# Exact-target Gibbs estimation of random-effect covariances Sigma.
#
# Bias-1 fix (the Laplace / PQL "approximation" bias): the Laplace marginal
# fits Sigma at the joint mode and reports a curvature-based covariance for the
# random effects. For binary / low-count responses with small groups the true
# random-effect conditional p(b_g | beta, Sigma, y) is non-Gaussian, so its
# Gaussian (Laplace) approximation under-disperses -> Sigma is biased low. This
# sampler removes that bias by sampling the exact joint posterior
# p(beta, {b_m}, {Sigma_m} | y) with a Metropolis-within-Gibbs scheme:
#
#   * b_{m,g} | beta, Sigma, y  -- random-walk Metropolis per (term, group)
#       (groups are conditionally independent given beta and the covariances),
#       proposal SHAPE from the Laplace per-group posterior covariance block
#       (return_re_cov) and a per-term adapted scale. MH against the exact group
#       likelihood + Gaussian RE prior corrects the non-Gaussianity.
#   * beta | b, Sigma, y    -- random-walk Metropolis, proposal shape from the
#       Laplace fixed-effect Hessian block (H_beta).
#   * Sigma_m | b_m         -- EXACT conjugate draw. For a CORRELATED term, under
#       an inverse-Wishart prior IW(nu0, Lambda0) and b_{m,g} ~ N(0, Sigma_m)
#       i.i.d. over G_m groups, Sigma_m | b_m ~ IW(nu0 + G_m, Lambda0 + sum_g
#       b_{m,g} b_{m,g}'). For an UNCORRELATED term Sigma_m is diagonal and each
#       variance has the scalar conjugate (inverse-gamma == 1x1 inverse-Wishart).
#       No linearization.
#
# Several random-effect terms (`(1 + x | g) + (1 | h)`, correlated and / or
# uncorrelated) are sampled jointly; a single-term model is the length-1 case.
#
# Composition: Laplace body (starting values + proposal shapes) + MH debias
# (exact b / beta conditionals) + conjugate outer (exact Sigma). The summary
# reuses `.re_cov_derived_summary` (shared with the grid integrator).

# inverse-Wishart sampler via Bartlett (native, no dependency) ---------------
# Sigma ~ IW(df, Lambda)  <=>  Sigma^{-1} ~ Wishart(df, Lambda^{-1}). Sample the
# Wishart by Bartlett (Lambda^{-1} = C C'; A lower-triangular with
# A_ii = sqrt(chisq_{df - i + 1}), A_ij ~ N(0,1) for i > j; W = C A A' C'), then
# invert. Requires df > p - 1 (so a 1x1 block, the scalar inverse-gamma, needs
# only df > 0).
.rinvwishart <- function(df, Lambda) {
  p <- nrow(Lambda)
  if (df <= p - 1L) {
    stop(sprintf("inverse-Wishart df = %g must exceed p - 1 = %d.", df, p - 1L),
         call. = FALSE)
  }
  V <- chol2inv(chol(Lambda))            # Lambda^{-1} (SPD)
  C <- t(chol(V))                        # V = C C'
  A <- matrix(0, p, p)
  for (i in seq_len(p)) {
    A[i, i] <- sqrt(stats::rchisq(1L, df - i + 1L))
    if (i > 1L) A[i, seq_len(i - 1L)] <- stats::rnorm(i - 1L)
  }
  CA <- C %*% A
  W  <- tcrossprod(CA)                   # Wishart(df, V) = Sigma^{-1}
  chol2inv(chol(W))                      # Sigma
}

# Per-observation log-likelihood matching tulpa's links (logit / log / identity
# / log). Full normalizing constants included (harmless: only differences enter
# the MH ratio, but keeps the routine self-contained / reusable).
.re_obs_loglik <- function(eta, y, n_trials, family, phi) {
  switch(family,
    binomial       = stats::dbinom(y, size = n_trials, prob = stats::plogis(eta),
                                   log = TRUE),
    poisson        = stats::dpois(y, lambda = exp(eta), log = TRUE),
    gaussian       = stats::dnorm(y, mean = eta, sd = sqrt(phi), log = TRUE),
    neg_binomial_2 = stats::dnbinom(y, size = phi, mu = exp(eta), log = TRUE),
    negbin         = stats::dnbinom(y, size = phi, mu = exp(eta), log = TRUE),
    stop(sprintf("tulpa_re_cov_gibbs: unsupported family '%s'.", family),
         call. = FALSE)
  )
}

# Symmetrize + eigen-floor a proposal covariance block to SPD, then return its
# lower Cholesky factor (the random-walk proposal shape).
.re_chol_spd <- function(S, floor = 1e-8) {
  S <- (S + t(S)) / 2
  e <- eigen(S, symmetric = TRUE)
  d <- pmax(e$values, floor)
  M <- e$vectors %*% (d * t(e$vectors))
  t(chol((M + t(M)) / 2))
}

# Build a per-block conjugate prior from the (possibly NULL) user overrides.
# A correlated block uses an inverse-Wishart IW(nu0, Lambda0); an uncorrelated
# block uses an independent scalar inverse-Wishart (== inverse-gamma) per
# variance. The overrides apply when their dimension matches the block (the
# documented single-covariance case); otherwise weakly-informative defaults.
.re_gibbs_block_prior <- function(bl, prior_df, prior_scale) {
  nc <- bl$nc
  if (bl$full) {
    nu0 <- if (is.null(prior_df)) nc + 1L else prior_df
    Lam <- if (!is.null(prior_scale) &&
               all(dim(as.matrix(prior_scale)) == nc)) as.matrix(prior_scale)
           else diag(nc)
    if (nu0 <= nc - 1L) {
      stop(sprintf("`prior_df` = %g must exceed n_coefs - 1 = %d for a proper ",
                   "inverse-Wishart prior.", nu0, nc - 1L), call. = FALSE)
    }
    list(full = TRUE, nu0 = nu0, Lambda0 = Lam)
  } else {
    nu0 <- if (is.null(prior_df)) 2 else prior_df
    lam <- if (!is.null(prior_scale) &&
               all(dim(as.matrix(prior_scale)) == nc))
             diag(as.matrix(prior_scale)) else rep(1, nc)
    if (nu0 <= 0) {
      stop(sprintf("`prior_df` = %g must be > 0 for a proper scalar ",
                   "inverse-gamma prior on a diagonal block.", nu0),
           call. = FALSE)
    }
    list(full = FALSE, nu0 = nu0, lambda0 = lam)
  }
}

# Exact conjugate draw of one block's Sigma given its random effects B_m
# (G_m x nc). Correlated: inverse-Wishart on the full matrix. Uncorrelated:
# independent scalar inverse-Wishart per diagonal variance.
.re_gibbs_draw_sigma <- function(B_m, pr) {
  G <- nrow(B_m); nc <- ncol(B_m)
  if (pr$full) {
    .rinvwishart(pr$nu0 + G, pr$Lambda0 + crossprod(B_m))
  } else {
    v <- vapply(seq_len(nc), function(i) {
      as.numeric(.rinvwishart(pr$nu0 + G,
                              matrix(pr$lambda0[i] + sum(B_m[, i]^2), 1L, 1L)))
    }, numeric(1))
    S <- matrix(0, nc, nc); diag(S) <- v; S
  }
}

#' Gibbs estimation of random-effect covariances (exact-target debias)
#'
#' @description
#' For one or more random-effects terms (e.g. `(1 + x | g)`, `(1 + x || g)`, or
#' several terms together), estimate the random-effect covariances `Sigma` by
#' sampling the exact joint posterior `p(beta, {b}, {Sigma} | y)` rather than
#' fixing each `Sigma` at the Laplace mode. This removes the Laplace / PQL
#' "approximation" bias that shrinks variance components low for binary and
#' low-count responses with small groups.
#'
#' @details
#' Metropolis-within-Gibbs targeting `p(beta, {b_m}, {Sigma_m} | y)`:
#'
#' - **`b_{m,g} | beta, Sigma, y`** -- random-walk Metropolis per (term, group)
#'   (groups are conditionally independent given `beta` and the covariances). The
#'   proposal *shape* is the Laplace per-group posterior covariance block
#'   (`return_re_cov` from [tulpa_laplace()]); the Metropolis acceptance is
#'   computed against the exact group log-likelihood plus the Gaussian RE
#'   log-prior, which corrects the non-Gaussianity the Laplace approximation
#'   misses. The linear predictor holds every other term's contribution fixed.
#' - **`beta | b, Sigma, y`** -- random-walk Metropolis, proposal shape from the
#'   Laplace fixed-effect Hessian block `H_beta`.
#' - **`Sigma_m | b_m`** -- exact conjugate draw. A correlated block draws the
#'   full matrix from `IW(prior_df + G_m, prior_scale + sum_g b_{m,g} b_{m,g}')`;
#'   an uncorrelated (diagonal) block draws each variance from its scalar
#'   conjugate (inverse-gamma). No linearization enters this step.
#'
#' A single Laplace solve provides the starting values (`beta`, `b`) and the
#' proposal shapes; the random-walk scales for the `beta` block and the per-term
#' `b` blocks are adapted toward their target acceptance during burn-in
#' (Robbins-Monro), then held fixed for the recorded sweeps. The covariance
#' summary marginalizes the derived scale / correlation parameters over the
#' posterior draws via the same machinery as [tulpa_re_cov_nested()].
#'
#' For `family = "gaussian"` the response is already conditionally Gaussian, so
#' there is no Laplace bias to remove; `phi` (the residual variance) is treated
#' as known. The sampler still runs and is useful as a reference / for `Sigma`
#' uncertainty.
#'
#' @param y,n_trials,X,family,phi Passed to the likelihood and to
#'   [tulpa_laplace()] for the pilot solve. `n_trials = NULL` defaults to 1.
#' @param re_terms Either a single random-effect term or a list of them. Each
#'   term is a list with `idx` (1-based group index per observation),
#'   `n_groups`, `n_coefs` (`c`), `Z` (the `n_obs x c` RE design; only required
#'   when `c > 1`), and `correlated` (`TRUE` for a full `Sigma`, `FALSE` for a
#'   diagonal one; defaults to `TRUE`). An optional `label` / `group_var` names
#'   the block. Any supplied `L` / `cov` / `sigma` is ignored -- `Sigma` is what
#'   this function samples.
#' @param n_iter Number of recorded post-burn-in sweeps (default 2000).
#' @param n_burnin Burn-in sweeps, used for proposal-scale adaptation (default
#'   1000).
#' @param thin Keep every `thin`-th recorded sweep (default 1).
#' @param prior_df Inverse-Wishart prior degrees of freedom. Applied to every
#'   correlated block (default `n_coefs + 1`, the minimal proper choice) and as
#'   the scalar inverse-gamma shape for every diagonal block (default 2). Must
#'   leave each block's prior proper.
#' @param prior_scale Inverse-Wishart prior scale matrix. Used for a block when
#'   its dimension matches (default `diag(n_coefs)`); otherwise the per-block
#'   default is used.
#' @param beta_prior_mean,beta_prior_sd Gaussian fixed-effect prior (defaults
#'   `0` and `100`). Scalars are recycled to `ncol(X)`.
#' @param seed Optional integer seed for reproducibility.
#' @param max_iter,tol,n_threads Pilot-solve controls (see [tulpa_laplace()]).
#'
#' @return A list with:
#'   - `posterior`: data frame with one row per parameter (`sigma_i`, `rho_ij`,
#'     `Sigma_ij`; prefixed by block label when there are several terms; diagonal
#'     blocks report no `rho`) and columns `mean`, `sd`, `median`, `ci_lo`,
#'     `ci_hi`.
#'   - `Sigma_mean`: the posterior mean of `Sigma` (a matrix for one block, a
#'     named list of matrices for several).
#'   - `Sigma_draws`: list of recorded draws; each element is the per-block list
#'     of `Sigma` matrices for that sweep.
#'   - `beta_draws` / `draws`: matrix of recorded `beta` draws; `draws` (with
#'     `means`, `param_names`, `process_info`) drives the generic `tulpa_fit`
#'     methods.
#'   - `accept`: list with `beta` and `b` acceptance rates over recorded sweeps.
#'   - `n_kept`, `n_coefs` (vector of per-block `c`), `prior`: bookkeeping.
#'
#' @seealso [tulpa_re_cov_nested()] for the grid-integration (summary-bias) fix;
#'   [tulpa_laplace()] for the pilot solve and per-group covariance blocks.
#'
#' @export
tulpa_re_cov_gibbs <- function(y, n_trials = NULL, X, re_terms,
                               family = "binomial", phi = 1.0,
                               n_iter = 2000L, n_burnin = 1000L, thin = 1L,
                               prior_df = NULL, prior_scale = NULL,
                               beta_prior_mean = 0, beta_prior_sd = 100,
                               seed = NULL,
                               max_iter = 100L, tol = 1e-8, n_threads = 1L) {

  if (!is.null(seed)) set.seed(as.integer(seed))

  re_terms <- .as_re_terms_list(re_terms)
  n_obs <- length(y)
  p     <- ncol(X)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  layout <- .re_cov_block_layout(re_terms, n_obs)
  M      <- length(layout)

  # --- per-block conjugate priors -------------------------------------------
  priors <- lapply(layout, .re_gibbs_block_prior, prior_df = prior_df,
                   prior_scale = prior_scale)

  # --- fixed-effect prior ----------------------------------------------------
  bmean <- if (length(beta_prior_mean) == 1L) rep(beta_prior_mean, p) else beta_prior_mean
  bsd   <- if (length(beta_prior_sd)   == 1L) rep(beta_prior_sd,   p) else beta_prior_sd
  if (length(bmean) != p || length(bsd) != p) {
    stop("`beta_prior_mean` / `beta_prior_sd` must be length 1 or ncol(X).",
         call. = FALSE)
  }
  log_prior_beta <- function(b) sum(stats::dnorm(b, bmean, bsd, log = TRUE))

  # --- pilot Laplace solve: starting values + proposal shapes ---------------
  # Sigma_m = I init for every block; mode gives beta and per-term b; H_beta and
  # cov_blocks (term-major then group order) give the proposal shapes.
  pilot <- tulpa_laplace(
    y = y, n_trials = n_trials, X = X,
    re_list = .re_cov_build_re_list(lapply(layout, function(bl) diag(bl$nc)),
                                    layout),
    family = family, phi = phi, return_hessian = TRUE, return_re_cov = TRUE,
    max_iter = max_iter, tol = tol, n_threads = n_threads
  )
  beta <- pilot$mode[seq_len(p)]
  re_vals <- pilot$mode[-seq_len(p)]

  L_beta <- if (!is.null(pilot$H_beta)) {
    .re_chol_spd(tryCatch(solve(pilot$H_beta), error = function(e) diag(p)))
  } else diag(p)

  # Slice the latent mode (term-major then group order) and the per-group
  # proposal blocks into per-term structures.
  B_list      <- vector("list", M)   # B_list[[m]]: G_m x nc_m random effects
  L_g_list    <- vector("list", M)   # per-group RW proposal Cholesky factors
  grp_obs     <- vector("list", M)   # per-group observation indices
  Z_g_list    <- vector("list", M)
  pos_re <- 0L; pos_cb <- 0L
  cb <- pilot$cov_blocks
  cb_ok <- !is.null(cb) && length(cb) == sum(vapply(layout, `[[`, integer(1),
                                                    "n_groups"))
  for (m in seq_len(M)) {
    bl <- layout[[m]]; G <- bl$n_groups; nc <- bl$nc
    len <- G * nc
    B_list[[m]] <- matrix(re_vals[pos_re + seq_len(len)], ncol = nc, byrow = TRUE)
    pos_re <- pos_re + len
    if (cb_ok) {
      L_g_list[[m]] <- lapply(cb[pos_cb + seq_len(G)], .re_chol_spd)
    } else {
      L_g_list[[m]] <- rep(list(diag(nc)), G)
    }
    pos_cb <- pos_cb + G
    go <- split(seq_len(n_obs), bl$idx)[as.character(seq_len(G))]
    grp_obs[[m]] <- go
    Z_g_list[[m]] <- lapply(seq_len(G), function(g) bl$Z[go[[g]], , drop = FALSE])
  }
  y_g_list <- lapply(seq_len(M), function(m)
    lapply(grp_obs[[m]], function(o) y[o]))
  n_g_list <- lapply(seq_len(M), function(m)
    lapply(grp_obs[[m]], function(o) n_trials[o]))

  # initial per-block Sigma from the pilot RE modes (method-of-moments), PD-clamped
  Sigma_list <- vector("list", M); Q_list <- vector("list", M)
  for (m in seq_len(M)) {
    bl <- layout[[m]]; B <- B_list[[m]]
    S <- if (bl$n_groups > bl$nc) stats::cov(B) else diag(bl$nc)
    if (!bl$full) S <- diag(diag(S), bl$nc)
    diag(S) <- pmax(diag(S), 1e-3)
    if (min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) < 1e-8)
      S <- diag(bl$nc)
    Sigma_list[[m]] <- S
    Q_list[[m]]     <- chol2inv(chol(S))
  }

  # per-term running RE contribution to the linear predictor, and its total
  re_contrib <- lapply(seq_len(M), function(m)
    rowSums(layout[[m]]$Z * B_list[[m]][layout[[m]]$idx, , drop = FALSE]))
  re_total <- Reduce(`+`, re_contrib)
  xb <- as.numeric(X %*% beta)

  # random-walk scales (adapted during burn-in): beta block + one per term
  s_beta   <- 2.4 / sqrt(p)
  s_b      <- vapply(layout, function(bl) 2.4 / sqrt(bl$nc), numeric(1))
  tgt_beta <- if (p > 1L) 0.234 else 0.44
  tgt_b    <- vapply(layout, function(bl) if (bl$nc > 1L) 0.234 else 0.44,
                     numeric(1))

  n_sweep <- as.integer(n_burnin + n_iter)
  keep_at <- seq.int(n_burnin + 1L, n_sweep, by = as.integer(thin))
  n_kept  <- length(keep_at)
  Sigma_draws <- vector("list", n_kept)
  beta_draws  <- matrix(NA_real_, n_kept, p)
  acc_beta_rec <- 0L; acc_b_rec <- 0L; n_b_rec <- 0L
  kept <- 0L

  for (sweep in seq_len(n_sweep)) {
    adapting <- sweep <= n_burnin
    gamma_t  <- 1 / sqrt(sweep)

    # --- beta | b, Sigma, y : RW Metropolis ---------------------------------
    eta_cur <- xb + re_total
    ll_cur  <- sum(.re_obs_loglik(eta_cur, y, n_trials, family, phi)) +
               log_prior_beta(beta)
    beta_prop <- beta + s_beta * as.numeric(L_beta %*% stats::rnorm(p))
    xb_prop   <- as.numeric(X %*% beta_prop)
    eta_prop  <- xb_prop + re_total
    ll_prop   <- sum(.re_obs_loglik(eta_prop, y, n_trials, family, phi)) +
                 log_prior_beta(beta_prop)
    acc_beta <- log(stats::runif(1)) < (ll_prop - ll_cur)
    if (isTRUE(acc_beta)) { beta <- beta_prop; xb <- xb_prop }
    if (adapting) s_beta <- exp(log(s_beta) + gamma_t * ((acc_beta) - tgt_beta))

    # --- b_{m,g} | beta, Sigma, y : per-(term, group) RW Metropolis ---------
    n_acc_b <- 0L; n_prop_b <- 0L
    for (m in seq_len(M)) {
      bl <- layout[[m]]; G <- bl$n_groups; nc <- bl$nc
      Q  <- Q_list[[m]]; Lg <- L_g_list[[m]]; B <- B_list[[m]]
      # offset = linear predictor minus this term's contribution (others fixed)
      base <- xb + (re_total - re_contrib[[m]])
      acc_m <- 0L
      for (g in seq_len(G)) {
        obs <- grp_obs[[m]][[g]]
        if (length(obs) == 0L) next
        bg  <- B[g, ]
        Zg  <- Z_g_list[[m]][[g]]
        eta_g_cur <- base[obs] + as.numeric(Zg %*% bg)
        ll_g_cur  <- sum(.re_obs_loglik(eta_g_cur, y_g_list[[m]][[g]],
                                        n_g_list[[m]][[g]], family, phi)) -
                     0.5 * as.numeric(t(bg) %*% Q %*% bg)
        bg_prop <- bg + s_b[m] * as.numeric(Lg[[g]] %*% stats::rnorm(nc))
        eta_g_prop <- base[obs] + as.numeric(Zg %*% bg_prop)
        ll_g_prop  <- sum(.re_obs_loglik(eta_g_prop, y_g_list[[m]][[g]],
                                         n_g_list[[m]][[g]], family, phi)) -
                      0.5 * as.numeric(t(bg_prop) %*% Q %*% bg_prop)
        if (log(stats::runif(1)) < (ll_g_prop - ll_g_cur)) {
          B[g, ] <- bg_prop; acc_m <- acc_m + 1L
        }
      }
      B_list[[m]] <- B
      # refresh this term's contribution and the running total
      new_contrib   <- rowSums(bl$Z * B[bl$idx, , drop = FALSE])
      re_total      <- re_total - re_contrib[[m]] + new_contrib
      re_contrib[[m]] <- new_contrib
      if (adapting) s_b[m] <- exp(log(s_b[m]) + gamma_t * (acc_m / G - tgt_b[m]))
      n_acc_b  <- n_acc_b + acc_m
      n_prop_b <- n_prop_b + G
    }

    # --- Sigma_m | b_m : exact conjugate draw per block ---------------------
    for (m in seq_len(M)) {
      Sigma_list[[m]] <- .re_gibbs_draw_sigma(B_list[[m]], priors[[m]])
      Q_list[[m]]     <- chol2inv(chol(Sigma_list[[m]]))
    }

    # --- record -------------------------------------------------------------
    if (!adapting && (sweep %in% keep_at)) {
      kept <- kept + 1L
      Sigma_draws[[kept]] <- Sigma_list
      beta_draws[kept, ]  <- beta
      acc_beta_rec <- acc_beta_rec + as.integer(acc_beta)
      acc_b_rec    <- acc_b_rec + n_acc_b
      n_b_rec      <- n_b_rec + n_prop_b
    }
  }

  w <- rep(1 / n_kept, n_kept)
  summ <- .re_cov_derived_summary(Sigma_draws, w, layout)

  # Fixed-effect posterior is the recorded beta draws directly (equal weight);
  # name the columns and expose the generic `tulpa_fit` accessors.
  beta_names <- colnames(X) %||% paste0("beta", seq_len(p))
  colnames(beta_draws) <- beta_names
  beta_mean <- colMeans(beta_draws)

  # Each sweep stores a per-block list of Sigma; for a single block unwrap to a
  # plain list of matrices (matching the Sigma_mean matrix-vs-list convention).
  Sigma_draws_out <- if (M == 1L) lapply(Sigma_draws, `[[`, 1L) else Sigma_draws

  list(
    posterior   = summ$posterior,
    Sigma_mean  = summ$Sigma_mean,
    Sigma_draws = Sigma_draws_out,
    beta_draws  = beta_draws,
    beta        = beta_mean,
    draws       = beta_draws,
    means       = beta_mean,
    param_names = beta_names,
    process_info = list(list(name = "fixed_effects", p = p,
                             coef_names = beta_names)),
    n_samples   = n_kept,
    n_params    = p,
    N           = n_obs,
    accept      = list(beta = acc_beta_rec / n_kept,
                       b    = if (n_b_rec > 0L) acc_b_rec / n_b_rec else NA_real_),
    n_kept      = n_kept,
    n_blocks    = M,
    n_coefs     = vapply(layout, `[[`, integer(1), "nc"),
    prior       = priors
  )
}
