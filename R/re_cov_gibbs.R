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
# (exact b / beta conditionals) + conjugate outer (exact Sigma). The pilot solve
# and the posterior summary live in R; the Metropolis-within-Gibbs sweep is the
# compiled engine (src/re_cov_gibbs{,_sweep}.h), driven by one per-row family
# likelihood source (the native GLMM oracle, src/glmm_oracle.h). The summary
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
#' @param prior_df Inverse-Wishart prior degrees of freedom. Applied to every
#'   correlated block (default `n_coefs + 1`, the minimal proper choice) and as
#'   the scalar inverse-gamma shape for every diagonal block (default 2). Must
#'   leave each block's prior proper.
#' @param prior_scale Inverse-Wishart prior scale matrix. Used for a block when
#'   its dimension matches (default `diag(n_coefs)`); otherwise the per-block
#'   default is used.
#' @param beta_prior_mean,beta_prior_sd Gaussian fixed-effect prior (defaults
#'   `0` and `100`). Scalars are recycled to `ncol(X)`.
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above). Recognized entries:
#'   \itemize{
#'     \item `n_iter`: recorded post-warmup sweeps (default 2000).
#'     \item `warmup`: warmup (burn-in) sweeps, used for proposal-scale
#'       adaptation (default 1000).
#'     \item `thin`: keep every `thin`-th recorded sweep (default 1).
#'     \item `seed`: optional integer seed for reproducibility.
#'     \item `max_iter`, `tol`, `n_threads`: pilot-solve controls (see
#'       [tulpa_laplace()]).
#'   }
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
#' @references
#' Lewandowski, Kurowicka & Joe (2009). Generating random correlation matrices
#' based on vines and extended onion method. \emph{Journal of Multivariate
#' Analysis} 100(9):1989-2001.
#' @examples
#' \donttest{
#' set.seed(1)
#' G <- 20L; per <- 12L; n <- G * per
#' grp <- rep(seq_len(G), each = per); x <- rnorm(n)
#' b <- cbind(rnorm(G, 0, 0.7), rnorm(G, 0, 0.5))     # random intercept + slope
#' eta <- -0.2 + 0.5 * x + b[grp, 1] + b[grp, 2] * x
#' y <- rbinom(n, 1L, plogis(eta))
#' re_term <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
#'                 correlated = TRUE)
#' fit <- tulpa_re_cov_gibbs(y, rep(1L, n), cbind(1, x), re_term,
#'                           family = "binomial",
#'                           control = list(n_iter = 300L, warmup = 150L))
#' fit$Sigma_mean        # exact-debias RE covariance posterior mean
#' }
#' @export
tulpa_re_cov_gibbs <- function(y, n_trials = NULL, X, re_terms,
                               family = "binomial", phi = 1.0,
                               prior_df = NULL, prior_scale = NULL,
                               beta_prior_mean = 0, beta_prior_sd = 100,
                               control = list()) {
  # Perf/numerical knobs live in `control = list()` (matching tulpa() /
  # tulpa_nested_laplace()); the signature carries only statistical arguments.
  n_iter    <- as.integer(control$n_iter %||% 2000L)
  warmup    <- as.integer(control$warmup %||% 1000L)
  thin      <- as.integer(control$thin %||% 1L)
  seed      <- control$seed
  max_iter  <- as.integer(control$max_iter %||% 100L)
  tol       <- control$tol %||% 1e-8
  n_threads <- as.integer(control$n_threads %||% 1L)

  .seed_scoped(seed)

  re_terms <- .as_re_terms_list(re_terms)
  n_obs <- length(y)
  p     <- ncol(X)
  if (is.null(n_trials)) n_trials <- rep(1L, n_obs)
  if (!family %in% c("binomial", "poisson", "gaussian",
                     "neg_binomial_2", "negbin")) {
    stop(sprintf("tulpa_re_cov_gibbs: unsupported family '%s'.", family),
         call. = FALSE)
  }
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
    .re_chol_spd(tryCatch(solve(pilot$H_beta), error = function(e) {
      warning("tulpa_re_cov_gibbs(): pilot fixed-effect Hessian not ",
              "invertible (", conditionMessage(e), "); using an identity ",
              "proposal scale for the beta block.", call. = FALSE)
      diag(p)
    }))
  } else diag(p)

  # Slice the latent mode (term-major then group order) and the per-group
  # proposal blocks into per-term structures.
  B_list   <- vector("list", M)   # B_list[[m]]: G_m x nc random effects (init)
  L_g_list <- vector("list", M)   # per-group RW proposal Cholesky factors
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
  }

  # initial per-block Sigma from the pilot RE modes (method-of-moments), PD-clamped
  Sigma_list <- vector("list", M)
  for (m in seq_len(M)) {
    bl <- layout[[m]]; B <- B_list[[m]]
    S <- if (bl$n_groups > bl$nc) stats::cov(B) else diag(bl$nc)
    if (!bl$full) S <- diag(diag(S), bl$nc)
    diag(S) <- pmax(diag(S), 1e-3)
    if (min(eigen(S, symmetric = TRUE, only.values = TRUE)$values) < 1e-8)
      S <- diag(bl$nc)
    Sigma_list[[m]] <- S
  }

  # --- per-block specs for the compiled sweep -------------------------------
  # The engine owns the shared linear predictor and the cross-block eta coupling;
  # each block carries its design (Z, idx), conjugate prior, initial RE / Sigma
  # and per-group proposal Cholesky factors.
  blocks <- lapply(seq_len(M), function(m) {
    bl <- layout[[m]]; pr <- priors[[m]]
    spec <- list(Z = bl$Z, idx = bl$idx, nc = bl$nc, full = isTRUE(bl$full),
                 n_groups = bl$n_groups, nu0 = pr$nu0,
                 b0 = B_list[[m]], Lg0 = L_g_list[[m]], Sigma0 = Sigma_list[[m]])
    if (pr$full) spec$Lambda0 <- pr$Lambda0 else spec$lambda0 <- pr$lambda0
    spec
  })

  out <- cpp_re_cov_gibbs_sweep(
    family = family, phi = phi,
    y = as.numeric(y), n_trials = as.numeric(n_trials),
    X = as.matrix(X), blocks = blocks,
    beta0 = as.numeric(beta), L_beta = as.matrix(L_beta),
    n_iter = as.integer(n_iter), n_burnin = as.integer(warmup),
    thin = as.integer(thin),
    beta_prior_mean = as.numeric(bmean), beta_prior_sd = as.numeric(bsd)
  )

  n_kept      <- out$n_kept
  Sigma_draws <- out$Sigma_draws        # list (n_kept) of per-block list of Sigma
  beta_draws  <- out$beta_draws

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

  .finalize_fit(list(
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
    accept      = list(beta = out$accept_beta, b = out$accept_b),
    n_kept      = n_kept,
    n_blocks    = M,
    n_coefs     = vapply(layout, `[[`, integer(1), "nc"),
    prior       = priors
  ), backend = "re_cov_gibbs", n_fixed = p, fixed_names = beta_names)
}
