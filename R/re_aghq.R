#' Adaptive Gauss-Hermite refinement of a grouped random-effect covariance
#'
#' @description
#' Refines a generalized linear mixed model's fixed effects and random-effect
#' covariance by replacing the per-group Laplace integral with `n_quad`-point
#' adaptive Gauss-Hermite quadrature (AGHQ). At `n_quad = 1` this is the joint
#' Laplace (glmer `nAGQ = 1`); higher `n_quad` reduces the small-cluster
#' attenuation of the variance components for binary / count data. Unlike
#' [agq_fit()] (intercept-only RE, built-in `binomial`/`poisson`/`gaussian`
#' likelihoods), this engine is **callback-driven**: the caller supplies the
#' per-group conditional likelihood, so a custom marginal (e.g. a
#' latent-state-integrated occupancy / detection likelihood, or the
#' latent-abundance-integrated N-mixture marginal) refines through the same
#' quadrature.
#'
#' The engine is **structure-agnostic**. It integrates the per-group marginal
#' \deqn{M_g = \int \exp\{\ell_g(b_g)\}\, N(b_g; 0, \Sigma)\, db_g,}
#' where \eqn{b_g} is the group's random-effect vector (dimension
#' \eqn{\sum_m c_m} over the RE terms) and \eqn{\ell_g(b_g)} is the group's
#' conditional log-likelihood when its linear predictors are perturbed by
#' \eqn{b_g}. How \eqn{b_g} enters the likelihood -- through one linear
#' predictor, or through several coupled arms at different observation
#' granularities (e.g. a per-site abundance arm and a per-visit detection arm
#' sharing a species grouping) -- lives entirely in the callback. The engine
#' only needs, per group, the value / gradient / Hessian of \eqn{\ell_g} in
#' \eqn{b_g} (for the mode) and \eqn{\ell_g} at the quadrature nodes (for the
#' sum). The fixed parameters `theta` and the log-Cholesky coordinates of
#' \eqn{\Sigma} are optimized jointly on \eqn{\sum_g \log M_g}; standard errors
#' come from the exact-marginal Hessian.
#'
#' Two callback forms select the structure (supply exactly one):
#'
#' * **`make_site`** -- the common **single-arm, per-row-separable** case
#'   (\eqn{\ell_g(b_g) = \sum_{i \in g} \log f_i(\eta_i + Z_i b_g)} for one
#'   linear predictor \eqn{\eta}). The engine builds the oracle from the
#'   per-row marginal and the RE design `Z` itself.
#' * **`make_group`** -- the **general / multi-arm** case. The caller supplies
#'   the per-group `b`-space oracle directly, so non-separable units (e.g. the
#'   visits of an N-mixture site coupled through the shared latent count) and
#'   random effects on several arms at once are handled with no engine change.
#'
#' Scope: one shared grouping factor across all RE terms (the per-group integral
#' factorizes). The total RE dimension per group should be small (the quadrature
#' grid is `n_quad^dim`).
#'
#' @param theta0 Initial fixed-parameter vector. The engine optimizes these
#'   jointly with the RE covariance; the callback interprets them.
#' @param re_terms A list of RE term specs (or one spec), each defining a
#'   covariance block: `n_coefs` (block dimension \eqn{c_m}), optional
#'   `correlated` (default `TRUE` for `c_m > 1`; `FALSE` gives a diagonal block),
#'   and `n_groups` (shared across terms). For the `make_site` path each term
#'   also carries `idx` (1-based group index, length `n_obs`) and, for a slope
#'   block, `Z` (the `n_obs x n_coefs` design). For the `make_group` path the
#'   per-observation `idx` / `Z` are optional -- the callback owns them -- and
#'   the term needs only `n_coefs` / `correlated` / `n_groups`.
#' @param Sigma0 List of initial per-term covariance matrices (the EM estimate).
#' @param make_site `function(theta)` for the single-arm separable case,
#'   returning a list with: `eta_re` (length `n_obs`, the RE-arm fixed
#'   predictor), `deriv = function(rows, eta)` returning `list(logL, d1, d2)`
#'   (per-row marginal log-likelihood and its first/second derivatives w.r.t.
#'   the RE-arm predictor `eta`, used for the per-group mode), and `lmat =
#'   function(rows, ETA)` returning a `length(rows) x ncol(ETA)` matrix of
#'   per-observation log-likelihoods over the quadrature node columns. Supply
#'   this or `make_group`, not both.
#' @param make_group `function(theta)` for the general / multi-arm case,
#'   returning a list with two per-group closures (let `d = sum(n_coefs)` be the
#'   group RE dimension):
#'   * `grad_hess(g, b)` -- for group `g` at RE value `b` (length `d`), the list
#'     `list(logL, grad, negH)`: the group conditional log-likelihood
#'     \eqn{\ell_g(b)}, its gradient \eqn{\partial \ell_g/\partial b} (length
#'     `d`), and the **data-only** observed information
#'     \eqn{-\partial^2 \ell_g/\partial b^2} (`d x d`; the engine adds the
#'     \eqn{\Sigma^{-1}} prior curvature).
#'   * `node_ll(g, B)` -- for group `g`, a numeric vector of length `nrow(B)`
#'     giving \eqn{\ell_g} at each quadrature node (rows of the `nrow x d`
#'     matrix `B` are candidate `b` vectors).
#'   The callback owns all arm / design / clamping bookkeeping. Supply this or
#'   `make_site`, not both.
#' @param n_obs Number of observations (length of each term's `idx`). Required
#'   for the `make_site` path; ignored for `make_group`.
#' @param keep Optional logical/integer mask of observations to include
#'   (default all; `make_site` path only). Rows outside `keep` are dropped from
#'   every group.
#' @param n_quad Quadrature nodes per RE dimension (default 9; `1` = Laplace).
#' @param lkj_eta LKJ shape for an optional correlation penalty on each
#'   *correlated* block (log-density `(eta - 1) log det R`, maximized at
#'   independence). `1` disables it; `> 1` regularizes a weakly-identified
#'   correlation off the boundary without touching the marginal SDs. The
#'   marginal SDs are otherwise unpenalized (pure ML), so the refinement debiases
#'   them rather than shrinking them.
#' @param maxit Optimizer iteration cap (default 200).
#'
#' @return A list with: `theta` (refined fixed parameters), `Sigma_list`
#'   (refined per-term covariance), `blup` / `blup_var` (per-term `n_groups x
#'   n_coefs` posterior mean / variance of the RE), `theta_cov` / `theta_se`
#'   (fixed-parameter covariance / SE from the marginal Hessian), `n_quad`,
#'   `lkj_eta`, and `converged`. Returns `NULL` if the RE terms do not share one
#'   grouping factor or the optimum is not usable (caller keeps its prior fit).
#' @export
tulpa_re_aghq <- function(theta0, re_terms, Sigma0,
                          make_site = NULL, make_group = NULL,
                          n_obs = NULL,
                          keep = NULL, n_quad = 9L, lkj_eta = 1,
                          maxit = 200L) {
  if (is.null(make_site) == is.null(make_group)) {
    stop("Supply exactly one of `make_site` (single-arm) or `make_group` ",
         "(general / multi-arm).", call. = FALSE)
  }
  single_arm <- !is.null(make_site)
  if (single_arm && is.null(n_obs)) {
    stop("`n_obs` is required for the `make_site` path.", call. = FALSE)
  }

  layout <- .re_cov_block_layout(.as_re_terms_list(re_terms), n_obs)

  # One shared grouping factor (the per-group integral factorizes only then).
  # The make_group path may omit the per-observation `idx`; the layout then
  # carries `idx = NULL`, and the shared-factor check reduces to matching
  # `n_groups` (identical(NULL, NULL) is TRUE).
  idx1 <- layout[[1L]]$idx
  ng   <- layout[[1L]]$n_groups
  same <- all(vapply(layout, function(b)
    identical(b$idx, idx1) && identical(b$n_groups, ng), logical(1)))
  if (!same) return(NULL)

  nc_terms <- vapply(layout, function(b) b$nc, integer(1))
  dtot     <- sum(nc_terms)
  coef_off <- cumsum(c(0L, nc_terms))

  # Fixed-parameter / RE-covariance parameter split. The RE block reuses tulpa's
  # log-Cholesky packing (.re_cov_theta_to_L_list); `theta` is everything else.
  re_par0 <- .re_cov_L_list_to_theta(lapply(Sigma0, .re_chol_spd), layout)
  n_theta <- length(theta0)

  gh <- gauss_hermite_prob(n_quad)              # nodes z, weights w (sum w = 1)
  Q  <- length(gh$nodes)
  grid   <- as.matrix(expand.grid(rep(list(seq_len(Q)), dtot)))
  Znodes <- matrix(gh$nodes[grid],        nrow = nrow(grid), ncol = dtot)
  logw_q <- rowSums(matrix(log(gh$weights)[grid], nrow = nrow(grid)))
  z2_q   <- rowSums(Znodes^2)
  cl <- function(e) pmin(pmax(e, -30), 30)

  block_diag <- function(mats) {
    if (length(mats) == 1L) return(mats[[1L]])
    out <- matrix(0, dtot, dtot); pos <- 0L
    for (m in mats) {
      k <- nrow(m); out[pos + seq_len(k), pos + seq_len(k)] <- m; pos <- pos + k
    }
    out
  }

  # -------------------------------------------------------------------------
  # Per-group conditional-likelihood oracle. `build_oracle(theta)` returns
  #   grad_hess(g, b) -> list(logL, grad, negH)   data-only value/score/info
  #   node_ll(g, B)   -> numeric over node rows    data log-lik at nodes
  # The integration core below is identical for both callback forms; only the
  # oracle differs. For make_site the engine assembles it from the per-row
  # marginal and the RE design; for make_group the caller supplies it directly.
  # -------------------------------------------------------------------------
  if (single_arm) {
    Zc <- do.call(cbind, lapply(layout, function(b) b$Z))   # n_obs x dtot
    if (is.null(keep)) keep <- rep(TRUE, n_obs)
    if (is.logical(keep)) keep <- which(keep)
    rows_by_g <- lapply(seq_len(ng), function(g) {
      r <- which(idx1 == g); r[r %in% keep]
    })
    active_groups <- which(lengths(rows_by_g) > 0L)

    build_oracle <- function(theta) {
      site <- make_site(theta)
      aRE  <- cl(site$eta_re)
      list(
        grad_hess = function(g, b) {
          rows <- rows_by_g[[g]]
          Zg <- Zc[rows, , drop = FALSE]
          gv <- site$deriv(rows, cl(aRE[rows] + as.numeric(Zg %*% b)))
          list(logL = sum(gv$logL),
               grad = as.numeric(crossprod(Zg, gv$d1)),
               negH = -crossprod(Zg, gv$d2 * Zg))
        },
        node_ll = function(g, B) {
          rows <- rows_by_g[[g]]
          Zg <- Zc[rows, , drop = FALSE]
          ETA <- cl(matrix(aRE[rows], length(rows), nrow(B)) + Zg %*% t(B))
          colSums(site$lmat(rows, ETA))
        })
    }
  } else {
    active_groups <- seq_len(ng)
    build_oracle  <- function(theta) make_group(theta)
  }

  # Per-group posterior mode of b (damped Newton on the penalized integrand) and
  # the precision -H there. Structure-agnostic via the oracle.
  grp_mode <- function(g, oracle, P) {
    bb <- numeric(dtot)
    for (it in seq_len(50L)) {
      gh <- oracle$grad_hess(g, bb)
      grad <- gh$grad - as.numeric(P %*% bb)
      negH <- gh$negH + P
      step <- tryCatch(solve(negH, grad), error = function(e) NULL)
      if (is.null(step)) break
      bb <- bb + step
      if (max(abs(step)) < 1e-9) break
    }
    gh <- oracle$grad_hess(g, bb)
    list(b = bb, negH = gh$negH + P)
  }

  # log M_g via adaptive GHQ centred at the mode (probabilist's convention):
  #   b_k = b_hat + L_c z_k,  C = L_c L_c' = (-H)^{-1}
  #   log M_g = -0.5 logdetSigma + log|L_c|
  #             + lse_k[ log w_k + ell_g(b_k) - 0.5 b_k' Sigma^{-1} b_k
  #                      + 0.5 z_k' z_k ]
  grp_logM <- function(g, oracle, P, logdetS, want_post = FALSE) {
    m <- grp_mode(g, oracle, P)
    Lc <- tryCatch(t(chol(solve(m$negH))), error = function(e) NULL)
    if (is.null(Lc)) return(NULL)
    B <- matrix(m$b, nrow(Znodes), dtot, byrow = TRUE) + Znodes %*% t(Lc)
    hvals <- oracle$node_ll(g, B) - 0.5 * rowSums((B %*% P) * B)
    terms <- logw_q + hvals + 0.5 * z2_q
    mx <- max(terms)
    logM <- -0.5 * logdetS + sum(log(diag(Lc))) + mx + log(sum(exp(terms - mx)))
    if (want_post) list(logM = logM, b = m$b, var = diag(solve(m$negH)))
    else logM
  }

  # LKJ(eta) penalty on each correlated block: (eta - 1) log det R, with
  # log det R = 2 sum log L_ii - 2 sum log sigma_i (sigma_i = sqrt(Sigma_ii)).
  # Marginal SDs are left unpenalized.
  lkj_logprior <- function(L_list) {
    if (lkj_eta == 1) return(0)
    s <- 0
    for (m in seq_along(layout)) {
      if (!layout[[m]]$full) next
      L <- L_list[[m]]
      sig <- sqrt(rowSums(L^2))
      s <- s + (lkj_eta - 1) * (2 * sum(log(diag(L))) - 2 * sum(log(pmax(sig, 1e-12))))
    }
    s
  }

  nll <- function(par) {
    theta <- par[seq_len(n_theta)]
    L_list <- .re_cov_theta_to_L_list(par[-seq_len(n_theta)], layout)
    Sig <- block_diag(lapply(L_list, tcrossprod)) + diag(1e-10, dtot)
    P <- tryCatch(solve(Sig), error = function(e) NULL)
    if (is.null(P)) return(1e10)
    logdetS <- as.numeric(determinant(Sig, logarithm = TRUE)$modulus)
    oracle <- build_oracle(theta)
    total <- 0
    for (g in active_groups) {
      lm <- grp_logM(g, oracle, P, logdetS)
      if (is.null(lm) || !is.finite(lm)) return(1e10)
      total <- total + lm
    }
    -(total + lkj_logprior(L_list))
  }

  opt <- stats::optim(c(theta0, re_par0), nll, method = "BFGS", hessian = TRUE,
                      control = list(maxit = maxit, reltol = 1e-9))
  V <- tryCatch(solve(opt$hessian), error = function(e) NULL)
  if (is.null(V) || any(!is.finite(opt$par))) return(NULL)

  theta_ref <- opt$par[seq_len(n_theta)]
  L_list <- .re_cov_theta_to_L_list(opt$par[-seq_len(n_theta)], layout)
  Sigma_list <- lapply(L_list, tcrossprod)

  # Per-group BLUPs + variances at the optimum.
  Sig <- block_diag(Sigma_list) + diag(1e-10, dtot)
  P <- solve(Sig)
  logdetS <- as.numeric(determinant(Sig, logarithm = TRUE)$modulus)
  oracle <- build_oracle(theta_ref)
  BHAT <- matrix(0, ng, dtot)
  BVAR <- matrix(rep(diag(Sig), each = ng), ng, dtot)   # empty groups -> prior
  for (g in active_groups) {
    post <- grp_logM(g, oracle, P, logdetS, want_post = TRUE)
    if (is.null(post)) return(NULL)
    BHAT[g, ] <- post$b
    BVAR[g, ] <- post$var
  }
  blup     <- lapply(seq_along(layout), function(m) BHAT[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  blup_var <- lapply(seq_along(layout), function(m) BVAR[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])

  list(
    theta      = theta_ref,
    Sigma_list = Sigma_list,
    blup       = blup,
    blup_var   = blup_var,
    theta_cov  = V[seq_len(n_theta), seq_len(n_theta), drop = FALSE],
    theta_se   = sqrt(pmax(diag(V)[seq_len(n_theta)], 0)),
    n_quad     = as.integer(n_quad),
    lkj_eta    = lkj_eta,
    converged  = isTRUE(opt$convergence == 0L)
  )
}
