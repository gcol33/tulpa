#' Adaptive Gauss-Hermite refinement of a grouped random-effect covariance
#'
#' @description
#' Refines a generalized linear mixed model's fixed effects and random-effect
#' covariance by replacing the per-group Laplace integral with `n_quad`-point
#' adaptive Gauss-Hermite quadrature (AGHQ). At `n_quad = 1` this is the joint
#' Laplace (glmer `nAGQ = 1`); higher `n_quad` reduces the small-cluster
#' attenuation of the variance components for binary / count data. Unlike
#' [agq_fit()] (intercept-only RE, built-in `binomial`/`poisson`/`gaussian`
#' likelihoods), this engine is **callback-driven** and handles **random
#' slopes and correlated multi-coefficient blocks** sharing one grouping
#' factor: the caller supplies the per-observation marginal likelihood through
#' `make_site`, so a custom marginal (e.g. a latent-state-integrated occupancy
#' or detection likelihood) refines through the same quadrature.
#'
#' The per-group marginal is
#' \deqn{M_g = \int \prod_{i \in g} f_i(\eta_i + Z_i b_g)\, N(b_g; 0, \Sigma)\, db_g,}
#' where \eqn{\eta_i} is the RE-bearing linear predictor's fixed part (supplied
#' by `make_site` as `eta_re`) and \eqn{f_i} is the per-observation marginal
#' (its value and first two derivatives in \eqn{\eta} supplied by `deriv` /
#' `lmat`). The fixed parameters `theta` and the log-Cholesky coordinates of
#' \eqn{\Sigma} are optimized jointly on \eqn{\sum_g \log M_g}; standard errors
#' come from the exact-marginal Hessian.
#'
#' Scope: one shared grouping factor across all RE terms (the per-group integral
#' factorizes), each term a `(1 | g)` / slope / correlated block. The total RE
#' dimension per group should be small (the quadrature grid is `n_quad^dim`).
#'
#' @param theta0 Initial fixed-parameter vector. The engine optimizes these
#'   jointly with the RE covariance; `make_site(theta)` interprets them.
#' @param re_terms A list of RE term specs (or one spec), each with `idx`
#'   (1-based group index, length `n_obs`), `n_groups`, `n_coefs`, optional `Z`
#'   (the `n_obs x n_coefs` design for a slope block) and `correlated`. All
#'   terms must share the same `idx` / `n_groups`. Passed to
#'   `.re_cov_block_layout()`.
#' @param Sigma0 List of initial per-term covariance matrices (the EM estimate).
#' @param make_site `function(theta)` returning a list with: `eta_re` (length
#'   `n_obs`, the RE-arm fixed predictor), `deriv = function(rows, eta)`
#'   returning `list(logL, d1, d2)` (per-row marginal log-likelihood and its
#'   first/second derivatives w.r.t. the RE-arm predictor `eta`, used for the
#'   per-group mode), and `lmat = function(rows, ETA)` returning a
#'   `length(rows) x ncol(ETA)` matrix of per-observation log-likelihoods over
#'   the quadrature node columns.
#' @param n_obs Number of observations (length of each term's `idx`).
#' @param keep Optional logical/integer mask of observations to include
#'   (default all). Rows outside `keep` are dropped from every group.
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
tulpa_re_aghq <- function(theta0, re_terms, Sigma0, make_site, n_obs,
                          keep = NULL, n_quad = 9L, lkj_eta = 1,
                          maxit = 200L) {
  layout <- .re_cov_block_layout(.as_re_terms_list(re_terms), n_obs)

  # One shared grouping factor (the per-group integral factorizes only then).
  idx1 <- layout[[1L]]$idx
  ng   <- layout[[1L]]$n_groups
  same <- all(vapply(layout, function(b)
    identical(b$idx, idx1) && identical(b$n_groups, ng), logical(1)))
  if (!same) return(NULL)

  nc_terms <- vapply(layout, function(b) b$nc, integer(1))
  dtot     <- sum(nc_terms)
  coef_off <- cumsum(c(0L, nc_terms))
  Zc <- do.call(cbind, lapply(layout, function(b) b$Z))   # n_obs x dtot

  if (is.null(keep)) keep <- rep(TRUE, n_obs)
  if (is.logical(keep)) keep <- which(keep)
  rows_by_g <- lapply(seq_len(ng), function(g) {
    r <- which(idx1 == g)
    r[r %in% keep]
  })

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

  # Per-group posterior mode of b (damped Newton on the penalized integrand) and
  # the precision -H there; arm-agnostic via the caller's `deriv`.
  grp_mode <- function(rows, a_g, Zg, P, deriv) {
    bb <- numeric(dtot)
    for (it in seq_len(50L)) {
      gv <- deriv(rows, cl(a_g + as.numeric(Zg %*% bb)))
      grad <- as.numeric(crossprod(Zg, gv$d1)) - as.numeric(P %*% bb)
      negH <- P - crossprod(Zg, gv$d2 * Zg)
      step <- tryCatch(solve(negH, grad), error = function(e) NULL)
      if (is.null(step)) break
      bb <- bb + step
      if (max(abs(step)) < 1e-9) break
    }
    gv <- deriv(rows, cl(a_g + as.numeric(Zg %*% bb)))
    list(b = bb, negH = P - crossprod(Zg, gv$d2 * Zg))
  }

  # log M_g via adaptive GHQ centred at the mode (probabilist's convention):
  #   b_k = b_hat + L_c z_k,  C = L_c L_c' = (-H)^{-1}
  #   log M_g = -0.5 logdetSigma + log|L_c|
  #             + lse_k[ log w_k + sum_i log f_i(b_k) - 0.5 b_k' Sigma^{-1} b_k
  #                      + 0.5 z_k' z_k ]
  grp_logM <- function(rows, a_g, Zg, P, logdetS, lmat, deriv, want_post = FALSE) {
    m <- grp_mode(rows, a_g, Zg, P, deriv)
    Lc <- tryCatch(t(chol(solve(m$negH))), error = function(e) NULL)
    if (is.null(Lc)) return(NULL)
    B <- matrix(m$b, nrow(Znodes), dtot, byrow = TRUE) + Znodes %*% t(Lc)
    ETA <- cl(matrix(a_g, length(rows), nrow(B)) + Zg %*% t(B))
    hvals <- colSums(lmat(rows, ETA)) - 0.5 * rowSums((B %*% P) * B)
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
    site <- make_site(theta)
    aRE <- cl(site$eta_re)
    total <- 0
    for (g in seq_len(ng)) {
      rows <- rows_by_g[[g]]
      if (!length(rows)) next
      lm <- grp_logM(rows, aRE[rows], Zc[rows, , drop = FALSE], P, logdetS,
                     site$lmat, site$deriv)
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
  site <- make_site(theta_ref)
  aRE <- cl(site$eta_re)
  BHAT <- matrix(0, ng, dtot)
  BVAR <- matrix(rep(diag(Sig), each = ng), ng, dtot)   # empty groups -> prior
  for (g in seq_len(ng)) {
    rows <- rows_by_g[[g]]
    if (!length(rows)) next
    post <- grp_logM(rows, aRE[rows], Zc[rows, , drop = FALSE], P, logdetS,
                     site$lmat, site$deriv, want_post = TRUE)
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
