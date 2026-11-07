#' Grouped random effects by Laplace-EM on a per-group oracle
#'
#' The `n_quad = 1` sibling of [tulpa_re_aghq()] on the same per-group oracle:
#' an expectation-conditional-maximisation (ECM) fit of a model whose
#' conditional likelihood factorises over the levels of one grouping factor.
#' Each iteration
#' \enumerate{
#'   \item solves every group's posterior mode `b_g` and covariance
#'     `C_g = Cov(b_g | y)` at the current parameters (the E-step, the same
#'     per-group solve the AGHQ engine reports as its BLUPs);
#'   \item updates each random-effect covariance in closed form,
#'     `Sigma_k = mean_g(b_gk b_gk' + C_gk)`, projected to its diagonal for an
#'     uncorrelated term and floored at `sigma_floor^2` on its eigenvalues;
#'   \item maximises the Laplace marginal over the fixed parameters `theta` at
#'     that covariance.
#' }
#' It stops when the Laplace log-marginal changes by less than `tol` relative to
#' its size. The fixed-parameter covariance is the inverse curvature of the
#' Laplace marginal in `theta` at the converged covariance, which is the Schur
#' complement of the random-effect block of the joint observed information.
#'
#' @inheritParams tulpa_re_aghq
#' @param sigma_floor Floor on every random-effect standard deviation, applied
#'   to the eigenvalues of each covariance update.
#' @param tol Relative tolerance on the change in the Laplace log-marginal.
#' @param max_iter Maximum EM iterations.
#' @param theta_iter Maximum BFGS iterations of each `theta` step.
#' @return A list with `theta`, `theta_cov`, `theta_se`, `Sigma_list`,
#'   `log_marginal` (Laplace), `blup`, `blup_var`, `blup_cov_g` (per-group
#'   posterior covariances), `blup_cross_g` (mode/theta cross-Hessian blocks,
#'   the `Bf` of [tulpa_re_aghq()]), `group_ok`, `converged`, `n_iter`,
#'   `log_marginal_trace`; `NULL` with a warning when the per-group solve fails
#'   at the start.
#' @seealso [tulpa_re_aghq()] for the exact-marginal (adaptive Gauss-Hermite)
#'   fit on the same oracle.
#' @export
tulpa_re_em <- function(theta0, re_terms, Sigma0,
                        make_site = NULL, make_group = NULL, oracle = NULL,
                        n_obs = NULL, keep = NULL,
                        theta_prior_sd = Inf, sigma_floor = 1e-4,
                        tol = 1e-8, max_iter = 200L, theta_iter = 100L) {
  native <- !is.null(oracle)
  if (!native && (is.null(make_site) == is.null(make_group))) {
    stop("Supply exactly one of `make_site` (single-arm), `make_group` ",
         "(general / multi-arm), or a prebuilt native `oracle`.", call. = FALSE)
  }
  if (!is.null(make_site) && is.null(n_obs)) {
    stop("`n_obs` is required for the `make_site` path.", call. = FALSE)
  }
  if (!is.numeric(sigma_floor) || length(sigma_floor) != 1L ||
      !is.finite(sigma_floor) || sigma_floor <= 0) {
    stop("`sigma_floor` must be a single positive number.", call. = FALSE)
  }
  layout   <- .aghq_layout(re_terms, n_obs)
  ng       <- layout[[1L]]$n_groups
  nc_terms <- vapply(layout, function(b) b$nc, integer(1))
  coef_off <- cumsum(c(0L, nc_terms))
  dtot     <- sum(nc_terms)
  full_vec <- vapply(layout, function(b) isTRUE(b$full), logical(1))
  n_theta  <- length(theta0)
  orc <- .aghq_oracle(layout, n_theta, make_site = make_site,
                      make_group = make_group, oracle = oracle, n_obs = n_obs,
                      keep = keep)
  ridge <- if (is.finite(theta_prior_sd)) 0.5 / theta_prior_sd^2 else 0

  pack <- function(theta, Sigma_list)
    c(theta, .re_cov_L_list_to_theta(lapply(Sigma_list, .re_chol_spd), layout))
  laplace <- function(par) cpp_aghq_objective(par, orc, nc_terms, full_vec, 1L, 1.0)

  theta <- as.numeric(theta0)
  Sigma <- lapply(Sigma0, as.matrix)
  par <- pack(theta, Sigma)
  lm_cur <- laplace(par)
  if (.aghq_is_fail(lm_cur)) {
    gok <- .aghq_group_status(par, orc, nc_terms, full_vec)
    warning("tulpa_re_em: the Laplace marginal is not defined at the starting ",
            "parameters -- the per-group solve failed for ",
            if (is.null(gok)) "at least one group"
            else .aghq_failed_group_phrase(gok),
            ". Returning NULL.", call. = FALSE)
    return(NULL)
  }

  # Closed-form covariance update from the per-group posterior moments.
  sigma_update <- function(bl) {
    lapply(seq_along(layout), function(m) {
      cols <- coef_off[m] + seq_len(nc_terms[m])
      B <- bl$bhat[, cols, drop = FALSE]
      S <- crossprod(B)
      for (g in seq_len(ng))
        S <- S + matrix(bl$bcov[g, cols, cols], nc_terms[m], nc_terms[m])
      S <- S / ng
      if (!full_vec[m]) S <- diag(diag(S), nc_terms[m])
      ee <- eigen((S + t(S)) / 2, symmetric = TRUE)
      ee$vectors %*% (pmax(ee$values, sigma_floor^2) * t(ee$vectors))
    })
  }

  trace <- lm_cur
  converged <- FALSE
  it <- 0L
  for (it in seq_len(max_iter)) {
    bl <- cpp_aghq_blups(par, orc, nc_terms, full_vec)
    if (!all(as.logical(bl$group_ok))) {
      warning("tulpa_re_em: the per-group posterior solve failed for ",
              .aghq_failed_group_phrase(as.logical(bl$group_ok)),
              " at iteration ", it, "; stopping at the last complete ",
              "iterate.", call. = FALSE)
      break
    }
    Sigma <- sigma_update(bl)
    re_par <- .re_cov_L_list_to_theta(lapply(Sigma, .re_chol_spd), layout)
    if (n_theta > 0L) {
      negf <- function(th) -laplace(c(th, re_par)) + ridge * sum(th^2)
      opt <- stats::optim(theta, negf, method = "BFGS",
                          control = list(maxit = theta_iter, reltol = 1e-10))
      theta <- opt$par
    }
    par <- c(theta, re_par)
    lm_new <- laplace(par)
    if (.aghq_is_fail(lm_new)) {
      warning("tulpa_re_em: the Laplace marginal became undefined at iteration ",
              it, "; stopping.", call. = FALSE)
      break
    }
    trace <- c(trace, lm_new)
    if (abs(lm_new - lm_cur) < tol * (1 + abs(lm_cur))) {
      lm_cur <- lm_new; converged <- TRUE; break
    }
    lm_cur <- lm_new
  }

  # Fixed-parameter covariance: inverse curvature of the Laplace marginal in
  # theta at the converged covariance.
  re_par <- par[-seq_len(n_theta)]
  theta_cov <- NULL
  if (n_theta > 0L) {
    negf <- function(th) -laplace(c(th, re_par)) + ridge * sum(th^2)
    H <- tryCatch(stats::optimHess(theta, negf), error = function(e) NULL)
    if (!is.null(H)) theta_cov <- tryCatch(.spd_inverse(H), error = function(e) NULL)
  }
  bl <- cpp_aghq_blups(par, orc, nc_terms, full_vec)
  BHAT <- bl$bhat; BVAR <- bl$bvar; BCOV <- bl$bcov; BCROSS <- bl$bcross
  blup     <- lapply(seq_along(layout), function(m)
    BHAT[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  blup_var <- lapply(seq_along(layout), function(m)
    BVAR[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE])
  list(
    theta        = theta,
    theta_cov    = theta_cov,
    theta_se     = if (is.null(theta_cov)) rep(NA_real_, n_theta)
                   else sqrt(pmax(diag(theta_cov), 0)),
    Sigma_list   = lapply(.re_cov_theta_to_L_list(re_par, layout), tcrossprod),
    log_marginal = lm_cur,
    blup         = blup,
    blup_var     = blup_var,
    blup_cov_g   = lapply(seq_len(ng), function(g)
      array(BCOV[g, , ], dim = c(dtot, dtot))),
    blup_cross_g = lapply(seq_len(ng), function(g)
      array(BCROSS[g, , ], dim = c(n_theta, dtot))),
    group_ok     = as.logical(bl$group_ok),
    converged    = converged,
    n_iter       = it,
    log_marginal_trace = trace
  )
}
