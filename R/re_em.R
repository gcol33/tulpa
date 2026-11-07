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
#'   \item moves the fixed parameters `theta` by BFGS on the Laplace marginal
#'     at that covariance, with its exact (envelope) gradient.
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
#' @param theta_iter Maximum BFGS steps on `theta` per EM iteration.
#' @param n_threads Threads for the per-group loops, used only when the
#'   oracle declares itself thread-safe. The result does not depend on it.
#' @return A list with `theta`, `theta_cov`, `theta_se`, `Sigma_list`,
#'   `log_marginal` (Laplace), `blup`, `blup_var`, `blup_cov_g` (per-group
#'   posterior covariances), `blup_cross_g` (mode/theta cross-Hessian blocks,
#'   the `Bf` of [tulpa_re_aghq()]), `group_ok`, `converged`, `n_iter`,
#'   `blup_cross_available`, `log_marginal_trace`; `NULL` with a warning when
#'   the per-group solve fails at the start.
#' @seealso [tulpa_re_aghq()] for the exact-marginal (adaptive Gauss-Hermite)
#'   fit on the same oracle.
#' @export
tulpa_re_em <- function(theta0, re_terms, Sigma0,
                        make_site = NULL, make_group = NULL, oracle = NULL,
                        n_obs = NULL, keep = NULL,
                        theta_prior_sd = Inf, sigma_floor = 1e-4,
                        tol = 1e-8, max_iter = 200L, theta_iter = 3L,
                        n_threads = 1L) {
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
  prec <- if (is.finite(theta_prior_sd)) 1 / theta_prior_sd^2 else 0

  res <- cpp_re_em(orc, as.numeric(theta0), lapply(Sigma0, as.matrix),
                   nc_terms, full_vec, prec, sigma_floor, tol,
                   as.integer(max_iter), as.integer(theta_iter),
                   as.integer(n_threads))
  if (!isTRUE(res$ok)) {
    par0 <- c(theta0, .re_cov_L_list_to_theta(lapply(Sigma0, .re_chol_spd), layout))
    gok <- .aghq_group_status(par0, orc, nc_terms, full_vec)
    warning("tulpa_re_em: the Laplace marginal is not defined at the starting ",
            "parameters -- the per-group solve failed for ",
            if (is.null(gok)) "at least one group"
            else .aghq_failed_group_phrase(gok),
            ". Returning NULL.", call. = FALSE)
    return(NULL)
  }
  group_ok <- as.logical(res$group_ok)
  if (!all(group_ok)) {
    warning("tulpa_re_em: the per-group posterior solve failed for ",
            .aghq_failed_group_phrase(group_ok), " at the final parameters; ",
            "their blup entries are NA.", call. = FALSE)
  }
  theta <- as.numeric(res$theta)
  theta_cov <- if (n_theta > 0L && isTRUE(res$theta_hessian_ok))
    tryCatch(.spd_inverse(res$theta_hessian), error = function(e) NULL) else NULL
  Sigma_list <- lapply(res$Sigma_list, as.matrix)
  # BLUP block (modes, covariances, and the mode/theta cross-Hessian where the
  # oracle supplies a theta-score) through the same extractor as tulpa_re_aghq().
  par <- c(theta, .re_cov_L_list_to_theta(lapply(Sigma_list, .re_chol_spd), layout))
  bl <- cpp_aghq_blups(par, orc, nc_terms, full_vec)
  BHAT <- bl$bhat; BVAR <- bl$bvar; BCOV <- bl$bcov; BCROSS <- bl$bcross
  list(
    theta        = theta,
    theta_cov    = theta_cov,
    theta_se     = if (is.null(theta_cov)) rep(NA_real_, n_theta)
                   else sqrt(pmax(diag(theta_cov), 0)),
    Sigma_list   = Sigma_list,
    log_marginal = res$log_marginal,
    blup         = lapply(seq_along(layout), function(m)
      BHAT[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE]),
    blup_var     = lapply(seq_along(layout), function(m)
      BVAR[, coef_off[m] + seq_len(nc_terms[m]), drop = FALSE]),
    blup_cov_g   = lapply(seq_len(ng), function(g)
      array(BCOV[g, , ], dim = c(dtot, dtot))),
    blup_cross_g = lapply(seq_len(ng), function(g)
      array(BCROSS[g, , ], dim = c(n_theta, dtot))),
    blup_cross_available = isTRUE(bl$bcross_available),
    group_ok     = as.logical(bl$group_ok),
    converged    = isTRUE(res$converged),
    n_iter       = as.integer(res$n_iter),
    log_marginal_trace = as.numeric(res$trace)
  )
}
