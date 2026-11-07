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
#'   \item moves the fixed parameters `theta` (see `theta_update`).
#' }
#' It stops when the Laplace log-marginal changes by less than `tol` relative to
#' its size.
#'
#' `theta_update = "marginal"` moves `theta` by BFGS on the Laplace marginal at
#' the current covariance, with its exact (envelope) gradient, and reports the
#' inverse curvature of that marginal as the `theta` covariance. Its gradient
#' differentiates each group's Hessian in `theta`, which the engine does by
#' central differences.
#'
#' `theta_update = "latent"` treats `theta` as a latent Gaussian variable with
#' prior `N(0, theta_prior_sd^2)`, as INLA treats fixed effects: `theta` and
#' every group's mode are solved jointly by a damped Newton on the penalized
#' joint log-density (the Hessian is an arrowhead, solved through the Schur
#' complement of the random-effect blocks), and the `theta` covariance is the
#' inverse of that Schur complement. It needs only first and second derivatives
#' of the group log-likelihood, including its `theta` score: a native oracle
#' supplies it, and a `make_group` callback through its optional `joint`
#' closure (below). With a Gaussian conditional and `theta` entering the linear
#' predictor, the two rules reach the same `theta`.
#'
#' @inheritParams tulpa_re_aghq
#' @param make_group As in [tulpa_re_aghq()]; the list it returns may also
#'   carry `joint(g, b)`, returning `list(logL, grad, negH, theta_grad,
#'   theta_info, cross)`: the `grad_hess` blocks plus the data theta score
#'   \eqn{\partial \ell_g / \partial\theta}, the theta observed information
#'   \eqn{-\partial^2 \ell_g / \partial\theta^2} (`n_theta x n_theta`) and the
#'   cross block \eqn{-\partial^2 \ell_g / \partial\theta \partial b}
#'   (`n_theta x d`). `theta_update = "latent"` requires it.
#' @param theta_prior_sd Prior standard deviation of `theta`, one value or one
#'   per coordinate; `Inf` leaves a coordinate unpenalized.
#' @param theta_update `"marginal"` (default) or `"latent"`; see Details.
#' @param b0 Optional `n_groups x d` matrix of starting group modes (`d` the
#'   total random-effect dimension, terms in `re_terms` order). Absent, the
#'   latent rule starts every group at zero and the marginal rule at each
#'   group's own solve.
#' @param newton_max Maximum joint Newton steps per iteration
#'   (`theta_update = "latent"`).
#' @param progress_label Label of the progress line and heartbeat (the
#'   `tulpa.nl_progress` option sets whether they are written).
#' @param aghq_debias Latent rule only: a per-axis node count `> 1` adds one
#'   adaptive Gauss-Hermite moment step at the converged mode. Each group's
#'   posterior second moment is integrated on the AGHQ engine's tensor grid
#'   (centred at the mode, scaled by the group's posterior covariance) and the
#'   covariance update takes their mean, which removes the Laplace
#'   small-cluster attenuation of the variance components; the group
#'   covariances and the `theta` covariance are then recomputed at the new
#'   covariance through the likelihood curvature. `0` (default) skips it.
#' @param sigma_floor Floor on every random-effect standard deviation, applied
#'   to the eigenvalues of each covariance update.
#' @param tol Relative tolerance on the change in the Laplace log-marginal.
#' @param max_iter Maximum EM iterations.
#' @param theta_iter Maximum BFGS steps on `theta` per EM iteration.
#' @param n_threads Threads for the per-group loops, used only when the
#'   oracle declares itself thread-safe. The result does not depend on it.
#' @return A list with `theta`, `theta_cov`, `theta_hessian` (the `theta`
#'   curvature `theta_cov` inverts; `theta_cov` is `NULL` when it is not
#'   positive definite), `theta_se`, `Sigma_list`,
#'   `log_marginal` (Laplace), `blup`, `blup_var`, `blup_cov_g` (per-group
#'   posterior covariances), `blup_cross_g` (mode/theta cross-Hessian blocks,
#'   the `Bf` of [tulpa_re_aghq()]), `group_ok`, `converged`, `n_iter`,
#'   `blup_cross_available`, `log_marginal_trace`, `stuck` (latent rule: the
#'   final joint Newton accepted no step at any damping, so `converged` is
#'   `FALSE`), `aghq_debiased` (whether the `aghq_debias` step ran; it leaves
#'   the Laplace fit in place when a factorization or quadrature sum fails);
#'   `NULL` with a warning when the per-group solve fails at the start.
#' @seealso [tulpa_re_aghq()] for the exact-marginal (adaptive Gauss-Hermite)
#'   fit on the same oracle.
#' @export
tulpa_re_em <- function(theta0, re_terms, Sigma0,
                        make_site = NULL, make_group = NULL, oracle = NULL,
                        n_obs = NULL, keep = NULL,
                        theta_prior_sd = Inf, sigma_floor = 1e-4,
                        tol = 1e-8, max_iter = 200L, theta_iter = 3L,
                        n_threads = 1L,
                        theta_update = c("marginal", "latent"), b0 = NULL,
                        newton_max = 30L, progress_label = "re-em",
                        aghq_debias = 0L) {
  theta_update <- match.arg(theta_update)
  aghq_debias <- as.integer(aghq_debias)
  if (length(aghq_debias) != 1L || is.na(aghq_debias) || aghq_debias < 0L) {
    stop("`aghq_debias` must be a single non-negative node count.", call. = FALSE)
  }
  if (aghq_debias > 1L && theta_update != "latent") {
    stop("`aghq_debias` is a step of the latent theta rule; use ",
         "tulpa_re_aghq() for the exact-marginal fit.", call. = FALSE)
  }
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
  if (!is.numeric(theta_prior_sd) || !(length(theta_prior_sd) %in% c(1L, n_theta)) ||
      anyNA(theta_prior_sd) || any(theta_prior_sd <= 0)) {
    stop("`theta_prior_sd` must be positive, one value or one per theta ",
         "coordinate.", call. = FALSE)
  }
  prec <- rep_len(ifelse(is.finite(theta_prior_sd), 1 / theta_prior_sd^2, 0),
                  n_theta)
  if (!is.null(b0)) {
    b0 <- as.matrix(b0)
    if (!identical(dim(b0), c(as.integer(ng), as.integer(dtot)))) {
      stop("`b0` must be an n_groups x d matrix (", ng, " x ", dtot, ").",
           call. = FALSE)
    }
    storage.mode(b0) <- "double"
  }
  popt <- getOption("tulpa.nl_progress", NULL)
  progress <- list(
    progress = isTRUE(popt$progress),
    file     = if (is.character(popt$progress_file)) popt$progress_file else "",
    every    = if (length(popt$progress_every)) as.integer(popt$progress_every) else 0L,
    throttle = if (length(popt$progress_throttle)) as.numeric(popt$progress_throttle) else 2,
    label    = as.character(progress_label))

  res <- cpp_re_em(orc, as.numeric(theta0), lapply(Sigma0, as.matrix),
                   nc_terms, full_vec, prec, sigma_floor, tol,
                   as.integer(max_iter), as.integer(theta_iter),
                   as.integer(n_threads), as.integer(theta_update == "latent"),
                   b0, as.integer(newton_max), progress, aghq_debias)
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
  # BLUP block. The latent rule's final joint solve carries the modes, their
  # covariances and the theta-b cross blocks itself; the marginal rule reads
  # them (the cross block where the oracle supplies a theta score) through the
  # same extractor as tulpa_re_aghq().
  if (theta_update == "latent") {
    BHAT <- res$bhat; BCOV <- res$bcov; BCROSS <- res$bcross
    BVAR <- matrix(vapply(seq_len(ng), function(g)
      diag(array(BCOV[g, , ], dim = c(dtot, dtot))), numeric(dtot)),
      ng, dtot, byrow = TRUE)
    bl <- list(bcross_available = n_theta > 0L, group_ok = res$group_ok)
  } else {
    par <- c(theta, .re_cov_L_list_to_theta(lapply(Sigma_list, .re_chol_spd), layout))
    bl <- cpp_aghq_blups(par, orc, nc_terms, full_vec)
    BHAT <- bl$bhat; BVAR <- bl$bvar; BCOV <- bl$bcov; BCROSS <- bl$bcross
  }
  list(
    theta        = theta,
    theta_cov    = theta_cov,
    theta_hessian = res$theta_hessian,
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
    stuck        = isTRUE(res$stuck),
    aghq_debiased = isTRUE(res$aghq_debias_ok),
    n_iter       = as.integer(res$n_iter),
    log_marginal_trace = as.numeric(res$trace)
  )
}


#' Closed-form covariance update of a grouped random-effect EM
#'
#' The covariance update every tulpa EM over a grouped random effect uses
#' (`tulpa_re_em()`, and compiled consumers through
#' `<tulpa/re_cov_update.h>`): the group-averaged posterior second moment of one
#' covariance block, symmetrized, projected to its diagonal for an uncorrelated
#' block, with its eigenvalues floored at `sigma_floor^2`.
#'
#' @param M The `c x c` group-averaged posterior second moment
#'   `mean_g(E[b_g] E[b_g]' + Cov(b_g))` of the block.
#' @param correlated `FALSE` keeps only the diagonal.
#' @param sigma_floor Floor on every standard deviation, applied to the
#'   eigenvalues.
#' @return The updated `c x c` covariance.
#' @export
#' @examples
#' M <- matrix(c(0.5, 0.2, 0.2, 1e-9), 2, 2)
#' tulpa_re_cov_update(M)
#' tulpa_re_cov_update(M, correlated = FALSE, sigma_floor = 0.01)
tulpa_re_cov_update <- function(M, correlated = TRUE, sigma_floor = 1e-4) {
  M <- as.matrix(M)
  if (!is.numeric(M) || nrow(M) != ncol(M) || anyNA(M)) {
    stop("`M` must be a square numeric matrix.", call. = FALSE)
  }
  if (!is.numeric(sigma_floor) || length(sigma_floor) != 1L ||
      !is.finite(sigma_floor) || sigma_floor <= 0) {
    stop("`sigma_floor` must be a single positive number.", call. = FALSE)
  }
  storage.mode(M) <- "double"
  cpp_re_cov_update(M, isTRUE(correlated), sigma_floor)
}
