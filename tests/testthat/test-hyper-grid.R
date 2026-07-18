# Generic outer hyperparameter-grid integrator (gcol33/tulpa#33).
#
# The driver wires a user-supplied inner-fit callback through the outer
# Cartesian grid + weight normalisation + posterior summary + law-of-total-
# covariance pooling. These tests pin the contract:
#   * structural correctness on a trivial flat-posterior inner_fit (weights,
#     names, shapes);
#   * hyperparameter recovery on a Gaussian-likelihood toy where the inner
#     fit returns the exact analytical log marginal and beta posterior;
#   * law-of-total-covariance vs the analytical answer (mixture of identical
#     Gaussian components: V_total = Cov(beta_means) + E[V_inner]);
#   * combine = "weighted_mean_only" / "none" gating;
#   * failed cells are mapped to weight 0 without aborting the integration.

# -----------------------------------------------------------------------------
# Toy inner_fit: Gaussian likelihood y ~ N(X beta, sigma^2 I), beta ~ N(0, tau^2 I).
# Hyperparameter is sigma; beta posterior is Gaussian closed-form. The log
# marginal up to a sigma-independent constant is the integrated-out density.
# -----------------------------------------------------------------------------
make_gaussian_inner_fit <- function(y, X, tau = 100) {
  p <- ncol(X)
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)
  prior_prec <- diag(1 / tau^2, p)
  n <- length(y)
  function(hypers) {
    sigma <- as.numeric(hypers["sigma"])
    if (!is.finite(sigma) || sigma <= 0) {
      return(list(log_marginal = -Inf))
    }
    Lambda <- XtX / sigma^2 + prior_prec
    V <- tryCatch(solve(Lambda), error = function(e) NULL)
    if (is.null(V)) return(list(log_marginal = -Inf))
    mu <- as.numeric(V %*% (Xty / sigma^2))
    resid <- y - as.numeric(X %*% mu)
    log_lik <- -0.5 * n * log(2 * pi) - n * log(sigma) -
               0.5 * sum(resid^2) / sigma^2
    log_prior_beta <- -0.5 * p * log(2 * pi) - p * log(tau) -
                      0.5 * sum(mu^2) / tau^2
    # Laplace marginal: log p(y|sigma) = log_lik + log_prior - 0.5 log|Lambda|
    #                                  + 0.5 p log(2 pi) at the mode.
    log_marg <- log_lik + log_prior_beta + 0.5 * p * log(2 * pi) -
                0.5 * as.numeric(determinant(Lambda,
                                              logarithm = TRUE)$modulus)
    list(log_marginal = log_marg,
         beta_mean    = stats::setNames(mu, paste0("beta", seq_len(p))),
         beta_cov     = V)
  }
}

# -----------------------------------------------------------------------------
# Structural smoke
# -----------------------------------------------------------------------------
test_that("tulpa_hyper_grid wires axes / weights / posterior fields", {
  specs <- list(
    hyper_axis_spec("alpha", grid = c(-1, 0, 1)),
    hyper_axis_spec("sigma", grid = c(0.5, 1, 2), log_scale = TRUE,
                    bounds = c(0, Inf))
  )
  # Flat inner: every cell has the same log marginal. Posterior over
  # hyperparameters should be uniform over the grid.
  inner_fit <- function(hypers) list(log_marginal = 0,
                                      beta_mean = c(b1 = 0),
                                      beta_cov  = matrix(1, 1, 1))
  res <- tulpa_hyper_grid(specs, inner_fit, n_draws = 0L)

  expect_s3_class(res, "tulpa_hyper_grid")
  expect_s3_class(res, "tulpa_fit")
  expect_equal(res$n_grid, 9L)
  expect_equal(dim(res$theta_grid), c(9L, 2L))
  expect_equal(colnames(res$theta_grid), c("alpha", "sigma"))
  expect_equal(res$theta_names, c("alpha", "sigma"))
  expect_equal(sum(res$weights), 1, tolerance = 1e-12)
  expect_true(all(abs(res$weights - 1 / 9) < 1e-10))

  # Hyper moments: uniform over a symmetric grid -> mean at centre.
  expect_equal(unname(res$theta_mean[["alpha"]]), 0, tolerance = 1e-10)
  expect_true(res$theta_sd[["alpha"]] > 0)
  expect_true(res$theta_sd[["sigma"]] > 0)
  expect_named(res$theta_median, c("alpha", "sigma"))
  expect_named(res$theta_ci_lo,  c("alpha", "sigma"))
  expect_named(res$theta_ci_hi,  c("alpha", "sigma"))

  # Combine = law_of_total_cov: weighted mean / cov returned even at n_draws=0.
  expect_equal(unname(res$beta), 0, tolerance = 1e-12)
  expect_equal(unname(res$beta_cov[1, 1]), 1, tolerance = 1e-12)
  expect_null(res$draws)
  expect_equal(res$param_names, "b1")
  expect_equal(res$n_failed, 0L)
})

test_that("log_prior tracks the refined grid, not the initial one (#198)", {
  set.seed(1)
  n <- 40L; X <- cbind(1, rnorm(n)); y <- as.numeric(X %*% c(0.5, -1) + rnorm(n))
  XtX <- crossprod(X); Xty <- crossprod(X, y)
  inner_fit <- function(hypers) {
    sigma <- as.numeric(hypers["sigma"])
    if (!is.finite(sigma) || sigma <= 0) return(list(log_marginal = -Inf))
    Lam <- XtX / sigma^2 + diag(1e-4, 2); V <- solve(Lam)
    mu <- as.numeric(V %*% (Xty / sigma^2))
    resid <- y - as.numeric(X %*% mu)
    list(log_marginal = -n * log(sigma) - 0.5 * sum(resid^2) / sigma^2,
         beta_mean = stats::setNames(mu, c("b0", "b1")), beta_cov = V)
  }
  specs <- list(hyper_axis_spec(
    "sigma", grid = c(0.2, 0.5, 1, 2), log_scale = TRUE, bounds = c(0, Inf),
    refinable = TRUE, log_prior = function(s) dexp(s, 1, log = TRUE)))

  res <- tulpa_hyper_grid(
    specs, inner_fit, combine = "law_of_total_cov", n_draws = 400L, seed = 7L,
    control = list(adaptive_grid = TRUE, adaptive_grid_edge_thresh = 1e-6,
                   adaptive_grid_max_passes = 3L))

  expect_gt(nrow(res$theta_grid), 4L)                        # refinement fired
  # The returned per-cell log-prior must span the FINAL grid...
  expect_length(res$log_prior, nrow(res$theta_grid))
  # ...and the per-draw hyper log-prior must not be NA (stale-index symptom).
  expect_false(anyNA(res$hyper_log_prior_draws))
  expect_length(res$hyper_log_prior_draws, 400L)
})

test_that("auto-wrapping accepts plain list specs", {
  specs <- list(
    list(name = "sigma", grid = c(0.5, 1, 2), log_scale = TRUE),
    list(name = "rho",   grid = c(0.2, 0.5, 0.8), bounds = c(0, 1))
  )
  inner_fit <- function(hypers) list(log_marginal = 0)
  res <- tulpa_hyper_grid(specs, inner_fit, combine = "none")
  expect_equal(res$theta_names, c("sigma", "rho"))
  expect_s3_class(res$hyper_specs[[1L]], "tulpa_hyper_axis_spec")
})

# -----------------------------------------------------------------------------
# Hyperparameter recovery: Gaussian likelihood, sigma grid covers the truth.
# Posterior should concentrate around the simulated sigma; the marginal
# weighted-mean / median should be close to the truth.
# -----------------------------------------------------------------------------
test_that("hyperparameter recovery on Gaussian likelihood", {
  skip_on_cran()
  set.seed(7L)
  n <- 400L; p <- 3L
  X <- cbind(1, matrix(rnorm(n * (p - 1L)), n, p - 1L))
  beta_true <- c(0.5, -1.2, 0.7)
  sigma_true <- 0.8
  y <- as.numeric(X %*% beta_true) + rnorm(n, 0, sigma_true)

  specs <- list(
    hyper_axis_spec("sigma",
                    grid = exp(seq(log(0.3), log(2), length.out = 25L)),
                    log_scale = TRUE, bounds = c(0, Inf))
  )
  res <- tulpa_hyper_grid(specs, make_gaussian_inner_fit(y, X), n_draws = 0L)

  expect_true(abs(res$theta_median[["sigma"]] - sigma_true) < 0.05)
  expect_true(abs(res$theta_mean[["sigma"]]   - sigma_true) < 0.05)
  expect_true(res$theta_ci_lo[["sigma"]] <
              res$theta_median[["sigma"]])
  expect_true(res$theta_ci_hi[["sigma"]] >
              res$theta_median[["sigma"]])

  # Beta recovery via law-of-total-cov.
  expect_equal(unname(res$beta), beta_true, tolerance = 0.1)
  # Diagonal SEs are sane (positive, on the right order of magnitude).
  ses <- sqrt(diag(res$beta_cov))
  expect_true(all(ses > 0))
  expect_true(all(ses < 1))
})

# -----------------------------------------------------------------------------
# Law-of-total-covariance: when every cell has the same beta_cov V_0 and the
# weighted variance of beta_means is V_means, total V = V_0 + V_means.
# -----------------------------------------------------------------------------
test_that("law-of-total-covariance matches analytical mixture variance", {
  # 3 cells, equal weight; beta_mean varies, beta_cov constant.
  specs <- list(hyper_axis_spec("theta", grid = c(-1, 0, 1)))
  V0    <- matrix(c(1, 0.3, 0.3, 1), 2, 2)
  means <- list(c(-2, 0.5), c(0, 1.0), c(2, 1.5))
  inner_fit <- function(hypers) {
    k <- match(round(hypers["theta"], 6), c(-1, 0, 1))
    list(log_marginal = 0, beta_mean = means[[k]], beta_cov = V0)
  }
  res <- tulpa_hyper_grid(specs, inner_fit, n_draws = 0L)

  expect_equal(res$weights, rep(1/3, 3L), tolerance = 1e-12)
  expected_mean <- colMeans(do.call(rbind, means))
  expect_equal(unname(res$beta), expected_mean, tolerance = 1e-12)

  Bmat <- do.call(rbind, means)
  cov_means <- crossprod(sweep(Bmat, 2, expected_mean)) / 3L  # population cov, 1/n
  expected_total <- V0 + cov_means
  expect_equal(unname(res$beta_cov), unname(expected_total),
               tolerance = 1e-12)
})

# -----------------------------------------------------------------------------
# combine = "weighted_mean_only" pools means but skips beta_cov / draws.
# combine = "none" returns the hyperparameter posterior with no beta fields.
# -----------------------------------------------------------------------------
test_that("combine modes gate the fixed-effect outputs", {
  specs <- list(hyper_axis_spec("a", grid = c(0, 1)))

  # weighted_mean_only: beta_cov not required.
  res_wm <- tulpa_hyper_grid(
    specs,
    function(hypers) list(log_marginal = 0,
                           beta_mean = c(b = as.numeric(hypers["a"]))),
    combine = "weighted_mean_only", n_draws = 0L
  )
  expect_equal(unname(res_wm$beta), 0.5, tolerance = 1e-12)
  expect_null(res_wm$beta_cov)
  expect_null(res_wm$draws)

  # none: beta fields all NULL.
  res_n <- tulpa_hyper_grid(
    specs,
    function(hypers) list(log_marginal = 0),
    combine = "none"
  )
  expect_null(res_n$beta)
  expect_null(res_n$beta_cov)
  expect_null(res_n$draws)
  expect_null(res_n$param_names)

  # Contract: "law_of_total_cov" with missing beta_cov errors clearly.
  expect_error(
    tulpa_hyper_grid(
      specs,
      function(hypers) list(log_marginal = 0,
                             beta_mean = c(b = 0)),
      combine = "law_of_total_cov"
    ),
    "NULL `beta_cov`"
  )
})

# -----------------------------------------------------------------------------
# Failed cells contribute weight 0 and don't abort the integration.
# -----------------------------------------------------------------------------
test_that("failed inner_fit cells get weight 0", {
  specs <- list(hyper_axis_spec("a", grid = c(0, 1, 2)))
  inner_fit <- function(hypers) {
    a <- hypers["a"]
    if (a == 1) stop("simulated failure")        # error path
    if (a == 2) list(log_marginal = NaN)         # non-finite path
    else        list(log_marginal = 0)           # the only good cell
  }
  res <- expect_warning(
    tulpa_hyper_grid(specs, inner_fit, combine = "none"),
    NA   # no warning expected; the surviving cell still carries mass
  )
  expect_equal(res$n_failed, 2L)
  expect_equal(res$weights, c(1, 0, 0), tolerance = 1e-12)
  expect_equal(unname(res$theta_mean[["a"]]), 0, tolerance = 1e-12)
})

# -----------------------------------------------------------------------------
# Spec validation
# -----------------------------------------------------------------------------
test_that("hyper_axis_spec validates its inputs", {
  expect_error(hyper_axis_spec("", grid = 1),       "non-empty")
  expect_error(hyper_axis_spec("a", grid = numeric(0)), "non-empty")
  expect_error(hyper_axis_spec("a", grid = c(-0.5, 1),  log_scale = TRUE),
               "negative")
  # `0` is allowed in a log-scale grid (no-effect atom).
  expect_silent(hyper_axis_spec("a", grid = c(0, 1, 2), log_scale = TRUE))
  expect_error(hyper_axis_spec("a", grid = c(0, 1),  bounds = c(0.5, 0.5)),
               "lower < upper")
  expect_error(hyper_axis_spec("a", grid = c(0, 1),  bounds = c(2, 3)),
               "outside `bounds`")
  expect_error(hyper_axis_spec("a", grid = 1,
                                log_prior = "not a function"),
               "must be NULL or a function")
})

# -----------------------------------------------------------------------------
# Adaptive refinement
# -----------------------------------------------------------------------------
test_that("adaptive_grid extends a too-narrow boundary", {
  # The inner_fit's mode (a unimodal Gaussian in mu) sits PAST the user's
  # initial grid endpoint; the boundary edge score should fire, and refinement
  # should append cells on the heavy side.
  inner_fit <- function(hypers) {
    mu <- as.numeric(hypers["mu"])
    list(log_marginal = -0.5 * (mu - 5)^2 / 0.3^2)
  }
  specs <- list(hyper_axis_spec("mu",
                                 grid = c(2.0, 2.5, 3.0, 3.5),
                                 refinable = TRUE))
  res <- tulpa_hyper_grid(specs, inner_fit, combine = "none",
                          control = list(adaptive_grid = TRUE,
                                          adaptive_grid_max_passes = 2L))
  expect_gt(nrow(res$theta_grid), 4L)
  expect_true("mu" %in% res$adaptive_grid_info$triggered_axes ||
              any(grepl("mu", res$adaptive_grid_info$triggered_axes)))
  # New cells appear on the heavy (max) side of the original grid.
  expect_true(any(res$theta_grid[, "mu"] > 3.5))
  # The refined posterior median should track the true mode closer than the
  # initial-grid posterior would have.
  expect_true(res$theta_median[["mu"]] > 3.5)
})

test_that("adaptive_grid skips non-refinable axes", {
  inner_fit <- function(hypers) list(log_marginal = 0)
  specs <- list(hyper_axis_spec("a", grid = c(0, 1, 2)))  # refinable = FALSE
  res <- tulpa_hyper_grid(specs, inner_fit, combine = "none",
                          control = list(adaptive_grid = TRUE))
  expect_equal(nrow(res$theta_grid), 3L)
  expect_null(res$adaptive_grid_info)
})

test_that("var_of_means_consistency adds slice points on a sharp posterior", {
  # Gaussian in log-sigma at 0 (sigma=1) with SD 0.2 on the log axis. A coarse
  # 5-point grid at log_sigma = (-1, -0.5, 0, 0.5, 1) collapses var-of-means
  # SD onto two adjacent cells -- well below the Laplace-at-mode SD -- so the
  # consistency pass fires and adds Laplace-guided slice points at
  # `mu +/- {0.7, 1.5} * sd_lap` on the log axis (the 0.05 log-tolerance dedup
  # in .hyper_propose_consistency_points keeps them).
  inner_fit <- function(hypers) {
    s <- as.numeric(hypers["sigma"])
    list(log_marginal = -0.5 * (log(s) - log(1.0))^2 / 0.1^2)
  }
  specs <- list(hyper_axis_spec(
    "sigma", grid = exp(seq(-1, 1, length.out = 5L)),
    log_scale = TRUE, bounds = c(0, Inf), refinable = TRUE))
  res <- tulpa_hyper_grid(specs, inner_fit, combine = "none",
                          control = list(var_of_means_consistency = TRUE))
  expect_true(!is.null(res$var_of_means_consistency_info))
  expect_gt(nrow(res$theta_grid), 5L)
  expect_gt(res$theta_sd[["sigma"]], 0)
})

test_that("duplicate axis names are rejected", {
  expect_error(
    tulpa_hyper_grid(list(
      hyper_axis_spec("sigma", grid = c(0.5, 1)),
      hyper_axis_spec("sigma", grid = c(1, 2))
    ), function(hypers) list(log_marginal = 0)),
    "Duplicate axis names"
  )
})
