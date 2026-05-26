# Front-door routing of a single correlated `(1 + x | g)` term. The conditional
# Laplace design path has no scalar sigma_re for a covariance, so tulpa()
# redirects to the RE-covariance integrator (nested-Laplace, default) or the
# exact Metropolis-within-Gibbs debias (control$re_cov = "gibbs").

make_corr_re_data <- function(seed = 3L, G = 50L, npg = 14L) {
  set.seed(seed)
  N <- G * npg
  g <- factor(rep(seq_len(G), each = npg))
  x <- rnorm(N)
  Sigma <- matrix(c(0.8^2, 0.3 * 0.8 * 0.5, 0.3 * 0.8 * 0.5, 0.5^2), 2)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), 2))
  eta <- -0.2 + 0.6 * x + u[as.integer(g), 1] + u[as.integer(g), 2] * x
  data.frame(y = rbinom(N, 1L, plogis(eta)), x = x, g = g)
}


test_that("re_cov backends are registered, reachable, and forceable", {
  expect_true(all(c("re_cov_nested", "re_cov_gibbs") %in% ALL_BACKENDS))
  expect_true(backend_is_reachable("re_cov_nested"))
  expect_true(backend_is_reachable("re_cov_gibbs"))
  expect_equal(get_backend_tier("re_cov_nested")$tier, 2L)
  expect_equal(get_backend_tier("re_cov_gibbs")$tier, 1L)
})


test_that("correlated (1 + x | g) routes to re_cov_nested under mode='laplace'", {
  skip_on_cran()
  d <- make_corr_re_data()
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace", control = list(seed = 1L, n_draws = 800L))

  expect_equal(fit$backend, "re_cov_nested")
  expect_equal(fit$inference_tier, 2L)
  expect_match(fit$selection_reason, "correlated random-slope")
  expect_s3_class(fit, "tulpa_fit")

  # coherent fixed-effect accessors (process_info => coef returns the 2 betas)
  cf <- coef(fit)
  expect_equal(unname(lengths(cf)), 2L)
  expect_true(all(c("2.5%", "97.5%") %in% colnames(confint(fit))))
  expect_equal(dim(vcov(fit)), c(2L, 2L))
  expect_equal(nrow(fit$draws), 800L)

  # Sigma posterior present with all derived parameters
  expect_setequal(fit$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
  # interval ordering + rho in range
  expect_true(all(fit$posterior$ci_lo <= fit$posterior$median + 1e-8))
  rho <- fit$posterior[fit$posterior$parameter == "rho_12", ]
  expect_gte(rho$ci_lo, -1); expect_lte(rho$ci_hi, 1)
})


test_that("control$re_cov='gibbs' routes to the exact Sigma debias", {
  skip_on_cran()
  d <- make_corr_re_data()
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "laplace",
               control = list(re_cov = "gibbs", n_iter = 500L, n_burnin = 250L,
                              seed = 1L))
  expect_equal(fit$backend, "re_cov_gibbs")
  expect_equal(fit$inference_tier, 1L)
  expect_false(is.null(fit$beta_draws))
  expect_equal(ncol(fit$draws), 2L)
  expect_setequal(fit$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
})


test_that("forcing mode='re_cov_nested' resolves directly", {
  skip_on_cran()
  d <- make_corr_re_data(seed = 4L, G = 30L, npg = 12L)
  fit <- tulpa(y ~ x + (1 + x | g), data = d, family = "binomial",
               mode = "re_cov_nested", control = list(seed = 2L, n_draws = 400L))
  expect_equal(fit$backend, "re_cov_nested")
})


test_that("uncorrelated (1 + x || g) still errors on the design path", {
  d <- make_corr_re_data(seed = 5L, G = 20L, npg = 10L)
  expect_error(
    suppressMessages(
      tulpa(y ~ x + (1 + x || g), data = d, family = "binomial",
            mode = "laplace")),
    "random slopes")
})
