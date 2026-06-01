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
  expect_match(fit$selection_reason, "random-slope term")
  expect_s3_class(fit, "tulpa_fit")

  # coherent fixed-effect accessors (process_info => coef returns the 2 betas)
  cf <- coef(fit)
  expect_length(cf, 2L)
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


test_that("uncorrelated (1 + x || g) routes to a diagonal-Sigma integration", {
  skip_on_cran()
  d <- make_corr_re_data(seed = 5L, G = 40L, npg = 12L)
  fit <- tulpa(y ~ x + (1 + x || g), data = d, family = "binomial",
               mode = "laplace", control = list(seed = 1L, n_draws = 400L))

  expect_equal(fit$backend, "re_cov_nested")
  expect_equal(fit$inference_tier, 2L)
  # diagonal block: marginal SDs and diagonal variances only -- NO correlation
  expect_setequal(fit$posterior$parameter,
                  c("sigma_1", "sigma_2", "Sigma_11", "Sigma_22"))
  expect_false(any(grepl("rho", fit$posterior$parameter)))
  expect_false(any(grepl("Sigma_12", fit$posterior$parameter)))
  # fixed-effect accessors still coherent
  expect_length(coef(fit), 2L)
  expect_equal(dim(vcov(fit)), c(2L, 2L))
})


# --- multi-term: (1 + x | g) + (1 | h), correlated block + scalar block ------
make_multi_re_data <- function(seed = 7L, G = 40L, H = 25L, npg = 12L) {
  set.seed(seed)
  N <- G * npg
  g <- factor(rep(seq_len(G), each = npg))
  h <- factor(sample.int(H, N, replace = TRUE))
  x <- rnorm(N)
  Sg <- matrix(c(0.8^2, 0.3 * 0.8 * 0.5, 0.3 * 0.8 * 0.5, 0.5^2), 2)
  ug <- t(t(chol(Sg)) %*% matrix(rnorm(2 * G), 2))
  uh <- rnorm(H, 0, 0.6)
  eta <- -0.2 + 0.6 * x + ug[as.integer(g), 1] + ug[as.integer(g), 2] * x +
    uh[as.integer(h)]
  data.frame(y = rbinom(N, 1L, plogis(eta)), x = x, g = g, h = h)
}


test_that("multi-term (1 + x | g) + (1 | h) integrates one block per term", {
  skip_on_cran()
  d <- make_multi_re_data()
  fit <- tulpa(y ~ x + (1 + x | g) + (1 | h), data = d, family = "binomial",
               mode = "laplace", control = list(seed = 2L, n_draws = 400L))

  expect_equal(fit$backend, "re_cov_nested")
  expect_match(fit$selection_reason, "2 block")
  # block-prefixed parameters: g is a full 2x2 Sigma, h a scalar variance
  expect_setequal(fit$posterior$parameter,
                  c("g.sigma_1", "g.sigma_2", "g.rho_12",
                    "g.Sigma_11", "g.Sigma_12", "g.Sigma_22",
                    "h.sigma_1", "h.Sigma_11"))
  # Sigma_mean is a named per-block list
  expect_named(fit$Sigma_mean, c("g", "h"))
  expect_equal(dim(fit$Sigma_mean$g), c(2L, 2L))
  expect_equal(dim(fit$Sigma_mean$h), c(1L, 1L))
  # fixed effects coherent
  expect_length(coef(fit), 2L)
  expect_equal(nrow(fit$draws), 400L)
})


test_that("multi-term routes to the Gibbs debias via control$re_cov", {
  skip_on_cran()
  d <- make_multi_re_data(seed = 8L, G = 30L, H = 18L, npg = 12L)
  fit <- tulpa(y ~ x + (1 + x | g) + (1 | h), data = d, family = "binomial",
               mode = "laplace",
               control = list(re_cov = "gibbs", n_iter = 400L, n_burnin = 200L,
                              seed = 1L))
  expect_equal(fit$backend, "re_cov_gibbs")
  expect_equal(fit$inference_tier, 1L)
  expect_setequal(fit$posterior$parameter,
                  c("g.sigma_1", "g.sigma_2", "g.rho_12",
                    "g.Sigma_11", "g.Sigma_12", "g.Sigma_22",
                    "h.sigma_1", "h.Sigma_11"))
  expect_equal(ncol(fit$draws), 2L)
})
