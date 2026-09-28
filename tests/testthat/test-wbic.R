# WBIC (Watanabe 2013): the expected negative log-likelihood under the posterior
# tempered at 1 / log(n). The SMC path passes through that distribution when
# `bridge_end < 1` and records its population there; these tests check the
# population against closed-form tempered posteriors, the criterion against its
# closed form on a conjugate regression, and the doors that refuse it.

# Tempered posterior of x_i ~ N(0, sp^2), with the unnormalized likelihood
# kernel -0.5 (x_i - mu_i)^2 / st_i^2 raised to the power b.
tempered_gauss <- function(mu, st, b, sp = 10) {
  v <- 1 / (1 / sp^2 + b / st^2)
  m <- v * b * mu / st^2
  list(mean = m, sd = sqrt(v),
       mean_loglik = sum(-0.5 * ((m - mu)^2 + v) / st^2))
}

test_that("wbic.default is minus the mean per-draw log-likelihood total", {
  set.seed(1)
  ll <- matrix(rnorm(50 * 8, -1), 50, 8)
  w <- wbic(ll)
  expect_s3_class(w, "tulpa_wbic")
  expect_equal(w$wbic, -mean(rowSums(ll)))
  expect_equal(w$beta, 1 / log(8))
  expect_equal(c(w$n_obs, w$n_draws), c(8L, 50L))
  # Streaming over column blocks gives the same total.
  gen <- tulpa_loglik(function(cols) ll[, cols, drop = FALSE],
                      n_obs = 8L, n_draws = 50L)
  expect_equal(wbic(gen)$wbic, w$wbic)
  expect_output(print(w), "WBIC")
})

test_that("wbic.default refuses what has no WBIC temperature", {
  expect_error(wbic(matrix(-1, 10, 2)), "n >= 3")
  expect_error(wbic("a"), "numeric draws x observations")
  expect_error(wbic(matrix(c(NA, -1), 4, 3)), "NA at")
})

skip_on_cran()

test_that("the two-leg SMC path records the tempered posterior exactly", {
  mu <- c(-1, 0.5, 2)
  st <- c(0.8, 1.2, 0.6)
  b  <- 0.3
  res <- cpp_smc_test(mu_target = mu, sigma_target = st, n_particles = 4000L,
                      n_mcmc_steps = 8L, seed = 3L, bridge_end = b)
  tr <- tempered_gauss(mu, st, b)
  expect_equal(as.numeric(res$tempered_means), tr$mean, tolerance = 0.05)
  expect_equal(as.numeric(res$tempered_sds), tr$sd, tolerance = 0.05)
  expect_equal(res$tempered_mean_loglik, tr$mean_loglik, tolerance = 0.05)

  # ...and the second leg still ends at the posterior.
  post <- tempered_gauss(mu, st, 1)
  expect_equal(as.numeric(res$means), post$mean, tolerance = 0.1)
  expect_equal(as.numeric(res$sds), post$sd, tolerance = 0.1)

  # The path coordinate crosses 1 at the leg boundary and ends at 2.
  temps <- res$temperatures
  expect_true(1 %in% temps)
  expect_equal(temps[length(temps)], 2)
  expect_true(all(diff(temps) > 0))
})

test_that("bridge_end = 1 is the one-leg path and records nothing", {
  a <- cpp_smc_test(mu_target = c(0, 1), sigma_target = c(1, 1),
                    n_particles = 300L, n_mcmc_steps = 3L, seed = 7L)
  b <- cpp_smc_test(mu_target = c(0, 1), sigma_target = c(1, 1),
                    n_particles = 300L, n_mcmc_steps = 3L, seed = 7L,
                    bridge_end = 1)
  expect_identical(a, b)
  expect_length(a$tempered_means, 0L)
  expect_true(is.na(a$tempered_mean_loglik))
  expect_error(cpp_smc_test(mu_target = 0, sigma_target = 1, bridge_end = 0),
               "bridge_end")
})

# Gaussian regression with the variance fixed and a N(0, sigma_beta^2) prior on
# every coefficient: the tempered posterior is Gaussian and the WBIC has a
# closed form,
#   n/2 log(2 pi s2) + (||y - X m||^2 + tr(X V X')) / (2 s2).
wbic_gauss_truth <- function(X, y, s2, sigma_beta) {
  n <- nrow(X)
  b <- 1 / log(n)
  V <- solve(diag(ncol(X)) / sigma_beta^2 + b * crossprod(X) / s2)
  m <- V %*% (b * crossprod(X, y) / s2)
  r <- y - X %*% m
  0.5 * n * log(2 * pi * s2) +
    (sum(r^2) + sum(diag(X %*% V %*% t(X)))) / (2 * s2)
}

test_that("wbic() on a front-door fit matches the conjugate closed form", {
  set.seed(4)
  n <- 150L
  d <- data.frame(x = rnorm(n))
  d$y <- 0.4 + 0.8 * d$x + rnorm(n, 0, 0.6)
  fit <- tulpa(y ~ x, d, family = "gaussian", phi = 0.36, mode = "smc",
               control = list(wbic = TRUE, n_particles = 2000L, seed = 2L))
  expect_equal(colnames(fit$tempered_draws), colnames(fit$draws))
  expect_equal(fit$tempered_beta, 1 / log(n))

  X <- cbind(1, d$x)
  truth <- wbic_gauss_truth(X, d$y, 0.36, fit$model_inputs$sigma_beta)
  w <- wbic(fit)
  expect_equal(w$beta, 1 / log(n))
  expect_equal(w$wbic, truth, tolerance = 0.5 / truth)

  # The tempered draws sit on the tempered posterior, wider than the
  # posterior the fit's own draws come from.
  expect_gt(stats::sd(fit$tempered_draws[, 2]), 1.5 * stats::sd(fit$draws[, 2]))
  V1 <- solve(diag(2) / fit$model_inputs$sigma_beta^2 + crossprod(X) / 0.36)
  m1 <- V1 %*% crossprod(X, d$y) / 0.36
  expect_equal(unname(colMeans(fit$draws)), as.numeric(m1), tolerance = 0.03)
})

test_that("the RE posterior survives the detour through the tempered one", {
  # The fixture and HMC reference of test-sample-glmm-structure.R: sigma_re
  # median 0.69, 90% interval (0.52, 0.94); slope 0.453.
  set.seed(11)
  J <- 20; n <- 200; gi <- sample(J, n, TRUE); u <- rnorm(J, 0, 0.7)
  x <- rnorm(n)
  d <- data.frame(y = 1 + .5 * x + u[gi] + rnorm(n, 0, .5), x, g = factor(gi))
  f <- suppressWarnings(tulpa(y ~ x + (1 | g), d, phi = .25, mode = "smc",
                              control = list(seed = 1L, wbic = TRUE)))
  s <- exp(f$draws[, "log_sigma_re"])
  expect_gt(stats::median(s), 0.55)
  expect_lt(stats::median(s), 0.85)
  expect_equal(mean(f$draws[, "x"]), 0.453, tolerance = 0.1)
  w <- wbic(f)
  expect_true(is.finite(w$wbic))
  expect_equal(nrow(f$tempered_draws), nrow(f$draws))
})

test_that("compare_models(criterion = 'wbic') ranks by the free energy", {
  set.seed(5)
  n <- 120L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.7 * d$x))
  ctl <- list(wbic = TRUE, n_particles = 800L, seed = 1L)
  full <- tulpa(y ~ x, d, family = "poisson", mode = "smc", control = ctl)
  null <- tulpa(y ~ 1, d, family = "poisson", mode = "smc", control = ctl)
  plain <- tulpa(y ~ x, d, family = "poisson", mode = "smc",
                 control = list(n_particles = 300L, seed = 1L))
  cm <- compare_models(full = full, null = null, plain = plain,
                       criterion = "wbic")
  expect_equal(cm$model, c("full", "null", "plain"))
  expect_equal(cm$delta[1], 0)
  expect_gt(cm$delta[2], 5)
  expect_true(is.na(cm$wbic[3]))
  expect_equal(sum(cm$weight, na.rm = TRUE), 1)
})

test_that("the tulpa_smc_fit C entry returns the tempered population", {
  set.seed(7)
  n <- 60L
  x <- rnorm(n)
  y <- rpois(n, exp(0.2 + 0.5 * x))
  X <- cbind(1, x)
  b <- 1 / log(n)
  cabi <- tulpa:::cpp_test_smc_cabi(y, rep(1L, n), X, "poisson", phi = 1,
                                    sigma_beta = 2.5, n_particles = 300L,
                                    seed = 4L, bridge_end = b)
  direct <- tulpa:::cpp_tulpa_sample_glmm(
    y = y, n_trials = rep(1L, n), X = X, family = "poisson", backend = "smc",
    phi = 1, sigma_beta = 2.5, seed = 4L, n_particles = 300L,
    n_mcmc_steps = 5L, ess_threshold = 0.5, smc_bridge_end = b)
  expect_equal(cabi$tempered_beta, b)
  expect_identical(unname(cabi$tempered_draws), unname(direct$tempered_draws))
  expect_identical(unname(cabi$draws), unname(direct$draws))

  plain <- tulpa:::cpp_test_smc_cabi(y, rep(1L, n), X, "poisson", phi = 1,
                                     sigma_beta = 2.5, n_particles = 100L,
                                     seed = 4L, bridge_end = 1)
  expect_null(plain$tempered_draws)
  expect_true(plain$tempered_null)
  expect_true(is.nan(plain$tempered_beta))
})

test_that("the WBIC knob is refused wherever it would be dropped", {
  set.seed(6)
  d <- data.frame(x = rnorm(40))
  d$y <- rpois(40, exp(0.2 + 0.3 * d$x))
  expect_error(tulpa(y ~ x, d, family = "poisson", mode = "hmc",
                     control = list(wbic = TRUE)),
               "mode = 'smc' only")
  expect_error(tulpa(y ~ x, d, family = "poisson", mode = "smc",
                     control = list(wbic = "yes")),
               "TRUE or FALSE")
  plain <- tulpa(y ~ x, d, family = "poisson", mode = "smc",
                 control = list(n_particles = 200L, seed = 1L))
  expect_null(plain$tempered_draws)
  expect_null(plain$tempered_beta)
  expect_error(wbic(plain), "control = list\\(wbic = TRUE\\)")
})
