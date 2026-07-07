# Second dispersion channel phi2 (gcol33/tulpa C7): configurable Student-t df
# through the registry, the Laplace kernel, and the samplers; the lognormal
# family registered with the variance convention; and the gaussian phi
# convention unified across backends (tulpa()'s phi = residual VARIANCE
# everywhere -- the SD-parameterized kernels receive sqrt(phi) at the
# boundary, so mode = 'laplace' and the R-side H_beta now describe the same
# model).

test_that("t-family registry ops honor phi2 and match dt()", {
  eta <- c(-0.5, 0.2, 1.4)
  y   <- c(0.3, -1.1, 2.0)
  s   <- 0.7

  for (nu in c(3, 10, 25)) {
    ll <- tulpa:::family_loglik(eta, y, "t", phi = s, phi2 = nu)
    expect_equal(ll, stats::dt((y - eta) / s, df = nu, log = TRUE) - log(s),
                 tolerance = 1e-12)
  }
  # Default df = 4 when phi2 is NULL.
  expect_equal(tulpa:::family_loglik(eta, y, "t", phi = s),
               stats::dt((y - eta) / s, df = 4, log = TRUE) - log(s),
               tolerance = 1e-12)
  # Score matches a numerical derivative at phi2 = 7.
  h <- 1e-6
  num <- (tulpa:::family_loglik(eta + h, y, "t", phi = s, phi2 = 7) -
          tulpa:::family_loglik(eta - h, y, "t", phi = s, phi2 = 7)) / (2 * h)
  expect_equal(tulpa:::family_score_eta(eta, y, "t", phi = s, phi2 = 7), num,
               tolerance = 1e-6)
  # Variance: phi^2 nu / (nu - 2).
  expect_equal(tulpa:::family_variance(eta, "t", phi = s, phi2 = 10),
               rep(s^2 * 10 / 8, 3))
})

test_that("phi2 is rejected for families without a second dispersion", {
  expect_error(tulpa:::family_loglik(0, 1, "gaussian", phi = 1, phi2 = 5),
               "no second dispersion")
  d <- data.frame(x = rnorm(30)); d$y <- rnorm(30)
  expect_error(
    tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi2 = 5),
    "no second dispersion")
  d$y2 <- rbinom(30, 1, 0.5)
  expect_error(
    tulpa(y2 ~ x, data = d, family = "binomial", mode = "laplace", phi2 = -1),
    "phi2")
})

test_that("Laplace t fit with phi2 matches an optim reference and differs from df 4", {
  skip_on_cran()
  set.seed(81)
  n <- 200L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(1, -0.6)) + 0.8 * rt(n, df = 10)
  d <- data.frame(y = y, x = X[, 2])

  fit10 <- tulpa(y ~ x, data = d, family = "t", mode = "laplace",
                 phi = 0.8, phi2 = 10)
  fit4  <- tulpa(y ~ x, data = d, family = "t", mode = "laplace", phi = 0.8)

  # Reference: penalized MAP of the same posterior (weak builtin prior
  # beta ~ N(0, 100^2)) via optim.
  nlp <- function(b, nu) {
    -sum(stats::dt((y - X %*% b) / 0.8, df = nu, log = TRUE) - log(0.8)) +
      sum(b^2) / (2 * 100^2)
  }
  ref10 <- stats::optim(c(0, 0), nlp, nu = 10, method = "BFGS")$par
  expect_equal(unname(coef(fit10)), ref10, tolerance = 1e-4)
  expect_false(isTRUE(all.equal(unname(coef(fit10)), unname(coef(fit4)),
                                tolerance = 1e-6)))
})

test_that("gaussian phi means the residual variance on every backend", {
  skip_on_cran()
  set.seed(83)
  n <- 60L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(1, -0.5)) + rnorm(n, 0, 2)   # variance 4
  d <- data.frame(y = y, x = X[, 2])

  # Laplace: a strong prior makes the likelihood/prior balance identify the
  # convention. MAP must be the variance-convention penalized WLS.
  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
               phi = 4, beta_prior = list(mean = 0, sd = 1))
  map_var <- solve(crossprod(X) / 4 + diag(1, 2), crossprod(X, y) / 4)
  expect_equal(unname(coef(fit)), as.numeric(map_var), tolerance = 1e-5)

  # ModelData NUTS: the posterior SD must match the variance convention.
  fit2 <- tulpa_sample_glmm(y, NULL, X, family = "gaussian", backend = "hmc",
                            phi = 2,   # the DIRECT door takes the residual SD
                            control = list(n_iter = 3000L, warmup = 1000L,
                                           n_chains = 2L, seed = 1L))
  sd_ref <- sqrt(diag(solve(crossprod(X) / 4)))
  expect_equal(unname(apply(fit2$draws[, 1:2], 2, sd)), sd_ref,
               tolerance = 0.15)

  # And through the front door (which converts variance -> SD for the kernel):
  fit3 <- tulpa(y ~ x, data = d, family = "gaussian", mode = "hmc", phi = 4,
                control = list(n_iter = 3000L, warmup = 1000L,
                               n_chains = 2L, seed = 1L))
  expect_equal(unname(apply(.fixed_draws_mat(fit3), 2, sd)), sd_ref,
               tolerance = 0.15)
})

test_that("lognormal is registered with the variance convention", {
  set.seed(85)
  eta <- c(0.2, 1.0); y <- c(1.5, 4.0); v <- 0.36

  expect_equal(tulpa:::family_loglik(eta, y, "lognormal", phi = v),
               stats::dlnorm(y, meanlog = eta, sdlog = sqrt(v), log = TRUE),
               tolerance = 1e-12)
  expect_equal(tulpa:::family_mean(eta, "lognormal", phi = v),
               exp(eta + v / 2))
  # Sample moments.
  ys <- tulpa:::family_sample(rep(0.5, 40000), "lognormal", phi = 0.25)
  expect_equal(mean(ys), exp(0.5 + 0.125), tolerance = 0.03)
  expect_equal(stats::var(ys),
               (exp(0.25) - 1) * exp(1 + 0.25), tolerance = 0.1)
})

test_that("lognormal fits through the front door", {
  skip_on_cran()
  set.seed(87)
  n <- 300L
  d <- data.frame(x = rnorm(n))
  d$y <- exp(1 + 0.5 * d$x + rnorm(n, 0, 0.6))   # log-scale variance 0.36

  fit <- tulpa(y ~ x, data = d, family = "lognormal", mode = "laplace",
               phi = 0.36)
  ref <- unname(coef(stats::lm(log(y) ~ x, data = d)))
  expect_equal(unname(coef(fit)), ref, tolerance = 1e-3)

  # And on the sampler tier.
  fit2 <- tulpa(y ~ x, data = d, family = "lognormal", mode = "hmc",
                phi = 0.36, control = list(n_iter = 2000L, warmup = 1000L,
                                           n_chains = 2L, seed = 2L))
  expect_equal(unname(coef(fit2)), ref, tolerance = 0.05)
})

test_that("posterior_predict honors the t df through phi2", {
  skip_on_cran()
  set.seed(89)
  n <- 300L
  d <- data.frame(x = rnorm(n))
  d$y <- 1 + 0.5 * d$x + 0.8 * rt(n, df = 20)
  fit <- tulpa(y ~ x, data = d, family = "t", mode = "laplace",
               phi = 0.8, phi2 = 20)

  yrep <- posterior_predict(fit, ndraws = 400L, seed = 3)
  # Residual variance of replicates ~ phi^2 nu/(nu-2) = 0.64 * 20/18 = 0.711.
  res_var <- mean(apply(yrep, 1, function(r) stats::var(r - fitted(fit))))
  expect_equal(res_var, 0.8^2 * 20 / 18, tolerance = 0.15)
})
