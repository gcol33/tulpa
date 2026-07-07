# Observation weights through the tulpa() front door (gcol33/tulpa C1).
# Equivalence anchor: a weight of 2 must reproduce the fit with that row
# duplicated, on both supported paths (non-spatial Laplace kernel and the R
# log-posterior builder). Zero weights must reproduce the subset fit, and the
# weighted gaussian MAP must equal the analytic penalized WLS solution.

test_that("weight 2 equals a duplicated row on the Laplace path", {
  skip_on_cran()
  set.seed(51)
  n <- 120L
  d <- data.frame(x = rnorm(n), g = factor(rep(letters[1:6], length.out = n)))
  d$y <- rpois(n, exp(0.3 + 0.6 * d$x))

  w <- rep(1, n); w[1:10] <- 2
  fit_w <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                 mode = "laplace", sigma_re = 1, weights = w)

  d_dup <- rbind(d[1:10, ], d)
  fit_d <- tulpa(y ~ x + (1 | g), data = d_dup, family = "poisson",
                 mode = "laplace", sigma_re = 1)

  expect_equal(coef(fit_w), coef(fit_d), tolerance = 1e-6)
  expect_equal(vcov(fit_w), vcov(fit_d), tolerance = 1e-5)
})

test_that("zero weights reproduce the subset fit", {
  skip_on_cran()
  set.seed(53)
  n <- 100L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.2 + 0.5 * d$x))

  w <- rep(1, n); w[81:100] <- 0
  fit_w <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                 weights = w)
  fit_s <- tulpa(y ~ x, data = d[1:80, ], family = "poisson", mode = "laplace")

  expect_equal(coef(fit_w), coef(fit_s), tolerance = 1e-6)
})

test_that("weighted gaussian Laplace MAP equals analytic penalized WLS", {
  skip_on_cran()
  set.seed(55)
  n <- 80L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(1, -0.5)) + rnorm(n, 0, 0.8)
  d <- data.frame(y = y, x = X[, 2])
  w <- runif(n, 0.5, 3)
  phi <- 0.64

  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
               phi = phi, weights = w)

  # MAP of the penalized weighted least squares with the built-in weak prior
  # beta ~ N(0, 100^2): (X'WX/phi + I/100^2)^{-1} X'W y / phi.
  A <- crossprod(X, w * X) / phi + diag(1e-4, 2)
  b <- crossprod(X, w * y) / phi
  beta_exact <- as.numeric(solve(A, b))

  expect_equal(unname(coef(fit)), beta_exact, tolerance = 1e-5)
})

test_that("weighted log-posterior equals the duplicated-row log-posterior", {
  set.seed(57)
  n <- 60L
  d <- data.frame(x = rnorm(n), g = factor(rep(1:5, length.out = n)))
  d$y <- rbinom(n, 1, plogis(0.2 + 0.7 * d$x))

  parsed <- tulpa:::tulpa_parse_formula(y ~ x + (1 | g))
  b_w    <- tulpa:::tulpa_build_model_data(parsed, d)
  d_dup  <- rbind(d[1:5, ], d)
  b_d    <- tulpa:::tulpa_build_model_data(parsed, d_dup)

  w <- rep(1, n); w[1:5] <- 2
  m_w <- tulpa:::build_glmm_logpost(b_w, "binomial", sigma_re = 1, weights = w)
  m_d <- tulpa:::build_glmm_logpost(b_d, "binomial", sigma_re = 1)

  expect_equal(m_w$dim, m_d$dim)
  for (i in 1:3) {
    theta <- rnorm(m_w$dim, sd = 0.5)
    expect_equal(m_w$log_posterior(theta), m_d$log_posterior(theta),
                 tolerance = 1e-12)
    expect_equal(m_w$grad_log_posterior(theta), m_d$grad_log_posterior(theta),
                 tolerance = 1e-12)
  }
})

test_that("all-ones weights equal an unweighted fit", {
  skip_on_cran()
  set.seed(59)
  d <- data.frame(x = rnorm(90))
  d$y <- rpois(90, exp(0.4 + 0.5 * d$x))

  fit_1 <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace",
                 weights = rep(1, 90))
  fit_0 <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  expect_equal(coef(fit_1), coef(fit_0), tolerance = 1e-10)
})

test_that("weights are validated and unsupported backends refuse", {
  d <- data.frame(x = rnorm(40), g = factor(rep(1:4, 10)))
  d$y <- rbinom(40, 1, 0.5)

  expect_error(
    tulpa(y ~ x, data = d, family = "binomial", mode = "laplace",
          weights = rep(-1, 40)),
    "non-negative")
  expect_error(
    tulpa(y ~ x, data = d, family = "binomial", mode = "laplace",
          weights = c(1, 2)),
    "length")
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "gibbs",
          weights = rep(1, 40)),
    "not supported by backend 'gibbs'")
})

test_that("weighted MALA shifts toward the up-weighted observations", {
  skip_on_cran()
  set.seed(61)
  n <- 200L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.5 + 0.8 * d$x))

  # Up-weighting everything by 4 sharpens the posterior but keeps the mean.
  fit_1 <- tulpa(y ~ x, data = d, family = "poisson", mode = "mala",
                 control = list(n_iter = 3000, warmup = 1000, epsilon = 0.05))
  fit_4 <- tulpa(y ~ x, data = d, family = "poisson", mode = "mala",
                 weights = rep(4, n),
                 control = list(n_iter = 3000, warmup = 1000, epsilon = 0.05))

  expect_equal(coef(fit_4), coef(fit_1), tolerance = 0.1)
  expect_lt(mean(diag(vcov(fit_4))), mean(diag(vcov(fit_1))))
})
