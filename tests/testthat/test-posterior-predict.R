# posterior_predict() / simulate.tulpa_fit (gcol33/tulpa C3): replicates are
# drawn from the family's sampling distribution at the per-draw linear
# predictor, so their moments must track the fitted mean and the family
# variance, RE draws must enter jointly with the fixed effects, and pp_check
# must fall back to generated replicates when no y_rep is stored.

test_that("family_sample draws match the family mean and variance", {
  set.seed(42)
  n <- 20000L
  eta <- rep(0.4, n)

  # poisson: mean = var = exp(eta)
  yp <- tulpa:::family_sample(eta, "poisson")
  expect_equal(mean(yp), exp(0.4), tolerance = 0.05)
  expect_equal(stats::var(yp), exp(0.4), tolerance = 0.05)

  # gaussian: mean = eta, var = phi
  yg <- tulpa:::family_sample(eta, "gaussian", phi = 2.5)
  expect_equal(mean(yg), 0.4, tolerance = 0.05)
  expect_equal(stats::var(yg), 2.5, tolerance = 0.1)

  # binomial with trials: mean = n p
  yb <- tulpa:::family_sample(eta, "binomial", n_trials = rep(10, n))
  expect_equal(mean(yb), 10 * plogis(0.4), tolerance = 0.05)

  # neg_binomial_2: var = mu + mu^2 / phi
  mu <- exp(0.4)
  ynb <- tulpa:::family_sample(eta, "neg_binomial_2", phi = 3)
  expect_equal(mean(ynb), mu, tolerance = 0.05)
  expect_equal(stats::var(ynb), mu + mu^2 / 3, tolerance = 0.1)

  # gamma: mean = mu, var = mu^2 / phi
  yga <- tulpa:::family_sample(eta, "gamma", phi = 4)
  expect_equal(mean(yga), mu, tolerance = 0.05)
  expect_equal(stats::var(yga), mu^2 / 4, tolerance = 0.1)

  # inverse_gaussian: mean = mu, var = phi * mu^3
  yig <- tulpa:::family_sample(eta, "inverse_gaussian", phi = 0.5)
  expect_equal(mean(yig), mu, tolerance = 0.05)
  expect_equal(stats::var(yig), 0.5 * mu^3, tolerance = 0.15)

  # beta: mean = mu_logit, var = mu(1-mu)/(phi+1)
  mub <- plogis(0.4)
  ybe <- tulpa:::family_sample(eta, "beta", phi = 8)
  expect_equal(mean(ybe), mub, tolerance = 0.02)
  expect_equal(stats::var(ybe), mub * (1 - mub) / 9, tolerance = 0.1)

  # beta_binomial: mean = n mu
  ybb <- tulpa:::family_sample(eta, "beta_binomial",
                               n_trials = rep(10, n), phi = 5)
  expect_equal(mean(ybb), 10 * mub, tolerance = 0.05)

  # t: mean = eta, var = phi^2 * nu/(nu-2) at nu = 4
  yt <- tulpa:::family_sample(eta, "t", phi = 1.5)
  expect_equal(mean(yt), 0.4, tolerance = 0.1)
  expect_equal(stats::var(yt), 1.5^2 * 4 / 2, tolerance = 0.6)
})

test_that("posterior_predict replicates track the data on a draws-tier fit", {
  skip_on_cran()
  set.seed(7)
  n <- 300L
  d <- data.frame(x = rnorm(n))
  d$y <- rpois(n, exp(0.5 + 0.7 * d$x))

  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "mala",
               control = list(n_iter = 2000, warmup = 500))
  yrep <- posterior_predict(fit, seed = 1)

  expect_true(is.matrix(yrep))
  expect_equal(ncol(yrep), n)
  expect_false(anyNA(yrep))
  # Replicated mean tracks the observed mean.
  expect_equal(mean(rowMeans(yrep)), mean(d$y), tolerance = 0.15)

  # ndraws subsamples rows.
  y50 <- posterior_predict(fit, ndraws = 50L)
  expect_equal(nrow(y50), 50L)

  # Seed makes it reproducible and restores the RNG state.
  r_before <- .Random.seed
  a <- posterior_predict(fit, ndraws = 20L, seed = 99)
  expect_identical(.Random.seed, r_before)
  b <- posterior_predict(fit, ndraws = 20L, seed = 99)
  expect_identical(a, b)
})

test_that("posterior_predict works on the draw-free Laplace tier", {
  skip_on_cran()
  set.seed(11)
  n <- 250L
  d <- data.frame(x = rnorm(n))
  d$y <- rnorm(n, 1 + 0.5 * d$x, sd = sqrt(2))

  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = 2)
  yrep <- posterior_predict(fit, ndraws = 300L, seed = 2)

  expect_equal(dim(yrep), c(300L, n))
  expect_equal(mean(rowMeans(yrep)), mean(d$y), tolerance = 0.1)
  # Replicate variance ~ fitted variance + phi, dominated by phi = 2.
  expect_equal(mean(apply(yrep, 1, stats::var)), stats::var(d$y),
               tolerance = 0.2)
})

test_that("posterior_predict includes random effects at the training data", {
  skip_on_cran()
  set.seed(21)
  n <- 400L
  g <- factor(rep(letters[1:8], length.out = n))
  b <- rnorm(8, 0, 1.5)
  d <- data.frame(x = rnorm(n), g = g)
  trials <- rep(10L, n)
  d$y <- rbinom(n, trials, plogis(0.2 + 0.4 * d$x + b[as.integer(g)]))

  fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "gibbs",
               n_trials = trials, control = list(n_iter = 1500, warmup = 500))
  yrep <- posterior_predict(fit, seed = 3)

  # Group means of the replicates track the group means of the data -- only
  # possible when the RE contribution is in the linear predictor.
  obs_gm <- tapply(d$y, g, mean)
  rep_gm <- tapply(colMeans(yrep), g, mean)
  expect_gt(stats::cor(obs_gm, rep_gm), 0.95)
})

test_that("posterior_predict at newdata is population-level", {
  skip_on_cran()
  set.seed(5)
  d <- data.frame(x = rnorm(150))
  d$y <- rnorm(150, 1 + 2 * d$x, 1)
  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi = 1)

  nd <- data.frame(x = c(-1, 0, 1))
  yrep <- posterior_predict(fit, newdata = nd, ndraws = 500L, seed = 4)
  expect_equal(dim(yrep), c(500L, 3L))
  expect_equal(colMeans(yrep), c(-1, 1, 3), tolerance = 0.2)
})

test_that("simulate.tulpa_fit follows the stats::simulate convention", {
  skip_on_cran()
  set.seed(6)
  d <- data.frame(x = rnorm(100))
  d$y <- rpois(100, exp(0.3 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  sim <- simulate(fit, nsim = 3, seed = 8)
  expect_s3_class(sim, "data.frame")
  expect_equal(dim(sim), c(100L, 3L))
  expect_named(sim, c("sim_1", "sim_2", "sim_3"))
})

test_that("pp_check falls back to posterior_predict when no y_rep is stored", {
  skip_on_cran()
  skip_if_not_installed("bayesplot")
  skip_if_not_installed("ggplot2")
  set.seed(9)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.2 + 0.4 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")

  p <- pp_check(fit, type = "dens_overlay")
  expect_s3_class(p, "ggplot")
})

test_that("posterior_predict rejects fits without a builtin family", {
  bad <- structure(list(family = list(name = "custom")), class = "tulpa_fit")
  expect_error(posterior_predict(bad), "builtin character family")
})
