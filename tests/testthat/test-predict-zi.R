# Observation-level accessors on a zero-inflated fit: the zero-inflation design
# is rebuilt from the fit's own `ziformula`, the response-scale quantities are
# the mixture's, and the pointwise log-likelihood behind the criteria is the
# mixture density. Every expectation is scored against the mixture written out
# by hand from coef() and the design, not against another accessor.

zi_predict_data <- function(n = 300, seed = 1) {
  set.seed(seed)
  d <- data.frame(x = stats::rnorm(n), z = stats::rnorm(n))
  y <- stats::rpois(n, exp(0.2 + 0.5 * d$x))
  y[stats::runif(n) < stats::plogis(-0.8 + 0.7 * d$z)] <- 0L
  d$y <- y
  d
}

zi_hand_linpred <- function(fit, d) {
  b <- coef(fit)
  list(eta = as.numeric(b["(Intercept)"] + b["x"] * d$x),
       lz  = as.numeric(b["zi_(Intercept)"] + b["zi_z"] * d$z))
}

test_that("fitted and predict read the mixture mean at the training and new data", {
  skip_if_fast()
  d <- zi_predict_data()
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace", ziformula = ~ z)
  h <- zi_hand_linpred(fit, d)
  mix_mean <- (1 - stats::plogis(h$lz)) * exp(h$eta)

  expect_equal(fitted(fit), mix_mean, tolerance = 1e-12)
  expect_equal(predict(fit, type = "response"), mix_mean, tolerance = 1e-12)
  expect_equal(predict(fit), h$eta, tolerance = 1e-12)

  nd <- d[1:7, ]
  expect_equal(predict(fit, newdata = nd, type = "response"), mix_mean[1:7],
               tolerance = 1e-12)
  expect_error(predict(fit, newdata = nd[, "x", drop = FALSE]), "z")
})

test_that("residuals use the mixture mean and variance", {
  skip_if_fast()
  d <- zi_predict_data()
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace", ziformula = ~ z)
  h <- zi_hand_linpred(fit, d)
  pi0 <- stats::plogis(h$lz); mu <- exp(h$eta)
  m <- (1 - pi0) * mu
  v <- (1 - pi0) * (mu + mu^2) - m^2
  expect_equal(residuals(fit, type = "response"), d$y - m, tolerance = 1e-12)
  expect_equal(residuals(fit), (d$y - m) / sqrt(v), tolerance = 1e-10)
})

test_that("response-scale bounds of a zero-inflated prediction are pinned", {
  skip_if_fast()
  d <- zi_predict_data()
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace", ziformula = ~ z)
  nd <- d[1:10, ]
  set.seed(42); before <- .Random.seed
  p1 <- predict(fit, newdata = nd, type = "response", se.fit = TRUE)
  expect_identical(.Random.seed, before)
  p2 <- predict(fit, newdata = nd, type = "response", se.fit = TRUE)
  expect_identical(p1, p2)
  expect_true(all(p1$lower < p1$fit & p1$fit < p1$upper))
  # se.fit is the count predictor's standard error from the count block of vcov.
  X <- cbind(1, nd$x)
  V <- vcov(fit)[c("(Intercept)", "x"), c("(Intercept)", "x")]
  expect_equal(p1$se.fit, sqrt(rowSums((X %*% V) * X)), tolerance = 1e-12)
})

test_that("posterior_predict and simulate draw the structural zeros", {
  skip_if_fast()
  d <- zi_predict_data(n = 400)
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace", ziformula = ~ z)
  yrep <- posterior_predict(fit, ndraws = 400, seed = 3)
  expect_equal(dim(yrep), c(400L, nrow(d)))
  h <- zi_hand_linpred(fit, d)
  p0 <- stats::plogis(h$lz) + (1 - stats::plogis(h$lz)) * exp(-exp(h$eta))
  # 160000 Bernoulli zero indicators: the Monte Carlo SE of the rate is < 0.002.
  expect_equal(mean(yrep == 0), mean(p0), tolerance = 0.01)
  expect_equal(mean(yrep), mean(fitted(fit)), tolerance = 0.02)

  s <- simulate(fit, nsim = 2, seed = 1)
  expect_equal(dim(s), c(nrow(d), 2L))
  expect_true(is.numeric(pit_residuals(fit, nsim = 20L)))
  expect_s3_class(test_dispersion(fit, nsim = 20L), "htest")
  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_type(check_model(fit, nsim = 20L), "list")
})

test_that("a hurdle fit predicts the truncated mean times the non-zero probability", {
  skip_if_fast()
  set.seed(5)
  n <- 300
  d <- data.frame(x = stats::rnorm(n))
  lam <- exp(0.5 + 0.3 * d$x)
  pos <- stats::qpois(stats::runif(n, stats::ppois(0, lam), 1), lam)
  d$y <- ifelse(stats::runif(n) < 0.3, 0L, as.integer(pmax(pos, 1)))
  fit <- tulpa(y ~ x, d, family = "truncated_poisson", mode = "laplace",
               ziformula = ~ 1)
  b <- coef(fit)
  mu <- exp(b["(Intercept)"] + b["x"] * d$x)
  hurdle_mean <- as.numeric((1 - stats::plogis(b["zi_(Intercept)"])) *
                              mu / (1 - exp(-mu)))
  expect_equal(fitted(fit), hurdle_mean, tolerance = 1e-12)
  expect_equal(predict(fit, type = "response"), hurdle_mean, tolerance = 1e-12)
  yrep <- posterior_predict(fit, ndraws = 50, seed = 2)
  expect_equal(mean(yrep == 0), mean(d$y == 0), tolerance = 0.05)
})

test_that("a sampler fit reads the zero-inflation block from the engine layout", {
  skip_if_fast()
  d <- zi_predict_data(n = 200)
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "hmc", ziformula = ~ z,
               control = list(n_iter = 150L, warmup = 75L, n_chains = 1L,
                              seed = 1L))
  eta <- tulpa:::.tulpa_eta_draws(fit, ndraws = 20, synth_seed = 9L)
  lz <- attr(eta, "logit_zi")
  expect_equal(dim(lz), dim(eta))
  D <- fit$draws
  S <- nrow(D)
  set.seed(9L)
  rows <- sample.int(S, 20)
  zi_cols <- grep("^beta_zi\\[|^zi_", colnames(D))
  expect_length(zi_cols, 2L)
  expect_equal(lz, D[rows, zi_cols, drop = FALSE] %*% t(cbind(1, d$z)),
               tolerance = 1e-12, ignore_attr = TRUE)
})

test_that("the pointwise log-likelihood of a zero-inflated fit is the mixture density", {
  skip_if_fast()
  d <- zi_predict_data()
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace", ziformula = ~ z)
  ll <- tulpa:::.tulpa_pointwise_loglik(fit)

  eta <- tulpa:::.tulpa_eta_draws(fit, synth_seed = 285603L)
  lz  <- attr(eta, "logit_zi")
  Y   <- matrix(d$y, nrow(eta), ncol(eta), byrow = TRUE)
  pz  <- stats::plogis(lz)
  hand <- ifelse(Y == 0,
                 log(pz + (1 - pz) * stats::dpois(0, exp(eta))),
                 log(1 - pz) + stats::dpois(Y, exp(eta), log = TRUE))
  expect_equal(unclass(ll), hand, tolerance = 1e-10, ignore_attr = TRUE)

  expect_equal(cpo(fit), cpo(hand), tolerance = 1e-10)
  skip_if_not_installed("loo")
  w_fit  <- suppressWarnings(loo::waic(fit))
  w_hand <- suppressWarnings(loo::waic(hand))
  expect_equal(w_fit$estimates, w_hand$estimates, tolerance = 1e-10)
})
