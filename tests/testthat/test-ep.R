# Expectation Propagation for a GLM (gcol33/tulpa C6). The definitive correctness
# anchor is that EP is EXACT for a Gaussian likelihood: the EP posterior must
# equal the closed-form Gaussian posterior. Also checked: Gauss-Hermite accuracy
# and logistic-GLM coefficient recovery.

test_that("Gauss-Hermite quadrature is accurate", {
  gh <- tulpa:::.gauss_hermite(20L)
  expect_equal(sum(gh$w), sqrt(pi), tolerance = 1e-10)
  # integral x^2 exp(-x^2) / integral exp(-x^2) = 1/2
  expect_equal(sum(gh$w * gh$x^2) / sum(gh$w), 0.5, tolerance = 1e-10)
  # odd moment is zero
  expect_equal(sum(gh$w * gh$x^3), 0, tolerance = 1e-8)
})

test_that("EP is exact for a Gaussian likelihood (closed-form tilted moments)", {
  skip_on_cran()
  set.seed(1)
  n <- 60L; X <- cbind(1, rnorm(n))
  beta <- c(0.5, -0.8); sig2 <- 0.7
  y <- as.numeric(X %*% beta) + rnorm(n, 0, sqrt(sig2))
  d <- data.frame(y = y, x = X[, 2])

  fit <- tulpa_ep(y ~ x, data = d, family = "gaussian", phi = sig2,
                  beta_prior = list(mean = 0, sd = 10))

  # Closed-form Gaussian posterior with prior N(0, 100 I).
  P0  <- diag(1 / 100, 2)
  Vex <- solve(P0 + crossprod(X) / sig2)
  mex <- as.numeric(Vex %*% (crossprod(X, y) / sig2))

  expect_equal(unname(coef(fit)), mex, tolerance = 1e-5)
  expect_equal(unname(vcov(fit)), unname(Vex), tolerance = 1e-5)
  expect_true(fit$converged)
})

test_that("EP log-marginal equals the exact Gaussian evidence", {
  skip_on_cran()
  set.seed(4)
  n <- 50L; X <- cbind(1, rnorm(n))
  sig2 <- 0.6; s0 <- 3
  y <- as.numeric(X %*% c(0.4, -0.7)) + rnorm(n, 0, sqrt(sig2))
  d <- data.frame(y = y, x = X[, 2])

  fit <- tulpa_ep(y ~ x, data = d, family = "gaussian", phi = sig2,
                  beta_prior = list(mean = 0, sd = s0))

  # Exact evidence of the conjugate linear model: y ~ N(0, s0^2 X X' + sig2 I).
  S <- s0^2 * tcrossprod(X) + diag(sig2, n)
  cS <- chol(S)
  logZ_exact <- -0.5 * n * log(2 * pi) - sum(log(diag(cS))) -
    0.5 * sum(backsolve(cS, y, transpose = TRUE)^2)

  expect_equal(fit$log_marginal, logZ_exact, tolerance = 1e-6)
  # logLik() now reads it.
  expect_equal(as.numeric(logLik(fit)), logZ_exact, tolerance = 1e-6)
})

test_that("EP log-marginal matches brute-force quadrature for a logistic GLM", {
  skip_on_cran()
  set.seed(5)
  n <- 80L; x <- rnorm(n)
  y <- rbinom(n, 1, plogis(-0.2 + 0.7 * x))
  d <- data.frame(y = y, x = x)
  s0 <- 2

  fit <- tulpa_ep(y ~ x, data = d, family = "binomial",
                  beta_prior = list(mean = 0, sd = s0))
  expect_true(is.finite(fit$log_marginal))

  # Brute-force 2-D quadrature over (b0, b1) of prod_i p(y_i | eta_i) N(b; 0, s0^2 I).
  gr <- seq(-3, 3, length.out = 161)
  h  <- gr[2] - gr[1]
  X  <- cbind(1, x)
  ll_grid <- outer(gr, gr, Vectorize(function(b0, b1) {
    eta <- b0 + b1 * x
    sum(tulpa:::family_loglik(eta, y, "binomial", n_trials = rep(1L, n))) +
      sum(stats::dnorm(c(b0, b1), 0, s0, log = TRUE))
  }))
  m <- max(ll_grid)
  logZ_quad <- m + log(sum(exp(ll_grid - m))) + 2 * log(h)

  # EP's evidence approximation is accurate to a few hundredths of a nat here.
  expect_equal(fit$log_marginal, logZ_quad, tolerance = 0.05)
})

test_that("EP is reachable as a registered backend via tulpa(mode = 'ep')", {
  skip_on_cran()
  set.seed(11)
  n <- 400L; x <- rnorm(n)
  y <- rbinom(n, 1, plogis(-0.2 + 0.7 * x))
  d <- data.frame(y = y, x = x)

  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep")
  expect_s3_class(fit, "tulpa_ep")
  expect_identical(fit$backend, "ep")
  expect_equal(fit$inference_tier, 2L)
  # Matches the direct fitter.
  direct <- tulpa_ep(y ~ x, data = d, family = "binomial")
  expect_equal(unname(coef(fit)), unname(coef(direct)), tolerance = 1e-8)
})

test_that("EP front door rejects random effects and a non-zero prior mean", {
  skip_on_cran()
  set.seed(12)
  d <- data.frame(y = rbinom(50, 1, 0.5), x = rnorm(50), g = gl(5, 10))
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "ep"),
    "fixed-effect GLM only"
  )
  expect_error(
    tulpa_ep(y ~ x, data = d, family = "binomial",
             beta_prior = list(mean = 1, sd = 10)),
    "mean-zero"
  )
})

test_that("EP front door forwards phi2 to ep_fit() (gcol33/tulpa#785)", {
  skip_on_cran()
  set.seed(1)
  n <- 160L; G <- 12L
  x <- rnorm(n); g <- factor(sample(seq_len(G), n, replace = TRUE))
  b0 <- rnorm(G, 0, 0.5); b1 <- rnorm(G, 0, 0.3)
  d <- data.frame(x = x, g = g,
                  y = 0.5 + x + b0[g] + b1[g] * x + 0.7 * rt(n, 4))

  direct <- tulpa_ep(y ~ x, data = d, family = "t", phi = 0.7, phi2 = 30)
  front  <- tulpa(y ~ x, data = d, family = "t", phi = 0.7, phi2 = 30,
                  mode = "ep")
  expect_equal(unname(coef(front)), unname(coef(direct)), tolerance = 1e-8)

  # phi2 actually moves the fit relative to the family's default.
  default <- tulpa_ep(y ~ x, data = d, family = "t", phi = 0.7)
  expect_true(any(abs(unname(coef(direct)) - unname(coef(default))) > 1e-3))

  expect_true("ep" %in% tulpa:::.phi2_backends())
})

test_that("EP recovers logistic-GLM coefficients", {
  skip_on_cran()
  set.seed(2)
  n <- 700L; x <- rnorm(n)
  b <- c(-0.3, 0.9)
  y <- rbinom(n, 1, plogis(b[1] + b[2] * x))
  d <- data.frame(y = y, x = x)

  fit <- tulpa_ep(y ~ x, data = d, family = "binomial")
  mle <- unname(coef(stats::glm(y ~ x, family = "binomial", data = d)))
  expect_lt(max(abs(unname(coef(fit)) - mle)), 0.1)
  expect_true(fit$converged)
  # PD covariance and draws.
  expect_true(all(eigen(vcov(fit), symmetric = TRUE, only.values = TRUE)$values > 0))
  expect_equal(ncol(fit$draws), 2L)
})

# gcol33/tulpa#882: the first column of cbind(successes, failures) was read as
# y against n_trials = 1, and y > n_trials was fitted with converged = TRUE.
# gcol33/tulpa#886: an incomplete row was dropped by model.frame() in silence.
test_that("EP refuses y > n_trials, a doubled denominator and NA rows", {
  set.seed(1)
  n <- 40L
  x <- rnorm(n)
  nt <- rep(5L, n)
  yb <- rbinom(n, nt, plogis(-0.5 + 0.8 * x))
  d <- data.frame(yb, nt, x)
  expect_error(tulpa_ep(yb ~ x, d, family = "binomial"),
               "with no `n_trials` requires a 0/1 response")
  expect_error(tulpa_ep(yb ~ x, d, family = "binomial", n_trials = 2L),
               "requires `y <= n_trials`")
  expect_error(tulpa_ep(cbind(yb, nt - yb) ~ x, d, family = "binomial",
                        n_trials = nt), "alongside a cbind")
  expect_error(tulpa_ep(cbind(yb, nt - yb) ~ x, d, family = "poisson"),
               "family = 'poisson' takes a single-column response")
  expect_error(tulpa_ep(yb ~ x, d, family = "poisson", n_trials = nt),
               "not read by family = 'poisson'")
  d$x[3] <- NA
  expect_error(tulpa_ep(yb ~ x, d, family = "binomial", n_trials = nt),
               "tulpa_ep: Non-finite value\\(s\\) in the model matrix")
})

test_that("EP decodes a cbind(successes, failures) response", {
  skip_on_cran()
  set.seed(1)
  n <- 200L
  x <- rnorm(n)
  nt <- rep(5L, n)
  yb <- rbinom(n, nt, plogis(-0.5 + 0.8 * x))
  d <- data.frame(yb, nt, x)
  pair <- tulpa_ep(cbind(yb, nt - yb) ~ x, d, family = "binomial")
  expl <- tulpa_ep(yb ~ x, d, family = "binomial", n_trials = nt)
  expect_equal(unname(coef(pair)), unname(coef(expl)), tolerance = 1e-8)
  mle <- unname(coef(stats::glm(cbind(yb, nt - yb) ~ x, binomial, data = d)))
  expect_lt(max(abs(unname(coef(pair)) - mle)), 0.02)
})
