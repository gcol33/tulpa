# The links carried on eta > 0: inverse (mu = 1/eta), 1mu2 (mu = 1/sqrt(eta))
# and sqrt (mu = eta^2). The first two have no mean outside the domain; sqrt has
# one, but it is the mirror image of the eta > 0 branch, so admitting it would
# make the mode a pair and the Hessian singular between them.
#
# These fits used to run against a clamp: eta was floored at 1e-10, which left mu
# constant in eta while mu_eta and mu_eta2 kept reporting -1e20 and 2e30. The
# value and its derivatives described different functions, and since the latent
# start is x = 0 -- hence eta = 0, the boundary -- every such fit began there.
# The domain is now a barrier the solver cannot cross, and the start is moved
# inside before the first iteration.

test_that("the log-likelihood is -Inf outside the eta > 0 links' domain", {
  cases <- list(
    list(family = "gamma_inverse",         y = 1.5, phi = 2.0),
    list(family = "inverse_gaussian_1mu2", y = 1.5, phi = 1.1),
    list(family = "poisson_sqrt",          y = 4,   phi = 1.0)
  )
  for (cs in cases) {
    for (eta in c(-3, -1e-8, 0)) {
      expect_identical(
        unname(cpp_family_terms(cs$y, 1L, eta, cs$family, cs$phi)[["log_lik"]]),
        -Inf,
        info = paste(cs$family, "at eta =", eta)
      )
    }
    # ... and finite immediately inside it.
    expect_true(
      is.finite(cpp_family_terms(cs$y, 1L, 0.7, cs$family, cs$phi)[["log_lik"]]),
      info = cs$family
    )
  }
})

test_that("the domain check leaves every other link alone", {
  cases <- list(
    list(family = "gamma",             y = 1.5, n = 1L,  phi = 2.0),
    list(family = "inverse_gaussian",  y = 1.5, n = 1L,  phi = 1.1),
    list(family = "poisson",           y = 4,   n = 1L,  phi = 1.0),
    list(family = "binomial",          y = 3,   n = 10L, phi = 1.0),
    list(family = "binomial_probit",   y = 3,   n = 10L, phi = 1.0),
    list(family = "gaussian",          y = 1.2, n = 1L,  phi = 1.5),
    list(family = "gaussian_log",      y = 1.2, n = 1L,  phi = 1.5)
  )
  for (cs in cases) {
    for (eta in c(-3, 0, 0.9)) {
      expect_true(
        is.finite(cpp_family_terms(cs$y, cs$n, eta, cs$family, cs$phi)[["log_lik"]]),
        info = paste(cs$family, "at eta =", eta)
      )
    }
  }
})

# H_beta is assembled in R, through the family registry, so a fit with
# return_hessian = TRUE exercises the engine and the R link layer together. That
# is the end-to-end path: before the link layer existed the mode-finding
# succeeded in C++ and the R side then failed to describe the same family.
test_that("a constrained-link fit returns a fixed-effect Hessian", {
  set.seed(15)
  n <- 200
  x <- runif(n, -1, 1)
  X <- cbind(1, x)
  mu <- 1 / as.vector(X %*% c(0.8, 0.25))
  y <- rgamma(n, shape = 6, rate = 6 / mu)

  fit <- tulpa_laplace(y, rep(1L, n), X, family = "gamma_inverse", phi = 6,
                       beta_prior = list(sd = Inf), return_hessian = TRUE)

  expect_true(fit$converged)
  expect_equal(dim(fit$H_beta), c(2L, 2L))
  expect_true(all(is.finite(fit$H_beta)))
  # Positive definite: the working weight has to be positive throughout, which
  # it is only if the fitted eta stayed inside the domain.
  expect_true(all(eigen(fit$H_beta, symmetric = TRUE, only.values = TRUE)$values > 0))

  # External check on the composed weight, free of dispersion conventions.
  # glm's unscaled covariance is (X' W X)^-1 with W = (dmu/deta)^2 / V and the
  # gamma variance function V = mu^2. Ours carries the shape, w = phi * W, so
  # H_beta = phi * (cov.unscaled)^-1 exactly.
  ref <- stats::glm(y ~ x, family = stats::Gamma(link = "inverse"))
  expect_equal(unname(fit$H_beta),
               unname(6 * solve(summary(ref)$cov.unscaled)),
               tolerance = 1e-4)
})

# With beta_prior$sd = Inf the fixed effects carry no penalty, so the Laplace
# mode IS the maximum-likelihood estimate and glm() is an exact reference rather
# than an approximate one. Any disagreement beyond solver tolerance is a defect,
# not a prior.

test_that("gamma with the inverse link reproduces the glm MLE", {
  set.seed(11)
  n <- 400
  x <- runif(n, -1, 1)
  X <- cbind(1, x)
  beta_true <- c(0.8, 0.25)          # eta in [0.55, 1.05] on x in [-1, 1]
  mu <- 1 / as.vector(X %*% beta_true)
  shape <- 6
  y <- rgamma(n, shape = shape, rate = shape / mu)

  ref <- stats::glm(y ~ x, family = stats::Gamma(link = "inverse"))
  fit <- tulpa_laplace(y, rep(1L, n), X, family = "gamma_inverse",
                       phi = shape, beta_prior = list(sd = Inf),
                       return_hessian = FALSE)

  expect_true(fit$converged)
  expect_false(isTRUE(fit$start_infeasible))
  expect_equal(as.numeric(fit$mode)[1:2], unname(stats::coef(ref)),
               tolerance = 1e-5)
  # The mode is interior: no fitted observation sits on or past the boundary.
  expect_true(all(as.vector(X %*% as.numeric(fit$mode)[1:2]) > 0))
})

test_that("inverse gaussian with the 1/mu^2 link reproduces the glm MLE", {
  set.seed(12)
  n <- 400
  x <- runif(n, -1, 1)
  X <- cbind(1, x)
  beta_true <- c(0.9, 0.3)           # eta = 1/mu^2 in [0.6, 1.2]
  eta <- as.vector(X %*% beta_true)
  mu <- 1 / sqrt(eta)
  disp <- 0.2
  y <- tulpa:::.rinvgauss(n, mu, lambda = 1 / disp)

  ref <- stats::glm(y ~ x, family = stats::inverse.gaussian(link = "1/mu^2"))
  fit <- tulpa_laplace(y, rep(1L, n), X, family = "inverse_gaussian_1mu2",
                       phi = disp, beta_prior = list(sd = Inf),
                       return_hessian = FALSE)

  expect_true(fit$converged)
  expect_false(isTRUE(fit$start_infeasible))
  expect_equal(as.numeric(fit$mode)[1:2], unname(stats::coef(ref)),
               tolerance = 1e-5)
  expect_true(all(as.vector(X %*% as.numeric(fit$mode)[1:2]) > 0))
})

test_that("the inverse-link estimator is unbiased across seeds", {
  # Reference agreement on one dataset says the solver finds the MLE. This says
  # the MLE it finds is centred on the truth, which is the claim a single fit
  # cannot support.
  beta_true <- c(0.8, 0.25)
  shape <- 6
  n <- 300
  n_seed <- 20

  est <- t(vapply(seq_len(n_seed), function(s) {
    set.seed(1000 + s)
    x <- runif(n, -1, 1)
    X <- cbind(1, x)
    mu <- 1 / as.vector(X %*% beta_true)
    y <- rgamma(n, shape = shape, rate = shape / mu)
    fit <- tulpa_laplace(y, rep(1L, n), X, family = "gamma_inverse",
                         phi = shape, beta_prior = list(sd = Inf),
                       return_hessian = FALSE)
    expect_true(fit$converged)
    as.numeric(fit$mode)[1:2]
  }, numeric(2)))

  # Monte Carlo SE of the mean over n_seed replicates; the bias has to sit
  # inside a few of those, not merely "look close".
  for (j in 1:2) {
    mc_se <- stats::sd(est[, j]) / sqrt(n_seed)
    expect_lt(abs(mean(est[, j]) - beta_true[j]), 3 * mc_se + 1e-8)
  }
})

test_that("a start outside the domain is moved inside, not clamped", {
  # x = 0 gives eta = 0 for every observation, which is the boundary. The
  # feasibility sweep has to find an interior start; if it silently clamped
  # instead, the fit would sit at mu = 1e10 with a huge reported gradient.
  set.seed(13)
  n <- 200
  x <- runif(n, -1, 1)
  X <- cbind(1, x)
  mu <- 1 / as.vector(X %*% c(0.8, 0.25))
  y <- rgamma(n, shape = 6, rate = 6 / mu)

  fit <- tulpa_laplace(y, rep(1L, n), X, family = "gamma_inverse",
                       phi = 6, beta_prior = list(sd = Inf),
                       return_hessian = FALSE)

  expect_false(isTRUE(fit$start_infeasible))
  expect_true(fit$converged)
  fitted_mu <- 1 / as.vector(X %*% as.numeric(fit$mode)[1:2])
  expect_true(all(fitted_mu > 0))
  expect_lt(max(fitted_mu), 100 * max(y))   # nowhere near the 1e10 clamp
})

test_that("the log link is unaffected by the feasibility sweep", {
  # The sweep only runs when the start has a non-finite objective, so a log-link
  # fit must reach the same MLE it always did.
  set.seed(14)
  n <- 300
  x <- runif(n, -1, 1)
  X <- cbind(1, x)
  mu <- exp(0.4 + 0.6 * x)
  y <- rgamma(n, shape = 5, rate = 5 / mu)

  ref <- stats::glm(y ~ x, family = stats::Gamma(link = "log"))
  fit <- tulpa_laplace(y, rep(1L, n), X, family = "gamma",
                       phi = 5, beta_prior = list(sd = Inf))

  expect_true(fit$converged)
  expect_equal(as.numeric(fit$mode)[1:2], unname(stats::coef(ref)),
               tolerance = 1e-5)
})
