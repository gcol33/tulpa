# The scale SGHMC and SGLD sample at (gcol33/tulpa#422).
#
# SGHMC injected friction noise with sd sqrt(2 * alpha * epsilon) where the
# scheme calls for sqrt(2 * alpha): substituting v = epsilon * r into the
# recursion gives an injected variance of 2 * alpha * eta * epsilon against the
# 2 * alpha * eta of Chen, Fox and Guestrin (2014) eq. (15) at eta = epsilon^2.
# Fluctuation-dissipation then fixes the sampled temperature at epsilon, so the
# chain targeted p(theta)^(1/epsilon): posterior means came out roughly right
# while every posterior sd was short by sqrt(epsilon) -- a factor of 32 at the
# default step size, with nothing in the output signalling it.
#
# Recovery of the MEAN is what the existing backend tests check, and it is
# exactly what this defect left intact. These read the sd against a posterior
# available in closed form.

# A gaussian arm at phi = 1 has a Gaussian posterior in beta under the ridge
# prior the front door applies, so mean and covariance are exact. phi = 1 also
# keeps the fixture clear of the residual-scale convention (gcol33/tulpa#560).
.scale_reference <- function(y, X) {
  prior_sd <- .tulpa_prior_sd("sample_glmm")
  V <- solve(crossprod(X) + diag(1 / prior_sd^2, ncol(X)))
  list(mean = drop(V %*% crossprod(X, y)), sd = sqrt(diag(V)))
}

.scale_fit <- function(y, X, backend, ...) {
  tulpa_sample_glmm(
    y = y, n_trials = rep(1L, length(y)), X = X,
    family = "gaussian", backend = backend, phi = 1,
    control = c(list(n_iter = 20000L, warmup = 5000L, seed = 7L), list(...)))
}

test_that("SGHMC samples the posterior sd, not a tempered one", {
  skip_if_not_slow()
  # A weakly informative design, so the step size the adapter settles on leaves
  # the discretisation bias of the scheme small (gcol33/tulpa#576).
  set.seed(404)
  n <- 200L
  X <- cbind(0.1, rnorm(n) * 0.1)
  y <- as.numeric(X %*% c(3, -6)) + rnorm(n, 0, 1)
  ref <- .scale_reference(y, X)

  fit <- .scale_fit(y, X, "sghmc")
  ratio <- apply(fit$draws, 2, sd) / ref$sd
  expect_true(all(ratio > 0.85 & ratio < 1.25),
              info = paste("sd ratio:", paste(round(ratio, 3), collapse = ", ")))
  expect_lt(max(abs(unname(fit$means) - ref$mean)), 0.05)

  # The pre-fix scaling put the chain at temperature epsilon, so its sd ratio
  # was sqrt(epsilon) -- about 0.32 at the step size this fixture adapts to,
  # and 0.03 at the sampler's own default.
  expect_gt(min(ratio), 2 * sqrt(fit$final_epsilon))
})

test_that("SGLD samples the posterior sd, and is pinned beside SGHMC", {
  skip_if_not_slow()
  # SGLD runs at a fixed small step size, so it needs a design its step can
  # traverse; its drift/noise pairing (0.5 * eps * grad against sd sqrt(eps))
  # was already the correct Langevin one and must stay that way.
  set.seed(505)
  n <- 150L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(0.3, -0.6)) + rnorm(n, 0, 1)
  ref <- .scale_reference(y, X)

  fit <- .scale_fit(y, X, "sgld")
  ratio <- apply(fit$draws, 2, sd) / ref$sd
  expect_true(all(ratio > 0.85 & ratio < 1.25),
              info = paste("sd ratio:", paste(round(ratio, 3), collapse = ", ")))
  expect_lt(max(abs(unname(fit$means) - ref$mean)), 0.05)
})

test_that("exact MCMC on the same fixture reproduces the closed form", {
  # The reference itself, checked against a backend that is not under test, so a
  # failure above reads as the stochastic-gradient scale and not as a wrong
  # posterior.
  skip_if_not_slow()
  set.seed(505)
  n <- 150L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(0.3, -0.6)) + rnorm(n, 0, 1)
  ref <- .scale_reference(y, X)

  fit <- .scale_fit(y, X, "hmc")
  ratio <- apply(fit$draws, 2, sd) / ref$sd
  expect_true(all(ratio > 0.95 & ratio < 1.05),
              info = paste("sd ratio:", paste(round(ratio, 3), collapse = ", ")))
  expect_lt(max(abs(unname(fit$means) - ref$mean)), 0.01)
})
