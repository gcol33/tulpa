# agq_fit() stops on the Newton decrement of the negative log marginal, its
# estimated distance above the optimum in nats, rather than on a relative change
# in the objective: on a log-likelihood of a few hundred a relative test stops
# wherever successive values agree to that tolerance times the objective's size.

test_that(".optim_to_stationary reaches the minimum of a shallow objective far from zero", {
  m <- c(1, -2, 0.5)
  cvec <- c(1e-3, 5e-3, 2e-2)
  fn <- function(x) 1e3 + 0.5 * sum(cvec * (x - m)^2) + 0.01 * (x[1] - m[1])^4
  gr <- function(x) cvec * (x - m) + c(0.04 * (x[1] - m[1])^3, 0, 0)
  for (g in list(gr, NULL)) {
    op <- tulpa:::.optim_to_stationary(c(0, 0, 0), fn, g, tol = 1e-12)
    expect_identical(op$convergence, 0L)
    expect_lt(max(abs(op$par - m)), 1e-4)
    expect_lt(op$newton_decrement, 1e-12)
    expect_equal(op$hessian, diag(cvec), tolerance = 1e-4)
  }
})

test_that("agq_fit()'s default tolerance reaches the maximum", {
  set.seed(20261102)
  G <- 40L; npg <- 8L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  y <- rbinom(N, 1, plogis(-0.4 + 0.9 * x + rnorm(G, sd = 0.8)[g]))
  for (nq in c(1L, 7L)) {
    f0 <- agq_fit(y, cbind(1, x), g, family = "binomial", n_quad = nq)
    f1 <- agq_fit(y, cbind(1, x), g, family = "binomial", n_quad = nq,
                  tol = 1e-14)
    expect_true(f0$converged)
    expect_lt(abs(f0$log_marginal - f1$log_marginal), 1e-7)
    expect_lt(abs(f0$sigma_re / f1$sigma_re - 1), 1e-4)
  }
  # glmmTMB's Laplace fit of the same data reads loglik -186.8586979 and
  # random-intercept SD 0.8160033.
  f <- agq_fit(y, cbind(1, x), g, family = "binomial", n_quad = 1L)
  expect_equal(f$log_marginal, -186.8586979, tolerance = 1e-8)
  expect_equal(f$sigma_re, 0.8160033, tolerance = 1e-5)
})
