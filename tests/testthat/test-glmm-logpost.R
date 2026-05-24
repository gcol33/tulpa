# test-glmm-logpost.R
# The Step-3 keystone: build_glmm_logpost(). Two things must hold:
#   1. grad_log_posterior matches the finite-difference gradient of
#      log_posterior (so gradient-based samplers/optimizers are correct);
#   2. the MAP of beta given the true RE sd recovers the simulating beta
#      (so the assembled log-posterior is the right target).

make_bundle <- function(formula, data) {
  tulpa_build_model_data(tulpa_parse_formula(formula), data)
}

test_that("gradient matches finite differences (binomial GLMM with RE)", {
  set.seed(11)
  ng <- 5L; per <- 12L; n <- ng * per
  g  <- rep(seq_len(ng), each = per)
  x  <- rnorm(n)
  nt <- rep(8L, n)
  eta <- 0.3 + 0.5 * x + rnorm(ng, 0, 0.6)[g]
  y  <- rbinom(n, nt, plogis(eta))
  d  <- data.frame(y = y, x = x, g = factor(g))

  bundle <- make_bundle(y ~ x + (1 | g), d)
  m <- build_glmm_logpost(bundle, "binomial", sigma_re = 0.6, n_trials = nt)

  set.seed(99)
  theta <- rnorm(m$dim, 0, 0.3)
  ana <- m$grad_log_posterior(theta)
  h <- 1e-6
  num <- vapply(seq_len(m$dim), function(i) {
    tp <- theta; tm <- theta; tp[i] <- tp[i] + h; tm[i] <- tm[i] - h
    (m$log_posterior(tp) - m$log_posterior(tm)) / (2 * h)
  }, numeric(1))
  expect_equal(ana, num, tolerance = 1e-4)
})

test_that("gradient matches finite differences (poisson, random slope)", {
  set.seed(22)
  ng <- 6L; per <- 15L; n <- ng * per
  g <- rep(seq_len(ng), each = per)
  x <- rnorm(n)
  eta <- 0.2 + 0.4 * x + rnorm(ng, 0, 0.5)[g] + x * rnorm(ng, 0, 0.3)[g]
  y <- rpois(n, exp(eta))
  d <- data.frame(y = y, x = x, g = factor(g))

  bundle <- make_bundle(y ~ x + (1 + x | g), d)
  m <- build_glmm_logpost(bundle, "poisson", sigma_re = 0.5)

  set.seed(7)
  theta <- rnorm(m$dim, 0, 0.2)
  ana <- m$grad_log_posterior(theta)
  h <- 1e-6
  num <- vapply(seq_len(m$dim), function(i) {
    tp <- theta; tm <- theta; tp[i] <- tp[i] + h; tm[i] <- tm[i] - h
    (m$log_posterior(tp) - m$log_posterior(tm)) / (2 * h)
  }, numeric(1))
  expect_equal(ana, num, tolerance = 1e-4)
})

test_that("MAP recovers the simulating beta (poisson GLMM, true sigma)", {
  set.seed(123)
  ng <- 50L; per <- 25L; n <- ng * per
  g  <- rep(seq_len(ng), each = per)
  x  <- rnorm(n)
  beta_true <- c(0.5, 0.8)
  sig <- 0.7
  u_true <- rnorm(ng, 0, sig)
  eta <- beta_true[1] + beta_true[2] * x + u_true[g]
  y <- rpois(n, exp(eta))
  d <- data.frame(y = y, x = x, g = factor(g))

  bundle <- make_bundle(y ~ x + (1 | g), d)
  m <- build_glmm_logpost(bundle, "poisson", sigma_re = sig)

  opt <- stats::optim(
    m$init, fn = m$log_posterior, gr = m$grad_log_posterior,
    method = "BFGS", control = list(fnscale = -1, maxit = 500)
  )
  expect_equal(opt$convergence, 0L)
  beta_hat <- m$unpack(opt$par)$beta
  expect_equal(beta_hat, beta_true, tolerance = 0.12)
})
