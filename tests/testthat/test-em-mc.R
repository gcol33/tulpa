# Monte-Carlo EM driver. Sibling of test-em-laplace.

# ============================================================================
# Toy: 2-component Gaussian mixture by Bernoulli classification.
#   z_i ~ Bernoulli(0.4)
#   y_i | z_i = 0 ~ N(-1, 1)
#   y_i | z_i = 1 ~ N(+2, 1)
# Goal: recover the mixture means (-1, +2) given y, treating z as latent.
# E-step: sample z | y, theta_curr (binary classification).
# M-step: refit two Gaussian means via weighted-mean (intercept-only blocks).
# ============================================================================

simulate_mixture <- function(n = 200L, seed = 1L) {
  set.seed(seed)
  z <- rbinom(n, 1L, 0.4)
  y <- ifelse(z == 0L, rnorm(n, -1, 1), rnorm(n, 2, 1))
  list(y = y, z_true = z)
}

make_e_step_sample <- function(y) {
  function(fits, n_mc, ...) {
    if (length(fits) == 0L) {
      # First iter: break symmetry by initialising with a soft
      # split at the median of y. Without this, naive 50/50 random
      # init leaves both components fitting the overall mean.
      probs <- ifelse(y > stats::median(y), 0.8, 0.2)
    } else {
      # Use last iter's first-draw fit means as current params.
      f0 <- fits[[1L]]
      mu0 <- f0$comp0$mode[1]
      mu1 <- f0$comp1$mode[1]
      log_p0 <- dnorm(y, mu0, 1, log = TRUE) + log(0.6)
      log_p1 <- dnorm(y, mu1, 1, log = TRUE) + log(0.4)
      m <- pmax(log_p0, log_p1)
      probs <- exp(log_p1 - m) / (exp(log_p0 - m) + exp(log_p1 - m))
    }
    lapply(seq_len(n_mc), function(s) rbinom(length(y), 1L, probs))
  }
}

make_m_step_encode <- function(y) {
  function(weights, ...) {
    z <- weights
    # comp0: obs with z=0; comp1: obs with z=1. Intercept-only Gaussian.
    list(
      comp0 = list(y = y[z == 0L],
                   X = matrix(1, sum(z == 0L), 1L),
                   family = "gaussian"),
      comp1 = list(y = y[z == 1L],
                   X = matrix(1, sum(z == 1L), 1L),
                   family = "gaussian")
    )
  }
}


test_that("tulpa_em_mc recovers mixture means on toy", {
  skip_on_cran()
  set.seed(501L)
  d <- simulate_mixture(n = 300L, seed = 501L)
  res <- tulpa_em_mc(
    e_step_sample = make_e_step_sample(d$y),
    m_step_encode = make_m_step_encode(d$y),
    n_mc = 10L,
    max_iter = 15L,
    tol = 5e-3,
    verbose = FALSE
  )

  expect_true(length(res$pooled) == 2L)
  expect_named(res$pooled, c("comp0", "comp1"))
  mu0 <- res$pooled$comp0$mean
  mu1 <- res$pooled$comp1$mean
  # Both components recovered within ~0.4 of truth on n=300.
  expect_lt(abs(mu0 - (-1)), 0.4)
  expect_lt(abs(mu1 - 2), 0.4)
  expect_gt(res$n_iter, 1L)
})


test_that("tulpa_em_mc respects n_mc_growth", {
  skip_on_cran()
  d <- simulate_mixture(n = 100L, seed = 502L)
  res <- tulpa_em_mc(
    e_step_sample = make_e_step_sample(d$y),
    m_step_encode = make_m_step_encode(d$y),
    n_mc = 5L,
    n_mc_growth = 1.5,
    n_mc_max = 50L,
    max_iter = 5L,
    tol = 1e-10,    # don't converge — exercise growth
    verbose = FALSE
  )
  # n_mc should grow over iterations.
  expect_true(all(diff(res$history$n_mc) >= 0L))
  expect_gt(res$n_mc_final, 5L)
})


test_that("tulpa_em_mc errors on bad e_step_sample return", {
  bad_sampler <- function(fits, n_mc, ...) "not a list"
  encoder <- function(w, ...) list(a = list(y = 1, X = matrix(1, 1, 1),
                                            family = "gaussian"))
  expect_error(
    tulpa_em_mc(bad_sampler, encoder, n_mc = 3L, max_iter = 1L,
                verbose = FALSE),
    "list of length"
  )
})


test_that("tulpa_em_mc errors on non-function callbacks", {
  expect_error(tulpa_em_mc(NULL, function(w, ...) list()),
               "must be a function")
  expect_error(tulpa_em_mc(function(f, n, ...) list(), 42),
               "must be a function")
})


test_that("tulpa_em_mc errors on bad n_mc / n_mc_growth", {
  e <- function(f, n, ...) replicate(n, list(), simplify = FALSE)
  m <- function(w, ...) list(a = list(y = 1, X = matrix(1, 1, 1),
                                      family = "gaussian"))
  expect_error(tulpa_em_mc(e, m, n_mc = 0L, verbose = FALSE),
               "n_mc")
  expect_error(tulpa_em_mc(e, m, n_mc = 5L, n_mc_growth = 0.5,
                           verbose = FALSE),
               "n_mc_growth")
  expect_error(tulpa_em_mc(e, m, n_mc = 10L, n_mc_max = 5L,
                           verbose = FALSE),
               "n_mc_max")
})
