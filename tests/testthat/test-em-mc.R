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
      # The previous M-step's point estimate, one entry per submodel.
      mu0 <- fits$comp0$mode[1]
      mu1 <- fits$comp1$mode[1]
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

skip_on_cran()


test_that("tulpa_em_mc recovers mixture means on toy", {
  skip_on_cran()
  set.seed(501L)
  d <- simulate_mixture(n = 300L, seed = 501L)
  res <- tulpa_em_mc(
    e_step_sample = make_e_step_sample(d$y),
    m_step_encode = make_m_step_encode(d$y),
    n_mc = 10L,
    max_iter = 15L,
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


test_that("n_mc grows only after a step swamped by Monte Carlo error", {
  skip_on_cran()
  d <- simulate_mixture(n = 100L, seed = 502L)
  res <- suppressWarnings(tulpa_em_mc(
    e_step_sample = make_e_step_sample(d$y),
    m_step_encode = make_m_step_encode(d$y),
    n_mc = 5L,
    n_mc_growth = 1.5,
    n_mc_max = 50L,
    max_iter = 12L,
    tol = 1e-10,    # unreachable: exercise growth
    verbose = FALSE
  ))
  h <- res$history
  grew <- diff(h$n_mc) > 0L
  expect_true(any(grew))
  # Iteration t + 1 draws more only when iteration t was swamped.
  expect_true(all(h$swamped[-nrow(h)][grew]))
  expect_true(all(h$n_mc[-1][!h$swamped[-nrow(h)] %in% TRUE] ==
                    h$n_mc[-nrow(h)][!h$swamped[-nrow(h)] %in% TRUE]))
  expect_identical(res$n_mc_final, h$n_mc[nrow(h)])
})


test_that(".em_mc_step reads the step against the Monte Carlo error", {
  pooled <- list(a = list(mean = c(1, 0), se = c(0.1, 0.1),
                          V_between = c(0.01, 0.01)))
  # mcse = sqrt(0.01 / 100) = 0.01; a step of 0.001 lies well inside it.
  s1 <- .em_mc_step(c(1.001, 0.001), pooled, n_mc = 100L, alpha = 0.25)
  expect_true(s1$swamped)
  expect_equal(s1$delta, 0.01)
  expect_equal(s1$mcse, 0.1)
  # A step of 0.05 is five Monte Carlo SEs per coordinate.
  s2 <- .em_mc_step(c(1.05, 0.05), pooled, n_mc = 100L, alpha = 0.25)
  expect_false(s2$swamped)
  expect_equal(s2$delta, 0.5)
  # A coefficient at zero is no harder to converge than any other.
  s3 <- .em_mc_step(c(1, 1e-6), pooled, n_mc = 100L, alpha = 0.25)
  expect_equal(s3$delta, 1e-5)
})


test_that("e_step_sample receives the pooled point estimate per submodel", {
  skip_on_cran()
  d <- simulate_mixture(n = 120L, seed = 503L)
  seen <- list()
  inner <- make_e_step_sample(d$y)
  e <- function(fits, n_mc, ...) {
    seen[[length(seen) + 1L]] <<- fits
    inner(fits, n_mc, ...)
  }
  res <- suppressWarnings(tulpa_em_mc(e, make_m_step_encode(d$y), n_mc = 4L,
                                      max_iter = 3L, verbose = FALSE))
  expect_identical(seen[[1]], list())
  f2 <- seen[[2]]
  expect_named(f2, c("comp0", "comp1"))
  expect_named(f2$comp0, c("mode", "beta", "se"))
  expect_identical(names(res$fits), names(f2))
  expect_length(res$draws, res$n_mc_final)
  expect_equal(res$fits$comp1$beta, res$pooled$comp1$mean)
  expect_equal(res$fits$comp1$mode,
               mean(vapply(res$draws, function(dr) dr$comp1$mode[1], 0)))
})


test_that("MCEM on a zero-inflated Poisson converges to the MLE", {
  skip_on_cran()
  set.seed(20261102)
  n <- 300L; x <- rnorm(n); v <- rnorm(n)
  z <- rbinom(n, 1, plogis(0.8 - 1.0 * v))
  y <- z * rpois(n, exp(1.0 + 0.5 * x))
  X <- cbind(1, x); Z <- cbind(1, v)
  e_step_sample <- function(fits, n_mc, ...) {
    w <- if (!length(fits)) ifelse(y > 0, 1, 0.5) else {
      psi <- plogis(drop(Z %*% fits$occ$mode))
      p0 <- psi * exp(-exp(drop(X %*% fits$abund$mode)))
      ifelse(y > 0, 1, p0 / (p0 + 1 - psi))
    }
    lapply(seq_len(n_mc), function(s) rbinom(n, 1, w))
  }
  m_step_encode <- function(weights, ...) list(
    occ   = list(y = rep(c(1, 0), each = n), X = rbind(Z, Z),
                 weights = c(weights, 1 - weights), family = "binomial"),
    abund = list(y = y, X = X, weights = weights, family = "poisson"))
  nll <- function(th) {
    psi <- plogis(drop(Z %*% th[1:2])); lam <- exp(drop(X %*% th[3:4]))
    -sum(ifelse(y == 0, log(1 - psi + psi * exp(-lam)),
                log(psi) + stats::dpois(y, lam, log = TRUE)))
  }
  op <- stats::optim(c(0, 0, 0, 0), nll, method = "BFGS",
                     control = list(reltol = 1e-14, maxit = 2000))
  se <- sqrt(diag(solve(stats::optimHess(op$par, nll))))
  set.seed(101)
  res <- tulpa_em_mc(e_step_sample, m_step_encode, n_mc = 10L,
                     verbose = FALSE)
  expect_true(res$converged)
  est <- c(res$pooled$occ$mean, res$pooled$abund$mean)
  expect_lt(max(abs(est - op$par) / se), 0.25)
})
