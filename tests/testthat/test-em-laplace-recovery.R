# Recovery tests for tulpa_em_laplace() (gcol33/tulpa#383).
#
# test-em-laplace.R mocks tulpa_laplace() and exercises engine plumbing only.
# This file fits, and its subject is the M-step CONTRACT for a soft latent
# label: the E-step's posterior weight w_i enters through the block's `weights`
# field, never as a fractional `y`.
#
# The M-step maximizes the expected complete-data log-likelihood, which for a
# Bernoulli latent is
#   Q = sum_i [ w_i log p_i + (1 - w_i) log(1 - p_i) ],
# a weighted Bernoulli likelihood carrying no binomial coefficient. Encoded as
# two rows per unit -- y = 1 at weight w_i, y = 0 at weight 1 - w_i -- that is
# exactly what glm(family = binomial, weights = ) maximizes, which is the
# reference below: it lives outside the engine, so agreeing with it makes the
# claim about Q a measurement rather than an assertion.
#
# A fractional `y` is a different object (the exact binomial density at a
# non-integer response, whose lchoose(1, w) normalizer is neither zero nor free
# of w) and is refused by .validate_family_support().

A0 <- 0.8; A1 <- -0.9      # occupancy (non-structural) logit coefficients
B0 <- 1.2; B1 <- 0.5       # abundance log-mean coefficients

sim_zip <- function(seed, n = 2000L) {
  set.seed(seed)
  Xo <- cbind(1, rnorm(n)); Xa <- cbind(1, rnorm(n))
  psi <- plogis(drop(Xo %*% c(A0, A1)))
  lam <- exp(drop(Xa %*% c(B0, B1)))
  list(y = rbinom(n, 1L, psi) * rpois(n, lam), Xo = Xo, Xa = Xa, n = n)
}

# The documented encoding: soft labels on `weights`, integer `y`.
zip_encode <- function(d) function(weights, ...) list(
  occ   = list(y = rep(c(1, 0), each = d$n), X = rbind(d$Xo, d$Xo),
               weights = c(weights, 1 - weights), family = "binomial"),
  abund = list(y = d$y, X = d$Xa, family = "poisson", weights = weights)
)

zip_e_step <- function(d) function(fits, ...) {
  if (!length(fits)) return(list(weights = pmax(as.numeric(d$y > 0), 0.5)))
  psi <- plogis(drop(d$Xo %*% fits$occ$mode))
  lam <- exp(drop(d$Xa %*% fits$abund$mode))
  num <- psi * exp(-lam)
  list(weights = ifelse(d$y > 0, 1, num / (num + (1 - psi))))
}

em_zip <- function(d, max_iter = 200L, damping = 0) {
  tulpa_em_laplace(zip_e_step(d), zip_encode(d), max_iter = max_iter,
                   tol = 1e-8, damping = damping, verbose = FALSE)
}

# Observed-data log-likelihood of the zero-inflated Poisson -- the quantity EM
# increases. Written out here rather than read off the fit, so the monotonicity
# check does not grade the engine against its own objective.
zip_loglik <- function(d, a, b) {
  psi <- plogis(drop(d$Xo %*% a)); lam <- exp(drop(d$Xa %*% b))
  sum(ifelse(d$y > 0,
             log(psi) + dpois(d$y, lam, log = TRUE),
             log((1 - psi) + psi * exp(-lam))))
}


# ---------------------------------------------------------------------------
# Tier 1: the contract at the block boundary.
# ---------------------------------------------------------------------------

test_that("a fractional binomial `y` is refused and names the weights channel", {
  e_step <- function(fits, ...) list(weights = rep(0.5, 6))
  m_step_encode <- function(weights, ...) list(
    occ = list(y = weights, X = cbind(1, rnorm(6)), family = "binomial")
  )
  expect_error(
    tulpa_em_laplace(e_step, m_step_encode, max_iter = 1L, verbose = FALSE),
    regexp = "Soft latent labels belong in `weights`"
  )
})


test_that("the two-row weighted encoding passes block validation", {
  n <- 6L; w <- runif(n); X <- cbind(1, rnorm(n))
  blk <- list(y = rep(c(1, 0), each = n), X = rbind(X, X),
              weights = c(w, 1 - w), family = "binomial")
  expect_true(tulpa:::.validate_submodel_block(blk, idx = "occ"))
})


test_that("a weights vector that does not match `y` is refused", {
  n <- 6L; X <- cbind(1, rnorm(n))
  blk <- list(y = rep_len(0:1, n), X = X, weights = runif(n - 2L),
              family = "binomial")
  expect_error(tulpa:::.validate_submodel_block(blk, idx = "occ"),
               regexp = "length\\(weights\\)")

  blk$weights <- as.character(runif(n))
  expect_error(tulpa:::.validate_submodel_block(blk, idx = "occ"),
               regexp = "`weights` must be numeric")
})


test_that("tulpa_laplace() rejects a weights vector it would read past", {
  n <- 20L; X <- cbind(1, rnorm(n)); y <- rbinom(n, 1L, 0.5)
  # The kernel borrows `weights` as a bare pointer indexed to N, so a short
  # vector used to read out of bounds and return a silently wrong mode.
  expect_error(
    tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, family = "binomial",
                  weights = runif(5L)),
    regexp = "`weights` must be a numeric vector of length 20"
  )
  expect_error(
    tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, family = "binomial",
                  weights = c(-1, runif(n - 1L))),
    regexp = "finite and non-negative"
  )
})


# ---------------------------------------------------------------------------
# Tier 2: the M-step objective, against a reference outside the engine.
# ---------------------------------------------------------------------------

test_that("the M-step's binomial arm reproduces glm(family = binomial, weights = )", {
  skip_on_cran()

  d <- sim_zip(101L)
  psi0 <- plogis(drop(d$Xo %*% c(0.4, -0.3)))
  lam0 <- exp(drop(d$Xa %*% c(1.0, 0.2)))
  w <- ifelse(d$y > 0, 1, psi0 * exp(-lam0) / (psi0 * exp(-lam0) + (1 - psi0)))

  blk <- tulpa_laplace(y = rep(c(1, 0), each = d$n),
                       n_trials = rep(1L, 2L * d$n),
                       X = rbind(d$Xo, d$Xo), weights = c(w, 1 - w),
                       family = "binomial", tol = 1e-12, max_iter = 500L)
  # glm warns "non-integer #successes" on a fractional prior weight: it is
  # noting that this is the weighted quasi-likelihood rather than an exact
  # binomial density, which is the same distinction the encoding rests on. The
  # score and information it maximizes are still Q's.
  ref <- suppressWarnings(
    stats::glm.fit(rbind(d$Xo, d$Xo), rep(c(1, 0), each = d$n),
                   weights = c(w, 1 - w), family = stats::binomial(),
                   control = list(epsilon = 1e-12, maxit = 200L)))
  # The residual gap is the engine's weak built-in fixed-effect ridge
  # (sigma_beta = 100, tau = 1e-4), which glm does not carry; measured 3.4e-07.
  expect_equal(blk$mode, unname(ref$coefficients), tolerance = 1e-5)

  # Same arm through the cbind(successes, failures) form on the original rows:
  # the two-row weighted encoding IS that likelihood.
  ref2 <- suppressWarnings(
    stats::glm.fit(d$Xo, cbind(w, 1 - w), family = stats::binomial(),
                   control = list(epsilon = 1e-12, maxit = 200L)))
  expect_equal(blk$mode, unname(ref2$coefficients), tolerance = 1e-5)

  # The weight reaches the curvature, not only the score: H_beta is the
  # weighted Fisher information plus that same ridge.
  mu <- plogis(drop(rbind(d$Xo, d$Xo) %*% blk$mode))
  Xs <- rbind(d$Xo, d$Xo)
  H_ref <- crossprod(Xs, Xs * (c(w, 1 - w) * mu * (1 - mu))) + diag(1e-4, 2L)
  expect_equal(blk$H_beta, H_ref, tolerance = 1e-8)

  # The abundance arm rides the same channel.
  pa <- tulpa_laplace(y = d$y, n_trials = rep(1L, d$n), X = d$Xa, weights = w,
                      family = "poisson", tol = 1e-12, max_iter = 500L)
  refp <- stats::glm.fit(d$Xa, d$y, weights = w, family = stats::poisson(),
                         control = list(epsilon = 1e-12, maxit = 200L))
  expect_equal(pa$mode, unname(refp$coefficients), tolerance = 1e-5)
})


test_that("zero-inflated Poisson EM recovers both arms", {
  skip_on_cran()

  res <- em_zip(sim_zip(201L))
  expect_true(res$converged)
  expect_equal(res$fits$occ$mode,   c(A0, A1), tolerance = 0.25)
  expect_equal(res$fits$abund$mode, c(B0, B1), tolerance = 0.12)
})


test_that("EM does not decrease the observed-data log-likelihood", {
  skip_on_cran()

  d <- sim_zip(301L, n = 800L)
  K <- 20L
  # The driver is deterministic, so refitting at max_iter = k reproduces
  # iteration k of one run.
  ll <- vapply(seq_len(K), function(k) {
    r <- em_zip(d, max_iter = k, damping = 0)
    zip_loglik(d, r$fits$occ$mode, r$fits$abund$mode)
  }, numeric(1))

  expect_gt(ll[K] - ll[1], 1)              # the run climbs
  # Undamped EM is monotone; past convergence the increments sit at the
  # floating-point floor, so the bound is a tolerance rather than zero
  # (measured worst decrease -2.6e-11 against a total climb of 78.8).
  expect_gt(min(diff(ll)), -1e-6)
})


# ---------------------------------------------------------------------------
# Tier 3: recovery across seeds.
# ---------------------------------------------------------------------------

test_that("ZIP EM is unbiased on both arms across seeds", {
  skip_if_not_slow()

  seeds <- 201L:212L
  est <- t(vapply(seeds, function(s) {
    r <- em_zip(sim_zip(s))
    expect_true(r$converged)
    c(r$fits$occ$mode, r$fits$abund$mode)
  }, numeric(4)))

  truth <- c(A0, A1, B0, B1)
  # Measured over these 12 seeds at n = 2000: mean estimates
  # (0.7961, -0.9027, 1.1968, 0.5007) against truth (0.8, -0.9, 1.2, 0.5),
  # per-seed sd (0.077, 0.053, 0.028, 0.015).
  expect_equal(colMeans(est), truth, tolerance = 0.05)
  for (j in seq_len(4L)) {
    expect_lt(abs(stats::median(est[, j]) - truth[j]), 0.06)
  }
})
