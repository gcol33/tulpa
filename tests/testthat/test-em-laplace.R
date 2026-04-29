# Tests for tulpa_em_laplace().
#
# We mock tulpa_laplace() with local_mocked_bindings() so the tests stay
# focused on engine plumbing — per-submodel family/offset routing,
# convergence, and validation — without exercising the C++ backend.

# Helper: build a minimal valid block. Tests override individual fields.
make_block <- function(family = "binomial",
                       n = 10L,
                       p = 2L,
                       n_trials = 1L,
                       offset = NULL) {
  list(
    y        = if (family == "gaussian") rnorm(n) else rep_len(0:1, n),
    n_trials = n_trials,
    X        = matrix(rnorm(n * p), nrow = n, ncol = p),
    family   = family,
    offset   = offset
  )
}

# Stub tulpa_laplace() that records calls and returns a deterministic mode.
make_recording_stub <- function(env) {
  env$calls <- list()
  function(...) {
    args <- list(...)
    env$calls[[length(env$calls) + 1L]] <- args
    n_fixed <- ncol(args$X)
    list(
      mode      = rep(0.1, n_fixed),
      converged = TRUE,
      n_iter    = 1L,
      H_beta    = diag(n_fixed)
    )
  }
}


test_that("smoke: two homogeneous binomial submodels converge in one step", {
  skip_if_not_installed("testthat", "3.0.0")

  capture_env <- new.env()
  stub <- make_recording_stub(capture_env)

  e_step <- function(fits, ...) {
    list(weights = rep(0.5, 10))
  }
  m_step_encode <- function(weights, ...) {
    list(
      psi = make_block("binomial", n = 10L, p = 2L),
      p   = make_block("binomial", n = 10L, p = 3L)
    )
  }

  local_mocked_bindings(tulpa_laplace = stub, .package = "tulpa")

  res <- tulpa_em_laplace(
    e_step        = e_step,
    m_step_encode = m_step_encode,
    max_iter      = 5L,
    tol           = 1e-4,
    verbose       = FALSE
  )

  expect_named(res, c("fits", "weights", "n_iter", "converged", "history"))
  expect_named(res$fits, c("psi", "p"))
  # With deterministic stub returning identical modes, the second iteration
  # has zero relative change -> convergence.
  expect_true(res$converged)
  expect_lte(res$n_iter, 5L)
  # Each iteration calls tulpa_laplace once per block.
  expect_equal(length(capture_env$calls), 2L * res$n_iter)
})


test_that("heterogeneous-family: binomial + poisson are forwarded per block", {
  capture_env <- new.env()
  stub <- make_recording_stub(capture_env)

  e_step <- function(fits, ...) list(weights = rep(0.5, 8))

  m_step_encode <- function(weights, ...) {
    # `zero` block has n_trials = NULL to exercise the default fill;
    # `counts` carries an explicit poisson family with offset.
    zero <- make_block("binomial", n = 8L, p = 2L, offset = rep(0.1, 8))
    zero$n_trials <- NULL
    list(
      zero   = zero,
      counts = make_block("poisson",  n = 8L, p = 3L,
                           offset = log(rep(2, 8)))
    )
  }

  local_mocked_bindings(tulpa_laplace = stub, .package = "tulpa")

  res <- tulpa_em_laplace(
    e_step        = e_step,
    m_step_encode = m_step_encode,
    max_iter      = 1L,
    verbose       = FALSE
  )

  expect_equal(length(capture_env$calls), 2L)

  # First call -> zero submodel (binomial, offset = rep(0.1, 8))
  call1 <- capture_env$calls[[1]]
  expect_equal(call1$family, "binomial")
  expect_equal(call1$offset, rep(0.1, 8))

  # Second call -> counts submodel (poisson, offset = log(rep(2, 8)))
  call2 <- capture_env$calls[[2]]
  expect_equal(call2$family, "poisson")
  expect_equal(call2$offset, log(rep(2, 8)))

  # n_trials default: NULL -> rep(1L, n).
  expect_equal(call1$n_trials, rep(1L, 8))
})


test_that("missing `family` field raises a clear error", {
  e_step <- function(fits, ...) list(weights = rep(0.5, 5))

  m_step_encode <- function(weights, ...) {
    bad <- list(
      y        = c(0, 1, 1, 0, 1),
      n_trials = NULL,
      X        = matrix(rnorm(10), nrow = 5, ncol = 2),
      offset   = NULL
      # `family` deliberately omitted
    )
    list(only = bad)
  }

  # No need to mock tulpa_laplace -- validation should fire before any call.
  expect_error(
    tulpa_em_laplace(
      e_step        = e_step,
      m_step_encode = m_step_encode,
      max_iter      = 1L,
      verbose       = FALSE
    ),
    regexp = "missing required field.*family"
  )
})


test_that("bad family string raises a clear error", {
  e_step <- function(fits, ...) list(weights = rep(0.5, 5))
  m_step_encode <- function(weights, ...) {
    blk <- make_block("binomial", n = 5L)
    blk$family <- "exotic_family"
    list(only = blk)
  }

  expect_error(
    tulpa_em_laplace(
      e_step        = e_step,
      m_step_encode = m_step_encode,
      max_iter      = 1L,
      verbose       = FALSE
    ),
    regexp = "must be one of.*exotic_family"
  )
})


test_that("offset length mismatch raises a clear error", {
  e_step <- function(fits, ...) list(weights = rep(0.5, 5))
  m_step_encode <- function(weights, ...) {
    blk <- make_block("poisson", n = 5L, offset = rep(0.0, 7))
    list(only = blk)
  }

  expect_error(
    tulpa_em_laplace(
      e_step        = e_step,
      m_step_encode = m_step_encode,
      max_iter      = 1L,
      verbose       = FALSE
    ),
    regexp = "length\\(offset\\)"
  )
})


test_that("correction = 'mi' / 'gibbs' raise not-implemented", {
  noop_e <- function(fits, ...) list(weights = numeric(0))
  noop_m <- function(weights, ...) list()

  expect_error(
    tulpa_em_laplace(noop_e, noop_m, correction = "mi", verbose = FALSE),
    regexp = "not yet implemented"
  )
  expect_error(
    tulpa_em_laplace(noop_e, noop_m, correction = "gibbs", verbose = FALSE),
    regexp = "not yet implemented"
  )
})
