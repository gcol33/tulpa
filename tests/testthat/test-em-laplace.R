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

  expect_named(res, c("fits", "weights", "n_iter", "converged", "history",
                      "correction"))
  expect_equal(res$correction, "none")
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


test_that("correction = 'mi' runs n_imputations draws and pools via Rubin", {
  capture_env <- new.env()
  stub <- make_recording_stub(capture_env)

  e_step <- function(fits, ...) list(weights = rep(0.5, 10))
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
    correction    = "mi",
    n_imputations = 4L,
    max_iter      = 2L,
    tol           = 1e-12,
    verbose       = FALSE
  )

  expect_equal(res$correction, "mi")
  expect_named(res$pooled, c("psi", "p"))
  expect_length(res$draws, 4L)
  # Each draw fits both submodels -> 2 calls per draw on top of EM calls.
  expect_equal(length(capture_env$calls), 2L * res$n_iter + 2L * 4L)
  # Pooled summaries carry mean / se / V_within / V_between / V_total / K.
  expect_named(res$pooled$psi,
               c("mean", "se", "V_within", "V_between", "V_total", "K"))
  expect_equal(res$pooled$psi$K, 4L)
  # Stub returns identical modes, so between-imputation variance is 0
  # and pooled se equals sqrt(within) = sqrt(1) = 1 (H_beta = I -> se = 1).
  expect_equal(res$pooled$psi$V_between, c(0, 0))
  expect_equal(res$pooled$psi$se, c(1, 1))
})


test_that("correction = 'gibbs' runs n_gibbs steps and pools via Rubin", {
  capture_env <- new.env()
  stub <- make_recording_stub(capture_env)

  e_step <- function(fits, ...) list(weights = rep(0.5, 10))
  m_step_encode <- function(weights, ...) {
    list(only = make_block("binomial", n = 10L, p = 2L))
  }

  local_mocked_bindings(tulpa_laplace = stub, .package = "tulpa")

  res <- tulpa_em_laplace(
    e_step        = e_step,
    m_step_encode = m_step_encode,
    correction    = "gibbs",
    n_gibbs       = 3L,
    max_iter      = 2L,
    tol           = 1e-12,
    verbose       = FALSE
  )

  expect_equal(res$correction, "gibbs")
  expect_named(res$pooled, "only")
  expect_length(res$draws, 3L)
  # 1 submodel x (EM iters + n_gibbs).
  expect_equal(length(capture_env$calls), res$n_iter + 3L)
  expect_equal(res$pooled$only$K, 3L)
})


test_that("correction = 'mi' invokes a user-supplied draw_z", {
  capture_env <- new.env()
  stub <- make_recording_stub(capture_env)
  draw_calls <- 0L
  capture_env$encode_weights <- list()

  draw_z_custom <- function(weights) {
    draw_calls <<- draw_calls + 1L
    rep(0L, length(weights))
  }

  e_step <- function(fits, ...) list(weights = rep(0.5, 6))
  m_step_encode <- function(weights, ...) {
    capture_env$encode_weights[[length(capture_env$encode_weights) + 1L]] <-
      weights
    list(only = make_block("binomial", n = 6L, p = 2L))
  }

  local_mocked_bindings(tulpa_laplace = stub, .package = "tulpa")

  res <- tulpa_em_laplace(
    e_step        = e_step,
    m_step_encode = m_step_encode,
    correction    = "mi",
    n_imputations = 5L,
    draw_z        = draw_z_custom,
    max_iter      = 1L,
    tol           = 1e-12,
    verbose       = FALSE
  )

  expect_equal(draw_calls, 5L)
  expect_equal(res$pooled$only$K, 5L)
  # MI draws (the tail of encode_weights) all see hard z = 0.
  mi_calls <- tail(capture_env$encode_weights, 5)
  expect_true(all(vapply(mi_calls,
                         function(w) all(w == 0L), logical(1))))
})


test_that("draw_z = NULL with non-numeric weights gives a clear error", {
  stub <- make_recording_stub(new.env())

  e_step <- function(fits, ...) list(weights = matrix(0.5, 4, 2))
  m_step_encode <- function(weights, ...) {
    list(only = make_block("binomial", n = 4L, p = 2L))
  }

  local_mocked_bindings(tulpa_laplace = stub, .package = "tulpa")

  expect_error(
    tulpa_em_laplace(
      e_step        = e_step,
      m_step_encode = m_step_encode,
      correction    = "mi",
      n_imputations = 2L,
      max_iter      = 1L,
      verbose       = FALSE
    ),
    regexp = "numeric vector of Bernoulli"
  )
})


test_that("bad n_imputations / n_gibbs / draw_z raise clear errors", {
  noop_e <- function(fits, ...) list(weights = rep(0.5, 4))
  noop_m <- function(weights, ...) {
    list(only = make_block("binomial", n = 4L, p = 2L))
  }

  expect_error(
    tulpa_em_laplace(noop_e, noop_m, correction = "mi",
                     n_imputations = 0L, max_iter = 1L, verbose = FALSE),
    regexp = "n_imputations"
  )
  expect_error(
    tulpa_em_laplace(noop_e, noop_m, correction = "gibbs",
                     n_gibbs = 0L, max_iter = 1L, verbose = FALSE),
    regexp = "n_gibbs"
  )
  expect_error(
    tulpa_em_laplace(noop_e, noop_m, correction = "mi",
                     draw_z = 42, max_iter = 1L, verbose = FALSE),
    regexp = "draw_z"
  )
})
