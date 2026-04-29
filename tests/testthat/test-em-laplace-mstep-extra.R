# ============================================================================
# tulpa_em_laplace: m_step_extra callback for non-eta parameters
#
# Pure plumbing test: stubs tulpa_laplace so the focus is the callback
# contract (fired between M-step and next E-step; receives the assembled
# fits + current weights; the returned list is used verbatim for the next
# iteration). No C++ Laplace call is exercised here.
# ============================================================================

# Tiny synthetic Poisson-like fixture. The "M-step" stub returns a fits list
# whose `mode` slot encodes a deterministic mapping from weights, so the test
# can verify (a) m_step_extra sees the M-step output and (b) its mutation
# survives into the next iteration.
make_fixture <- function(n = 30L, n_submodels = 2L, seed = 1L) {
  set.seed(seed)
  X <- cbind(1, rnorm(n))
  list(
    n = n,
    n_submodels = n_submodels,
    sub_names = paste0("m", seq_len(n_submodels)),
    X = X,
    y = rpois(n, lambda = 2),
    n_trials = rep(1L, n)
  )
}

# Stub Laplace fit: returns a minimal object with the fields tulpa_em_laplace
# touches downstream (mode, H_beta). The actual values don't matter for the
# m_step_extra plumbing test — they just need to be present and stable.
make_fake_laplace <- function() {
  function(y, n_trials, X, re_list = list(), family = "binomial",
           spatial = NULL, max_iter = 100L, tol = 1e-6,
           n_threads = 1L, return_hessian = TRUE, ...) {
    p <- ncol(X)
    list(
      mode = rep(0.0, p),
      H_beta = diag(p),
      family = family,
      n_obs = length(y),
      phi = NA_real_  # placeholder; m_step_extra is the one that fills this
    )
  }
}

# 1D Newton step on f(phi) = (phi - target)^2 / 2 -> phi := target after one step.
# Used to verify that m_step_extra mutations land in the fits object that the
# engine carries to the next iteration.
newton_phi <- function(phi_old, target) {
  grad <- phi_old - target
  hess <- 1.0
  phi_old - grad / hess  # = target
}

test_that("m_step_extra fires once per iteration and mutations persist", {
  fx <- make_fixture()
  fake_laplace <- make_fake_laplace()

  call_log <- new.env(parent = emptyenv())
  call_log$n_calls <- 0L
  call_log$phis_seen <- list()
  call_log$e_step_phis <- list()

  e_step <- function(fits, ...) {
    # The engine carries the m_step_extra-mutated fits into the next E-step.
    # Capture phi to confirm the mutation reached us.
    call_log$e_step_phis[[length(call_log$e_step_phis) + 1L]] <-
      vapply(fits, function(f) f$phi %||% NA_real_, numeric(1))
    list(weights = rep(0.5, fx$n), delta = 1e-8)
  }

  m_step_encode <- function(weights, ...) {
    specs <- lapply(fx$sub_names, function(nm) {
      list(y = fx$y, n_trials = fx$n_trials, X = fx$X)
    })
    names(specs) <- fx$sub_names
    specs
  }

  # phi targets: each iteration nudges phi toward 2.5 from whatever the
  # previous iteration left behind. Newton on a quadratic converges in one
  # step, so phi should equal 2.5 after the first call.
  m_step_extra <- function(fits, weights, ...) {
    call_log$n_calls <- call_log$n_calls + 1L
    call_log$phis_seen[[call_log$n_calls]] <-
      vapply(fits, function(f) f$phi %||% NA_real_, numeric(1))
    for (k in seq_along(fits)) {
      old <- fits[[k]]$phi
      if (is.null(old) || is.na(old)) old <- 1.0
      fits[[k]]$phi <- newton_phi(old, target = 2.5)
    }
    fits
  }

  testthat::local_mocked_bindings(tulpa_laplace = fake_laplace, .package = "tulpa")

  res <- tulpa_em_laplace(
    e_step = e_step,
    m_step_encode = m_step_encode,
    init = list(),
    m_step_extra = m_step_extra,
    family = "poisson",
    max_iter = 3L, tol = 1e-12, damping = 0,
    correction = "none",
    verbose = FALSE
  )

  # Callback fired once per EM iteration that ran.
  expect_equal(call_log$n_calls, res$convergence$n_iter)

  # Mutated phi survives into the final fits object.
  expect_equal(res$fits$m1$phi, 2.5)
  expect_equal(res$fits$m2$phi, 2.5)

  # Each m_step_extra call sees a freshly rebuilt fits list (M-step
  # re-runs tulpa_laplace), so the stub phi is NA at entry every time.
  expect_true(all(vapply(call_log$phis_seen, function(p) all(is.na(p)),
                         logical(1))))

  # The E-steps that run *after* the first iteration receive the
  # mutated fits — phi reaches them as 2.5. The very first E-step (init)
  # runs before any M-step or m_step_extra, so it sees `init` (an empty
  # list) and we don't constrain that one.
  if (length(call_log$e_step_phis) >= 2L) {
    expect_equal(call_log$e_step_phis[[2]], c(m1 = 2.5, m2 = 2.5))
  }
})

test_that("m_step_extra = NULL is identical to omitting the argument", {
  fx <- make_fixture()
  fake_laplace <- make_fake_laplace()

  e_step <- function(fits, ...) list(weights = rep(0.5, fx$n), delta = 1e-8)
  m_step_encode <- function(weights, ...) {
    specs <- lapply(fx$sub_names, function(nm) {
      list(y = fx$y, n_trials = fx$n_trials, X = fx$X)
    })
    names(specs) <- fx$sub_names
    specs
  }

  testthat::local_mocked_bindings(tulpa_laplace = fake_laplace, .package = "tulpa")

  res_default <- tulpa_em_laplace(
    e_step = e_step, m_step_encode = m_step_encode, init = list(),
    family = "poisson", max_iter = 2L, tol = 1e-12, damping = 0,
    correction = "none", verbose = FALSE
  )
  res_null <- tulpa_em_laplace(
    e_step = e_step, m_step_encode = m_step_encode, init = list(),
    m_step_extra = NULL,
    family = "poisson", max_iter = 2L, tol = 1e-12, damping = 0,
    correction = "none", verbose = FALSE
  )

  # Drop history df (has identical content but env-tagged); compare the rest.
  expect_equal(res_default$fits, res_null$fits)
  expect_equal(res_default$weights, res_null$weights)
  expect_equal(res_default$convergence$n_iter, res_null$convergence$n_iter)
  expect_equal(res_default$convergence$converged, res_null$convergence$converged)
})

test_that("m_step_extra rejects bad shapes with clear errors", {
  fx <- make_fixture()
  fake_laplace <- make_fake_laplace()

  e_step <- function(fits, ...) list(weights = rep(0.5, fx$n), delta = 1e-8)
  m_step_encode <- function(weights, ...) {
    specs <- lapply(fx$sub_names, function(nm) {
      list(y = fx$y, n_trials = fx$n_trials, X = fx$X)
    })
    names(specs) <- fx$sub_names
    specs
  }

  testthat::local_mocked_bindings(tulpa_laplace = fake_laplace, .package = "tulpa")

  # Wrong length.
  bad_len <- function(fits, weights, ...) fits[1]
  expect_error(
    tulpa_em_laplace(
      e_step = e_step, m_step_encode = m_step_encode, init = list(),
      m_step_extra = bad_len, max_iter = 1L, correction = "none", verbose = FALSE
    ),
    "expected 2"
  )

  # Wrong names.
  bad_names <- function(fits, weights, ...) {
    out <- fits
    names(out) <- c("x", "y")
    out
  }
  expect_error(
    tulpa_em_laplace(
      e_step = e_step, m_step_encode = m_step_encode, init = list(),
      m_step_extra = bad_names, max_iter = 1L, correction = "none", verbose = FALSE
    ),
    "preserve submodel names"
  )

  # Non-list return.
  bad_type <- function(fits, weights, ...) "oops"
  expect_error(
    tulpa_em_laplace(
      e_step = e_step, m_step_encode = m_step_encode, init = list(),
      m_step_extra = bad_type, max_iter = 1L, correction = "none", verbose = FALSE
    ),
    "must return a list"
  )

  # Non-function value.
  expect_error(
    tulpa_em_laplace(
      e_step = e_step, m_step_encode = m_step_encode, init = list(),
      m_step_extra = 42, max_iter = 1L, correction = "none", verbose = FALSE
    ),
    "must be NULL or a function"
  )
})
