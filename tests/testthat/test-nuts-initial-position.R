skip_on_cran_budget()

test_that("C callables reject non-finite initial densities and gradients", {
  load_cpp_fixture("nuts_initial_position")
  for (chains in c(1L, 2L)) {
    for (bad in c(NaN, NA_real_, Inf, -Inf)) {
      for (fault in 1:3) {
        message <- if (fault == 2L) {
          "non-finite gradient at the initial position, coordinate 5 \\(chain 1\\)"
        } else {
          "non-finite log density at the initial position \\(chain 1\\)"
        }
        expect_silent(expect_error(cpp_test_nuts_initial_position(
          matrix(0, chains, 5), fault = fault, bad_value = bad), message))
      }
    }
  }
})

test_that("sampler controls preserve initial-position validation", {
  load_cpp_fixture("nuts_initial_position")
  controls <- list(list(), list(L = 5L), list(walnuts = TRUE))
  for (control in controls) {
    for (chains in c(1L, 2L)) {
      for (epsilon in c(0, 0.1)) {
        for (fault in 1:2) {
          message <- if (fault == 1L) "non-finite log density" else
            "non-finite gradient at the initial position, coordinate 5"
          args <- c(list(init = matrix(0, chains, 5), fault = fault,
                         epsilon = epsilon), control)
          expect_silent(expect_error(do.call(cpp_test_nuts_initial_position, args), message))
        }
      }
    }
  }
})

test_that("every dispersed and explicitly supplied start is checked", {
  load_cpp_fixture("nuts_initial_position")
  for (fault in 1:3) {
    message <- if (fault == 2L) {
      "non-finite gradient at the initial position, coordinate 5 \\(chain 2\\)"
    } else {
      "non-finite log density at the initial position \\(chain 2\\)"
    }
    for (warmup in c(0L, 20L)) {
      starts <- matrix(0, 3, 5)
      if (warmup == 0L) starts[2, ] <- 0.3
      expect_silent(expect_error(cpp_test_nuts_initial_position(
        starts, fault = fault, origin_only = TRUE, n_warmup = warmup), message))
    }
  }
})

test_that("verification rejects non-finite analytic and numerical gradients", {
  load_cpp_fixture("nuts_initial_position")
  for (fault in 1:5) {
    message <- switch(fault, "non-finite log density",
      "non-finite gradient at the initial position, coordinate 5",
      "non-finite log density",
      "non-finite gradient at the initial position, coordinate 1",
      "non-finite gradient at the initial position, coordinate 1")
    expect_silent(expect_error(cpp_test_nuts_initial_position(
      matrix(0, 1, 5), fault = fault, direct = TRUE), message))
  }
  expect_silent(expect_error(cpp_test_nuts_initial_position(
    matrix(0, 1, 5), fault = 4L, numerical = TRUE),
    "non-finite gradient at the initial position, coordinate 1"))
  for (chains in c(1L, 2L)) {
    for (fault in 4:5) {
      expect_silent(expect_error(cpp_test_nuts_initial_position(
        matrix(0, chains, 5), fault = fault),
        "non-finite gradient at the initial position, coordinate 1"))
    }
  }
})

test_that("finite gradient disagreement keeps the numerical fallback", {
  load_cpp_fixture("nuts_initial_position")
  expect_warning(fit <- cpp_test_nuts_initial_position(
    matrix(0.3, 1, 5), fault = 6L, epsilon = 0.1), "Gradient mismatch")
  expect_true(all(is.finite(fit$draws[[1]])))
  expect_gt(nrow(unique(fit$draws[[1]])), 1L)
})

test_that("finite initial positions sample reproducibly with each control", {
  load_cpp_fixture("nuts_initial_position")
  controls <- list(list(), list(L = 5L), list(walnuts = TRUE))
  for (control in controls) {
    for (chains in c(1L, 2L)) {
      args <- c(list(init = matrix(0.3, chains, 5), epsilon = 0.1), control)
      expect_silent(fit <- do.call(cpp_test_nuts_initial_position, args))
      expect_identical(fit, do.call(cpp_test_nuts_initial_position, args))
      expect_identical(fit$epsilon, rep(0.1, chains))
      for (draws in fit$draws) {
        expect_true(all(is.finite(draws)))
        expect_gt(nrow(unique(draws)), 1L)
      }
      expect_true(all(is.finite(unlist(fit$log_prob))))
    }
  }
})

test_that("a gradient-mode change validates each chain under the selected mode", {
  load_cpp_fixture("nuts_initial_position")
  starts <- rbind(rep(0.3, 5), rep(0, 5))
  expect_warning(expect_error(cpp_test_nuts_initial_position(starts, fault = 7L),
    "non-finite gradient at the initial position, coordinate 1 \\(chain 2\\)"),
    "Gradient mismatch")
})

test_that("generic R entries reject a non-finite target at initialization", {
  X <- matrix(1, 4, 1)
  for (entry in list(tulpa:::cpp_tulpa_fit_generic,
                     tulpa:::cpp_tulpa_fit_generic_chains)) {
    expect_silent(expect_error(entry(y_r = c(0, NaN, 0, 0), X_r = X,
      n_iter = 12L, n_warmup = 6L, verbose = FALSE),
      "non-finite log density at the initial position \\(chain 1\\)"))
  }
  for (bad in c(NaN, Inf, -Inf)) {
    expect_silent(expect_error(tulpa:::cpp_tulpa_fit_generic(
      y_r = rep(0, 4), X_r = X, init = c(bad, 0),
      n_iter = 12L, n_warmup = 6L, verbose = FALSE),
      "non-finite log density at the initial position \\(chain 1\\)"))
    starts <- matrix(0, 2, 2)
    starts[2, 1] <- bad
    expect_silent(expect_error(tulpa:::cpp_tulpa_fit_generic_chains(
      y_r = rep(0, 4), X_r = X, n_chains = 2L, init = starts,
      n_iter = 12L, n_warmup = 6L, verbose = FALSE),
      "non-finite log density at the initial position \\(chain 2\\)"))
  }
  expect_silent(fit <- tulpa:::cpp_tulpa_fit_generic(
    y_r = c(-0.2, 0, 0.1, 0.2), X_r = X,
    n_iter = 30L, n_warmup = 10L, verbose = FALSE))
  expect_true(all(is.finite(fit$draws)))
  expect_true(all(is.finite(fit$log_prob)))
})
