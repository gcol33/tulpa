# The control surface: every door checks its keys, every default has one home.
#
# gcol33/tulpa#673 -- fit_st_nested() had no control-key check at all, so a
# misspelled knob was accepted in silence and the fit ran at the default, and
# two knobs it reads were undocumented with one of them an inline literal.
# gcol33/tulpa#675 -- control$sigma_eps was a second spelling of `phi`, in the
# other convention (an SD against a variance).
# gcol33/tulpa#676 -- the tulpa() dispatch restated defaults the backend
# signatures already carry, so the same number lived in two files.

test_that("fit_st_nested checks its control keys", {
  keys <- tulpa:::.CONTROL_KEYS$st_nested
  expect_true(is.character(keys) && length(keys) > 5L)
  # Both knobs the driver reads and the roxygen did not name.
  expect_true("rho_spatial" %in% keys)
  expect_true("within_cell" %in% keys)

  expect_error(
    tulpa:::tulpa_check_control(list(rho_spatail = 0.5), keys, "fit_st_nested"),
    "rho_spatail")
  expect_silent(
    tulpa:::tulpa_check_control(list(rho_spatial = 0.5), keys, "fit_st_nested"))
})

test_that("the proper-CAR mixing default has one home", {
  # It was an inline literal in fit_st_nested(); .NL_ST_GRID is where every
  # other spatiotemporal grid default lives.
  expect_equal(tulpa:::.nl_st_default("rho_spatial"), 0.9)
  expect_true("rho_spatial" %in% names(tulpa:::.NL_ST_GRID))
  src <- paste(deparse(tulpa:::fit_st_nested), collapse = " ")
  expect_match(src, 'nl_st_default("rho_spatial")', fixed = TRUE)
})

test_that("control$sigma_eps is refused by name", {
  skip_on_cran()
  set.seed(5)
  n <- 80L
  d <- data.frame(x = rnorm(n), g = gl(8, 10))
  d$y <- rnorm(n)
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "agq",
          control = list(sigma_eps = 0.5)),
    "second spelling")
  expect_false("sigma_eps" %in% tulpa:::.CONTROL_KEYS$tulpa)
})

test_that("the dispatch omits a knob the caller did not set", {
  # .drop_null() is what lets a backend's own formal supply the default, so the
  # number lives in one file.
  expect_identical(tulpa:::.drop_null(list(a = 1, b = NULL, c = "x")),
                   list(a = 1, c = "x"))
  expect_identical(tulpa:::.drop_null(list()), list())

  # And the defaults the dispatch used to restate are still the backends' own.
  expect_identical(formals(tulpa:::mala)$n_iter, 2000L)
  expect_identical(formals(tulpa:::pathfinder)$n_draws, 1000L)
  expect_identical(formals(tulpa:::imh_laplace)$n_iter, 2000L)
  expect_identical(formals(tulpa:::agq_fit)$n_quad, 7L)

  src <- paste(deparse(tulpa:::.tulpa_fitter_args), collapse = " ")
  expect_false(grepl("control$n_iter %||% 2000L", src, fixed = TRUE))
  expect_false(grepl("control$n_draws %||% 1000L", src, fixed = TRUE))
})

test_that("a fit still honours a knob the caller does set", {
  skip_on_cran()
  set.seed(6)
  n <- 120L
  d <- data.frame(x = rnorm(n))
  d$y <- rbinom(n, 1, plogis(0.3 + 0.6 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "mala",
               control = list(n_iter = 300L, warmup = 100L, seed = 1L))
  expect_identical(nrow(fit$draws), 200L)
})
