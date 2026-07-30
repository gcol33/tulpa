# A fit's display is driven by the structure the fit carries, not by which
# function produced it: any fit with a random-effect covariance reports it, and
# any fit with s(...) terms lists and plots them. These are the tests that keep
# the sections reachable from the one front door.

test_that("the display sections are silent on a fit carrying neither structure", {
  # print.tulpa_fit() calls both unconditionally, so a fit with no random effects
  # and no smoothers must produce nothing rather than an empty table.
  bare <- structure(list(), class = "tulpa_fit")
  expect_silent(tulpa:::.print_re_section(bare))
  expect_silent(tulpa:::.print_smooth_section(bare))
})

test_that("the smooth view is offered on every fit and refuses one with no smoothers", {
  expect_true("smooth" %in% eval(formals(tulpa:::plot.tulpa_fit)$type))
  expect_true("term" %in% names(formals(tulpa:::plot.tulpa_fit)))
  bare <- structure(list(), class = "tulpa_fit")
  expect_error(plot(bare, type = "smooth"), "no s\\(\\.\\.\\.\\) smoother terms")
})

test_that("a random-intercept fit reports its covariance through print()", {
  skip_on_cran()
  set.seed(4)
  n <- 200L
  g <- rep(seq_len(10), length.out = n)
  d <- data.frame(x = rnorm(n), g = factor(g))
  d$y <- rnorm(n, 0.3 + 0.5 * d$x + rnorm(10, 0, 0.6)[g], 1)

  fit <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian",
               mode = "laplace", sigma_re = 0.6)

  out <- utils::capture.output(print(fit))
  expect_true(any(grepl("Random effects", out, fixed = TRUE)))
  # sigma_re was supplied, so VarCorr must label it conditioned rather than
  # present the input as an estimate.
  expect_true(any(grepl("conditioned", out, fixed = TRUE)))
})

test_that("a smoother fit lists its smoothers alongside the tier's own report", {
  skip_on_cran()
  set.seed(5)
  d <- data.frame(x = runif(200, -2, 2))
  d$y <- rpois(200, exp(0.3 + sin(2 * d$x)))

  fit <- tulpa(y ~ s(x, k = 12), data = d, family = "poisson")

  out <- utils::capture.output(print(fit))
  expect_true(any(grepl("s(x)", out, fixed = TRUE)))
  # A smoother routes to the nested-Laplace tier, whose print method composes
  # its own hyperparameter report with the generic fixed-effect body. Both
  # halves must be there: dispatching to the tier must not drop the fixed
  # effects, and printing the fit must not drop the integrated hyperparameters.
  expect_true(any(grepl("hyperparameters", out, fixed = TRUE)))
  expect_true(any(grepl("Fixed effects", out, fixed = TRUE)))

  sm <- smooth_effects(fit)
  expect_true(is.data.frame(sm) && nrow(sm) == 12L)

  grDevices::pdf(NULL)
  on.exit(grDevices::dev.off(), add = TRUE)
  expect_silent(plot(fit, type = "smooth"))
})
