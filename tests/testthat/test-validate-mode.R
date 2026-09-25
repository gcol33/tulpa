# validate_mode(): a fit matches the backend that ran or the tier it ran in
# (gcol33/tulpa#879 -- it compared against the tier name only, so a fit made
# with mode = "laplace" failed validate_mode(fit, "laplace")).

.vm_fit <- function(backend, tier_name, tier) {
  structure(list(backend = backend, inference_mode = tier_name,
                 inference_tier = tier), class = "tulpa_fit")
}

test_that("validate_mode matches the backend and the tier", {
  f <- .vm_fit("laplace", "structured", 2L)
  expect_true(validate_mode(f, "laplace"))
  expect_true(validate_mode(f, "LAPLACE"))
  expect_true(validate_mode(f, "structured"))
  expect_false(validate_mode(f, "hmc", error = FALSE))
  expect_false(validate_mode(f, "exact", error = FALSE))
  expect_error(validate_mode(f, "hmc"),
               "fit used backend 'laplace' \\(structured, Tier 2\\) but expected 'hmc'")

  g <- .vm_fit("re_cov_gibbs", "exact", 1L)
  expect_true(validate_mode(g, "re_cov_gibbs"))
  expect_true(validate_mode(g, "exact"))
  expect_false(validate_mode(g, "structured", error = FALSE))
})

test_that("validate_mode accepts a set of modes and refuses unknown names", {
  f <- .vm_fit("mala", "exact", 1L)
  expect_true(validate_mode(f, c("laplace", "mala")))
  expect_false(validate_mode(f, c("laplace", "hmc"), error = FALSE))
  expect_error(validate_mode(f, c("laplace", "hmc")), "'laplace' or 'hmc'")
  expect_error(validate_mode(f, "lapalce"), "unknown mode\\(s\\): 'lapalce'")
  expect_error(validate_mode(f, character(0)), "non-empty character vector")
  expect_error(validate_mode(f, NA_character_), "non-empty character vector")
  expect_error(validate_mode(f, 1), "non-empty character vector")
})

test_that("validate_mode accepts the mode a tulpa() fit was run with", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(30))
  d$y <- rpois(30, exp(1 + d$x))
  f <- tulpa(y ~ x, d, family = "poisson", mode = "laplace")
  expect_true(validate_mode(f, "laplace"))
  expect_true(validate_mode(f, f$backend))
  expect_true(validate_mode(f, "structured"))
  expect_true(validate_mode(f, c("laplace", "hmc")))
})
