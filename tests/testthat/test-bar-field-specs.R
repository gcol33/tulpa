# Exported varying-coefficient bar expansion (gcol33/tulpa#93). The same bar
# grammar spatial() and the inline temporal() field constructor consume,
# surfaced for downstream packages: tulpa_bar_field_specs() expands a
# ~ 1 + w || node bar into one (column_name, weight, is_intercept) spec per
# design column, and tulpa_is_spatial_bar() recognizes the bar. The wrapper
# delegates to the same .bar_field_columns() the engine's own fitters use, so
# the consumer cannot drift from the engine.

test_that("intercept-only bar expands to one unweighted intercept spec", {
  d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))
  specs <- tulpa_bar_field_specs(~ 1 || cell, d)

  expect_length(specs, 1L)
  expect_identical(specs[[1]]$column_name, "Intercept")
  expect_true(specs[[1]]$is_intercept)
  expect_null(specs[[1]]$weight)
  expect_identical(attr(specs, "node"), "cell")
  expect_false(attr(specs, "correlated"))
})

test_that("intercept + slope bar expands to intercept + weighted spec", {
  d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))
  specs <- tulpa_bar_field_specs(~ 1 + w || cell, d)

  expect_length(specs, 2L)

  expect_identical(specs[[1]]$column_name, "Intercept")
  expect_true(specs[[1]]$is_intercept)
  expect_null(specs[[1]]$weight)

  expect_identical(specs[[2]]$column_name, "w")
  expect_false(specs[[2]]$is_intercept)
  expect_equal(specs[[2]]$weight, d$w)
  expect_length(specs[[2]]$weight, nrow(d))

  expect_identical(attr(specs, "node"), "cell")
})

test_that("slope-only bar drops the intercept", {
  d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))
  specs <- tulpa_bar_field_specs(~ 0 + w || cell, d)

  expect_length(specs, 1L)
  expect_identical(specs[[1]]$column_name, "w")
  expect_false(specs[[1]]$is_intercept)
  expect_equal(specs[[1]]$weight, d$w)
})

test_that("a single bar | flags correlated fields", {
  d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))
  specs <- tulpa_bar_field_specs(~ 1 + w | cell, d)

  expect_length(specs, 2L)
  expect_true(attr(specs, "correlated"))
  expect_identical(attr(specs, "node"), "cell")
})

test_that("the wrapper matches the engine's internal column expansion", {
  adj <- matrix(0, 5, 5)
  for (i in 1:4) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))

  f <- spatial(graph = adj, formula = ~ 1 + w || cell)
  internal <- tulpa:::.bar_field_columns(f, d)
  exported <- tulpa_bar_field_specs(~ 1 + w || cell, d)

  expect_identical(internal[[1]]$name, exported[[1]]$column_name)
  expect_identical(internal[[2]]$name, exported[[2]]$column_name)
  # The intercept's all-ones weight is dropped to NULL in the exported contract.
  expect_null(exported[[1]]$weight)
  expect_equal(internal[[2]]$weight, exported[[2]]$weight)
})

test_that("tulpa_is_spatial_bar recognizes bars and rejects non-bars", {
  expect_true(tulpa_is_spatial_bar(~ 1 + w || cell))
  expect_true(tulpa_is_spatial_bar(~ 1 + w | cell))
  expect_true(tulpa_is_spatial_bar(~ 1 || cell))
  # A bare bar language object, not wrapped in a formula.
  expect_true(tulpa_is_spatial_bar(quote(1 + w || cell)))

  # A plain formula with no bar.
  expect_false(tulpa_is_spatial_bar(~ 1 + w))
  # A non-bar term (a plain call).
  expect_false(tulpa_is_spatial_bar(quote(1 + w)))
  # Non-formula / non-language inputs.
  expect_false(tulpa_is_spatial_bar("cell"))
  expect_false(tulpa_is_spatial_bar(42))
  # A two-sided formula is not a one-sided bar.
  expect_false(tulpa_is_spatial_bar(y ~ x))
})

test_that("tulpa_bar_field_specs enforces the bar grammar", {
  d <- data.frame(cell = rep(1:5, each = 4), site = rep(1:2, 10), w = rnorm(20))

  # No bar.
  expect_error(tulpa_bar_field_specs(~ 1 + w, d), "grouping bar")
  # Two-sided formula.
  expect_error(tulpa_bar_field_specs(y ~ 1 + w || cell, d), "one-sided")
  # Nested grouping.
  expect_error(tulpa_bar_field_specs(~ 1 || cell / site, d), "Nested grouping")
  # Interaction / expression RHS.
  expect_error(tulpa_bar_field_specs(~ 1 || cell:site, d),
               "single bare node-index")
})
