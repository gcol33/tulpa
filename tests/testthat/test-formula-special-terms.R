# Parser recognition of spatial(col) / temporal(col) special terms (Increment 2a
# of the gibbs-spatial front-door work). These name a data column; the spec
# (type, adjacency, coords) arrives via tulpa()'s spatial=/temporal= argument.
# The term is stripped from the fixed-effects formula.

rhs_labels <- function(pf) attr(terms(pf$fixed_formula), "term.labels")

test_that("spatial(col) is recorded and stripped from the fixed formula", {
  pf <- tulpa_parse_formula(y ~ x + spatial(region))
  expect_identical(pf$spatial_var, "region")
  expect_null(pf$temporal_var)
  expect_identical(rhs_labels(pf), "x")          # spatial() removed
  expect_false("region" %in% all.vars(pf$fixed_formula))
})

test_that("temporal(col) is recorded and stripped", {
  pf <- tulpa_parse_formula(y ~ x + temporal(week))
  expect_identical(pf$temporal_var, "week")
  expect_null(pf$spatial_var)
  expect_identical(rhs_labels(pf), "x")
})

test_that("spatial + temporal + RE coexist in one formula", {
  pf <- tulpa_parse_formula(y ~ x + spatial(region) + temporal(week) + (1 | site))
  expect_identical(pf$spatial_var, "region")
  expect_identical(pf$temporal_var, "week")
  expect_equal(pf$n_re_terms, 1L)
  expect_identical(rhs_labels(pf), "x")
})

test_that("plain formulas are unaffected (no regression)", {
  pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 | g))
  expect_null(pf$spatial_var)
  expect_null(pf$temporal_var)
  expect_setequal(rhs_labels(pf), c("x1", "x2"))
  expect_equal(pf$n_re_terms, 1L)
})

test_that("invalid special terms error clearly", {
  expect_error(tulpa_parse_formula(y ~ spatial(a) + spatial(b)),
               "At most one spatial")
  expect_error(tulpa_parse_formula(y ~ spatial(a + b)),
               "one bare column name")
})
