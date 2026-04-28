test_that("findbars finds bar terms in formula AST", {
  f <- y ~ x + (1 | group)
  bars <- findbars(f[[3]])
  expect_length(bars, 1)
  expect_equal(deparse(bars[[1]][[1]]), "|")
})

test_that("findbars handles multiple RE terms", {
  f <- y ~ x + (1 | group) + (x || site)
  bars <- findbars(f[[3]])
  expect_length(bars, 2)
})

test_that("findbars returns NULL for no RE", {
  f <- y ~ x1 + x2
  bars <- findbars(f[[3]])
  expect_null(bars)
})

test_that("nobars removes bar terms from AST", {
  f <- y ~ x + (1 | group)
  clean <- nobars(f[[3]])
  # Should just be x (no bar term)
  expect_false(grepl("|", deparse(clean), fixed = TRUE))
})

test_that("nobars handles formula with only RE terms", {
  f <- y ~ (1 | group)
  clean <- nobars(f[[3]])
  expect_null(clean)
})

test_that("tulpa_parse_formula separates fixed and random", {
  pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 | group))
  expect_s3_class(pf, "tulpa_parsed_formula")
  expect_equal(pf$response, "y")
  expect_equal(pf$n_re_terms, 1)
  expect_equal(pf$random_effects[[1]]$group_var, "group")
  expect_true(pf$random_effects[[1]]$has_intercept)
  expect_true(pf$random_effects[[1]]$correlated)
})

test_that("|| expands (1 + x || group) into independent bars (lme4-equivalent)", {
  pf <- tulpa_parse_formula(y ~ x + (1 + x || group))
  # (1 + x || g) -> (1 | g) + (0 + x | g): two bars on the same group,
  # each correlated=TRUE because each has at most one coefficient.
  expect_equal(pf$n_re_terms, 2)
  expect_equal(pf$random_effects[[1]]$group_var, "group")
  expect_equal(pf$random_effects[[2]]$group_var, "group")

  # First bar: intercept-only
  expect_true(pf$random_effects[[1]]$has_intercept)
  expect_length(pf$random_effects[[1]]$slope_terms, 0)

  # Second bar: slope-only, no intercept
  expect_false(pf$random_effects[[2]]$has_intercept)
  expect_length(pf$random_effects[[2]]$slope_terms, 1)

  # Both individual bars carry correlated=TRUE (a single coef can't be
  # uncorrelated with itself); the diagonal structure comes from being
  # split across separate bars.
  expect_true(pf$random_effects[[1]]$correlated)
  expect_true(pf$random_effects[[2]]$correlated)
})

test_that("|| with bare slope expands as (x || g) -> (1 | g) + (0 + x | g)", {
  pf <- tulpa_parse_formula(y ~ (x || g))
  expect_equal(pf$n_re_terms, 2)
  expect_true(pf$random_effects[[1]]$has_intercept)
  expect_length(pf$random_effects[[1]]$slope_terms, 0)
  expect_false(pf$random_effects[[2]]$has_intercept)
  expect_length(pf$random_effects[[2]]$slope_terms, 1)
})

test_that("|| with no intercept does not invent one", {
  pf <- tulpa_parse_formula(y ~ (0 + x1 + x2 || g))
  # No intercept; each slope on its own bar.
  expect_equal(pf$n_re_terms, 2)
  expect_false(pf$random_effects[[1]]$has_intercept)
  expect_false(pf$random_effects[[2]]$has_intercept)
})

test_that("tulpa_parse_formula handles crossed RE", {
  pf <- tulpa_parse_formula(y ~ x + (1 | site) + (1 | year))
  expect_equal(pf$n_re_terms, 2)
  expect_equal(pf$random_effects[[1]]$group_var, "site")
  expect_equal(pf$random_effects[[2]]$group_var, "year")
})

test_that("tulpa_parse_formula handles nested RE (a/b)", {
  pf <- tulpa_parse_formula(y ~ x + (1 | region/site))
  # Should expand to (1|region) + (1|region:site)
  expect_equal(pf$n_re_terms, 2)
  expect_equal(pf$random_effects[[1]]$group_var, "region")
  expect_equal(pf$random_effects[[2]]$group_var, "region:site")
})

test_that("tulpa_parse_formula handles random slope without intercept", {
  pf <- tulpa_parse_formula(y ~ x + (0 + x | group))
  expect_equal(pf$n_re_terms, 1)
  expect_false(pf$random_effects[[1]]$has_intercept)
})

test_that("tulpa_build_model_data constructs matrices", {
  set.seed(1)
  df <- data.frame(
    y = rnorm(100),
    x = rnorm(100),
    group = rep(1:10, each = 10)
  )
  pf <- tulpa_parse_formula(y ~ x + (1 | group))
  md <- tulpa_build_model_data(pf, df)

  expect_equal(md$n_obs, 100)
  expect_equal(md$n_fixed, 2)  # intercept + x
  expect_equal(md$n_re_terms, 1)
  expect_equal(md$re_terms[[1]]$n_groups, 10)
  expect_equal(length(md$re_terms[[1]]$group_idx), 100)
  expect_equal(length(md$y), 100)
})

test_that("tulpa_build_model_data handles random slopes", {
  set.seed(2)
  df <- data.frame(
    y = rnorm(60),
    x = rnorm(60),
    group = rep(letters[1:6], each = 10)
  )
  pf <- tulpa_parse_formula(y ~ x + (1 + x | group))
  md <- tulpa_build_model_data(pf, df)

  expect_equal(md$re_terms[[1]]$n_coefs, 2)  # intercept + slope
  expect_equal(md$re_terms[[1]]$n_groups, 6)
  expect_false(is.null(md$re_terms[[1]]$slope_matrix))
  expect_equal(ncol(md$re_terms[[1]]$slope_matrix), 1)
})

test_that("print.tulpa_parsed_formula works", {
  # Use only correlated bars so the count is unambiguous post-|| expansion.
  pf <- tulpa_parse_formula(y ~ x + (1 | g) + (1 + x | site))
  expect_output(print(pf), "tulpa parsed formula")
  expect_output(print(pf), "Random effects: 2")
})

# ============================================================================
# P1.1: Edge cases that would break deparse+regex
# ============================================================================

test_that("has_implicit_intercept works on AST, not strings", {
  # These are the cases that break regex: poly(), backticks, I()
  expect_true(has_implicit_intercept(quote(x)))
  expect_true(has_implicit_intercept(quote(1 + x)))
  expect_true(has_implicit_intercept(quote(x + z)))
  expect_false(has_implicit_intercept(quote(0 + x)))
  expect_false(has_implicit_intercept(0))
  expect_false(has_implicit_intercept(quote(-1 + x)))
})

test_that("formula parser handles poly(x, 2) in slopes", {
  set.seed(42)
  df <- data.frame(
    y = rnorm(50),
    x = rnorm(50),
    group = rep(1:5, each = 10)
  )
  pf <- tulpa_parse_formula(y ~ poly(x, 2) + (1 + poly(x, 2) | group))
  expect_equal(pf$n_re_terms, 1)
  expect_true(pf$random_effects[[1]]$has_intercept)
  expect_length(pf$random_effects[[1]]$slope_terms, 1)

  md <- tulpa_build_model_data(pf, df)
  # poly(x, 2) produces 2 columns
  expect_equal(md$re_terms[[1]]$n_coefs, 3)  # intercept + 2 poly columns
})

test_that("formula parser handles backtick-quoted names", {
  df <- data.frame(
    y = rnorm(30),
    `weird name` = rnorm(30),
    group = rep(1:3, each = 10),
    check.names = FALSE
  )
  pf <- tulpa_parse_formula(y ~ `weird name` + (1 + `weird name` | group))
  expect_equal(pf$n_re_terms, 1)
  expect_true(pf$random_effects[[1]]$has_intercept)

  md <- tulpa_build_model_data(pf, df)
  expect_equal(md$re_terms[[1]]$n_coefs, 2)  # intercept + slope
})

test_that("formula parser handles I(x^2) in slopes", {
  set.seed(99)
  df <- data.frame(
    y = rnorm(40),
    x = rnorm(40),
    group = rep(1:4, each = 10)
  )
  pf <- tulpa_parse_formula(y ~ x + I(x^2) + (1 + I(x^2) | group))
  expect_equal(pf$n_re_terms, 1)
  expect_true(pf$random_effects[[1]]$has_intercept)

  md <- tulpa_build_model_data(pf, df)
  expect_equal(md$re_terms[[1]]$n_coefs, 2)  # intercept + I(x^2)
})

test_that("decompose_bar_lhs extracts slope language objects", {
  d <- decompose_bar_lhs(quote(1 + x + z))
  expect_true(d$has_intercept)
  expect_length(d$slope_terms, 2)
  expect_equal(deparse(d$slope_terms[[1]]), "x")
  expect_equal(deparse(d$slope_terms[[2]]), "z")

  d2 <- decompose_bar_lhs(quote(0 + x))
  expect_false(d2$has_intercept)
  expect_length(d2$slope_terms, 1)

  d3 <- decompose_bar_lhs(quote(1))
  expect_true(d3$has_intercept)
  expect_length(d3$slope_terms, 0)
})

test_that("formula parser handles multiple slopes correctly", {
  set.seed(7)
  df <- data.frame(
    y = rnorm(50),
    x1 = rnorm(50),
    x2 = rnorm(50),
    group = rep(1:5, each = 10)
  )
  pf <- tulpa_parse_formula(y ~ x1 + x2 + (1 + x1 + x2 | group))
  md <- tulpa_build_model_data(pf, df)
  expect_equal(md$re_terms[[1]]$n_coefs, 3)  # intercept + x1 + x2
  expect_equal(ncol(md$re_terms[[1]]$slope_matrix), 2)
})

# ============================================================================
# Response handling: language objects, no deparse/reparse round-trip
# ============================================================================

test_that("parsed$response_expr is a language object, not deparsed text", {
  pf <- tulpa_parse_formula(y ~ x)
  expect_true(is.language(pf$response_expr))
  expect_identical(pf$response_expr, quote(y))
  expect_equal(pf$response, "y")
})

test_that("cbind(succ, fail) responses survive round-tripping", {
  set.seed(11)
  df <- data.frame(
    succ  = sample.int(10, 30, replace = TRUE),
    fail  = sample.int(10, 30, replace = TRUE),
    x     = rnorm(30),
    group = rep(1:3, each = 10)
  )
  pf <- tulpa_parse_formula(cbind(succ, fail) ~ x + (1 | group))
  expect_true(is.call(pf$response_expr))
  md <- tulpa_build_model_data(pf, df)
  expect_true(is.matrix(md$y))
  expect_equal(ncol(md$y), 2L)
  expect_equal(nrow(md$y), 30L)
})

test_that("backtick-quoted response is evaluated correctly", {
  df <- data.frame(check.names = FALSE,
                   `weird name` = rnorm(20),
                   x = rnorm(20),
                   g = rep(1:2, each = 10))
  pf <- tulpa_parse_formula(`weird name` ~ x + (1 | g))
  md <- tulpa_build_model_data(pf, df)
  expect_equal(length(md$y), 20L)
})

# ============================================================================
# Nested grouping: structured group_vars instead of "a:b" colon-string
# ============================================================================

test_that("nested grouping carries structured group_vars vector", {
  pf <- tulpa_parse_formula(y ~ (1 | a/b))
  expect_equal(pf$n_re_terms, 2L)
  expect_equal(pf$random_effects[[1]]$group_vars, "a")
  expect_equal(pf$random_effects[[2]]$group_vars, c("a", "b"))
  # Display label still uses "a:b" for the nested level
  expect_equal(pf$random_effects[[2]]$group_var, "a:b")
})

test_that("nested grouping builds an interaction factor on the right columns", {
  set.seed(13)
  df <- data.frame(
    y = rnorm(60),
    a = rep(letters[1:3], each = 20),
    b = rep(1:4, times = 15)
  )
  pf <- tulpa_parse_formula(y ~ (1 | a/b))
  md <- tulpa_build_model_data(pf, df)
  expect_equal(md$re_terms[[1]]$n_groups, 3L)
  expect_equal(md$re_terms[[2]]$n_groups, 12L)  # 3 * 4 fully crossed
  expect_equal(md$re_terms[[2]]$group_vars, c("a", "b"))
})

test_that("missing nested-group columns produce a clear error", {
  df <- data.frame(y = 1:6, a = rep(letters[1:3], 2))
  pf <- tulpa_parse_formula(y ~ (1 | a/b))
  expect_error(tulpa_build_model_data(pf, df), "not found in data")
})

# ============================================================================
# Offset extraction
# ============================================================================

test_that("offset(log(area)) is extracted from fixed-effect RHS", {
  set.seed(17)
  df <- data.frame(
    y    = rpois(40, 1),
    x    = rnorm(40),
    area = runif(40, 1, 100),
    g    = rep(1:4, each = 10)
  )
  pf <- tulpa_parse_formula(y ~ x + offset(log(area)) + (1 | g))
  md <- tulpa_build_model_data(pf, df)
  # Offset is exposed as a numeric vector
  expect_true(is.numeric(md$offset))
  expect_equal(length(md$offset), 40L)
  expect_equal(md$offset, log(df$area))
  # Design matrix excludes the offset
  expect_equal(colnames(md$X), c("(Intercept)", "x"))
})

test_that("absent offset() returns NULL, not zeros", {
  df <- data.frame(y = 1:5, x = 1:5)
  pf <- tulpa_parse_formula(y ~ x)
  md <- tulpa_build_model_data(pf, df)
  expect_null(md$offset)
})

# ============================================================================
# Vestigial fields: terms = deparse(lhs) is gone
# ============================================================================

test_that("RE specs no longer carry a deparsed `terms` field", {
  pf <- tulpa_parse_formula(y ~ x + (1 + x | g))
  expect_null(pf$random_effects[[1]]$terms)
  # slope_terms (language objects) is the source of truth
  expect_length(pf$random_effects[[1]]$slope_terms, 1)
  expect_equal(pf$random_effects[[1]]$slope_terms[[1]], quote(x))
})

# ============================================================================
# Slope formula carries the original formula's environment
# ============================================================================

test_that("slope construction can find user-scope helpers via formula env", {
  # Define a helper in the calling scope; reference it from a slope.
  custom_scale <- function(z) (z - mean(z)) / sd(z)
  set.seed(19)
  df <- data.frame(y = rnorm(30), x = rnorm(30), g = rep(1:3, each = 10))
  f <- y ~ custom_scale(x) + (1 + custom_scale(x) | g)
  pf <- tulpa_parse_formula(f)
  # Slope build must not bail out looking for `custom_scale`
  md <- tulpa_build_model_data(pf, df)
  expect_equal(md$re_terms[[1]]$n_coefs, 2L)
})
