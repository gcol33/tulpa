set.seed(11)
d <- data.frame(
  y = rbinom(60, 1, 0.4), x = rnorm(60),
  hab = factor(rep("A", 60)),
  g = factor(rep(1:6, 10))
)

test_that("a one-level factor in the fixed effects is named", {
  expect_error(
    tulpa(y ~ x + hab, data = d, family = "binomial", mode = "laplace"),
    "`hab` in the fixed-effects formula is categorical with 1 level \\(only \"A\"\\)"
  )
})

test_that("a one-level character or logical column is named too", {
  d$chr <- "k"
  d$flag <- TRUE
  expect_error(
    tulpa(y ~ x + chr, data = d, family = "binomial", mode = "laplace"),
    "`chr` in the fixed-effects formula"
  )
  expect_error(
    tulpa(y ~ x + flag, data = d, family = "binomial", mode = "laplace"),
    "`flag` in the fixed-effects formula"
  )
})

test_that("a one-level factor in a random-slope term is named", {
  expect_error(
    tulpa(y ~ x + (1 + hab | g), data = d, family = "binomial",
          mode = "laplace", sigma_re = 1),
    "`hab` in a random-slope term"
  )
})

test_that("a factor with two levels is not refused", {
  d$hab <- factor(rep(c("A", "B"), 30))
  parsed <- tulpa_parse_formula(y ~ x + hab)
  md <- tulpa_build_model_data(parsed, d)
  expect_equal(ncol(md$X), 3L)
})
