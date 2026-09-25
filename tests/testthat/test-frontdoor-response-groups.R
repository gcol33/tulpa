# The formula front door's reading of the response and the grouping factors
# (gcol33/tulpa#880, #881): inputs refused or normalised at the door rather
# than reaching a backend that fails in its own words.

.rg_data <- function() {
  set.seed(1)
  n <- 200L
  x <- rnorm(n)
  data.frame(g = factor(sample(1:10, n, TRUE)), x = x,
             yb = rbinom(n, 1, plogis(x)), y = rpois(n, exp(0.5 + 0.3 * x)))
}

test_that("a non-numeric response is refused by name (#880)", {
  d <- .rg_data()
  d$yl <- d$yb == 1
  expect_error(tulpa(yl ~ x, d, family = "poisson"),
               "response is logical, which family = 'poisson' does not read")
  d$f <- factor(ifelse(d$yl, "a", "b"))
  expect_error(tulpa(f ~ x, d, family = "binomial"),
               "response is a factor, which family = 'binomial' does not read")
  expect_error(tulpa_laplace(as.character(d$yb), rep(1L, 200), cbind(1, d$x),
                             family = "binomial"),
               "response is a character")
})

test_that("an NA group or random slope is refused at the door (#880)", {
  d <- .rg_data()
  d$g[5] <- NA
  for (m in c("laplace", "mala", "eb")) {
    expect_error(tulpa(y ~ x + (1 | g), d, family = "poisson", mode = m,
                       sigma_re = if (m == "laplace") 1),
                 "Missing value\\(s\\) in the grouping variable `g` \\(1 row\\(s\\), first at row 5\\)")
  }
  d <- .rg_data()
  d$z <- rnorm(200)
  d$z[7] <- NA
  expect_error(tulpa(y ~ x + (1 + z | g), d, family = "poisson"),
               "random-slope covariate\\(s\\) of `g` \\(1 row\\(s\\), first at row 7\\)")
  fam <- tulpa_family("pois", function(eta, params, n_obs, ...) {
    rpois(n_obs, exp(eta[[1]]))
  })
  d$g[5] <- NA
  expect_error(prior_predict(y ~ x + (1 | g), fam, d),
               "prior_predict: Missing value\\(s\\) in the grouping variable")
})

test_that("tulpa_build_model_data drops unused grouping levels (#880, #881)", {
  d <- data.frame(y = 1:6, g = factor(c("a", "a", "c", "c", "d", "d"),
                                      levels = c("a", "b", "c", "d", "e")))
  b <- tulpa_build_model_data(tulpa_parse_formula(y ~ 1 + (1 | g)), d)
  expect_equal(b$re_terms[[1]]$n_groups, 3L)
  expect_equal(b$re_terms[[1]]$levels, c("a", "c", "d"))
  expect_equal(b$re_terms[[1]]$group_idx, c(1L, 1L, 2L, 2L, 3L, 3L))
})

test_that("a logical binomial response is read as 0/1 on every door (#880)", {
  skip_on_cran()
  d <- .rg_data()
  d$yl <- d$yb == 1
  for (m in c("laplace", "eb")) {
    a <- tulpa(yl ~ x + (1 | g), d, family = "binomial", mode = m,
               sigma_re = if (m == "laplace") 0.7)
    b <- tulpa(yb ~ x + (1 | g), d, family = "binomial", mode = m,
               sigma_re = if (m == "laplace") 0.7)
    expect_equal(coef(a), coef(b))
  }
  expect_equal(tulpa_laplace(d$yl, rep(1L, 200), cbind(1, d$x),
                             family = "binomial")$mode,
               tulpa_laplace(d$yb, rep(1L, 200), cbind(1, d$x),
                             family = "binomial")$mode)
})
