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

test_that("a random-effects-only formula is refused by name where unsupported (#881)", {
  d <- .rg_data()
  for (m in c("eb", "structured", "agq", "re_cov_gibbs")) {
    expect_error(tulpa(y ~ 0 + (1 | g), d, family = "poisson", mode = m),
                 "needs at least one fixed-effect column", info = m)
  }
})

test_that("an aliased fixed-effect column is dropped with a warning (#881)", {
  set.seed(1)
  x <- rnorm(50)
  X <- cbind(`(Intercept)` = 1, x = x, x2 = 2 * x, z = rnorm(50))
  expect_warning(Xd <- .drop_aliased_fixed(X),
                 "rank deficient; dropping 1 aliased column\\(s\\): x2")
  expect_equal(colnames(Xd), c("(Intercept)", "x", "z"))
  expect_identical(.drop_aliased_fixed(X[, -3]), X[, -3])
})

test_that("a rank-deficient formula fits the identified columns (#881)", {
  skip_on_cran()
  set.seed(1)
  n <- 200L
  d <- data.frame(g = factor(sample(1:12, n, TRUE)), x = rnorm(n),
                  z = rnorm(n))
  d$y <- rpois(n, exp(0.5 + 0.3 * d$x))
  expect_warning(
    fit <- tulpa(y ~ x + z + poly(x, 2) + (1 | g), d, family = "poisson",
                 mode = "laplace", sigma_re = 0.5),
    "dropping 1 aliased column\\(s\\): poly\\(x, 2\\)1")
  expect_equal(names(coef(fit)), c("(Intercept)", "x", "z", "poly(x, 2)2"))
  expect_true(all(is.finite(sqrt(diag(vcov(fit))))))
})

test_that("VarCorr labels a sampled Sigma_mean 'sampled' (#881)", {
  base <- list(re_layout = list(list(group_var = "g", n_coefs = 1L,
                                     coef_labels = "(Intercept)")),
               Sigma_mean = matrix(0.25))
  est <- structure(base, class = "tulpa_fit")
  smp <- structure(c(base, list(Sigma_draws = list(matrix(0.2), matrix(0.3)))),
                   class = "tulpa_fit")
  expect_equal(VarCorr(est)$source, "estimated")
  expect_equal(VarCorr(smp)$source, "sampled")
  expect_equal(VarCorr(smp)$sd, 0.5)
})

test_that("a non-normal tulpa_prior is refused as beta_prior (#893)", {
  expect_error(.beta_prior_fields(prior_half_normal(1)),
               "`beta_prior` is a half_normal prior.*prior_normal\\(mean, sd\\)")
  expect_error(.normalize_beta_prior(prior_half_normal(1), 2L),
               "half_normal prior")
  expect_equal(.beta_prior_fields(prior_normal(0, 2))$sd, 2)
  expect_equal(.beta_prior_fields(list(sd = 3))$sd, 3)
  set.seed(1)
  d <- data.frame(x = rnorm(30))
  d$y <- 1 + 2 * d$x + rnorm(30)
  expect_error(tulpa(y ~ x, d, family = "gaussian", mode = "laplace", phi = 1,
                     beta_prior = prior_half_normal(1)),
               "half_normal prior")
})

test_that("re_prior keys the resolved backend does not read are refused (#893)", {
  expect_silent(.check_re_prior_backend(list(sigma_re_scale = 1), "hmc"))
  expect_silent(.check_re_prior_backend(list(prior_sigma = c(1, 0.05), eta = 2),
                                        "re_cov_nested"))
  expect_error(.check_re_prior_backend(list(sigma_re_scale = 0.01), "mala"),
               "not read by backend 'mala'.*`sigma_re_scale` \\(read by hmc, ess")
  expect_error(.check_re_prior_backend(list(prior_df = 3), "re_cov_nested"),
               "`prior_df` \\(read by re_cov_gibbs\\)")
  d <- .rg_data()
  expect_error(tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "mala",
                     re_prior = list(sigma_re_scale = 0.01)),
               "not read by backend 'mala'")
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
