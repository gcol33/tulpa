# cpo() / dic() on a tulpa_fit, and loo's waic() / loo() generics dispatching
# to it. Every door reads the pointwise log-likelihood compare_models() reads,
# so the numbers agree with the criteria layer run on that matrix by hand.

criteria_fit_data <- function(n = 120L, seed = 11L) {
  set.seed(seed)
  d <- data.frame(x = rnorm(n), time = rep(1:6, length.out = n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x))
  d$yb <- rbinom(n, 1L, stats::plogis(-0.2 + 0.7 * d$x))
  d
}

test_that("the default doors name `object` for an object with no method", {
  expect_error(cpo(list(a = 1)), "cpo\\(\\): `object` must be")
  expect_error(dic(list(a = 1)), "dic\\(\\): `object` must be")
  expect_error(cpo("a"), "class character")
})

test_that("a fit with no pointwise log-likelihood is refused, naming its class", {
  bare <- structure(list(backend = "laplace", family = "poisson"),
                    class = c("my_fit", "tulpa_fit"))
  for (door in list(cpo, dic)) {
    expect_error(door(bare),
                 "class my_fit/tulpa_fit \\(backend 'laplace'\\).*no response")
  }
  custom <- structure(list(y = 1:3, family = list(name = "custom")),
                      class = "tulpa_fit")
  expect_error(cpo(custom), "not a single built-in family name")
  skip_if_not_installed("loo")
  expect_error(loo::waic(bare), "waic\\(\\): the fit of class my_fit/tulpa_fit")
  expect_error(loo::loo(bare), "loo\\(\\): the fit of class my_fit/tulpa_fit")
})

test_that("a stored log_lik is read directly; DIC then reports dbar only", {
  set.seed(2)
  ll <- matrix(stats::dnorm(rnorm(60 * 8), log = TRUE), 60, 8)
  fit <- structure(list(draws = list(log_lik = ll), draws_kind = "iid"),
                   class = "tulpa_fit")
  expect_equal(cpo(fit)$lpml, cpo(ll)$lpml)
  d <- dic(fit)
  expect_equal(d$dbar, -2 * mean(rowSums(ll)))
  expect_true(is.na(d$dic))
})

test_that("iid fit: cpo / dic shapes, waic / loo equal loo on the matrix", {
  skip_on_cran()
  d <- criteria_fit_data()
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  ll <- .tulpa_pointwise_loglik(fit)
  expect_equal(dim(ll), c(nrow(ll), nrow(d)))
  expect_identical(.tulpa_pointwise_loglik(fit), ll)

  cp <- cpo(fit)
  expect_s3_class(cp, "tulpa_criteria")
  expect_equal(nrow(cp$pointwise), nrow(d))
  expect_equal(cp$pointwise$cpo, exp(cp$pointwise$elpd_loo))
  expect_equal(cp$lpml, cpo(ll)$lpml)

  dc <- dic(fit)
  expect_s3_class(dc, "tulpa_criteria")
  expect_equal(dc$dic, dc$dbar + dc$p_dic)
  # Two fixed effects and an exact-Gaussian-ish posterior: p_DIC near 2.
  expect_lt(abs(dc$p_dic - 2), 0.5)

  cm <- compare_models(a = fit, criterion = "waic")
  expect_equal(cm$elpd, tulpa_criteria(ll, criteria = "waic")$elpd_waic)

  skip_if_not_installed("loo")
  w <- suppressWarnings(loo::waic(fit))
  expect_s3_class(w, "waic")
  expect_equal(w$estimates, suppressWarnings(loo::waic(ll))$estimates)
  expect_equal(w$estimates["elpd_waic", "Estimate"], cm$elpd)

  l <- suppressWarnings(loo::loo(fit))
  expect_s3_class(l, "psis_loo")
  expect_equal(l$estimates, suppressWarnings(loo::loo(ll, r_eff = 1))$estimates)
})

test_that("chain fit: loo() reads relative efficiencies over the chains", {
  skip_on_cran()
  skip_if_not_installed("loo")
  d <- criteria_fit_data(seed = 12L)
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
               control = list(n_iter = 300L, warmup = 150L, n_chains = 2L,
                              seed = 1L))
  expect_true(.tulpa_is_chain(fit))
  ll <- .tulpa_pointwise_loglik(fit)
  cid <- .tulpa_chain_id(fit, nrow(ll))
  expect_equal(sort(unique(cid)), 1:2)

  r_eff <- loo::relative_eff(exp(ll), chain_id = cid)
  l <- suppressWarnings(loo::loo(fit))
  expect_equal(l$diagnostics$r_eff, r_eff)
  expect_equal(l$estimates,
               suppressWarnings(loo::loo(ll, r_eff = r_eff))$estimates)
  expect_equal(suppressWarnings(loo::waic(fit))$estimates,
               suppressWarnings(loo::waic(ll))$estimates)
  expect_equal(compare_models(a = fit, criterion = "loo")$elpd,
               tulpa_criteria(ll, criteria = "loo")$elpd_loo)

  dc <- dic(fit)
  expect_true(is.finite(dc$dic) && dc$p_dic > 0)
  expect_equal(nrow(cpo(fit)$pointwise), nrow(d))
})

test_that("nested-Laplace fit: the grid-mixture read is repeatable", {
  skip_on_cran()
  d <- criteria_fit_data(seed = 13L)
  fit <- tulpa(yb ~ x, data = d, family = "binomial",
               temporal = temporal_rw1("time"), mode = "nested_laplace")
  seed_before <- get0(".Random.seed", envir = globalenv())
  c1 <- cpo(fit)
  expect_identical(get0(".Random.seed", envir = globalenv()), seed_before)
  expect_equal(cpo(fit)$lpml, c1$lpml)
  expect_equal(nrow(c1$pointwise), nrow(d))
  dc <- dic(fit)
  expect_true(is.finite(dc$dic) && dc$p_dic > 0)
  expect_equal(compare_models(a = fit, criterion = "waic")$elpd,
               tulpa_criteria(.tulpa_pointwise_loglik(fit),
                              criteria = "waic")$elpd_waic)
})

test_that("a fit that stores no response is refused on every door", {
  skip_on_cran()
  set.seed(14)
  d <- data.frame(x = rnorm(90))
  d$ycat <- factor(sample(c("a", "b", "c"), 90, TRUE))
  fit <- tulpa(ycat ~ x, data = d, family = "multinomial")
  fit$y <- NULL
  expect_error(cpo(fit), "class tulpa_multinomial/tulpa_categorical/tulpa_fit")
  expect_error(dic(fit), "no pointwise log-likelihood")
  expect_true(is.na(compare_models(a = fit, criterion = "waic")$elpd))
})
