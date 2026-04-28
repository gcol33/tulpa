# Prior predictive simulation: dispatch via tulpa_family$simulate_fn

make_gaussian_family <- function() {
  tulpa_family(
    name = "gaussian",
    simulate_fn = function(eta, params, n_obs, ...) {
      rnorm(n_obs, eta[[1]], params$sigma_y)
    },
    extra_params = list(sigma_y = prior_half_normal(1))
  )
}

make_poisson_family <- function() {
  tulpa_family(
    name = "poisson",
    simulate_fn = function(eta, params, n_obs, ...) {
      rpois(n_obs, exp(pmin(eta[[1]], 20)))
    }
  )
}

set.seed(42)
df <- data.frame(
  y = rep(0, 30),
  x = rnorm(30),
  g = factor(rep(1:5, each = 6))
)


test_that("tulpa_family validates required slots", {
  expect_error(tulpa_family(name = "x", simulate_fn = NULL),
               "must be a function")
  expect_error(tulpa_family(name = "x", simulate_fn = function() NULL,
                            process_names = character(0)),
               "non-empty")
  fam <- make_gaussian_family()
  expect_s3_class(fam, "tulpa_family")
  expect_named(fam$extra_params, "sigma_y")
})


test_that("prior_predict returns a tulpa_prior_predict with correct shapes", {
  fam <- make_gaussian_family()
  pp <- prior_predict(y ~ x, fam, df, n_draws = 10, seed = 1)

  expect_s3_class(pp, "tulpa_prior_predict")
  expect_equal(pp$n_draws, 10)
  expect_equal(pp$n_obs, 30)
  expect_length(pp$y, 10)
  expect_length(pp$theta, 10)
  expect_length(pp$linpred, 10)
  expect_length(pp$y[[1]], 30)
  expect_length(pp$linpred[[1]][[1]], 30)
})


test_that("prior_predict is reproducible with seed", {
  fam <- make_gaussian_family()
  pp1 <- prior_predict(y ~ x, fam, df, n_draws = 5, seed = 123)
  pp2 <- prior_predict(y ~ x, fam, df, n_draws = 5, seed = 123)
  expect_identical(pp1$y, pp2$y)
  expect_identical(pp1$theta, pp2$theta)
})


test_that("prior_predict draws differ across seeds", {
  fam <- make_gaussian_family()
  pp1 <- prior_predict(y ~ x, fam, df, n_draws = 5, seed = 1)
  pp2 <- prior_predict(y ~ x, fam, df, n_draws = 5, seed = 2)
  expect_false(identical(pp1$y, pp2$y))
})


test_that("prior_predict handles random effect terms", {
  fam <- make_gaussian_family()
  pp <- prior_predict(y ~ x + (1 | g), fam, df, n_draws = 5, seed = 7)

  expect_length(pp$theta[[1]]$u[[1]], 1L)  # one RE term -> one u block per process
  re_block <- pp$theta[[1]]$u[[1]][[1]]
  expect_equal(nrow(re_block), 5)  # 5 levels of g
  expect_equal(ncol(re_block), 1)  # intercept-only
})


test_that("prior_predict samples extra_params from family priors", {
  fam <- make_gaussian_family()
  pp <- prior_predict(y ~ x, fam, df, n_draws = 50, seed = 4)

  sig <- vapply(pp$theta, function(t) t$extras$sigma_y, numeric(1))
  expect_true(all(sig > 0))               # half-normal positivity
  expect_true(length(unique(sig)) > 1L)   # they vary across draws
})


test_that("prior_predict works without extra_params", {
  fam <- make_poisson_family()
  pp <- prior_predict(y ~ x, fam, df, n_draws = 5, seed = 9)
  expect_length(pp$y, 5)
  expect_true(all(vapply(pp$y, is.numeric, logical(1))))
})


test_that("prior_predict rejects bad inputs", {
  fam <- make_gaussian_family()
  expect_error(prior_predict(y ~ x, "not a family", df), "tulpa_family")
  expect_error(prior_predict(y ~ x, fam, df, priors = list(),
                              n_draws = 1),
               "tulpa_priors")
  expect_error(prior_predict(y ~ x, fam, df, n_draws = 0), "positive integer")
})


test_that("prior_predict respects custom priors", {
  fam <- make_gaussian_family()
  tight <- tulpa_priors(beta = prior_normal(0, 0.1))
  loose <- tulpa_priors(beta = prior_normal(0, 5))

  pp_t <- prior_predict(y ~ x, fam, df, priors = tight,
                        n_draws = 100, seed = 42)
  pp_l <- prior_predict(y ~ x, fam, df, priors = loose,
                        n_draws = 100, seed = 42)

  beta_t <- do.call(rbind, lapply(pp_t$theta, function(t) t$beta[[1]]))
  beta_l <- do.call(rbind, lapply(pp_l$theta, function(t) t$beta[[1]]))

  expect_lt(sd(beta_t[, 2]), sd(beta_l[, 2]))
})


test_that("print method runs", {
  fam <- make_gaussian_family()
  pp <- prior_predict(y ~ x, fam, df, n_draws = 3, seed = 0)
  expect_output(print(pp), "tulpa prior predictive draws")
})
