# A joint model with no latent block (gcol33/tulpa#939): `prior = list()` is a
# list of zero blocks, so each arm's predictor is its fixed effects and the
# outer grid is the per-arm dispersion axes alone.

.no_latent_arms <- function() {
  set.seed(11)
  n <- 100
  x <- rnorm(n)
  X <- cbind(`(Intercept)` = 1, x = x)
  list(
    num = list(y = rpois(n, exp(0.8 + 0.3 * x)), n_trials = rep(1L, n),
               X = X, family = "poisson"),
    den = list(y = rgamma(n, 4, 4 / exp(0.4 - 0.2 * x)), n_trials = rep(1L, n),
               X = X, family = "gamma", phi = 4))
}

test_that("an empty prior is a list of zero blocks", {
  expect_true(.is_multi_block_prior(list()))
  expect_false(.is_multi_block_prior(list(type = "icar")))
})

test_that("an empty prior fits the arms on the dispersion axes alone", {
  skip_on_cran()
  resp <- .no_latent_arms()
  pg <- list(den = c(1, 2, 3, 4, 6, 9))
  ctl <- list(axis_refine = "none", prune = FALSE)
  fit <- tulpa_nested_laplace_joint(responses = resp, prior = list(),
                                    phi_grid = pg, control = ctl)
  expect_identical(colnames(fit$theta_grid), "phi_den")
  expect_equal(as.numeric(fit$theta_grid[, 1]), pg$den)
  expect_true(all(is.finite(fit$log_marginal)))

  # The documented silent block -- one iid unit held at sigma = 0 -- puts
  # nothing on the predictor, so it is the same model and must give the same
  # surface and the same fixed-effect posterior.
  silent <- tulpa_nested_laplace_joint(
    responses = resp,
    prior = list(list(type = "iid", n_units = 1L, sigma_grid = 0,
                      obs_idx = list(rep(1L, 100), rep(1L, 100)))),
    phi_grid = pg, control = ctl)
  expect_equal(fit$log_marginal, silent$log_marginal, tolerance = 1e-8)
  expect_equal(fit$weights, silent$weights, tolerance = 1e-8)
  expect_equal(coef(fit), coef(silent), tolerance = 1e-8)
  expect_equal(vcov(fit), vcov(silent), tolerance = 1e-8)
})

test_that("an empty prior with no dispersion axis is one cell at the GLM mode", {
  skip_on_cran()
  resp <- .no_latent_arms()["num"]
  fit <- tulpa_nested_laplace_joint(responses = resp, prior = list())
  expect_equal(nrow(fit$theta_grid), 1L)
  expect_equal(ncol(fit$theta_grid), 0L)
  ref <- stats::glm(resp$num$y ~ resp$num$X[, "x"], family = poisson)
  expect_equal(unname(coef(fit)), unname(coef(ref)), tolerance = 1e-4)
})

test_that("the registry door names the doors that take a model with no block", {
  expect_error(
    tulpa_nested_laplace(y = rpois(10, 2), n_trials = rep(1L, 10),
                         X = matrix(1, 10, 1), family = "poisson",
                         prior = list()),
    "no latent block")
})
