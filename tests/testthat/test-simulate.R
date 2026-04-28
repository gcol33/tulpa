# tulpa_simulate: generic simulator dispatching through family$simulate_fn

make_gaussian_family <- function() {
  tulpa_family(
    name = "gaussian",
    simulate_fn = function(eta, params, n_obs, ...) {
      rnorm(n_obs, eta[[1]], params$sigma_y)
    },
    extra_params = list(sigma_y = prior_half_normal(1))
  )
}

set.seed(7)
df <- data.frame(
  y = rep(0, 25),
  x = rnorm(25),
  g = factor(rep(1:5, each = 5))
)


test_that("tulpa_simulate with NULL theta is a single prior draw", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x, fam, df, theta = NULL, n_sims = 1, seed = 1)

  expect_s3_class(s, "tulpa_simulate")
  expect_length(s$y, 1)
  expect_length(s$y[[1]], 25)
  expect_length(s$theta, 1)
  expect_length(s$theta[[1]]$beta[[1]], 2L)
})


test_that("tulpa_simulate accepts a fixed theta list", {
  fam <- make_gaussian_family()
  theta <- list(
    beta = list(y = c(0.0, 1.0)),
    u = list(y = list()),
    extras = list(sigma_y = 0.01)
  )
  s <- tulpa_simulate(y ~ x, fam, df, theta = theta, n_sims = 4, seed = 11)

  expect_length(s$y, 4)
  # With sigma_y near zero, simulated y should be ~ X %*% beta
  X <- model.matrix(~ x, df)
  expected <- as.numeric(X %*% theta$beta$y)
  for (k in seq_along(s$y)) {
    expect_lt(max(abs(s$y[[k]] - expected)), 0.1)
  }
})


test_that("tulpa_simulate validates theta against parsed model", {
  fam <- make_gaussian_family()
  bad <- list(beta = list(y = c(1.0)),  # too short
              u = list(y = list()),
              extras = list(sigma_y = 1))
  expect_error(tulpa_simulate(y ~ x, fam, df, theta = bad), "expected 2")

  miss_extras <- list(beta = list(y = c(0, 1)), u = list(y = list()),
                      extras = list())
  expect_error(tulpa_simulate(y ~ x, fam, df, theta = miss_extras),
               "sigma_y")
})


test_that("tulpa_simulate is reproducible with seed", {
  fam <- make_gaussian_family()
  s1 <- tulpa_simulate(y ~ x, fam, df, n_sims = 3, seed = 99)
  s2 <- tulpa_simulate(y ~ x, fam, df, n_sims = 3, seed = 99)
  expect_identical(s1$y, s2$y)
})


test_that("tulpa_simulate works with random effects", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x + (1 | g), fam, df, n_sims = 2, seed = 3)
  expect_length(s$y[[1]], 25)
  expect_equal(nrow(s$theta[[1]]$u[[1]][[1]]), 5)
})


test_that("tulpa_simulate dispatches via family$simulate_fn (no model branch)", {
  # Custom family: response is a constant tag
  custom <- tulpa_family(
    name = "tag",
    simulate_fn = function(eta, params, n_obs, ...) rep("tag", n_obs)
  )
  s <- tulpa_simulate(y ~ x, custom, df, theta = list(
    beta = list(y = c(0, 0)), u = list(y = list()), extras = list()
  ), n_sims = 2)
  expect_true(all(s$y[[1]] == "tag"))
})


test_that("print method runs", {
  fam <- make_gaussian_family()
  s <- tulpa_simulate(y ~ x, fam, df, n_sims = 1, seed = 0)
  expect_output(print(s), "tulpa simulated datasets")
})
