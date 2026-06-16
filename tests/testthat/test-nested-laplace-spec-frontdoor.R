# tulpa_nested_laplace(spec = ) front door: a single tulpa_spatial / tulpa_gp /
# tulpa_hsgp spec routes to the matching nested kernel through the shared
# spec -> prior converter (.spatial_spec_to_nl_prior). Areal was already
# supported; gp / nngp / hsgp are wired here (gcol33/tulpa#80). SPDE keeps its
# own (range, sigma) FEM integrator and declines with a redirect.

test_that("areal spec routes through tulpa_nested_laplace()", {
  skip_on_cran()
  set.seed(11)
  n <- 30L
  W <- matrix(0, n, n)
  for (i in 1:(n - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
  u <- as.numeric(scale(cumsum(rnorm(n))))
  y <- rbinom(n, 1, plogis(u))
  sp <- spatial_car(adjacency = W, level = "obs")
  sp$type <- "icar"
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, n), X = matrix(1, n, 1),
                              spec = sp, data = data.frame(i = seq_len(n)),
                              family = "binomial")
  expect_s3_class(fit, "tulpa_nested_laplace")
  expect_true(is.finite(fit$theta_median))
})

test_that("continuous gp/nngp spec routes through tulpa_nested_laplace()", {
  skip_on_cran()
  set.seed(12)
  ng <- 40L
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(coords[, 1] * 2 - coords[, 2]))
  y <- rbinom(ng, 1, plogis(fld))
  dg <- data.frame(lon = coords[, 1], lat = coords[, 2])
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, ng), X = matrix(1, ng, 1),
                              spec = spatial_gp(~ lon + lat, nn = 8),
                              data = dg, family = "binomial")
  expect_s3_class(fit, "tulpa_nested_laplace")
  expect_setequal(names(fit$theta_mean), c("sigma2", "phi_gp"))
  expect_true(all(is.finite(fit$theta_mean)))
})

test_that("continuous hsgp spec routes through tulpa_nested_laplace()", {
  skip_on_cran()
  set.seed(13)
  ng <- 40L
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(coords[, 1] * 2 - coords[, 2]))
  y <- rbinom(ng, 1, plogis(fld))
  dg <- data.frame(lon = coords[, 1], lat = coords[, 2])
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(1L, ng), X = matrix(1, ng, 1),
                              spec = spatial_hsgp(~ lon + lat, m = 6, c = 1.5),
                              data = dg, family = "binomial")
  expect_s3_class(fit, "tulpa_nested_laplace")
  expect_setequal(names(fit$theta_mean), c("sigma2", "lengthscale"))
  expect_true(all(is.finite(fit$theta_mean)))
})

test_that("SPDE spec declines with a fit_spde() redirect", {
  sp <- list(type = "spde")
  class(sp) <- c("tulpa_spatial", "list")
  expect_error(
    tulpa_nested_laplace(y = rbinom(5, 1, 0.5), n_trials = rep(1L, 5),
                         X = matrix(1, 5, 1), spec = sp,
                         data = data.frame(i = 1:5), family = "binomial"),
    "fit_spde"
  )
})

test_that("unknown spatial type errors clearly", {
  sp <- list(type = "wobble", adjacency = diag(3))
  class(sp) <- c("tulpa_spatial", "list")
  expect_error(.prior_from_spatial_spec(sp, data.frame(i = 1:3)),
               "Unknown spatial type")
})
