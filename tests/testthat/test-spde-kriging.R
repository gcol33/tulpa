# SPDE field kriging through predict() (gcol33/tulpa C4, SPDE slice). A fitted
# Matern field is interpolated to arbitrary coordinates by re-projecting the
# posterior-mean mesh-node field through the spec's mesh. Re-projecting at the
# training coordinates must reproduce X beta + A_train %*% w_hat, which validates
# the projection; include_field = FALSE returns the fixed-effect (population)
# prediction.

test_that("predict() kriges the SPDE Matern field to new coordinates", {
  skip_on_cran()
  skip_if_not_installed("tulpaMesh")

  set.seed(101)
  n <- 200L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  spec <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.2, 0.5),
                       cutoff = 0.05, nu = 1)
  w <- as.numeric(rnorm(spec$n_mesh, 0, 0.6)); w <- w - mean(w)
  d$y <- as.numeric(2.0 + 0.5 * d$x + as.numeric(spec$A %*% w) + rnorm(n, 0, 0.3))

  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = d, family = "gaussian", spatial = spec, mode = "laplace"
  )))
  skip_if(is.null(fit$spatial_effects), "SPDE Laplace fit carries no field")

  X <- model.matrix(~ x, data = d)
  eta_fixed <- as.numeric(X %*% coef(fit)[colnames(X)])
  eta_full  <- eta_fixed + as.numeric(spec$A %*% fit$spatial_effects)

  # Re-projecting the field through the mesh at the training coordinates
  # reproduces X beta + A_train %*% w_hat -> the kriging projection is correct.
  p_new <- predict(fit, newdata = d, include_field = TRUE)
  expect_equal(p_new, eta_full, tolerance = 1e-6)

  # include_field = FALSE is the fixed-effect (population) prediction.
  p_pop <- predict(fit, newdata = d, include_field = FALSE)
  expect_equal(p_pop, eta_fixed, tolerance = 1e-8)
  expect_gt(max(abs(p_new - p_pop)), 1e-6)     # the field is non-trivial

  # A fresh (unobserved) location kriges to a finite value.
  nd <- data.frame(lon = 0.5, lat = 0.5, x = 0.1)
  expect_true(is.finite(predict(fit, newdata = nd, include_field = TRUE)))

  # se.fit together with the field is refused (field covariance not propagated).
  expect_error(predict(fit, newdata = nd, se.fit = TRUE),
               "not supported")
})

test_that("a custom (mesh-less) SPDE spec cannot krige to new coordinates", {
  skip_on_cran()
  skip_if_not_installed("fmesher")

  set.seed(7)
  n <- 120L
  coords <- cbind(runif(n), runif(n))
  mesh <- fmesher::fm_mesh_2d(loc = coords, max.edge = c(0.2, 0.5), cutoff = 0.06)
  fem  <- fmesher::fm_fem(mesh)
  A    <- as(fmesher::fm_basis(mesh, loc = coords), "CsparseMatrix")
  spec <- spatial_spde_custom(C = fem$c0, G = fem$g1, A = A, nu = 1)
  w <- as.numeric(rnorm(spec$n_mesh, 0, 0.5)); w <- w - mean(w)
  x <- rnorm(n)
  d <- data.frame(y = 1 + 0.4 * x + as.numeric(spec$A %*% w) + rnorm(n, 0, 0.3),
                  x = x, lon = coords[, 1], lat = coords[, 2])

  fit <- suppressWarnings(suppressMessages(tulpa(
    y ~ x, data = d, family = "gaussian", spatial = spec, mode = "laplace"
  )))
  skip_if(is.null(fit$spatial_effects), "SPDE Laplace fit carries no field")

  # No mesh / coordinate formula on a custom spec -> re-projection is refused
  # with a clear message rather than a wrong answer.
  expect_error(predict(fit, newdata = d, include_field = TRUE),
               "mesh and a coordinate formula")
})
