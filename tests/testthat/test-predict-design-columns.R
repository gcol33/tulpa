# A fit made from a design matrix (fit_spde(), no formula) predicts at
# `newdata` by reading each design column under its name.

skip_on_cran()

.design_fit_fixture <- local({
  fx <- NULL
  function() {
    if (is.null(fx)) {
      set.seed(2)
      n <- 150
      d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
      sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.12, 0.3))
      d$y <- rpois(n, exp(1 + 0.5 * d$x + sin(3 * d$lon)))
      X <- model.matrix(~ x, d)
      fx <<- list(d = d, sp = sp,
                  fit = fit_spde(d$y, X, sp, family = "poisson",
                                 range = 0.4, sigma = 0.5))
    }
    fx
  }
})

test_that("predict(newdata) on a fit_spde() fit rebuilds the design by column", {
  skip_on_cran()
  fx <- .design_fit_fixture()
  fit <- fx$fit
  nd <- data.frame(lon = c(0.2, 0.5, 0.8), lat = c(0.3, 0.6, 0.4),
                   x = c(0, 1, -0.5))
  A_new <- tulpaMesh::fem_matrices(fx$sp$mesh,
                                   obs_coords = cbind(nd$lon, nd$lat))$A
  ref <- as.numeric(cbind(1, nd$x) %*% coef(fit)) +
    as.numeric(A_new %*% fit$spatial_effects)
  expect_equal(predict(fit, nd), ref, tolerance = 1e-10)
  # At the training rows it reproduces the in-sample prediction.
  expect_equal(predict(fit, fx$d), predict(fit), tolerance = 1e-10)
  expect_error(predict(fit, nd[, c("lon", "lat")]), "missing: x")
})

test_that("predict(newdata) refuses a design with unnamed columns", {
  skip_on_cran()
  fx <- .design_fit_fixture()
  fit <- fx$fit
  colnames(fit$model_matrix) <- NULL
  expect_error(predict(fit, fx$d[1:2, ]), "unnamed columns")
})

test_that("predict(newdata) refuses a fit whose offset was a vector", {
  skip_on_cran()
  fx <- .design_fit_fixture()
  fit <- fx$fit
  fit$offset <- rep(0.1, nrow(fx$d))
  expect_error(predict(fit, fx$d[1:2, ]), "offset")
})
