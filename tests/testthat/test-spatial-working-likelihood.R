test_that("fixed working predictors give the same dense Gaussian field coordinates", {
  set.seed(196)
  S <- 4L; N <- 16L
  d <- data.frame(lon = rep(c(0, .3, .7, 1), each = 4), lat = rep(c(.2, .8, .1, .6), each = 4))
  X <- cbind(1, seq(-1, 1, length.out = N)); y <- rnorm(N)
  gp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(spatial_gp(~ lon + lat, nn = 3L, cov = 'exponential'), d))
  gp$parameterization <- 'auto'; gp$gp_parameterization <- 1L
  gp$conditional_coordinates <- TRUE
  gp$range_prior <- list(type = 'lognormal', meanlog = 0, sdlog = 1)
  ordinary <- gp
  ordinary$conditional_coordinates <- FALSE
  full <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = gp)
  scaled <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = ordinary, point = full$x)
  expect_true(full$conditional)
  expect_false(scaled$conditional)
  expect_equal(full$raw_lp, scaled$raw_lp, tolerance = 1e-9)
  expect_equal(full$stored, scaled$stored, tolerance = 1e-9)
  expect_equal(full$raw_gradient, scaled$raw_gradient, tolerance = 1e-9)
  expect_true(full$verified && scaled$verified)
  bym <- list(type = 'bym2', parameterization = 'auto', spatial_idx = rep(1:S, each = 4),
    n_spatial_units = S, adj_row_ptr = c(0L, 1L, 3L, 5L, 6L),
    adj_col_idx = c(1L, 0L, 2L, 1L, 3L, 2L), n_neighbors = c(1L, 2L, 2L, 1L))
  for (sp in list(gp, bym)) for (restricted in c(FALSE, TRUE)) {
    if (restricted) sp$restriction_basis <- qr.Q(qr(X))
    at <- function(anchor) tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp,
      shift = .2, working_eta = anchor)
    base <- at(NULL)
    for (anchor in c(-2, 1, 3)) {
      r <- at(anchor)
      expect_equal(r$x, base$x, tolerance = 2e-6)
      expect_equal(r$jacobian, base$jacobian, tolerance = 2e-6)
      expect_equal(r$lp, base$lp, tolerance = 2e-6)
      expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
      expect_true(r$verified)
    }
    expect_error(at(c(1, 2)), 'one eta per process')
    expect_error(at(NA_real_), 'must be finite')
    expect_error(at(Inf), 'must be finite')
  }
  for (restricted in c(FALSE, TRUE)) {
    point <- c(.1, -.2, -.3, -.4, rep(0, S))
    sp <- gp
    if (restricted) sp$restriction_basis <- qr.Q(qr(X))
    r <- tulpa:::cpp_test_field_coordinates(y, X, spatial_spec = sp, point = point,
      native_point = FALSE, working_eta = 1)
    K <- exp(point[3]) * exp(-as.matrix(dist(gp$coords)) / exp(point[4])) + diag(1e-8, S)
    P <- if (restricted) diag(N) - tcrossprod(sp$restriction_basis) else diag(N)
    G <- P %*% diag(S)[gp$obs_to_loc + 1L, ]
    V <- diag(N) + G %*% K %*% t(G)
    mean <- drop(K %*% crossprod(G, solve(V, y - X %*% point[1:2])))
    expect_equal(r$stored[5:8], unname(mean), tolerance = 2e-6)
  }
})
