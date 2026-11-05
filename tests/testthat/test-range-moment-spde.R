.range_spde_fixture <- function() {
  C <- Matrix::Diagonal(2)
  G <- Matrix::Matrix(matrix(c(1, -1, -1, 1), 2), sparse = TRUE)
  spatial_spde_custom(C, G, Matrix::Diagonal(2), nu = 1,
    prior_range = c(1, .5), prior_sigma = c(1, .1))
}

# In the mass-normalized constant and contrast coordinates, the SPDE
# precision eigenvalues are tau^2 kappa^4 and tau^2 (kappa^2 + 2)^2.
.range_spde_loglik <- function(r, sigma = .8) {
  k2 <- 8 / r^2
  tau2 <- 1 / (4 * pi * k2 * sigma^2)
  v <- c(1 + 2 * 2.5^2 + 1 / (tau2 * k2^2),
         1 + 1 / (tau2 * (k2 + 2)^2))
  y <- c(.3 + .7, .3 - .7) / sqrt(2)
  -log(2 * pi) - .5 * sum(log(v) + y^2 / v)
}

test_that("SPDE matrix storage preserves the model's constant null mode", {
  set.seed(63)
  seed <- .Random.seed
  sp <- .range_spde_fixture()
  G <- as(sp$G, "generalMatrix")
  full <- spatial_spde_custom(sp$C, G, sp$A, nu = 1,
    prior_range = c(1, .5), prior_sigma = c(1, .1))
  expect_equal(sp$G1_x, full$G1_x)
  expect_equal(sp$G1_i, full$G1_i)
  expect_equal(sp$G1_p, full$G1_p)
  expect_equal(length(sp$G1_x), 4L)
  expect_equal(tulpa:::.hyper_spde_boundary(list(family = "gaussian", y = c(.3, .7)), sp), 1)
  expect_identical(.Random.seed, seed)
})

test_that("an existing finite-mesh range mean converges to exact Gaussian integration", {
  skip_on_cran()
  sp <- .range_spde_fixture()
  prior <- tulpa:::.spatial_spec_to_nl_prior(sp)
  prior$prior_range <- sp$prior_range
  prior$sigma_grid <- .8
  lambda <- -log(.5)
  integrand <- function(h, power) vapply(h, function(x) {
    if (x == 0 || !is.finite(x)) return(0)
    exp(.range_spde_loglik(1 / x) - lambda * x) * lambda / x^power
  }, numeric(1))
  z <- integrate(integrand, 0, Inf, power = 0, rel.tol = 1e-9)$value
  exact_mean <- integrate(integrand, 0, Inf, power = 1, rel.tol = 1e-9)$value / z
  fit_at <- function(top) {
    p <- prior
    p$range_grid <- exp(seq(log(.02), log(top), by = .025))
    suppressWarnings(tulpa_nested_laplace(c(.3, .7), c(1L, 1L), matrix(1, 2, 1),
      prior = list(p), family = "gaussian", phi = 1,
      control = list(progress = FALSE, auto_recenter = FALSE, prune = FALSE,
        axis_refine = "none", diagnose_k = FALSE, diagnose_skew = FALSE)))
  }
  fits <- lapply(c(1500, 3000), fit_at)
  for (fit in fits) {
    expect_true(fit$moment_existence$b1.range$mean_exists)
    expect_false(fit$moment_existence$b1.range$sd_exists)
    expect_equal(fit$theta_mean[["b1.range"]], exact_mean, tolerance = .002)
    expect_identical(fit$theta_sd[["b1.range"]], Inf)
    tg <- tulpa_theta_matrix(fit)
    lm <- vapply(tg[, "b1.range"], .range_spde_loglik, numeric(1))
    expect_equal(fit$log_marginal - fit$log_hyperprior, lm, tolerance = 1e-5)
  }
  expect_equal(fits[[1]]$theta_mean, fits[[2]]$theta_mean, tolerance = .002)
})

test_that("Gaussian SPDE level decay honours a flat intercept and observed mesh components", {
  sp <- .range_spde_fixture()
  object <- list(family = "gaussian", y = c(.3, .7), model_matrix = matrix(1, 2, 1),
    beta_prior = list(sd = Inf))
  expect_equal(tulpa:::.hyper_spde_boundary(object, sp), 0)
  object$beta_prior$sd <- 2.5
  expect_equal(tulpa:::.hyper_spde_boundary(object, sp), 1)
  arms <- list(a = list(family = "gaussian", y = c(.3, .7), X = matrix(1, 2, 1)),
               b = list(family = "gaussian", y = c(.1, .5), X = matrix(1, 2, 1)))
  joint <- sp
  for (key in c("A_x", "A_i", "A_p")) joint[[key]] <- rep(list(sp[[key]]), 2L)
  expect_equal(tulpa:::.hyper_spde_boundary(list(responses = arms), joint), 1)
  pinned <- spatial_spde_custom(Matrix::Diagonal(2, c(1, 0)), sp$G, sp$A,
    prior_range = c(1, .5), prior_sigma = c(1, .1))
  expect_equal(tulpa:::.hyper_spde_boundary(object, pinned), 0)
})

test_that("direct SPDE integration retains range log moments on the fit", {
  skip_on_cran()
  fit <- fit_spde(c(.3, .7), matrix(1, 2, 1), .range_spde_fixture(),
    family = "gaussian", phi = 1,
    control = list(method = "grid", n_grid = 9L, progress = FALSE, diagnose_k = FALSE))
  expect_identical(fit$theta_log_mean, fit$nested$theta_log_mean)
  expect_identical(fit$theta_log_sd, fit$nested$theta_log_sd)
  expect_true(is.finite(fit$theta_log_mean[["range"]]))
  expect_identical(fit$nested$range_sd, Inf)
})

test_that("SPDE boundary components use sparse observed loadings", {
  n <- 200L
  sp <- spatial_spde_custom(Matrix::Diagonal(n), Matrix::sparseMatrix(i = integer(),
    j = integer(), dims = c(n, n)), Matrix::Diagonal(n),
    prior_range = c(1, .5), prior_sigma = c(1, .1))
  expect_equal(tulpa:::.hyper_spde_boundary(list(family = "gaussian", y = rep(0, n)), sp), n)
  G <- Matrix::bandSparse(n, k = c(-1, 0, 1),
    diagonals = list(rep(-1, n - 1), c(1, rep(2, n - 2), 1), rep(-1, n - 1)))
  chain <- spatial_spde_custom(sp$C, G, sp$A, prior_range = c(1, .5), prior_sigma = c(1, .1))
  expect_equal(tulpa:::.hyper_spde_boundary(list(family = "gaussian", y = rep(0, n)), chain), 1)
})

test_that("HSGP multiscale and varying-coefficient grids retain divergent range rows", {
  skip_on_cran()
  set.seed(58)
  d <- data.frame(lon = runif(16), lat = runif(16), x = rnorm(16), y = rnorm(16))
  sp <- validate_gp(spatial_multiscale(~lon + lat, approx = "hsgp", m = 3L,
    c_boundary = 2, range_local = c(.1, .5), range_regional = c(.5, 2),
    scale_coords = FALSE), d)
  blocks <- tulpa:::.multiscale_nl_blocks(sp)
  for (b in 1:2) {
    blocks[[b]]$sigma2_grid <- .4
    blocks[[b]]$lengthscale_grid <- exp(seq(log(.05), log(100), length.out = 7))
  }
  fit <- suppressWarnings(tulpa_nested_laplace(d$y, rep(1L, 16), cbind(1, d$x),
    prior = blocks, family = "gaussian", phi = .16,
    control = list(integration = "grid", progress = FALSE, auto_recenter = FALSE,
      axis_refine = "none", diagnose_k = FALSE, diagnose_skew = FALSE)))
  expect_true(all(fit$theta_mean[c("b1.lengthscale", "b2.lengthscale")] == Inf))
  expect_true(all(spatial_range(fit)[c("range[local]", "range[regional]"), "sd"] == Inf))
  basis <- tulpa:::cpp_hsgp_basis_2d(as.matrix(d[, c("lon", "lat")]), 3L, 2)
  svc <- list(type = "hsgp", phi = list(basis$phi_basis), eigenvalues = basis$lambda_eig,
    n_obs_per_arm = 16L, m_total = 9L, sigma2_grid = .4,
    lengthscale_grid = c(.1, .4, 2, 10), prior_range = c(.5, .5), svc_column = 2L)
  varying <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = d$y, X = cbind(1, d$x), family = "gaussian", phi = .16)),
    prior = list(svc), control = list(progress = FALSE, auto_recenter = FALSE,
      axis_refine = "none", diagnose_k = FALSE, diagnose_skew = FALSE)))
  expect_identical(varying$theta_mean[["b1.lengthscale"]], Inf)
  expect_identical(varying$theta_sd[["b1.lengthscale"]], Inf)
})
