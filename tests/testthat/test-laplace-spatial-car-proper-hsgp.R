# Fixed-hyperparameter (single-point) Laplace for proper-CAR and HSGP fields
# (gcol33/tulpa#79). The single-point kernels reuse the SAME block factories +
# dense spec solver as the nested integrators, so the mode (and, for CAR_proper,
# the log-marginal) equal the nested kernel evaluated at one grid cell; the HSGP
# log-marginal is checked against a brute-force Laplace. Both are wired into
# dispatch_laplace_spatial / tulpa_laplace with a Schur-complement marginal SE.

ring_adj <- function(n) {
  W <- matrix(0, n, n)
  for (i in 1:(n - 1)) { W[i, i + 1] <- 1; W[i + 1, i] <- 1 }
  W[1, n] <- 1; W[n, 1] <- 1
  W
}

test_that("CAR_proper single-point == nested kernel at one (tau, rho) cell", {
  skip_on_cran()
  set.seed(7)
  n <- 24L
  W <- ring_adj(n)
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  u <- as.numeric(scale(cumsum(rnorm(n))))
  y <- rbinom(n, 1, plogis(u))
  X <- matrix(1, n, 1)
  tau <- 1.7; rho <- 0.6
  args <- list(y = as.numeric(y), n = rep(1L, n), X = X, re_idx = rep(0, n),
               n_re_groups = 0L, sigma_re = 1.0, spatial_idx = 1:n,
               n_spatial_units = n, adj_row_ptr = csr$row_ptr,
               adj_col_idx = csr$col_idx, n_neighbors = csr$n_neighbors,
               family = "binomial")
  sp <- do.call(tulpa:::cpp_laplace_fit_car_proper,
                c(args, list(tau_spatial = tau, rho = rho)))
  ns <- do.call(tulpa:::cpp_nested_laplace_car_proper,
                c(args, list(tau_grid = tau, rho_grid = rho)))
  expect_equal(as.numeric(sp$mode), as.numeric(ns$modes[1, ]), tolerance = 1e-8)
  expect_equal(sp$log_marginal, as.numeric(ns$log_marginal)[1], tolerance = 1e-8)
})

test_that("HSGP single-point matches nested mode and a brute-force Laplace", {
  skip_on_cran()
  set.seed(8)
  ng <- 50L
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(sin(3 * coords[, 1]) + coords[, 2]))
  y <- rbinom(ng, 1, plogis(fld))
  X <- matrix(1, ng, 1)
  basis <- tulpa:::cpp_hsgp_basis_2d(coords, 6L, 1.5)
  sig2 <- 0.8; ell <- 0.5
  args <- list(y = as.numeric(y), n = rep(1L, ng), X = X, re_idx = rep(0, ng),
               n_re_groups = 0L, sigma_re = 1.0, phi_basis = basis$phi_basis,
               lambda_eig = basis$lambda_eig, family = "binomial")
  sp <- do.call(tulpa:::cpp_laplace_fit_hsgp,
                c(args, list(sigma2 = sig2, lengthscale = ell)))
  ns <- do.call(tulpa:::cpp_nested_laplace_hsgp,
                c(args, list(sigma2_grid = sig2, lengthscale_grid = ell)))
  expect_equal(as.numeric(sp$mode), as.numeric(ns$modes[1, ]), tolerance = 1e-7)

  # Brute-force Laplace log p(y | theta) at the fitted mode.
  S <- sig2 * sqrt(2 * pi) * ell * exp(-0.5 * ell^2 * basis$lambda_eig)
  D <- cbind(X, sweep(basis$phi_basis, 2, sqrt(S), `*`))
  eta <- as.numeric(D %*% sp$mode); pr <- plogis(eta)
  fstar <- sum(dbinom(y, 1, pr, log = TRUE)) +
    dnorm(sp$mode[1], 0, 100, log = TRUE) +
    sum(dnorm(sp$mode[-1], 0, 1, log = TRUE))
  H <- t(D) %*% ((pr * (1 - pr)) * D)
  H[1, 1] <- H[1, 1] + 1 / 100^2
  diag(H)[-1] <- diag(H)[-1] + 1
  d <- ncol(D)
  lm_bf <- fstar + 0.5 * d * log(2 * pi) -
    0.5 * as.numeric(determinant(H, logarithm = TRUE)$modulus)
  expect_equal(sp$log_marginal, lm_bf, tolerance = 1e-6)
})

test_that("tulpa_laplace routes a proper-CAR spec with a PD marginal H_beta", {
  skip_on_cran()
  set.seed(3)
  n <- 28L
  W <- ring_adj(n)
  u <- as.numeric(scale(cumsum(rnorm(n))))
  y <- rbinom(n, 1, plogis(0.3 + u))
  X <- cbind(1, rnorm(n))
  sp <- spatial_car(adjacency = W, level = "obs", proper = TRUE)
  sp$spatial_idx <- 1:n
  fit <- tulpa_laplace(y = y, n_trials = rep(1L, n), X = X, spatial = sp,
                       family = "binomial", return_hessian = TRUE)
  expect_true(fit$converged)
  expect_false(is.null(fit$H_beta))
  expect_true(all(eigen(fit$H_beta, only.values = TRUE)$values > 0))
})

test_that("tulpa_laplace routes an HSGP spec with a PD marginal H_beta", {
  skip_on_cran()
  set.seed(5)
  ng <- 60L
  coords <- cbind(runif(ng), runif(ng))
  fld <- as.numeric(scale(cos(4 * coords[, 1]) - coords[, 2]))
  y <- rbinom(ng, 1, plogis(0.2 + fld))
  X <- cbind(1, rnorm(ng))
  dh <- data.frame(lon = coords[, 1], lat = coords[, 2])
  sp <- validate_hsgp(spatial_hsgp(~ lon + lat, m = 6, c = 1.5), dh)
  fit <- tulpa_laplace(y = y, n_trials = rep(1L, ng), X = X, spatial = sp,
                       family = "binomial", return_hessian = TRUE)
  expect_true(fit$converged)
  expect_false(is.null(fit$H_beta))
  expect_true(all(eigen(fit$H_beta, only.values = TRUE)$values > 0))
})
