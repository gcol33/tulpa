# test-rational-spde.R
# Tests for the rational SPDE (fractional Matern smoothness) path (gcol33/tulpa#71).

test_that("rational_spde_coefficients returns integer flag for integer nu", {
  rat <- rational_spde_coefficients(1.0)
  expect_true(rat$is_integer)
  expect_equal(length(rat$poles), 0)

  rat2 <- rational_spde_coefficients(2.0)
  expect_true(rat2$is_integer)
})

test_that("rational_spde_coefficients returns BRASIL roots for fractional nu", {
  # gcol33/tulpa#71: the documented method. Fractional nu returns the rSPDE
  # rational roots from the BRASIL best-rational approximation, not the old
  # self-derived two-pole approximation.
  rat <- rational_spde_coefficients(0.5, m = 4L)
  expect_false(rat$is_integer)
  expect_length(rat$rb, 4L)            # m denominator factors (-> Pl)
  expect_length(rat$rc, 4L)            # m numerator factors   (-> Pr)
  expect_true(all(is.finite(rat$rb)) && all(is.finite(rat$rc)))
  expect_true(is.finite(rat$scale))
  expect_equal(rat$alpha, 1.5)         # nu + 1 in 2D
  expect_equal(rat$beta, 0.75)

  # Higher order lowers the rational approximation error of the spectral density.
  e3 <- rational_spde_coefficients(0.5, m = 3L)$error
  e5 <- rational_spde_coefficients(0.5, m = 5L)$error
  expect_lt(e5, e3)
})

test_that("spatial_spde accepts fractional nu (gcol33/tulpa#71)", {
  coords <- cbind(runif(20), runif(20))
  expect_s3_class(spatial_spde(coords, nu = 0.5), "tulpa_spatial")
  expect_s3_class(spatial_spde(coords, nu = 1.5), "tulpa_spatial")
  expect_error(spatial_spde(coords, nu = -0.5), "non-negative")
})

test_that("fit_spde with nu = 0 runs without error", {
  # nu=0 (alpha=1) is the first-order operator -- ill-conditioned, rarely used.
  set.seed(42)
  n_obs <- 80
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords, nu = 0)
  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)
  result <- fit_spde(y, X, spec, family = "binomial", range = 0.5, sigma = 0.3)
  expect_true(result$n_iter > 0)
})

test_that("fit_spde works with nu = 1 (alpha = 2, standard)", {
  set.seed(42)
  n_obs <- 80
  coords <- cbind(runif(n_obs), runif(n_obs))
  spec <- spatial_spde(coords, nu = 1)
  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)
  result <- fit_spde(y, X, spec, family = "binomial", range = 0.3, sigma = 0.5)
  expect_true(result$converged)
  expect_true(is.finite(result$log_marginal))
})

# Helper: a 1D circulant FEM wrapped as a custom SPDE spec, where the operator
# spectrum is known and dense checks are cheap.
.frac_1d_spec <- function(n = 150L, nu = 0.5, order = 2L,
                          range_true = 0.15, sigma_true = 1.0) {
  h <- 1 / n
  Cd <- Matrix::Diagonal(n, rep(h, n))
  G <- Matrix::bandSparse(n, n, c(-1, 0, 1),
                          list(rep(-1 / h, n - 1), rep(2 / h, n), rep(-1 / h, n - 1)))
  G <- as(G, "CsparseMatrix"); G[1, n] <- -1 / h; G[n, 1] <- -1 / h
  A <- as(Matrix::Diagonal(n), "CsparseMatrix")
  spec <- spatial_spde_custom(C = Cd, G = G, A = A, nu = nu,
                              prior_range = c(range_true, 0.5),
                              prior_sigma = c(sigma_true, 0.5))
  spec$rational_order <- order
  spec
}

test_that("fractional single-point Laplace recovers the field", {
  skip_on_cran()
  n <- 150L
  spec <- .frac_1d_spec(n = n, nu = 0.5, order = 2L)
  set.seed(11)
  a <- tulpa:::.spde_assemble_at(spec, range = 0.15, sigma = 1.0, order = 2L)
  ch <- Matrix::Cholesky(a$Q, LDL = FALSE, perm = TRUE)
  x_true <- as.numeric(Matrix::solve(ch, Matrix::solve(ch, rnorm(n), system = "Lt"),
                                     system = "Pt"))
  u_true <- as.numeric(a$Pr %*% x_true)
  y <- 0.5 + u_true + rnorm(n, 0, sqrt(0.05))
  X <- matrix(1, n, 1)

  fit <- tulpa:::laplace_spde_at(y = y, n_trials = rep(1L, n), X = X, spatial = spec,
                                 family = "gaussian", phi = 0.05,
                                 range = 0.15, sigma = 1.0)
  u_hat <- fit$mode[-1]                      # mesh block is field-space
  expect_gt(cor(u_hat, u_true), 0.9)
})

test_that("fractional nested fit_spde recovers (range, sigma) -- Gaussian", {
  skip_on_cran()
  n <- 150L
  range_true <- 0.15; sigma_true <- 1.0
  spec <- .frac_1d_spec(n = n, nu = 0.5, order = 2L,
                        range_true = range_true, sigma_true = sigma_true)
  set.seed(11)
  a <- tulpa:::.spde_assemble_at(spec, range = range_true, sigma = sigma_true, order = 2L)
  ch <- Matrix::Cholesky(a$Q, LDL = FALSE, perm = TRUE)
  x_true <- as.numeric(Matrix::solve(ch, Matrix::solve(ch, rnorm(n), system = "Lt"),
                                     system = "Pt"))
  u_true <- as.numeric(a$Pr %*% x_true)
  y <- 0.5 + u_true + rnorm(n, 0, sqrt(0.05))
  X <- matrix(1, n, 1)

  fit <- fit_spde(y = y, X = X, spatial = spec, family = "gaussian", phi = 0.05,
                  method = "grid", n_grid = 9L, diagnose_k = FALSE)
  # The nested marginal must be peaked near the truth (interior, not boundary).
  expect_gt(fit$nested$range_mean, range_true * 0.5)
  expect_lt(fit$nested$range_mean, range_true * 2.0)
  expect_gt(fit$nested$sigma_mean, sigma_true * 0.6)
  expect_lt(fit$nested$sigma_mean, sigma_true * 1.6)
})

test_that("fractional nested fit_spde recovers range on a 2D mesh -- Gaussian", {
  skip_on_cran()
  skip_if_not_installed("tulpaMesh")
  set.seed(2024)
  np <- 400L
  coords <- cbind(runif(np), runif(np))
  range_true <- 0.3; sigma_true <- 1.5
  spec <- spatial_spde(coords, nu = 0.5, max_edge = c(0.07, 0.2),
                       prior_range = c(range_true, 0.5), prior_sigma = c(sigma_true, 0.5))
  spec$rational_order <- 2L

  a <- tulpa:::.spde_assemble_at(spec, range = range_true, sigma = sigma_true, order = 2L)
  ns <- length(a$keep)
  ch <- Matrix::Cholesky(a$Q, LDL = FALSE, perm = TRUE)
  x_true <- as.numeric(Matrix::solve(ch, Matrix::solve(ch, rnorm(ns), system = "Lt"),
                                     system = "Pt"))
  field_full <- numeric(spec$n_mesh)
  field_full[a$keep] <- as.numeric(a$Pr %*% x_true)
  u <- as.numeric(spec$A %*% field_full)
  y <- 0.0 + u + rnorm(np, 0, 0.3)
  X <- matrix(1, np, 1)

  fit <- fit_spde(y = y, X = X, spatial = spec, family = "gaussian", phi = 0.09,
                  method = "grid", n_grid = 9L, diagnose_k = FALSE)
  # Range estimate must be interior to the grid (the old broken path pegged it)
  # and recover near the truth.
  rg <- sort(unique(fit$nested$range_grid))
  expect_gt(fit$nested$range_best, rg[1])
  expect_lt(fit$nested$range_best, rg[length(rg)])
  expect_gt(fit$nested$range_mean, range_true * 0.5)
  expect_lt(fit$nested$range_mean, range_true * 1.7)
  expect_gt(fit$nested$sigma_mean, sigma_true * 0.6)
  expect_lt(fit$nested$sigma_mean, sigma_true * 1.6)
})

test_that("fractional nu: fixed-hyper NUTS runs, joint NUTS gated, analytic-Q SE gated", {
  spec <- .frac_1d_spec(n = 40L, nu = 0.5)
  # Joint-over-hypers fractional NUTS stays gated -- the rational roots are not
  # differentiable in kappa (gcol33/tulpa#85).
  expect_error(
    tulpa_nuts_spde(y = rnorm(40), X = matrix(1, 40, 1), spatial = spec,
                    family = "gaussian", joint = TRUE),
    "fractional"
  )
  # Fixed-hyper fractional NUTS now runs: the rational precision Q is precomputed
  # in R and passed to the sampler (#85).
  set.seed(1)
  fit <- tulpa_nuts_spde(y = rnorm(40), X = matrix(1, 40, 1), spatial = spec,
                         family = "gaussian", n_iter = 60L, n_warmup = 30L)
  expect_true(isTRUE(fit$rational))
  expect_true(is.matrix(fit$draws))
  # The analytic integer-Q rebuild helper stays integer-only: the fractional
  # marginal SE goes through the rational assembly (.spde_assemble_at), not this
  # shifted-alpha rebuild.
  expect_error(
    tulpa:::.spde_precision_Q(spec, kappa = 5, tau_spde = 1),
    "integer-nu only"
  )
})

test_that("different integer nu values produce different log-marginals", {
  set.seed(42)
  n_obs <- 100
  coords <- cbind(runif(n_obs), runif(n_obs))
  y <- rbinom(n_obs, 1, 0.4)
  X <- matrix(1, nrow = n_obs, ncol = 1)
  nu_vals <- c(0, 1, 2)
  lmls <- numeric(length(nu_vals))
  for (i in seq_along(nu_vals)) {
    spec <- spatial_spde(coords, nu = nu_vals[i])
    result <- fit_spde(y, X, spec, family = "binomial", range = 0.3, sigma = 0.5)
    lmls[i] <- result$log_marginal
  }
  expect_true(length(unique(round(lmls, 2))) > 1)
})
