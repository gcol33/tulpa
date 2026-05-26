# Tests for tulpa_nmix_laplace_car_proper() -- nested-Laplace fit of the
# spatial Royle (2004) N-mixture model with a proper CAR(rho) prior on
# log lambda.

grid_adj_csr_car <- function(n_row, n_col) {
  n_units <- n_row * n_col
  neighbours_for <- function(s) {
    r <- ((s - 1L) %/% n_col) + 1L
    c <- ((s - 1L) %% n_col) + 1L
    out <- integer(0)
    if (r > 1L)    out <- c(out, s - n_col)
    if (r < n_row) out <- c(out, s + n_col)
    if (c > 1L)    out <- c(out, s - 1L)
    if (c < n_col) out <- c(out, s + 1L)
    out
  }
  cols <- lapply(seq_len(n_units), neighbours_for)
  n_neighbors <- vapply(cols, length, integer(1))
  list(
    n_units     = n_units,
    n_neighbors = n_neighbors,
    adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx = as.integer(unlist(cols) - 1L)
  )
}

simulate_nmix_car_proper <- function(seed,
                                     n_row = 6, n_col = 6,
                                     J = 4,
                                     beta_lambda = c(log(4), 0.4),
                                     beta_p = c(qlogis(0.45), -0.25),
                                     rho_true = 0.85,
                                     tau_true = 1.0,
                                     r = Inf) {
  set.seed(seed)
  n_sites <- n_row * n_col
  adj <- grid_adj_csr_car(n_row, n_col)

  W <- matrix(0, n_sites, n_sites)
  for (s in seq_len(n_sites)) {
    a <- adj$adj_row_ptr[s] + 1L
    b <- adj$adj_row_ptr[s + 1L]
    if (b >= a) for (kk in a:b) {
      t <- adj$adj_col_idx[kk] + 1L
      W[s, t] <- 1
    }
  }
  D <- diag(adj$n_neighbors)
  Q <- D - rho_true * W
  Sigma <- solve(Q) / tau_true
  # Symmetrise to clear tiny round-off.
  Sigma <- (Sigma + t(Sigma)) / 2
  z_field <- as.vector(chol(Sigma) %*% rnorm(n_sites))
  z_field <- z_field - mean(z_field)

  elev <- rnorm(n_sites)
  X_lambda <- cbind(intercept = 1, elev = elev)
  eta_lam <- as.vector(X_lambda %*% beta_lambda) + z_field
  lambda  <- exp(eta_lam)

  wind <- matrix(rnorm(n_sites * J), n_sites, J)
  X_p <- cbind(intercept = 1, wind = as.vector(t(wind)))
  site_idx <- as.integer(rep(seq_len(n_sites), each = J))

  eta_p_mat <- matrix(0, n_sites, J)
  for (s in seq_len(n_sites)) {
    eta_p_mat[s, ] <- beta_p[1] + beta_p[2] * wind[s, ]
  }
  p_mat <- plogis(eta_p_mat)

  N <- if (is.finite(r)) rnbinom(n_sites, size = r, mu = lambda) else rpois(n_sites, lambda)
  y_mat <- matrix(0L, n_sites, J)
  for (s in seq_len(n_sites)) for (j in seq_len(J)) {
    y_mat[s, j] <- rbinom(1, N[s], p_mat[s, j])
  }
  y_long <- as.integer(as.vector(t(y_mat)))

  list(
    y = y_long, site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    map_site_to_unit = seq_len(n_sites),
    adj = adj,
    z_true = z_field,
    rho_true = rho_true,
    tau_true = tau_true,
    r_true = r,
    beta_lambda_true = beta_lambda,
    beta_p_true = beta_p,
    n_sites = n_sites
  )
}

test_that("Spatial N-mix CAR_proper nested Laplace runs end-to-end", {
  dat <- simulate_nmix_car_proper(seed = 11)
  fit <- suppressWarnings(tulpa_nmix_laplace_car_proper(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 4L)),
    rho_grid = c(0.3, 0.6, 0.85),
    K_max    = max(dat$y) + 200L,
    max_iter = 100L, tol = 1e-6
  ))
  expect_equal(nrow(fit$theta_grid), length(fit$tau_grid) * length(fit$rho_grid))
  # theta_grid always carries the dispersion axis (tau, rho, r); Poisson uses r = Inf.
  expect_equal(ncol(fit$theta_grid), 3L)
  expect_true(all(is.finite(fit$log_marginal)))
  expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
  expect_true(all(fit$boundary_max < 1e-2))
  # log_det_Q_rho should be finite and monotonic-decreasing in rho.
  expect_true(all(is.finite(fit$log_det_Q_rho)))
  expect_true(all(diff(fit$log_det_Q_rho) < 0))
})

test_that("CAR_proper recovers rho near truth on simulated data", {
  skip_on_cran()
  dat <- simulate_nmix_car_proper(seed = 13, n_row = 8, n_col = 8,
                                  rho_true = 0.85, tau_true = 1.0)
  fit <- tulpa_nmix_laplace_car_proper(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = exp(seq(log(0.3), log(10), length.out = 5L)),
    rho_grid = c(0.2, 0.5, 0.7, 0.85, 0.95),
    max_iter = 120L, tol = 1e-6
  )
  # Slopes are well-identified; intercepts ride the N-mix identifiability ridge.
  expect_lt(abs(fit$beta_lambda_mean["elev"] - dat$beta_lambda_true[2]), 0.25)
  expect_lt(abs(fit$beta_p_mean["wind"]      - dat$beta_p_true[2]),      0.25)
  # rho posterior mean should be in the upper half of (0,1) for a true rho = 0.85.
  expect_gt(fit$rho_mean, 0.5)
  # Spatial field recovery: correlation with truth should be positive.
  rho_z <- cor(fit$z_mean, dat$z_true)
  expect_gt(rho_z, 0.3)
})

test_that("CAR_proper at low rho is close to non-spatial fit", {
  skip_on_cran()
  # Use truth with no spatial structure (rho effectively zero via z_scale).
  set.seed(17)
  n_row <- 6; n_col <- 6; J <- 4
  adj <- grid_adj_csr_car(n_row, n_col)
  n_sites <- adj$n_units
  beta_lam <- c(log(4), 0.4); beta_p <- c(qlogis(0.45), -0.25)
  elev <- rnorm(n_sites)
  X_lambda <- cbind(intercept = 1, elev = elev)
  lambda <- exp(as.vector(X_lambda %*% beta_lam))
  wind <- matrix(rnorm(n_sites * J), n_sites, J)
  X_p <- cbind(intercept = 1, wind = as.vector(t(wind)))
  site_idx <- as.integer(rep(seq_len(n_sites), each = J))
  eta_p_mat <- matrix(beta_p[1], n_sites, J)
  for (s in seq_len(n_sites)) eta_p_mat[s, ] <- beta_p[1] + beta_p[2] * wind[s, ]
  p_mat <- plogis(eta_p_mat)
  N <- rpois(n_sites, lambda)
  y_mat <- matrix(0L, n_sites, J)
  for (s in seq_len(n_sites)) for (j in seq_len(J)) {
    y_mat[s, j] <- rbinom(1, N[s], p_mat[s, j])
  }
  y_long <- as.integer(as.vector(t(y_mat)))

  # Strict low-rho high-tau grid drives the CAR field toward zero.
  fit_sp <- tulpa_nmix_laplace_car_proper(
    y = y_long, site_idx = site_idx,
    map_site_to_unit = seq_len(n_sites),
    X_lambda = X_lambda, X_p = X_p,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    n_spatial = adj$n_units,
    tau_grid = c(1e3, 1e4),
    rho_grid = c(0.1, 0.3),
    max_iter = 80L, tol = 1e-6
  )
  fit_ns <- tulpa_nmix_laplace(
    y = y_long, site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    max_iter = 80L, tol = 1e-6
  )
  expect_lt(max(abs(fit_sp$beta_lambda_mean - fit_ns$beta_lambda)), 0.10)
  expect_lt(max(abs(fit_sp$beta_p_mean      - fit_ns$beta_p)),      0.10)
  expect_lt(max(abs(fit_sp$z_mean)), 0.1)
})

test_that("Print method runs for CAR_proper fit", {
  dat <- simulate_nmix_car_proper(seed = 19, n_row = 4, n_col = 4)
  fit <- tulpa_nmix_laplace_car_proper(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = c(1, 5),
    rho_grid = c(0.5, 0.85),
    max_iter = 50L
  )
  expect_output(print(fit), "spatial N-mixture")
  expect_output(print(fit), "CAR")
})

test_that("CAR_proper NB integrates r as a third grid axis", {
  skip_on_cran()
  dat <- simulate_nmix_car_proper(seed = 24, n_row = 6, n_col = 6, J = 5, r = 2)
  fit <- suppressWarnings(tulpa_nmix_laplace_car_proper(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 4L)),
    rho_grid = c(0.3, 0.7, 0.95),
    mixture = "NB", r_grid = c(1, 2, 5, 15),
    max_iter = 100L, tol = 1e-6
  ))
  expect_identical(fit$mixture, "NB")
  expect_true(any(fit$converged))
  expect_true(is.finite(fit$r_mean) && fit$r_mean > 0)
  expect_true(all(c("tau", "rho", "r") %in% colnames(fit$theta_grid)))
  expect_equal(length(fit$weights), nrow(fit$theta_grid))
})
