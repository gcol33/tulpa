# Tests for tulpa_nmix_laplace_bym2() -- nested-Laplace fit of the spatial
# Royle (2004) N-mixture model with a BYM2 (Riebler et al. 2016) prior on the
# abundance-arm spatial offset.

grid_adj_csr_bym2 <- function(n_row, n_col) {
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

simulate_nmix_bym2 <- function(seed,
                               n_row = 6, n_col = 6,
                               J = 4,
                               beta_lambda = c(log(4), 0.4),
                               beta_p = c(qlogis(0.45), -0.25),
                               sigma_true = 1.0,
                               rho_true = 0.7) {
  set.seed(seed)
  n_sites <- n_row * n_col
  adj <- grid_adj_csr_bym2(n_row, n_col)

  W <- matrix(0, n_sites, n_sites)
  for (s in seq_len(n_sites)) {
    a <- adj$adj_row_ptr[s] + 1L
    b <- adj$adj_row_ptr[s + 1L]
    if (b >= a) for (kk in a:b) {
      t <- adj$adj_col_idx[kk] + 1L
      W[s, t] <- 1
    }
  }
  Q <- diag(as.numeric(adj$n_neighbors)) - W
  eig <- eigen(Q, symmetric = TRUE)
  nz_vals <- eig$values[abs(eig$values) > 1e-10]
  scale_factor <- exp(mean(log(nz_vals)))

  # Draw v from scaled ICAR (sum-to-zero), w from N(0, I).
  # Pseudo-inverse via spectral decomposition.
  inv_eig <- ifelse(abs(eig$values) > 1e-10, 1 / eig$values, 0)
  Sigma_v <- eig$vectors %*% diag(inv_eig) %*% t(eig$vectors)
  Sigma_v <- (Sigma_v + t(Sigma_v)) / 2
  v_raw <- as.vector(MASS::mvrnorm(1, mu = rep(0, n_sites), Sigma = Sigma_v))
  v_field <- v_raw - mean(v_raw)
  w_field <- rnorm(n_sites)

  a_coef <- sigma_true * sqrt(rho_true / scale_factor)
  b_coef <- sigma_true * sqrt(1 - rho_true)
  phi <- a_coef * v_field + b_coef * w_field

  elev <- rnorm(n_sites)
  X_lambda <- cbind(intercept = 1, elev = elev)
  eta_lam <- as.vector(X_lambda %*% beta_lambda) + phi
  lambda  <- exp(eta_lam)

  wind <- matrix(rnorm(n_sites * J), n_sites, J)
  X_p <- cbind(intercept = 1, wind = as.vector(t(wind)))
  site_idx <- as.integer(rep(seq_len(n_sites), each = J))

  eta_p_mat <- matrix(0, n_sites, J)
  for (s in seq_len(n_sites)) {
    eta_p_mat[s, ] <- beta_p[1] + beta_p[2] * wind[s, ]
  }
  p_mat <- plogis(eta_p_mat)

  N <- rpois(n_sites, lambda)
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
    v_true = v_field,
    w_true = w_field,
    phi_true = phi,
    sigma_true = sigma_true,
    rho_true = rho_true,
    scale_factor = scale_factor,
    beta_lambda_true = beta_lambda,
    beta_p_true = beta_p,
    n_sites = n_sites
  )
}

test_that("BYM2 nested Laplace runs end-to-end", {
  dat <- simulate_nmix_bym2(seed = 21)
  fit <- suppressWarnings(tulpa_nmix_laplace_bym2(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    sigma_grid = c(0.5, 1.0, 1.5),
    rho_grid = c(0.3, 0.7),
    K_max    = max(dat$y) + 200L,
    max_iter = 100L, tol = 1e-6
  ))
  expect_equal(nrow(fit$theta_grid),
               length(fit$sigma_grid) * length(fit$rho_grid))
  expect_equal(ncol(fit$theta_grid), 2L)
  expect_true(all(is.finite(fit$log_marginal)))
  expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
  expect_true(all(fit$boundary_max < 1e-2))
  expect_true(is.numeric(fit$scale_factor) && fit$scale_factor > 0)
  expect_equal(length(fit$v_mean), dat$adj$n_units)
  expect_equal(length(fit$w_mean), dat$adj$n_units)
  expect_equal(length(fit$phi_mean), dat$adj$n_units)
})

test_that("BYM2 recovers slopes and total offset on simulated data", {
  skip_on_cran()
  dat <- simulate_nmix_bym2(seed = 23, n_row = 8, n_col = 8,
                            sigma_true = 1.0, rho_true = 0.7)
  fit <- tulpa_nmix_laplace_bym2(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    sigma_grid = exp(seq(log(0.3), log(2.5), length.out = 5L)),
    rho_grid = c(0.1, 0.3, 0.5, 0.7, 0.9),
    max_iter = 120L, tol = 1e-6
  )
  # Slopes are well-identified; intercepts ride the N-mix identifiability ridge.
  expect_lt(abs(fit$beta_lambda_mean["elev"] - dat$beta_lambda_true[2]), 0.30)
  expect_lt(abs(fit$beta_p_mean["wind"]      - dat$beta_p_true[2]),      0.30)
  # Total offset phi should correlate positively with the truth.
  rho_phi <- cor(fit$phi_mean, dat$phi_true)
  expect_gt(rho_phi, 0.3)
})

test_that("BYM2 at small sigma collapses to a non-spatial fit", {
  skip_on_cran()
  set.seed(27)
  n_row <- 6; n_col <- 6; J <- 4
  adj <- grid_adj_csr_bym2(n_row, n_col)
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

  # Tiny sigma_grid drives the spatial offset to ~0; estimates should agree
  # closely with a non-spatial fit on the same data.
  fit_sp <- tulpa_nmix_laplace_bym2(
    y = y_long, site_idx = site_idx,
    map_site_to_unit = seq_len(n_sites),
    X_lambda = X_lambda, X_p = X_p,
    adj_row_ptr = adj$adj_row_ptr,
    adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    n_spatial = adj$n_units,
    sigma_grid = c(0.01, 0.05),
    rho_grid = c(0.3, 0.7),
    max_iter = 80L, tol = 1e-6
  )
  fit_ns <- tulpa_nmix_laplace(
    y = y_long, site_idx = site_idx,
    X_lambda = X_lambda, X_p = X_p,
    max_iter = 80L, tol = 1e-6
  )
  expect_lt(max(abs(fit_sp$beta_lambda_mean - fit_ns$beta_lambda)), 0.10)
  expect_lt(max(abs(fit_sp$beta_p_mean      - fit_ns$beta_p)),      0.10)
  expect_lt(max(abs(fit_sp$phi_mean)), 0.1)
})

test_that("Print method runs for BYM2 fit", {
  dat <- simulate_nmix_bym2(seed = 29, n_row = 4, n_col = 4)
  fit <- tulpa_nmix_laplace_bym2(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    sigma_grid = c(0.5, 1.0),
    rho_grid = c(0.3, 0.7),
    max_iter = 50L
  )
  expect_output(print(fit), "spatial N-mixture")
  expect_output(print(fit), "BYM2")
  expect_output(print(fit), "sigma_mean")
})
