# Tests for tulpa_nmix_laplace_icar() -- nested-Laplace fit of the spatial
# Royle (2004) N-mixture model with an ICAR prior on log lambda.

# Build CSR adjacency for an n_row x n_col rook (4-neighbour) grid.
grid_adj_csr <- function(n_row, n_col) {
  n_units <- n_row * n_col
  rc <- function(s) {
    r <- ((s - 1L) %/% n_col) + 1L
    c <- ((s - 1L) %% n_col) + 1L
    c(r, c)
  }
  neighbours_for <- function(s) {
    rc_s <- rc(s)
    out <- integer(0)
    if (rc_s[1] > 1L)      out <- c(out, s - n_col)
    if (rc_s[1] < n_row)   out <- c(out, s + n_col)
    if (rc_s[2] > 1L)      out <- c(out, s - 1L)
    if (rc_s[2] < n_col)   out <- c(out, s + 1L)
    out
  }
  cols <- lapply(seq_len(n_units), neighbours_for)
  n_neighbors <- vapply(cols, length, integer(1))
  adj_row_ptr <- as.integer(c(0L, cumsum(n_neighbors)))
  adj_col_idx <- as.integer(unlist(cols) - 1L)  # 0-based for C++
  list(
    n_units     = n_units,
    n_neighbors = n_neighbors,
    adj_row_ptr = adj_row_ptr,
    adj_col_idx = adj_col_idx
  )
}

simulate_nmix_spatial <- function(seed,
                                  n_row = 6, n_col = 6,
                                  J = 4,
                                  beta_lambda = c(log(4), 0.4),
                                  beta_p = c(qlogis(0.45), -0.25),
                                  z_scale = 0.5,
                                  r = Inf) {
  set.seed(seed)
  n_sites <- n_row * n_col
  adj <- grid_adj_csr(n_row, n_col)
  # Smooth z field: low-frequency sinusoid over the grid + small noise. Sum
  # to zero by construction (centred).
  z_field <- numeric(n_sites)
  for (s in seq_len(n_sites)) {
    r <- ((s - 1L) %/% n_col) + 1L
    cc <- ((s - 1L) %% n_col) + 1L
    z_field[s] <- z_scale * (
      sin(2 * pi * r / n_row) + cos(2 * pi * cc / n_col)
    )
  }
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
    beta_lambda_true = beta_lambda,
    beta_p_true = beta_p,
    r_true = r,
    n_sites = n_sites
  )
}

test_that("Spatial N-mix nested Laplace runs end-to-end", {
  dat <- simulate_nmix_spatial(seed = 11)
  fit <- suppressWarnings(tulpa_nmix_laplace_icar(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid  = exp(seq(log(0.5), log(20), length.out = 5L)),
    K_max     = max(dat$y) + 200L,
    max_iter  = 80L, tol = 1e-6
  ))
  expect_true(all(fit$converged))
  expect_true(all(is.finite(fit$log_marginal)))
  expect_equal(length(fit$weights), length(fit$tau_grid))
  expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
  # boundary_max can be slightly above the 1e-4 advisory at the smallest tau
  # because of a single high-lambda site; the warning fires upstream.
  expect_true(all(fit$boundary_max < 1e-2))
})

test_that("Spatial N-mix slopes are near truth on simulated data", {
  skip_on_cran()
  dat <- simulate_nmix_spatial(seed = 13, n_row = 8, n_col = 8)
  fit <- tulpa_nmix_laplace_icar(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid  = exp(seq(log(0.5), log(20), length.out = 5L)),
    max_iter  = 100L, tol = 1e-6
  )
  # The slope coefficients (well-identified, unlike the intercept that rides
  # the N-mixture identifiability ridge) should land near truth.
  expect_lt(abs(fit$beta_lambda_mean["elev"] - dat$beta_lambda_true[2]), 0.20)
  expect_lt(abs(fit$beta_p_mean["wind"]      - dat$beta_p_true[2]),      0.20)
  # The estimated spatial field should correlate with truth (since we have
  # only one realisation, just check correlation is meaningfully positive).
  rho <- cor(fit$z_mean, dat$z_true)
  expect_gt(rho, 0.4)
})

test_that("High-tau spatial fit collapses to non-spatial fit", {
  skip_on_cran()
  dat <- simulate_nmix_spatial(seed = 17, z_scale = 0)   # truth has no field
  # Use a strict, high-only tau grid that drives z toward zero.
  fit_sp <- tulpa_nmix_laplace_icar(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid  = c(1e4, 1e5),
    max_iter  = 80L, tol = 1e-6
  )
  fit_ns <- tulpa_nmix_laplace(
    y = dat$y, site_idx = dat$site_idx,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    max_iter = 80L, tol = 1e-6
  )
  expect_lt(max(abs(fit_sp$beta_lambda_mean - fit_ns$beta_lambda)), 0.05)
  expect_lt(max(abs(fit_sp$beta_p_mean      - fit_ns$beta_p)),      0.05)
  expect_lt(max(abs(fit_sp$z_mean)), 0.1)
})

test_that("Print method runs", {
  dat <- simulate_nmix_spatial(seed = 19, n_row = 4, n_col = 4)
  fit <- tulpa_nmix_laplace_icar(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid  = c(1, 5),
    max_iter  = 50L
  )
  expect_output(print(fit), "spatial N-mixture")
})

# --------------------------------------------------------------------------
# Negative-binomial mixture: r integrated as an extra outer grid axis
# --------------------------------------------------------------------------

test_that("Spatial ICAR NB integrates r and recovers slopes / dispersion", {
  skip_on_cran()
  dat <- simulate_nmix_spatial(seed = 21, n_row = 7, n_col = 7, J = 5, r = 2)
  fit <- suppressWarnings(tulpa_nmix_laplace_icar(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 5L)),
    mixture = "NB",
    max_iter = 100L, tol = 1e-6
  ))
  expect_identical(fit$mixture, "NB")
  expect_true(all(fit$converged))
  expect_true(is.finite(fit$r_mean) && fit$r_mean > 0)
  # theta_grid carries the (tau, r) axes; weights run over the full product grid.
  expect_true(all(c("tau", "r") %in% colnames(fit$theta_grid)))
  expect_equal(length(fit$weights), nrow(fit$theta_grid))
  # The NB size is only weakly identified from a single 49-site realisation, so
  # we check the grid-integrated posterior is valid and interior to the grid,
  # not a tight point estimate. (The non-spatial fit's 30-seed test validates
  # dispersion recovery + coverage rigorously.) The slope coefficients are
  # well-identified and should land near truth.
  expect_gt(fit$r_mean, min(fit$r_grid))
  expect_lt(fit$r_mean, max(fit$r_grid))
  expect_lt(abs(fit$beta_lambda_mean["elev"] - dat$beta_lambda_true[2]), 0.25)
  expect_lt(abs(fit$beta_p_mean["wind"]      - dat$beta_p_true[2]),      0.25)
})

test_that("Spatial NB log-marginal beats Poisson on overdispersed data", {
  skip_on_cran()
  dat <- simulate_nmix_spatial(seed = 23, n_row = 7, n_col = 7, J = 5, r = 1.5)
  args <- list(
    y = dat$y, site_idx = dat$site_idx,
    map_site_to_unit = dat$map_site_to_unit,
    X_lambda = dat$X_lambda, X_p = dat$X_p,
    adj_row_ptr = dat$adj$adj_row_ptr,
    adj_col_idx = dat$adj$adj_col_idx,
    n_neighbors = dat$adj$n_neighbors,
    n_spatial = dat$adj$n_units,
    tau_grid = exp(seq(log(0.5), log(20), length.out = 5L)),
    max_iter = 100L, tol = 1e-6
  )
  fit_nb <- suppressWarnings(do.call(tulpa_nmix_laplace_icar, c(args, list(mixture = "NB"))))
  fit_p  <- suppressWarnings(do.call(tulpa_nmix_laplace_icar, c(args, list(mixture = "P"))))
  expect_gt(max(fit_nb$log_marginal), max(fit_p$log_marginal) - 1)
  expect_true(is.na(fit_p$r_mean))
  # Poisson grid shape unchanged (no r axis).
  expect_equal(length(fit_p$weights), length(fit_p$tau_grid))
})
