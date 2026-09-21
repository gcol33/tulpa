# Spatial fixtures shared by the sparse-Cholesky, nested-Laplace and
# selected-inversion tests. A helper file rather than a test file sourced from
# the others: sourcing a test file runs every test_that() block in it again,
# once per file that sources it.

# The 4-connected neighbours of every cell of an nrow x ncol lattice, cells
# numbered row-major from 1, each cell's neighbours in the order up, down,
# left, right. The one construction behind both views below, so a fixture
# reading the list and one reading the CSR cannot come to describe different
# graphs.
grid_neighbours <- function(nrow, ncol) {
  lapply(seq_len(nrow * ncol), function(i) {
    r <- (i - 1) %/% ncol + 1
    c <- (i - 1) %% ncol + 1
    nb <- integer(0)
    if (r > 1)    nb <- c(nb, i - ncol)
    if (r < nrow) nb <- c(nb, i + ncol)
    if (c > 1)    nb <- c(nb, i - 1L)
    if (c < ncol) nb <- c(nb, i + 1L)
    as.integer(nb)
  })
}

# The same lattice as the 0-based CSR spec the spatial kernels take.
make_grid_adjacency <- function(nrow, ncol) {
  neighbors <- grid_neighbours(nrow, ncol)
  n_neighbors <- vapply(neighbors, length, integer(1))
  list(
    n = nrow * ncol,
    adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx = as.integer(unlist(neighbors) - 1L),
    n_neighbors = as.integer(n_neighbors)
  )
}

# Helper: simulate binomial spatial data on a grid
simulate_spatial_data <- function(n_sites, n_obs_per_site, beta0, tau_spatial, adj) {
  set.seed(42)
  n_obs <- n_sites * n_obs_per_site

  # Simulate ICAR spatial effects (approximate: draw from N(0, 1/tau) then smooth)
  spatial_raw <- rnorm(n_sites, 0, 1 / sqrt(tau_spatial))

  # Simple spatial smoothing via neighbor averaging (2 passes)
  for (pass in 1:2) {
    smoothed <- numeric(n_sites)
    for (s in seq_len(n_sites)) {
      start <- adj$adj_row_ptr[s] + 1L
      end <- adj$adj_row_ptr[s + 1]
      if (end >= start) {
        nb_idx <- adj$adj_col_idx[start:end] + 1L  # back to 1-based
        smoothed[s] <- mean(spatial_raw[nb_idx])
      }
    }
    spatial_raw <- smoothed
  }
  spatial_raw <- spatial_raw - mean(spatial_raw)

  # Design: intercept only
  X <- matrix(1, nrow = n_obs, ncol = 1)

  # Site assignments (1-based)
  spatial_idx <- rep(seq_len(n_sites), each = n_obs_per_site)

  # Linear predictor
  eta <- beta0 + spatial_raw[spatial_idx]
  p <- plogis(eta)

  # Binomial response
  n_trials <- rep(1L, n_obs)
  y <- rbinom(n_obs, size = n_trials, prob = p)

  list(
    y = as.integer(y),
    n_trials = as.integer(n_trials),
    X = X,
    spatial_idx = as.integer(spatial_idx),
    eta_true = eta,
    spatial_true = spatial_raw
  )
}
