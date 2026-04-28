# Tests for proper-CAR spatial wiring in tulpa.
#
# These exercise:
#   - The R-side spatial_car_proper() constructor.
#   - The C++ dispatch: spatial_type_str = "car_proper" routes to
#     SpatialType::CAR_PROPER and allocates a logit_rho_car parameter.
#   - The handcoded log_post / gradient code paths for CAR_PROPER —
#     verified against the numerical reference via gradient_check_only.

# --- Helper: build a chain adjacency matrix -------------------------------
chain_adj <- function(n) {
  adj <- matrix(0L, n, n)
  for (i in seq_len(n - 1)) {
    adj[i, i + 1] <- 1L
    adj[i + 1, i] <- 1L
  }
  adj
}

# --- Helper: build CSR triple from adjacency ------------------------------
adj_to_csr <- function(adj) {
  n <- nrow(adj)
  row_ptr <- integer(n + 1)
  col_idx <- integer(0)
  n_neighbors <- integer(n)
  for (i in seq_len(n)) {
    nbrs <- which(adj[i, ] != 0)
    n_neighbors[i] <- length(nbrs)
    col_idx <- c(col_idx, nbrs - 1L)
    row_ptr[i + 1] <- row_ptr[i] + length(nbrs)
  }
  list(row_ptr = row_ptr, col_idx = col_idx, n_neighbors = n_neighbors)
}

# --- Helper: simulate from a proper-CAR field -----------------------------
# phi ~ N(0, (tau * (D - rho * W))^{-1})
sim_car_proper_field <- function(adj, tau = 1.0, rho = 0.7, seed = 42) {
  set.seed(seed)
  n <- nrow(adj)
  D <- diag(rowSums(adj))
  Q <- tau * (D - rho * adj)
  L <- chol(Q)
  z <- rnorm(n)
  # phi solves L^T phi = z, then phi has covariance Q^{-1}
  backsolve(L, z)
}

test_that("spatial_car_proper() returns a tulpa_spatial with type='car_proper'", {
  adj <- chain_adj(5)
  spec <- spatial_car_proper(adj, level = "group", group_var = "site")
  expect_s3_class(spec, "tulpa_spatial")
  expect_equal(spec$type, "car_proper")
  expect_true(spec$proper)
  expect_equal(spec$n_spatial, 5)
  expect_false(is.null(spec$rho_bounds))
  expect_named(spec$rho_bounds, c("lower", "upper"))
})

test_that("spatial_car(adj, proper = TRUE) is equivalent to spatial_car_proper", {
  adj <- chain_adj(6)
  a <- spatial_car_proper(adj, level = "group", group_var = "x")
  b <- spatial_car(adj, level = "group", group_var = "x", proper = TRUE)
  expect_equal(a$type, b$type)
  expect_equal(a$rho_bounds, b$rho_bounds)
  expect_equal(a$n_spatial, b$n_spatial)
})

test_that("CAR_PROPER handcoded gradient agrees with numerical reference", {
  # Small chain adjacency (10 sites) keeps the dense O(n^3) Cholesky
  # in the proper-CAR log-prior cheap.
  set.seed(2026)
  n_sites <- 10
  n_per <- 5
  adj <- chain_adj(n_sites)
  csr <- adj_to_csr(adj)

  # True parameters
  tau_true <- 1.5
  rho_true <- 0.6
  phi_true <- sim_car_proper_field(adj, tau = tau_true, rho = rho_true,
                                   seed = 7)

  # Simulate binomial data: y_num successes / y_denom trials with logit eta
  N <- n_sites * n_per
  site <- rep(seq_len(n_sites), each = n_per)
  x <- scale(rnorm(N))[, 1]
  beta_num <- c(0.0, 0.5)        # intercept + slope
  eta <- beta_num[1] + beta_num[2] * x + phi_true[site]
  prob <- 1 / (1 + exp(-eta))
  trials <- 10L
  y_num <- rbinom(N, size = trials, prob = prob)
  y_denom <- rep(trials, N)

  X_num <- cbind(1, x)
  X_denom <- matrix(1, N, 1)  # placeholder, unused for binomial

  spatial_params <- list(
    type = "car_proper",
    group = as.integer(site),
    n_units = n_sites,
    adj_row_ptr = csr$row_ptr,
    adj_col_idx = csr$col_idx,
    n_neighbors = csr$n_neighbors,
    bym2_scale = 1.0,
    rho_lower = 0.0,
    rho_upper = 1.0,
    parameterization = "standard"
  )

  re_params <- list(
    group = integer(N),
    n_groups = 0L,
    n_terms = 0L,
    parameterization = 1L,
    group_matrix = matrix(0L, 1, 1),
    n_groups_vec = integer(0),
    has_slopes = FALSE,
    has_correlated_slopes = FALSE,
    n_coefs_vec = integer(0),
    correlated_vec = integer(0),
    n_chol_vec = integer(0),
    slope_matrices = list()
  )

  temporal_params <- list(
    type = "none",
    time_idx = integer(N),
    group_idx = integer(N),
    n_times = 0L,
    n_groups = 0L,
    n_params = 0L,
    cyclic = FALSE,
    shared = TRUE,
    tau_shape = 1.0,
    tau_rate = 0.01
  )

  prior_params <- list(
    sigma_beta = 2.5,
    sigma_re_scale = 1.0,
    phi_shape = 1.0,
    phi_rate = 0.01,
    tau_spatial_shape = 1.0,
    tau_spatial_rate = 0.01
  )

  zi_params <- list(type = "none", X = matrix(1, 1, 1), prior_sd = 2.5,
                   X_oi = matrix(1, 1, 1), p_oi = 0L, oi_prior_sd = 2.5,
                   p_zi = 0L)
  latent_params <- list(has_latent = FALSE, n_factors = 0L,
                       shared = TRUE, scale = TRUE,
                       constraint = 0L, sigma_prior_rate = 1.0)
  st_params <- list(has_spatiotemporal = FALSE)
  tvc_params <- list(has_tvc = FALSE)
  svc_params <- list(has_svc = FALSE)

  # Initial parameter vector layout for binomial + CAR_PROPER:
  #   [beta_num(2), beta_denom(1), log_tau_spatial, logit_rho_car, phi(n_sites)]
  # (binomial has no phi/overdispersion params)
  init <- c(0.0, 0.5,           # beta_num: intercept + slope
            0.0,                # beta_denom (placeholder, unused)
            log(tau_true),      # log_tau_spatial
            0.0,                # logit_rho_car (rho ~ 0.5)
            phi_true)           # phi (initialized at truth)

  # Run gradient check (no sampling)
  res <- tulpa:::cpp_hmc_fit(
    q_init = init,
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    y_num_cont = numeric(N),
    y_denom_cont = numeric(N),
    X_num = X_num,
    X_denom = X_denom,
    model_type_str = "binomial",
    re_params = re_params,
    spatial_params = spatial_params,
    temporal_params = temporal_params,
    prior_params = prior_params,
    zi_params = zi_params,
    latent_params = latent_params,
    st_params = st_params,
    tvc_params = tvc_params,
    svc_params = svc_params,
    n_iter = 1L, n_warmup = 0L, L = 1L,
    n_chains = 1L, seed = 1L,
    n_threads = 1L, verbose = FALSE,
    gradient_mode_str = "H",
    max_treedepth = 4L, metric_str = "diag",
    adapt_delta = -1.0, riemannian = -1L,
    gradient_check_only = TRUE
  )

  # gradient_check_only returns a list with $h_vs_n (handcoded vs numerical)
  expect_true(is.list(res))
  if (!is.null(res$h_vs_n) && res$h_vs_n >= 0) {
    # Handcoded gradient must agree with finite differences to ~1e-3
    # (loose because the proper-CAR log-det rho gradient uses an
    # analytical formula compared to log-post numerical FD).
    expect_lt(res$h_vs_n, 1e-3)
  }
})
