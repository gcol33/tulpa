empty_hmc_re_params <- function(N) list(
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

empty_hmc_spatial_params <- function(N) list(
  type = "none",
  group = integer(N),
  n_units = 0L,
  adj_row_ptr = integer(1),
  adj_col_idx = integer(0),
  n_neighbors = integer(0),
  bym2_scale = 1.0,
  rho_lower = 0.0,
  rho_upper = 1.0,
  parameterization = "standard"
)

empty_hmc_temporal_params <- function(N) list(
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

default_hmc_prior_params <- function() list(
  sigma_beta = 2.5,
  sigma_re_scale = 1.0,
  phi_shape = 1.0,
  phi_rate = 0.01,
  tau_spatial_shape = 1.0,
  tau_spatial_rate = 0.01
)

run_hmc_builder_gradient_check <- function(q_init, y_num, y_denom, X_num,
                                           zi_params) {
  N <- length(y_num)
  tulpa:::cpp_hmc_fit(
    q_init = q_init,
    y_num = as.integer(y_num),
    y_denom = as.integer(y_denom),
    y_num_cont = numeric(N),
    y_denom_cont = numeric(N),
    X_num = X_num,
    X_denom = matrix(1, N, 1),
    model_type_str = "binomial",
    re_params = empty_hmc_re_params(N),
    spatial_params = empty_hmc_spatial_params(N),
    temporal_params = empty_hmc_temporal_params(N),
    prior_params = default_hmc_prior_params(),
    zi_params = zi_params,
    latent_params = list(has_latent = FALSE, n_factors = 0L, shared = TRUE,
                         scale = TRUE, constraint = 0L,
                         sigma_prior_rate = 1.0),
    st_params = list(has_spatiotemporal = FALSE),
    tvc_params = list(has_tvc = FALSE),
    svc_params = list(has_svc = FALSE),
    n_iter = 1L,
    n_warmup = 0L,
    L = 1L,
    n_chains = 1L,
    seed = 1L,
    n_threads = 1L,
    verbose = FALSE,
    gradient_mode_str = "auto",
    max_treedepth = 4L,
    metric_str = "diag",
    adapt_delta = -1.0,
    riemannian = -1L,
    gradient_check_only = TRUE
  )
}

test_that("cpp_hmc_fit builds baseline legacy ModelData after split", {
  N <- 5L
  X_num <- cbind(1, seq_len(N) / N)
  res <- run_hmc_builder_gradient_check(
    q_init = c(0.1, -0.2, 0.0),
    y_num = c(0L, 1L, 1L, 2L, 3L),
    y_denom = rep(4L, N),
    X_num = X_num,
    zi_params = list(type = "none", X = matrix(1, 1, 1), prior_sd = 2.5,
                     X_oi = matrix(1, 1, 1), p_oi = 0L,
                     oi_prior_sd = 2.5, p_zi = 0L)
  )

  expect_true(is.list(res))
  expect_equal(res$n_params, 3L)
  expect_named(res, c("h_vs_n", "ar_vs_n", "a_vs_n", "h_vs_ar", "tol",
                      "h_ok", "ar_ok", "a_ok", "n_params"))
})

test_that("cpp_hmc_fit preserves OI-only p_zi and X_oi boundaries", {
  N <- 5L
  X_num <- cbind(1, seq_len(N) / N)
  X_oi <- matrix(c(1, 0, 1, 0, 1), ncol = 1)
  res <- run_hmc_builder_gradient_check(
    q_init = c(0.1, -0.2, 0.0, -1.0),
    y_num = c(0L, 4L, 1L, 4L, 2L),
    y_denom = rep(4L, N),
    X_num = X_num,
    zi_params = list(type = "oi_binomial", X = matrix(1, 1, 1),
                     prior_sd = 2.5, X_oi = X_oi, p_oi = 1L,
                     oi_prior_sd = 2.5, p_zi = 0L)
  )

  expect_true(is.list(res))
  expect_equal(res$n_params, 4L)
  expect_named(res, c("h_vs_n", "ar_vs_n", "a_vs_n", "h_vs_ar", "tol",
                      "h_ok", "ar_ok", "a_ok", "n_params"))
})
