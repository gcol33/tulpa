# Tests for compute_log_post = compute_log_prior + compute_log_lik_only
#
# gcol33/tulpa#6 prereq: SMC requires log_prior and log_lik as separable
# callables (target = log_prior + beta * log_lik). The split is exact by
# construction (compute_log_lik_only is derived by subtraction), but we
# still verify it numerically across a few representative model shapes
# to guard against accidental sign / accumulator regressions.

# ---------------------------------------------------------------------------
# Shared fixture-builders (match the simple-sampler subset used by ESS /
# SGHMC / SGLD via populate_model_data_simple).
# ---------------------------------------------------------------------------

empty_re_params <- function(N) list(
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

re_params_with_groups <- function(group_idx, n_groups) list(
  group = as.integer(group_idx),
  n_groups = as.integer(n_groups),
  n_terms = 1L,
  parameterization = 1L,
  group_matrix = matrix(0L, 1, 1),
  n_groups_vec = as.integer(n_groups),
  has_slopes = FALSE,
  has_correlated_slopes = FALSE,
  n_coefs_vec = integer(0),
  correlated_vec = integer(0),
  n_chol_vec = integer(0),
  slope_matrices = list()
)

empty_spatial_params <- function(N) list(
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

icar_spatial_params <- function(site_idx, adj) list(
  type = "icar",
  group = as.integer(site_idx),
  n_units = as.integer(nrow(adj)),
  adj_row_ptr = as.integer(adj$row_ptr),
  adj_col_idx = as.integer(adj$col_idx),
  n_neighbors = as.integer(adj$n_neighbors),
  bym2_scale = 1.0,
  rho_lower = 0.0,
  rho_upper = 1.0,
  parameterization = "standard"
)

empty_temporal_params <- function(N) list(
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

default_prior_params <- function() list(
  sigma_beta = 2.5,
  sigma_re_scale = 1.0,
  phi_shape = 1.0,
  phi_rate = 0.01,
  tau_spatial_shape = 1.0,
  tau_spatial_rate = 0.01
)

empty_zi_params <- function() list(
  type = "none", X = matrix(1, 1, 1), prior_sd = 2.5,
  X_oi = matrix(1, 1, 1), p_oi = 0L, oi_prior_sd = 2.5,
  p_zi = 0L
)

# Build a chain adjacency in CSR form (small enough to keep ICAR cheap).
chain_adj_csr <- function(n) {
  row_ptr <- integer(n + 1)
  col_idx <- integer(0)
  n_neighbors <- integer(n)
  for (i in seq_len(n)) {
    nbrs <- c()
    if (i > 1) nbrs <- c(nbrs, i - 1L)
    if (i < n) nbrs <- c(nbrs, i + 1L)
    n_neighbors[i] <- length(nbrs)
    col_idx <- c(col_idx, nbrs - 1L)
    row_ptr[i + 1] <- row_ptr[i] + length(nbrs)
  }
  list(row_ptr = row_ptr, col_idx = col_idx, n_neighbors = n_neighbors,
       n_units = n)
}

# Stub for ICAR helper above: nrow(adj) where adj is the CSR list.
nrow.list <- function(x) NULL  # no-op so the icar fn above can use nrow on a matrix-like

# ---------------------------------------------------------------------------
# Generic checker: log_post == log_prior + log_lik_only at random draws.
# ---------------------------------------------------------------------------
check_split <- function(fixture, params, tol = 1e-8, label = "") {
  args <- c(list(params = as.numeric(params)), fixture)
  lp_full  <- do.call(tulpa:::cpp_compute_log_post_test, args)
  lp_prior <- do.call(tulpa:::cpp_compute_log_prior_test, args)
  lp_lik   <- do.call(tulpa:::cpp_compute_log_lik_only_test, args)

  expect_true(is.finite(lp_full),
              info = paste0(label, ": log_post not finite (got ", lp_full, ")"))
  expect_true(is.finite(lp_prior), info = paste0(label, ": log_prior not finite"))
  expect_true(is.finite(lp_lik),   info = paste0(label, ": log_lik not finite"))

  resid <- abs(lp_full - (lp_prior + lp_lik))
  expect_lt(resid, tol,
            label = paste0(label, ": |log_post - (prior + lik)| = ", resid))
  invisible(resid)
}

# ---------------------------------------------------------------------------
# Fixture 1: binomial-only, no spatial / temporal / RE.
# ---------------------------------------------------------------------------
make_fixture_binomial_only <- function(seed = 1) {
  set.seed(seed)
  N <- 40
  p_num <- 2
  X_num <- cbind(1, scale(rnorm(N))[, 1])
  X_denom <- matrix(1, N, 1)  # placeholder
  trials <- 8L
  beta_true <- c(-0.3, 0.6)
  eta <- as.numeric(X_num %*% beta_true)
  prob <- 1 / (1 + exp(-eta))
  y_num <- as.integer(rbinom(N, size = trials, prob = prob))
  y_denom <- as.integer(rep(trials, N))

  list(
    y_num = y_num,
    y_denom = y_denom,
    y_denom_cont = numeric(N),
    X_num = X_num,
    X_denom = X_denom,
    model_type_str = "binomial",
    re_params = empty_re_params(N),
    spatial_params = empty_spatial_params(N),
    temporal_params = empty_temporal_params(N),
    prior_params = default_prior_params(),
    zi_params = empty_zi_params()
  )
}

# Layout (binomial, no extras): [beta_num(p_num), beta_denom(1)]
make_init_binomial_only <- function(seed = 11) {
  set.seed(seed)
  c(rnorm(2, sd = 0.3),  # beta_num
    0.0)                 # beta_denom placeholder
}


# ---------------------------------------------------------------------------
# Fixture 2: poisson_gamma with random effect.
# Model layout: [beta_num(p_num), beta_denom(1), log_sigma_re, re(n_groups),
#                log_phi_denom]   (poisson_gamma has phi_denom but no phi_num)
# ---------------------------------------------------------------------------
make_fixture_poisson_re <- function(seed = 2) {
  set.seed(seed)
  N <- 60
  n_groups <- 6L
  group <- as.integer(rep(seq_len(n_groups), length.out = N) - 1L)  # 0-based
  X_num <- cbind(1, scale(rnorm(N))[, 1])
  X_denom <- matrix(1, N, 1)
  beta_true <- c(0.5, 0.4)
  re_true <- rnorm(n_groups, sd = 0.3)
  eta <- as.numeric(X_num %*% beta_true) + re_true[group + 1L]
  y_num <- as.integer(rpois(N, exp(pmin(eta, 5))))

  list(
    y_num = y_num,
    y_denom = integer(N),
    y_denom_cont = pmax(rgamma(N, shape = 2, rate = 1), 1e-3),
    X_num = X_num,
    X_denom = X_denom,
    model_type_str = "poisson_gamma",
    re_params = re_params_with_groups(group, n_groups),
    spatial_params = empty_spatial_params(N),
    temporal_params = empty_temporal_params(N),
    prior_params = default_prior_params(),
    zi_params = empty_zi_params()
  )
}

make_init_poisson_re <- function(n_groups, seed = 22) {
  set.seed(seed)
  c(rnorm(2, sd = 0.3),               # beta_num
    0.0,                               # beta_denom placeholder
    log(0.5),                          # log_sigma_re
    rnorm(n_groups, sd = 0.2),         # re
    log(1.5))                          # log_phi_denom
}


# ---------------------------------------------------------------------------
# Fixture 3: binomial with ICAR spatial.
# Layout: [beta_num(p_num), beta_denom(1), log_tau_spatial, phi_spatial(n_sites)]
# ---------------------------------------------------------------------------
make_fixture_binomial_icar <- function(seed = 3) {
  set.seed(seed)
  n_sites <- 8L
  n_per <- 5L
  N <- n_sites * n_per
  csr <- chain_adj_csr(n_sites)
  csr$adj <- list(row_ptr = csr$row_ptr, col_idx = csr$col_idx,
                  n_neighbors = csr$n_neighbors)
  site <- as.integer(rep(seq_len(n_sites), each = n_per) - 1L)  # 0-based
  X_num <- cbind(1, scale(rnorm(N))[, 1])
  X_denom <- matrix(1, N, 1)
  beta_true <- c(0.0, 0.5)
  phi_true <- rnorm(n_sites, sd = 0.4)
  phi_true <- phi_true - mean(phi_true)  # ICAR identifiability
  eta <- as.numeric(X_num %*% beta_true) + phi_true[site + 1L]
  prob <- 1 / (1 + exp(-eta))
  trials <- 10L
  y_num <- as.integer(rbinom(N, size = trials, prob = prob))
  y_denom <- as.integer(rep(trials, N))

  spatial <- list(
    type = "icar",
    group = site,
    n_units = as.integer(n_sites),
    adj_row_ptr = csr$row_ptr,
    adj_col_idx = csr$col_idx,
    n_neighbors = csr$n_neighbors,
    bym2_scale = 1.0,
    rho_lower = 0.0,
    rho_upper = 1.0,
    parameterization = "standard"
  )

  list(
    fixture = list(
      y_num = y_num,
      y_denom = y_denom,
      y_denom_cont = numeric(N),
      X_num = X_num,
      X_denom = X_denom,
      model_type_str = "binomial",
      re_params = empty_re_params(N),
      spatial_params = spatial,
      temporal_params = empty_temporal_params(N),
      prior_params = default_prior_params(),
      zi_params = empty_zi_params()
    ),
    n_sites = n_sites,
    phi_true = phi_true
  )
}

make_init_binomial_icar <- function(n_sites, seed = 33) {
  set.seed(seed)
  c(rnorm(2, sd = 0.3),         # beta_num
    0.0,                         # beta_denom placeholder
    log(2.0),                    # log_tau_spatial
    rnorm(n_sites, sd = 0.2))    # phi_spatial
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test_that("binomial-only: log_post = log_prior + log_lik (multiple draws)", {
  fix <- make_fixture_binomial_only(seed = 101)
  resids <- replicate(5, {
    init <- make_init_binomial_only(seed = sample.int(.Machine$integer.max, 1))
    check_split(fix, init, tol = 1e-8, label = "binomial-only")
  })
  expect_true(all(resids < 1e-8))
})

test_that("poisson_gamma + RE: log_post = log_prior + log_lik (multiple draws)", {
  fix <- make_fixture_poisson_re(seed = 202)
  n_groups <- fix$re_params$n_groups
  resids <- replicate(5, {
    init <- make_init_poisson_re(n_groups,
                                 seed = sample.int(.Machine$integer.max, 1))
    check_split(fix, init, tol = 1e-8, label = "poisson_gamma + RE")
  })
  expect_true(all(resids < 1e-8))
})

test_that("binomial + ICAR: log_post = log_prior + log_lik (multiple draws)", {
  bundle <- make_fixture_binomial_icar(seed = 303)
  fix <- bundle$fixture
  n_sites <- bundle$n_sites
  resids <- replicate(5, {
    init <- make_init_binomial_icar(n_sites,
                                    seed = sample.int(.Machine$integer.max, 1))
    check_split(fix, init, tol = 1e-8, label = "binomial + ICAR")
  })
  expect_true(all(resids < 1e-8))
})

test_that("layout total_params matches manual init lengths", {
  # Sanity: the test wrappers report the same param count as the fixtures use.
  fix1 <- make_fixture_binomial_only(seed = 1)
  expect_equal(
    do.call(tulpa:::cpp_log_post_split_n_params, fix1[setdiff(names(fix1), "params")]),
    length(make_init_binomial_only())
  )

  fix2 <- make_fixture_poisson_re(seed = 1)
  n_groups2 <- fix2$re_params$n_groups
  expect_equal(
    do.call(tulpa:::cpp_log_post_split_n_params, fix2[setdiff(names(fix2), "params")]),
    length(make_init_poisson_re(n_groups2))
  )

  bundle3 <- make_fixture_binomial_icar(seed = 1)
  expect_equal(
    do.call(tulpa:::cpp_log_post_split_n_params, bundle3$fixture[setdiff(names(bundle3$fixture), "params")]),
    length(make_init_binomial_icar(bundle3$n_sites))
  )
})
