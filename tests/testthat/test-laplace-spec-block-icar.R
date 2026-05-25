# GMRF latent block in the spec-driven Laplace solver (L2 of the solver
# unification, see clean_migration.md). cpp_laplace_spec_test_icar drives the
# generalized laplace_mode_spec_dense_impl with a single ICAR LatentBlock built
# exactly as cpp_nested_laplace_icar builds it (same add_icar_prior /
# log_prior_icar / center_effects callbacks) and a builtin_family_spec
# likelihood. So the spec-driven inner solve must reproduce the family-enum
# nested kernel's inner solve at a single hyperparameter cell:
#   * the mode must match (full gradient: likelihood cross-terms + block prior
#     gradient + centering),
#   * the Laplace log-marginal must match exactly. The nested kernel now routes
#     its inner solve through the same spec path (L3), so both include the
#     beta-prior log-density -- the convention gap the earlier fixtures flagged
#     is reconciled. Exact log-marginal agreement also pins the shared
#     log-likelihood and the shared Hessian (via log|H|).

# Chain graph on K nodes: s ~ s-1, s+1. CSR adjacency with 0-based col idx.
build_chain_adj <- function(K) {
  nbrs <- lapply(seq_len(K), function(s) c(if (s > 1L) s - 1L, if (s < K) s + 1L))
  n_neighbors <- vapply(nbrs, length, integer(1))
  list(
    adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx = as.integer(unlist(nbrs) - 1L),
    n_neighbors = as.integer(n_neighbors)
  )
}

# Spec mode/log-marginal must match the nested ICAR kernel at one tau cell.
expect_icar_match <- function(family, y, n_trials, X, spatial_idx, K, adj,
                              tau_spatial, phi = 1.0,
                              re_idx = integer(0), n_re_groups = 0L,
                              sigma_re = 1.0, sigma_beta = 100,
                              tol = 1e-5) {
  ref <- tulpa:::cpp_nested_laplace_icar(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.numeric(re_idx) else numeric(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    spatial_idx = as.integer(spatial_idx), n_spatial_units = K,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_grid = tau_spatial, family = family, phi = phi,
    max_iter = 300L, tol = 1e-11, n_threads = 1L
  )
  spec <- tulpa:::cpp_laplace_spec_test_icar(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.integer(re_idx) else integer(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    spatial_idx = as.integer(spatial_idx), n_spatial_units = K,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    tau_spatial = tau_spatial, sigma_beta = sigma_beta,
    family = family, phi = phi, max_iter = 300L, tol = 1e-11, n_threads = 1L
  )

  expect_true(spec$converged, info = paste0(family, " ICAR: converged"))
  ref_mode <- as.numeric(ref$modes[1, ])
  expect_equal(spec$mode, ref_mode, tolerance = tol,
               info = paste0(family, " ICAR: mode"))

  # L3: the nested kernel routes through the same spec inner solve, so it now
  # includes the beta-prior log-density too -- the log-marginals match exactly.
  expect_equal(spec$log_marginal, ref$log_marginal[1],
               tolerance = tol,
               info = paste0(family, " ICAR: log_marginal"))
}

test_that("spec ICAR block reproduces the nested ICAR kernel (no RE)", {
  set.seed(11L)
  K <- 10L
  adj <- build_chain_adj(K)
  N <- 120L
  spatial_idx <- sample.int(K, N, replace = TRUE)
  x_cov <- rnorm(N)
  X <- cbind(1, x_cov)
  beta <- c(0.1, 0.5)
  phi_field <- as.numeric(scale(sin(seq_len(K)), scale = FALSE))  # sum-to-zero-ish
  eta <- as.numeric(X %*% beta) + phi_field[spatial_idx]
  ones <- rep(1L, N)

  expect_icar_match("gaussian", eta + rnorm(N, 0, 0.6), ones, X,
                    spatial_idx, K, adj, tau_spatial = 2.0, phi = 0.6)
  expect_icar_match("poisson", rpois(N, exp(eta)), ones, X,
                    spatial_idx, K, adj, tau_spatial = 1.5, phi = 1.0)
  expect_icar_match("binomial", rbinom(N, 12L, plogis(eta)), rep(12L, N), X,
                    spatial_idx, K, adj, tau_spatial = 1.0, phi = 1.0)
})

test_that("spec ICAR block reproduces the nested ICAR kernel (with iid RE)", {
  set.seed(23L)
  K <- 8L
  adj <- build_chain_adj(K)
  N <- 150L
  n_re_groups <- 10L
  sigma_re <- 0.5
  spatial_idx <- sample.int(K, N, replace = TRUE)
  re_idx <- sample.int(n_re_groups, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  beta <- c(0.2, -0.3)
  phi_field <- as.numeric(scale(cos(seq_len(K)), scale = FALSE))
  u <- rnorm(n_re_groups, sd = sigma_re)
  eta <- as.numeric(X %*% beta) + phi_field[spatial_idx] + u[re_idx]
  ones <- rep(1L, N)

  expect_icar_match("poisson", rpois(N, exp(eta)), ones, X,
                    spatial_idx, K, adj, tau_spatial = 1.5, phi = 1.0,
                    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re)
  expect_icar_match("binomial", rbinom(N, 8L, plogis(eta)), rep(8L, N), X,
                    spatial_idx, K, adj, tau_spatial = 1.0, phi = 1.0,
                    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re)
})
