# BYM2 (two-block) GMRF in the spec-driven Laplace solver (L2, see
# clean_migration.md). cpp_laplace_spec_test_bym2 drives the generalized
# laplace_mode_spec_dense_impl with the same phi (ICAR, centered) + theta (IID)
# LatentBlocks cpp_nested_laplace_bym2 builds, with grid-dependent d_fac. This
# exercises the paths the single ICAR block does not: d_fac != 1 on the eta
# mixing and the block x block likelihood cross-term (both blocks active at the
# same observation). As in the ICAR test, the spec log-marginal matches the
# nested kernel up to the beta-prior log-density the nested kernel omits.

build_chain_adj <- function(K) {
  nbrs <- lapply(seq_len(K), function(s) c(if (s > 1L) s - 1L, if (s < K) s + 1L))
  n_neighbors <- vapply(nbrs, length, integer(1))
  list(
    adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
    adj_col_idx = as.integer(unlist(nbrs) - 1L),
    n_neighbors = as.integer(n_neighbors)
  )
}

expect_bym2_match <- function(family, y, n_trials, X, spatial_idx, K, adj,
                              scale_factor, sigma_spatial, rho, phi = 1.0,
                              re_idx = integer(0), n_re_groups = 0L,
                              sigma_re = 1.0, sigma_beta = 100,
                              tol = 1e-5) {
  ref <- tulpa:::cpp_nested_laplace_bym2(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.numeric(re_idx) else numeric(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    spatial_idx = as.integer(spatial_idx), n_spatial_units = K,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    scale_factor = scale_factor,
    sigma_spatial_grid = sigma_spatial, rho_grid = rho,
    family = family, phi = phi, max_iter = 300L, tol = 1e-11, n_threads = 1L
  )
  spec <- tulpa:::cpp_laplace_spec_test_bym2(
    y = y, n = n_trials, X = X,
    re_idx = if (n_re_groups > 0L) as.integer(re_idx) else integer(0),
    n_re_groups = n_re_groups, sigma_re = sigma_re,
    spatial_idx = as.integer(spatial_idx), n_spatial_units = K,
    adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
    n_neighbors = adj$n_neighbors,
    scale_factor = scale_factor, sigma_spatial = sigma_spatial, rho = rho,
    sigma_beta = sigma_beta, family = family, phi = phi,
    max_iter = 300L, tol = 1e-11, n_threads = 1L
  )

  expect_true(spec$converged, info = paste0(family, " BYM2: converged"))
  ref_mode <- as.numeric(ref$modes[1, ])
  expect_equal(spec$mode, ref_mode, tolerance = tol,
               info = paste0(family, " BYM2: mode"))

  p <- ncol(X)
  beta <- spec$mode[seq_len(p)]
  tau_beta <- 1 / (sigma_beta * sigma_beta)
  beta_density <- sum(-0.5 * tau_beta * beta^2) + 0.5 * p * log(tau_beta / (2 * pi))
  expect_equal(spec$log_marginal, ref$log_marginal[1] + beta_density,
               tolerance = tol,
               info = paste0(family, " BYM2: log_marginal up to beta density"))
}

test_that("spec BYM2 blocks reproduce the nested BYM2 kernel (no RE)", {
  set.seed(101L)
  K <- 10L
  adj <- build_chain_adj(K)
  N <- 140L
  spatial_idx <- sample.int(K, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  beta <- c(0.1, 0.4)
  field <- as.numeric(scale(sin(seq_len(K)), scale = FALSE))
  eta <- as.numeric(X %*% beta) + field[spatial_idx]
  ones <- rep(1L, N)

  expect_bym2_match("gaussian", eta + rnorm(N, 0, 0.6), ones, X,
                    spatial_idx, K, adj, scale_factor = 1.0,
                    sigma_spatial = 0.8, rho = 0.6, phi = 0.6)
  expect_bym2_match("poisson", rpois(N, exp(eta)), ones, X,
                    spatial_idx, K, adj, scale_factor = 1.0,
                    sigma_spatial = 0.7, rho = 0.5, phi = 1.0)
  expect_bym2_match("binomial", rbinom(N, 10L, plogis(eta)), rep(10L, N), X,
                    spatial_idx, K, adj, scale_factor = 1.0,
                    sigma_spatial = 0.9, rho = 0.4, phi = 1.0)
})

test_that("spec BYM2 blocks reproduce the nested BYM2 kernel (with iid RE)", {
  set.seed(202L)
  K <- 8L
  adj <- build_chain_adj(K)
  N <- 150L
  n_re_groups <- 10L
  sigma_re <- 0.5
  spatial_idx <- sample.int(K, N, replace = TRUE)
  re_idx <- sample.int(n_re_groups, N, replace = TRUE)
  X <- cbind(1, rnorm(N))
  beta <- c(0.15, -0.25)
  field <- as.numeric(scale(cos(seq_len(K)), scale = FALSE))
  u <- rnorm(n_re_groups, sd = sigma_re)
  eta <- as.numeric(X %*% beta) + field[spatial_idx] + u[re_idx]
  ones <- rep(1L, N)

  expect_bym2_match("poisson", rpois(N, exp(eta)), ones, X,
                    spatial_idx, K, adj, scale_factor = 1.0,
                    sigma_spatial = 0.7, rho = 0.5, phi = 1.0,
                    re_idx = re_idx, n_re_groups = n_re_groups, sigma_re = sigma_re)
})
