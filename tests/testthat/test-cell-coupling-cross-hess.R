# Cross-arm Hessian regression for the per-cell branch
# (gcol33/tulpa#32 Layer B.2 cross_hess). Uses a 2-arm bivariate
# Gaussian CellCouplingSpec with a fixed precision matrix Lambda so
# the off-diagonal entry of `arm_cross_hess` exercises the
# scatter_cross_chain_{dense,sparse} helpers.
#
# Per-cell joint density at cell c with one row per arm:
#   log p_cell = -0.5 (eta_c - y_c)' Lambda (eta_c - y_c) + const
# Mode (no priors on beta, single ICAR field z, single beta intercept
# per arm) solves a 2 x (n_s + 2) linear system. With Lambda non-
# diagonal, the joint Hessian wires the (beta_0, beta_1) and
# (beta_0, z[c]), (beta_1, z[c]) cross-arm cells.

.chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx = as.integer(unlist(nbr)) - 1L,
        n_neighbors = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

setup_bivariate_data <- function(seed = 1L, n_s = 10L) {
    set.seed(seed)
    adj <- .chain_adj(n_s)
    spatial_idx <- seq_len(n_s)
    z_true <- rnorm(n_s, 0, 0.5)
    y0 <- 0.2 + z_true + rnorm(n_s, 0, 0.4)
    y1 <- -0.1 + 0.8 * z_true + rnorm(n_s, 0, 0.4)
    list(
        n_s = n_s, adj = adj,
        spatial_idx = spatial_idx,
        y0 = y0, y1 = y1
    )
}

build_arm_biv <- function(d, y_vec) {
    list(
        y           = y_vec,
        n_trials    = rep(1L, d$n_s),
        X           = matrix(1, nrow = d$n_s, ncol = 1),
        spatial_idx = d$spatial_idx,
        family      = "gaussian",
        phi         = 1,
        coupled     = TRUE,
        cell_obs_map = seq_len(d$n_s)
    )
}

prior_spec_biv <- function(d) {
    list(type            = "icar",
         n_spatial_units = d$adj$n_spatial_units,
         adj_row_ptr     = d$adj$adj_row_ptr,
         adj_col_idx     = d$adj$adj_col_idx,
         n_neighbors     = d$adj$n_neighbors,
         sigma_grid      = c(0.4, 0.8, 1.5))
}

test_that("dense and sparse paths agree under non-zero cross_hess", {
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 2.0, lam11 = 1.5, lam01 = 0.7
    )
    expect_true(cpp_cell_coupling_registry_has("test_bivariate_gaussian"))

    d <- setup_bivariate_data(seed = 11L)

    res_dense <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11)
    )

    res_sparse <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11,
                         force_sparse = TRUE)
    )

    expect_equal(res_sparse$log_marginal,
                 res_dense$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_sparse$modes,
                 res_dense$modes,
                 tolerance = 1e-8)
    expect_equal(res_sparse$theta_mean,
                 res_dense$theta_mean,
                 tolerance = 1e-8)
    expect_equal(res_sparse$theta_sd,
                 res_dense$theta_sd,
                 tolerance = 1e-8)
})

test_that("cross_hess scatter reduces to independent fits when lam01 = 0", {
    # lam01 = 0 -> the joint density factorises into two independent
    # Gaussian likelihoods per arm; the coupled fit must agree with two
    # separate uncoupled fits on the same data (up to FP noise).
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 2.0, lam11 = 1.5, lam01 = 0.0
    )

    d <- setup_bivariate_data(seed = 12L)

    res_coupled <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11)
    )

    res_coupled_sparse <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11,
                         force_sparse = TRUE)
    )

    expect_equal(res_coupled_sparse$log_marginal,
                 res_coupled$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_coupled_sparse$modes,
                 res_coupled$modes,
                 tolerance = 1e-8)
})

test_that("opposite-sign cross_hess (lam01 < 0) still gives consistent dense/sparse", {
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 3.0, lam11 = 2.0, lam01 = -1.1
    )

    d <- setup_bivariate_data(seed = 13L)

    res_dense <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11)
    )

    res_sparse <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11,
                         force_sparse = TRUE)
    )

    expect_equal(res_sparse$log_marginal,
                 res_dense$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_sparse$modes,
                 res_dense$modes,
                 tolerance = 1e-8)
})
