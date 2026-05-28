# Recovery / regression test for the cell-coupling per-cell branch
# (gcol33/tulpa#32 Layer B.1). Compares the per-cell branch's mode +
# log-marginal to the existing per-obs scatter for the same data.
#
# The test spec `TestSeparableBernoulliCoupling` reproduces single-arm
# binomial (n_trials = 1) as a CellCouplingSpec with one row per cell,
# so the two paths should agree to numerical precision (no summation
# order differences -- each cell has exactly one row).

setup_test_data <- function(seed = 1L, n_s = 12L, N = 60L) {
    set.seed(seed)
    adj_row_ptr <- c(0L, 1L,
                     3L, 4L, 6L, 8L, 9L, 11L, 13L, 14L, 16L, 18L, 19L, 20L)
    adj_col_idx <- c(2L,
                     1L, 3L, 2L,
                     3L, 5L, 4L, 6L, 5L,
                     7L, 9L, 8L, 10L, 9L,
                     11L, 13L, 12L,
                     11L, 12L) - 1L
    n_neighbors <- c(1L, 2L, 1L, 2L, 2L, 1L, 2L, 2L, 1L, 2L, 2L, 1L)
    spatial_idx <- as.integer(rep(seq_len(n_s), length.out = N))
    eta_true <- 0.3 + rnorm(n_s, 0, 0.6)
    p_true   <- 1 / (1 + exp(-eta_true[spatial_idx]))
    y <- rbinom(N, 1, p_true)
    list(
        n_s = n_s, adj_row_ptr = adj_row_ptr,
        adj_col_idx = adj_col_idx, n_neighbors = n_neighbors,
        spatial_idx = spatial_idx,
        y = y, N = N
    )
}

build_arm <- function(d, coupled) {
    arm <- list(
        y           = d$y,
        n_trials    = rep(1L, d$N),
        X           = matrix(1, nrow = d$N, ncol = 1),
        spatial_idx = d$spatial_idx,
        family      = "binomial",
        phi         = 1
    )
    if (coupled) {
        arm$coupled      <- TRUE
        arm$cell_obs_map <- seq_len(d$N)
    }
    arm
}

prior_spec <- function(d) {
    list(type = "icar",
         n_spatial_units = d$n_s,
         adj_row_ptr     = d$adj_row_ptr,
         adj_col_idx     = d$adj_col_idx,
         n_neighbors     = d$n_neighbors,
         sigma_grid      = c(0.4, 0.8, 1.5))
}

test_that("per-cell branch matches per-obs path: mode and log-marginal", {
    cpp_register_test_separable_bernoulli_coupling()
    expect_true(cpp_cell_coupling_registry_has("test_separable_bernoulli"))

    d <- setup_test_data(seed = 2L)

    res_baseline <- tulpa_nested_laplace_joint(
        responses = list(occ = build_arm(d, coupled = FALSE)),
        prior     = prior_spec(d),
        control   = list(max_iter = 60L, tol = 1e-10)
    )

    res_coupled <- tulpa_nested_laplace_joint(
        responses     = list(occ = build_arm(d, coupled = TRUE)),
        prior         = prior_spec(d),
        cell_coupling = "test_separable_bernoulli",
        control       = list(max_iter = 60L, tol = 1e-10)
    )

    expect_equal(res_coupled$log_marginal,
                 res_baseline$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_coupled$modes,
                 res_baseline$modes,
                 tolerance = 1e-8)
    expect_equal(res_coupled$theta_mean,
                 res_baseline$theta_mean,
                 tolerance = 1e-8)
    expect_equal(res_coupled$theta_sd,
                 res_baseline$theta_sd,
                 tolerance = 1e-8)
})

test_that("kernel rejects coupled = TRUE with the separable default spec", {
    d <- setup_test_data(seed = 3L, N = 20L)
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = build_arm(d, coupled = TRUE)),
            prior     = prior_spec(d),
            cell_coupling = "separable",
            control   = list(max_iter = 5L)
        ),
        regexp = "coupled = TRUE"
    )
})

test_that("kernel rejects spec arm_ids() that disagree with arms' coupled flags", {
    cpp_register_test_separable_bernoulli_coupling()
    d <- setup_test_data(seed = 4L, N = 20L)
    expect_error(
        tulpa_nested_laplace_joint(
            responses     = list(occ = build_arm(d, coupled = FALSE)),
            prior         = prior_spec(d),
            cell_coupling = "test_separable_bernoulli",
            control       = list(max_iter = 5L)
        ),
        regexp = "spec lists arm 1"
    )
})
