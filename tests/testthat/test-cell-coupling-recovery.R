# Recovery / regression test for the cell-coupling per-cell branch
# (gcol33/tulpa#32 Layer B.1). Compares the per-cell branch's mode +
# log-marginal to the existing per-obs scatter for the same data.
#
# The test spec `TestSeparableBernoulliCoupling` reproduces single-arm
# binomial (n_trials = 1) as a CellCouplingSpec with one row per cell,
# so the two paths should agree to numerical precision (no summation
# order differences -- each cell has exactly one row).

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

setup_test_data <- function(seed = 1L, n_s = 12L, N = 60L) {
    set.seed(seed)
    adj <- .chain_adj(n_s)
    spatial_idx <- as.integer(rep(seq_len(n_s), length.out = N))
    eta_true <- 0.3 + rnorm(n_s, 0, 0.6)
    p_true   <- 1 / (1 + exp(-eta_true[spatial_idx]))
    y <- rbinom(N, 1, p_true)
    list(
        n_s = n_s, adj = adj,
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
    list(type            = "icar",
         n_spatial_units = d$adj$n_spatial_units,
         adj_row_ptr     = d$adj$adj_row_ptr,
         adj_col_idx     = d$adj$adj_col_idx,
         n_neighbors     = d$adj$n_neighbors,
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

test_that("fisher step-curvature reproduces the observed-Hessian fit (Bernoulli is curvature-invariant)", {
    # The single-arm Bernoulli per-cell spec writes the canonical-link Fisher
    # weight p(1-p) as its curvature, so observed == expected information
    # cell-by-cell (no latent mixture, no missing-information term).
    # control$hessian = "fisher" (Expected step curvature) must therefore
    # reproduce the default observed-Hessian fit to numerical precision -- the
    # fisher path is a correct no-op for a curvature-invariant spec.
    cpp_register_test_separable_bernoulli_coupling()
    d <- setup_test_data(seed = 2L)

    res_observed <- tulpa_nested_laplace_joint(
        responses     = list(occ = build_arm(d, coupled = TRUE)),
        prior         = prior_spec(d),
        cell_coupling = "test_separable_bernoulli",
        control       = list(max_iter = 60L, tol = 1e-10)
    )
    res_fisher <- tulpa_nested_laplace_joint(
        responses     = list(occ = build_arm(d, coupled = TRUE)),
        prior         = prior_spec(d),
        cell_coupling = "test_separable_bernoulli",
        control       = list(max_iter = 60L, tol = 1e-10, hessian = "fisher")
    )

    expect_equal(res_fisher$log_marginal, res_observed$log_marginal, tolerance = 1e-9)
    expect_equal(res_fisher$modes,        res_observed$modes,        tolerance = 1e-8)
    expect_equal(res_fisher$theta_mean,   res_observed$theta_mean,   tolerance = 1e-8)
    expect_equal(res_fisher$theta_sd,     res_observed$theta_sd,     tolerance = 1e-8)
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

test_that("sparse per-cell branch matches dense per-obs baseline (B.2)", {
    cpp_register_test_separable_bernoulli_coupling()
    d <- setup_test_data(seed = 5L)

    res_dense <- tulpa_nested_laplace_joint(
        responses = list(occ = build_arm(d, coupled = FALSE)),
        prior     = prior_spec(d),
        control   = list(max_iter = 60L, tol = 1e-10)
    )

    res_sparse_coupled <- tulpa_nested_laplace_joint(
        responses     = list(occ = build_arm(d, coupled = TRUE)),
        prior         = prior_spec(d),
        cell_coupling = "test_separable_bernoulli",
        control       = list(max_iter = 60L, tol = 1e-10,
                             force_sparse = TRUE)
    )

    expect_equal(res_sparse_coupled$log_marginal,
                 res_dense$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_sparse_coupled$modes,
                 res_dense$modes,
                 tolerance = 1e-8)
    expect_equal(res_sparse_coupled$theta_mean,
                 res_dense$theta_mean,
                 tolerance = 1e-8)
    expect_equal(res_sparse_coupled$theta_sd,
                 res_dense$theta_sd,
                 tolerance = 1e-8)
})

test_that("sparse separable default stays byte-identical under cell_coupling", {
    d <- setup_test_data(seed = 6L)

    res_sparse <- tulpa_nested_laplace_joint(
        responses = list(occ = build_arm(d, coupled = FALSE)),
        prior     = prior_spec(d),
        control   = list(max_iter = 60L, tol = 1e-10,
                         force_sparse = TRUE)
    )
    res_dense <- tulpa_nested_laplace_joint(
        responses = list(occ = build_arm(d, coupled = FALSE)),
        prior     = prior_spec(d),
        control   = list(max_iter = 60L, tol = 1e-10)
    )

    expect_equal(res_sparse$log_marginal,
                 res_dense$log_marginal,
                 tolerance = 1e-9)
    expect_equal(res_sparse$theta_mean,
                 res_dense$theta_mean,
                 tolerance = 1e-8)
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
