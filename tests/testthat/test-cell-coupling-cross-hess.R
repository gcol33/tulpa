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

test_that("fisher step-curvature retains the true Gaussian cross-Hessian", {
    # The bivariate Gaussian spec's cross term lam01 is the genuine Fisher
    # off-diagonal -- a Gaussian's Hessian equals its precision, which equals
    # its Fisher information -- NOT a mixture missing-information term. Under
    # control$hessian = "fisher" (Expected step curvature) it must be retained,
    # so the coupled fit reproduces the default observed-Hessian fit to
    # precision. Dropping cross-Hessians is correct only for a mixture spec
    # (occu_cover); a spurious global cross-drop here would shift the mode.
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 2.0, lam11 = 1.5, lam01 = 0.7
    )
    d <- setup_bivariate_data(seed = 11L)

    res_observed <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11)
    )
    res_fisher <- tulpa_nested_laplace_joint(
        responses = list(a = build_arm_biv(d, d$y0),
                         b = build_arm_biv(d, d$y1)),
        prior     = prior_spec_biv(d),
        cell_coupling = "test_bivariate_gaussian",
        control   = list(max_iter = 80L, tol = 1e-11, hessian = "fisher")
    )

    expect_equal(res_fisher$log_marginal, res_observed$log_marginal, tolerance = 1e-9)
    expect_equal(res_fisher$modes,        res_observed$modes,        tolerance = 1e-8)
    expect_equal(res_fisher$theta_mean,   res_observed$theta_mean,   tolerance = 1e-8)
    expect_equal(res_fisher$theta_sd,     res_observed$theta_sd,     tolerance = 1e-8)
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

# --------------------------------------------------------------------------- #
# gcol33/tulpa#86: a shared latent field AND a per-group RE block on the same  #
# predictor of a coupled arm. The multi-block prior machinery composes the     #
# field precision (BYM2) block-diagonally with the iid RE precision; both      #
# enter each arm's eta through their own projection maps and both variance     #
# components integrate on the outer grid. Exercised on the genuinely coupled   #
# cell path (lam01 != 0) so the field+RE assembly is verified alongside the    #
# cross-arm Hessian, not just on the separable per-obs sum.                    #
# --------------------------------------------------------------------------- #

.chain_adj_re <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Build a coupled two-arm dataset whose shared eta carries a BYM2 field over
# spatial units AND an iid effect over groups, plus the multi-block prior that
# stacks both. `grp` maps each spatial unit to one of n_g RE groups.
.setup_field_re <- function(seed, n_s, n_g, sigma_w, sigma_u, adj, grp) {
    set.seed(seed)
    w <- rnorm(n_s, 0, sigma_w); w <- w - mean(w)
    u <- rnorm(n_g, 0, sigma_u)
    eta_a <- 0.3 + w + u[grp]
    eta_b <- -0.2 + w + u[grp]
    y0 <- eta_a + rnorm(n_s, 0, 0.35)
    y1 <- eta_b + rnorm(n_s, 0, 0.35)
    arm <- function(y) list(
        y = y, n_trials = rep(1L, n_s), X = matrix(1, n_s, 1),
        spatial_idx = seq_len(n_s), re_idx = rep(0, n_s),
        n_re_groups = 0L, sigma_re = 1.0, family = "gaussian", phi = 1,
        coupled = TRUE, cell_obs_map = seq_len(n_s))
    prior <- list(
        list(type = "bym2", n_spatial_units = n_s,
             adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
             n_neighbors = adj$n_neighbors, scale_factor = 1.0,
             sigma_grid = seq(0.2, 1.4, length.out = 7L),
             rho_grid   = seq(0.1, 0.95, length.out = 7L),
             spatial_idx = list(seq_len(n_s), seq_len(n_s))),
        list(type = "iid", n_units = n_g,
             sigma_grid = seq(0.2, 1.4, length.out = 7L),
             obs_idx = list(grp, grp)))
    list(responses = list(a = arm(y0), b = arm(y1)), prior = prior)
}

test_that("coupled field + per-group RE: dense and sparse paths agree (#86)", {
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 2.0, lam11 = 1.5, lam01 = 0.6)
    n_s <- 30L; n_g <- 6L
    adj <- .chain_adj_re(n_s)
    grp <- rep(seq_len(n_g), length.out = n_s)
    dat <- .setup_field_re(seed = 21L, n_s = n_s, n_g = n_g,
                           sigma_w = 0.8, sigma_u = 0.7, adj = adj, grp = grp)

    res_dense <- tulpa_nested_laplace_joint(
        responses = dat$responses, prior = dat$prior,
        cell_coupling = "test_bivariate_gaussian",
        control = list(adaptive_grid = FALSE, max_iter = 80L, tol = 1e-11,
                       diagnose_k = FALSE))
    res_sparse <- tulpa_nested_laplace_joint(
        responses = dat$responses, prior = dat$prior,
        cell_coupling = "test_bivariate_gaussian",
        control = list(adaptive_grid = FALSE, max_iter = 80L, tol = 1e-11,
                       diagnose_k = FALSE, force_sparse = TRUE))

    # The combined latent block is assembled: two block moments, a field and a
    # per-group RE, each with a finite variance component.
    expect_length(res_dense$block_moments, 2L)
    expect_identical(res_dense$block_moments[[1]]$type, "bym2")
    expect_identical(res_dense$block_moments[[2]]$type, "iid")
    expect_true(is.finite(res_dense$block_moments[[1]]$mean[["sigma"]]))
    expect_true(is.finite(res_dense$block_moments[[2]]$mean[["sigma"]]))

    # Dense and sparse Cholesky paths must be identical -- the field+RE block
    # assembly cannot diverge between them.
    expect_equal(res_sparse$log_marginal, res_dense$log_marginal, tolerance = 1e-9)
    expect_equal(res_sparse$modes,        res_dense$modes,        tolerance = 1e-8)
    expect_equal(res_sparse$theta_mean,   res_dense$theta_mean,   tolerance = 1e-8)
    expect_equal(res_sparse$theta_sd,     res_dense$theta_sd,     tolerance = 1e-8)
})

test_that("coupled field + per-group RE: the RE variance component is integrated (#86)", {
    skip_on_cran()
    cpp_register_test_bivariate_gaussian_coupling(
        lam00 = 2.0, lam11 = 1.5, lam01 = 0.6)
    n_s <- 60L; n_g <- 10L
    adj <- .chain_adj_re(n_s)
    grp <- rep(seq_len(n_g), length.out = n_s)
    sigma_w <- 0.8; sigma_u <- 0.7

    # field-only fit on the SAME data: the engine's field-variance integration is
    # already correct on its own (this is the reference that the field component
    # is not the thing under test here).
    field_only_prior <- function(dat) dat$prior[1]
    est <- t(vapply(1:8, function(seed) {
        dat <- .setup_field_re(seed = 100L + seed, n_s = n_s, n_g = n_g,
                               sigma_w = sigma_w, sigma_u = sigma_u,
                               adj = adj, grp = grp)
        both <- tulpa_nested_laplace_joint(
            responses = dat$responses, prior = dat$prior,
            cell_coupling = "test_bivariate_gaussian",
            control = list(adaptive_grid = FALSE, max_iter = 80L, tol = 1e-9,
                           diagnose_k = FALSE))
        only <- tulpa_nested_laplace_joint(
            responses = dat$responses, prior = field_only_prior(dat),
            cell_coupling = "test_bivariate_gaussian",
            control = list(adaptive_grid = FALSE, max_iter = 80L, tol = 1e-9,
                           diagnose_k = FALSE))
        c(iid       = both$block_moments[[2]]$mean[["sigma"]],
          field     = both$block_moments[[1]]$mean[["sigma"]],
          field_only = only$block_moments[[1]]$mean[["sigma"]])
    }, numeric(3)))

    # The per-group RE variance is integrated, not dropped: it recovers across
    # seeds. This is the #86 capability -- a second (RE) latent block layered on
    # the same predictor as the field, with its own variance component on the
    # outer grid.
    expect_equal(mean(est[, "iid"]), sigma_u, tolerance = 0.25)
    expect_true(all(est[, "iid"] > 0.2 & is.finite(est[, "iid"])))

    # The field component stays finite and positive when combined with the RE
    # block. Its point estimate is jointly under-identified against the RE in
    # this synthetic bivariate-Gaussian coupling (a smooth field competes with
    # an interleaved per-group effect at n_s = 60), so the combined split is not
    # a recovery target; the field-only fit on the same data recovers sigma_w,
    # confirming the field integration itself is correct.
    expect_true(all(est[, "field"] > 0 & is.finite(est[, "field"])))
    expect_equal(mean(est[, "field_only"]), sigma_w, tolerance = 0.20)
})
