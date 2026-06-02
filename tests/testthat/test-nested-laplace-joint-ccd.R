# CCD outer integration for the joint multi-block path (gcol33/tulpa#59).
#
# CCD places a central composite design around the joint hyperparameter mode
# for >= 3 transformable axes -- far fewer inner solves than the tensor product
# -- and falls back to the tensor grid for <= 2 axes or an unguessable axis.
# These tests cover: (1) the engage path produces the right node count and a
# posterior that matches a fine tensor grid (the reference integration) in the
# tighter-posterior regime where CCD is accurate; (2) the decline paths
# (CAR_proper rho_car, < 3 axes) fall back to the tensor grid.

.chain_adj_ccd <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Two-arm joint (binomial occ + gaussian pos) sharing a smooth (structured)
# field so the BYM2 mixing rho has an interior posterior mode.
.sim_joint_ccd <- function(seed, N, n_s) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))   # structured
    theta <- rnorm(n_s)
    w_s   <- 0.9 * (sqrt(0.7) * phi + sqrt(0.3) * theta)
    x     <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]; spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)
    list(
        adj = .chain_adj_ccd(n_s),
        responses = list(
            occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                       spatial_idx = spatial_idx, re_idx = rep(0, N),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "binomial", phi = 1.0),
            pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                       spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)),
                       n_re_groups = 0L, sigma_re = 1.0,
                       family = "gaussian", phi = 0.5)))
}

.bym2_copy_block <- function(adj, sigma_grid, rho_grid, sp) {
    list(type = "bym2", spatial_idx = sp,
         n_spatial_units = adj$n_spatial_units,
         adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
         n_neighbors = adj$n_neighbors, scale_factor = 1.0,
         sigma_grid = sigma_grid, rho_grid = rho_grid)
}

test_that("CCD engages for a 3-axis BYM2 copy block and counts nodes", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    # 3 axes (sigma, alpha, rho) -> CCD with 1 + 2*3 + 2^3 = 15 nodes.
    expect_identical(fit$integration, "ccd")
    expect_equal(length(fit$log_marginal), 1L + 2L * 3L + 2L^3L)
    expect_true(all(is.finite(fit$theta_mean)))
    expect_true(abs(sum(fit$weights) - 1) < 1e-8)
})

test_that("CCD matches a fine tensor grid in the tighter-posterior regime", {
    skip_on_cran()
    skip_if_fast()
    # Larger N -> tighter, more Gaussian hyperparameter posterior, the regime
    # CCD is designed for (large spatial fields). CCD then reproduces the fine
    # tensor grid's posterior means for the well-identified field amplitude
    # (sigma) and copy coefficient (alpha). The BYM2 mixing weight rho stays
    # right-skewed even at this N, so its posterior MEAN is the known weak spot
    # of a Gaussian central-composite design (INLA's CCD shares this); it is
    # checked against a looser band, not the few-percent the regular axes hit.
    sim <- .sim_joint_ccd(7L, N = 4000L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    copy_spec <- list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2))

    fit_ccd <- tulpa_nested_laplace_joint(
        sim$responses,
        list(.bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)),
        copy = copy_spec,
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit_ccd$integration, "ccd")

    fine <- .bym2_copy_block(sim$adj,
                             exp(seq(log(0.2), log(2.0), length.out = 7)),
                             seq(0.05, 0.95, length.out = 7), sp)
    fit_fine <- suppressWarnings(tulpa_nested_laplace_joint(
        sim$responses, list(fine),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = seq(0.1, 2.0, length.out = 7)),
        control = list(integration = "grid", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE)))

    rel <- function(nm) abs(fit_ccd$theta_mean[[nm]] - fit_fine$theta_mean[[nm]]) /
                        max(abs(fit_fine$theta_mean[[nm]]), 0.1)
    expect_lt(rel("b1.sigma"), 0.12)
    expect_lt(rel("b1.alpha"), 0.12)
    expect_lt(rel("b1.rho"),   0.30)
})

test_that("CCD declines to the tensor grid on an unguessable axis (CAR_proper)", {
    skip_on_cran()
    sim <- .sim_joint_ccd(99L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    car <- list(type = "car_proper", spatial_idx = sp,
                n_spatial_units = sim$adj$n_spatial_units,
                adj_row_ptr = sim$adj$adj_row_ptr,
                adj_col_idx = sim$adj$adj_col_idx,
                n_neighbors = sim$adj$n_neighbors,
                sigma_grid = c(0.5, 1.0), rho_car_grid = c(0.5, 0.9))
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(car),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    # rho_car support is the adjacency eigenvalue interval -> not safely
    # transformable -> CCD declines, tensor grid runs (2 * 3 * 2 = 12 cells).
    expect_identical(fit$integration, "grid")
    expect_equal(length(fit$log_marginal), 12L)
})

test_that("CCD declines for <= 2 transformable axes (single ICAR copy)", {
    skip_on_cran()
    sim <- .sim_joint_ccd(5L, N = 600L, n_s = 30L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    icar <- list(type = "icar", spatial_idx = sp,
                 n_spatial_units = sim$adj$n_spatial_units,
                 adj_row_ptr = sim$adj$adj_row_ptr,
                 adj_col_idx = sim$adj$adj_col_idx,
                 n_neighbors = sim$adj$n_neighbors,
                 sigma_grid = c(0.4, 0.8, 1.2))
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(icar),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.5, 1.0, 1.5)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    # 2 axes (sigma, alpha) -> below the CCD threshold -> tensor (3 * 3 = 9).
    expect_identical(fit$integration, "grid")
    expect_equal(length(fit$log_marginal), 9L)
})

test_that("CCD rides the latent axes and crosses an active phi tensor (gcol33/tulpa#61)", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    phi_axis <- c(0.4, 0.6)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        phi_grid = list(pos = phi_axis),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    # An active phi axis no longer disables CCD: 15 latent CCD nodes
    # (1 + 2*3 + 2^3) crossed with the 2-point phi tensor = 30 cells.
    expect_identical(fit$integration, "ccd")
    expect_equal(length(fit$log_marginal), (1L + 2L * 3L + 2L^3L) * length(phi_axis))
    expect_true("phi_pos" %in% names(fit$theta_mean))
    expect_true(all(is.finite(fit$theta_mean)))
    expect_true(abs(sum(fit$weights) - 1) < 1e-8)
    # The integrated phi sits inside the supplied tensor support.
    expect_gte(fit$theta_mean[["phi_pos"]], min(phi_axis))
    expect_lte(fit$theta_mean[["phi_pos"]], max(phi_axis))
})

test_that("CCD x phi matches the tensor grid x phi in the tighter-posterior regime", {
    skip_on_cran()
    skip_if_fast()
    sim <- .sim_joint_ccd(7L, N = 4000L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    phi_axis <- exp(seq(log(0.3), log(0.8), length.out = 4))

    fit_ccd <- tulpa_nested_laplace_joint(
        sim$responses,
        list(.bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        phi_grid = list(pos = phi_axis),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit_ccd$integration, "ccd")

    fine <- .bym2_copy_block(sim$adj,
                             exp(seq(log(0.2), log(2.0), length.out = 7)),
                             seq(0.05, 0.95, length.out = 7), sp)
    fit_grid <- suppressWarnings(tulpa_nested_laplace_joint(
        sim$responses, list(fine),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = seq(0.1, 2.0, length.out = 7)),
        phi_grid = list(pos = phi_axis),
        control = list(integration = "grid", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE)))
    expect_identical(fit_grid$integration, "grid")

    rel <- function(nm) abs(fit_ccd$theta_mean[[nm]] - fit_grid$theta_mean[[nm]]) /
                        max(abs(fit_grid$theta_mean[[nm]]), 0.1)
    expect_lt(rel("b1.sigma"), 0.15)
    expect_lt(rel("b1.alpha"), 0.15)
    expect_lt(rel("phi_pos"),  0.15)
})

test_that("integration = 'grid' forces the tensor product even for >= 3 axes", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(integration = "grid", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "grid")
    expect_equal(length(fit$log_marginal), 27L)   # 3 * 3 * 3
})
