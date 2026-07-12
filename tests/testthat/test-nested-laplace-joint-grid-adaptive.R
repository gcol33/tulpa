# Adaptive-lattice outer integration for the joint multi-block path
# (integration = "grid_adaptive"; nested_laplace_joint_adaptive.R).
#
# The adaptive grid seeds from a coarse subsample, floods outward on the fine
# lattice, and keeps only the cells within a log-density cutoff of the peak. It
# is therefore a mass-concentrated SUBSET of the dense tensor at uniform lattice
# spacing, re-evaluated by the main kernel call with the plain softmax weight.
# The claim these tests pin:
#   * it engages, reports integration = "grid_adaptive", keeps FEWER cells than
#     the dense tensor, and normalises (sum(weights) == 1);
#   * its posterior (hyperparameter means, per-axis medians) MATCHES the dense
#     tensor (integration = "grid") to the cutoff-truncation tolerance -- the
#     omitted cells each carry dense-grid weight < exp(-cutoff);
#   * it is reproducible run to run;
#   * it declines back to the dense tensor when the kept region would rival it.

.chain_adj_ga <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# Two-arm joint (binomial occ + gaussian pos) sharing a smooth structured field,
# so the BYM2 mixing rho and the copy alpha have interior posterior modes and a
# fine tensor grid brackets a genuine peak the adaptive flood can concentrate on.
.sim_joint_ga <- function(seed, N, n_s) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
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
        adj = .chain_adj_ga(n_s),
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

.bym2_copy_block_ga <- function(adj, sigma_grid, rho_grid, sp) {
    list(type = "bym2", spatial_idx = sp,
         n_spatial_units = adj$n_spatial_units,
         adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
         n_neighbors = adj$n_neighbors, scale_factor = 1.0,
         sigma_grid = sigma_grid, rho_grid = rho_grid)
}

.fit_ga <- function(sim, integration, sigma_grid, rho_grid, alpha_grid,
                    extra = list()) {
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block_ga(sim$adj, sigma_grid, rho_grid, sp)
    tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = alpha_grid),
        control = utils::modifyList(
            list(integration = integration, diagnose_k = FALSE,
                 var_of_means_consistency = FALSE), extra))
}

test_that("grid_adaptive engages, keeps fewer cells, and normalises", {
    skip_on_cran()
    sim <- .sim_joint_ga(2024L, N = 1500L, n_s = 50L)
    sg <- c(0.2, 0.5, 0.9, 1.5, 2.5, 4.0)
    rg <- c(0.1, 0.3, 0.5, 0.7, 0.9)
    ag <- c(0.2, 0.5, 0.9, 1.4, 2.0)
    n_dense <- length(sg) * length(rg) * length(ag)   # 150

    fit <- .fit_ga(sim, "grid_adaptive", sg, rg, ag)
    expect_identical(fit$integration, "grid_adaptive")
    expect_true(abs(sum(fit$weights) - 1) < 1e-8)
    expect_true(all(is.finite(fit$theta_mean)))
    # A peaked posterior: the flood must skip a real fraction of the dense grid.
    expect_lt(length(fit$log_marginal), n_dense)
    expect_false(is.null(fit$adaptive_grid_info))
    expect_equal(fit$adaptive_grid_info$n_dense, n_dense)
})

test_that("grid_adaptive matches the dense tensor posterior", {
    skip_on_cran()
    sim <- .sim_joint_ga(2024L, N = 1500L, n_s = 50L)
    sg <- c(0.2, 0.5, 0.9, 1.5, 2.5, 4.0)
    rg <- c(0.1, 0.3, 0.5, 0.7, 0.9)
    ag <- c(0.2, 0.5, 0.9, 1.4, 2.0)

    # The dense reference is a 150-cell tensor -- it legitimately trips the
    # >50-cell grid-size notice; that warning is about the reference, not the
    # feature under test, so suppress it here.
    fit_dense <- suppressWarnings(.fit_ga(sim, "grid",          sg, rg, ag))
    fit_adapt <- .fit_ga(sim, "grid_adaptive", sg, rg, ag)

    # Hyperparameter posterior means agree: the adaptive grid is the dense tensor
    # with the far-tail cells (dense weight < exp(-cutoff)) omitted, so the two
    # weighted means differ only by that truncated mass.
    for (nm in names(fit_dense$theta_mean)) {
        d <- fit_dense$theta_mean[[nm]]; a <- fit_adapt$theta_mean[[nm]]
        expect_equal(a, d, tolerance = 0.02,
                     info = paste("theta_mean axis", nm))
    }
    # Per-axis medians (the reported hyperparameter summary) agree too.
    for (nm in names(fit_dense$theta_median)) {
        expect_equal(fit_adapt$theta_median[[nm]], fit_dense$theta_median[[nm]],
                     tolerance = 0.05, info = paste("theta_median axis", nm))
    }
})

test_that("grid_adaptive is reproducible run to run", {
    skip_on_cran()
    sim <- .sim_joint_ga(7L, N = 1200L, n_s = 45L)
    # 4 x 4 x 3 = 48 cells: at the min-cells floor so the flood is attempted,
    # and reproducible whether it engages or (on this sim) declines to dense.
    sg <- c(0.4, 0.9, 1.6, 2.8); rg <- c(0.2, 0.45, 0.65, 0.85); ag <- c(0.3, 0.8, 1.5)
    f1 <- .fit_ga(sim, "grid_adaptive", sg, rg, ag)
    f2 <- .fit_ga(sim, "grid_adaptive", sg, rg, ag)
    expect_equal(length(f1$log_marginal), length(f2$log_marginal))
    expect_equal(sort(f1$log_marginal), sort(f2$log_marginal), tolerance = 1e-10)
    expect_equal(f1$theta_mean, f2$theta_mean, tolerance = 1e-10)
})

test_that("grid_adaptive folds a phi axis into the lattice and matches dense", {
    skip_on_cran()
    # 4-D outer lattice: sigma, rho, alpha (latent) + phi_pos (Beta/gaussian
    # dispersion). The phi axis is crossed into the SAME adaptive flood here (not
    # a tensor cross on top), so this pins that the phi column folds correctly on
    # the engaged path and the fit matches the dense tensor.
    sim <- .sim_joint_ga(3L, N = 2200L, n_s = 55L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block_ga(sim$adj, c(0.3, 0.6, 1.0, 1.7, 2.8),
                               c(0.2, 0.5, 0.8), sp)
    phi_axis <- exp(seq(log(0.15), log(1.5), length.out = 5))
    mk <- function(integ) tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.4, 0.8, 1.3, 1.9)),
        phi_grid = list(pos = phi_axis),
        control = list(integration = integ, diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    fit_dense <- suppressWarnings(mk("grid"))
    fit_adapt <- mk("grid_adaptive")

    expect_identical(fit_adapt$integration, "grid_adaptive")
    expect_true("phi_pos" %in% colnames(fit_adapt$theta_grid))
    expect_lt(length(fit_adapt$log_marginal), length(fit_dense$log_marginal))
    expect_true(abs(sum(fit_adapt$weights) - 1) < 1e-8)
    for (nm in names(fit_dense$theta_mean)) {
        expect_equal(fit_adapt$theta_mean[[nm]], fit_dense$theta_mean[[nm]],
                     tolerance = 0.03, info = paste("phi-fold theta_mean", nm))
    }
})

test_that("grid_adaptive works at 2 latent axes (ICAR sigma + copy alpha)", {
    skip_on_cran()
    sim <- .sim_joint_ga(11L, N = 1400L, n_s = 48L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    # ICAR block: sigma only (no rho) -> 2 latent axes with the copy alpha.
    # 10 x 6 = 60 cells, above adaptive_grid_min_cells so the flood engages.
    blk <- list(type = "icar", spatial_idx = sp,
                n_spatial_units = sim$adj$n_spatial_units,
                adj_row_ptr = sim$adj$adj_row_ptr,
                adj_col_idx = sim$adj$adj_col_idx,
                n_neighbors = sim$adj$n_neighbors, scale_factor = 1.0,
                sigma_grid = exp(seq(log(0.3), log(3.5), length.out = 10)))
    ctrl <- list(integration = NULL, diagnose_k = FALSE,
                 var_of_means_consistency = FALSE)
    mk <- function(integ) tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = c(0.3, 0.6, 0.9, 1.2, 1.5, 1.8)),
        control = utils::modifyList(ctrl, list(integration = integ)))
    fit_dense <- suppressWarnings(mk("grid"))   # 60-cell dense reference (>50 notice)
    fit_adapt <- mk("grid_adaptive")
    expect_identical(fit_adapt$integration, "grid_adaptive")
    expect_lt(length(fit_adapt$log_marginal), length(fit_dense$log_marginal))
    for (nm in names(fit_dense$theta_mean)) {
        expect_equal(fit_adapt$theta_mean[[nm]], fit_dense$theta_mean[[nm]],
                     tolerance = 0.03, info = paste("2-axis theta_mean", nm))
    }
})

test_that("grid_adaptive declines to the dense tensor on a small outer grid", {
    skip_on_cran()
    sim <- .sim_joint_ga(5L, N = 900L, n_s = 40L)
    # 2 x 2 x 2 = 8 cells, well below adaptive_grid_min_cells (48): the builder
    # must decline BEFORE any inner solve and hand back the dense tensor, so the
    # small-grid case never pays the coarse-seed overhead.
    fit <- .fit_ga(sim, "grid_adaptive", c(0.5, 1.5), c(0.4, 0.8), c(0.6, 1.2))
    expect_identical(fit$integration, "grid")     # declined to the tensor
    expect_null(fit$adaptive_grid_info)
    expect_equal(length(fit$log_marginal), 2L * 2L * 2L)
    # And the same fit forced dense is identical (decline is a pure no-op).
    fit_grid <- .fit_ga(sim, "grid", c(0.5, 1.5), c(0.4, 0.8), c(0.6, 1.2))
    expect_equal(sort(fit$log_marginal), sort(fit_grid$log_marginal),
                 tolerance = 1e-10)
})
