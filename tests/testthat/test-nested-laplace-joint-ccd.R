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
                       family = "gaussian", phi = 0.25)))
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

test_that("verbose announces the engaged outer integrator at selection time (gcol33/tulpa#63)", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    copy_spec <- list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2))
    base_ctrl <- list(diagnose_k = FALSE, var_of_means_consistency = FALSE,
                      verbose = TRUE)

    # 3 axes with integration = "ccd": the CCD engages and is announced.
    expect_message(
        tulpa_nested_laplace_joint(
            sim$responses, list(blk), copy = copy_spec,
            control = c(base_ctrl, list(integration = "ccd"))),
        "outer integration: CCD \\(3 latent axes")

    # integration = "grid" forces the tensor product; the line names it.
    expect_message(
        tulpa_nested_laplace_joint(
            sim$responses, list(blk), copy = copy_spec,
            control = c(base_ctrl, list(integration = "grid"))),
        "outer integration: tensor grid \\([0-9]+ cells\\)")
})

test_that("CCD matches a fine tensor grid in the tighter-posterior regime", {
    skip_on_cran()
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

test_that("the auto default uses the tensor grid at 3 axes (gcol33/tulpa#59)", {
    skip_on_cran()
    # No `integration` in control -> "auto", which keeps the cheaper, more
    # ridge-robust tensor grid at <= 3 axes and reserves CCD for >= 4 (where the
    # k^d blow-up bites hardest). A single BYM2 copy block is 3 axes (sigma,
    # alpha, rho), so the default integrates on the 3 * 3 * 3 = 27 tensor cells.
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(diagnose_k = FALSE, var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "grid")
    expect_equal(length(fit$log_marginal), 27L)
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

# --------------------------------------------------------------------------- #
# The decline is recorded on the fit (gcol33/tulpa#315)                        #
# --------------------------------------------------------------------------- #
#
# `fit$integration` names the integrator that RAN, and `.nl_node_support()` keys
# the interval construction off it, so "grid" cannot distinguish a tensor grid
# the caller chose from one a declined CCD fell back to. Each case below asserts
# the decline FIRED (or did not) before reading the reason, so none can pass
# vacuously on a fit that took the other branch.

test_that("an engaged CCD records no decline", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "ccd")
    expect_identical(fit$integration_requested, "ccd")
    expect_true(is.na(fit$integration_declined))
})

test_that("a tensor grid nobody asked to replace records no decline", {
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
    expect_identical(fit$integration_requested, "grid")
    expect_true(is.na(fit$integration_declined))
})

test_that("an unguessable axis records why the CCD declined", {
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
    # The decline fired: 3 axes were requested as a CCD and a tensor grid ran.
    expect_identical(fit$integration_requested, "ccd")
    expect_false(identical(fit$integration, "ccd"))
    # rho_car's support is the adjacency eigenvalue interval, the one axis the
    # transform registry will not guess.
    expect_identical(fit$integration_declined, "unguessable_axis")
})

test_that("a CCD below its axis threshold records the axis count", {
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
    expect_identical(fit$integration_requested, "ccd")
    expect_false(identical(fit$integration, "ccd"))
    expect_identical(fit$integration_declined, "axis_count")
})

test_that("every recorded decline reason is one of the named ones", {
    expect_true(all(c("axis_count", "unguessable_axis", "degenerate_axis",
                      "modefind_ridge", "modefind_boundary",
                      "modefind_degenerate", "modefind_failed",
                      "hessian_singular", "hessian_not_pd",
                      "copy_atom_components", "copy_atom_mass") %in%
                    tulpa:::.CCD_DECLINE_REASONS))
})

test_that("the CCD designs a copy scale on its declared log coordinate", {
    expect_identical(
        tulpa:::.joint_ccd_coord_tags(c("b1.sigma", "b1.alpha", "phi_pos"),
                                      c("log", "identity", "log")),
        c("log", "log", "log"))
    # Only a copy scale whose levels carry the "no coupling" node splits the
    # design; one bracketing the continuum alone stays a single component.
    expect_identical(
        tulpa:::.joint_ccd_atom_axes(c("b1.sigma", "b1.alpha", "b2.alpha"),
                                     list(c(0.5, 1), c(0, 0.5), c(0.5, 1))),
        2L)
    expect_identical(
        tulpa:::.joint_ccd_atom_axes(c("sigma", "rho"), list(c(0.5, 1), c(0, 1))),
        integer(0))
})

test_that("a copy atom splits the CCD into one design per coupling configuration", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0, 0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "ccd")
    acol <- grep("alpha$", colnames(fit$theta_grid), value = TRUE)
    expect_length(acol, 1L)
    al <- as.numeric(fit$theta_grid[, acol])
    # The design coordinate is log alpha, so no node leaves the support.
    expect_true(all(al >= 0))
    # The "no coupling" atom is reachable, which a single affine design over
    # the three axes never places.
    expect_true(any(al == 0))
    # 3-axis continuum design (1 + 2*3 + 2^3) plus the 2-axis design at
    # alpha = 0 (1 + 2*2 + 2^2).
    expect_identical(nrow(fit$theta_grid), 24L)
    # Each component carries the prior mass its configuration declares, so the
    # atom's design weights sum to the declared atom mass and the whole design
    # to one.
    expect_equal(sum(fit$dnode), 1, tolerance = 1e-10)
    expect_equal(sum(fit$dnode[al == 0]), tulpa:::.TULPA_COPY_ATOM_MASS,
                 tolerance = 1e-10)
})

test_that("a split CCD matches a fine tensor grid carrying the same atom", {
    skip_on_cran()
    # The mixture the split integrates is the one the tensor rule integrates as
    # levels, so the two agree on the posterior of the copy scale -- including
    # the share of it sitting on "no coupling", which the unsplit design could
    # not reach at all.
    sim <- .sim_joint_ccd(7L, N = 4000L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)

    fit_ccd <- tulpa_nested_laplace_joint(
        sim$responses,
        list(.bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = c(0, 0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit_ccd$integration, "ccd")

    fine <- .bym2_copy_block(sim$adj,
                             exp(seq(log(0.2), log(2.0), length.out = 7)),
                             seq(0.05, 0.95, length.out = 7), sp)
    fit_fine <- suppressWarnings(tulpa_nested_laplace_joint(
        sim$responses, list(fine),
        copy = list(arm = "pos", block = 1L,
                    alpha_grid = c(0, seq(0.1, 2.0, length.out = 7))),
        control = list(integration = "grid", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE)))

    rel <- function(nm) abs(fit_ccd$theta_mean[[nm]] - fit_fine$theta_mean[[nm]]) /
                        max(abs(fit_fine$theta_mean[[nm]]), 0.1)
    expect_lt(rel("b1.sigma"), 0.12)
    expect_lt(rel("b1.alpha"), 0.12)

    # The atom is reachable: the split places cells at alpha = 0 and gives them
    # the prior mass they declare, so the two rules agree on how much posterior
    # ends up there. This fixture couples strongly, so both send ~0 to "no
    # coupling" -- which the unsplit design could not report, having never
    # evaluated it.
    at <- function(f) {
        a <- as.numeric(f$theta_grid[, grep("alpha$", colnames(f$theta_grid))])
        list(prior = sum((f$dnode %||% rep(1, length(a)))[a == 0]),
             post  = sum(f$weights[a == 0]))
    }
    expect_gt(at(fit_ccd)$prior, 0)
    expect_lt(abs(at(fit_ccd)$post - at(fit_fine)$post), 0.15)
})

test_that("a split CCD keeps a usable outer Pareto-k on its grid proposal", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0, 0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = TRUE,
                       var_of_means_consistency = FALSE))
    expect_identical(fit$integration, "ccd")
    # The mode-Hessian splice is withheld on a split design, so the diagnostic
    # reports the grid-built proposal it actually used rather than one that
    # silently did not take.
    expect_false(identical(fit$pareto_k_proposal_source, "mode_hessian"))
    expect_true(fit$pareto_k_proposal_source %in%
                c("grid_moment", "grid_mixture"))
    expect_true(is.finite(fit$pareto_k))
})

test_that("a copy atom's design mass follows control$copy_atom_mass", {
    skip_on_cran()
    sim <- .sim_joint_ccd(2024L, N = 800L, n_s = 40L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- .bym2_copy_block(sim$adj, c(0.3, 0.6, 1.0), c(0.3, 0.7, 0.9), sp)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0, 0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       var_of_means_consistency = FALSE,
                       copy_atom_mass = 0.2))
    expect_identical(fit$integration, "ccd")
    acol <- grep("alpha$", colnames(fit$theta_grid), value = TRUE)
    al <- as.numeric(fit$theta_grid[, acol])
    expect_equal(sum(fit$dnode[al == 0]), 0.2, tolerance = 1e-10)
})
