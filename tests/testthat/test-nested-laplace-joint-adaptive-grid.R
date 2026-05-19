# Recovery test for the adaptive alpha-grid refinement path added to
# `tulpa_nested_laplace_joint()`. The default is `adaptive_grid = FALSE`;
# this file is the only place that exercises the `TRUE` path end-to-end.
#
# The test rig deliberately gives the user an `alpha_grid` whose maximum
# (0.6) sits below `alpha_true = 2.0`. With `adaptive_grid = FALSE` the
# joint posterior is truncated at the boundary and the alpha mean collapses
# at the largest grid value. With `adaptive_grid = TRUE` the refinement
# helper detects boundary mass on the alpha axis, appends densification +
# outward extension points, and re-evaluates the kernel on those new
# triples. The resulting posterior shifts outward and the
# `adaptive_grid_info` field documents the action.

# --------------------------------------------------------------------------- #
# Helpers (chain ICAR adjacency + joint sim — mirrors -joint-icar)            #
# --------------------------------------------------------------------------- #

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

.simulate_joint_icar_strong <- function(N = 600, n_s = 50, sigma = 1.0,
                                         alpha_true = 2.0,
                                         beta_occ = c(-0.3, 0.5),
                                         beta_pos = c(0.2, -0.4),
                                         sd_pos = 0.3, seed = 6) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(rnorm(n_s, 0, sigma / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    x <- rnorm(N)
    Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% beta_occ) + phi_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% beta_pos) + alpha_true * phi_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, sd_pos)
    list(N = N, n_s = n_s,
         spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
         truth = list(sd_pos = sd_pos, alpha = alpha_true))
}

.fit_joint_icar <- function(sim, alpha_grid, adaptive_grid) {
    adj <- .chain_adj(sim$n_s)
    arm_occ <- list(
        y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
        X = sim$Xocc, spatial_idx = sim$spatial_idx,
        re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
        family = "binomial", phi = 1.0
    )
    arm_pos <- list(
        y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
        X = sim$Xpos, spatial_idx = sim$spi_pos,
        re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L, sigma_re = 1.0,
        family = "gaussian", phi = sim$truth$sd_pos
    )
    prior <- list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid = c(0.6, 1.0, 1.5)
    )
    # var_of_means_consistency adds its own slice cells (gcol33/tulpa#21);
    # disable here so the cell-count book-keeping in test 2 reflects the
    # adaptive_grid path alone.
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        copy = list(arm = "pos", alpha_grid = alpha_grid),
        adaptive_grid = adaptive_grid,
        var_of_means_consistency = FALSE
    )
}


# --------------------------------------------------------------------------- #
# 1. Default (adaptive_grid = FALSE) preserves legacy fixed-grid behaviour    #
# --------------------------------------------------------------------------- #

test_that("adaptive_grid = FALSE leaves grid and result unchanged", {
    sim <- .simulate_joint_icar_strong(seed = 6)
    fit <- .fit_joint_icar(sim, alpha_grid = c(0.2, 0.4, 0.6),
                            adaptive_grid = FALSE)
    expect_null(fit$adaptive_grid_info)
    expect_equal(length(fit$log_marginal), 9L)         # 3 sigma x 3 alpha
    expect_equal(max(fit$theta_grid[, "alpha"]), 0.6)  # boundary intact
})


# --------------------------------------------------------------------------- #
# 2. adaptive_grid = TRUE refines beyond the user's alpha boundary             #
#    when the truncated grid traps posterior mass at the edge                  #
# --------------------------------------------------------------------------- #

test_that("adaptive_grid = TRUE extends alpha when boundary carries mass", {
    sim <- .simulate_joint_icar_strong(seed = 6)
    fit_F <- .fit_joint_icar(sim, alpha_grid = c(0.2, 0.4, 0.6),
                              adaptive_grid = FALSE)
    fit_T <- .fit_joint_icar(sim, alpha_grid = c(0.2, 0.4, 0.6),
                              adaptive_grid = TRUE)

    # Refinement metadata is populated.
    expect_false(is.null(fit_T$adaptive_grid_info))
    expect_true("alpha" %in%
                strsplit(paste(fit_T$adaptive_grid_info$triggered_axes,
                                collapse = ","), ",")[[1]])
    expect_gt(sum(fit_T$adaptive_grid_info$n_points_added), 0L)

    # Grid grew and now reaches beyond the user-supplied boundary.
    expect_gt(length(fit_T$log_marginal), length(fit_F$log_marginal))
    expect_gt(max(fit_T$theta_grid[, "alpha"]),
              max(fit_F$theta_grid[, "alpha"]) + 1e-6)

    # Mode-tracked path: new cells = new axis points (slice), not new ×
    # other_cartesian. With sigma_grid = 3 levels (icar, no rho axis), the
    # legacy cartesian path would have added `n_new_axis_pts * 3` cells;
    # the slice path adds exactly `n_new_axis_pts`. Three boundary-side
    # extension points => 3 new cells, 3 kernel solves, total 9 + 3 = 12.
    expect_equal(length(fit_T$log_marginal) - length(fit_F$log_marginal),
                  sum(fit_T$adaptive_grid_info$n_points_added))
    expect_lte(sum(fit_T$adaptive_grid_info$n_points_added), 6L)

    # Posterior alpha mean moves outward (FALSE truncated at alpha_max =
    # 0.6; TRUE lets mass past that ceiling toward alpha_true = 2.0).
    # 0.4 is well above MCSE for n=600, sigma=1.
    expect_lt(fit_F$theta_mean[["alpha"]], 0.6 + 1e-6)
    expect_gt(fit_T$theta_mean[["alpha"]], fit_F$theta_mean[["alpha"]] + 0.4)
})


# --------------------------------------------------------------------------- #
# 3. adaptive_grid = TRUE is a no-op when the grid already covers the         #
#    posterior (boundary mass below `adaptive_grid_edge_thresh`)               #
# --------------------------------------------------------------------------- #

test_that("adaptive_grid = TRUE leaves grid alone when boundary is empty", {
    sim <- .simulate_joint_icar_strong(seed = 6)
    # Wide grid: well past alpha_true = 2.0 (boundary near 4 has
    # negligible weight under any seed of this simulator).
    sp_wide <- c(0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0)
    fit_F <- .fit_joint_icar(sim, alpha_grid = sp_wide,
                              adaptive_grid = FALSE)
    fit_T <- .fit_joint_icar(sim, alpha_grid = sp_wide,
                              adaptive_grid = TRUE)

    expect_null(fit_T$adaptive_grid_info)
    expect_equal(length(fit_T$log_marginal), length(fit_F$log_marginal))
    expect_equal(fit_T$theta_mean[["alpha"]], fit_F$theta_mean[["alpha"]])
})
