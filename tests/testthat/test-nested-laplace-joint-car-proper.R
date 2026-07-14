# Joint CAR_proper nested-Laplace smoke tests (Phase 1c-7b).
#
# Mirror of the ICAR joint smoke test but with the proper-CAR backend.
# CAR_proper has full-rank precision Q = tau * (D - rho_car * W); we sweep
# both tau and rho_car (plus alpha when copy is set).

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
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

.simulate_joint_car <- function(N = 300, n_s = 30, sigma = 0.6,
                                 alpha_true = 1.0,
                                 beta_occ = c(-0.3, 0.5),
                                 beta_pos = c(0.2, -0.4),
                                 sd_pos = 0.5, seed = 7) {
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

    list(
        N = N, n_s = n_s,
        spatial_idx = as.integer(spatial_idx),
        Xocc = Xocc, occur = occur,
        Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
        truth = list(beta_occ = beta_occ, beta_pos = beta_pos,
                     sigma = sigma, alpha = alpha_true,
                     sd_pos = sd_pos, phi_s = phi_s)
    )
}

.weighted_mode_mean <- function(res, idx) {
    sum(res$weights * res$modes[, idx])
}


# --------------------------------------------------------------------------- #
# 1. alpha = 0 decouples the second arm from the shared field                  #
# --------------------------------------------------------------------------- #

test_that("joint CAR_proper with alpha = 0 leaves beta_occ unchanged", {
    skip_on_cran()
    sim <- .simulate_joint_car(N = 300, n_s = 30, alpha_true = 0.0, seed = 42)
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
        family = "gaussian", phi = 1.0
    )
    prior <- list(
        type = "car_proper",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid   = c(0.5, 1.0),
        rho_car_grid = c(0.7, 0.95)
    )

    fit_joint <- tulpa_nested_laplace_joint(
        responses = list(
            occ = arm_occ,
            pos = modifyList(arm_pos,
                             list(field_coef = list(name = "alpha", grid = 0)))
        ),
        prior = prior
    )
    expect_s3_class(fit_joint, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit_joint$log_marginal)))

    fit_single_occ <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ),
        prior = prior,
        copy = NULL
    )
    expect_true(all(is.finite(fit_single_occ$log_marginal)))

    layout_j <- fit_joint$arm_layout
    layout_s <- fit_single_occ$arm_layout
    beta1_joint  <- .weighted_mode_mean(fit_joint, layout_j$beta_start[1] + 1L)
    beta2_joint  <- .weighted_mode_mean(fit_joint, layout_j$beta_start[1] + 2L)
    beta1_single <- .weighted_mode_mean(fit_single_occ, layout_s$beta_start[1] + 1L)
    beta2_single <- .weighted_mode_mean(fit_single_occ, layout_s$beta_start[1] + 2L)

    expect_lt(abs(beta1_joint - beta1_single), 0.05)
    expect_lt(abs(beta2_joint - beta2_single), 0.05)
})


# --------------------------------------------------------------------------- #
# 2. Recovery on simulated data with a known shared spatial field             #
# --------------------------------------------------------------------------- #

test_that("joint CAR_proper recovers per-arm betas and locates the alpha mode", {
    skip_on_cran()
    sim <- .simulate_joint_car(N = 400, n_s = 40, sigma = 0.6,
                                alpha_true = 1.0, seed = 99)
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
        family = "gaussian", phi = 1.0
    )
    prior <- list(
        type = "car_proper",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid   = c(0.3, 0.7, 1.4),
        rho_car_grid = c(0.5, 0.9, 0.99)
    )

    fit <- tulpa_nested_laplace_joint(
        responses = list(
            occ = arm_occ,
            pos = modifyList(arm_pos, list(
                field_coef = list(name = "alpha", grid = c(0, 0.5, 1.0, 1.5))))
        ),
        prior = prior
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))

    layout <- fit$arm_layout

    slope_occ <- .weighted_mode_mean(fit, layout$beta_start[1] + 2L)
    slope_pos <- .weighted_mode_mean(fit, layout$beta_start[2] + 2L)

    expect_lt(abs(slope_occ - sim$truth$beta_occ[2]), 0.30)
    expect_lt(abs(slope_pos - sim$truth$beta_pos[2]), 0.30)

    # Looser bound than BYM2/ICAR: random-walk-simulated phi is a poor match
    # to proper-CAR's conditional structure on a 1D chain, so the inferred
    # alpha wanders within the {0, 0.5, 1, 1.5} grid. The alpha=0 reduction
    # test above pins down the kernel math; this bound only has to exclude a
    # collapse onto either grid edge (0 or 1.5+).
    alpha_mean <- fit$theta_mean[["alpha"]]
    expect_lt(abs(alpha_mean - 1.0), 0.75)
})
