# Joint BYM2 nested-Laplace smoke tests (Phase 1c).
#
# Outer grid is (sigma_occ, rho [, sigma_pos]) after gcol33/tulpa#18.
# alpha = sigma_pos / sigma_occ is derived post-hoc on the joint posterior.
#
# 1. sigma_pos = 0 reduces the joint to two independent fits — beta_occ should
#    match the single-arm binomial fit and beta_pos should match the
#    single-arm gaussian fit (both at the best grid point), within tolerance.
# 2. Joint fit on simulated data with a true shared spatial field recovers
#    the per-arm betas and locates a sensible (sigma_occ, rho, sigma_pos)
#    maximum / alpha posterior mean.

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

# Tiny 1D-chain adjacency in CSR (0-based) — keeps the BYM2 prior well-defined.
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

.simulate_joint <- function(N = 200, n_s = 25, sigma = 0.6, rho = 0.7,
                            alpha_true = 1.0,
                            beta_occ = c(-0.3, 0.5),
                            beta_pos = c(0.2, -0.4),
                            sd_pos = 0.5, seed = 7) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- rnorm(n_s, 0, 1)
    theta <- rnorm(n_s, 0, 1)
    w_s   <- sigma * (sqrt(rho) * phi + sqrt(1 - rho) * theta)

    x <- rnorm(N)
    Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% beta_occ) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))

    is_pos    <- occur == 1L
    Xpos      <- Xocc[is_pos, , drop = FALSE]
    spi_pos   <- spatial_idx[is_pos]
    eta_pos   <- as.numeric(Xpos %*% beta_pos) + alpha_true * w_s[spi_pos]
    y_pos     <- rnorm(sum(is_pos), eta_pos, sd_pos)

    list(
        N = N, n_s = n_s,
        spatial_idx = as.integer(spatial_idx),
        Xocc = Xocc, occur = occur,
        Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
        truth = list(beta_occ = beta_occ, beta_pos = beta_pos,
                     sigma = sigma, rho = rho, alpha = alpha_true,
                     sd_pos = sd_pos, w_s = w_s)
    )
}

.weighted_mode_mean <- function(res, idx) {
    # Posterior-weighted mean of a single latent slot across grid points.
    sum(res$weights * res$modes[, idx])
}


# --------------------------------------------------------------------------- #
# 1. alpha = 0 decouples the second arm from the shared field                  #
# --------------------------------------------------------------------------- #

test_that("joint BYM2 with sigma_pos = 0 leaves beta_occ unchanged from single-arm", {
    sim <- .simulate_joint(N = 300, n_s = 30, alpha_true = 0.0, seed = 42)
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
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.4, 0.8), rho_grid = c(0.5, 0.9)
    )

    fit_joint <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        copy = list(arm = "pos", sigma_pos_grid = 0.0)
    )
    expect_s3_class(fit_joint, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit_joint$log_marginal)))

    # Single-arm fit on occ only (no copy, no second arm).
    fit_single_occ <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ),
        prior = prior,
        copy = NULL
    )
    expect_true(all(is.finite(fit_single_occ$log_marginal)))

    # Compare beta_occ posterior weighted mean.
    layout_j <- fit_joint$arm_layout
    beta1_joint  <- .weighted_mode_mean(fit_joint, layout_j$beta_start[1] + 1L)
    beta2_joint  <- .weighted_mode_mean(fit_joint, layout_j$beta_start[1] + 2L)
    layout_s <- fit_single_occ$arm_layout
    beta1_single <- .weighted_mode_mean(fit_single_occ, layout_s$beta_start[1] + 1L)
    beta2_single <- .weighted_mode_mean(fit_single_occ, layout_s$beta_start[1] + 2L)

    # alpha=0 fully decouples arm 2's contribution to the shared spatial field
    # in the eta-builder. The remaining coupling is *prior*-only and goes
    # through the shared (phi, theta) block, so the two fits won't be bit-
    # identical, but should be within numerical noise on a centered design.
    expect_lt(abs(beta1_joint - beta1_single), 0.05)
    expect_lt(abs(beta2_joint - beta2_single), 0.05)
})


# --------------------------------------------------------------------------- #
# 2. Recovery on simulated data with a known shared spatial field             #
# --------------------------------------------------------------------------- #

test_that("joint BYM2 recovers per-arm betas and locates the alpha mode", {
    sim <- .simulate_joint(N = 400, n_s = 40, sigma = 0.6, rho = 0.7,
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
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.3, 0.6, 1.0),
        rho_grid   = c(0.3, 0.7)
    )

    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        copy = list(arm = "pos",
                    sigma_pos_grid = c(0.0, 0.3, 0.6, 1.0))
    )
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))

    layout <- fit$arm_layout

    # Per-arm beta posterior means (intercept absorbs the centered spatial
    # mean, so check the slope only).
    slope_occ <- .weighted_mode_mean(fit, layout$beta_start[1] + 2L)
    slope_pos <- .weighted_mode_mean(fit, layout$beta_start[2] + 2L)

    expect_lt(abs(slope_occ - sim$truth$beta_occ[2]), 0.30)
    expect_lt(abs(slope_pos - sim$truth$beta_pos[2]), 0.30)

    # alpha = sigma_pos / sigma_occ posterior mean should land near the
    # true 1.0. Computed post-hoc from the joint posterior over
    # (sigma_occ, sigma_pos) — see gcol33/tulpa#18.
    alpha_mean <- fit$theta_mean[["alpha"]]
    expect_lt(abs(alpha_mean - 1.0), 0.6)
})


# --------------------------------------------------------------------------- #
# 3. Regression: gaussian copy arm at the true noise scale (gcol33/tulpa#17)  #
# --------------------------------------------------------------------------- #
# Before the fix, post-Newton `center_effects` shifted the phi block to mean
# zero without compensating each arm's intercept, which dragged eta off the
# Newton mode and tanked log_lik for the gaussian copy arm. At phi =
# sigma_pos_true the bug made sigma_pos = 0 the best cell by 500+ log units.
# The fix: shift the per-arm intercept (first beta column) by
# arm_sigma * d_phi_base_unit * mean(phi) so eta — and hence log_marginal —
# is invariant under the centering.

test_that("joint BYM2 with gaussian copy arm prefers alpha = alpha_true at true phi", {
    sim <- .simulate_joint(N = 300, n_s = 25,
                            sigma = 0.6, rho = 0.7,
                            alpha_true = 1.0, sd_pos = 0.3, seed = 50101)
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
    # True donor amplitude is sigma=0.6 in this simulator. Pin sigma_occ to
    # the truth so the test isolates the copy-arm mode along sigma_pos —
    # the centering-bug failure mode is on sigma_pos, not on sigma_occ.
    sigma_occ_true <- 0.6
    prior <- list(
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = sigma_occ_true,
        rho_grid   = c(0.5, 0.7, 0.9)
    )

    sigma_pos_grid <- c(0.0, 0.3, 0.6, 0.9)
    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        copy = list(arm = "pos", sigma_pos_grid = sigma_pos_grid)
    )

    # alpha = sigma_pos / sigma_occ; sigma_occ pinned at 0.6, alpha_true = 1.
    df <- data.frame(sigma_pos = fit$theta_grid[, "sigma_pos"],
                     log_marginal = fit$log_marginal)
    best_sp <- df$sigma_pos[which.max(df$log_marginal)]
    expect_equal(best_sp, sigma_occ_true)

    max_by_sp <- vapply(sigma_pos_grid,
                        function(sp) max(df$log_marginal[df$sigma_pos == sp]),
                        numeric(1))
    # sigma_pos = 0 must be the worst cell at the true noise scale (this
    # fails catastrophically before the fix).
    expect_lt(max_by_sp[1], max_by_sp[3] - 50)
})
