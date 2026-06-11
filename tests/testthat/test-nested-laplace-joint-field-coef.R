# Per-arm field_coef tests (gcol33/tulpa#32, Change 1).
#
# `tulpa_nested_laplace_joint()` lets each arm declare a per-arm
# `field_coef` controlling its multiplier on the shared latent field's
# amplitude:
#
#   * field_coef = 1            -- donor (default; existing non-copy arms).
#   * field_coef = 0            -- this arm carries NO field contribution.
#   * field_coef = "alpha"      -- the per-arm scale is a named outer-grid
#                                  hyperparam.
#   * field_coef = list(name=, grid=) -- embedded axis declaration.
#
# The single-block copy coefficient is declared on the arm via `field_coef`;
# the separate `copy =` argument is multi-block only (the single-block shim was
# removed -- pre-release: no back-compat shims).
#
# This file covers:
#   (1) field_coef = 0 on arm 2 reproduces an independent single-arm fit
#       (arm 2 sees no spatial field; arm 1 still does).
#   (2) field_coef = list(name = "alpha", grid = ) runs, and the removed
#       single-block `copy =` shim now errors.
#   (3) A 3-arm fit with (1, 0, 0.5) constants runs end-to-end and
#       recovers per-arm slopes.

# --------------------------------------------------------------------------- #
# Helpers (mirror test-nested-laplace-joint-icar.R)                            #
# --------------------------------------------------------------------------- #

.fc_chain_adj <- function(n_s) {
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

.fc_simulate_joint_icar <- function(N = 300, n_s = 30, sigma = 0.6,
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

.fc_weighted_mode_mean <- function(res, idx) {
    sum(res$weights * res$modes[, idx])
}


# --------------------------------------------------------------------------- #
# 1. field_coef = 0 reproduces the independent fit                             #
# --------------------------------------------------------------------------- #

test_that("field_coef = 0 on arm 2 matches a no-field independent fit", {
    sim <- .fc_simulate_joint_icar(N = 300, n_s = 30, alpha_true = 0.0,
                                    seed = 42)
    adj <- .fc_chain_adj(sim$n_s)

    arm_occ <- list(
        y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
        X = sim$Xocc, spatial_idx = sim$spatial_idx,
        re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
        family = "binomial", phi = 1.0,
        field_coef = 1
    )
    arm_pos_no_field <- list(
        y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
        X = sim$Xpos, spatial_idx = sim$spi_pos,
        re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L, sigma_re = 1.0,
        family = "gaussian", phi = 1.0,
        field_coef = 0
    )
    prior <- list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid = c(0.5, 1.0)
    )

    # Joint fit with field_coef = 0 on the positive arm.
    fit_joint <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos_no_field),
        prior = prior
    )
    expect_s3_class(fit_joint, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit_joint$log_marginal)))

    # Reference: independent single-arm OLS for the positive arm (no
    # field). With field_coef = 0 the positive arm's eta is exactly
    # X_pos %*% beta_pos, so the posterior mean of beta_pos should match
    # OLS within Laplace numerical noise.
    ols_pos <- lm.fit(sim$Xpos, sim$y_pos)$coefficients

    layout <- fit_joint$arm_layout
    beta_pos1_joint <- .fc_weighted_mode_mean(fit_joint,
                                               layout$beta_start[2] + 1L)
    beta_pos2_joint <- .fc_weighted_mode_mean(fit_joint,
                                               layout$beta_start[2] + 2L)
    expect_lt(abs(beta_pos1_joint - ols_pos[[1L]]), 0.10)
    expect_lt(abs(beta_pos2_joint - ols_pos[[2L]]), 0.10)

    # Arm 1 (occ) still carries the spatial field: its beta should match
    # the single-arm spatial fit.
    fit_single_occ <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ),
        prior = prior
    )
    layout_s <- fit_single_occ$arm_layout
    beta1_joint  <- .fc_weighted_mode_mean(fit_joint,
                                            layout$beta_start[1] + 1L)
    beta2_joint  <- .fc_weighted_mode_mean(fit_joint,
                                            layout$beta_start[1] + 2L)
    beta1_single <- .fc_weighted_mode_mean(fit_single_occ,
                                            layout_s$beta_start[1] + 1L)
    beta2_single <- .fc_weighted_mode_mean(fit_single_occ,
                                            layout_s$beta_start[1] + 2L)
    expect_lt(abs(beta1_joint - beta1_single), 0.05)
    expect_lt(abs(beta2_joint - beta2_single), 0.05)
})


# --------------------------------------------------------------------------- #
# 2. field_coef = "alpha" matches the legacy `copy = ...` call                 #
# --------------------------------------------------------------------------- #

test_that("field_coef = list(name='alpha', grid=) runs; single-block copy= errors", {
    sim <- .fc_simulate_joint_icar(N = 400, n_s = 40, sigma = 0.6,
                                    alpha_true = 1.0, seed = 99)
    adj <- .fc_chain_adj(sim$n_s)

    alpha_grid <- c(0, 0.5, 1.0, 1.5)

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
        sigma_grid = c(0.3, 0.7, 1.4)
    )

    # The copy coefficient is declared on the arm itself.
    arm_pos_new <- arm_pos
    arm_pos_new$field_coef <- list(name = "alpha", grid = alpha_grid)
    fit_new <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos_new),
        prior = prior
    )
    expect_s3_class(fit_new, "tulpa_nested_laplace_joint")
    expect_true("alpha" %in% colnames(fit_new$theta_grid))

    # The single-block `copy =` back-compat shim was removed (pre-release: no
    # shims). Passing it now errors, pointing at the per-arm field_coef spec.
    expect_error(
        tulpa_nested_laplace_joint(
            responses = list(occ = arm_occ, pos = arm_pos),
            prior = prior,
            copy = list(arm = "pos", alpha_grid = alpha_grid)
        ),
        "field_coef"
    )
})


# --------------------------------------------------------------------------- #
# 3. 3-arm fit with (1, 0, 0.5) constants runs end-to-end                      #
# --------------------------------------------------------------------------- #

test_that("3-arm (field_coef = 1, 0, 0.5) runs and recovers betas", {
    set.seed(11)
    n_s <- 30L
    N   <- 300L
    sigma <- 0.6
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(rnorm(n_s, 0, sigma / sqrt(n_s)))
    phi_s <- rw - mean(rw)

    # Three arms, all gaussian for simplicity, sharing the chain ICAR field.
    # Arm 1: full field (coef = 1).
    # Arm 2: no field (coef = 0) -- pure regression on its design.
    # Arm 3: scaled field (coef = 0.5).
    beta1 <- c(-0.2, 0.4)
    beta2 <- c( 0.5, -0.3)
    beta3 <- c( 0.1,  0.6)

    sd1 <- 0.4; sd2 <- 0.4; sd3 <- 0.4

    x1 <- rnorm(N); X1 <- cbind(1, x1)
    x2 <- rnorm(N); X2 <- cbind(1, x2)
    x3 <- rnorm(N); X3 <- cbind(1, x3)

    eta1 <- as.numeric(X1 %*% beta1) + 1.0 * phi_s[spatial_idx]
    eta2 <- as.numeric(X2 %*% beta2) + 0.0 * phi_s[spatial_idx]
    eta3 <- as.numeric(X3 %*% beta3) + 0.5 * phi_s[spatial_idx]
    y1 <- rnorm(N, eta1, sd1)
    y2 <- rnorm(N, eta2, sd2)
    y3 <- rnorm(N, eta3, sd3)

    adj <- .fc_chain_adj(n_s)

    arm_make <- function(y, X, sd, fc) {
        list(
            y = y, n_trials = rep(1L, N),
            X = X, spatial_idx = spatial_idx,
            re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1.0,
            family = "gaussian", phi = sd,
            field_coef = fc
        )
    }
    responses <- list(
        a1 = arm_make(y1, X1, sd1, 1),
        a2 = arm_make(y2, X2, sd2, 0),
        a3 = arm_make(y3, X3, sd3, 0.5)
    )
    prior <- list(
        type = "icar",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors,
        sigma_grid = c(0.4, 0.8)
    )

    fit <- tulpa_nested_laplace_joint(responses = responses, prior = prior)
    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))

    layout <- fit$arm_layout
    slope1 <- .fc_weighted_mode_mean(fit, layout$beta_start[1] + 2L)
    slope2 <- .fc_weighted_mode_mean(fit, layout$beta_start[2] + 2L)
    slope3 <- .fc_weighted_mode_mean(fit, layout$beta_start[3] + 2L)

    # Recovery tolerances are loose -- the goal is to demonstrate the
    # 3-arm fit runs end-to-end and gives sensible per-arm slopes, not
    # to test calibration. (Tight recovery lives in the dedicated
    # multi-recovery file.)
    expect_lt(abs(slope1 - beta1[2]), 0.30)
    expect_lt(abs(slope2 - beta2[2]), 0.30)
    expect_lt(abs(slope3 - beta3[2]), 0.30)
})
