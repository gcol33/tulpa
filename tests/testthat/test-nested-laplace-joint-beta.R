# Joint BYM2 nested-Laplace with beta-positive arm (Phase 1c-7c).
#
# Verifies that the joint kernel handles a beta likelihood as the second
# (positive) arm: same code path as binomial+gaussian, just routed through
# the beta branch of grad_hess_for_family.

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

.simulate_joint_beta <- function(N = 400, n_s = 30, sigma = 0.6, rho = 0.7,
                                  alpha_true = 1.0,
                                  beta_occ = c(-0.3, 0.5),
                                  beta_pos = c(0.2, -0.4),
                                  phi_beta = 20.0, seed = 7) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi_s   <- rnorm(n_s, 0, 1)
    theta_s <- rnorm(n_s, 0, 1)
    w_s     <- sigma * (sqrt(rho) * phi_s + sqrt(1 - rho) * theta_s)

    x <- rnorm(N)
    Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% beta_occ) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))

    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% beta_pos) + alpha_true * w_s[spi_pos]
    mu_pos  <- plogis(eta_pos)
    a       <- mu_pos * phi_beta
    b       <- (1 - mu_pos) * phi_beta
    y_pos   <- rbeta(sum(is_pos), a, b)
    # Pull samples off the open-interval boundaries so the log-density is
    # well-defined.
    y_pos   <- pmin(pmax(y_pos, 1e-6), 1 - 1e-6)

    list(
        N = N, n_s = n_s,
        spatial_idx = as.integer(spatial_idx),
        Xocc = Xocc, occur = occur,
        Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
        truth = list(beta_occ = beta_occ, beta_pos = beta_pos,
                     sigma = sigma, rho = rho, alpha = alpha_true,
                     phi_beta = phi_beta)
    )
}

.weighted_mode_mean <- function(res, idx) {
    sum(res$weights * res$modes[, idx])
}

test_that("joint BYM2 with beta-positive arm runs and recovers betas", {
    sim <- .simulate_joint_beta(N = 500, n_s = 40, alpha_true = 1.0,
                                 phi_beta = 30.0, seed = 13)
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
        family = "beta", phi = 30.0
    )
    prior <- list(
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.4, 0.8),
        rho_grid   = c(0.5, 0.9)
    )

    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        copy = list(arm = "pos",
                    alpha_grid = c(0, 0.5, 1.0, 1.5))
    )

    expect_s3_class(fit, "tulpa_nested_laplace_joint")
    expect_true(all(is.finite(fit$log_marginal)))

    layout <- fit$arm_layout
    slope_occ <- .weighted_mode_mean(fit, layout$beta_start[1] + 2L)
    slope_pos <- .weighted_mode_mean(fit, layout$beta_start[2] + 2L)

    expect_lt(abs(slope_occ - sim$truth$beta_occ[2]), 0.30)
    expect_lt(abs(slope_pos - sim$truth$beta_pos[2]), 0.30)
})
