# Joint nested-Laplace: phi_grid as a residual-SD hyperparameter axis.
#
# `phi_grid` lifts each Gaussian/lognormal arm's residual SD off the
# parse-time scalar onto the outer grid. The kernel rewrites `arm.phi`
# at each grid point before the inner Newton solve, and `phi_<arm>`
# enters the joint posterior alongside (sigma_occ, rho, sigma_pos).
#
# Two recovery tests:
#   1. Gaussian copy arm — phi_grid recovers true residual SD.
#   2. Lognormal copy arm — first-class lognormal family with
#      `eta = E[log y]` and `-log(y)` Jacobian; phi_grid recovers
#      true residual SD on the log scale.
#
# Both pin (sigma_occ, rho, sigma_pos) at truth on a tight grid to
# isolate the phi_pos axis — the failure mode here is mis-identification
# of residual noise from spatial signal, not the spatial hyperparameters.

# --------------------------------------------------------------------------- #
# Shared helpers                                                              #
# --------------------------------------------------------------------------- #

.chain_adj_pg <- function(n_s) {
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

# Joint BYM2 simulator. `noise_kind` controls the second arm:
#   "gaussian"  — y_pos = eta_pos + N(0, sd_pos^2)
#   "lognormal" — y_pos = exp(eta_pos + N(0, sd_pos^2))
.simulate_joint_pg <- function(N = 400, n_s = 40,
                               sigma = 0.6, rho = 0.7, alpha_true = 1.0,
                               beta_occ = c(-0.3, 0.5),
                               beta_pos = c(0.2, -0.4),
                               sd_pos = 0.3,
                               noise_kind = c("gaussian", "lognormal"),
                               seed = 7) {
    noise_kind <- match.arg(noise_kind)
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi_s <- rnorm(n_s, 0, 1)
    theta <- rnorm(n_s, 0, 1)
    w_s   <- sigma * (sqrt(rho) * phi_s + sqrt(1 - rho) * theta)

    x <- rnorm(N)
    Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% beta_occ) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))

    is_pos    <- occur == 1L
    Xpos      <- Xocc[is_pos, , drop = FALSE]
    spi_pos   <- spatial_idx[is_pos]
    eta_pos   <- as.numeric(Xpos %*% beta_pos) + alpha_true * w_s[spi_pos]
    epsilon   <- rnorm(sum(is_pos), 0, sd_pos)
    y_pos <- if (noise_kind == "gaussian") {
        eta_pos + epsilon
    } else {
        exp(eta_pos + epsilon)
    }

    list(
        N = N, n_s = n_s,
        spatial_idx = as.integer(spatial_idx),
        Xocc = Xocc, occur = occur,
        Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
        truth = list(beta_occ = beta_occ, beta_pos = beta_pos,
                     sigma = sigma, rho = rho, alpha = alpha_true,
                     sd_pos = sd_pos, noise_kind = noise_kind)
    )
}


# --------------------------------------------------------------------------- #
# 1. Gaussian arm: phi_grid recovers true residual SD                         #
# --------------------------------------------------------------------------- #

test_that("phi_grid on gaussian copy arm recovers true residual SD", {
    sd_pos_true <- 0.3
    sim <- .simulate_joint_pg(N = 500, n_s = 30,
                              sigma = 0.6, rho = 0.7,
                              alpha_true = 1.0, sd_pos = sd_pos_true,
                              noise_kind = "gaussian", seed = 60101)
    adj <- .chain_adj_pg(sim$n_s)

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
        family = "gaussian",
        phi = 1.0  # placeholder; overridden by phi_grid below
    )
    # Pin spatial hyperparameters at truth so the test isolates phi_pos.
    prior <- list(
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = sim$truth$sigma,
        rho_grid   = sim$truth$rho
    )

    phi_axis <- exp(seq(log(0.08), log(1.0), length.out = 9))
    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior     = prior,
        copy      = list(arm = "pos", sigma_pos_grid = sim$truth$sigma),
        phi_grid  = list(pos = phi_axis)
    )

    expect_true("phi_pos" %in% colnames(fit$theta_grid))
    expect_true(all(is.finite(fit$log_marginal)))

    # Posterior mean recovers true sd_pos within ~25% relative.
    phi_mean <- fit$theta_mean[["phi_pos"]]
    expect_lt(abs(phi_mean - sd_pos_true) / sd_pos_true, 0.25)

    # Mode of the phi_pos marginal is near truth (within one grid step).
    df <- data.frame(phi = fit$theta_grid[, "phi_pos"], lm = fit$log_marginal)
    max_per_phi <- vapply(phi_axis, function(p) {
        sel <- abs(df$phi - p) < 1e-8
        if (any(sel)) max(df$lm[sel]) else -Inf
    }, numeric(1))
    best_phi <- phi_axis[which.max(max_per_phi)]
    # Truth (0.3) lies between phi_axis grid points; require best cell within
    # one log-step of truth.
    log_step <- diff(log(phi_axis))[1]
    expect_lt(abs(log(best_phi) - log(sd_pos_true)), 1.1 * log_step)
})


# --------------------------------------------------------------------------- #
# 2. Lognormal arm: phi_grid recovers true log-scale residual SD              #
# --------------------------------------------------------------------------- #

test_that("phi_grid on lognormal copy arm recovers true log-scale residual SD", {
    sd_pos_true <- 0.4  # SD on the log scale
    sim <- .simulate_joint_pg(N = 500, n_s = 30,
                              sigma = 0.6, rho = 0.7,
                              alpha_true = 1.0, sd_pos = sd_pos_true,
                              noise_kind = "lognormal", seed = 60201)
    adj <- .chain_adj_pg(sim$n_s)

    expect_true(all(sim$y_pos > 0))  # lognormal data must be strictly positive

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
        family = "lognormal",  # first-class lognormal, eta = E[log y]
        phi = 1.0
    )
    prior <- list(
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = sim$truth$sigma,
        rho_grid   = sim$truth$rho
    )

    phi_axis <- exp(seq(log(0.1), log(1.2), length.out = 9))
    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior     = prior,
        copy      = list(arm = "pos", sigma_pos_grid = sim$truth$sigma),
        phi_grid  = list(pos = phi_axis)
    )

    expect_true(all(is.finite(fit$log_marginal)))

    phi_mean <- fit$theta_mean[["phi_pos"]]
    expect_lt(abs(phi_mean - sd_pos_true) / sd_pos_true, 0.25)

    df <- data.frame(phi = fit$theta_grid[, "phi_pos"], lm = fit$log_marginal)
    max_per_phi <- vapply(phi_axis, function(p) {
        sel <- abs(df$phi - p) < 1e-8
        if (any(sel)) max(df$lm[sel]) else -Inf
    }, numeric(1))
    best_phi <- phi_axis[which.max(max_per_phi)]
    log_step <- diff(log(phi_axis))[1]
    expect_lt(abs(log(best_phi) - log(sd_pos_true)), 1.1 * log_step)
})


# --------------------------------------------------------------------------- #
# 3. Lognormal-vs-Gaussian-on-log(y) cross-check: log_marginal differs only   #
#    by the constant -sum(log y) Jacobian.                                    #
# --------------------------------------------------------------------------- #

test_that("lognormal == gaussian(log y) at any (sigma, rho, sigma_pos, phi)", {
    sim <- .simulate_joint_pg(N = 300, n_s = 25,
                              sigma = 0.6, rho = 0.7,
                              alpha_true = 1.0, sd_pos = 0.35,
                              noise_kind = "lognormal", seed = 60301)
    adj <- .chain_adj_pg(sim$n_s)

    n_pos <- length(sim$y_pos)
    arm_occ <- list(
        y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
        X = sim$Xocc, spatial_idx = sim$spatial_idx,
        re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
        family = "binomial", phi = 1.0
    )
    common_pos <- list(
        n_trials = rep(1L, n_pos), X = sim$Xpos, spatial_idx = sim$spi_pos,
        re_idx = rep(0, n_pos), n_re_groups = 0L, sigma_re = 1.0,
        phi = sim$truth$sd_pos
    )
    arm_pos_ln <- c(list(y = sim$y_pos,           family = "lognormal"),
                     common_pos)
    arm_pos_gn <- c(list(y = log(sim$y_pos),      family = "gaussian"),
                     common_pos)

    prior <- list(
        type = "bym2",
        n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = sim$truth$sigma,
        rho_grid   = sim$truth$rho
    )

    fit_ln <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos_ln),
        prior     = prior,
        copy      = list(arm = "pos", sigma_pos_grid = sim$truth$sigma)
    )
    fit_gn <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos_gn),
        prior     = prior,
        copy      = list(arm = "pos", sigma_pos_grid = sim$truth$sigma)
    )

    # log p_lognormal(y|eta,phi) = log p_gaussian(log y|eta,phi) - log(y).
    # Hessian wrt eta and the Newton mode are identical (the -log(y) term
    # is eta-independent), so log_marginal differs by exactly -sum(log y).
    jacobian <- -sum(log(sim$y_pos))
    expect_equal(fit_ln$log_marginal - fit_gn$log_marginal,
                 rep(jacobian, length(fit_ln$log_marginal)),
                 tolerance = 1e-6)

    # Modes (latent fixed effects + spatial field) are bit-identical
    # because the Newton inner solve sees the same gradient/Hessian.
    expect_equal(fit_ln$modes, fit_gn$modes, tolerance = 1e-6)
})
