# Joint multi-block nested-Laplace recovery test (Phase J-C1).
#
# Statistical validation, not a smoke test. Simulates 30 datasets from a
# two-arm joint model
#   arm1 (binomial occupancy): eta1 = X1 beta1 + sigma_occ * w[s_idx_1]
#                                              + ar[t_idx_1] + iota[o_idx_1]
#   arm2 (beta cover):          eta2 = X2 beta2 + sigma_pos * w[s_idx_2]
#                                              + ar[t_idx_2] + iota[o_idx_2]
# with shared latent fields
#   w     = sqrt(rho) * phi + sqrt(1-rho) * theta    (BYM2-style)
#   ar    = AR1(rho_ar, marginal precision tau_ar)
#   iota  = IID N(0, sigma_iid^2)
# truth: sigma_occ = 0.6, sigma_pos = 0.9 (alpha = 1.5), rho_b = 0.7,
#        tau_ar = 10, rho_ar = 0.8, sigma_iid = 0.3.
#
# Fits via tulpa_nested_laplace_joint() with multi-block prior
#   list(BYM2, AR1, IID)   copy block = 1 (BYM2) on arm "pos".
#
# Recovery thresholds (calibrated on 3-seed pilot at N=500 in
# dev_notes/pilot_joint_multi_recovery.R):
#   * median |sigma_occ_hat - 0.6| / 0.6 < 0.30
#   * median |sigma_pos_hat - 0.9| / 0.9 < 0.30
#   * median |alpha_hat     - 1.5| / 1.5 < 0.40
#   * 95% CI coverage >= 0.80 on each of sigma_occ, sigma_pos.
#
# AR1 tau and rho are not asserted on -- with n_years = 10 plus competing
# BYM2 / IID variance the AR1 marginal posterior is identification-limited
# (pilot shows tau biased +100% with grid mean near upper edge). The
# headline recoveries (sigma_occ, sigma_pos, alpha) are the contract.

skip_on_cran()

# Chain adjacency (25 sites). Same shape as the alpha-ridge regression
# test; keeps the BYM2 scale_factor = 1 calibration roughly correct on
# the spatial component.
.chain_adj_rec <- function(n_s) {
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

# Joint multi-block simulator. Two arms drawing from the same latent
# spatial / temporal / observer fields with per-arm sigma on the spatial
# component (copy semantics).
.sim_joint_multi_rec <- function(seed,
                                  N1 = 500L, N2 = 500L,
                                  n_sites = 25L, n_years = 10L,
                                  n_obs = 30L,
                                  sigma_occ = 0.6, sigma_pos = 0.9,
                                  rho_b = 0.7,
                                  tau_ar = 10, rho_ar = 0.8,
                                  sigma_iid = 0.3,
                                  betaO = c(-0.5, 0.5),
                                  betaP = c(0.2, -0.3),
                                  phi_b = 25) {
    set.seed(seed)

    # Unit-variance BYM2-style field (phi and theta both centered, iid).
    phi_f   <- rnorm(n_sites); phi_f   <- phi_f   - mean(phi_f)
    theta_f <- rnorm(n_sites); theta_f <- theta_f - mean(theta_f)
    w_unit  <- sqrt(rho_b) * phi_f + sqrt(1 - rho_b) * theta_f

    sd_ar  <- 1 / sqrt(tau_ar)
    inn_sd <- sd_ar * sqrt(1 - rho_ar^2)
    ar_t   <- numeric(n_years)
    ar_t[1L] <- rnorm(1, 0, sd_ar)
    for (t in 2:n_years) {
        ar_t[t] <- rho_ar * ar_t[t - 1L] + rnorm(1, 0, inn_sd)
    }

    iota_o <- rnorm(n_obs, 0, sigma_iid)

    s_idx_1 <- sample.int(n_sites, N1, replace = TRUE)
    s_idx_2 <- sample.int(n_sites, N2, replace = TRUE)
    t_idx_1 <- sample.int(n_years, N1, replace = TRUE)
    t_idx_2 <- sample.int(n_years, N2, replace = TRUE)
    o_idx_1 <- sample.int(n_obs,   N1, replace = TRUE)
    o_idx_2 <- sample.int(n_obs,   N2, replace = TRUE)

    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- as.numeric(X1 %*% betaO) +
            sigma_occ * w_unit[s_idx_1] +
            ar_t[t_idx_1] + iota_o[o_idx_1]
    eta2 <- as.numeric(X2 %*% betaP) +
            sigma_pos * w_unit[s_idx_2] +
            ar_t[t_idx_2] + iota_o[o_idx_2]

    y1 <- rbinom(N1, 1L, plogis(eta1))
    mu2 <- plogis(eta2)
    y2 <- rbeta(N2, mu2 * phi_b, (1 - mu2) * phi_b)
    y2 <- pmin(pmax(y2, 1e-6), 1 - 1e-6)

    list(
        arm_occ = list(
            y = as.numeric(y1), n_trials = rep(1L, N1), X = X1,
            re_idx = rep(0, N1), n_re_groups = 0L, sigma_re = 1.0,
            family = "binomial", phi = 1.0
        ),
        arm_pos = list(
            y = y2, n_trials = rep(1L, N2), X = X2,
            re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
            family = "beta", phi = phi_b
        ),
        s_idx_1 = s_idx_1, s_idx_2 = s_idx_2,
        t_idx_1 = t_idx_1, t_idx_2 = t_idx_2,
        o_idx_1 = o_idx_1, o_idx_2 = o_idx_2
    )
}

test_that("joint multi-block (BYM2+AR1+IID) recovers sigma_occ/sigma_pos/alpha (30 seeds)", {
    adj      <- .chain_adj_rec(25L)
    n_sites  <- adj$n_spatial_units
    n_years  <- 10L
    n_obs    <- 30L
    seeds    <- 9001:9030
    n_seeds  <- length(seeds)

    # Grids bracket truth on every axis; cell count = 3*2*3 * 3*2 * 3 = 324.
    sigma_grid     <- c(0.3, 0.6, 1.0)
    sigma_pos_grid <- c(0.45, 0.9, 1.6)
    rho_grid       <- c(0.5, 0.8)
    tau_grid       <- c(4, 10, 25)
    rho_ar1_grid   <- c(0.5, 0.85)
    sigma_iid_grid <- c(0.15, 0.3, 0.6)

    truth <- list(sigma_occ = 0.6, sigma_pos = 0.9, alpha = 1.5,
                  rho_b = 0.7, tau_ar = 10, rho_ar = 0.8,
                  sigma_iid = 0.3)

    sigma_occ_hat <- numeric(n_seeds)
    sigma_pos_hat <- numeric(n_seeds)
    sigma_occ_sd  <- numeric(n_seeds)
    sigma_pos_sd  <- numeric(n_seeds)
    alpha_hat     <- numeric(n_seeds)
    alpha_sd      <- numeric(n_seeds)

    for (i in seq_along(seeds)) {
        sim <- .sim_joint_multi_rec(seeds[i], n_sites = n_sites,
                                     n_years = n_years, n_obs = n_obs)

        prior <- list(
            list(
                type = "bym2",
                n_spatial_units = adj$n_spatial_units,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                sigma_grid = sigma_grid, rho_grid = rho_grid,
                spatial_idx = list(sim$s_idx_1, sim$s_idx_2)
            ),
            list(
                type = "ar1",
                n_times = n_years,
                tau_grid = tau_grid, rho_grid = rho_ar1_grid,
                temporal_idx = list(sim$t_idx_1, sim$t_idx_2)
            ),
            list(
                type = "iid",
                n_units = n_obs,
                sigma_grid = sigma_iid_grid,
                obs_idx = list(sim$o_idx_1, sim$o_idx_2)
            )
        )

        fit <- suppressWarnings(
            tulpa_nested_laplace_joint(
                responses = list(occ = sim$arm_occ, pos = sim$arm_pos),
                prior = prior,
                copy = list(block = 1L, arm = "pos",
                            sigma_pos_grid = sigma_pos_grid),
                adaptive_grid = FALSE,
                max_iter = 60L, tol = 1e-5
            )
        )

        expect_true(all(is.finite(fit$log_marginal)),
                    info = sprintf("seed %d: non-finite log_marginal",
                                    seeds[i]))

        bm1 <- fit$block_moments[[1L]]
        sigma_occ_hat[i] <- bm1$mean[["sigma_occ"]]
        sigma_pos_hat[i] <- bm1$mean[["sigma_pos"]]
        sigma_occ_sd[i]  <- bm1$sd[["sigma_occ"]]
        sigma_pos_sd[i]  <- bm1$sd[["sigma_pos"]]
        alpha_hat[i] <- fit$theta_mean[["alpha"]]
        alpha_sd[i]  <- fit$theta_sd[["alpha"]]
    }

    rel_err <- function(hat, tr) abs(hat - tr) / abs(tr)
    med_occ   <- median(rel_err(sigma_occ_hat, truth$sigma_occ))
    med_pos   <- median(rel_err(sigma_pos_hat, truth$sigma_pos))
    med_alpha <- median(rel_err(alpha_hat,     truth$alpha))

    # 95% CI coverage on the two arm-anchoring amplitudes.
    cover_occ <- mean(
        truth$sigma_occ >= sigma_occ_hat - 1.96 * sigma_occ_sd &
        truth$sigma_occ <= sigma_occ_hat + 1.96 * sigma_occ_sd
    )
    cover_pos <- mean(
        truth$sigma_pos >= sigma_pos_hat - 1.96 * sigma_pos_sd &
        truth$sigma_pos <= sigma_pos_hat + 1.96 * sigma_pos_sd
    )

    # Diagnostic line on test failure -- keeps repro cheap.
    info_str <- sprintf(
        "median |bias|/truth: sigma_occ=%.2f sigma_pos=%.2f alpha=%.2f | coverage: sigma_occ=%.2f sigma_pos=%.2f",
        med_occ, med_pos, med_alpha, cover_occ, cover_pos
    )

    expect_lt(med_occ,   0.30, label = info_str)
    expect_lt(med_pos,   0.30, label = info_str)
    expect_lt(med_alpha, 0.40, label = info_str)
    expect_gte(cover_occ, 0.80, label = info_str)
    expect_gte(cover_pos, 0.80, label = info_str)
})

# --------------------------------------------------------------------------- #
# J-C2: INLA cross-check on a 5-seed subset.                                  #
#                                                                             #
# Re-fit the same joint two-arm data with INLA's bym2 + ar1 + iid latents.    #
# INLA's BYM2 uses scale.model = TRUE (geomean of marginal variance = 1) so   #
# we pass scale_factor = compute_bym2_scale(adjacency) on the tulpa side for  #
# convention compatibility. Compares across-seed mean posterior moments       #
# (mean, sd) on sigma_occ, sigma_pos, alpha, sigma_iid -- the cleanly        #
# identified hyperparameters. Skipped on:                                     #
#   - CRAN (INLA isn't on CRAN, and the joint INLA fit takes 5-10s/seed)     #
#   - any machine without INLA installed.                                     #
#                                                                             #
# rho_b (BYM2 mixing), tau / rho_ar (AR1 hyperparameters) are not asserted:  #
# their posteriors are dominated by prior identification at this data size   #
# (n_years = 10, single noisy realization of an iid simulator) and the two   #
# engines genuinely disagree on the within-prior pull. The recovery test     #
# above plus the parity test in test-nested-laplace-joint-multi.R cover the  #
# joint engine's correctness; this test validates that the well-identified   #
# sigma marginals match an independent INLA implementation.                  #
# --------------------------------------------------------------------------- #

test_that("INLA joint fit agrees with tulpa joint multi-block (5 seeds)", {
    skip_on_cran()
    skip_if_not_installed("INLA")

    adj_sparse <- .chain_adj_rec(25L)
    n_sites    <- adj_sparse$n_spatial_units
    n_years    <- 10L
    n_obs      <- 30L
    seeds      <- 9001:9005

    adj_dense <- matrix(0L, n_sites, n_sites)
    for (s in seq_len(n_sites - 1L)) {
        adj_dense[s, s + 1L] <- 1L
        adj_dense[s + 1L, s] <- 1L
    }
    sf <- compute_bym2_scale(adj_dense)

    sigma_grid     <- c(0.3, 0.6, 1.0)
    sigma_pos_grid <- c(0.45, 0.9, 1.6)
    rho_grid       <- c(0.5, 0.8)
    tau_grid       <- c(4, 10, 25)
    rho_ar1_grid   <- c(0.5, 0.85)
    sigma_iid_grid <- c(0.15, 0.3, 0.6)

    keys     <- c("sigma_occ", "sigma_pos", "alpha", "sigma_iid")
    tul_mean <- matrix(NA_real_, length(seeds), length(keys),
                       dimnames = list(NULL, keys))
    tul_sd   <- tul_mean
    inla_mean <- tul_mean
    inla_sd   <- tul_mean

    for (i in seq_along(seeds)) {
        sim <- .sim_joint_multi_rec(seeds[i], n_sites = n_sites,
                                     n_years = n_years, n_obs = n_obs)
        n1 <- length(sim$arm_occ$y); n2 <- length(sim$arm_pos$y)

        # --- tulpa joint multi-block fit (compute_bym2_scale aligns with INLA's
        # scale.model = TRUE).
        prior <- list(
            list(type = "bym2",
                 n_spatial_units = adj_sparse$n_spatial_units,
                 adj_row_ptr = adj_sparse$adj_row_ptr,
                 adj_col_idx = adj_sparse$adj_col_idx,
                 n_neighbors = adj_sparse$n_neighbors,
                 scale_factor = sf,
                 sigma_grid = sigma_grid, rho_grid = rho_grid,
                 spatial_idx = list(sim$s_idx_1, sim$s_idx_2)),
            list(type = "ar1", n_times = n_years,
                 tau_grid = tau_grid, rho_grid = rho_ar1_grid,
                 temporal_idx = list(sim$t_idx_1, sim$t_idx_2)),
            list(type = "iid", n_units = n_obs,
                 sigma_grid = sigma_iid_grid,
                 obs_idx = list(sim$o_idx_1, sim$o_idx_2))
        )
        fit_t <- suppressWarnings(
            tulpa_nested_laplace_joint(
                responses = list(occ = sim$arm_occ, pos = sim$arm_pos),
                prior = prior,
                copy = list(block = 1L, arm = "pos",
                            sigma_pos_grid = sigma_pos_grid),
                adaptive_grid = FALSE, max_iter = 60L, tol = 1e-5
            )
        )
        bm1 <- fit_t$block_moments[[1L]]
        bm3 <- fit_t$block_moments[[3L]]
        tul_mean[i, "sigma_occ"] <- bm1$mean[["sigma_occ"]]
        tul_mean[i, "sigma_pos"] <- bm1$mean[["sigma_pos"]]
        tul_mean[i, "alpha"]     <- fit_t$theta_mean[["alpha"]]
        tul_mean[i, "sigma_iid"] <- bm3$mean[["sigma"]]
        tul_sd[i, "sigma_occ"] <- bm1$sd[["sigma_occ"]]
        tul_sd[i, "sigma_pos"] <- bm1$sd[["sigma_pos"]]
        tul_sd[i, "alpha"]     <- fit_t$theta_sd[["alpha"]]
        tul_sd[i, "sigma_iid"] <- bm3$sd[["sigma"]]

        # --- INLA joint fit: stack arms; binomial on row block 1, beta on
        # row block 2; spatial copy via "copy = 'area1'".
        Y <- matrix(NA, n1 + n2, 2)
        Y[1:n1, 1]                 <- sim$arm_occ$y
        Y[(n1 + 1):(n1 + n2), 2]   <- sim$arm_pos$y
        dat <- list(
            Y = Y,
            int1 = c(rep(1, n1), rep(NA, n2)),
            int2 = c(rep(NA, n1), rep(1, n2)),
            x1   = c(sim$arm_occ$X[, 2], rep(NA, n2)),
            x2   = c(rep(NA, n1), sim$arm_pos$X[, 2]),
            area1 = c(sim$s_idx_1, rep(NA, n2)),
            area2 = c(rep(NA, n1), sim$s_idx_2),
            year  = c(sim$t_idx_1, sim$t_idx_2),
            obs   = c(sim$o_idx_1, sim$o_idx_2)
        )
        Ntr <- c(rep(1L, n1), rep(NA, n2))

        formula_inla <- Y ~ -1 + int1 + int2 + x1 + x2 +
            f(area1, model = "bym2", graph = adj_dense,
              scale.model = TRUE, constr = TRUE) +
            f(area2, copy = "area1", fixed = FALSE,
              hyper = list(beta = list(fixed = FALSE,
                                        prior = "normal",
                                        param = c(0, 1)))) +
            f(year, model = "ar1") +
            f(obs,  model = "iid")

        fit_i <- INLA::inla(
            formula_inla, data = dat,
            family = c("binomial", "beta"), Ntrials = Ntr,
            control.predictor = list(compute = FALSE),
            control.compute   = list(config = FALSE, dic = FALSE, waic = FALSE),
            control.inla      = list(strategy = "gaussian",
                                      int.strategy = "ccd")
        )
        hp <- fit_i$summary.hyperpar
        prec_area <- hp["Precision for area1", c("mean", "sd")]
        beta_area <- hp["Beta for area2",      c("mean", "sd")]
        prec_obs  <- hp["Precision for obs",   c("mean", "sd")]

        # sigma_occ = 1/sqrt(prec_area). Delta-method sd.
        m_occ <- 1 / sqrt(prec_area[["mean"]])
        s_occ <- prec_area[["sd"]] / (2 * prec_area[["mean"]]^1.5)
        m_alp <- beta_area[["mean"]]; s_alp <- beta_area[["sd"]]
        m_iid <- 1 / sqrt(prec_obs[["mean"]])
        s_iid <- prec_obs[["sd"]] / (2 * prec_obs[["mean"]]^1.5)
        m_pos <- m_occ * m_alp
        s_pos <- sqrt((s_occ * m_alp)^2 + (m_occ * s_alp)^2)

        inla_mean[i, ] <- c(m_occ, m_pos, m_alp, m_iid)
        inla_sd[i, ]   <- c(s_occ, s_pos, s_alp, s_iid)
    }

    # Across-seed averages.
    tul_mean_avg  <- colMeans(tul_mean)
    inla_mean_avg <- colMeans(inla_mean)
    tul_sd_avg    <- colMeans(tul_sd)
    inla_sd_avg   <- colMeans(inla_sd)

    rel_mean <- abs(tul_mean_avg - inla_mean_avg) / abs(inla_mean_avg)
    rel_sd   <- abs(tul_sd_avg - inla_sd_avg)     / abs(inla_sd_avg)

    info_str <- paste0("\n",
        sprintf("  %-10s  tulpa %.3f +/- %.3f  INLA %.3f +/- %.3f  rel(mean)=%.2f rel(sd)=%.2f\n",
                 keys,
                 tul_mean_avg, tul_sd_avg,
                 inla_mean_avg, inla_sd_avg,
                 rel_mean, rel_sd),
        collapse = ""
    )

    # Means: within 15% on each well-identified quantity.
    expect_lt(rel_mean[["sigma_occ"]], 0.15, label = info_str)
    expect_lt(rel_mean[["sigma_pos"]], 0.15, label = info_str)
    expect_lt(rel_mean[["alpha"]],     0.15, label = info_str)
    expect_lt(rel_mean[["sigma_iid"]], 0.15, label = info_str)

    # SDs: discrete tulpa grid produces ~30-60% wider/narrower posterior SD
    # than INLA's continuous integration -- expected. Cap at 0.60 so the test
    # still catches order-of-magnitude regressions.
    expect_lt(rel_sd[["sigma_occ"]], 0.60, label = info_str)
    expect_lt(rel_sd[["sigma_pos"]], 0.60, label = info_str)
    expect_lt(rel_sd[["alpha"]],     0.80, label = info_str)
    expect_lt(rel_sd[["sigma_iid"]], 0.95, label = info_str)
})
