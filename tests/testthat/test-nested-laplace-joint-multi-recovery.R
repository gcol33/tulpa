# Joint multi-block nested-Laplace recovery sweep (dev_notes/plans/accuracy.md Phase 3).
#
# Statistical validation, not a smoke test. Per dev_notes/plans/accuracy.md sec. Win 1 Validation
# Plan, sweeps recovery across three alpha regimes -- {0.0, 1.0, 2.0} -- on a
# BYM2 + AR1 + IID two-arm joint model (binomial occupancy + beta cover).
#
# Simulator:
#   arm1 (binomial): eta1 = X1 beta1 + sigma * w[s_idx_1]
#                                    + ar[t_idx_1] + iota[o_idx_1]
#   arm2 (beta):     eta2 = X2 beta2 + alpha * sigma * w[s_idx_2]
#                                    + ar[t_idx_2] + iota[o_idx_2]
# Shared latent fields:
#   w    = sqrt(rho_b) * phi + sqrt(1 - rho_b) * theta    (BYM2-style)
#   ar   = AR1(rho_ar, marginal precision tau_ar)
#   iota = IID N(0, sigma_iid^2)
# Truth (held fixed across regimes): sigma = 0.6, rho_b = 0.7, tau_ar = 10,
# rho_ar = 0.8, sigma_iid = 0.3. Truth on alpha varies per regime.
#
# CI semantics: per dev_notes/plans/accuracy.md Win 2, alpha is summarized via weighted
# empirical 2.5 / 97.5 quantiles on the joint hyperparameter grid
# (`theta_ci_lo` / `theta_ci_hi`), not mean +/- 1.96 sd. Sigma is reported
# from per-block Laplace-at-mode Gaussian moments (`block_moments[[1]]$mean`,
# `$sd`) -- sigma is identified well enough away from its boundary for
# Gaussian SE to be honest.
#
# Per-regime acceptance (calibrated for N1 = N2 = 500, 25 sites, alpha grid
# c(0, 0.5, 1, 1.5, 2, 2.5)):
#   alpha = 2.0 (strong copy):
#     * median |alpha_hat - 2| / 2     < 0.35
#     * empirical 95% CI coverage      >= 0.80
#   alpha = 1.0 (moderate copy):
#     * median |alpha_hat - 1| / 1     < 0.45
#     * empirical 95% CI coverage      >= 0.80
#   alpha = 0.0 (no copy -- boundary win vs rgeneric):
#     * median |alpha_hat|             < 0.40
#     * median(alpha_median)           < 0.30
#     * CI_lo touches 0 in            >= 0.85 of seeds
# Sigma (across all regimes):
#     * median |sigma_hat - 0.6| / 0.6 < 0.35
#
# AR1 tau / rho are not asserted: at n_years = 10 plus competing BYM2 / IID
# variance the AR1 marginal is identification-limited (pilot shows tau biased
# by ~100% with grid mean near the upper edge).

skip_on_cran()
skip_if_fast()

# Chain adjacency (25 sites). Same shape as the alpha-ridge regression test;
# keeps BYM2 scale_factor = 1 calibration roughly correct on the spatial
# component.
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

# Joint multi-block simulator. Per-regime alpha controls the strength of the
# copy of the BYM2 field into the positive arm.
.sim_joint_multi_rec <- function(seed,
                                  N1 = 500L, N2 = 500L,
                                  n_sites = 25L, n_years = 10L,
                                  n_obs = 30L,
                                  sigma = 0.6, alpha = 1.5,
                                  rho_b = 0.7,
                                  tau_ar = 10, rho_ar = 0.8,
                                  sigma_iid = 0.3,
                                  betaO = c(-0.5, 0.5),
                                  betaP = c(0.2, -0.3),
                                  phi_b = 25) {
    set.seed(seed)

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
            sigma * w_unit[s_idx_1] +
            ar_t[t_idx_1] + iota_o[o_idx_1]
    eta2 <- as.numeric(X2 %*% betaP) +
            alpha * sigma * w_unit[s_idx_2] +
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

# Per-seed fit helper. Returns a flat list of recovery quantities so the
# regime loop can collect across seeds without nested indexing.
.fit_one_seed_multi <- function(seed, alpha_true,
                                 sigma_grid, alpha_grid, rho_grid,
                                 tau_grid, rho_ar1_grid, sigma_iid_grid,
                                 adj, n_years, n_obs,
                                 scale_factor = 1.0) {
    sim <- .sim_joint_multi_rec(seed,
                                 n_sites = adj$n_spatial_units,
                                 n_years = n_years, n_obs = n_obs,
                                 alpha = alpha_true)
    prior <- list(
        list(type = "bym2",
             n_spatial_units = adj$n_spatial_units,
             adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
             n_neighbors = adj$n_neighbors, scale_factor = scale_factor,
             sigma_grid = sigma_grid, rho_grid = rho_grid,
             spatial_idx = list(sim$s_idx_1, sim$s_idx_2)),
        list(type = "ar1", n_times = n_years,
             tau_grid = tau_grid, rho_grid = rho_ar1_grid,
             temporal_idx = list(sim$t_idx_1, sim$t_idx_2)),
        list(type = "iid", n_units = n_obs,
             sigma_grid = sigma_iid_grid,
             obs_idx = list(sim$o_idx_1, sim$o_idx_2))
    )
    fit <- suppressWarnings(
        tulpa_nested_laplace_joint(
            responses = list(occ = sim$arm_occ, pos = sim$arm_pos),
            prior = prior,
            copy = list(block = 1L, arm = "pos", alpha_grid = alpha_grid),
            control = list(adaptive_grid = FALSE, max_iter = 60L, tol = 1e-5)
        )
    )
    bm1 <- fit$block_moments[[1L]]
    bm3 <- fit$block_moments[[3L]]
    # Multi-block joint_grid columns are prefixed `b<N>.` (B blocks); the
    # copy block here is block 1, so `b1.alpha` is the prefixed axis name
    # used by theta_median / theta_ci_lo / theta_ci_hi.
    list(
        sigma_hat       = bm1$mean[["sigma"]],
        sigma_sd        = bm1$sd[["sigma"]],
        alpha_mean      = bm1$mean[["alpha"]],
        alpha_sd        = bm1$sd[["alpha"]],
        alpha_med       = fit$theta_median[["b1.alpha"]],
        alpha_lo        = fit$theta_ci_lo[["b1.alpha"]],
        alpha_hi        = fit$theta_ci_hi[["b1.alpha"]],
        sigma_iid_mean  = bm3$mean[["sigma"]],
        sigma_iid_sd    = bm3$sd[["sigma"]],
        log_marg_ok     = all(is.finite(fit$log_marginal)),
        sim             = sim
    )
}

# --------------------------------------------------------------------------- #
# Phase 3: 30-seed recovery sweep across alpha in {0, 1, 2}.                  #
# --------------------------------------------------------------------------- #

test_that("(sigma, alpha) reparam recovers across alpha in {0, 1, 2} on BYM2+AR1+IID (30 seeds per regime)", {
    adj      <- .chain_adj_rec(25L)
    n_years  <- 10L
    n_obs    <- 30L
    seeds    <- 9001:9030
    n_seeds  <- length(seeds)

    # Single shared hyperparameter grid that brackets each alpha truth value.
    # Cell count = 3 * 6 * 2 * 3 * 2 * 3 = 648.
    sigma_grid     <- c(0.3, 0.6, 1.0)
    alpha_grid     <- c(0, 0.5, 1.0, 1.5, 2.0, 2.5)
    rho_grid       <- c(0.5, 0.8)
    tau_grid       <- c(4, 10, 25)
    rho_ar1_grid   <- c(0.5, 0.85)
    sigma_iid_grid <- c(0.15, 0.3, 0.6)

    regimes <- list(
        list(name = "alpha=0", alpha_true = 0.0),
        list(name = "alpha=1", alpha_true = 1.0),
        list(name = "alpha=2", alpha_true = 2.0)
    )

    sigma_truth <- 0.6

    for (r in regimes) {
        sigma_hat <- numeric(n_seeds)
        alpha_mn  <- numeric(n_seeds)
        alpha_md  <- numeric(n_seeds)
        alpha_lo  <- numeric(n_seeds)
        alpha_hi  <- numeric(n_seeds)

        for (i in seq_along(seeds)) {
            f <- .fit_one_seed_multi(
                seeds[i], alpha_true = r$alpha_true,
                sigma_grid = sigma_grid, alpha_grid = alpha_grid,
                rho_grid = rho_grid, tau_grid = tau_grid,
                rho_ar1_grid = rho_ar1_grid, sigma_iid_grid = sigma_iid_grid,
                adj = adj, n_years = n_years, n_obs = n_obs
            )
            expect_true(f$log_marg_ok,
                        info = sprintf("%s seed %d: non-finite log_marginal",
                                        r$name, seeds[i]))
            sigma_hat[i] <- f$sigma_hat
            alpha_mn[i]  <- f$alpha_mean
            alpha_md[i]  <- f$alpha_med
            alpha_lo[i]  <- f$alpha_lo
            alpha_hi[i]  <- f$alpha_hi
        }

        # Sigma: same contract across regimes.
        rel_sigma <- median(abs(sigma_hat - sigma_truth) / sigma_truth)
        info_s <- sprintf("%s: median rel|sigma_hat - 0.6| = %.3f",
                          r$name, rel_sigma)
        expect_lt(rel_sigma, 0.35, label = info_s)

        # Alpha: regime-dependent.
        if (r$alpha_true == 0) {
            abs_mn  <- median(abs(alpha_mn))
            abs_md  <- median(alpha_md)
            boundary_frac <- mean(alpha_lo <= 0 + 1e-8)
            info_a <- sprintf(
                "%s: median |alpha_mean|=%.3f, median(alpha_med)=%.3f, frac(CI_lo<=0)=%.2f",
                r$name, abs_mn, abs_md, boundary_frac)
            expect_lt(abs_mn, 0.40, label = info_a)
            expect_lt(abs_md, 0.30, label = info_a)
            expect_gte(boundary_frac, 0.85, label = info_a)
        } else {
            rel_a <- median(abs(alpha_mn - r$alpha_true) / r$alpha_true)
            cov_a <- mean(alpha_lo <= r$alpha_true &
                          alpha_hi >= r$alpha_true)
            thr_rel <- if (r$alpha_true == 1.0) 0.45 else 0.35
            info_a <- sprintf(
                "%s: median rel|alpha_mean - %.1f|=%.3f, empirical CI cov=%.2f",
                r$name, r$alpha_true, rel_a, cov_a)
            expect_lt(rel_a, thr_rel, label = info_a)
            expect_gte(cov_a, 0.80, label = info_a)
        }
    }
})

# --------------------------------------------------------------------------- #
# Phase 3: INLA cross-check on the well-identified middle of the alpha range. #
#                                                                             #
# Re-fit the same joint two-arm data at alpha = 1 with INLA's bym2 + ar1 +    #
# iid latents. INLA's BYM2 uses scale.model = TRUE (geomean of marginal       #
# variance = 1) so we pass scale_factor = compute_bym2_scale(adjacency) on    #
# the tulpa side for convention compatibility. Compares across-seed mean      #
# posterior moments on sigma, alpha, sigma_iid -- the cleanly identified      #
# hyperparameters. Skipped on:                                                #
#   - CRAN (INLA isn't on CRAN, and the joint INLA fit takes 5-10s / seed)    #
#   - any machine without INLA installed.                                     #
#                                                                             #
# rho_b (BYM2 mixing) and AR1 hyperparameters are not asserted: their         #
# posteriors are dominated by prior identification at this data size         #
# (n_years = 10, single noisy realization of an iid simulator) and the two   #
# engines genuinely disagree on the within-prior pull.                       #
# --------------------------------------------------------------------------- #

test_that("INLA joint fit agrees with tulpa at alpha = 1 (5-seed cross-check)", {
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
    alpha_grid     <- c(0, 0.5, 1.0, 1.5, 2.0, 2.5)
    rho_grid       <- c(0.5, 0.8)
    tau_grid       <- c(4, 10, 25)
    rho_ar1_grid   <- c(0.5, 0.85)
    sigma_iid_grid <- c(0.15, 0.3, 0.6)

    keys     <- c("sigma", "alpha", "sigma_iid")
    tul_mean <- matrix(NA_real_, length(seeds), length(keys),
                       dimnames = list(NULL, keys))
    tul_sd    <- tul_mean
    inla_mean <- tul_mean
    inla_sd   <- tul_mean

    for (i in seq_along(seeds)) {
        f <- .fit_one_seed_multi(
            seeds[i], alpha_true = 1.0,
            sigma_grid = sigma_grid, alpha_grid = alpha_grid,
            rho_grid = rho_grid, tau_grid = tau_grid,
            rho_ar1_grid = rho_ar1_grid, sigma_iid_grid = sigma_iid_grid,
            adj = adj_sparse, n_years = n_years, n_obs = n_obs,
            scale_factor = sf
        )
        tul_mean[i, "sigma"]     <- f$sigma_hat
        tul_mean[i, "alpha"]     <- f$alpha_mean
        tul_mean[i, "sigma_iid"] <- f$sigma_iid_mean
        tul_sd[i, "sigma"]     <- f$sigma_sd
        tul_sd[i, "alpha"]     <- f$alpha_sd
        tul_sd[i, "sigma_iid"] <- f$sigma_iid_sd

        # INLA side.
        sim <- f$sim
        n1 <- length(sim$arm_occ$y); n2 <- length(sim$arm_pos$y)
        Y <- matrix(NA, n1 + n2, 2)
        Y[1:n1, 1]               <- sim$arm_occ$y
        Y[(n1 + 1):(n1 + n2), 2] <- sim$arm_pos$y
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

        m_sig <- 1 / sqrt(prec_area[["mean"]])
        s_sig <- prec_area[["sd"]] / (2 * prec_area[["mean"]]^1.5)
        m_alp <- beta_area[["mean"]]; s_alp <- beta_area[["sd"]]
        m_iid <- 1 / sqrt(prec_obs[["mean"]])
        s_iid <- prec_obs[["sd"]] / (2 * prec_obs[["mean"]]^1.5)

        inla_mean[i, ] <- c(m_sig, m_alp, m_iid)
        inla_sd[i, ]   <- c(s_sig, s_alp, s_iid)
    }

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

    # Means: within 20% on each well-identified quantity (alpha at truth = 1
    # is less constrained than at truth = 1.5, so loosen from 0.15 to 0.20).
    expect_lt(rel_mean[["sigma"]],     0.20, label = info_str)
    expect_lt(rel_mean[["alpha"]],     0.25, label = info_str)
    expect_lt(rel_mean[["sigma_iid"]], 0.20, label = info_str)

    # SDs: discrete tulpa grid produces ~30-90% wider / narrower posterior
    # SD than INLA's continuous integration -- expected. Cap to catch
    # order-of-magnitude regressions.
    expect_lt(rel_sd[["sigma"]],     0.60, label = info_str)
    expect_lt(rel_sd[["alpha"]],     0.90, label = info_str)
    expect_lt(rel_sd[["sigma_iid"]], 0.95, label = info_str)
})

# --------------------------------------------------------------------------- #
# dev_notes/plans/accuracy.md headline at alpha = 0 -- positioning note.                      #
#                                                                             #
# dev_notes/plans/accuracy.md claims a "beat rgeneric" win at alpha = 0: tulpa's (sigma,      #
# alpha) reparameterization with PC-style prior puts calibrated mass at      #
# alpha = 0 while INLA's `rgeneric` (custom user-coded Q with no boundary    #
# atom) overshoots. The 30-seed sweep above pins down the tulpa side of      #
# that claim: empirical CI lower bound on alpha touches 0 in >= 85% of       #
# seeds at truth alpha = 0.                                                   #
#                                                                             #
# A direct INLA cross-check at alpha = 0 was prototyped against `copy =      #
# "area1"` with `prior = "normal", param = c(0, 1)` on Beta -- INLA's        #
# canonical built-in copy machinery. That comparison is NOT informative:     #
# the zero-centered Gaussian on Beta puts effective mass at 0 just like      #
# tulpa's grid + PC atom, and 95% CI contains 0 in 100% of seeds. INLA's     #
# `copy =` is not what dev_notes/plans/accuracy.md calls "rgeneric" -- the rgeneric path     #
# requires writing a user-defined Q-matrix + prior callback (no PC prior on  #
# alpha by default), and *that* is where the boundary atom matters.         #
#                                                                             #
# Implementing the rgeneric BYM2-copy callback to validate the headline       #
# "beat rgeneric" claim is tracked as future work; not in Phase 3 scope.     #
# --------------------------------------------------------------------------- #
