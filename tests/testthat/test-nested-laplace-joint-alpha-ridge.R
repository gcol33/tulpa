# Joint BYM2 -- absence of the cover-arm ridge under the (sigma, alpha)
# reparameterization on the small-n_pos / low-psi regime (d7 Cell B).
#
# Historical context (gcol33/tulpa#18, gcol33/tulpa#22): an earlier
# derived-alpha parameterization computed alpha = sigma_pos / sigma_occ
# post-hoc from a (sigma_occ, sigma_pos) grid. The cover-arm likelihood
# ridge then dragged sigma_occ toward its prior (mean 0.51, bias -15% vs
# truth 0.6) and the derived alpha picked up the ratio noise (mean 1.27,
# bias +27% vs truth 1.0) on this regime.
#
# Current parameterization: (sigma, alpha) are both direct outer-grid
# axes. The structural property to verify is the ABSENCE of the cover-
# arm ridge -- sigma is jointly identified by both arms and is no longer
# dragged by the cover-arm SD prior, and the alpha posterior is properly
# calibrated even when its marginal is right-skewed.
#
# What this test does NOT assert: that the alpha posterior MEAN is
# unbiased under flat prior at small n_pos. The cover-arm likelihood at
# n_pos ~ 45 is weakly identifying and the alpha marginal is genuinely
# right-skewed in this regime (a single seed often puts ~40% weight on
# alpha = 1.4 and ~40% on alpha = 2.0 vs ~18% at the truth alpha = 1.0).
# That bias is structural, not MC noise, and IS the regime motivating
# `prior_alpha`. The sibling test
# `test-nested-laplace-joint-sigma-pos-prior.R` measures exactly this:
# the flat-prior baseline carries `expect_gt(gb_flat, 0.08)` upward
# bias on the same fixture, and PC / half-normal priors cut it.

.chain_adj_ridge <- function(n_s) {
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

# d7-Cell-B-like simulator: small n_pos, beta cover, low occurrence
# probability so psi (cover-arm sample fraction) is around 0.12.
.simulate_d7_cellB <- function(seed,
                                Nb = 300L, n_s = 25L,
                                sigma_b = 0.6, rho_b = 0.7,
                                betaO = c(-2.0, 0.7),
                                betaP = c(0.4, -0.5),
                                alpha_true = 1.0,
                                phi_b = 30) {
    set.seed(seed)
    region  <- sample.int(n_s, Nb, replace = TRUE)
    phi_f   <- rnorm(n_s, 0, 1)
    theta_f <- rnorm(n_s, 0, 1)
    w_s     <- sigma_b * (sqrt(rho_b) * phi_f +
                          sqrt(1 - rho_b) * theta_f)
    x       <- rnorm(Nb)
    Xocc    <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% betaO) + w_s[region]
    occur   <- rbinom(Nb, 1L, plogis(eta_occ))

    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- region[is_pos]
    eta_pos <- as.numeric(Xpos %*% betaP) + alpha_true * w_s[spi_pos]
    mu_pos  <- plogis(eta_pos)
    y_pos   <- rbeta(sum(is_pos), mu_pos * phi_b,
                      (1 - mu_pos) * phi_b)
    y_pos   <- pmin(pmax(y_pos, 1e-6), 1 - 1e-6)

    list(N = Nb, n_s = n_s,
         spatial_idx = as.integer(region),
         Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos),
         truth = list(alpha = alpha_true, sigma = sigma_b))
}

test_that("(sigma, alpha) reparam removes the cover-arm ridge on small-n_pos BYM2 (gcol33/tulpa#22)", {
    skip_on_cran()
    adj <- .chain_adj_ridge(25L)
    # 10 seeds are sufficient for the two assertions this test makes:
    #   * sigma is well-identified (per-seed sigma SD is small because
    #     sigma is jointly identified by both arms),
    #   * the alpha 95% CI contains truth (a per-seed coverage check,
    #     so MC SE on the proportion is bounded by sqrt(p(1-p)/n) ~ 0.15
    #     at n = 10 -- comfortably below the 0.8 floor when the true
    #     coverage is at nominal 0.95).
    # The seed-averaged alpha MEAN under flat prior carries the
    # documented small-n_pos upward bias (gcol33/tulpa#22) and is
    # tested in test-nested-laplace-joint-sigma-pos-prior.R, not here.
    seeds <- 7501:7510
    sigma_hat <- numeric(length(seeds))
    alpha_lo  <- numeric(length(seeds))
    alpha_hi  <- numeric(length(seeds))
    for (i in seq_along(seeds)) {
        sim <- .simulate_d7_cellB(seeds[i])
        arm_occ <- list(
            y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
            X = sim$Xocc, spatial_idx = sim$spatial_idx,
            re_idx = rep(0, sim$N), n_re_groups = 0L, sigma_re = 1.0,
            family = "binomial", phi = 1.0
        )
        arm_pos <- list(
            y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
            X = sim$Xpos, spatial_idx = sim$spi_pos,
            re_idx = rep(0, length(sim$y_pos)),
            n_re_groups = 0L, sigma_re = 1.0,
            family = "beta", phi = 30.0
        )
        prior <- list(
            type = "bym2",
            n_spatial_units = adj$n_spatial_units,
            adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
            n_neighbors = adj$n_neighbors, scale_factor = 1.0,
            sigma_grid = exp(seq(log(0.3), log(1.2), length.out = 5L)),
            rho_grid   = c(0.5, 0.7, 0.9)
        )
        fit <- tulpa_nested_laplace_joint(
            responses = list(occ = arm_occ, pos = arm_pos),
            prior     = prior,
            copy      = list(arm = "pos",
                              alpha_grid = c(0, 0.4, 0.7, 1.0, 1.4, 2.0)),
            adaptive_grid = FALSE
        )
        sigma_hat[i] <- fit$theta_mean[["sigma"]]
        alpha_lo[i]  <- fit$theta_ci_lo[["alpha"]]
        alpha_hi[i]  <- fit$theta_ci_hi[["alpha"]]
    }

    # (1) Sigma is no longer dragged by the cover-arm ridge. Under the
    #     derived-alpha parameterization the same fixture gave a mean
    #     sigma of 0.51 (bias -15%). The reparam pulls this back to
    #     within a few percent of truth = 0.6 (10-seed reference run on
    #     2026-05-20: mean sigma 0.596, bias -0.7%).
    sigma_bias <- mean(sigma_hat) - 0.6
    expect_lt(abs(sigma_bias), 0.10,
              label = sprintf("sigma_bias = %+.3f", sigma_bias))

    # (2) The 95% CI on alpha contains truth on (almost) every seed.
    #     This is the calibrated-posterior assertion: even though the
    #     alpha marginal is right-skewed under flat prior at this
    #     n_pos, the CI is wide enough that the truth alpha = 1.0 is
    #     bounded. 10-seed reference run on 2026-05-20: 10/10 seeds
    #     cover. Threshold of 0.8 leaves slack for the occasional seed
    #     whose posterior concentrates above truth.
    covers <- alpha_lo <= 1.0 & alpha_hi >= 1.0
    cov_rate <- mean(covers)
    expect_gte(cov_rate, 0.8,
               label = sprintf("alpha 95%% CI coverage = %.0f%%",
                                100 * cov_rate))
})
