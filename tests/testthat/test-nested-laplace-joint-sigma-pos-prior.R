# Regularizing hyperprior on sigma_pos (gcol33/tulpa#22).
#
# At small n_pos the cover-arm likelihood is weakly identifying and the
# marginal on sigma_pos is right-skewed. The derived
#   alpha = sigma_pos / sigma_occ
# inherits the skew: even after marginalizing the joint sigma grid (the
# tulpa#21 fix to plug-in-MAP-from-skewed-axis), the posterior median of
# alpha overshoots truth on the D7 Cell B regime (n_pos ~ 45,
# alpha_true = 1.0). The well-identified sigma_occ axis doesn't
# compensate.
#
# A regularizing hyperprior on sigma_pos -- PC (`pc.prec`) or half-normal
# -- tightens the upper tail of the weakly-identified marginal without
# biasing the modal cell when the data identifies it. This test fits the
# D7 Cell B-like fixture 30 times with and without the prior and asserts:
#
#   (1) the flat-prior baseline carries the documented small-n_pos
#       upward bias on alpha (sanity check that the fixture exercises
#       the regime tulpa#22 targets),
#   (2) the documented-default pc.prec(U=1.0, alpha=0.01) cuts the
#       alpha geometric bias by ~half (target on this fixture: <= 9%;
#       observed ~7.3% on a 30-seed sweep dated 2026-05-19; allows
#       headroom for seed variation),
#   (3) a stronger half_normal(scale=0.5) gives at least 15% relative
#       reduction vs flat (the documented scale=1.0 default barely
#       changes the bias on this fixture; the test exercises a regime
#       where half_normal demonstrably regularizes),
#   (4) the regularizer on sigma_pos does not corrupt the
#       well-identified donor-arm amplitude sigma_occ (cross-axis
#       coupling check; observed pc.prec drag ~0.06 on truth = 0.6).
#
# The #22 issue's stated target of <5% geom bias on INLAabun's D7 Cell B
# is a downstream validation on INLAabun's own harness (its fixture
# starts at +8.5% flat bias); this fixture is similar but not identical
# and starts at +13.9% flat bias, so the in-tulpa target is calibrated
# to "the prior regularizes meaningfully and consistently" rather than a
# specific absolute floor.
#
# Skipped on CRAN: 30 joint BYM2 fits x 3 prior settings cost ~30-40s.

.chain_adj_pp <- function(n_s) {
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

# d7-Cell-B-like simulator. Same shape as test-nested-laplace-joint-
# alpha-ridge.R's local `.simulate_d7_cellB` but renamed to avoid name
# collisions across the test files.
.simulate_d7_pp <- function(seed,
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

.fit_pp <- function(sim, adj, prior_sigma_pos = NULL) {
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
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior     = prior,
        copy      = list(arm = "pos",
                          sigma_pos_grid = exp(seq(log(0.15),
                                                   log(1.2),
                                                   length.out = 5L))),
        prior_sigma_pos = prior_sigma_pos,
        adaptive_grid   = FALSE
    )
}

test_that("pc.prec on sigma_pos cuts alpha bias at small n_pos (gcol33/tulpa#22)", {
    skip_on_cran()
    adj   <- .chain_adj_pp(25L)
    seeds <- 7501:7530
    n_seeds <- length(seeds)

    alpha_flat   <- numeric(n_seeds)
    alpha_pc     <- numeric(n_seeds)
    alpha_hn     <- numeric(n_seeds)
    sig_occ_flat <- numeric(n_seeds)
    sig_occ_pc   <- numeric(n_seeds)

    for (i in seq_along(seeds)) {
        sim <- .simulate_d7_pp(seeds[i])
        f_flat <- .fit_pp(sim, adj, prior_sigma_pos = NULL)
        f_pc   <- .fit_pp(sim, adj,
                          prior_sigma_pos = list("pc.prec", c(1.0, 0.01)))
        f_hn   <- .fit_pp(sim, adj,
                          prior_sigma_pos = list("half_normal", 0.5))
        alpha_flat[i]   <- f_flat$theta_median[["alpha"]]
        alpha_pc[i]     <- f_pc$theta_median[["alpha"]]
        alpha_hn[i]     <- f_hn$theta_median[["alpha"]]
        sig_occ_flat[i] <- f_flat$theta_mean[["sigma_occ"]]
        sig_occ_pc[i]   <- f_pc$theta_mean[["sigma_occ"]]
    }

    geom_bias <- function(hat, truth) {
        ok <- is.finite(hat) & hat > 0
        exp(mean(log(hat[ok]))) - truth
    }
    gb_flat <- geom_bias(alpha_flat, 1.0)
    gb_pc   <- geom_bias(alpha_pc,   1.0)
    gb_hn   <- geom_bias(alpha_hn,   1.0)

    info_str <- sprintf(
        "geom_bias alpha: flat=%.3f  pc.prec=%.3f  half_normal=%.3f | mean sigma_occ: flat=%.3f pc=%.3f",
        gb_flat, gb_pc, gb_hn, mean(sig_occ_flat), mean(sig_occ_pc)
    )

    # (1) Sanity: flat-prior baseline carries small-n_pos upward bias.
    #     Precondition for the regularization test below -- if this
    #     fails, the fixture has stopped exercising the regime tulpa#22
    #     targets and the rest of the assertions aren't meaningful.
    expect_gt(gb_flat, 0.08, label = info_str)

    # (2) Documented default `pc.prec(U=1.0, alpha=0.01)` regularizes
    #     meaningfully. On a 30-seed sweep (2026-05-19) this cut bias
    #     from +13.9% (flat) to +7.3% on the same fixture; 0.09 leaves
    #     headroom for seed-set variation.
    expect_lt(abs(gb_pc), 0.09, label = info_str)

    # (3) `half_normal(scale = 0.5)` gives at least 15% relative
    #     reduction vs flat (the looser scale=1.0 default barely moves
    #     the bias on this fixture -- the test exercises the regime
    #     where half_normal demonstrably regularizes).
    expect_lt(abs(gb_hn), 0.85 * abs(gb_flat), label = info_str)

    # (4) The regularizer on sigma_pos doesn't corrupt the
    #     well-identified donor-arm amplitude. Truth is 0.6; pc.prec(1,
    #     0.01) pulls sigma_occ down via joint coupling but should stay
    #     within ~0.10 of truth (observed offset ~0.06 on 30 seeds).
    expect_lt(abs(mean(sig_occ_pc) - 0.6), 0.10, label = info_str)
})

test_that("PC / half_normal log-density is finite at sigma = 0 boundary", {
    # Regression for the `ok <- s > 0` -> `ok <- s >= 0` fix. The PC
    # prior density at sigma=0 is the finite limit `lambda`, not zero;
    # half-normal at 0 is `2 / (scale * sqrt(2 * pi))`. Earlier impl
    # returned `-Inf` for s = 0, which zeroed the sigma_pos = 0 grid
    # cell entirely and biased derived alpha when the truth sat on the
    # boundary (e.g. INLAabun D3 alpha_true = 0.0).
    f_pc <- tulpa:::.joint_parse_sigma_prior(list("pc.prec", c(1.0, 0.01)),
                                              "prior_sigma_pos")
    expect_true(is.finite(f_pc(0)))
    expect_equal(f_pc(0), log(-log(0.01) / 1.0))

    f_hn <- tulpa:::.joint_parse_sigma_prior(list("half_normal", 0.5),
                                              "prior_sigma_pos")
    expect_true(is.finite(f_hn(0)))
    expect_equal(f_hn(0), log(2) - 0.5 * log(2 * pi) - log(0.5))
})

test_that("sigma_pos = 0 grid cell is not zeroed by the prior", {
    # Integration-level check: with a sigma_pos grid that includes 0 and
    # truth alpha = 0 (sigma_pos truth = 0), the posterior should
    # concentrate near 0. Pre-fix the boundary bug pushed the posterior
    # to the next-smallest cell, surfacing as a +500% bias on derived
    # alpha. One seed is enough to catch a regression of the boundary.
    skip_on_cran()
    adj <- .chain_adj_pp(25L)
    sim <- .simulate_d7_pp(seed = 7501L, alpha_true = 0.0)
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
        sigma_grid = c(0.3, 0.6, 0.9),
        rho_grid   = c(0.5, 0.7, 0.9)
    )
    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior     = prior,
        copy      = list(arm = "pos",
                          sigma_pos_grid = c(0.0, 0.3, 0.6, 0.9, 1.2)),
        prior_sigma_pos = list("pc.prec", c(1.0, 0.01)),
        adaptive_grid   = FALSE
    )
    # With sigma_pos_truth = 0, alpha posterior median should be small
    # (well below 0.2). Pre-fix the median jumped to ~0.5 because the
    # zero cell was eliminated. 0.2 is generous headroom for one seed.
    expect_lt(fit$theta_median[["alpha"]], 0.2)
})
