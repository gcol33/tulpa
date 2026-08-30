# Regularizing hyperprior on alpha (gcol33/tulpa#22).
#
# At small n_pos the cover-arm likelihood is weakly identifying and the
# marginal on alpha is right-skewed. The posterior median of alpha
# overshoots truth on the D7 Cell B regime (n_pos ~ 45, alpha_true = 1.0).
# The well-identified sigma axis doesn't compensate.
#
# A regularizing hyperprior on alpha -- PC (`pc.prec`) or half-normal --
# tightens the upper tail of the weakly-identified marginal without
# biasing the modal cell when the data identifies it. This test fits the
# D7 Cell B-like fixture 30 times with and without the prior and asserts:
#
#   (1) the flat-prior baseline carries the documented small-n_pos
#       upward bias on alpha (sanity check that the fixture exercises
#       the regime tulpa#22 targets),
#   (2) the documented-default pc.prec(U=4.0, alpha=0.01) cuts the
#       alpha geometric bias without over-shrinking past truth,
#   (3) half_normal(scale=1.0) does the same, more strongly,
#   (4) the regularizer on alpha does not corrupt the well-identified
#       donor-arm amplitude sigma (cross-axis coupling check).
#
# What U means here is set by the measure the alpha axis carries with no
# prior: the axis is integrated on log alpha, so the flat default is
# already a 1/alpha shrinkage, and a PC prior only shrinks harder than the
# baseline once lambda = -log(a)/U is large enough to beat it. That is why
# the recommended U is not "the upper end of plausible alpha" alone.
# Swept on this fixture's own 50 seeds (dev_notes/audit0830/
# sweep_alpha_prior.R), alpha geometric bias against truth 1.0, with the
# coupled donor sigma against its truth of 0.6:
#
#   flat  +0.056 (sigma 0.644)   pc U=1  -0.205 (0.746)   hn 0.25 -0.373
#   pc U=8 +0.116 (0.614)        pc U=2  -0.071 (0.683)   hn 0.5  -0.161
#   pc U=4 +0.049 (0.635)        pc U=0.5 -0.350 (0.838)  hn 1    +0.040
#                                pc U=0.25 -0.491 (0.941) hn 2    +0.148
#
# Monotone in U, crossing truth near U ~ 3.4, and the over-shrinking arms
# reproduce tulpa#22's cross-axis mechanism exactly: pulling alpha below 1
# lifts the coupled sigma above 0.6, up to 0.94 at U=0.25. The defaults
# moved from U=8 / scale=2 to U=4 / scale=1 when the joint driver's priors
# stopped being integrated in the wrong coordinate (gcol33/tulpa#625); the
# earlier numbers were calibrated against a realised prior of pi(a)/a.
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

.fit_pp <- function(sim, adj, prior_alpha = NULL) {
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
        responses = list(
            occ = arm_occ,
            pos = modifyList(arm_pos, list(field_coef = list(
                name = "alpha", grid = c(0, 0.4, 0.7, 1.0, 1.4, 2.0))))
        ),
        prior     = prior,
        prior_alpha   = prior_alpha,
        control = list(adaptive_grid = FALSE)
    )
}

test_that("alpha hyperprior gently reduces small-n_pos bias without corrupting sigma (gcol33/tulpa#22)", {
    skip_on_cran()
    adj   <- .chain_adj_pp(25L)
    # 50 seeds: the per-seed alpha median is high-variance (per-seed sd ~0.16),
    # so a small seed set gives an unstable geometric-bias baseline.
    seeds <- 7501:7550
    n_seeds <- length(seeds)

    alpha_flat <- numeric(n_seeds)
    alpha_pc   <- numeric(n_seeds)
    alpha_hn   <- numeric(n_seeds)
    sig_flat   <- numeric(n_seeds)
    sig_pc     <- numeric(n_seeds)

    for (i in seq_along(seeds)) {
        sim <- .simulate_d7_pp(seeds[i])
        f_flat <- .fit_pp(sim, adj, prior_alpha = NULL)
        f_pc   <- .fit_pp(sim, adj,
                          prior_alpha = list("pc.prec", c(4.0, 0.01)))
        f_hn   <- .fit_pp(sim, adj,
                          prior_alpha = list("half_normal", 1.0))
        alpha_flat[i] <- f_flat$theta_median[["alpha"]]
        alpha_pc[i]   <- f_pc$theta_median[["alpha"]]
        alpha_hn[i]   <- f_hn$theta_median[["alpha"]]
        sig_flat[i]   <- f_flat$theta_mean[["sigma"]]
        sig_pc[i]     <- f_pc$theta_mean[["sigma"]]
    }

    geom_bias <- function(hat, truth) {
        ok <- is.finite(hat) & hat > 0
        exp(mean(log(hat[ok]))) - truth
    }
    gb_flat <- geom_bias(alpha_flat, 1.0)
    gb_pc   <- geom_bias(alpha_pc,   1.0)
    gb_hn   <- geom_bias(alpha_hn,   1.0)

    info_str <- sprintf(
        "geom_bias alpha: flat=%.3f pc.prec(U=4)=%.3f half_normal(1)=%.3f | mean sigma: flat=%.3f pc=%.3f",
        gb_flat, gb_pc, gb_hn, mean(sig_flat), mean(sig_pc)
    )

    # (1) Sanity: the fixture exercises the small-n_pos regime, so the flat
    #     baseline carries a small upward bias on alpha (measured 0.056).
    expect_gt(gb_flat, 0.02, label = info_str)

    # (2) The recommended default pc.prec(U=4, alpha=0.01) shrinks the bias
    #     toward zero (measured 0.049) without over-shrinking past the truth.
    #     A too-small U drives the alpha median below 1 and, through the
    #     alpha * sigma copy axis, inflates the coupled donor sigma; U=4 keeps
    #     alpha on the correct side of the truth.
    expect_lt(abs(gb_pc), 0.09, label = info_str)   # bias stays small
    expect_gt(gb_pc, -0.05, label = info_str)       # no over-shrink past truth

    # (3) half_normal(scale = 1.0) is likewise gentle: it regularizes the tail
    #     without over-shrinking (measured 0.040 bias).
    expect_lt(abs(gb_hn), 0.12, label = info_str)
    expect_gt(gb_hn, -0.05, label = info_str)

    # (5) tulpa#22's claim itself, which the bounds above do not test: the
    #     regularizer BUYS something against the axis's own no-prior measure.
    #     Both fits are deterministic on these seeds, so this is an exact
    #     comparison, not a sampling one (measured 0.049 and 0.040 against
    #     0.056). It is the assertion that fails if a default is recommended
    #     that shrinks less than the flat axis already does.
    expect_lt(abs(gb_pc), abs(gb_flat), label = info_str)
    expect_lt(abs(gb_hn), abs(gb_flat), label = info_str)

    # (4) The alpha regularizer leaves the well-identified donor amplitude
    #     sigma near its truth of 0.6 (measured 0.635 under pc.prec(U=4)); the
    #     flat baseline recovers sigma to 0.644.
    expect_lt(abs(mean(sig_pc) - 0.6),   0.12, label = info_str)
    expect_lt(abs(mean(sig_flat) - 0.6), 0.10, label = info_str)
})

test_that("PC / half_normal log-density is finite at the 0 boundary", {
    # Regression for the `ok <- s > 0` -> `ok <- s >= 0` fix. The PC
    # prior density at 0 is the finite limit `lambda`, not zero;
    # half-normal at 0 is `2 / (scale * sqrt(2 * pi))`. Earlier impl
    # returned `-Inf` for s = 0, which zeroed the alpha = 0 grid cell
    # entirely and biased the posterior when the truth sat on the
    # boundary (e.g. INLAabun D3 alpha_true = 0.0).
    f_pc <- tulpa:::.joint_parse_sigma_prior(list("pc.prec", c(1.0, 0.01)),
                                              "prior_alpha")
    expect_true(is.finite(f_pc(0)))
    expect_equal(f_pc(0), log(-log(0.01) / 1.0))

    f_hn <- tulpa:::.joint_parse_sigma_prior(list("half_normal", 0.5),
                                              "prior_alpha")
    expect_true(is.finite(f_hn(0)))
    expect_equal(f_hn(0), log(2) - 0.5 * log(2 * pi) - log(0.5))
})

test_that("alpha = 0 grid cell is not zeroed by the prior", {
    # Integration-level check: with an alpha grid that includes 0 and
    # truth alpha = 0, the posterior should concentrate near 0. Pre-fix
    # the boundary bug pushed the posterior to the next-smallest cell,
    # surfacing as a +500% bias on alpha. One seed is enough to catch a
    # regression of the boundary.
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
        responses = list(
            occ = arm_occ,
            pos = modifyList(arm_pos, list(field_coef = list(
                name = "alpha", grid = c(0, 0.5, 1.0, 1.5, 2.0))))
        ),
        prior     = prior,
        prior_alpha     = list("pc.prec", c(1.0, 0.01)),
        control = list(adaptive_grid = FALSE)
    )
    # With alpha_truth = 0, the alpha posterior median should be small
    # (well below 0.2). Pre-fix the median jumped to ~0.5 because the
    # zero cell was eliminated. 0.2 is generous headroom for one seed.
    expect_lt(fit$theta_median[["alpha"]], 0.2)
})
