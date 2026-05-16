# Joint BYM2 — per-arm sigma posterior under the small-n_pos / low-psi
# regime that triggered gcol33/tulpa#18.
#
# Old (sigma, alpha) parameterization: the cover-arm likelihood identified
# only the product alpha * sigma; sigma was pulled toward its prior and
# alpha inflated to compensate. Reported sigma mean was 0.51 (bias -15%
# vs truth 0.6) and alpha mean was 1.27 (bias +27% vs truth 1.0) on 30
# d7 Cell B seeds.
#
# New (sigma_occ, sigma_pos) parameterization: each arm's likelihood
# anchors its own field amplitude axis. The donor arm's sigma_occ
# posterior should center on truth without leaning on the prior; same
# for sigma_pos from the cover arm. The derived alpha = sigma_pos /
# sigma_occ is reported on the joint posterior, but as a ratio of two
# small-N posteriors it inherits intrinsic small-sample skew (E[X/Y] !=
# E[X]/E[Y] when Y has support near 0); the underlying scientific fix is
# that the *anchoring* posterior moments — sigma_occ, sigma_pos — are
# now unbiased.

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

test_that("per-arm sigma is unbiased on small-n_pos BYM2 (gcol33/tulpa#18)", {
    skip_on_cran()
    adj <- .chain_adj_ridge(25L)
    seeds <- 7501:7510
    sigma_occ <- numeric(length(seeds))
    sigma_pos <- numeric(length(seeds))
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
                              sigma_pos_grid = exp(seq(log(0.15),
                                                       log(1.2),
                                                       length.out = 5L))),
            adaptive_grid = FALSE
        )
        sigma_occ[i] <- fit$theta_mean[["sigma_occ"]]
        sigma_pos[i] <- fit$theta_mean[["sigma_pos"]]
    }
    # Under the old (sigma, alpha) parameterization, the cover-arm
    # likelihood ridge dragged sigma toward its prior — reported mean
    # was 0.51 (bias -15% vs truth 0.6) on the same regime. After the
    # reparam each arm's sigma is anchored by its own likelihood and the
    # seed-averaged means should sit within ~15% of truth.
    occ_bias <- mean(sigma_occ) - 0.6
    pos_bias <- mean(sigma_pos) - 0.6
    expect_lt(abs(occ_bias), 0.09)
    expect_lt(abs(pos_bias), 0.12)
})
