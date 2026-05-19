# Joint BYM2 — sigma and alpha posteriors under the small-n_pos / low-psi
# regime (d7 Cell B).
#
# Historical context (gcol33/tulpa#18, gcol33/tulpa#22): an earlier
# derived-alpha parameterization computed alpha = sigma_pos / sigma_occ
# post-hoc from a (sigma_occ, sigma_pos) grid; the cover-arm likelihood
# ridge then dragged sigma_occ toward its prior (mean 0.51, bias -15% vs
# truth 0.6) and the derived alpha picked up the ratio noise (mean 1.27,
# bias +27% vs truth 1.0) on this regime.
#
# Current parameterization: (sigma, alpha) are both direct outer-grid axes.
# Sigma is jointly identified by both arms and alpha is grid-evaluated
# without ratio noise, so the seed-averaged posterior means should sit
# close to truth even when n_pos is small.

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

test_that("sigma and alpha are unbiased on small-n_pos BYM2 (gcol33/tulpa#22)", {
    skip_on_cran()
    adj <- .chain_adj_ridge(25L)
    seeds <- 7501:7510
    sigma_hat <- numeric(length(seeds))
    alpha_hat <- numeric(length(seeds))
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
        alpha_hat[i] <- fit$theta_mean[["alpha"]]
    }
    # Under the derived-alpha parameterization the same regime showed
    # sigma mean 0.51 (bias -15%) and alpha mean 1.27 (bias +27%). With
    # (sigma, alpha) both as direct grid axes the seed-averaged means
    # should sit within ~15-25% of truth.
    sigma_bias <- mean(sigma_hat) - 0.6
    alpha_bias <- mean(alpha_hat) - 1.0
    expect_lt(abs(sigma_bias), 0.12)
    expect_lt(abs(alpha_bias), 0.25)
})
