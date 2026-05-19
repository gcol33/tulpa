# Phase 1 gate: 5-seed recovery smoke on the (sigma, alpha) reparameterization
# (gcol33/tulpa#22). Three regimes -- alpha_true in {0, 1, 2} -- covering no
# copy, equal-scale, strong copy. The 30-seed multi-block test in
# test-nested-laplace-joint-multi-recovery.R is the precise recovery contract
# at alpha = 1.5; this 5-seed smoke is a light per-regime regression that
# guards the (sigma, alpha) axis structure across the full alpha range.
#
# Layout: two-arm joint (binomial + binomial), single BYM2 block, chain
# adjacency on 25 sites. Identical simulator structure to the 30-seed test
# but smaller (no AR1/IID extras, faster fit).
#
# Diagnostic context (recorded so future regressions can be triaged fast):
# single-arm Laplace on the occupancy arm alone gives sigma_hat ~ 0.38 at
# N1 = 400, 25 chain sites, 3-point sigma grid {0.3, 0.6, 1.0}. The joint
# kernel matches that single-arm result to within 0.01 at alpha = 0 and
# alpha = 1, then pulls sigma_hat back toward truth at alpha = 2 when the
# copy arm carries enough signal to disambiguate sigma (sigma_hat ~ 0.57).
# This is the expected behaviour of the (sigma, alpha) parameterisation:
# the donor arm alone fixes sigma at small N only weakly, and cross-arm
# information at large alpha tightens the marginal. Thresholds below are
# therefore widened relative to the 30-seed multi-block contract.
#
# Acceptance thresholds (calibrated on the first 5-seed run):
#   * sigma:          median |sigma_hat - 0.6| / 0.6   < 0.45  per regime
#   * alpha = 0:      median |alpha_hat|               < 0.30
#                     median(alpha_median)             < 0.50
#   * alpha in {1,2}: median |alpha_hat - true| / true < 0.50
#                     95% CI coverage >= 0.60 (>= 3/5 seeds)
#
# Regression target: any change that pushes sigma_hat below ~0.3 (e.g. an
# off-by-alpha^2 error in the joint Hessian at alpha = 0) or distorts the
# alpha posterior off-grid (e.g. axis-swap in the kernel) will trip these.

skip_on_cran()

.phase1_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx     = as.integer(unlist(nbr)) - 1L,
        n_neighbors     = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

# Two-arm BYM2 simulator with explicit (sigma, alpha) truth on the copy arm.
.phase1_sim_joint <- function(seed, n_sites, N1, N2,
                              sigma_true, alpha_true, rho_b = 0.7,
                              beta1 = c(-0.5,  0.4),
                              beta2 = c( 0.3, -0.2)) {
    set.seed(seed)
    phi   <- rnorm(n_sites); phi   <- phi   - mean(phi)
    theta <- rnorm(n_sites); theta <- theta - mean(theta)
    w_unit <- sqrt(rho_b) * phi + sqrt(1 - rho_b) * theta

    s1 <- sample.int(n_sites, N1, replace = TRUE)
    s2 <- sample.int(n_sites, N2, replace = TRUE)
    X1 <- cbind(1, rnorm(N1))
    X2 <- cbind(1, rnorm(N2))

    eta1 <- as.numeric(X1 %*% beta1) +              sigma_true * w_unit[s1]
    eta2 <- as.numeric(X2 %*% beta2) + alpha_true * sigma_true * w_unit[s2]

    y1 <- rbinom(N1, 1L, plogis(eta1))
    y2 <- rbinom(N2, 1L, plogis(eta2))

    list(
        occ = list(y = y1, n_trials = rep(1L, N1), X = X1,
                   spatial_idx = s1, family = "binomial"),
        pos = list(y = y2, n_trials = rep(1L, N2), X = X2,
                   spatial_idx = s2, family = "binomial")
    )
}

test_that("(sigma, alpha) reparam recovers sigma/alpha across alpha in {0, 1, 2} (5 seeds)", {
    n_sites <- 25L
    N1      <- 400L
    N2      <- 400L
    adj     <- .phase1_chain_adj(n_sites)

    sigma_grid <- c(0.3, 0.6, 1.0)
    alpha_grid <- c(0, 0.5, 1.0, 1.5, 2.0, 2.5)
    rho_grid   <- c(0.5, 0.85)

    prior_tmpl <- c(list(type       = "bym2",
                         sigma_grid = sigma_grid,
                         rho_grid   = rho_grid),
                     adj,
                     list(scale_factor = 1.0))
    copy_spec <- list(arm = "pos", alpha_grid = alpha_grid)

    sigma_true <- 0.6
    regimes <- list(
        list(name = "alpha=0", alpha_true = 0.0),
        list(name = "alpha=1", alpha_true = 1.0),
        list(name = "alpha=2", alpha_true = 2.0)
    )
    seeds <- 7001L:7005L

    for (r in regimes) {
        sigma_hat <- numeric(length(seeds))
        alpha_hat <- numeric(length(seeds))
        alpha_med <- numeric(length(seeds))
        alpha_lo  <- numeric(length(seeds))
        alpha_hi  <- numeric(length(seeds))

        for (i in seq_along(seeds)) {
            sim <- .phase1_sim_joint(seeds[i],
                                     n_sites = n_sites, N1 = N1, N2 = N2,
                                     sigma_true = sigma_true,
                                     alpha_true = r$alpha_true)
            fit <- suppressWarnings(
                tulpa_nested_laplace_joint(
                    responses = sim,
                    prior     = prior_tmpl,
                    copy      = copy_spec,
                    adaptive_grid            = FALSE,
                    var_of_means_consistency = FALSE,
                    max_iter = 60L, tol = 1e-5
                )
            )
            expect_true(all(is.finite(fit$log_marginal)),
                        info = sprintf("%s seed %d: non-finite log_marginal",
                                        r$name, seeds[i]))

            sigma_hat[i] <- fit$theta_mean[["sigma"]]
            alpha_hat[i] <- fit$theta_mean[["alpha"]]
            alpha_med[i] <- fit$theta_median[["alpha"]]
            alpha_lo[i]  <- fit$theta_ci_lo[["alpha"]]
            alpha_hi[i]  <- fit$theta_ci_hi[["alpha"]]
        }

        rel_sig <- median(abs(sigma_hat - sigma_true) / sigma_true)
        info_sig <- sprintf("%s: median rel|sigma|=%.3f (sigma_hat=%s)",
                            r$name, rel_sig,
                            paste(sprintf("%.3f", sigma_hat), collapse = ","))
        expect_lt(rel_sig, 0.45, label = info_sig)

        if (r$alpha_true == 0) {
            abs_a    <- median(abs(alpha_hat))
            med_a    <- median(alpha_med)
            info_a   <- sprintf("%s: median |alpha|=%.3f, median(median)=%.3f",
                                r$name, abs_a, med_a)
            expect_lt(abs_a, 0.30, label = info_a)
            expect_lt(med_a, 0.50, label = info_a)
        } else {
            rel_a <- median(abs(alpha_hat - r$alpha_true) / r$alpha_true)
            cov_a <- mean(alpha_lo <= r$alpha_true &
                          alpha_hi >= r$alpha_true)
            info_a <- sprintf("%s: median rel|alpha|=%.3f, CI cov=%.2f",
                              r$name, rel_a, cov_a)
            expect_lt(rel_a, 0.50, label = info_a)
            expect_gte(cov_a, 0.60, label = info_a)
        }
    }
})
