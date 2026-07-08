# Parallel-vs-serial parity for the joint BYM2 nested-Laplace driver.
#
# The PR 1 speedup (dev_notes/speedup.md) adds outer-grid OpenMP via
# `n_threads_outer`. Posterior weights, means, SDs, and the derived alpha
# quantiles must match the serial path to within numerical noise from
# step-halving ordering. The parallel path warm-starts every cell from the
# centre-cell pilot mode rather than the previous-cell mode, so per-cell
# Newton iter counts can differ by +/- a few; the *converged log-marginal*
# should not.

.chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_neighbors <- vapply(nbr, length, integer(1))
    list(
        adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
        adj_col_idx    = as.integer(unlist(nbr)) - 1L,
        n_neighbors    = as.integer(n_neighbors),
        n_spatial_units = n_s
    )
}

.sim_joint_bym2 <- function(seed) {
    set.seed(seed)
    N <- 400L; n_s <- 60L
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- rnorm(n_s); theta <- rnorm(n_s)
    w_s   <- 0.6 * (sqrt(0.7) * phi + sqrt(0.3) * theta)
    x     <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos))
}

.fit_joint <- function(sim, n_threads_outer) {
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
        family = "gaussian", phi = 0.5
    )
    prior <- list(
        type = "bym2", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9)
    )
    arm_pos$field_coef <- list(name = "alpha", grid = c(0.3, 0.6, 1.0, 1.4, 1.8))
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        control = list(n_threads = 1L, n_threads_outer = n_threads_outer,
                       adaptive_grid = FALSE, var_of_means_consistency = FALSE)
    )
}

.fit_joint_tile <- function(sim, n_threads_outer, tile_warm) {
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
        family = "gaussian", phi = 0.5
    )
    prior <- list(
        type = "bym2", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9)
    )
    arm_pos$field_coef <- list(name = "alpha", grid = c(0.3, 0.6, 1.0, 1.4, 1.8))
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        control = list(n_threads = 1L, n_threads_outer = n_threads_outer,
                       tile_warm = tile_warm,
                       adaptive_grid = FALSE, var_of_means_consistency = FALSE)
    )
}

# Multi-block joint with two full-rank blocks (RW1 temporal + IID), fit through
# the SPARSE driver (force_sparse) so the outer grid runs through
# run_multi_block_nested_laplace_joint_sparse_impl with its per-outer-thread
# builder / scratch / spec / cache pool (gcol33/tulpa#58). Full-rank blocks make
# the inner Newton converge cleanly (rank-deficient ICAR/BYM2 over-iterate on the
# field null space -- a separate inner-solve issue), so serial and parallel land
# on the same converged mode and the parity is exact up to FP noise.
.fit_joint_rw1_iid_sparse <- function(seed, n_threads_outer) {
    set.seed(seed)
    N <- 360L; n_t <- 6L; n_u <- N
    t_idx <- sample.int(n_t, N, replace = TRUE)
    x <- rnorm(N); Xocc <- cbind(1, x)
    f_t <- cumsum(rnorm(n_t, sd = 0.5)); f_t <- f_t - mean(f_t)
    eta_occ <- as.numeric(Xocc %*% c(-0.2, 0.4)) + f_t[t_idx] + rnorm(N, 0, 0.3)
    occur <- rbinom(N, 1, plogis(eta_occ))
    is_pos <- occur == 1L
    Xpos  <- Xocc[is_pos, , drop = FALSE]
    t_pos <- t_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.1, -0.3)) + f_t[t_pos]
    y_pos <- rnorm(sum(is_pos), eta_pos, 0.5)
    arm_occ <- list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                    spatial_idx = rep(0L, N), re_idx = rep(0, N),
                    n_re_groups = 0L, sigma_re = 1.0, family = "binomial",
                    phi = 1.0)
    arm_pos <- list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                    spatial_idx = rep(0L, length(y_pos)),
                    re_idx = rep(0, length(y_pos)), n_re_groups = 0L,
                    sigma_re = 1.0, family = "gaussian", phi = 0.5)
    prior <- list(
        list(type = "rw1", temporal_idx = list(t_idx, t_pos), n_times = n_t,
             tau_grid = c(0.5, 1.0, 2.0)),
        list(type = "iid", obs_idx = list(seq_len(N), which(is_pos)),
             n_units = n_u, sigma_grid = c(0.5, 1.0))
    )
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos), prior = prior,
        control = list(n_threads = 1L, n_threads_outer = n_threads_outer,
                       force_sparse = TRUE, integration = "grid",
                       max_iter = 100L, tol = 1e-8, diagnose_k = FALSE,
                       var_of_means_consistency = FALSE))
}

test_that("multi-block SPARSE path parallel matches serial across seeds", {
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise the parallel sparse path")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))

    seeds <- 7160L + seq_len(4L)
    for (seed in seeds) {
        fit_s <- .fit_joint_rw1_iid_sparse(seed, n_threads_outer = 1L)
        fit_p <- .fit_joint_rw1_iid_sparse(seed, n_threads_outer = n_outer)
        expect_true(all(fit_s$n_iter < 100L),
                    info = sprintf("seed=%d serial converged", seed))
        expect_equal(fit_p$log_marginal, fit_s$log_marginal, tolerance = 1e-6,
                     info = sprintf("seed=%d log_marginal", seed))
        expect_equal(fit_p$weights, fit_s$weights, tolerance = 1e-6,
                     info = sprintf("seed=%d weights", seed))
        expect_equal(fit_p$theta_mean, fit_s$theta_mean, tolerance = 1e-5,
                     info = sprintf("seed=%d theta_mean", seed))
        expect_equal(fit_p$modes, fit_s$modes, tolerance = 1e-4,
                     info = sprintf("seed=%d modes", seed))
    }
})

test_that("joint BYM2 parallel path matches serial across seeds", {
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise the parallel path")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))

    seeds <- 7100L + seq_len(5L)
    for (seed in seeds) {
        sim <- .sim_joint_bym2(seed)
        fit_s <- .fit_joint(sim, n_threads_outer = 1L)
        fit_p <- .fit_joint(sim, n_threads_outer = n_outer)

        # Log-marginal per cell should match to numerical precision. The two
        # paths warm-start differently so Newton iter counts may differ but
        # the converged log-marginal at each cell is determined by the
        # likelihood and prior, not the path taken.
        expect_equal(fit_p$log_marginal, fit_s$log_marginal, tolerance = 1e-6,
                     info = sprintf("seed=%d log_marginal", seed))

        # Posterior weights derive from log_marginal, so they must match too.
        expect_equal(fit_p$weights, fit_s$weights, tolerance = 1e-6,
                     info = sprintf("seed=%d weights", seed))

        # Theta moments (sigma, rho, alpha) and posterior means/SDs.
        expect_equal(fit_p$theta_mean, fit_s$theta_mean, tolerance = 1e-6,
                     info = sprintf("seed=%d theta_mean", seed))
        expect_equal(fit_p$theta_sd, fit_s$theta_sd, tolerance = 1e-4,
                     info = sprintf("seed=%d theta_sd", seed))

        # Per-cell Newton modes — same posterior weights AND same modes mean
        # downstream posterior moments computed via sum(w * mode_col) match.
        expect_equal(fit_p$modes, fit_s$modes, tolerance = 1e-4,
                     info = sprintf("seed=%d modes", seed))
    }
})

test_that("joint BYM2 tile_warm path matches no-tile path across seeds", {
    # Phase 2 (dev_notes/speedup.md, gcol33/tulpa#26): tile_warm groups
    # outer-grid cells sharing every coordinate except the copy coefficient
    # alpha and warm-starts the alpha cells from their tile pilot. The
    # joint mode at every cell is unchanged by the warm-start (Newton
    # converges to the same MAP given enough iters), so the tile path
    # must match the no-tile path at this problem size where both Newton
    # paths converge cleanly inside `max_iter`.
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise tile_warm")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))

    seeds <- 7200L + seq_len(5L)
    iters_off_total <- 0L
    iters_on_total  <- 0L
    for (seed in seeds) {
        sim <- .sim_joint_bym2(seed)
        fit_off <- .fit_joint_tile(sim, n_outer, tile_warm = FALSE)
        fit_on  <- .fit_joint_tile(sim, n_outer, tile_warm = TRUE)

        # Posterior equivalence: log-marginal per cell must match (warm-
        # start choice only affects iter count, not the converged mode at
        # this scale).
        expect_equal(fit_on$log_marginal, fit_off$log_marginal,
                     tolerance = 1e-5,
                     info = sprintf("seed=%d log_marginal", seed))
        expect_equal(fit_on$weights, fit_off$weights, tolerance = 1e-5,
                     info = sprintf("seed=%d weights", seed))
        expect_equal(fit_on$theta_mean, fit_off$theta_mean,
                     tolerance = 1e-5,
                     info = sprintf("seed=%d theta_mean", seed))
        expect_equal(fit_on$theta_sd, fit_off$theta_sd, tolerance = 1e-4,
                     info = sprintf("seed=%d theta_sd", seed))
        expect_equal(fit_on$modes, fit_off$modes, tolerance = 1e-4,
                     info = sprintf("seed=%d modes", seed))

        iters_off_total <- iters_off_total + sum(fit_off$n_iter)
        iters_on_total  <- iters_on_total  + sum(fit_on$n_iter)
    }
    # Iter count is data-dependent: on a small grid with ~9 tiles the
    # Tier-2 pilot pass can absorb most of Tier-3's savings. Check that
    # the *aggregate* iter count over 5 seeds is no worse than the no-
    # tile baseline (within a 10% tolerance for Tier-2 overhead).
    expect_lte(iters_on_total, ceiling(iters_off_total * 1.1),
               label = "aggregate Newton iters over 5 seeds")
})

test_that("tile_warm is a no-op when n_threads_outer == 1", {
    skip_on_cran()
    # Serial path: tile metadata is ignored by run_nested_laplace_grid, so
    # tile_warm = TRUE vs FALSE must produce identical results. Guards
    # against accidental tile-path activation outside the parallel branch.
    sim <- .sim_joint_bym2(7301L)
    fit_off <- .fit_joint_tile(sim, n_threads_outer = 1L, tile_warm = FALSE)
    fit_on  <- .fit_joint_tile(sim, n_threads_outer = 1L, tile_warm = TRUE)
    expect_identical(fit_on$log_marginal, fit_off$log_marginal)
    expect_identical(fit_on$n_iter, fit_off$n_iter)
})
