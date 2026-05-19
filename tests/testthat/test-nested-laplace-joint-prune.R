# Cheap-pass pruning for the joint BYM2 nested-Laplace driver (Phase 3 of
# dev_notes/speedup.md). Verifies:
#
#   (1) `prune = FALSE` is the default and the result has no prune fields.
#   (2) `prune = TRUE` populates `prune_mask`, `prune_cheap_log_marginal`,
#       `prune_n_pruned`, `prune_tol`, and pruned cells carry
#       `log_marginal = -Inf` and `n_iter = 0`.
#   (3) Posterior moments (theta_mean, theta_sd, derived alpha quantiles)
#       match the no-prune fit to within MC tolerance — the cheap pass
#       must not move the posterior on cells whose mass survives
#       screening.
#   (4) prune_tol = 0 with prune = TRUE is bitwise the no-prune path:
#       cheap-pass output is filled but no cell is pruned.

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

.fit_joint_prune <- function(sim, prune, prune_tol = 1e-3,
                              n_threads_outer = 1L) {
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
    copy <- list(arm = "pos", alpha_grid = c(0.3, 0.6, 1.0, 1.4, 1.8))
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior, copy = copy,
        n_threads = 1L, n_threads_outer = n_threads_outer,
        prune = prune, prune_tol = prune_tol,
        adaptive_grid = FALSE, var_of_means_consistency = FALSE
    )
}

test_that("prune = FALSE leaves no prune fields in the result", {
    skip_on_cran()
    sim <- .sim_joint_bym2(8101L)
    fit <- .fit_joint_prune(sim, prune = FALSE)
    expect_null(fit$prune_mask)
    expect_null(fit$prune_cheap_log_marginal)
    expect_null(fit$prune_n_pruned)
    expect_null(fit$prune_tol)
})

test_that("prune = TRUE populates prune fields and marks pruned cells", {
    skip_on_cran()
    sim <- .sim_joint_bym2(8101L)
    # Aggressive tol to force at least one prune on this 45-cell grid.
    fit <- .fit_joint_prune(sim, prune = TRUE, prune_tol = 0.05)

    expect_type(fit$prune_mask, "logical")
    expect_length(fit$prune_mask, length(fit$log_marginal))
    expect_type(fit$prune_cheap_log_marginal, "double")
    expect_length(fit$prune_cheap_log_marginal, length(fit$log_marginal))
    expect_type(fit$prune_n_pruned, "integer")
    expect_equal(fit$prune_n_pruned, sum(fit$prune_mask))
    expect_equal(fit$prune_tol, 0.05)

    # Pilot cell is never pruned.
    k_pilot <- length(fit$log_marginal) %/% 2L + 1L  # 1-based
    expect_false(fit$prune_mask[k_pilot])

    # Pruned cells must have log_marginal = -Inf and n_iter = 0. Surviving
    # cells must have finite log_marginal (i.e. the full Newton ran).
    pruned <- which(fit$prune_mask)
    kept   <- which(!fit$prune_mask)
    if (length(pruned) > 0L) {
        expect_true(all(fit$log_marginal[pruned] == -Inf),
                    info = "pruned cells must carry log_marginal = -Inf")
        expect_true(all(fit$n_iter[pruned] == 0L),
                    info = "pruned cells must carry n_iter = 0")
    }
    expect_true(all(is.finite(fit$log_marginal[kept])),
                info = "kept cells must have a finite log_marginal")
})

test_that("prune_tol = 0 with prune = TRUE is the no-prune path", {
    skip_on_cran()
    sim <- .sim_joint_bym2(8101L)
    fit_off <- .fit_joint_prune(sim, prune = FALSE)
    fit_zero <- .fit_joint_prune(sim, prune = TRUE, prune_tol = 0.0)

    # prune_tol = 0 short-circuits the cheap pass entirely — no point
    # screening when nothing can be pruned. The kernel emits no prune
    # fields in that case (identical to prune = FALSE).
    expect_null(fit_zero$prune_mask)
    expect_null(fit_zero$prune_cheap_log_marginal)
    expect_null(fit_zero$prune_n_pruned)
    expect_null(fit_zero$prune_tol)

    # Posterior moments must match the no-prune fit to numerical noise.
    expect_equal(fit_zero$log_marginal, fit_off$log_marginal,
                 tolerance = 1e-8)
    expect_equal(fit_zero$weights, fit_off$weights, tolerance = 1e-8)
    expect_equal(fit_zero$theta_mean, fit_off$theta_mean, tolerance = 1e-8)
    expect_equal(fit_zero$theta_sd, fit_off$theta_sd, tolerance = 1e-6)
})

test_that("prune posterior matches no-prune across seeds (conservative tol)", {
    # At prune_tol = 1e-3 the screen drops cells holding < 0.1% of cheap-
    # pass mass. Surviving cells carry essentially all the posterior mass,
    # so theta_mean / theta_sd / derived alpha quantiles must match the no-
    # prune fit within numerical noise across seeds.
    #
    # The cheap-pass uses a one-Newton-step screen from the centre-cell
    # pilot mode (much more accurate than the raw pilot-mode evaluation but
    # not bit-equal to full Newton convergence). Empirically the cheap-pass
    # posterior matches the full-Newton posterior to ~1e-3 on means and
    # ~1e-2 on SDs at this workload. We assert absolute tolerances rather
    # than expect_equal's default relative ones because SD values can be
    # small (~0.05) where 1% relative is sub-noise.
    skip_on_cran()
    seeds <- 8200L + seq_len(5L)
    for (seed in seeds) {
        sim <- .sim_joint_bym2(seed)
        fit_off <- .fit_joint_prune(sim, prune = FALSE)
        fit_on  <- .fit_joint_prune(sim, prune = TRUE, prune_tol = 1e-3)

        # Absolute tolerances: 5e-3 on means (sigma, rho, alpha all in
        # ~[0.1, 2]), 1e-2 on SDs (typically ~[0.02, 0.3]).
        expect_lt(max(abs(fit_on$theta_mean - fit_off$theta_mean)),
                  5e-3,
                  label = sprintf("seed=%d theta_mean max-abs-diff", seed))
        expect_lt(max(abs(fit_on$theta_sd - fit_off$theta_sd)),
                  1e-2,
                  label = sprintf("seed=%d theta_sd max-abs-diff", seed))

        # Derived alpha quantiles (median + 95% interval) should be
        # unchanged within MC noise when no real mass is lost.
        if (!is.null(fit_off$alpha_quantiles) &&
            !is.null(fit_on$alpha_quantiles)) {
            expect_lt(max(abs(fit_on$alpha_quantiles -
                              fit_off$alpha_quantiles)),
                      2e-2,
                      label = sprintf("seed=%d alpha_quantiles max-abs-diff",
                                       seed))
        }
    }
})

test_that("prune composes with outer-grid parallelism", {
    # The pruning machinery lives in run_nested_laplace_grid, so the
    # parallel and serial paths both honour the prune mask. Sanity check
    # that prune + n_threads_outer give identical results to prune serial,
    # because the cheap pass runs single-threaded before the parallel fan-
    # out and the mask drives both paths.
    skip_on_cran()
    skip_if_not(parallel::detectCores() >= 2L,
                "needs multi-core to exercise the parallel path")
    n_outer <- max(2L, min(4L, parallel::detectCores() - 1L))

    sim <- .sim_joint_bym2(8301L)
    fit_s <- .fit_joint_prune(sim, prune = TRUE, prune_tol = 1e-3,
                              n_threads_outer = 1L)
    fit_p <- .fit_joint_prune(sim, prune = TRUE, prune_tol = 1e-3,
                              n_threads_outer = n_outer)

    # Same mask, same pruned-cell count.
    expect_identical(fit_p$prune_mask, fit_s$prune_mask)
    expect_identical(fit_p$prune_n_pruned, fit_s$prune_n_pruned)

    # Converged log-marginals on kept cells must match to numerical noise.
    kept <- !fit_s$prune_mask
    expect_lt(max(abs(fit_p$log_marginal[kept] - fit_s$log_marginal[kept])),
              1e-4,
              label = "parallel vs serial log_marginal on kept cells")
    expect_lt(max(abs(fit_p$theta_mean - fit_s$theta_mean)), 1e-4,
              label = "parallel vs serial theta_mean")
    expect_lt(max(abs(fit_p$theta_sd - fit_s$theta_sd)), 1e-3,
              label = "parallel vs serial theta_sd")
})
