# Rank-faithful cheap-pass prune + safety gate (gcol33/tulpa#43).
#
# Regression for the cheap-pass mis-rank that dropped the true posterior mode
# on a real 1.16M-row BYM2 cover hurdle. The fixed-pilot one-Newton-step screen
# mis-ranked far cells by O(1e5) log-units (the inner latent mode moves
# substantially across the (sigma, rho, alpha) grid), so the pruned fit kept a
# cell whose true full-grid weight was ~1e-321. The fix:
#   1. Rank-faithful screen: a chained lattice sweep, warm-starting each cell's
#      short Newton run from the previous screened cell's quasi-mode, so every
#      cheap mode stays near its cell's true mode.
#   2. Safety gate: if the cheap-screen argmax disagrees with the full-solve
#      argmax (or the kept posterior collapses onto a cell the screen badly
#      mis-estimated), warn and fall back to the full grid.
#
# This file builds a tractable-but-stressing fit whose inner mode MOVES across
# the outer grid (strong spatial field, wide sigma/rho range, tight Gaussian
# positive arm) and asserts the pruned posterior recovers the SAME mode and
# theta_mean as the full grid. It also asserts the safety gate triggers a
# warning + fallback on a constructed mis-rank.

.mr_chain_adj <- function(n_s) {
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

# Strong, smooth spatial field so the inner latent mode depends strongly on
# (sigma, rho); tight Gaussian positive arm so the field is pinned hard and the
# inner mode at the true cell is far from where a distant pilot would suggest.
.mr_sim <- function(seed) {
    set.seed(seed)
    N <- 800L; n_s <- 80L
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- as.numeric(arima.sim(list(ar = 0.85), n_s))
    theta <- rnorm(n_s)
    w_s   <- 1.4 * (sqrt(0.85) * phi / sd(phi) + sqrt(0.15) * theta)
    x     <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.2, 0.5)) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]
    spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.1, -0.3)) + 1.0 * w_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, 0.25)
    list(N = N, n_s = n_s, spatial_idx = as.integer(spatial_idx),
         Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi_pos = as.integer(spi_pos))
}

.mr_fit <- function(sim, prune, prune_tol = 1e-3, n_threads_outer = 1L) {
    adj <- .mr_chain_adj(sim$n_s)
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
        family = "gaussian", phi = 0.25
    )
    prior <- list(
        type = "bym2", n_spatial_units = adj$n_spatial_units,
        adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
        n_neighbors = adj$n_neighbors, scale_factor = 1.0,
        sigma_grid = c(0.05, 0.2, 0.6, 1.5, 3.0),
        rho_grid   = c(0.1, 0.4, 0.7, 0.95)
    )
    arm_pos$field_coef <- list(name = "alpha", grid = c(0.2, 0.6, 1.0, 1.5))
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior,
        control = list(n_threads = 1L, n_threads_outer = n_threads_outer,
                       prune = prune, prune_tol = prune_tol,
                       adaptive_grid = FALSE, var_of_means_consistency = FALSE,
                       diagnose_k = FALSE)
    )
}

test_that("pruned posterior recovers the full-grid mode (rank-faithful screen)", {
    skip_on_cran()
    sim <- .mr_sim(4242L)
    fit_full  <- .mr_fit(sim, prune = FALSE)
    fit_prune <- expect_warning(.mr_fit(sim, prune = TRUE, prune_tol = 1e-3),
                                regexp = NA)  # rank-faithful: NO fallback warning

    # The cheap screen must actually prune something (otherwise the test is
    # vacuous) but must keep the true mode.
    expect_gt(fit_prune$prune_n_pruned, 0L)

    k_full  <- which.max(fit_full$log_marginal)
    expect_false(fit_prune$prune_mask[k_full],
                 info = "the full-grid argmax cell must NOT be pruned")

    # Pruned argmax cell == full argmax cell.
    expect_equal(which.max(fit_prune$log_marginal), k_full)

    # theta_mean recovered to tight tolerance (was off by ~0.72 with the
    # fixed-pilot screen; rank-faithful screen recovers to ~1e-3).
    expect_lt(max(abs(fit_prune$theta_mean - fit_full$theta_mean)), 1e-2,
              label = "pruned vs full theta_mean max-abs-diff")
    # And no silent fallback fired (the screen was rank-faithful here).
    expect_null(fit_prune$prune_fallback_triggered)
})

test_that("parallel per-tile cheap screen is rank-faithful (gcol33/tulpa#68)", {
    skip_on_cran()
    # The cheap screen parallelises by splitting the load-bearing global
    # neighbour chain into independent per-tile chains (one tile per non-alpha
    # outer coordinate), run concurrently on outer-thread worker slots. This
    # must keep the same rank faithfulness as the serial chain: still keep the
    # true mode, still recover the full-grid theta_mean, and agree with the
    # serial screen's prune decisions.
    sim <- .mr_sim(4242L)
    fit_full <- .mr_fit(sim, prune = FALSE)
    fit_ser  <- .mr_fit(sim, prune = TRUE, prune_tol = 1e-3,
                        n_threads_outer = 1L)
    fit_par  <- expect_warning(
        .mr_fit(sim, prune = TRUE, prune_tol = 1e-3, n_threads_outer = 4L),
        regexp = NA)  # rank-faithful: NO fallback warning

    k_full <- which.max(fit_full$log_marginal)

    # The parallel screen prunes something but never the true mode.
    expect_gt(fit_par$prune_n_pruned, 0L)
    expect_false(fit_par$prune_mask[k_full],
                 info = "parallel screen must NOT prune the full-grid argmax")
    expect_equal(which.max(fit_par$log_marginal), k_full)

    # Rank-faithful: parallel-screen posterior recovers the full grid to the
    # same order of accuracy as the serial screen.
    expect_lt(max(abs(fit_par$theta_mean - fit_full$theta_mean)), 2e-2,
              label = "parallel-screen vs full theta_mean max-abs-diff")
    expect_null(fit_par$prune_fallback_triggered)

    # The two-tier tile-pilot backbone gives the parallel screen the same
    # near-neighbour warm-starts as the serial chain, so it reaches essentially
    # the same prune decisions: the survivor count matches within a couple of
    # borderline cells and the marginalised posterior agrees closely.
    expect_lte(abs(fit_par$prune_n_pruned - fit_ser$prune_n_pruned), 2L)
    expect_lt(max(abs(fit_par$theta_mean - fit_ser$theta_mean)), 2e-2,
              label = "parallel vs serial screen theta_mean max-abs-diff")
})

test_that("safety gate warns and falls back on argmax disagreement", {
    skip_on_cran()
    # Construct a pruned-result shape whose cheap-screen argmax disagrees with
    # the full-solve argmax, and a resolve_full thunk standing in for the
    # full-grid re-solve. The gate must warn and return the full-grid result.
    res_pruned <- list(
        log_marginal = c(-10, -Inf, -5, -Inf),   # full argmax = cell 3 (kept)
        prune_mask = c(FALSE, TRUE, FALSE, TRUE),
        prune_cheap_log_marginal = c(-10, -1, -5, -8),  # cheap argmax = cell 2
        prune_argmax_disagree = TRUE,
        prune_cheap_full_gap = 0,
        prune_n_pruned = 2L
    )
    full_sentinel <- list(log_marginal = c(-1, -2, -3, -4), ITS_THE_FULL = TRUE)

    expect_warning(
        .joint_prune_safety_gate(res_pruned,
                                 resolve_full = function() full_sentinel),
        regexp = "falling back to the full grid")
    out <- suppressWarnings(
        .joint_prune_safety_gate(res_pruned,
                                 resolve_full = function() full_sentinel))
    expect_true(isTRUE(out$ITS_THE_FULL))
    expect_true(isTRUE(out$prune_fallback_triggered))
    expect_match(out$prune_fallback_reason, "argmax")
})

test_that("safety gate warns and falls back on gap-collapse", {
    skip_on_cran()
    # Posterior collapses onto one kept cell whose cheap-vs-full gap is huge:
    # argmax agrees, but the screen badly mis-estimated the cell the whole
    # posterior sits on, so the prune is not trustworthy.
    res_pruned <- list(
        # Spike on cell 1 (ESS ~ 1); cells 2..4 pruned.
        log_marginal = c(0, -Inf, -Inf, -Inf),
        prune_mask = c(FALSE, TRUE, TRUE, TRUE),
        prune_cheap_log_marginal = c(-500, -10, -12, -15),  # cheap on cell1 way off
        prune_argmax_disagree = FALSE,  # cheap argmax cell would be 2, but we
                                        # force the disagree flag off to isolate
                                        # the gap-collapse path
        prune_cheap_full_gap = 500,     # |full_lm[1] - cheap_lm[1]| = 500
        prune_n_pruned = 3L
    )
    full_sentinel <- list(log_marginal = c(-1, -2, -3, -4), ITS_THE_FULL = TRUE)

    expect_warning(
        .joint_prune_safety_gate(res_pruned,
                                 resolve_full = function() full_sentinel),
        regexp = "falling back to the full grid")
    out <- suppressWarnings(
        .joint_prune_safety_gate(res_pruned,
                                 resolve_full = function() full_sentinel))
    expect_true(isTRUE(out$ITS_THE_FULL))
    expect_true(isTRUE(out$prune_fallback_triggered))
    expect_match(out$prune_fallback_reason, "gap|collapse")
})

test_that("safety gate is a no-op when the ranking is trustworthy", {
    skip_on_cran()
    # Argmax agrees, gap small, ESS healthy: no fallback, returns res as-is.
    res_pruned <- list(
        log_marginal = c(-1, -1.2, -1.1, -Inf),
        prune_mask = c(FALSE, FALSE, FALSE, TRUE),
        prune_cheap_log_marginal = c(-1.05, -1.25, -1.15, -8),
        prune_argmax_disagree = FALSE,
        prune_cheap_full_gap = 0.05,
        prune_n_pruned = 1L
    )
    out <- expect_warning(
        .joint_prune_safety_gate(res_pruned,
                                 resolve_full = function()
                                     stop("must not re-solve")),
        regexp = NA)
    expect_null(out$prune_fallback_triggered)
})

test_that("safety gate is a no-op when no prune ran", {
    skip_on_cran()
    res_noprune <- list(log_marginal = c(-1, -2, -3))  # no prune_mask
    out <- .joint_prune_safety_gate(res_noprune,
                                    resolve_full = function()
                                        stop("must not re-solve"))
    expect_identical(out, res_noprune)
})
