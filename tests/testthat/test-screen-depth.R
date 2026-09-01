# Cheap-screen depth (`control$screen_iters`) on the joint nested-Laplace
# front door.
#
# The screen runs a short inner Newton on EVERY cell of the outer grid to rank
# them, then the full solve runs on the survivors. Its depth is therefore a
# cost-against-ranking-fidelity trade: each extra step is paid on every cell,
# including the ones the screen keeps and then solves in full.
#
# What is asserted here:
#   (1) the depth the driver ran at is echoed back on the fit, so the knob is
#       observable rather than inferred from timings,
#   (2) a shallow screen and the default one keep the same argmax cell and the
#       same fixed-effect table, which is what says the ranking is faithful at
#       both depths on this fixture,
#   (3) the R door refuses a depth that is not a single integer >= 1,
#   (4) a fit that does not screen is untouched by the knob.
#
# The fixture is the one test-nested-laplace-joint-prune.R screens: a two-arm
# BYM2 joint fit over a 45-cell (sigma x rho x alpha) grid, at a seed where the
# prune safety gate does not fall back.

.screen_chain_adj <- function(n_s) {
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

.screen_sim <- function(seed) {
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

# `screen_iters = NULL` leaves the knob out of `control` entirely, which is how
# the default is exercised as a default rather than as a restated value.
.screen_fit <- function(sim, prune = TRUE, screen_iters = NULL,
                        prune_tol = 1e-3) {
    adj <- .screen_chain_adj(sim$n_s)
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
        sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9)
    )
    arm_pos$field_coef <- list(name = "alpha", grid = c(0.3, 0.6, 1.0, 1.4, 1.8))
    ctrl <- list(n_threads = 1L, n_threads_outer = 1L,
                 prune = prune, prune_tol = prune_tol,
                 adaptive_grid = FALSE, var_of_means_consistency = FALSE)
    if (!is.null(screen_iters)) ctrl$screen_iters <- screen_iters
    tulpa_nested_laplace_joint(
        responses = list(occ = arm_occ, pos = arm_pos),
        prior = prior, control = ctrl
    )
}

test_that("the screening depth the driver ran at is reported on the fit", {
    skip_on_cran()
    sim <- .screen_sim(8101L)

    fit_default <- .screen_fit(sim)
    expect_identical(as.integer(fit_default$prune_screen_iters),
                     as.integer(tulpa:::.nl_screen("iters")))

    fit_two <- .screen_fit(sim, screen_iters = 2L)
    expect_identical(as.integer(fit_two$prune_screen_iters), 2L)
})

test_that("a one-step screen ranks the grid the same way the default does", {
    # The screen only has to RANK the cells, and each is warm-started from its
    # already-screened lattice neighbour, so a shallow screen should keep the
    # same argmax and drop cells carrying the same negligible mass. Tolerances
    # are absolute and read off the prune threshold: a cell can hold up to
    # `prune_tol` of the screening mass and still be dropped, so two depths that
    # disagree about one such cell move a coefficient by that order.
    skip_on_cran()
    sim <- .screen_sim(8101L)

    f1 <- .screen_fit(sim, screen_iters = 1L)
    f5 <- .screen_fit(sim, screen_iters = 5L)

    # Both fits screened rather than falling back through the safety gate,
    # which is the premise of every comparison below.
    expect_false(isTRUE(f1$prune_fallback_triggered))
    expect_false(isTRUE(f5$prune_fallback_triggered))

    expect_identical(which.max(f1$log_marginal), which.max(f5$log_marginal))

    expect_lt(max(abs(unname(coef(f1)) - unname(coef(f5)))), 5e-3)
    expect_lt(max(abs(summary(f1)$std.error - summary(f5)$std.error)), 1e-2)
    expect_lt(max(abs(f1$theta_mean - f5$theta_mean)), 5e-3)
})

test_that("the R door refuses a screening depth that is not an integer >= 1", {
    skip_on_cran()
    sim <- .screen_sim(8101L)
    for (bad in list(0L, -1L, 2.5, c(1L, 2L), "3", NA_integer_)) {
        expect_error(.screen_fit(sim, screen_iters = bad), "screen_iters")
    }
})

test_that("a fit that does not screen is untouched by the depth", {
    # With `prune = FALSE` the cheap sweep never runs, so the knob has nothing
    # to reach and the fit must be the one the default produces, field for
    # field, with no screening report attached.
    skip_on_cran()
    sim <- .screen_sim(8101L)

    fit_off <- .screen_fit(sim, prune = FALSE)
    fit_off_deep <- .screen_fit(sim, prune = FALSE, screen_iters = 1L)

    expect_null(fit_off$prune_screen_iters)
    expect_null(fit_off_deep$prune_screen_iters)
    expect_null(fit_off_deep$prune_mask)

    expect_equal(fit_off_deep$log_marginal, fit_off$log_marginal,
                 tolerance = 0)
    expect_equal(fit_off_deep$theta_mean, fit_off$theta_mean, tolerance = 0)
    expect_equal(fit_off_deep$theta_sd, fit_off$theta_sd, tolerance = 0)
})
