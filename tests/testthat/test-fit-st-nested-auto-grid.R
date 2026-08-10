# Auto mode-Hessian outer-grid recentering for fit_st_nested() (gcol33/tulpa#291).
#
# fit_st_nested()'s tau_spatial x tau_temporal [x rho] tensor grid had no
# mode-find machinery at all before this (unlike every other nested-Laplace
# family #289/#290 fixed, which all had SOME existing curvature-finding
# machinery to reuse). This builds one from scratch: a box-constrained
# derivative-free optim() over the unconstrained per-axis coordinate,
# triggered the same way as every other family (`pareto_k_regime ==
# "collapsed_edge"`), then a refit on a grid re-centred at the mode. See
# `.st_auto_grid_rescue()` (R/fit_st_nested_auto_grid.R).
#
# Note on triggering the rescue in a test: `pareto_k_regime` is a WHOLE-GRID
# effective-sample-size verdict (`.joint_pareto_grid_regime()`), not a
# per-axis one -- it only reports "collapsed_edge" when the FULL joint grid's
# quadrature ESS falls below 2, which needs the tensor's OTHER axis to also
# be reasonably concentrated (not necessarily railed itself), or both axes
# genuinely outside their default span at once. A companion axis that is
# merely "well identified but not particularly concentrated on this coarse a
# grid" can keep the joint ESS above 2 even while the other axis is visibly
# pinned to a boundary -- the recipes below were tuned to avoid that.

.chain_adj_st <- function(n_s) {
    adj <- matrix(0, n_s, n_s)
    for (i in seq_len(n_s - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
    adj
}

# One simulated RW1 path with known precision `tau`, sum-to-zero centred to
# match the field's own identifiability constraint.
.rw1_path_st <- function(n, tau) {
    v <- cumsum(c(0, rnorm(n - 1L, 0, 1 / sqrt(tau))))
    v - mean(v)
}

test_that("a true tau_spatial above the default ceiling (16) recenters the grid", {
    skip_on_cran()
    set.seed(7)
    n_s <- 14L; n_t <- 8L; n_per <- 60L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 60)      # true tau_spatial = 60, above the 16 ceiling
    vt <- .rw1_path_st(n_t, tau = 2)       # true tau_temporal well inside [0.25, 16]
    x  <- rnorm(length(spatial_idx), 0, 0.3)
    eta <- 0.1 + 0.3 * x + us[spatial_idx] + vt[tidx]
    y  <- rbinom(length(eta), 1, plogis(eta))

    fit <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                         family = "binomial", temporal_type = "rw1")

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_true(fit$outer_grid_recenter_attempts >= 1L)
    # The recentered axis brackets the true value AND extends past the
    # retired 16 ceiling.
    expect_gt(max(fit$theta_grid[, "tau_spatial"]), 16)
    expect_lt(min(fit$theta_grid[, "tau_spatial"]), 60)
    expect_gt(max(fit$theta_grid[, "tau_spatial"]), 60)
    expect_true(all(is.finite(fit$log_marginal)))
})

test_that("a true tau_temporal below the default floor (0.25) recenters the grid", {
    skip_on_cran()
    set.seed(101)
    n_s <- 12L; n_t <- 16L; n_per <- 150L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 4)       # sits exactly on a default grid node
    vt <- .rw1_path_st(n_t, tau = 0.03)    # true tau_temporal well below the 0.25 floor
    x  <- rnorm(length(spatial_idx), 0, 0.2)
    eta <- 0.1 + 0.2 * x + us[spatial_idx] + vt[tidx]
    y  <- rbinom(length(eta), 1, plogis(eta))

    fit <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                         family = "binomial", temporal_type = "rw1")

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_lt(min(fit$theta_grid[, "tau_temporal"]), 0.25)
    expect_lt(min(fit$theta_grid[, "tau_temporal"]), 0.03)
    expect_gt(max(fit$theta_grid[, "tau_temporal"]), 0.03)
    expect_true(all(is.finite(fit$log_marginal)))
})

# One railed fixture, reused by every provenance case below.
.st_railed_fixture <- function() {
    set.seed(7)
    n_s <- 14L; n_t <- 8L; n_per <- 60L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 60)
    vt <- .rw1_path_st(n_t, tau = 2)
    x  <- rnorm(length(spatial_idx), 0, 0.3)
    eta <- 0.1 + 0.3 * x + us[spatial_idx] + vt[tidx]
    list(y = rbinom(length(eta), 1, plogis(eta)), X = cbind(1, x),
         spatial_idx = spatial_idx, adj = adj, tidx = tidx, n_t = n_t)
}

.st_fit_railed <- function(control = list()) {
    f <- .st_railed_fixture()
    fit_st_nested(f$y, f$X, f$spatial_idx, f$adj, f$tidx, f$n_t,
                  family = "binomial", temporal_type = "rw1", control = control)
}

test_that("a genuinely non-default tau bound pins BOTH precision axes and stays fixed", {
    skip_on_cran()
    # `tau_lower` / `tau_upper` are the shared bounds of both precision axes, so
    # on an rw1 fit (no rho axis) pinning one pins every axis the grid has.
    fit <- .st_fit_railed(list(tau_lower = 0.2))

    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$outer_grid_recenter_declined, "grid_knobs_overridden")
    expect_identical(fit$pareto_k_regime, "collapsed_edge")
    expect_true("tau_spatial" %in% fit$pareto_k_grid_edge_axes)
})

# gcol33/tulpa#294: presence is not provenance. A knob set to the engine's own
# default expresses no preference -- the natural thing for a wrapper package
# that exposes its own argument defaulted to the engine's value -- and used to
# disable the recenter for every fit that wrapper made.
test_that("grid knobs at their own default values do NOT disable the rescue", {
    skip_on_cran()
    fit <- .st_fit_railed(list(n_grid_spatial = 4L, n_grid_temporal = 4L,
                               tau_lower = 0.25, tau_upper = 16))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$outer_grid_pinned_axes, character(0))
    expect_gt(max(fit$theta_grid[, "tau_spatial"]), 16)
})

test_that("an auto_grid()-marked knob is a declared default, so the rescue stays live", {
    skip_on_cran()
    # The SAME `tau_lower = 0.2` as the pinned case above -- byte-identical
    # starting grid, only the declaration differs.
    fit <- .st_fit_railed(list(tau_lower = auto_grid(0.2)))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$outer_grid_pinned_axes, character(0))
    expect_gt(max(fit$theta_grid[, "tau_spatial"]), 16)
})

test_that("a pin is PER AXIS: pinning rho leaves both precision axes free", {
    skip_on_cran()
    set.seed(31)
    n_s <- 14L; n_t <- 16L; n_per <- 100L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 60)
    vt <- .rw1_path_st(n_t, tau = 0.05)
    x  <- rnorm(length(spatial_idx), 0, 0.2)
    eta <- 0.1 + 0.2 * x + us[spatial_idx] + vt[tidx]
    y  <- rbinom(length(eta), 1, plogis(eta))

    fit <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                         family = "binomial", temporal_type = "ar1",
                         control = list(rho_lower = 0.2))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$outer_grid_pinned_axes, "rho")
    # The pinned axis keeps exactly the nodes its knobs built ...
    expect_equal(sort(unique(fit$theta_grid[, "rho"])),
                 seq(0.2, 0.9, length.out = 3L))
    # ... while the free precision axes are recentred past the retired ceiling.
    expect_gt(max(fit$theta_grid[, "tau_spatial"]), 16)
})

test_that(".st_pinned_axes reads value, mark and axis mapping", {
    pinned <- tulpa:::.st_pinned_axes
    # No knobs, engine defaults, and marked knobs are all defaults.
    expect_identical(pinned(list()), character(0))
    expect_identical(pinned(list(tau_lower = 0.25, n_grid_rho = 3L)), character(0))
    expect_identical(pinned(list(n_grid_spatial = auto_grid(9L))), character(0))
    # The shared precision bounds pin BOTH precision axes; the rho knobs only rho.
    expect_setequal(pinned(list(tau_upper = 40)), c("tau_spatial", "tau_temporal"))
    expect_identical(pinned(list(rho_lower = -0.5)), "rho")
    expect_identical(pinned(list(n_grid_spatial = 7L)), "tau_spatial")
})

test_that("a well-identified fit (mode inside the default span) is byte-stable across repeated calls", {
    skip_on_cran()
    set.seed(42)
    n_s <- 10L; n_t <- 6L; n_per <- 40L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 2)
    vt <- .rw1_path_st(n_t, tau = 2)
    x  <- rnorm(length(spatial_idx))
    eta <- 0.2 + 0.5 * x + us[spatial_idx] + vt[tidx]
    y  <- rbinom(length(eta), 1, plogis(eta))

    fit_a <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                           family = "binomial", temporal_type = "rw1")
    fit_b <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                           family = "binomial", temporal_type = "rw1")

    expect_identical(fit_a$outer_grid_placement, "fixed")
    expect_null(fit_a$outer_grid_recenter_attempts)
    expect_identical(fit_a$theta_grid, fit_b$theta_grid)
    expect_identical(fit_a$weights, fit_b$weights)
    expect_identical(fit_a$log_marginal, fit_b$log_marginal)
    expect_identical(fit_a$spatial_effects, fit_b$spatial_effects)
})

test_that("ar1's rho axis rides along with the recenter when both precision axes collapse", {
    skip_on_cran()
    set.seed(31)
    n_s <- 14L; n_t <- 16L; n_per <- 100L
    adj <- .chain_adj_st(n_s)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    tidx <- sample(n_t, length(spatial_idx), TRUE)
    us <- .rw1_path_st(n_s, tau = 60)
    vt <- .rw1_path_st(n_t, tau = 0.05)
    x  <- rnorm(length(spatial_idx), 0, 0.2)
    eta <- 0.1 + 0.2 * x + us[spatial_idx] + vt[tidx]
    y  <- rbinom(length(eta), 1, plogis(eta))

    fit <- fit_st_nested(y, cbind(1, x), spatial_idx, adj, tidx, n_t,
                         family = "binomial", temporal_type = "ar1")

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(ncol(fit$theta_grid), 3L)
    expect_true(all(c("tau_spatial", "tau_temporal", "rho") %in% colnames(fit$theta_grid)))
    # rho stays within its model-level stationarity bound regardless of how
    # far the recenter pushes it.
    expect_true(all(fit$theta_grid[, "rho"] > -1 & fit$theta_grid[, "rho"] < 1))
    expect_true(all(is.finite(fit$log_marginal)))

    # A mode-SD bound declines the axis it bound on, NOT the whole pass
    # (gcol33/tulpa#387). On this fixture `rho`'s mode SD reaches the ceiling
    # while both precision axes resolve theirs, so the precision axes are still
    # re-placed and `rho` keeps the nodes it came in with. Declining the pass
    # outright would discard two placements to say nothing about the third.
    expect_identical(unname(fit$outer_grid_recenter_sd_clamp[["rho"]]),
                     "ceiling")
    expect_identical(names(fit$outer_grid_recenter_sd_declined), "rho")
    expect_identical(unname(fit$outer_grid_recenter_sd_declined[["rho"]]),
                     "sd_ceiling_unresolved")
    # The declined axis reports the SD the mode-find MEASURED but no SD it was
    # laid from, because it was not laid.
    expect_true(is.finite(fit$outer_grid_recenter_sd_raw[["rho"]]))
    expect_false("rho" %in% names(fit$outer_grid_recenter_sd_used))
    # ... and the axes that did resolve moved off their defaults.
    expect_true(all(c("tau_spatial", "tau_temporal") %in%
                        names(fit$outer_grid_recenter_sd_used)))
    expect_identical(unname(fit$outer_grid_recenter_sd_clamp[["tau_spatial"]]),
                     "none")
})
