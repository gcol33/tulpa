# Auto mode-Hessian outer-grid recentering (gcol33/tulpa#289, #292, #293).
#
# The joint driver's `sigma_grid` default (`exp(seq(log(0.1), log(3),
# length.out = 5))`) is a starting axis, not a hard ceiling: when a fit's
# field-SD posterior rails the top node (pareto_k_regime = "collapsed_edge"
# on the sigma axis), the driver re-centres the axis on the mode-Hessian its
# own outer Pareto-k diagnostic already computed and refits, up to two
# attempts (the second adding a light default PC(U=3, alpha=0.01) prior when
# geometry alone does not settle the mode). A PINNED `sigma_grid` always wins;
# an axis left NULL, marked with `auto_grid()`, or equal to the engine's own
# default axis is a default the recenter may move (gcol33/tulpa#293).
#
# gcol33/tulpa#292: the tests below through "BYM2's finer grid" run at the
# default `control$diagnose_k = TRUE`, so the rescue's mode-Hessian comes from
# the outer Pareto-k diagnostic that already ran. The block below
# ("at diagnose_k = FALSE") re-runs the two collapse cases at the PRODUCTION
# default `diagnose_k = FALSE` and confirms the recenter still engages: the
# placement-only mode-Hessian (`.joint_attach_pareto_k_placement()`) supplies
# the same curvature without the full diagnostic ever running.

.chain_adj_ag <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn),
         n_spatial_units = n_s)
}

# A sparse, strongly separated occurrence pattern -- a few units almost
# always positive, the rest almost always negative -- the "sparse,
# weakly-identified species" regime gcol33/tulpa#289 targets: the field-SD
# posterior wants to sit well past the old fixed ceiling of 3.0.
.sparse_icar_arm <- function(n_s = 20L, n_per = 6L, seed = 11) {
    set.seed(seed)
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    base_p <- rep(0.02, n_s); base_p[1:5] <- 0.95
    y <- rbinom(length(spatial_idx), 1, base_p[spatial_idx])
    X <- cbind(1, rnorm(length(y), 0, 0.05))
    list(
        arm = list(y = as.numeric(y), n_trials = rep(1L, length(y)),
                  X = X, spatial_idx = as.integer(spatial_idx),
                  re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
                  family = "binomial", phi = 1.0),
        adj = .chain_adj_ag(n_s)
    )
}

test_that("auto-recenter resolves a sigma-axis edge collapse and stays spread", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors)

    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior)

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_true(fit$outer_grid_recenter_attempts >= 1L)
    # The recentered axis sits above the retired 3.0 ceiling.
    expect_gt(max(fit$theta_grid[, "sigma"]), 3.0)
    expect_true(all(is.finite(fit$log_marginal)))
})

test_that("the SAME data with a tiny explicit sigma_grid stays collapsed (override never touched)", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors,
                  sigma_grid = c(0.1, 0.5, 1, 2, 3))

    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior)

    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(fit$pareto_k_regime, "collapsed_edge")
    expect_true("sigma" %in% fit$pareto_k_grid_edge_axes)
})

# gcol33/tulpa#293: a wrapper package (tulpaObs's occu_cover) computes the
# engine's own default axis itself and writes it onto the block -- it also
# derives the copy arm's amplitude axis from that vector -- so every auto fit
# arrived carrying a non-NULL `prior$sigma_grid` and the rescue's `!is.null()`
# guard read it as a user override. The two cases below are the same data as
# the collapse test above, with the axis arriving the two ways a caller can
# declare "this is my default, not the user's choice".
test_that("an engine-default sigma_grid handed back in still auto-recenters", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors,
                  sigma_grid = exp(seq(log(0.1), log(3), length.out = 5)))

    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior,
                                      control = list(diagnose_k = FALSE))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_gt(max(fit$theta_grid[, "sigma"]), 3.0)
    expect_null(fit$outer_grid_recenter_declined)
})

# The exact single-block shape `occu_cover()` sends: one shared icar field
# carrying the engine's own default `sigma_grid`, a second arm reading it through
# an on-arm `field_coef` copy coefficient, and the outer k-hat diagnostic off.
# This is the fit gcol33/tulpa#293 reports as never recentering.
test_that("the occu_cover single-block shape (stamped default + field_coef) recenters", {
    skip_on_cran()
    sim <- .sparse_icar_arm()
    n_s <- sim$adj$n_spatial_units
    set.seed(7)
    N2 <- 200L
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X2 <- cbind(1, rnorm(N2))
    y2 <- rnorm(N2, as.numeric(X2 %*% c(0.1, 0.2)), 0.4)
    arm_pos <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                    spatial_idx = as.integer(s2),
                    re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                    family = "gaussian", phi = 1.0,
                    field_coef = list(name = "alpha", grid = c(0, 0.5, 1.0)))
    prior <- list(type = "icar", n_spatial_units = n_s,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors,
                  sigma_grid = exp(seq(log(0.1), log(3), length.out = 5)))

    fit <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = sim$arm, pos = arm_pos), prior = prior,
        control = list(diagnose_k = FALSE)))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_gt(max(fit$theta_grid[, "sigma"]), 3.0)
    expect_null(fit$outer_grid_recenter_declined)
})

test_that("an auto_grid()-marked axis auto-recenters; the same axis unmarked does not", {
    skip_on_cran()
    sim  <- .sparse_icar_arm()
    base <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                 adj_row_ptr = sim$adj$adj_row_ptr,
                 adj_col_idx = sim$adj$adj_col_idx,
                 n_neighbors = sim$adj$n_neighbors)
    axis <- c(0.1, 0.5, 1, 2, 3)   # not the engine default: provenance is the mark

    marked <- tulpa_nested_laplace_joint(
        responses = list(occ = sim$arm),
        prior = c(base, list(sigma_grid = auto_grid(axis))),
        control = list(diagnose_k = FALSE))
    expect_identical(marked$outer_grid_placement, "auto_recentered")
    expect_gt(max(marked$theta_grid[, "sigma"]), 3.0)

    pinned <- tulpa_nested_laplace_joint(
        responses = list(occ = sim$arm),
        prior = c(base, list(sigma_grid = axis)),
        control = list(diagnose_k = FALSE))
    expect_identical(pinned$outer_grid_placement, "fixed")
    expect_identical(pinned$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(max(pinned$theta_grid[, "sigma"]), 3.0)
})

test_that("a copy block's engine-default donor axis auto-recenters (multi-block)", {
    skip_on_cran()
    set.seed(11)
    n_s <- 20L; n_per <- 6L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    f1 <- rep(0, n_s); f1[1:5] <- 3; f1[6:20] <- -0.6
    f1 <- f1 - mean(f1)
    sigma1_true <- 3.0
    alpha_true  <- 0.5

    y1 <- rbinom(length(spatial_idx), 1, plogis(sigma1_true * f1[spatial_idx]))
    X1 <- cbind(1, rnorm(length(y1), 0, 0.05))

    N2 <- 300L
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X2 <- cbind(1, rnorm(N2))
    y2 <- rnorm(N2, X2 %*% c(0.1, 0.2) + alpha_true * sigma1_true * f1[s2], 0.3)

    arm1 <- list(y = as.numeric(y1), n_trials = rep(1L, length(y1)),
                X = X1, re_idx = rep(0, length(y1)), n_re_groups = 0L,
                sigma_re = 1.0, family = "binomial", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                family = "gaussian", phi = 1.0)

    adj <- .chain_adj_ag(n_s)
    # The wrapper-package shape: the donor axis IS the engine default, written
    # onto the block rather than left NULL.
    block <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  spatial_idx = list(spatial_idx, s2),
                  sigma_grid = exp(seq(log(0.1), log(3), length.out = 5)))

    fit <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2),
        prior = list(block),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.2, 0.4, 0.6, 0.8))),
        control = list(diagnose_k = FALSE)
    ))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_gt(max(fit$theta_grid[, "b1.sigma"]), 3.0)
    expect_lt(abs(fit$block_moments[[1L]]$mean[["sigma"]] - sigma1_true), 1.0)
})

test_that("a well-identified fit is byte-stable across repeated calls (no rescue touched it)", {
    skip_on_cran()
    set.seed(42)
    n_s <- 20L; n_per <- 60L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    rw    <- cumsum(rnorm(n_s, 0, 1.0 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    X <- cbind(1, rnorm(length(spatial_idx)))
    eta <- as.numeric(X %*% c(-0.2, 0.4)) + phi_s[spatial_idx]
    y   <- rbinom(length(eta), 1, plogis(eta))
    arm <- list(y = as.numeric(y), n_trials = rep(1L, length(y)),
                X = X, spatial_idx = as.integer(spatial_idx),
                re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
                family = "binomial", phi = 1.0)
    adj <- .chain_adj_ag(n_s)
    prior <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors)

    fit_a <- tulpa_nested_laplace_joint(responses = list(occ = arm), prior = prior)
    fit_b <- tulpa_nested_laplace_joint(responses = list(occ = arm), prior = prior)

    expect_identical(fit_a$outer_grid_placement, "fixed")
    expect_null(fit_a$outer_grid_recenter_attempts)
    expect_identical(fit_a$outer_grid_recenter_declined, "grid_not_collapsed")
    expect_identical(fit_a$theta_grid, fit_b$theta_grid)
    expect_identical(fit_a$weights, fit_b$weights)
    expect_identical(fit_a$log_marginal, fit_b$log_marginal)
    expect_identical(fit_a$theta_mean, fit_b$theta_mean)
})

test_that("BYM2's finer (sigma, rho) grid also carries the new field, untouched when unneeded", {
    skip_on_cran()
    set.seed(42)
    n_s <- 20L; n_per <- 60L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    rw    <- cumsum(rnorm(n_s, 0, 1.0 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    X <- cbind(1, rnorm(length(spatial_idx)))
    eta <- as.numeric(X %*% c(-0.2, 0.4)) + phi_s[spatial_idx]
    y   <- rbinom(length(eta), 1, plogis(eta))
    arm <- list(y = as.numeric(y), n_trials = rep(1L, length(y)),
                X = X, spatial_idx = as.integer(spatial_idx),
                re_idx = rep(0, length(y)), n_re_groups = 0L, sigma_re = 1.0,
                family = "binomial", phi = 1.0)
    adj <- .chain_adj_ag(n_s)
    prior <- list(type = "bym2", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, scale_factor = 1.0)

    fit <- tulpa_nested_laplace_joint(responses = list(occ = arm), prior = prior)
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$pareto_k_regime, "spread")
})

test_that("a multi-block copy field's donor sigma axis auto-recenters on collapse", {
    skip_on_cran()
    set.seed(11)
    n_s <- 20L; n_per <- 6L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    f1 <- rep(0, n_s); f1[1:5] <- 3; f1[6:20] <- -0.6
    f1 <- f1 - mean(f1)
    sigma1_true <- 3.0
    alpha_true  <- 0.5

    y1 <- rbinom(length(spatial_idx), 1, plogis(sigma1_true * f1[spatial_idx]))
    X1 <- cbind(1, rnorm(length(y1), 0, 0.05))

    N2 <- 300L
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X2 <- cbind(1, rnorm(N2))
    y2 <- rnorm(N2, X2 %*% c(0.1, 0.2) + alpha_true * sigma1_true * f1[s2], 0.3)

    arm1 <- list(y = as.numeric(y1), n_trials = rep(1L, length(y1)),
                X = X1, re_idx = rep(0, length(y1)), n_re_groups = 0L,
                sigma_re = 1.0, family = "binomial", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                family = "gaussian", phi = 1.0)

    adj <- .chain_adj_ag(n_s)
    block <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  spatial_idx = list(spatial_idx, s2))   # sigma_grid unset -> default

    fit <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2),
        prior = list(block),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.2, 0.4, 0.6, 0.8)))
    ))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_gt(max(fit$theta_grid[, "b1.sigma"]), 3.0)
    expect_lt(abs(fit$block_moments[[1L]]$mean[["sigma"]] - sigma1_true), 1.0)
})

# gcol33/tulpa#292: the rescue must engage at diagnose_k = FALSE too -- the
# production default, and the setting the cover-glaser deliverable runs
# under -- not just when the full outer Pareto-k diagnostic happens to have
# computed the mode-Hessian already.
test_that("auto-recenter engages at diagnose_k = FALSE (single-block icar)", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors)

    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior,
                                      control = list(diagnose_k = FALSE))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_true(fit$outer_grid_recenter_attempts >= 1L)
    expect_gt(max(fit$theta_grid[, "sigma"]), 3.0)
    expect_true(all(is.finite(fit$log_marginal)))
    # The diagnostic itself stayed off throughout -- the recenter used only
    # the placement-only mode-Hessian, not a computed outer k-hat.
    expect_true(is.na(fit$pareto_k))
})

test_that("an explicit sigma_grid override is never touched at diagnose_k = FALSE", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors,
                  sigma_grid = c(0.1, 0.5, 1, 2, 3))

    fit <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior,
                                      control = list(diagnose_k = FALSE))

    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(fit$pareto_k_regime, "collapsed_edge")
    expect_true("sigma" %in% fit$pareto_k_grid_edge_axes)
})

test_that("a multi-block copy field's donor sigma axis auto-recenters at diagnose_k = FALSE", {
    skip_on_cran()
    set.seed(11)
    n_s <- 20L; n_per <- 6L
    spatial_idx <- rep(seq_len(n_s), each = n_per)
    f1 <- rep(0, n_s); f1[1:5] <- 3; f1[6:20] <- -0.6
    f1 <- f1 - mean(f1)
    sigma1_true <- 3.0
    alpha_true  <- 0.5

    y1 <- rbinom(length(spatial_idx), 1, plogis(sigma1_true * f1[spatial_idx]))
    X1 <- cbind(1, rnorm(length(y1), 0, 0.05))

    N2 <- 300L
    s2 <- sample.int(n_s, N2, replace = TRUE)
    X2 <- cbind(1, rnorm(N2))
    y2 <- rnorm(N2, X2 %*% c(0.1, 0.2) + alpha_true * sigma1_true * f1[s2], 0.3)

    arm1 <- list(y = as.numeric(y1), n_trials = rep(1L, length(y1)),
                X = X1, re_idx = rep(0, length(y1)), n_re_groups = 0L,
                sigma_re = 1.0, family = "binomial", phi = 1.0)
    arm2 <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                re_idx = rep(0, N2), n_re_groups = 0L, sigma_re = 1.0,
                family = "gaussian", phi = 1.0)

    adj <- .chain_adj_ag(n_s)
    block <- list(type = "icar", n_spatial_units = adj$n_spatial_units,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  spatial_idx = list(spatial_idx, s2))   # sigma_grid unset -> default

    fit <- suppressWarnings(tulpa_nested_laplace_joint(
        responses = list(occ = arm1, pos = arm2),
        prior = list(block),
        copy = list(list(arm = "pos", block = 1L, alpha_grid = c(0.2, 0.4, 0.6, 0.8))),
        control = list(diagnose_k = FALSE)
    ))

    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_identical(fit$pareto_k_regime, "spread")
    expect_gt(max(fit$theta_grid[, "b1.sigma"]), 3.0)
    expect_lt(abs(fit$block_moments[[1L]]$mean[["sigma"]] - sigma1_true), 1.0)
    expect_true(is.na(fit$pareto_k))
})

test_that("diagnose_k = FALSE recenter matches diagnose_k = TRUE's grid placement", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors)

    fit_diag   <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior,
                                             control = list(diagnose_k = TRUE))
    fit_nodiag <- tulpa_nested_laplace_joint(responses = list(occ = sim$arm), prior = prior,
                                             control = list(diagnose_k = FALSE))

    # Both recenter around the same mode regardless of whether the full outer
    # k-hat diagnostic supplied the mode-Hessian or the placement-only path
    # did -- both reach it via the identical FD-Hessian computation
    # (.joint_pareto_grid_mode_cov()) at the same modal cell, so a loose
    # tolerance (rather than an exact match) absorbs any warm-start-path
    # floating point noise.
    expect_identical(fit_diag$outer_grid_placement, "auto_recentered")
    expect_identical(fit_nodiag$outer_grid_placement, "auto_recentered")
    expect_equal(fit_diag$theta_grid, fit_nodiag$theta_grid, tolerance = 1e-6)
})

test_that("control$auto_recenter = FALSE holds the default grid where it is", {
    skip_on_cran()
    sim   <- .sparse_icar_arm()
    prior <- list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
                  adj_row_ptr = sim$adj$adj_row_ptr,
                  adj_col_idx = sim$adj$adj_col_idx,
                  n_neighbors = sim$adj$n_neighbors)

    fit <- tulpa_nested_laplace_joint(
        responses = list(occ = sim$arm), prior = prior,
        control = list(diagnose_k = FALSE, auto_recenter = FALSE))

    # Same data as the collapse test above; the opt-out integrates the default
    # axis as given, railed and all, and says so.
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$outer_grid_recenter_declined, "auto_recenter_disabled")
    expect_identical(fit$pareto_k_regime, "collapsed_edge")
    expect_equal(max(fit$theta_grid[, "sigma"]), 3.0)
})
