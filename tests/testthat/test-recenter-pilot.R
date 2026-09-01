# The outer-grid placement pilot (gcol33/tulpa#636).
#
# Placement reads an argmax cell and an FD curvature stencil off `log_marginal`,
# and reads neither off the integration the detecting pass paid for -- so a
# placement that fires discards every inner Newton solve of the grid it detected
# on. `control$recenter_pilot` detects on a THINNED grid over the same spans and
# solves the full grid once, at the placed axes.
#
# Three properties carry the feature, and each is asserted below rather than
# argued for:
#   1. a thinned axis is a coarser read of the SAME span -- both endpoints kept,
#      which is what makes the rail test transport;
#   2. a path that stores its axes PRE-PAIRED is never thinned per axis, since
#      that would re-pair the grid rather than thin it;
#   3. a fit whose placement declines is BIT-IDENTICAL to the same fit with the
#      knob off -- the pilot is a pre-screen, so it can only add a placement.

# --- 1. the axis thinner -----------------------------------------------------

test_that("a thinned axis keeps both endpoints and is a subset of its input", {
    ax <- exp(seq(log(0.1), log(3), length.out = 9))
    for (n in 2:8) {
        th <- .nl_pilot_axis(ax, n)
        expect_true(length(th) <= n)
        expect_true(length(th) >= 2L)
        expect_equal(th[1L], ax[1L])
        expect_equal(th[length(th)], ax[length(ax)])
        expect_true(all(th %in% ax))
        expect_false(is.unsorted(th))
    }
})

test_that("an axis already at or below the pilot resolution is left alone", {
    expect_null(.nl_pilot_axis(c(1, 2, 3), 3))
    expect_null(.nl_pilot_axis(c(1, 2), 3))
    expect_null(.nl_pilot_axis(1, 3))
    # A matrix axis (mcar / miid's log-Cholesky field, tgmrf's bounds box) is a
    # coordinate block, not a set of nodes on one axis.
    expect_null(.nl_pilot_axis(matrix(1:6, 3, 2), 2))
})

test_that("control$recenter_pilot resolves to a resolution, and refuses junk", {
    expect_true(is.na(.nl_pilot_n(list())))
    expect_true(is.na(.nl_pilot_n(list(recenter_pilot = FALSE))))
    expect_identical(.nl_pilot_n(list(recenter_pilot = TRUE)),
                     as.integer(.nl_recenter("pilot_n")))
    expect_identical(.nl_pilot_n(list(recenter_pilot = 4L)), 4L)
    expect_error(.nl_pilot_n(list(recenter_pilot = 1L)), "integer")
    expect_error(.nl_pilot_n(list(recenter_pilot = "coarse")), "integer")
})

# --- 2. paired axes are not thinned per axis ---------------------------------

test_that("a path that crosses its own axes thins each; a paired one does not", {
    # `.NL_PATH_CROSSES` is the declared property, and the two consumers of it
    # must agree with the tables they are declared beside: every path named in
    # `.NL_PATH_AXES` needs an entry, and the registry -- whose axes
    # `.nl_fill_family_axes()` expands into one row per tuple -- must be the
    # paired one.
    expect_true(all(names(.NL_PATH_AXES) %in% names(.NL_PATH_CROSSES)))
    expect_false(.nl_path_crosses("registry"))
    expect_true(.nl_path_crosses("joint_single"))
    expect_true(.nl_path_crosses("copy"))
    expect_false(.nl_path_crosses("no_such_path"))

    # bym2 on the registry path holds (sigma, rho) PAIRED, so neither is thinned.
    blk <- list(type = "bym2",
                sigma_grid = rep(exp(seq(log(0.1), log(3), length.out = 5)), 4),
                rho_grid   = rep(seq(0.1, 0.9, length.out = 4), each = 5))
    r <- .nl_pilot_block(blk, "registry", 3L)
    expect_identical(r$moved, character(0))
    expect_true(all(c("sigma_grid", "rho_grid") %in% r$kept))
    expect_identical(r$block$sigma_grid, blk$sigma_grid)
    expect_identical(r$block$rho_grid, blk$rho_grid)

    # The same family on the joint single-block path crosses them itself, so
    # each is an axis and each thins.
    r2 <- .nl_pilot_block(list(type = "bym2",
                               sigma_grid = exp(seq(log(0.1), log(3), length.out = 5)),
                               rho_grid   = seq(0.1, 0.9, length.out = 4)),
                          "joint_single", 3L)
    expect_setequal(r2$moved, c("sigma_grid", "rho_grid"))
    expect_length(r2$block$sigma_grid, 3L)
    expect_length(r2$block$rho_grid, 3L)
})

test_that("an axis the caller left absent is thinned from the engine's own default", {
    # The driver would expand an absent axis to the full default, so the pilot
    # has to materialise it at the pilot resolution -- onto the PILOT block,
    # which no rescue reads for provenance.
    r <- .nl_pilot_block(list(type = "icar"), "joint_single", 3L)
    expect_identical(r$moved, "sigma_grid")
    expect_length(r$block$sigma_grid, 3L)
    full <- .nl_grid_axis(.nl_path_axis_key("icar", "joint_single", "sigma_grid"))
    expect_equal(range(r$block$sigma_grid), range(full))
})

test_that("a copy coefficient is thinned by RESOLUTION, and a stated axis is kept", {
    # The alpha axis carries prior structure -- an atom at 0 against a slab --
    # so a subsample of a STATED axis would state a different axis.
    kept <- .nl_pilot_alpha(list(arm = "cov", alpha_grid = c(0, 0.5, 1, 2)), 3L)
    expect_false(kept$moved)
    expect_identical(kept$spec$alpha_grid, c(0, 0.5, 1, 2))
    expect_null(kept$spec$alpha_n)

    moved <- .nl_pilot_alpha(list(arm = "cov"), 3L)
    expect_true(moved$moved)
    expect_identical(moved$spec$alpha_n, 3L)
    expect_null(moved$spec$alpha_grid)

    # `field_coef` is the single-block spelling of the same declaration.
    fc <- .nl_pilot_field_coef(list(field_coef = "alpha"), 3L)
    expect_true(fc$moved)
    expect_identical(fc$arm$field_coef[["n"]], 3L)
    expect_false(.nl_pilot_field_coef(list(field_coef = 1), 3L)$moved)
    expect_false(.nl_pilot_field_coef(list(), 3L)$moved)
    expect_false(.nl_pilot_field_coef(
        list(field_coef = list(name = "alpha", grid = c(0, 1, 2, 3))), 3L)$moved)
})

test_that("a pilot with nothing left to thin is not a pilot", {
    # Every axis already at or below the resolution: a pilot would be a second
    # copy of the same grid, so the fit runs exactly as it does with the knob off.
    p <- .nl_recenter_pilot(list(type = "icar", sigma_grid = c(0.5, 1, 2)),
                            phi_grid = NULL, copy = NULL, responses = NULL,
                            n = 3L, enabled = TRUE)
    expect_false(p$active)
    # Disabled placement leaves the pilot off whatever the axes look like --
    # nothing would read a pilot's detection.
    off <- .nl_recenter_pilot(list(type = "icar"), NULL, NULL, NULL,
                              n = 3L, enabled = FALSE)
    expect_false(off$active)
    expect_false(.nl_recenter_pilot(list(type = "icar"), NULL, NULL, NULL,
                                    n = NA_integer_, enabled = TRUE)$active)
})

# --- 3. end to end on the joint front door -----------------------------------

.pilot_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# A donor ICAR arm plus a gaussian arm reading the field through a copy
# coefficient and carrying its own dispersion axis: three outer axes
# (sigma, alpha, phi), the shape the fit gcol33/tulpa#636 measures has.
.pilot_sim <- function(n_s = 20L, n_per = 6L, N2 = 40L, f_coef = 0.2, seed = 11) {
    set.seed(seed)
    si <- rep(seq_len(n_s), each = n_per)
    base_p <- rep(0.02, n_s); base_p[1:5] <- 0.95
    y <- rbinom(length(si), 1, base_p[si])
    X <- cbind(1, rnorm(length(y), 0, 0.05))
    occ <- list(y = as.numeric(y), n_trials = rep(1L, length(y)), X = X,
                spatial_idx = as.integer(si), re_idx = rep(0, length(y)),
                n_re_groups = 0L, sigma_re = 1.0, family = "binomial", phi = 1.0)
    s2 <- sample.int(n_s, N2, replace = TRUE)
    f  <- rnorm(n_s, 0, 1.5)
    X2 <- cbind(1, rnorm(N2, 0, 1))
    y2 <- 0.4 + 0.8 * X2[, 2] + f_coef * f[s2] + rnorm(N2, 0, 0.5)
    cov <- list(y = y2, n_trials = rep(1L, N2), X = X2,
                spatial_idx = as.integer(s2), re_idx = rep(0, N2),
                n_re_groups = 0L, sigma_re = 1.0, family = "gaussian",
                phi = 0.25, field_coef = "alpha")
    list(responses = list(occ = occ, cov = cov), adj = .pilot_chain_adj(n_s))
}

.pilot_phi <- list(occ = NULL, cov = exp(seq(log(0.2), log(1.5), length.out = 4)))

.pilot_fit <- function(sim, prior, pilot, extra = list()) {
    ctrl <- c(list(diagnose_k = FALSE, progress = FALSE), extra)
    if (!isFALSE(pilot)) ctrl$recenter_pilot <- pilot
    tulpa_nested_laplace_joint(responses = sim$responses, prior = prior,
                               phi_grid = .pilot_phi, control = ctrl)
}

.pilot_prior <- function(sim, ...) {
    c(list(type = "icar", n_spatial_units = sim$adj$n_spatial_units,
           adj_row_ptr = sim$adj$adj_row_ptr,
           adj_col_idx = sim$adj$adj_col_idx,
           n_neighbors = sim$adj$n_neighbors), list(...))
}

test_that("a fit whose placement declines is bit-identical with the pilot on", {
    skip_on_cran()
    sim <- .pilot_sim()
    # A pinned axis declines the placement outright, which is the branch where
    # the pilot must cost the fit nothing beyond its own cells.
    prior <- .pilot_prior(sim, sigma_grid = exp(seq(log(1), log(12), length.out = 5)))
    off <- .pilot_fit(sim, prior, FALSE)
    on  <- .pilot_fit(sim, prior, TRUE)

    expect_identical(off$outer_grid_placement, "fixed")
    expect_identical(on$outer_grid_placement, "fixed")
    expect_identical(on$outer_grid_recenter_declined,
                     off$outer_grid_recenter_declined)
    expect_equal(on$theta_grid, off$theta_grid, tolerance = 0)
    expect_equal(on$log_marginal, off$log_marginal, tolerance = 0)
    expect_equal(on$weights, off$weights, tolerance = 0)
    expect_equal(unname(coef(on)), unname(coef(off)), tolerance = 0)
})

test_that("the pilot records what it thinned and what it detected on", {
    skip_on_cran()
    sim   <- .pilot_sim()
    fit   <- .pilot_fit(sim, .pilot_prior(sim), TRUE)
    p     <- fit$outer_grid_pilot

    expect_false(is.null(p))
    expect_identical(p$n_pilot, as.integer(.nl_recenter("pilot_n")))
    # Every axis of this shape is thinnable: the donor SD, the copy
    # coefficient's resolution, and the per-arm dispersion axis.
    expect_true("sigma_grid" %in% p$axes)
    expect_true(any(startsWith(p$axes, "phi_")))
    expect_true(any(startsWith(p$axes, "alpha")))
    # The detecting grid is strictly smaller than the reported one, and the
    # record says what the placement was decided from -- the reported grid is
    # not the grid the trigger was read on.
    expect_lt(p$cells, nrow(fit$theta_grid))
    expect_true(p$regime %in% c("spread", "collapsed_edge", "collapsed_interior"))
    expect_true(is.finite(p$ess_grid))
})

test_that("the pilot is a pre-screen: it never costs a placement the full grid would make", {
    skip_on_cran()
    # The engine's own collapse fixture, whose default sigma axis rails: both
    # arms must place, and the piloted one must place onto an axis that clears
    # the retired 3.0 ceiling just as the un-piloted one does.
    sim   <- .pilot_sim(N2 = 8L, f_coef = 0.05)
    prior <- .pilot_prior(sim)
    off <- .pilot_fit(sim, prior, FALSE)
    on  <- .pilot_fit(sim, prior, TRUE)
    if (identical(off$outer_grid_placement, "auto_recentered")) {
        expect_identical(on$outer_grid_placement, "auto_recentered")
        expect_gt(max(on$theta_grid[, "sigma"]), 3.0)
    }
    # Whatever each arm placed, the REPORTED grid is the full one: a pilot grid
    # is a detector and is never integrated.
    expect_identical(nrow(on$theta_grid), nrow(off$theta_grid))
    expect_gt(nrow(on$theta_grid), on$outer_grid_pilot$cells)
})

test_that("recenter_pilot is refused where the placement pass is off", {
    skip_on_cran()
    sim  <- .pilot_sim()
    fit  <- .pilot_fit(sim, .pilot_prior(sim), TRUE,
                       extra = list(auto_recenter = FALSE))
    # Nothing would read a pilot's detection, so no pilot runs and the fit says
    # what it always said.
    expect_null(fit$outer_grid_pilot)
    expect_identical(fit$outer_grid_recenter_declined, "auto_recenter_disabled")
})

# --- 4. what a recentred axis's reported h/sd actually is ---------------------

test_that("a recentred axis's h/sd is 1.25 * sd_used / sd_realized, not 1.25", {
    skip_on_cran()
    # gcol33/tulpa#636 read `outer_grid_h_over_sd = 6.03` on a recentred axis
    # against the "1.25 by construction" the node layout gives, and the two are
    # different quantities: the layout's 1.25 is in PLACEMENT SDs, while the
    # reported ratio divides by the weighted posterior SD the placed grid
    # realizes. Wherever the clamp SUBSTITUTED a bound they cannot agree, and
    # the identity below is what says which of the two a large reading is.
    sim   <- .pilot_sim(seed = 1)
    fit   <- .pilot_fit(sim, .pilot_prior(sim), FALSE)
    expect_identical(fit$outer_grid_placement, "auto_recentered")

    tags <- .joint_pareto_axis_tags(fit)
    skip_if(.k_is_decline(tags), "axis tags declined")
    j  <- match("sigma", colnames(fit$theta_grid))
    hs <- .nl_axis_h_over_sd(fit, "sigma", tags[j])

    spacing_sd <- 2 * .nl_recenter("span") / (.nl_recenter("n_pts") - 1)
    sd_used    <- unname(fit$outer_grid_recenter_sd_used[["sigma"]])
    # The realized spread, read off the placed grid's own weights in the axis's
    # own coordinate -- the denominator the report uses.
    u  <- log(sort(unique(fit$theta_grid[, "sigma"])))
    mw <- .nl_axis_marginal_w(fit, "sigma")
    uu <- log(mw$vals)
    mu <- sum(mw$w * uu)
    sd_realized <- sqrt(max(0, sum(mw$w * uu^2) - mu^2))

    expect_equal(hs, spacing_sd * sd_used / sd_realized, tolerance = 1e-8)
    # The node spacing itself IS `spacing_sd * sd_used`, which is the half of
    # the claim that does hold by construction.
    expect_equal(median(diff(u)), spacing_sd * sd_used, tolerance = 1e-8)
    # And where the floor bound, the reported ratio exceeds 1.25 by exactly the
    # factor the substitution widened by.
    if (identical(unname(fit$outer_grid_recenter_sd_clamp[["sigma"]]), "floor")) {
        expect_gt(sd_used, unname(fit$outer_grid_recenter_sd_raw[["sigma"]]))
        expect_gt(hs, spacing_sd)
    }
})

test_that("the grid ESS helper is the one statistic behind every read of it", {
    w <- c(0.25, 0.25, 0.25, 0.25)
    expect_equal(.nl_grid_ess(w), 4)
    expect_equal(.nl_grid_ess(c(1, 0, 0, 0)), 1)
    # Unnormalised input is the same read, and a cell the grid gave no mass is
    # not a cell the integration averaged over.
    expect_equal(.nl_grid_ess(c(2, 2, 2, 2)), 4)
    expect_equal(.nl_grid_ess(c(1, 1, NA, -1, 0)), 2)
    expect_true(is.na(.nl_grid_ess(numeric(0))))
    expect_true(is.na(.nl_grid_ess(c(0, 0))))
    expect_true(is.na(.nl_grid_ess(NULL)))
})
