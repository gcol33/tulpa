# Per-arm dispersion axes are placed like a prior block's scale axis
# (gcol33/tulpa#663).
#
# `.NL_REGISTRY_AXIS_FIELD` is keyed by prior BLOCK type, and a `phi_grid` axis
# is a hyperparameter of an ARM, so no rescue used to walk it: a fit whose own
# `grid_coarsest_axis` named `phi_pos` was naming the one axis nothing could
# move, and the advice that came with it (add nodes) was the one lever that did
# not apply. These cover the slot source, the provenance question that decides
# whether an axis may move, and the placement itself.

# --------------------------------------------------------------------------- #
# Slot source, provenance, triggers                                           #
# --------------------------------------------------------------------------- #

test_that("phi provenance reads auto_grid() marks by arm and strips them", {
    named <- list(pos = auto_grid(c(1, 2, 4)), occ = c(0.5, 1))
    p <- .nl_phi_provenance(named, c("occ", "pos"))
    expect_identical(p$auto, "pos")
    expect_false(is_auto_grid(p$phi_grid$pos))
    expect_identical(as.numeric(p$phi_grid$pos), c(1, 2, 4))
    expect_identical(as.numeric(p$phi_grid$occ), c(0.5, 1))

    # A positional list is keyed through `arm_names`, so the record is by arm
    # either way.
    pos <- .nl_phi_provenance(list(NULL, auto_grid(c(1, 2, 4))),
                              c("occ", "pos"))
    expect_identical(pos$auto, "pos")
    # `attr<-`(NULL, ...) DELETES a list element, which would renumber every
    # arm after it -- a NULL entry has to survive the strip untouched.
    expect_length(pos$phi_grid, 2L)
    expect_null(pos$phi_grid[[1L]])

    expect_identical(.nl_phi_provenance(NULL, "pos")$auto, character(0))
})

test_that("a dispersion axis is a pin unless the caller marked it", {
    # `.nl_axis_is_pinned()`'s "equal to the engine's own default" branch has no
    # counterpart: the engine has no default dispersion axis, so every axis that
    # exists was written by a caller and only the marker separates the two.
    expect_false(.nl_phi_axis_is_pinned("pos", "pos"))
    expect_true(.nl_phi_axis_is_pinned("pos", character(0)))
    expect_true(.nl_phi_axis_is_pinned("pos", "occ"))
})

test_that("phi slots are read off the fit's grid, not the argument", {
    res <- list(theta_grid = matrix(
        0, 2, 3, dimnames = list(NULL, c("sigma", "alpha", "phi_pos"))))
    s <- .nl_phi_axis_slots(res, list(occ = NULL, pos = c(1, 2)))
    expect_length(s, 1L)
    expect_identical(s[[1L]]$arm, "pos")
    expect_identical(s[[1L]]$axis, "phi_pos")

    # An arm the fit carries no column for -- a length-1 entry, which
    # `.normalise_phi_grid()` reads as no axis -- offers no slot.
    expect_length(.nl_phi_axis_slots(res, list(occ = c(1, 2))), 0L)
    expect_length(.nl_phi_axis_slots(res, NULL), 0L)
})

test_that("the placement stencil is asked for only when an axis wants it", {
    # A grid whose phi axis is resolved (spacing well under one posterior SD)
    # against one whose weight sits on a single node.
    fine <- exp(seq(log(0.9), log(1.1), length.out = 9))
    res_ok <- list(
        theta_grid = matrix(fine, ncol = 1L, dimnames = list(NULL, "phi_pos")),
        log_marginal = -0.5 * ((log(fine) - log(1)) / 0.5)^2)
    expect_false(.nl_placement_axis_wanted(res_ok, "phi_pos"))

    coarse <- exp(seq(log(1), log(60), length.out = 4))
    res_bad <- list(
        theta_grid = matrix(coarse, ncol = 1L, dimnames = list(NULL, "phi_pos")),
        log_marginal = -0.5 * ((log(coarse) - log(7.7)) / 0.01)^2)
    expect_true(.nl_placement_axis_wanted(res_bad, "phi_pos"))

    # Naming no axis is the gate that keeps every fit with nothing movable on
    # exactly the test it had before.
    expect_false(.nl_placement_axis_wanted(res_bad, character(0)))
    expect_false(.nl_placement_axis_wanted(res_bad, "phi_absent"))
})

test_that("a later rescue's record is appended to an earlier one's", {
    prev <- list(outer_grid_placement = "auto_recentered",
                 outer_grid_recenter_axes = "sigma",
                 outer_grid_recenter_attempts = 1L,
                 outer_grid_recenter_sd_used = c(sigma = 0.4),
                 outer_grid_prior_added = TRUE)
    new <- list(outer_grid_placement = "auto_recentered",
                outer_grid_recenter_axes = "phi_pos",
                outer_grid_recenter_attempts = 2L,
                outer_grid_recenter_sd_used = c(phi_pos = 0.2))
    m <- .nl_carry_recenter_stamps(new, prev)
    expect_setequal(m$outer_grid_recenter_axes, c("sigma", "phi_pos"))
    expect_equal(m$outer_grid_recenter_sd_used,
                 c(sigma = 0.4, phi_pos = 0.2))
    expect_identical(m$outer_grid_recenter_attempts, 3L)
    expect_true(m$outer_grid_prior_added)

    # An unplaced predecessor contributes nothing.
    expect_identical(
        .nl_carry_recenter_stamps(new, list(outer_grid_placement = "fixed")),
        new)
})

test_that("a field-SD decline survives a later dispersion placement (#720)", {
    # The stamp flow when the field-SD rescue declines and the dispersion rescue
    # then places: the refit carries no stamps, and the predecessor is UNPLACED.
    prev <- .nl_decline_axis(
        list(outer_grid_placement = "fixed",
             outer_grid_recenter_declined = "axis_pinned"),
        "sigma", "axis_pinned")
    new <- list(outer_grid_placement = "auto_recentered",
                outer_grid_recenter_axes = "phi_pos")
    res <- .nl_decline_recenter(.nl_carry_recenter_stamps(new, prev),
                                "grid_resolves_posterior")
    expect_identical(res$outer_grid_placement, "auto_recentered")
    expect_null(res$outer_grid_recenter_declined)
    expect_identical(res$outer_grid_axis_declined[["sigma"]], "axis_pinned")
    expect_identical(.tulpa_grid_axis_lever(
        list(coarsest = "sigma", axis_declined = res$outer_grid_axis_declined)),
        .tulpa_grid_axis_lever(
            list(coarsest = "sigma", axis_declined = c(sigma = "axis_pinned"))))

    # The new fit's own record wins on a shared axis, and an axis the new fit
    # moved drops the decline it carried.
    prev2 <- list(outer_grid_placement = "auto_recentered",
                  outer_grid_axis_declined = c(sigma = "grid_not_collapsed",
                                               phi_pos = "no_usable_curvature"))
    new2 <- list(outer_grid_placement = "auto_recentered",
                 outer_grid_recenter_axes = "phi_pos",
                 outer_grid_axis_declined = c(sigma = "axis_pinned"))
    m <- .nl_carry_recenter_stamps(new2, prev2)
    expect_identical(m$outer_grid_axis_declined, c(sigma = "axis_pinned"))
})

# --------------------------------------------------------------------------- #
# Reliability + resolution reporting                                          #
# --------------------------------------------------------------------------- #

test_that("grid reliability counts the cells solved, not the ones with weight", {
    rel <- .tulpa_grid_reliability(list(weights = c(1, rep(0, 123))))
    expect_identical(rel$n_grid, 124L)
    expect_equal(rel$ess_grid, 1)
    expect_equal(rel$max_weight, 1)
    expect_equal(rel$rel_ess_grid, 1 / 124)

    # A healthy grid is unchanged: nothing was filtered out of it.
    w <- rep(1, 8)
    expect_identical(.tulpa_grid_reliability(list(weights = w))$n_grid, 8L)
    expect_equal(.tulpa_grid_reliability(list(weights = w))$ess_grid, 8)
})

test_that("the coarsest axis reports the lever it actually has", {
    pinned <- .tulpa_grid_axis_lever(
        list(coarsest = "phi_pos", axis_declined = c(phi_pos = "axis_pinned")))
    expect_match(pinned, "auto_grid", fixed = TRUE)
    expect_match(pinned, "PINNED", fixed = TRUE)

    off <- .tulpa_grid_axis_lever(
        list(coarsest = "phi_pos",
             axis_declined = c(phi_pos = "auto_recenter_disabled")))
    expect_match(off, "auto_recenter", fixed = TRUE)

    # An axis the placement pass never spoke about keeps the generic advice.
    expect_null(.tulpa_grid_axis_lever(
        list(coarsest = "sigma", axis_declined = c(phi_pos = "axis_pinned"))))
    expect_null(.tulpa_grid_axis_lever(
        list(coarsest = "phi_pos", axis_declined = character(0))))
})

# --------------------------------------------------------------------------- #
# End to end: a coarse dispersion axis, placed and not placed                 #
# --------------------------------------------------------------------------- #


# Gaussian copy arm on a BYM2 field, spatial hyperparameters pinned at truth so
# the fit turns on the dispersion axis alone. Residual SD 0.3, so `phi` (a
# VARIANCE at every R-level door) is 0.09.


test_that("a pinned dispersion axis is left alone and says why", {
    skip_on_cran()
    fx <- .pgp_fixture()
    fit <- tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        phi_grid = list(pos = .PGP_COARSE))

    expect_identical(fit$outer_grid_placement, "fixed")
    # The per-axis record, which is what a reader of `grid_coarsest_axis` needs:
    # the whole-fit slot holds one reason for the fit, and on a fit where a
    # sibling axis moved it would not be about this one.
    expect_identical(fit$outer_grid_axis_declined[["phi_pos"]], "axis_pinned")
    expect_null(fit$outer_grid_recenter_axes)
    # The declared nodes are still the ones integrated (the adaptive refinement
    # adds cells between them; it never moves the span).
    nodes <- sort(unique(fit$theta_grid[, "phi_pos"]))
    expect_equal(range(nodes), range(.PGP_COARSE))
})

test_that("a marked dispersion axis is placed onto its own posterior", {
    skip_on_cran()
    fx <- .pgp_fixture()
    pinned <- tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        phi_grid = list(pos = .PGP_COARSE))
    placed <- tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        phi_grid = list(pos = auto_grid(.PGP_COARSE)))

    expect_identical(placed$outer_grid_placement, "auto_recentered")
    expect_true("phi_pos" %in% placed$outer_grid_recenter_axes)
    expect_false("phi_pos" %in% names(placed$outer_grid_axis_declined))
    # The field SD axis is pinned at truth and stayed put next to the moved
    # dispersion axis; its reason is still on the fit (gcol33/tulpa#720).
    expect_false("sigma" %in% placed$outer_grid_recenter_axes)
    expect_identical(placed$outer_grid_axis_declined[["sigma"]], "axis_pinned")

    # The placement is the estimate, not only the report. On the declared span
    # the dispersion posterior sits between nodes and comes back at roughly
    # twice the truth; placed, it recovers it.
    err <- function(f) abs(f$theta_mean[["phi_pos"]] - fx$truth_phi) /
        fx$truth_phi
    expect_gt(err(pinned), 0.5)
    expect_lt(err(placed), 0.15)

    # And the span it was placed on is orders narrower than the declared one.
    span <- function(f) diff(range(log(f$theta_grid[, "phi_pos"])))
    expect_lt(span(placed), 0.25 * span(pinned))
})

test_that("auto_recenter = FALSE holds a marked dispersion axis too", {
    skip_on_cran()
    fx <- .pgp_fixture(N = 800, n_s = 25)
    fit <- tulpa_nested_laplace_joint(
        responses = fx$responses, prior = fx$prior,
        phi_grid = list(pos = auto_grid(.PGP_COARSE)),
        control = list(auto_recenter = FALSE))
    expect_identical(fit$outer_grid_placement, "fixed")
    expect_identical(fit$outer_grid_axis_declined[["phi_pos"]],
                     "auto_recenter_disabled")
})

test_that("a fit with no dispersion axis carries no per-axis decline", {
    skip_on_cran()
    fx <- .pgp_fixture(N = 800, n_s = 25)
    fit <- tulpa_nested_laplace_joint(responses = fx$responses,
                                      prior = fx$prior)
    expect_false("phi_pos" %in% colnames(fit$theta_grid))
    expect_false("phi_pos" %in% names(fit$outer_grid_axis_declined))
    # The pinned field SD axis is the only one the placement pass spoke about.
    expect_identical(fit$outer_grid_axis_declined, c(sigma = "axis_pinned"))
})
