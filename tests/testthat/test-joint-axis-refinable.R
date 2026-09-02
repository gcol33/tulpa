# A caller-stated outer-grid axis is a bound (gcol33/tulpa#658).
#
# Refinability used to be decided by axis NAME -- `alpha` and every `phi_*` axis
# opted into the adaptive-grid and consistency passes whether or not the caller
# had written their nodes down -- so a fit could integrate a range the caller
# never asked for and report it nowhere. It is decided by PROVENANCE now: an
# axis whose nodes the caller stated is densified inside that range and never
# extended past its ends; one the engine placed may still be followed out.
#
# The contract asserted here:
#   * the per-axis mode reaches the spec field `.hyper_refinable_axes()` reads;
#   * a non-extendable axis's proposals stay inside its declared node span,
#     on both the boundary and the consistency path;
#   * provenance -- not the axis name -- is what picks the default;
#   * an engine-placed axis still extends exactly as it did;
#   * the reported span separates the half-node-step term from refinement.


# --------------------------------------------------------------------------- #
# 1. The spec field                                                            #
# --------------------------------------------------------------------------- #

test_that("hyper_axis_spec carries `extend`, defaulting to the historical answer", {
    s <- hyper_axis_spec("alpha", grid = c(0.2, 0.5), log_scale = TRUE,
                         refinable = TRUE)
    expect_true(s$extend)
    expect_true(tulpa:::.hyper_axis_may_extend(s))

    b <- hyper_axis_spec("alpha", grid = c(0.2, 0.5), log_scale = TRUE,
                         refinable = TRUE, extend = FALSE)
    expect_false(b$extend)
    expect_false(tulpa:::.hyper_axis_may_extend(b))

    # A spec built before the field existed says nothing about it and keeps the
    # historical answer, so no caller of the generic passes is silently bounded.
    legacy <- s
    legacy$extend <- NULL
    expect_true(tulpa:::.hyper_axis_may_extend(legacy))

    expect_error(hyper_axis_spec("alpha", grid = c(0.2, 0.5), extend = NA),
                 "extend")
    expect_error(hyper_axis_spec("alpha", grid = c(0.2, 0.5),
                                 extend = c(TRUE, FALSE)),
                 "extend")
})


test_that("the node limit is the declared span, and skips a log axis's atom", {
    b <- hyper_axis_spec("alpha", grid = c(0, 0.2, 0.35, 0.5), log_scale = TRUE,
                         bounds = c(0, Inf), refinable = TRUE, extend = FALSE)
    expect_equal(tulpa:::.hyper_axis_node_limit(b), c(0.2, 0.5))

    e <- hyper_axis_spec("alpha", grid = c(0, 0.2, 0.5), log_scale = TRUE,
                         bounds = c(0, Inf), refinable = TRUE, extend = TRUE)
    expect_null(tulpa:::.hyper_axis_node_limit(e))

    # One continuum node leaves no interior, so nothing may be proposed.
    one <- hyper_axis_spec("alpha", grid = c(0, 0.3), log_scale = TRUE,
                           bounds = c(0, Inf), refinable = TRUE, extend = FALSE)
    expect_length(tulpa:::.hyper_clip_to_node_limit(c(0.1, 0.3, 0.9), one), 0L)
})


# --------------------------------------------------------------------------- #
# 2. The per-axis mode reaches the field the refinement passes read            #
# --------------------------------------------------------------------------- #

test_that(".joint_axis_specs reads the resolved mode, not the axis name", {
    grids <- list(sigma = c(0.5, 1, 2),
                  alpha = c(0, 0.2, 0.5),
                  phi_y = c(0.5, 1, 2))
    cp <- list(has_copy = TRUE)

    specs <- tulpa:::.joint_axis_specs(
        grids, cp, axis_refine = c(alpha = "densify", phi_y = "extend"))
    by <- stats::setNames(specs, vapply(specs, `[[`, character(1), "name"))

    expect_true(by$alpha$refinable)
    expect_false(by$alpha$extend)
    expect_true(by$phi_y$refinable)
    expect_true(by$phi_y$extend)
    expect_false(by$sigma$refinable)

    # This is the field `.hyper_refinable_axes()` reads, which is what the issue
    # says was already in place and unreachable.
    expect_equal(tulpa:::.hyper_refinable_names(specs), c("alpha", "phi_y"))

    none <- tulpa:::.joint_axis_specs(
        grids, cp, axis_refine = c(alpha = "none", phi_y = "none"))
    expect_length(tulpa:::.hyper_refinable_names(none), 0L)

    # No provenance supplied -- a spec list rebuilt from an assembled grid, which
    # no refinement pass reads -- keeps the driver's own eligibility.
    bare <- tulpa:::.joint_axis_specs(grids, cp)
    expect_equal(tulpa:::.hyper_refinable_names(bare), c("alpha", "phi_y"))
})


# --------------------------------------------------------------------------- #
# 3. Proposals on a bounded axis                                               #
# --------------------------------------------------------------------------- #

test_that("a bounded axis is densified inside its span and never past it", {
    lev <- c(0.2, 0.35, 0.5)
    mk <- function(extend) {
        hyper_axis_spec("alpha", grid = lev, log_scale = TRUE,
                        bounds = c(0, Inf), refinable = TRUE, extend = extend)
    }
    e <- mk(TRUE)
    b <- mk(FALSE)

    pts_e <- tulpa:::.hyper_propose_axis_extension(e, lev, "max")
    expect_gt(max(pts_e), 0.5)

    # The midpoint between the two outermost levels, and nothing else: on a log
    # axis that is the geometric mean.
    pts_b <- tulpa:::.hyper_propose_axis_extension(b, lev, "max")
    expect_equal(pts_b, sqrt(0.35 * 0.5))
    expect_true(all(pts_b > 0.35 & pts_b < 0.5))

    pts_lo <- tulpa:::.hyper_propose_axis_extension(b, lev, "min")
    expect_equal(pts_lo, sqrt(0.2 * 0.35))

    # The clip is on the proposal, not only on the caller's request, so an
    # explicit `extend_ok` cannot walk a bounded axis outward.
    forced <- tulpa:::.hyper_propose_axis_extension(b, lev, "max",
                                                    extend_ok = TRUE)
    expect_true(all(forced >= 0.2 & forced <= 0.5))

    # Interior densification is inside by construction, so the two agree there.
    di_e <- tulpa:::.hyper_propose_interior_densification(e, lev, 2L,
                                                          do_left = TRUE,
                                                          do_right = TRUE)
    di_b <- tulpa:::.hyper_propose_interior_densification(b, lev, 2L,
                                                          do_left = TRUE,
                                                          do_right = TRUE)
    expect_equal(di_b, di_e)
})


test_that("consistency points are clipped to a bounded axis's span", {
    lev <- c(0.5, 1, 2)
    mk <- function(extend) {
        hyper_axis_spec("phi_y", grid = lev, log_scale = TRUE,
                        bounds = c(0, Inf), refinable = TRUE, extend = extend)
    }

    # A spread this narrow places every point inside the span, so the clip must
    # not touch them: a bounded axis is still refined, only not widened.
    inside_e <- tulpa:::.hyper_propose_consistency_points(mk(TRUE), 1, 0.35, lev)
    inside_b <- tulpa:::.hyper_propose_consistency_points(mk(FALSE), 1, 0.35, lev)
    expect_gt(length(inside_b), 0L)
    expect_equal(inside_b, inside_e)
    expect_true(all(inside_b >= 0.5 & inside_b <= 2))

    # A wide one places them all outside; the extendable axis takes them and the
    # bounded one takes none.
    wide_e <- tulpa:::.hyper_propose_consistency_points(mk(TRUE), 1, 1, lev)
    wide_b <- tulpa:::.hyper_propose_consistency_points(mk(FALSE), 1, 1, lev)
    expect_true(any(wide_e > 2))
    expect_true(any(wide_e < 0.5))
    expect_length(wide_b, 0L)
})


# --------------------------------------------------------------------------- #
# 4. Provenance, not the axis name, picks the default                          #
# --------------------------------------------------------------------------- #

test_that("a stated alpha axis is a bound and a placed one is not", {
    arm <- function(fc) {
        tulpa:::.normalise_arm_field_coef(list(field_coef = fc), 1L)
    }
    stated <- arm(list(name = "alpha", grid = c(0.2, 0.5)))
    marked <- arm(list(name = "alpha", grid = auto_grid(c(0.2, 0.5))))
    by_n   <- arm(list(name = "alpha", n = 9L))
    named  <- arm("alpha")
    deflt  <- arm(list(name = "alpha",
                       grid = tulpa:::.nl_grid_axis("copy_alpha")))

    expect_true(tulpa:::.joint_axis_is_stated("alpha", list(stated)))
    # `as.numeric()` drops the marker, so the normaliser has to record it.
    expect_true(marked$field_coef_axis$grid_auto)
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(marked)))
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(by_n)))
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(named)))
    # Nodes that ARE the engine's own axis carry nothing a statement would add.
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(deflt)))
    # No copy arm at all.
    expect_false(tulpa:::.joint_axis_is_stated("alpha", list(list())))
})


test_that("a phi axis is stated when the caller gave its nodes", {
    st <- tulpa:::.joint_axis_is_stated("phi_y", list(),
                                        phi_grid = list(y = c(0.5, 1, 2)),
                                        arm_names = c("y", "z"))
    expect_true(st)

    au <- tulpa:::.joint_axis_is_stated("phi_y", list(),
                                        phi_grid = list(y = auto_grid(c(0.5, 1, 2))),
                                        arm_names = c("y", "z"))
    expect_false(au)

    # Positional `phi_grid`, the other shape `.normalise_phi_grid()` accepts.
    pos <- tulpa:::.joint_axis_is_stated("phi_z", list(),
                                         phi_grid = list(NULL, c(0.5, 1, 2)),
                                         arm_names = c("y", "z"))
    expect_true(pos)

    # An arm with no axis of its own.
    expect_false(tulpa:::.joint_axis_is_stated("phi_z", list(),
                                               phi_grid = list(y = c(0.5, 1)),
                                               arm_names = c("y", "z")))
})


test_that(".joint_axis_refine_modes resolves every axis of one grid", {
    grids <- list(sigma = c(0.5, 1), alpha = c(0, 0.2, 0.5),
                  phi_y = c(0.5, 1, 2))
    cp <- list(has_copy = TRUE)
    arm_stated <- tulpa:::.normalise_arm_field_coef(
        list(field_coef = list(name = "alpha", grid = c(0.2, 0.5))), 1L)
    arm_placed <- tulpa:::.normalise_arm_field_coef(
        list(field_coef = list(name = "alpha", n = 9L)), 1L)

    m <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_stated),
                                          phi_grid = list(y = c(0.5, 1, 2)),
                                          arm_names = "y")
    expect_equal(m[["alpha"]], "densify")
    expect_equal(m[["phi_y"]], "densify")
    expect_equal(m[["sigma"]], "none")

    p <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_placed),
                                          phi_grid = NULL, arm_names = "y")
    expect_equal(p[["alpha"]], "extend")
    # With no caller nodes recorded there is no statement to honour, so the
    # axis falls to the engine-placed default.
    expect_equal(p[["phi_y"]], "extend")

    u <- tulpa:::.joint_axis_refine_modes(grids, cp, list(arm_stated),
                                          phi_grid = list(y = c(0.5, 1, 2)),
                                          arm_names = "y",
                                          user = c(alpha = "extend",
                                                   phi_y = "none"))
    expect_equal(u[["alpha"]], "extend")
    expect_equal(u[["phi_y"]], "none")

    # A fit with no copy arm has no alpha axis to resolve.
    nc <- tulpa:::.joint_axis_refine_modes(grids, list(has_copy = FALSE),
                                           list(list()))
    expect_false("alpha" %in% names(nc))
})


# --------------------------------------------------------------------------- #
# 5. The control knob                                                          #
# --------------------------------------------------------------------------- #

test_that("control$axis_refine is validated against the fit's own axes", {
    axes <- c("sigma", "alpha", "phi_y")

    expect_null(tulpa:::.joint_check_axis_refine(NULL, axes))
    expect_equal(tulpa:::.joint_check_axis_refine(c(alpha = "none"), axes),
                 c(alpha = "none"))
    expect_equal(tulpa:::.joint_check_axis_refine(list(alpha = "densify"), axes),
                 c(alpha = "densify"))

    # One unnamed value applies to every refinable axis, and only to those.
    blanket <- tulpa:::.joint_check_axis_refine("none", axes)
    expect_equal(sort(names(blanket)), c("alpha", "phi_y"))
    expect_true(all(blanket == "none"))

    expect_error(tulpa:::.joint_check_axis_refine(c(alpha = "widen"), axes),
                 "unknown mode")
    expect_error(tulpa:::.joint_check_axis_refine(c(alfa = "none"), axes),
                 "does not have")
    # Asking for nodes on an axis this driver never places any on is an error,
    # not a silent no-op that leaves the caller believing it took effect.
    expect_error(tulpa:::.joint_check_axis_refine(c(sigma = "extend"), axes),
                 "only \"none\"")
    expect_equal(tulpa:::.joint_check_axis_refine(c(sigma = "none"), axes),
                 c(sigma = "none"))
    expect_error(tulpa:::.joint_check_axis_refine(c("none", "extend"), axes),
                 "named by axis")
})


# --------------------------------------------------------------------------- #
# 6. The reported span, and the two widening terms                             #
# --------------------------------------------------------------------------- #

test_that("axis_span separates the half-node-step term from refinement", {
    two <- c(0.2, 0.5)
    spec <- hyper_axis_spec("alpha", grid = two, log_scale = TRUE,
                            bounds = c(0, Inf), refinable = TRUE,
                            extend = FALSE)
    tg <- matrix(two, ncol = 1L, dimnames = list(NULL, "alpha"))
    sp <- tulpa:::.joint_axis_span(tg, tg, list(spec))[["alpha"]]

    expect_equal(sp$nodes, two)
    expect_equal(sp$refine, "densify")
    expect_equal(unname(sp$n_nodes), c(2L, 2L))
    expect_equal(sp$integrated, sp$declared)
    # For k equally log-spaced nodes the declared support is k / (k - 1) times
    # the node range on the axis's own coordinate: 2x at two nodes.
    expect_equal(diff(log(sp$declared)) / diff(log(sp$nodes)), 2)

    nine <- exp(seq(log(0.2), log(0.5), length.out = 9L))
    spec9 <- hyper_axis_spec("alpha", grid = nine, log_scale = TRUE,
                             bounds = c(0, Inf), refinable = TRUE,
                             extend = FALSE)
    tg9 <- matrix(nine, ncol = 1L, dimnames = list(NULL, "alpha"))
    sp9 <- tulpa:::.joint_axis_span(tg9, tg9, list(spec9))[["alpha"]]
    expect_equal(diff(log(sp9$declared)) / diff(log(sp9$nodes)), 9 / 8)

    # A grid refinement moved past the declared end: the second term is then
    # visible as `integrated` reaching past `declared`.
    ext <- hyper_axis_spec("alpha", grid = two, log_scale = TRUE,
                           bounds = c(0, Inf), refinable = TRUE, extend = TRUE)
    tgf <- matrix(c(0.08, 0.2, 0.5, 1.25), ncol = 1L,
                  dimnames = list(NULL, "alpha"))
    spe <- tulpa:::.joint_axis_span(tg, tgf, list(ext))[["alpha"]]
    expect_equal(spe$refine, "extend")
    expect_equal(spe$nodes, two)
    expect_gt(spe$integrated[2L], spe$declared[2L])
    expect_lt(spe$integrated[1L], spe$declared[1L])
    expect_equal(unname(spe$n_nodes), c(2L, 4L))

    # An axis with a single continuum level has no span to report, matching what
    # `.hyper_grid_supports()` leaves out.
    one <- hyper_axis_spec("alpha", grid = c(0, 0.3), log_scale = TRUE,
                           bounds = c(0, Inf))
    tg1 <- matrix(c(0, 0.3), ncol = 1L, dimnames = list(NULL, "alpha"))
    expect_null(tulpa:::.joint_axis_span(tg1, tg1, list(one)))
})


# --------------------------------------------------------------------------- #
# 7. End to end: the fit integrates where the caller said                      #
# --------------------------------------------------------------------------- #

# The refinement fixture of test-nested-laplace-joint-adaptive-grid.R: a copy
# axis whose maximum (0.6) sits below the truth (2.0), so the boundary carries
# mass and the pass fires on every arm below.
.axr_sim <- function(seed = 6L, N = 600L, n_s = 50L) {
    set.seed(seed)
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    n_nb <- vapply(nbr, length, integer(1))
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    rw    <- cumsum(stats::rnorm(n_s, 0, 1 / sqrt(n_s)))
    phi_s <- rw - mean(rw)
    Xocc  <- cbind(1, stats::rnorm(N))
    occur <- stats::rbinom(N, 1, stats::plogis(
        as.numeric(Xocc %*% c(-0.3, 0.5)) + phi_s[spatial_idx]))
    is_pos <- occur == 1L
    Xpos <- Xocc[is_pos, , drop = FALSE]
    spi  <- spatial_idx[is_pos]
    y_pos <- stats::rnorm(sum(is_pos),
                          as.numeric(Xpos %*% c(0.2, -0.4)) + 2.0 * phi_s[spi],
                          0.3)
    list(N = N, n_s = n_s, n_nb = n_nb, nbr = nbr,
         spatial_idx = as.integer(spatial_idx), Xocc = Xocc, occur = occur,
         Xpos = Xpos, y_pos = y_pos, spi = as.integer(spi))
}

.axr_fit <- function(sim, alpha_grid, control = list()) {
    ctrl <- utils::modifyList(
        list(adaptive_grid = TRUE, diagnose_k = FALSE,
             var_of_means_consistency = FALSE), control)
    tulpa_nested_laplace_joint(
        responses = list(
            occ = list(y = as.numeric(sim$occur), n_trials = rep(1L, sim$N),
                       X = sim$Xocc, spatial_idx = sim$spatial_idx,
                       re_idx = rep(0, sim$N), n_re_groups = 0L,
                       sigma_re = 1.0, family = "binomial", phi = 1.0),
            pos = list(y = sim$y_pos, n_trials = rep(1L, length(sim$y_pos)),
                       X = sim$Xpos, spatial_idx = sim$spi,
                       re_idx = rep(0, length(sim$y_pos)), n_re_groups = 0L,
                       sigma_re = 1.0, family = "gaussian", phi = 0.09,
                       field_coef = list(name = "alpha", grid = alpha_grid))),
        prior = list(type = "icar", n_spatial_units = sim$n_s,
                     adj_row_ptr = as.integer(c(0L, cumsum(sim$n_nb))),
                     adj_col_idx = as.integer(unlist(sim$nbr)) - 1L,
                     n_neighbors = as.integer(sim$n_nb),
                     sigma_grid = c(0.6, 1.0, 1.5)),
        control = ctrl)
}

test_that("a stated copy axis is densified, never extended, under adaptive_grid", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)

    fit <- .axr_fit(sim, stated)
    lev <- sort(unique(as.numeric(fit$theta_grid[, "alpha"])))

    # The pass still fires -- this is densify, not refusal -- and every node it
    # placed sits inside the range the caller wrote down.
    expect_false(is.null(fit$adaptive_grid_info))
    expect_gt(sum(fit$adaptive_grid_info$n_points_added), 0L)
    expect_gt(length(lev), length(stated))
    expect_equal(range(lev), range(stated))

    # The same fixture with the axis declared a default extends past 0.6, so the
    # difference is provenance and nothing else.
    fit_auto <- .axr_fit(sim, auto_grid(stated))
    expect_gt(max(fit_auto$theta_grid[, "alpha"]), max(stated) + 1e-6)

    # And the caller can restore that per axis without touching adaptive_grid.
    fit_ext <- .axr_fit(sim, stated,
                        control = list(axis_refine = c(alpha = "extend")))
    expect_gt(max(fit_ext$theta_grid[, "alpha"]), max(stated) + 1e-6)
})


test_that("axis_refine = 'none' keeps the copy axis out of refinement entirely", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)

    fit <- .axr_fit(sim, stated,
                    control = list(axis_refine = c(alpha = "none")))
    expect_null(fit$adaptive_grid_info)
    expect_equal(length(fit$log_marginal), 9L)   # 3 sigma x 3 alpha, untouched
    expect_equal(sort(unique(as.numeric(fit$theta_grid[, "alpha"]))), stated)
})


test_that("a fit reports the span it worked over on a stated axis", {
    skip_on_cran()
    sim <- .axr_sim()
    stated <- c(0.2, 0.4, 0.6)
    fit <- .axr_fit(sim, stated)

    sp <- fit$axis_span[["alpha"]]
    expect_false(is.null(sp))
    expect_equal(sp$refine, "densify")
    expect_equal(sp$nodes, range(stated))
    # The declared support is the stated nodes widened by half a node step at
    # each end -- the only widening term left once the axis cannot be extended.
    expect_lt(sp$declared[1L], min(stated))
    expect_gt(sp$declared[2L], max(stated))
    # Densification adds interior nodes, which narrows the outermost cells; the
    # integrated span therefore sits inside the declared one and still contains
    # every node.
    expect_gte(sp$integrated[1L], sp$declared[1L])
    expect_lte(sp$integrated[2L], sp$declared[2L])
    expect_lte(sp$integrated[1L], min(stated))
    expect_gte(sp$integrated[2L], max(stated))
    # `axis_support` is the same interval, read off the same final grid.
    expect_equal(sp$integrated, fit$axis_support[["alpha"]])
})
