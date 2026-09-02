# CCD placement: what it costs, what it reports, and what it says while it runs
# (gcol33/tulpa#652, gcol33/tulpa#653).
#
# The joint CCD mode-find reads the outer log-posterior through the same inner
# Newton solve an outer grid cell costs, so placing a design is paid for in the
# same currency as the integration it replaces. Two things follow, and these
# tests hold both:
#
#   * the placement has a CEILING in inner solves (#653). It is projected before
#     any solve is paid for, metered while they are, and reported afterwards, and
#     it declines to the tensor grid under its own reason rather than under a
#     mode-find failure.
#   * the placement SAYS SO while it runs (#652). Before this the phase emitted
#     nothing at all -- the outer-grid progress reporter belongs to the grid loop,
#     which the mode-find precedes -- so a working placement and a hung process
#     were indistinguishable from outside.
#
# Almost everything here drives the placement against synthetic log-posteriors,
# so the eval counts are exact arithmetic rather than timing, and the whole file
# runs on CRAN apart from the two front-door fits at the end.
#
# Where the spend is charged matters for reading these: EVERY probe -- seed,
# step calibration, mode-find round, line search -- reaches the objective
# through one evaluator, and that evaluator is the single place the meter is
# charged. So a test that hands one of those passes a bare objective is testing
# its heartbeat, not its accounting; the accounting is asserted on the
# production path, through `.joint_ccd_grid()`.

# --- a concave quadratic in u-space, evaluated through the design's own
# --- coordinate mapping, with every batch it is handed recorded.
#
# `.joint_ccd_grid()` hands `eval_logpost` PHYSICAL theta (columns named by the
# axes), so the fixture maps back through the same per-axis forward transform
# the design places its nodes on.
.ccd_bud_eval <- function(tags, m, prec, rec) {
    function(theta) {
        theta <- as.matrix(theta)
        rec$calls <- rec$calls + 1L
        rec$rows  <- rec$rows + nrow(theta)
        rec$sizes <- c(rec$sizes, nrow(theta))
        u <- theta
        for (j in seq_along(tags)) {
            u[, j] <- tulpa:::.joint_pareto_fwd(tags[[j]], theta[, j])
        }
        apply(u, 1L, function(ui) -0.5 * sum(prec * (ui - m)^2))
    }
}

.ccd_bud_rec <- function() {
    e <- new.env(parent = emptyenv())
    e$calls <- 0L; e$rows <- 0; e$sizes <- numeric(0)
    e
}

# A four-axis multi-block layout the transform registry can read: a BYM2 block
# carrying (sigma, rho, alpha) and an ICAR block carrying (sigma). Three levels
# an axis, so the tensor grid it replaces is 81 cells. No copy atom (no level at
# alpha = 0), so the design is a single mixture component on all four axes.
.ccd_bud_fixture <- function() {
    axis_names   <- c("b1.sigma", "b1.rho", "b1.alpha", "b2.sigma")
    axis_offsets <- c(0L, 3L, 4L)
    prepared     <- list(list(type = "bym2"), list(type = "icar"))
    # The coordinate each axis is DESIGNED on, taken from the engine's own
    # resolver so the fixture's objective is quadratic in the same u the design
    # places its nodes in: a copy alpha's continuum is designed in log alpha,
    # the BYM2 mixing weight on the logit.
    tags <- tulpa:::.joint_ccd_coord_tags(
        axis_names,
        tulpa:::.joint_ccd_axis_tags(axis_names, axis_offsets, prepared))
    list(
        axis_names   = axis_names,
        axis_offsets = axis_offsets,
        prepared     = prepared,
        axis_values  = list(c(0.5, 1, 2), c(0.3, 0.6, 0.9),
                            c(0.5, 1, 2), c(0.5, 1, 2)),
        tags         = tags,
        # The mode, in u-space: inside the box, inside one trust radius of the
        # grid-median seed, and away from every axis edge.
        mode         = c(0.3, 0.6, -0.2, 0.15),
        prec         = c(2, 3, 1.5, 2.5))
}

.ccd_bud_cfg <- function(...) {
    utils::modifyList(as.list(tulpa:::.CCD_PLACEMENT), list(...))
}

# The three arithmetic constants this file's exact eval counts rest on, at
# d = 4: one stencil, one line search, one closing stencil.
.CCD_BUD_STENCIL <- 1 + 2 * 4 + 4 * choose(4, 2)     # 33
.CCD_BUD_HALVE   <- tulpa:::.ccd_placement("max_halve") + 1


# --- 1. the cost arithmetic --------------------------------------------------

test_that("the stencil / round / attempt projections are the issue's arithmetic", {
    cfg <- .ccd_bud_cfg()
    # One stencil: centre + 2d axials + 4 C(d, 2) mixed corners. 33 at d = 4 is
    # the number gcol33/tulpa#653 is written around.
    expect_equal(tulpa:::.ccd_stencil_evals(4), 33)
    expect_equal(tulpa:::.ccd_stencil_evals(3), 19)
    expect_equal(tulpa:::.ccd_stencil_evals(2), 9)
    expect_equal(tulpa:::.ccd_stencil_evals(1), 3)
    # One round: the stencil plus the batched line search, which always
    # evaluates all `max_halve + 1` candidates in one call.
    expect_equal(tulpa:::.ccd_round_evals(4, cfg), 33 + cfg$max_halve + 1)
    # The cheapest placement that can still produce a design: opening centre
    # evaluation, one round, closing stencil.
    expect_equal(tulpa:::.ccd_min_evals(4, cfg), 1 + 2 * 33 + (cfg$max_halve + 1))
    # The designed first attempt, and the escalation behind it.
    expect_equal(tulpa:::.ccd_first_evals(4, cfg),
                 1 + cfg$first_rounds * (33 + cfg$max_halve + 1) + 33)
    expect_equal(tulpa:::.ccd_rescue_evals(4, 256, cfg),
                 256 + cfg$calibrate_rounds * 33 +
                     1 + cfg$max_rounds * (33 + cfg$max_halve + 1) + 33)
    # A component with no free axis is one node at the origin of the copy
    # coordinates and costs nothing to place.
    expect_equal(tulpa:::.ccd_min_evals(0L, cfg), 0)
    expect_equal(tulpa:::.ccd_first_evals(0L, cfg), 0)
    expect_equal(tulpa:::.ccd_design_nodes(0L), 1)
    # The design node count is read off the design builder, not restated.
    expect_equal(tulpa:::.ccd_design_nodes(4L), ccd_grid(4L)$n_points)
    expect_equal(tulpa:::.ccd_design_nodes(4L), 25)
})

test_that("the seed projection follows the seed pass's own dispatch", {
    cfg <- .ccd_bud_cfg()
    # Under the cap the joint grid is evaluated in one batch.
    expect_equal(tulpa:::.ccd_seed_evals(c(3L, 3L, 3L), cfg), 27)
    # Above it the pass falls back to the coordinate sweep, one call per axis
    # carrying at least two levels.
    big <- rep(8L, 4L)                                   # 4096 > 256
    expect_equal(tulpa:::.ccd_seed_evals(big, cfg), 32)
    # A pinned axis contributes no sweep call: 9 * 1 * 9 * 9 = 729 is past the
    # cap, so the sweep visits the three axes carrying levels.
    expect_equal(tulpa:::.ccd_seed_evals(c(9L, 1L, 9L, 9L), cfg), 27)
})

test_that("the budget is the tensor grid it replaces, floored at the first attempt", {
    cfg <- .ccd_bud_cfg()
    # 81 tensor cells, a 25-node design: at evals_per_cell = 1 the cost-neutral
    # ceiling is 56, but the designed first attempt (354) is exempt, so the
    # ceiling bounds the ESCALATION and not the path the CCD was built for.
    expect_equal(tulpa:::.ccd_budget(81, 25, 4L, cfg),
                 tulpa:::.ccd_first_evals(4L, cfg))
    # With the floor off the multiple is a hard cap on the whole placement.
    expect_equal(tulpa:::.ccd_budget(81, 25, 4L, .ccd_bud_cfg(budget_floor = FALSE)),
                 56)
    expect_equal(
        tulpa:::.ccd_budget(81, 25, 4L,
                            .ccd_bud_cfg(budget_floor = FALSE,
                                         evals_per_cell = 2)),
        112)
    # A grid large enough that the cost argument, not the floor, governs.
    expect_equal(tulpa:::.ccd_budget(5000, 25, 4L, cfg), 4975)
    # Inf switches the ceiling off outright.
    expect_identical(tulpa:::.ccd_budget(81, 25, 4L,
                                         .ccd_bud_cfg(evals_per_cell = Inf)),
                     Inf)
    # A design that is not smaller than the grid it replaces has nothing to
    # spend once the floor is off.
    expect_equal(tulpa:::.ccd_budget(10, 25, 4L,
                                     .ccd_bud_cfg(budget_floor = FALSE)), 0)
})

test_that("the placement registry holds the loop caps the projection reads", {
    # Pinned deliberately: these are the numbers the mode-find loops run on AND
    # the numbers the up-front projection is computed from, so a change to
    # either has to move both together.
    expect_identical(tulpa:::.ccd_placement("first_rounds"), 8L)
    expect_identical(tulpa:::.ccd_placement("max_rounds"), 30L)
    expect_identical(tulpa:::.ccd_placement("calibrate_rounds"), 4L)
    expect_identical(tulpa:::.ccd_placement("seed_max_pts"), 256L)
    expect_identical(tulpa:::.ccd_placement("max_halve"), 6L)
    expect_equal(tulpa:::.ccd_placement("evals_per_cell"), 1)
    expect_true(tulpa:::.ccd_placement("budget_floor"))
    # The curvature-reuse path shipped in 0.2.14 opt-in, was measured against a
    # full stencil every round, and was removed in 0.2.15: it spent no fewer
    # inner solves and centred the design somewhere worse (gcol33/tulpa#662).
    # The knob is gone from the registry AND refused at the door, so a caller
    # still passing it is told rather than silently ignored.
    expect_false("stencil_reuse" %in% names(tulpa:::.CCD_PLACEMENT))
    expect_false("refresh_every" %in% names(tulpa:::.CCD_PLACEMENT))
    expect_error(tulpa:::.ccd_placement("stencil_reuse"),
                 "Unknown CCD placement")
    expect_error(tulpa:::.ccd_placement("no_such_knob"), "Unknown CCD placement")
    # The loop defaults ARE the registry, not a second copy beside it.
    expect_identical(formals(tulpa:::.joint_ccd_modefind)$max_rounds,
                     quote(.ccd_placement("max_rounds")))
    expect_identical(formals(tulpa:::.joint_ccd_modefind)$max_halve,
                     quote(.ccd_placement("max_halve")))
    expect_identical(formals(tulpa:::.joint_ccd_grid_seed)$max_pts,
                     quote(.ccd_placement("seed_max_pts")))
    expect_identical(formals(tulpa:::.joint_ccd_calibrate_step)$rounds,
                     quote(.ccd_placement("calibrate_rounds")))
})

test_that("control knobs resolve onto the registry, and bad ones are refused", {
    expect_identical(tulpa:::.ccd_placement_args(list()),
                     as.list(tulpa:::.CCD_PLACEMENT))
    cfg <- tulpa:::.ccd_placement_args(
        list(ccd_budget = 0.5, ccd_budget_floor = FALSE))
    expect_equal(cfg$evals_per_cell, 0.5)
    expect_false(cfg$budget_floor)
    # The removed knobs resolve onto nothing, and the front door refuses them.
    expect_null(tulpa:::.ccd_placement_args(
        list(ccd_stencil_reuse = TRUE))$stencil_reuse)
    expect_error(
        tulpa_check_control(list(ccd_stencil_reuse = TRUE),
                            tulpa:::.CONTROL_KEYS$nested_laplace_joint,
                            "tulpa_nested_laplace_joint"),
        "ccd_stencil_reuse")
    # Everything the caller did not name keeps the registry's value.
    expect_identical(cfg$max_rounds, tulpa:::.ccd_placement("max_rounds"))
    expect_error(tulpa:::.ccd_placement_args(list(ccd_budget = -1)),
                 "ccd_budget")
    expect_error(tulpa:::.ccd_placement_args(list(ccd_budget = c(1, 2))),
                 "ccd_budget")
    # Inf is the documented way to place without a ceiling.
    expect_identical(
        tulpa:::.ccd_placement_args(list(ccd_budget = Inf))$evals_per_cell, Inf)
})

test_that("placement_budget is a named decline reason", {
    expect_true("placement_budget" %in% tulpa:::.CCD_DECLINE_REASONS)
    # And it is the only one about whether a definable design is WORTH placing;
    # the others all say a design could not be defined.
    expect_true(all(c("axis_count", "unguessable_axis", "degenerate_axis",
                      "modefind_ridge", "modefind_boundary",
                      "modefind_degenerate", "modefind_failed",
                      "hessian_singular", "hessian_not_pd",
                      "copy_atom_components", "copy_atom_mass",
                      "placement_budget") %in%
                    tulpa:::.CCD_DECLINE_REASONS))
})


# --- 2. the running meter ----------------------------------------------------

test_that("the meter charges before the batch and refuses one that would cross", {
    m <- tulpa:::.ccd_meter_new(10, list(on = FALSE, file = ""))
    tulpa:::.ccd_meter_spend(m, 4)
    tulpa:::.ccd_meter_spend(m, 6)
    expect_equal(m$evals, 10)
    # The batch that would cross is never paid for: the spend stays where it
    # was, so a declined placement reports what it actually cost.
    expect_error(tulpa:::.ccd_meter_spend(m, 1), class = "tulpa_ccd_budget")
    expect_equal(m$evals, 10)
    # No meter and no ceiling are both no-ops.
    expect_silent(tulpa:::.ccd_meter_spend(NULL, 1e6))
    inf <- tulpa:::.ccd_meter_new(Inf, list(on = FALSE, file = ""))
    tulpa:::.ccd_meter_spend(inf, 1e6)
    expect_equal(inf$evals, 1e6)
})

test_that("a budget signal is re-raised through the stencil error handlers", {
    # The mode-find absorbs a failed stencil, which must not swallow the budget:
    # the signal has to unwind to .joint_ccd_grid() so the fit declines under
    # its own reason instead of as a mode-find failure.
    bud <- structure(class = c("tulpa_ccd_budget", "error", "condition"),
                     list(message = "ceiling", call = NULL))
    expect_error(tulpa:::.ccd_rethrow_budget(bud), class = "tulpa_ccd_budget")
    expect_null(tulpa:::.ccd_rethrow_budget(simpleError("a numerical failure")))
})


# --- 3. the reported cost matches the stencil arithmetic ---------------------

test_that("a placement reports the inner solves it spent, to the eval", {
    fx  <- .ccd_bud_fixture()
    rec <- .ccd_bud_rec()
    # The fixture's objective is quadratic in the coordinate the design is laid
    # on, which is what makes the round count -- and so the eval count -- exact.
    expect_identical(unname(fx$tags), c("log", "logit01", "log", "log"))
    res <- tulpa:::.joint_ccd_grid(
        fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
        .ccd_bud_eval(fx$tags, fx$mode, fx$prec, rec))

    # The design was placed: the default ceiling did not fire.
    expect_null(res$declined)
    expect_true(is.matrix(res$grid))
    expect_identical(nrow(res$grid), ccd_grid(4L)$n_points)

    # Two Newton rounds on an exact quadratic: the first steps to the mode, the
    # second finds a zero gradient there. So the placement is the opening centre
    # evaluation, two rounds of (stencil + line search) and the closing stencil
    # that supplies the design's curvature.
    expect_identical(res$modefind_rounds, 2L)
    expect_equal(res$modefind_evals,
                 1 + 2 * (.CCD_BUD_STENCIL + .CCD_BUD_HALVE) + .CCD_BUD_STENCIL)
    expect_equal(res$modefind_evals, 114)
    # The counter is the number of rows the objective was actually handed, not
    # a projection of it.
    expect_equal(res$modefind_evals, rec$rows)
    # Reported against the ceiling it was spent under, so the two are readable
    # together, and with a wall-clock reading of the phase.
    expect_equal(res$modefind_budget,
                 tulpa:::.ccd_budget(81, 25, 4L, .ccd_bud_cfg()))
    expect_true(is.numeric(res$modefind_seconds) &&
                res$modefind_seconds >= 0)
    # Every stencil went out as ONE batched call, which is what lets the probes
    # fan out across the outer-grid threads.
    expect_true(.CCD_BUD_STENCIL %in% rec$sizes)
})


# --- 4. the ceiling declines, up front and mid-placement ---------------------

test_that("a hopeless ceiling declines before a single inner solve is paid", {
    fx  <- .ccd_bud_fixture()
    rec <- .ccd_bud_rec()
    # 81 tensor cells against a 25-node design leaves 56 solves at the
    # cost-neutral multiple, and the cheapest placement that could still produce
    # a design over four axes needs 74. Nothing to buy, so nothing is spent.
    cfg <- .ccd_bud_cfg(budget_floor = FALSE)
    expect_gt(tulpa:::.ccd_min_evals(4L, cfg), tulpa:::.ccd_budget(81, 25, 4L, cfg))
    res <- tulpa:::.joint_ccd_grid(
        fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
        .ccd_bud_eval(fx$tags, fx$mode, fx$prec, rec), cfg = cfg)
    expect_identical(res$declined, "placement_budget")
    expect_identical(rec$calls, 0L)
    expect_equal(res$modefind_evals, 0)
    expect_equal(res$modefind_budget, 56)
})

test_that("a ceiling reached mid-placement declines with the spend recorded", {
    fx  <- .ccd_bud_fixture()
    rec <- .ccd_bud_rec()
    # A ceiling of 112: above the 74 the up-front projection tests, below the
    # 114 this placement takes. The abort lands on the closing stencil, whose
    # 33 rows would cross, so the spend stops at the two completed rounds.
    cfg <- .ccd_bud_cfg(budget_floor = FALSE, evals_per_cell = 2)
    res <- tulpa:::.joint_ccd_grid(
        fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
        .ccd_bud_eval(fx$tags, fx$mode, fx$prec, rec), cfg = cfg)
    expect_identical(res$declined, "placement_budget")
    expect_null(res$grid)
    expect_equal(res$modefind_budget, 112)
    expect_equal(res$modefind_evals,
                 1 + 2 * (.CCD_BUD_STENCIL + .CCD_BUD_HALVE))
    expect_equal(res$modefind_evals, 81)
    # Never more than the ceiling: the crossing batch is refused, not paid.
    expect_lte(res$modefind_evals, res$modefind_budget)
    expect_equal(res$modefind_evals, rec$rows)
})

test_that("the ceiling is inert where it does not fire", {
    fx <- .ccd_bud_fixture()
    r1 <- .ccd_bud_rec(); r2 <- .ccd_bud_rec()
    # The shipped default against no ceiling at all: same design, same centre,
    # same scale, same cost. A fit inside its budget is the fit it was before.
    a <- tulpa:::.joint_ccd_grid(
        fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
        .ccd_bud_eval(fx$tags, fx$mode, fx$prec, r1))
    b <- tulpa:::.joint_ccd_grid(
        fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
        .ccd_bud_eval(fx$tags, fx$mode, fx$prec, r2),
        cfg = .ccd_bud_cfg(evals_per_cell = Inf))
    expect_equal(a$grid, b$grid)
    expect_equal(a$u_hat, b$u_hat)
    expect_equal(a$L_scale, b$L_scale)
    expect_equal(a$dnode, b$dnode)
    expect_equal(r1$rows, r2$rows)
    expect_identical(b$modefind_budget, Inf)
})

test_that("metering does not move the mode-find's numbers", {
    fx    <- .ccd_bud_fixture()
    eval1 <- function(U) {
        U <- matrix(U, ncol = 4L)
        apply(U, 1L, function(u) -0.5 * sum(fx$prec * (u - fx$mode)^2))
    }
    args <- list(u0 = rep(0, 4L), eval1 = eval1, lower = rep(-5, 4L),
                 upper = rep(5, 4L), h = rep(0.1, 4L), trust = rep(2, 4L))
    bare  <- do.call(tulpa:::.joint_ccd_modefind, args)
    metered <- do.call(tulpa:::.joint_ccd_modefind,
                       c(args, list(meter = tulpa:::.ccd_meter_new(
                           Inf, list(on = FALSE, file = "")))))
    expect_identical(bare$status, "ok")
    expect_identical(bare$par, metered$par)
    expect_identical(bare$hess, metered$hess)
    expect_identical(bare$value, metered$value)
    expect_identical(bare$converged, metered$converged)
})


# --- 5. the heartbeat (gcol33/tulpa#652) -------------------------------------

.ccd_bud_quad_eval <- function(m, prec) {
    function(U) {
        U <- matrix(U, ncol = length(m))
        apply(U, 1L, function(u) -0.5 * sum(prec * (u - m)^2))
    }
}

.ccd_bud_run_modefind <- function(meter, m = c(0.4, -0.3, 0.2),
                                  prec = c(2, 3, 1.5)) {
    tulpa:::.joint_ccd_modefind(
        u0 = rep(0, length(m)), eval1 = .ccd_bud_quad_eval(m, prec),
        lower = rep(-5, length(m)), upper = rep(5, length(m)),
        h = rep(0.1, length(m)), trust = rep(2, length(m)), meter = meter)
}

test_that("the mode-find emits one heartbeat line per round", {
    meter <- tulpa:::.ccd_meter_new(500, list(on = TRUE, file = ""))
    out <- capture.output(mf <- .ccd_bud_run_modefind(meter))
    expect_identical(mf$status, "ok")
    rounds <- grep("^\\[ccd mode-find\\] round [0-9]+/", out)
    expect_identical(length(rounds), as.integer(meter$rounds))
    expect_gte(meter$rounds, 1L)
    # The line carries what distinguishes converging from crawling from stuck:
    # which round of how many, the spend against the ceiling, the step size, the
    # log-posterior, and enough timing to project the phase.
    first <- out[rounds[1L]]
    expect_match(first, "round 1/")
    expect_match(first, "[0-9]+/500 evals")
    expect_match(first, "\\|step\\|")
    expect_match(first, "logpost")
    expect_match(first, "elapsed")
    expect_match(first, "/eval")
    # And the phase says when it is over, so a silent gap afterwards is the
    # grid loop rather than the mode-find.
    expect_true(any(grepl("^\\[ccd mode-find\\] mode-find done", out)))
})

test_that("the heartbeat is silent when progress is off", {
    meter <- tulpa:::.ccd_meter_new(500, list(on = FALSE, file = ""))
    expect_output(.ccd_bud_run_modefind(meter), NA)
    # The rounds are still counted: the count is the report, the line is the
    # narration, and switching the narration off does not stop the reporting.
    expect_gte(meter$rounds, 1L)
})

test_that("the step calibration and the seed pass narrate their own probes", {
    # Both run probes before the rescue mode-find starts, and both were silent;
    # #652 asks for them on the same heartbeat. Each is handed an EVALUATOR, so
    # driving them with a bare objective tests what the `meter` argument buys
    # here: the heartbeat. The spend those probes add is charged one level down,
    # at the single point every probe passes through, and is asserted on the
    # production path in the test below.
    m     <- c(0.4, -0.3, 0.2)
    prec  <- c(2, 3, 1.5)
    meter <- tulpa:::.ccd_meter_new(2000, list(on = TRUE, file = ""))
    out <- capture.output(
        h <- tulpa:::.joint_ccd_calibrate_step(
            rep(0, 3L), .ccd_bud_quad_eval(m, prec), span = rep(1, 3L),
            meter = meter))
    expect_length(h, 3L)
    expect_true(any(grepl("^\\[ccd mode-find\\] step calibration [0-9]+/",
                          out)))

    u_vals <- list(c(-1, 0, 1), c(-1, 0, 1), c(-1, 0, 1))
    meter2 <- tulpa:::.ccd_meter_new(2000, list(on = TRUE, file = ""))
    out2 <- capture.output(
        u <- tulpa:::.joint_ccd_grid_seed(rep(0, 3L), u_vals,
                                          .ccd_bud_quad_eval(m, prec),
                                          meter = meter2))
    expect_length(u, 3L)
    expect_true(any(grepl("^\\[ccd mode-find\\] grid seed \\(27 points\\)",
                          out2)))
})

test_that("the escalation's probes are charged to the same placement budget", {
    # The first attempt rails to the box edge, so the placement escalates: joint
    # grid seed, step calibration, then the rescue mode-find. Every probe of
    # those passes is a full inner solve on a real fit, so all of them have to
    # be in the placement's spend -- a budget counting only the mode-find rounds
    # would under-count the escalation, which is exactly where gcol33/tulpa#653's
    # unbounded cost lives.
    fx  <- .ccd_bud_fixture()
    rec <- .ccd_bud_rec()
    op  <- options(tulpa.nl_progress = list(progress = TRUE,
                                            progress_file = ""))
    on.exit(options(op), add = TRUE)
    out <- capture.output(
        res <- tulpa:::.joint_ccd_grid(
            fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
            # A mode far outside the axis box: the mode-find runs to the edge
            # and is declined as boundary-supported, from either seed.
            .ccd_bud_eval(fx$tags, rep(40, 4L), fx$prec, rec),
            cfg = .ccd_bud_cfg(evals_per_cell = Inf)))
    # The escalation ran, and both of its passes reached the one heartbeat.
    expect_true(any(grepl("^\\[ccd mode-find\\] grid seed \\(81 points\\)", out)))
    expect_true(any(grepl("^\\[ccd mode-find\\] step calibration [0-9]+/", out)))
    # It declined for what it is -- a boundary-supported hyperparameter -- and
    # not for cost, so nothing here is an artefact of the ceiling.
    expect_identical(res$declined, "modefind_boundary")
    # The seed pass evaluates the whole 3^4 latent grid in one batch, and that
    # batch is in the charged total.
    expect_true(81 %in% rec$sizes)
    # The spend is every row the objective was handed -- seed and calibration
    # included, not the mode-find rounds alone.
    expect_equal(res$modefind_evals, rec$rows)
    expect_gt(res$modefind_evals,
              tulpa:::.ccd_first_evals(4L, .ccd_bud_cfg()))
})

test_that("a configured heartbeat file carries the grid reporter's wire format", {
    f <- tempfile(fileext = ".hb")
    on.exit(unlink(f), add = TRUE)
    meter <- tulpa:::.ccd_meter_new(500, list(on = TRUE, file = f))
    invisible(capture.output(.ccd_bud_run_modefind(meter)))
    expect_true(file.exists(f))
    # "<done> <total> <elapsed_s> <eta_s>", the four numbers the outer-grid
    # heartbeat file already holds, so a detached reader parsing that file sees
    # liveness through the placement phase.
    txt <- readLines(f, warn = FALSE)
    expect_length(txt, 1L)
    nums <- suppressWarnings(as.numeric(strsplit(trimws(txt), "[[:space:]]+")[[1L]]))
    expect_length(nums, 4L)
    expect_false(anyNA(nums))
    expect_equal(nums[2L], 500)
    expect_equal(nums[1L], meter$evals)
})

test_that("the heartbeat rides the fit's own progress record", {
    # #652 asks for the SAME switch the rest of the path uses, not a second one:
    # the joint front door writes `tulpa.nl_progress` once per fit and the
    # placement phase reads it there.
    op <- options(tulpa.nl_progress = list(progress = TRUE,
                                           progress_every = 0L,
                                           progress_throttle = 2,
                                           progress_file = "hb.txt"))
    on.exit(options(op), add = TRUE)
    p <- tulpa:::.ccd_progress_opts()
    expect_true(p$on)
    expect_identical(p$file, "hb.txt")
    options(tulpa.nl_progress = list(progress = FALSE, progress_file = ""))
    expect_false(tulpa:::.ccd_progress_opts()$on)
    # Outside a fit there is no record, and the phase is silent.
    options(tulpa.nl_progress = NULL)
    expect_false(tulpa:::.ccd_progress_opts()$on)
})

test_that("a placement outside a fit narrates nothing", {
    fx  <- .ccd_bud_fixture()
    rec <- .ccd_bud_rec()
    op  <- options(tulpa.nl_progress = NULL)
    on.exit(options(op), add = TRUE)
    expect_output(
        tulpa:::.joint_ccd_grid(
            fx$axis_names, fx$axis_offsets, fx$prepared, fx$axis_values,
            .ccd_bud_eval(fx$tags, fx$mode, fx$prec, rec)),
        NA)
})


# --- 6. the finite-difference stencil ---------------------------------------
#
# 0.2.14 shipped a curvature-reuse path here -- an axial-only stencil plus an SR1
# secant update of the off-diagonal block -- opt-in and unmeasured. It was
# measured over 16 paired fits and removed (gcol33/tulpa#662): it spent no fewer
# inner solves, because the secant model lengthened the walk by roughly the
# per-round saving, and it centred the design somewhere worse. What survives is
# the assertion the reuse test carried about the FULL stencil, which is now the
# only one the mode-find takes.

test_that("the full stencil reads an exact quadratic's own gradient and Hessian", {
    m    <- c(0.4, -0.3, 0.2)
    prec <- c(2, 3, 1.5)
    ev   <- .ccd_bud_quad_eval(m, prec)
    u    <- c(0.1, 0.05, -0.2)
    h    <- rep(0.1, 3L)
    fullst <- tulpa:::.joint_ccd_fd_stencil(u, ev, h)
    expect_equal(fullst$hess, -diag(prec), tolerance = 1e-6)
    expect_equal(fullst$grad, -prec * (u - m), tolerance = 1e-6)
    # One batched call, `1 + 2d + 4 * C(d, 2)` points: the centre, the axials
    # and the mixed corners, so every inner solve in a round fans out across the
    # outer-grid threads.
    sizes <- integer(0)
    ev_rec <- function(U) {
        sizes <<- c(sizes, nrow(matrix(U, ncol = length(m))))
        ev(U)
    }
    tulpa:::.joint_ccd_fd_stencil(u, ev_rec, h)
    expect_identical(sizes, as.integer(1L + 2L * 3L + 4L * choose(3L, 2L)))
})

# --- 7. the whole transport, on a fit ----------------------------------------

.ccd_bud_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

.ccd_bud_sim <- function(seed, N, n_s) {
    set.seed(seed)
    spatial_idx <- sample.int(n_s, N, replace = TRUE)
    phi   <- as.numeric(scale(cumsum(rnorm(n_s, sd = 0.6))))
    theta <- rnorm(n_s)
    w_s   <- 0.9 * (sqrt(0.7) * phi + sqrt(0.3) * theta)
    x     <- rnorm(N); Xocc <- cbind(1, x)
    eta_occ <- as.numeric(Xocc %*% c(-0.3, 0.5)) + w_s[spatial_idx]
    occur   <- rbinom(N, 1, plogis(eta_occ))
    is_pos  <- occur == 1L
    Xpos    <- Xocc[is_pos, , drop = FALSE]; spi_pos <- spatial_idx[is_pos]
    eta_pos <- as.numeric(Xpos %*% c(0.2, -0.4)) + w_s[spi_pos]
    y_pos   <- rnorm(sum(is_pos), eta_pos, 0.5)
    list(adj = .ccd_bud_chain_adj(n_s),
         responses = list(
             occ = list(y = as.numeric(occur), n_trials = rep(1L, N), X = Xocc,
                        spatial_idx = spatial_idx, re_idx = rep(0, N),
                        n_re_groups = 0L, sigma_re = 1.0,
                        family = "binomial", phi = 1.0),
             pos = list(y = y_pos, n_trials = rep(1L, length(y_pos)), X = Xpos,
                        spatial_idx = spi_pos, re_idx = rep(0, length(y_pos)),
                        n_re_groups = 0L, sigma_re = 1.0,
                        family = "gaussian", phi = 0.25)))
}

test_that("a fit reports the placement it declined on cost", {
    skip_on_cran()
    sim <- .ccd_bud_sim(2024L, N = 600L, n_s = 30L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- list(type = "bym2", spatial_idx = sp,
                n_spatial_units = sim$adj$n_spatial_units,
                adj_row_ptr = sim$adj$adj_row_ptr,
                adj_col_idx = sim$adj$adj_col_idx,
                n_neighbors = sim$adj$n_neighbors, scale_factor = 1.0,
                sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9))
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = list(integration = "ccd", diagnose_k = FALSE,
                       progress = FALSE,
                       var_of_means_consistency = FALSE,
                       ccd_budget = 0, ccd_budget_floor = FALSE))
    # The whole transport: control knob -> scoped option -> the placement's cfg
    # -> the up-front decline -> the record -> the field on the fit.
    expect_identical(fit$integration_requested, "ccd")
    expect_false(identical(fit$integration, "ccd"))
    expect_identical(fit$integration_declined, "placement_budget")
    expect_equal(fit$ccd_modefind_evals, 0)
    expect_equal(fit$ccd_modefind_budget, 0)
    # And the tensor grid it fell back to actually integrated the model.
    expect_true(is.matrix(fit$theta_grid))
    expect_true(any(is.finite(fit$weights)))
})

test_that("a fit inside its budget reports what the placement cost", {
    skip_on_cran()
    sim <- .ccd_bud_sim(2024L, N = 600L, n_s = 30L)
    sp  <- list(sim$responses$occ$spatial_idx, sim$responses$pos$spatial_idx)
    blk <- list(type = "bym2", spatial_idx = sp,
                n_spatial_units = sim$adj$n_spatial_units,
                adj_row_ptr = sim$adj$adj_row_ptr,
                adj_col_idx = sim$adj$adj_col_idx,
                n_neighbors = sim$adj$n_neighbors, scale_factor = 1.0,
                sigma_grid = c(0.3, 0.6, 1.0), rho_grid = c(0.3, 0.7, 0.9))
    ctl <- list(integration = "ccd", diagnose_k = FALSE, progress = FALSE,
                var_of_means_consistency = FALSE)
    fit <- tulpa_nested_laplace_joint(
        sim$responses, list(blk),
        copy = list(arm = "pos", block = 1L, alpha_grid = c(0.3, 0.7, 1.2)),
        control = ctl)
    # Whatever the placement decided, the cost of deciding it is on the fit --
    # which is what lets the CCD be judged on a real model rather than argued
    # from the round caps.
    expect_true(is.numeric(fit$ccd_modefind_evals))
    expect_gt(fit$ccd_modefind_evals, 0)
    expect_lte(fit$ccd_modefind_evals, fit$ccd_modefind_budget)
    expect_true(is.numeric(fit$ccd_modefind_seconds))
    expect_gte(fit$ccd_modefind_rounds, 1L)
})
