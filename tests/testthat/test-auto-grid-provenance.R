
test_that("both registry tau defaults are recognised, an unrelated tau grid is not", {
    for (d in list(tulpa:::.nl_grid_axis("gmrf_tau"),
                   exp(seq(log(0.3), log(30), length.out = 5)))) {
        expect_false(tulpa:::.nl_axis_is_pinned(
            list(type = "icar", tau_grid = d), "tau_grid"))
    }
    expect_true(tulpa:::.nl_axis_is_pinned(
        list(type = "icar", tau_grid = c(0.5, 1, 2)), "tau_grid"))
    # A field with no engine default of its own is always a pin when present.
    expect_true(tulpa:::.nl_axis_is_pinned(
        list(type = "bym2", rho_grid = c(0.3, 0.6)), "rho_grid"))
})

test_that("axis aliases cover the bare, block-prefixed and coerced spellings", {
    expect_identical(tulpa:::.nl_axis_alias("sigma"), c("sigma", "theta"))
    # One block: a bare or coerced name can only be that block's axis.
    expect_identical(tulpa:::.nl_axis_alias("sigma", 1L, n_blocks = 1L),
                     c("b1.sigma", "sigma", "theta"))
    # Several blocks: every axis is prefixed, so an unprefixed match would be
    # attributing another block's axis.
    expect_identical(tulpa:::.nl_axis_alias("sigma", 2L, n_blocks = 3L),
                     "b2.sigma")
})

test_that("an edge hit is found under any of the three spellings", {
    hit <- function(edge, ..., blocks = list()) {
        tulpa:::.nl_edge_axis_hit(
            list(pareto_k_grid_edge_axes = edge, blocks = blocks), ...)
    }
    expect_true(hit("sigma", "sigma"))
    expect_true(hit("theta", "sigma"))                    # coerced vector grid
    expect_true(hit("b2.sigma", "sigma", 2L,
                    blocks = list(1, 2, 3)))              # multi-block
    expect_false(hit("b2.sigma", "sigma", 1L, blocks = list(1, 2, 3)))
    expect_false(hit(c("rho", "alpha"), "sigma"))
    expect_false(hit(character(0), "sigma"))
})

test_that("a decline reason is stamped only when the recenter did not run", {
    declined <- tulpa:::.nl_decline_recenter(
        list(outer_grid_placement = "fixed"), "axis_pinned")
    expect_identical(declined$outer_grid_recenter_declined, "axis_pinned")
    ran <- tulpa:::.nl_decline_recenter(
        list(outer_grid_placement = "auto_recentered"), "axis_pinned")
    expect_null(ran$outer_grid_recenter_declined)
})

# The joint placement pass over one family, driven with a
# `refit(prior, prior_sigma)` callback.
.place_field <- function(res, prior, prior_sigma, refit, auto = logical(0),
                         enabled = TRUE, max_attempts = 2L) {
    tulpa:::.joint_place_axes(
        res, list(prior = prior, prior_sigma = prior_sigma),
        refit = function(st, from) refit(st$prior, st$prior_sigma),
        families = tulpa:::.joint_placement_families(auto = auto)["field"],
        enabled = enabled, max_attempts = max_attempts)
}
# A one-block icar fit on the precision axis `b1.tau`, its marginal a Gaussian
# in log tau centred at `mode` with SD 0.5, laid on `nodes`. The default axis
# spaces its nodes 0.58 apart in log tau, so a centre inside it is resolved and
# one past its top node (30) rails there.
.tau_fit <- function(mode, nodes = tulpa:::.nl_grid_axis("gmrf_tau"),
                     cov = 0.25) {
    list(theta_grid = cbind(b1.tau = nodes),
         log_marginal = -0.5 * ((log(nodes) - log(mode)) / 0.5)^2,
         outer_mode_u = log(mode), outer_mode_cov_u = matrix(cov, 1, 1),
         outer_mode_axis_tags = "log", outer_mode_axis_names = "b1.tau",
         outer_grid_placement = "fixed", blocks = list(list(type = "icar")))
}
.place_copy_sd <- function(res, prior, copy, cp, prior_sigma, refit,
                           auto = list(), enabled = TRUE, max_attempts = 2L) {
    tulpa:::.joint_place_axes(
        res, list(prior = prior, prior_sigma = prior_sigma, copy = copy,
                  cp = cp),
        refit = function(st, from) refit(st$prior, st$prior_sigma),
        families = tulpa:::.joint_placement_families(auto = auto)["copy_sigma"],
        enabled = enabled, max_attempts = max_attempts)
}

test_that("auto_grid() marks a prior specification as well as a grid or a knob", {
    m <- auto_grid(list("pc.prec", c(U = 3, alpha = 0.01)))
    expect_true(is_auto_grid(m))
    expect_true(is.list(m))
    expect_identical(m[[1L]], "pc.prec")
    # A scalar knob.
    k <- auto_grid(4L)
    expect_true(is_auto_grid(k))
    expect_identical(as.numeric(k), 4)
    expect_error(auto_grid(list()), "non-empty")
})

skip_on_cran_budget()
# Outer-axis provenance for the auto-recenter passes (gcol33/tulpa#293).
#
# The rescues in R/nested_laplace_auto_grid.R must recentre a DEFAULT axis and
# leave a USER PIN alone. Field presence cannot answer that question: a wrapper
# package that computes the engine's own default itself writes a non-NULL
# `prior$sigma_grid` on a fit where the user named no grid, which the original
# `!is.null()` guard read as an override -- every `occu_cover()` fit's
# auto-recenter was inert for exactly that reason. These are structural tests of
# the predicate, the marker, and the axis-name resolution: no fit runs, so they
# hold at every test tier.

test_that("auto_grid marks a grid and is_auto_grid reads the mark back", {
    skip_on_cran_budget()
    g <- auto_grid(c(0.5, 1, 2))
    expect_true(is_auto_grid(g))
    expect_identical(as.numeric(g), c(0.5, 1, 2))
    expect_false(is_auto_grid(c(0.5, 1, 2)))
    expect_false(is_auto_grid(NULL))
    expect_false(is_auto_grid("sigma"))
    # as.numeric() strips it -- the documented "mark last" contract.
    expect_false(is_auto_grid(as.numeric(g)))
    expect_error(auto_grid(numeric(0)), "non-empty")
    expect_error(auto_grid(c(1, NA)), "non-empty|NA")
})

# gcol33/tulpa#360: two families store their axis as a matrix of pre-paired
# coordinates, so a coercion that flattens destroys the axis it is marking and
# leaves those families with no way to declare a default at all.
test_that("a matrix-valued axis keeps its shape through the mark", {
    skip_on_cran_budget()
    m <- matrix(as.numeric(1:6), nrow = 3, ncol = 2,
                dimnames = list(NULL, c("d1", "off")))
    g <- auto_grid(m)
    expect_true(is_auto_grid(g))
    expect_true(is.matrix(g))
    expect_identical(dim(g), c(3L, 2L))
    expect_identical(dimnames(g), dimnames(m))
    expect_identical(as.numeric(g), as.numeric(m))
    expect_identical(storage.mode(g), "double")
    # An integer matrix is coerced in place, not flattened.
    gi <- auto_grid(matrix(1:6, nrow = 2))
    expect_identical(dim(gi), c(2L, 3L))
    expect_identical(storage.mode(gi), "double")
})

test_that("a marked matrix axis survives every consumer of the mark", {
    skip_on_cran_budget()
    p <- 2L
    lc <- tulpa:::.mcar_default_logchol_grid(p)
    expect_true(is.matrix(lc))
    blk <- list(type = "miid", n_units = 5L, n_fields = p,
                logchol_grid = auto_grid(lc))

    # (a) the provenance record names the field and hands back a bare matrix.
    prov <- tulpa:::.nl_grid_provenance(blk)
    expect_identical(prov$auto, c(logchol_grid = TRUE))
    expect_true(is.matrix(prov$prior$logchol_grid))
    expect_identical(dim(prov$prior$logchol_grid), dim(lc))
    expect_false(is_auto_grid(prov$prior$logchol_grid))

    # (b) the provenance predicate reads it as a DEFAULT, which is the whole
    # point: unmarked, the same matrix reads as a pin.
    expect_false(tulpa:::.nl_axis_is_pinned(blk, "logchol_grid",
                                            prov$auto, type = "miid"))
    expect_true(tulpa:::.nl_axis_is_pinned(list(type = "miid",
                                                logchol_grid = lc),
                                           "logchol_grid", character(0),
                                           type = "miid"))

    # (c) the #352 consumption check sees a numeric axis the path reads, so it
    # neither refuses nor records it.
    expect_null(tulpa:::.nl_check_axis_fields(blk, "registry",
                                              auto = prov$auto))

    # (d) the axis still drives the grid builder, unchanged by the round trip.
    ref <- tulpa:::.nl_block_axis_grid(list(type = "miid", n_units = 5L,
                                            n_fields = p, logchol_grid = lc))
    got <- tulpa:::.nl_block_axis_grid(prov$prior)
    expect_equal(got$grid, ref$grid)
    expect_identical(got$names, ref$names)
})

test_that("a pinned axis is pinned and every flavour of default is not", {
    skip_on_cran_budget()
    base <- list(type = "icar")
    # No axis at all.
    expect_false(tulpa:::.nl_axis_is_pinned(base, "sigma_grid"))
    # A genuine pin.
    pinned <- c(list(sigma_grid = c(0.1, 0.5, 1, 2, 3)), base)
    expect_true(tulpa:::.nl_axis_is_pinned(pinned, "sigma_grid"))
    # The same pin, declared a default by the caller.
    marked <- pinned
    marked$sigma_grid <- auto_grid(pinned$sigma_grid)
    expect_false(tulpa:::.nl_axis_is_pinned(marked, "sigma_grid"))
    # ... or declared through the stripped provenance record.
    expect_false(tulpa:::.nl_axis_is_pinned(pinned, "sigma_grid", "sigma_grid"))
    # The engine's OWN default axis, handed back in unmarked (the #293 case).
    eng <- pinned
    eng$sigma_grid <- tulpa:::.nl_grid_axis("field_sd")
    expect_false(tulpa:::.nl_axis_is_pinned(eng, "sigma_grid"))
    # ... and the same node set crossed against a second axis (bym2's paired
    # (sigma, rho) grid) is still that default node set.
    gr <- expand.grid(sigma = tulpa:::.nl_grid_axis("field_sd"),
                      rho = tulpa:::.nl_grid_axis("bym2_rho"))
    crossed <- pinned
    crossed$sigma_grid <- gr$sigma
    expect_false(tulpa:::.nl_axis_is_pinned(crossed, "sigma_grid"))
    # A default axis with ONE node moved is a pin again.
    moved <- eng
    moved$sigma_grid[5L] <- 6
    expect_true(tulpa:::.nl_axis_is_pinned(moved, "sigma_grid"))
})

test_that("provenance strips markers and records them per block", {
    skip_on_cran_budget()
    single <- list(type = "icar", sigma_grid = auto_grid(c(1, 2, 3)),
                   rho_grid = c(0.2, 0.9))
    pv <- tulpa:::.nl_grid_provenance(single)
    expect_identical(pv$auto, c(sigma_grid = TRUE))
    expect_false(is_auto_grid(pv$prior$sigma_grid))
    expect_identical(pv$prior$sigma_grid, c(1, 2, 3))
    expect_identical(tulpa:::.nl_auto_fields_at(pv$auto), c(sigma_grid = TRUE))

    multi <- list(list(type = "icar", sigma_grid = auto_grid(c(1, 2))),
                  list(type = "iid",  sigma_grid = c(0.3, 0.6)))
    pvm <- tulpa:::.nl_grid_provenance(multi)
    expect_identical(pvm$auto[[1L]], c(sigma_grid = TRUE))
    expect_identical(pvm$auto[[2L]], logical(0))
    expect_false(is_auto_grid(pvm$prior[[1L]]$sigma_grid))
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 1L),
                     c(sigma_grid = TRUE))
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 2L), logical(0))
    # Out-of-range / absent records read as "nothing declared", never an error.
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 9L), logical(0))
    expect_identical(tulpa:::.nl_auto_fields_at(NULL, 1L), logical(0))
    # A bare character vector of declared fields is the placeable default.
    expect_identical(tulpa:::.nl_auto_fields_at("sigma_grid"),
                     c(sigma_grid = TRUE))
})

# gcol33/tulpaObs#361: PROVENANCE (whose nodes these are) and PLACEMENT POLICY
# (may the pass move them) are different questions, and a package with its own
# measured answer to the second had only the user's pin to express it with --
# which is then what the fit reports back to a user who pinned nothing.
test_that("a declared default can ask to be integrated as written", {
    skip_on_cran_budget()
    held  <- auto_grid(c(0.5, 1, 2), place = FALSE)
    place <- auto_grid(c(0.5, 1, 2))
    expect_true(is_auto_grid(held))
    expect_false(auto_grid_place(held))
    expect_true(auto_grid_place(place))
    # An unmarked value is held because it reads as a pin, not because it asked.
    expect_true(auto_grid_place(c(0.5, 1, 2)))
    expect_identical(as.numeric(held), c(0.5, 1, 2))
    expect_error(auto_grid(c(1, 2), place = NA), "TRUE or FALSE")
    expect_error(auto_grid(c(1, 2), place = "no"), "TRUE or FALSE")

    # Both halves survive the record, and neither attribute survives the strip.
    blk <- list(type = "icar", sigma_grid = held)
    pv  <- tulpa:::.nl_grid_provenance(blk)
    expect_identical(pv$auto, c(sigma_grid = FALSE))
    expect_null(attributes(pv$prior$sigma_grid))

    # Three answers, and the engine ACTS the same on the last two.
    hold <- function(b, auto = logical(0))
        tulpa:::.nl_axis_hold(b, "sigma_grid", auto, type = ".joint_areal")
    expect_null(hold(list(type = "icar", sigma_grid = place)))
    expect_identical(hold(pv$prior, pv$auto), "default_axis_pinned")
    expect_identical(hold(list(type = "icar", sigma_grid = c(0.3, 0.9, 2))),
                     "axis_pinned")
    expect_true(tulpa:::.nl_axis_is_pinned(pv$prior, "sigma_grid", pv$auto,
                                           type = ".joint_areal"))
    # Read off the value as well as off the record, since a call site may hold
    # the axis before the front door strips it.
    expect_identical(hold(blk), "default_axis_pinned")
})

test_that("axis index resolves by name, then by being the only log axis", {
    skip_on_cran_budget()
    expect_identical(tulpa:::.nl_axis_index(c("sigma", "rho"), c("sigma", "theta")), 1L)
    expect_identical(tulpa:::.nl_axis_index(c("rho", "sigma"), c("sigma", "theta")), 2L)
    # icar's theta_names says "tau" while the coerced grid says "theta": the
    # lone log axis IS the family's scale axis whatever it is named.
    expect_identical(tulpa:::.nl_axis_index("tau", c("sigma", "theta"), "log"), 1L)
    # A lone BOUNDED axis is declined rather than recentred on a guessed support.
    expect_true(is.na(tulpa:::.nl_axis_index("rho", c("sigma", "theta"), "logit01")))
    expect_true(is.na(tulpa:::.nl_axis_index(c("rho", "alpha"),
                                             c("sigma", "theta"),
                                             c("logit01", "identity"))))
    expect_true(is.na(tulpa:::.nl_axis_index(NULL, "sigma")))
})

test_that("the field placement honours a pin and passes a default", {
    skip_on_cran_budget()
    # No fit: a stub `res` carrying the grid, marginal and outer mode the
    # placement reads, railed on the top node of the default precision axis,
    # and a refit that records whether it was called.
    railed <- .tau_fit(100)
    calls <- new.env(parent = emptyenv())
    calls$n <- 0L
    refit <- function(prior_i, prior_sigma_i) {
        calls$n <- calls$n + 1L
        calls$grid <- as.numeric(prior_i[[1L]]$tau_grid)
        # The refit's grid holds the mode it was laid at, resolved, so the loop
        # stops after one attempt.
        .tau_fit(100, nodes = calls$grid)
    }
    icar <- list(type = "icar")

    # 1. Axis absent -> placed past the rail.
    out <- .place_field(railed, list(icar), NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)
    expect_gt(max(calls$grid), 30)
    expect_true(is_auto_grid(out$st$prior[[1L]]$tau_grid))
    expect_null(out$res$outer_grid_recenter_declined)
    expect_identical(out$res$outer_grid_recenter_axes, "b1.tau")
    expect_null(out$res$outer_grid_axis_declined)

    # 2. Engine default handed in unmarked -> still placed (#293).
    calls$n <- 0L
    out <- .place_field(
        railed, list(c(icar, list(tau_grid = tulpa:::.nl_grid_axis("gmrf_tau")))),
        NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)

    # 3. A marked caller default -> placed.
    calls$n <- 0L
    out <- .place_field(
        railed, list(c(icar, list(tau_grid = auto_grid(c(0.5, 2, 8))))),
        NULL, refit, auto = list("tau_grid"))
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)

    # 4. A genuine pin -> untouched, with the reason recorded.
    calls$n <- 0L
    out <- .place_field(
        railed, list(c(icar, list(tau_grid = c(0.5, 1, 2, 4, 8)))), NULL, refit)
    expect_identical(out$res$outer_grid_placement, "fixed")
    expect_identical(out$res$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(out$res$outer_grid_axis_declined, c(b1.tau = "axis_pinned"))
    expect_identical(calls$n, 0L)

    # 5. A grid that resolves its posterior -> untouched, different reason.
    calls$n <- 0L
    out <- .place_field(.tau_fit(3), list(icar), NULL, refit)
    expect_identical(out$res$outer_grid_recenter_declined,
                     "grid_resolves_posterior")
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.tau = "grid_resolves_posterior"))
    expect_identical(calls$n, 0L)

    # 6. Railed, but no curvature to place on -> untouched, own reason.
    calls$n <- 0L
    blind <- utils::modifyList(railed, list(outer_mode_u = NULL,
                                            outer_mode_cov_u = NULL))
    out <- .place_field(blind, list(icar), NULL, refit)
    expect_identical(out$res$outer_grid_recenter_declined, "no_usable_curvature")
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.tau = "no_usable_curvature"))
    expect_identical(calls$n, 0L)

    out <- .place_field(railed, list(icar), NULL, refit, enabled = FALSE)
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.tau = "auto_recenter_disabled"))

    # 7. A block the registry binds no axis to offers the family no slot, so
    #    the pass stamps nothing at all.
    out <- .place_field(railed, list(list(type = "lf")), NULL, refit)
    expect_null(out$res$outer_grid_recenter_declined)
    expect_null(out$res$outer_grid_axis_declined)
    expect_identical(calls$n, 0L)
})

# gcol33/tulpa#297: the second attempt's regularizing PC prior was suppressed by
# ANY supplied prior_sigma, decided by presence -- so a wrapper stamping a
# prior_sigma of its own silently turned the escalation into a second geometry
# recenter while still reporting `attempts = 2`.
test_that("a door prior_sigma pins by provenance, not presence", {
    skip_on_cran_budget()
    pinned <- function(ps) tulpa:::.nl_prior_sigma_pinned_at(
        ps, list(list(type = "icar")), 1L)
    expect_false(pinned(NULL))
    # The engine's own default handed back in carries no information a pin adds.
    expect_false(pinned(tulpa:::.nl_recenter("sigma_pc_prior")))
    # A caller declaring its own spec a default.
    expect_false(pinned(auto_grid(list("pc.prec", c(U = 1, alpha = 0.01)))))
    # A different prior, undeclared: a real choice.
    expect_true(pinned(list("pc.prec", c(U = 1, alpha = 0.01))))
    expect_true(pinned(list("half_normal", 2)))
    # A per-block door spec pins only the block it names.
    per_block <- list(list(block = 2L, prior = list("pc.prec", c(1, 0.01))))
    two <- list(list(type = "icar"), list(type = "icar"))
    expect_false(tulpa:::.nl_prior_sigma_pinned_at(per_block, two, 1L))
    expect_true(tulpa:::.nl_prior_sigma_pinned_at(per_block, two, 2L))
})

test_that("the second recenter attempt engages the PC prior unless it was PINNED", {
    skip_on_cran_budget()
    # The escalation rides the copy block's field SD, the one family that
    # carries it (`escalate = TRUE` in `.joint_placement_families()`).
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "b1.sigma",
                     outer_mode_u = c(log(3), 0.4),
                     outer_mode_cov_u = diag(c(0.25, 0.09)),
                     outer_mode_axis_tags = c("log", "identity"),
                     outer_mode_axis_names = c("b1.sigma", "b1.alpha"),
                     outer_grid_placement = "fixed",
                     blocks = list(list(type = "icar")))
    cp <- list(has_copy = TRUE, copy_blocks_zero = 0L)
    calls <- new.env(parent = emptyenv())
    # Never leaves its rail, so the loop runs its full two attempts.
    refit <- function(prior_i, prior_sigma_i) {
        calls$last_prior_sigma <- prior_sigma_i
        stub_res
    }
    blocks <- list(list(type = "icar"))
    pc <- tulpa:::.nl_recenter("sigma_pc_prior")
    on_b1 <- list(list(block = 1L, prior = pc))

    # No prior at all -> attempt 2 adds the engine's PC prior on the copy block.
    out <- .place_copy_sd(stub_res, blocks, NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_recenter_attempts, 2L)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, on_b1)
    expect_null(out$res$outer_grid_prior_declined)

    # A wrapper's own default, declared -> the escalation still engages, and the
    # engine's prior replaces the declared default on the copy block.
    wrapper_default <- auto_grid(list("pc.prec", c(U = 1, alpha = 0.01)))
    out <- .place_copy_sd(stub_res, blocks, NULL, cp, wrapper_default, refit)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, on_b1)
    expect_null(out$res$outer_grid_prior_declined)

    # A deliberate, undeclared choice -> held, and the suppression is legible
    # rather than an `attempts = 2` that quietly did the same thing twice.
    chosen <- list("pc.prec", c(U = 1, alpha = 0.01))
    out <- .place_copy_sd(stub_res, blocks, NULL, cp, chosen, refit)
    expect_identical(out$res$outer_grid_recenter_attempts, 2L)
    expect_false(out$res$outer_grid_prior_added)
    expect_identical(out$res$outer_grid_prior_declined, "prior_pinned")
    expect_equal(calls$last_prior_sigma, chosen)
})

test_that("a prior_sigma stated on a block pins the escalation on that block only", {
    skip_on_cran_budget()
    pinned <- tulpa:::.nl_prior_sigma_pinned_at
    icar <- function(anchor = NULL) c(list(type = "icar"),
                                      if (!is.null(anchor)) list(prior_sigma = anchor))
    expect_true(pinned(NULL, list(icar(c(1, 0.01))), 1L))
    expect_false(pinned(NULL, list(icar(), icar(c(1, 0.01))), 1L))
    expect_true(pinned(NULL, list(icar(), icar(c(1, 0.01))), 2L))
    expect_true(pinned(NULL, icar(c(1, 0.01)), 1L))
    # The engine's own anchor, or one the caller marked a default, adds nothing.
    expect_false(pinned(NULL, list(icar(c(3, 0.01))), 1L))
    expect_false(pinned(NULL, list(icar(auto_grid(c(1, 0.01)))), 1L))
    expect_false(pinned(NULL, list(icar(), icar()), 2L))

    # A railed copy block whose own anchor the caller stated: the escalated door
    # spec would replace that anchor's density, so the second attempt holds it.
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "b1.sigma",
                     outer_mode_u = c(log(3), 0.4),
                     outer_mode_cov_u = diag(c(0.25, 0.09)),
                     outer_mode_axis_tags = c("log", "identity"),
                     outer_mode_axis_names = c("b1.sigma", "b1.alpha"),
                     outer_grid_placement = "fixed",
                     blocks = list(list(type = "icar")))
    cp <- list(has_copy = TRUE, copy_blocks_zero = 0L)
    calls <- new.env(parent = emptyenv())
    refit <- function(prior_i, prior_sigma_i) {
        calls$last_prior_sigma <- prior_sigma_i
        stub_res
    }
    out <- .place_copy_sd(stub_res, list(icar(c(1, 0.01))), NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_recenter_attempts, 2L)
    expect_false(out$res$outer_grid_prior_added)
    expect_identical(out$res$outer_grid_prior_declined, "prior_pinned")
    expect_null(calls$last_prior_sigma)

    pc <- tulpa:::.nl_recenter("sigma_pc_prior")
    # A stated anchor on ANOTHER block neither pins the copy block's escalation
    # nor is replaced: the door spec names the copy block alone, so block 2
    # keeps its own anchor.
    stub_two <- stub_res
    stub_two$outer_mode_axis_names <- c("b1.sigma", "b1.alpha", "b2.sigma")
    stub_two$outer_mode_u <- c(log(3), 0.4, 0)
    stub_two$outer_mode_cov_u <- diag(c(0.25, 0.09, 0.25))
    stub_two$outer_mode_axis_tags <- c("log", "identity", "log")
    stub_two$blocks <- list(list(type = "icar"), list(type = "icar"))
    refit_two <- function(prior_i, prior_sigma_i) {
        calls$last_prior_sigma <- prior_sigma_i
        stub_two
    }
    out <- .place_copy_sd(stub_two, list(icar(), icar(c(1, 0.01))), NULL, cp,
                          NULL, refit_two)
    expect_true(out$res$outer_grid_prior_added)
    expect_null(out$res$outer_grid_prior_declined)
    expect_equal(calls$last_prior_sigma, list(list(block = 1L, prior = pc)))

    # A wrapper default at the door keeps reaching the other block, while the
    # copy block takes the engine prior.
    wrapper_default <- auto_grid(list("pc.prec", c(U = 1, alpha = 0.01)))
    out <- .place_copy_sd(stub_two, list(icar(), icar()), NULL, cp,
                          wrapper_default, refit_two)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, list(
        list(block = 1L, prior = pc),
        list(block = 2L, prior = list("pc.prec", c(U = 1, alpha = 0.01)))))

    # A block anchor at the engine's own value escalates as before.
    out <- .place_copy_sd(stub_res, list(icar(c(3, 0.01))), NULL, cp, NULL, refit)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, list(list(block = 1L, prior = pc)))
})

test_that("the escalated per-block prior_sigma parses to the copy block alone", {
    skip_on_cran_budget()
    # The per-block door spec the escalation writes parses and is applied to the
    # copy block's field-SD axis alone.
    fn <- tulpa:::.joint_parse_hyperprior(
        list(list(block = 1L, prior = tulpa:::.nl_recenter("sigma_pc_prior"))),
        "prior_sigma")
    expect_s3_class(fn, "tulpa_block_hyperprior")
    expect_identical(tulpa:::.joint_hp_named_blocks(fn), 1L)
    expect_null(tulpa:::.joint_hp_fn_for_block(fn, 2L))
})

test_that("the multi-block rescue reports a pinned copy-block axis", {
    skip_on_cran_budget()
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "b1.sigma",
                     outer_mode_u = c(log(3), 0.4),
                     outer_mode_cov_u = diag(c(0.25, 0.09)),
                     outer_mode_axis_tags = c("log", "identity"),
                     outer_mode_axis_names = c("b1.sigma", "b1.alpha"),
                     outer_grid_placement = "fixed",
                     blocks = list(list(type = "icar")))
    cp <- list(has_copy = TRUE, copy_blocks_zero = 0L)
    refit <- function(prior_i, prior_sigma_i) {
        utils::modifyList(stub_res, list(pareto_k_regime = "spread",
                                         pareto_k_grid_edge_axes = character(0),
                                         .grid = prior_i[[1L]]$sigma_grid))
    }

    # Default (absent) donor axis -> recenters.
    out <- .place_copy_sd(
        stub_res, list(list(type = "icar")), NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_gt(max(out$res$.grid), 3)
    expect_identical(out$res$outer_grid_recenter_axes, "b1.sigma")
    expect_null(out$res$outer_grid_axis_declined)

    # Pinned donor axis -> untouched, reason recorded (was previously
    # indistinguishable from "did not need it").
    out <- .place_copy_sd(
        stub_res, list(list(type = "icar", sigma_grid = c(0.1, 0.5, 1, 2, 3))),
        NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "fixed")
    expect_identical(out$res$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.sigma = "axis_pinned"))

    out <- .place_copy_sd(
        stub_res, list(list(type = "icar")), NULL, cp, NULL, refit,
        enabled = FALSE)
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.sigma = "auto_recenter_disabled"))

    # Two collapsed copy blocks and one attempt: both are laid from the one
    # mode in the same refit, and both say they ran out of attempts -- the
    # refit here leaves them railed.
    two <- utils::modifyList(stub_res, list(
        pareto_k_grid_edge_axes = c("b1.sigma", "b2.sigma"),
        outer_mode_u = c(log(3), log(3)),
        outer_mode_cov_u = diag(c(0.25, 0.25)),
        outer_mode_axis_tags = c("log", "log"),
        outer_mode_axis_names = c("b1.sigma", "b2.sigma"),
        blocks = list(list(type = "icar"), list(type = "icar"))))
    still <- function(prior_i, prior_sigma_i) two
    out <- .place_copy_sd(
        two, list(list(type = "icar"), list(type = "icar")), NULL,
        list(has_copy = TRUE, copy_blocks_zero = c(0L, 1L)), NULL, still,
        max_attempts = 1L)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(out$res$outer_grid_recenter_axes, c("b1.sigma", "b2.sigma"))
    expect_identical(out$res$outer_grid_axis_declined,
                     c(b1.sigma = "attempts_exhausted",
                       b2.sigma = "attempts_exhausted"))

    # Engine default on the donor block (the shape a wrapper package writes)
    # -> recenters.
    out <- .place_copy_sd(
        stub_res,
        list(list(type = "icar", sigma_grid = tulpa:::.nl_grid_axis("field_sd"))),
        NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
})
