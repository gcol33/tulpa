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

test_that("a pinned axis is pinned and every flavour of default is not", {
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
                      rho = c(0.2, 0.5, 0.8, 0.95))
    crossed <- pinned
    crossed$sigma_grid <- gr$sigma
    expect_false(tulpa:::.nl_axis_is_pinned(crossed, "sigma_grid"))
    # A default axis with ONE node moved is a pin again.
    moved <- eng
    moved$sigma_grid[5L] <- 6
    expect_true(tulpa:::.nl_axis_is_pinned(moved, "sigma_grid"))
})

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

test_that("provenance strips markers and records them per block", {
    single <- list(type = "icar", sigma_grid = auto_grid(c(1, 2, 3)),
                   rho_grid = c(0.2, 0.9))
    pv <- tulpa:::.nl_grid_provenance(single)
    expect_identical(pv$auto, "sigma_grid")
    expect_false(is_auto_grid(pv$prior$sigma_grid))
    expect_identical(pv$prior$sigma_grid, c(1, 2, 3))
    expect_identical(tulpa:::.nl_auto_fields_at(pv$auto), "sigma_grid")

    multi <- list(list(type = "icar", sigma_grid = auto_grid(c(1, 2))),
                  list(type = "iid",  sigma_grid = c(0.3, 0.6)))
    pvm <- tulpa:::.nl_grid_provenance(multi)
    expect_identical(pvm$auto[[1L]], "sigma_grid")
    expect_identical(pvm$auto[[2L]], character(0))
    expect_false(is_auto_grid(pvm$prior[[1L]]$sigma_grid))
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 1L), "sigma_grid")
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 2L), character(0))
    # Out-of-range / absent records read as "nothing declared", never an error.
    expect_identical(tulpa:::.nl_auto_fields_at(pvm$auto, 9L), character(0))
    expect_identical(tulpa:::.nl_auto_fields_at(NULL, 1L), character(0))
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

test_that("axis index resolves by name, then by being the only log axis", {
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

test_that("recentering from a fit resolves a prefixed axis and its curvature", {
    mode_u <- c(log(3), 0.4)
    cov_u  <- diag(c(0.25, 0.09))
    ax <- tulpa:::.nl_axis_recenter_from_fit(
        mode_u, cov_u, c("log", "identity"), c("b1.sigma", "b1.alpha"),
        "sigma", block_index = 1L, n_blocks = 1L)
    expect_length(ax, 5L)
    expect_true(all(diff(ax) > 0))
    # Centred on the mode, spanning +/- 2.5 SD on the log axis.
    expect_equal(ax[3L], 3, tolerance = 1e-8)
    expect_equal(log(ax[5L]) - log(ax[1L]), 2 * 2.5 * 0.5, tolerance = 1e-8)
    # An axis on another transform, or absent, declines.
    expect_null(tulpa:::.nl_axis_recenter_from_fit(
        mode_u, cov_u, c("log", "identity"), c("b1.sigma", "b1.alpha"),
        "alpha", block_index = 1L, n_blocks = 1L))
    expect_null(tulpa:::.nl_axis_recenter_from_fit(
        mode_u, cov_u, c("log", "identity"), c("b1.sigma", "b1.alpha"),
        "sigma", block_index = 2L, n_blocks = 3L))
    expect_null(tulpa:::.nl_axis_recenter_from_fit(NULL, cov_u, "log", "sigma",
                                                   "sigma"))
})

test_that("a decline reason is stamped only when the recenter did not run", {
    declined <- tulpa:::.nl_decline_recenter(
        list(outer_grid_placement = "fixed"), "axis_pinned")
    expect_identical(declined$outer_grid_recenter_declined, "axis_pinned")
    ran <- tulpa:::.nl_decline_recenter(
        list(outer_grid_placement = "auto_recentered"), "axis_pinned")
    expect_null(ran$outer_grid_recenter_declined)
})

test_that("the single-block rescue guard honours a pin and passes a default", {
    # No fit: a stub `res` carrying only the diagnostic fields the rescue reads,
    # and a refit that records whether it was called.
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "sigma",
                     pareto_k_grid_edge_sides = "upper",
                     pareto_k_mode_u = log(3),
                     pareto_k_cov_u = matrix(0.25, 1, 1),
                     pareto_k_axis_tags = "log",
                     pareto_k_axis_names = "sigma",
                     outer_grid_placement = "fixed")
    calls <- new.env(parent = emptyenv())
    calls$n <- 0L
    refit <- function(prior_i, prior_sigma_i) {
        calls$n <- calls$n + 1L
        calls$grid <- prior_i$sigma_grid
        # Refit lands on a spread grid, so the loop stops after one attempt.
        utils::modifyList(stub_res, list(pareto_k_regime = "spread",
                                         pareto_k_grid_edge_axes = character(0)))
    }
    icar <- list(type = "icar")

    # 1. Axis absent -> recenters.
    out <- tulpa:::.joint_sigma_grid_rescue(stub_res, icar, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)
    expect_gt(max(calls$grid), 3)
    expect_null(out$res$outer_grid_recenter_declined)

    # 2. Engine default handed in unmarked -> still recenters (#293).
    calls$n <- 0L
    out <- tulpa:::.joint_sigma_grid_rescue(
        stub_res, c(icar, list(sigma_grid = tulpa:::.nl_grid_axis("field_sd"))),
        NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)

    # 3. A marked caller default -> recenters.
    calls$n <- 0L
    out <- tulpa:::.joint_sigma_grid_rescue(
        stub_res, c(icar, list(sigma_grid = auto_grid(c(0.2, 0.6, 1.8)))),
        NULL, refit, auto = "sigma_grid")
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(calls$n, 1L)

    # 4. A genuine pin -> untouched, with the reason recorded.
    calls$n <- 0L
    out <- tulpa:::.joint_sigma_grid_rescue(
        stub_res, c(icar, list(sigma_grid = c(0.1, 0.5, 1, 2, 3))), NULL, refit)
    expect_identical(out$res$outer_grid_placement, "fixed")
    expect_identical(out$res$outer_grid_recenter_declined, "axis_pinned")
    expect_identical(calls$n, 0L)

    # 5. A grid that never collapsed -> untouched, different reason.
    calls$n <- 0L
    spread <- utils::modifyList(stub_res,
                                list(pareto_k_regime = "spread",
                                     pareto_k_grid_edge_axes = character(0)))
    out <- tulpa:::.joint_sigma_grid_rescue(spread, icar, NULL, refit)
    expect_identical(out$res$outer_grid_recenter_declined, "grid_not_collapsed")
    expect_identical(calls$n, 0L)

    # 6. Collapsed, but no curvature to recentre on -> untouched, own reason.
    calls$n <- 0L
    blind <- utils::modifyList(stub_res, list(pareto_k_mode_u = NULL,
                                              pareto_k_cov_u = NULL))
    out <- tulpa:::.joint_sigma_grid_rescue(blind, icar, NULL, refit)
    expect_identical(out$res$outer_grid_recenter_declined, "no_usable_curvature")
    expect_identical(calls$n, 0L)

    # 7. A prior shape this rescue does not cover stamps nothing at all -- the
    #    multi-block rescue chained after it owns that fit.
    out <- tulpa:::.joint_sigma_grid_rescue(stub_res, list(icar), NULL, refit)
    expect_null(out$res$outer_grid_recenter_declined)
})

# gcol33/tulpa#297: the second attempt's regularizing PC prior was suppressed by
# ANY supplied prior_sigma, decided by presence -- so a wrapper stamping a
# prior_sigma of its own silently turned the escalation into a second geometry
# recenter while still reporting `attempts = 2`.
test_that(".nl_prior_sigma_is_pinned reads provenance, not presence", {
    pinned <- tulpa:::.nl_prior_sigma_is_pinned
    expect_false(pinned(NULL))
    # The engine's own default handed back in carries no information a pin adds.
    expect_false(pinned(tulpa:::.nl_recenter("sigma_pc_prior")))
    # A caller declaring its own spec a default.
    expect_false(pinned(auto_grid(list("pc.prec", c(U = 1, alpha = 0.01)))))
    # A different prior, undeclared: a real choice.
    expect_true(pinned(list("pc.prec", c(U = 1, alpha = 0.01))))
    expect_true(pinned(list("half_normal", 2)))
})

test_that("the second recenter attempt engages the PC prior unless it was PINNED", {
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "sigma",
                     pareto_k_grid_edge_sides = "upper",
                     pareto_k_mode_u = log(3),
                     pareto_k_cov_u = matrix(0.25, 1, 1),
                     pareto_k_axis_tags = "log",
                     pareto_k_axis_names = "sigma",
                     outer_grid_placement = "fixed")
    calls <- new.env(parent = emptyenv())
    # Never leaves collapsed_edge, so the loop runs its full two attempts.
    refit <- function(prior_i, prior_sigma_i) {
        calls$last_prior_sigma <- prior_sigma_i
        stub_res
    }
    icar <- list(type = "icar")
    pc <- tulpa:::.nl_recenter("sigma_pc_prior")

    # No prior at all -> attempt 2 adds the engine's PC prior.
    out <- tulpa:::.joint_sigma_grid_rescue(stub_res, icar, NULL, refit)
    expect_identical(out$res$outer_grid_recenter_attempts, 2L)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, pc)
    expect_null(out$res$outer_grid_prior_declined)

    # A wrapper's own default, declared -> the escalation still engages, and the
    # engine's prior replaces the declared default.
    wrapper_default <- auto_grid(list("pc.prec", c(U = 1, alpha = 0.01)))
    out <- tulpa:::.joint_sigma_grid_rescue(stub_res, icar, wrapper_default, refit)
    expect_true(out$res$outer_grid_prior_added)
    expect_equal(calls$last_prior_sigma, pc)
    expect_null(out$res$outer_grid_prior_declined)

    # A deliberate, undeclared choice -> held, and the suppression is legible
    # rather than an `attempts = 2` that quietly did the same thing twice.
    chosen <- list("pc.prec", c(U = 1, alpha = 0.01))
    out <- tulpa:::.joint_sigma_grid_rescue(stub_res, icar, chosen, refit)
    expect_identical(out$res$outer_grid_recenter_attempts, 2L)
    expect_false(out$res$outer_grid_prior_added)
    expect_identical(out$res$outer_grid_prior_declined, "prior_pinned")
    expect_equal(calls$last_prior_sigma, chosen)
})

test_that("auto_grid() marks a prior specification as well as a grid or a knob", {
    m <- auto_grid(list("pc.prec", c(U = 3, alpha = 0.01)))
    expect_true(is_auto_grid(m))
    expect_true(is.list(m))
    expect_identical(m[[1L]], "pc.prec")
    # A scalar control knob (fit_st_nested's grid knobs, gcol33/tulpa#294).
    k <- auto_grid(4L)
    expect_true(is_auto_grid(k))
    expect_identical(as.numeric(k), 4)
    expect_error(auto_grid(list()), "non-empty")
})

test_that("the multi-block rescue reports a pinned copy-block axis", {
    stub_res <- list(pareto_k_regime = "collapsed_edge",
                     pareto_k_grid_edge_axes = "b1.sigma",
                     pareto_k_mode_u = c(log(3), 0.4),
                     pareto_k_cov_u = diag(c(0.25, 0.09)),
                     pareto_k_axis_tags = c("log", "identity"),
                     pareto_k_axis_names = c("b1.sigma", "b1.alpha"),
                     outer_grid_placement = "fixed",
                     blocks = list(list(type = "icar")))
    cp <- list(has_copy = TRUE, copy_blocks_zero = 0L)
    refit <- function(prior_i, prior_sigma_i) {
        utils::modifyList(stub_res, list(pareto_k_regime = "spread",
                                         pareto_k_grid_edge_axes = character(0),
                                         .grid = prior_i[[1L]]$sigma_grid))
    }

    # Default (absent) donor axis -> recenters.
    out <- tulpa:::.joint_multi_sigma_grid_rescue(
        stub_res, list(list(type = "icar")), NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_gt(max(out$res$.grid), 3)

    # Pinned donor axis -> untouched, reason recorded (was previously
    # indistinguishable from "did not need it").
    out <- tulpa:::.joint_multi_sigma_grid_rescue(
        stub_res, list(list(type = "icar", sigma_grid = c(0.1, 0.5, 1, 2, 3))),
        NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "fixed")
    expect_identical(out$res$outer_grid_recenter_declined, "axis_pinned")

    # Engine default on the donor block (the shape a wrapper package writes)
    # -> recenters.
    out <- tulpa:::.joint_multi_sigma_grid_rescue(
        stub_res,
        list(list(type = "icar", sigma_grid = tulpa:::.nl_grid_axis("field_sd"))),
        NULL, cp, NULL, refit)
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
})
