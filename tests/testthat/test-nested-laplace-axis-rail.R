# A default outer axis that does not contain its own posterior mode
# (gcol33/tulpa#361).
#
# `.NL_GRID` fixes every default span as a constant, so a fit whose posterior
# sits past one end integrates a tail at any spacing. The engine's rescue
# (gcol33/tulpa#290) is what moves such an axis, and #361 is the two places it
# could not reach: it recentred ONE axis per family and only in the `log`
# coordinate, so a BYM2 `rho` railed against its 0.95 ceiling was detected
# (`pareto_k_grid_edge_axes` names it) and then left there.
#
# Detection is per axis here, off the fit's own marginal -- the same marginal
# `.nl_axis_quantiles()` reports that axis's median and interval from -- rather
# than off the whole tensor's `ess_grid`.

.rail_chain_adj <- function(n_s) {
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn <- vapply(nbr, length, integer(1))
    list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
         adj_col_idx = as.integer(unlist(nbr)) - 1L,
         n_neighbors = as.integer(nn))
}

# A one-axis stub result carrying `w` as its marginal over `vals`.
.rail_stub <- function(vals, w, name = "tau") {
    tg <- matrix(vals, ncol = 1L, dimnames = list(NULL, name))
    list(theta_grid = tg, theta_names = name, weights = w / sum(w),
         log_marginal = log(w / sum(w)))
}

skip_on_cran_budget()

test_that("an axis is railed exactly when its own marginal peaks at an endpoint", {
    skip_on_cran_budget()
    up   <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.001, 0.02, 0.35, 0.63))
    down <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.70, 0.25, 0.04, 0.01))
    mid  <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.05, 0.60, 0.30, 0.05))
    flat <- .rail_stub(c(0.2, 0.5, 0.8, 0.95), c(0.26, 0.25, 0.245, 0.245))

    expect_identical(.nl_axis_rail(up,   "tau")$side, "upper")
    expect_identical(.nl_axis_rail(down, "tau")$side, "lower")
    expect_null(.nl_axis_rail(mid,  "tau"))
    # Maximal at an endpoint by 0.015 of the weight: the argmax of a marginal
    # this flat is arbitrary, and a recenter would lay a coarser grid.
    expect_null(.nl_axis_rail(flat, "tau"))

    expect_identical(.nl_railed_axes(up),  "tau:upper")
    expect_identical(.nl_railed_axes(mid), character(0))
})

test_that("the rail verdict does not move with the axis's node count", {
    skip_on_cran_budget()
    # gcol33/tulpa#375. The materiality guard reads a per-NODE quantity, so it
    # has to be compared against a per-node reference. Against a fixed SHARE the
    # same posterior read at more nodes carries less on any one of them, and a
    # longer axis becomes a weaker detector -- which is the opposite of what
    # #361's longer `bym2_rho` span was meant to compose with.
    #
    # One marginal shape, read at six node counts over ONE fixed span in the
    # axis's own coordinate. The span is what fixes the posterior geometry; the
    # node count is the only thing that moves.
    span <- stats::qlogis(c(0.2, 0.95))
    ms   <- c(4L, 5L, 6L, 8L, 12L, 20L)
    read_at <- function(m, shape) {
        u <- seq(span[1L], span[2L], length.out = m)
        w <- shape(u); w <- w / sum(w)
        .rail_stub(stats::plogis(u), w, "rho")
    }

    # Piled against the ceiling: a mode 1.5 logit units PAST the top node, so
    # every read has its argmax at that node whatever the resolution.
    piled <- function(u) exp(-0.5 * ((u - (span[2L] + 1.5)) / 2.1)^2)
    lift  <- vapply(ms, function(m) .nl_axis_rail(read_at(m, piled), "rho")$lift,
                    numeric(1))
    expect_true(all(lift >= 2))
    # The share it used to be read against falls by more than 3x over the same
    # reads, and drops through 0.5 -- the defect, in one line.
    share <- vapply(ms, function(m) .nl_axis_rail(read_at(m, piled), "rho")$mass,
                    numeric(1))
    expect_gt(share[1L] / share[length(share)], 3)
    expect_gt(share[1L], 0.5)
    expect_lt(share[length(share)], 0.5)

    # Flat to within the weights' own noise: the argmax is arbitrary and lands
    # on the top node by construction. Declines at every node count -- the
    # guard's own job, which the rescale must not cost.
    for (m in ms) {
        set.seed(400L + m)
        w <- rep(1, m) + stats::runif(m, -0.02, 0.02)
        w[m] <- max(w) + 0.01
        expect_null(.nl_axis_rail(.rail_stub(
            stats::plogis(seq(span[1L], span[2L], length.out = m)),
            w / sum(w), "rho"), "rho"))
    }

    # And a marginal that merely tilts: 1.6x from the bottom node to the top
    # over the whole span, no curvature to recentre onto.
    gentle <- function(u) exp(0.47 * (u - u[1L]) / diff(range(u)))
    for (m in ms) expect_null(.nl_axis_rail(read_at(m, gentle), "rho"))
})

skip_on_cran()

test_that("a BYM2 fit whose mixing weight rails is moved off its ceiling", {
    skip_on_cran()
    # gcol33/tulpa#357's census configuration: 100 chain-adjacent regions, whose
    # `bym2_rho` posterior is maximal at a 0.95 top node. Pre-#361 the fit
    # reported a rho median of 0.89 against a span stopping at 0.95, with the
    # upper interval bound extrapolated PAST the support, and recorded the
    # decline reason `grid_not_collapsed` on a grid that had collapsed.
    #
    # The 0.95 ceiling is supplied here through `auto_grid()` rather than taken
    # from `.NL_GRID`: #361's span change carried the default to 0.999, which
    # CONTAINS this fixture's mode, so the engine's own axis no longer rails on
    # it. `auto_grid()` is the marker for "a default the engine may move", so
    # the rescue path exercised is identical and the statement no longer moves
    # with the default's value. The block below asserts what the longer default
    # does on the same data.
    S <- 100L
    set.seed(505L + S + 5000L)
    eff <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4)), scale = FALSE)) +
           rnorm(S, 0, 0.2)
    set.seed(505L + S)
    idx <- rep(seq_len(S), each = 10L)
    X <- cbind(1, rnorm(length(idx)))
    y <- as.numeric(X %*% c(-0.2, 0.7)) + eff[idx] +
         rnorm(length(idx), 0, sqrt(0.5))

    gr <- expand.grid(sigma = .nl_grid_axis("field_sd"),
                      rho = c(0.2, 0.5, 0.8, 0.95), KEEP.OUT.ATTRS = FALSE)
    prior <- c(list(type = "bym2", n_spatial_units = S, spatial_idx = idx,
                    scale_factor = 1,
                    sigma_grid = auto_grid(gr$sigma),
                    rho_grid   = auto_grid(gr$rho)), .rail_chain_adj(S))
    # The grids are held as placed (`axis_refine = "none"`): refinement follows
    # a mixing weight out past an end node holding mass on its own, and the
    # statements here are about what the placement pass does.
    ctrl <- list(max_iter = 200L, tol = 1e-9, n_threads = 1L,
                 diagnose_k = FALSE, diagnose_skew = FALSE,
                 axis_refine = "none")
    fit_held <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = 0.5,
        control = c(ctrl, list(auto_recenter = FALSE))))
    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = prior,
        family = "gaussian", phi = 0.5, control = ctrl))

    # Held where it is, the axis stops at 0.95 with its marginal still climbing.
    expect_identical(max(fit_held$theta_grid[, "b1.rho"]), 0.95)
    expect_identical(.nl_axis_rail(fit_held, "b1.rho")$side, "upper")
    expect_true("b1.rho:upper" %in% fit_held$outer_grid_railed_axes)

    # Placed, the axis brackets its own mode and the reported interval is
    # inside the support the mixing weight lives on.
    expect_identical(fit$outer_grid_placement, "auto_recentered")
    expect_true("b1.rho" %in% fit$outer_grid_recenter_axes)
    expect_gt(max(fit$theta_grid[, "b1.rho"]), 0.95)
    expect_null(.nl_axis_rail(fit, "b1.rho"))
    # A dense 60 x 80 (log sigma, logit rho) tensor puts rho's median at 0.850
    # and its 95% interval at 0.574 to 0.991; the held axis reads the median
    # 0.881. The placed axis's interval brackets that median.
    expect_lt(as.numeric(fit$theta_ci_lo[["b1.rho"]]), 0.850)
    expect_gt(as.numeric(fit$theta_ci_hi[["b1.rho"]]), 0.850)
    # Both reads report a mixing weight inside its own support, which is
    # gcol33/tulpa#369 and not placement -- before that the held axis overshot
    # to 1.0028 and the moved one to 1.0012.
    expect_lt(as.numeric(fit$theta_ci_hi[["b1.rho"]]), 1)
    expect_lt(as.numeric(fit_held$theta_ci_hi[["b1.rho"]]), 1)

    # The engine's own axis, on the same data, held as laid. gcol33/tulpa#361's
    # span change is what makes this a different statement from the block
    # above: a bounded axis's SPAN and its RESOLUTION are not interchangeable,
    # so a span topping out at 0.95 could not report a bound above 0.97642 at
    # any node count.
    default_prior <- c(list(type = "bym2", n_spatial_units = S,
                            spatial_idx = idx, scale_factor = 1),
                       .rail_chain_adj(S))
    fit_default <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = default_prior,
        family = "gaussian", phi = 0.5,
        control = c(ctrl, list(auto_recenter = FALSE))))
    expect_identical(max(fit_default$theta_grid[, "b1.rho"]), 0.999)
    # It contains its own mode, so nothing rails.
    expect_null(.nl_axis_rail(fit_default, "b1.rho"))
    expect_identical(fit_default$outer_grid_railed_axes, character(0))
    # And it reports past the retired cap, inside the support.
    expect_gt(as.numeric(fit_default$theta_ci_hi[["b1.rho"]]), 0.97642)
    expect_lt(as.numeric(fit_default$theta_ci_hi[["b1.rho"]]), 1)
    expect_gt(as.numeric(fit_default$theta_median[["b1.rho"]]),
              as.numeric(fit_held$theta_median[["b1.rho"]]))

    # The placement pass re-places this fit even though nothing rails, because
    # the default `(sigma, rho)` grid does not resolve its own posterior
    # (gcol33/tulpa#361), and the reported mixing weight stays inside its
    # support.
    fit_resolve <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = default_prior,
        family = "gaussian", phi = 0.5, control = ctrl))
    expect_identical(fit_resolve$outer_grid_placement, "auto_recentered")
    expect_identical(fit_resolve$outer_grid_railed_axes, character(0))
    expect_lt(as.numeric(fit_resolve$theta_ci_hi[["b1.rho"]]), 1)

    # A per-axis policy name is not a placement setting.
    expect_error(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, length(y)), X = X, prior = default_prior,
        family = "gaussian", phi = 0.5,
        control = c(ctrl, list(auto_recenter = "rail"))),
        "TRUE or FALSE")
})
