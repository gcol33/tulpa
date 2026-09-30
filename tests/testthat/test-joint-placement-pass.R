# The joint placement pass (R/joint_placement_pass.R, gcol33/tulpa#925): every
# movable axis laid from one outer mode, the copy scale among them, and what
# the var-of-means consistency pass does with that mode afterwards.

# --- the copy scale ----------------------------------------------------------

# A single-block detecting fit whose copy-scale continuum sits on one node, 20
# posterior SDs per node, with the outer mode a mode-find reached for it.
.pp_alpha_stub <- function() {
    a <- c(0, exp(seq(log(0.1), log(3), length.out = 9)))
    list(theta_grid = matrix(a, ncol = 1L, dimnames = list(NULL, "alpha")),
         log_marginal = c(-50, -0.5 * ((log(a[-1]) - log(0.358)) / 0.02)^2),
         outer_mode_u = log(0.36), outer_mode_cov_u = matrix(0.02^2, 1, 1),
         outer_mode_axis_tags = "log", outer_mode_axis_names = "alpha",
         outer_grid_placement = "fixed")
}

test_that("an unresolved copy scale is placed with its atom kept, marked", {
    fams <- .joint_placement_families()["alpha"]
    fc <- list(name = "alpha",
               grid = auto_grid(c(0, exp(seq(log(0.1), log(3), length.out = 9)))))
    st <- list(prior = list(type = "icar"),
               responses = list(occ = list(), pos = list(field_coef = fc)))
    seen <- new.env(parent = emptyenv())
    out <- .joint_place_axes(.pp_alpha_stub(), st, families = fams,
                             refit = function(st, from) {
                                 seen$st <- st
                                 g <- as.numeric(st$responses$pos$field_coef$grid)
                                 list(theta_grid = matrix(g, ncol = 1L,
                                          dimnames = list(NULL, "alpha")),
                                      log_marginal = c(-50, -0.5 * ((log(g[-1]) -
                                          log(0.36)) / 0.02)^2),
                                      outer_grid_placement = "fixed")
                             })
    expect_identical(out$res$outer_grid_placement, "auto_recentered")
    expect_identical(out$res$outer_grid_recenter_axes, "alpha")
    g <- seen$st$responses$pos$field_coef$grid
    # The atom and five continuum nodes, the continuum at the mode, and written
    # back as the engine's own default.
    expect_identical(g[1L], 0)
    expect_length(g, 6L)
    expect_equal(log(g[4L]), log(0.36))
    expect_true(is_auto_grid(g))
    expect_null(seen$st$responses$pos$field_coef[["n"]])
})

test_that("a stated copy scale is held, and says so", {
    fams <- .joint_placement_families()["alpha"]
    fc <- list(name = "alpha", grid = c(0, 0.2, 0.5, 1, 2))
    st <- list(prior = list(type = "icar"),
               responses = list(occ = list(), pos = list(field_coef = fc)))
    out <- .joint_place_axes(.pp_alpha_stub(), st, families = fams,
                             refit = function(st, from) stop("no refit"))
    expect_identical(out$res$outer_grid_axis_declined, c(alpha = "axis_pinned"))
    expect_identical(out$res$outer_grid_recenter_declined, "axis_pinned")
})

test_that("a copy scale heaviest on its atom is neither railed nor placed", {
    a <- c(0, 0.1, 0.3, 1, 3)
    res <- list(theta_grid = matrix(a, ncol = 1L, dimnames = list(NULL, "alpha")),
                log_marginal = c(0, -3, -6, -9, -12))
    # The mass is on the "no coupling" model, not past the span.
    expect_null(.nl_axis_rail(res, "alpha"))
    expect_false(.nl_axis_placement_fires(res, "alpha", "log"))
    # Mass on the continuum's top node IS a rail, read on the continuum.
    res$log_marginal <- c(-20, -12, -9, -6, 0)
    expect_identical(.nl_axis_rail(res, "alpha")$side, "upper")
    expect_true(.nl_axis_placement_fires(res, "alpha", "log"))
    # The resolution read leaves the atom out.
    expect_true(is.finite(.nl_axis_h_over_sd(res, "alpha", "log")))
})

test_that("the pilot thins a marked copy scale and keeps a stated one", {
    marked <- auto_grid(c(0, exp(seq(log(0.1), log(3), length.out = 9))))
    thin <- .nl_pilot_field_coef(list(field_coef = list(name = "alpha",
                                                          grid = marked)), 3L)
    expect_true(thin$moved)
    g <- thin$arm$field_coef$grid
    expect_equal(as.numeric(g), c(0, 0.1, sqrt(0.3), 3))
    expect_true(is_auto_grid(g))
    stated <- .nl_pilot_field_coef(list(field_coef = list(
        name = "alpha", grid = c(0, 0.2, 0.5, 1, 2))), 3L)
    expect_false(stated$moved)
})

# --- the consistency pass at the found mode ----------------------------------

test_that("points at the mode are one SD apart, inside a stated span", {
    spec <- hyper_axis_spec("phi_pos", grid = c(1, 3.91, 15.3, 60),
                            log_scale = TRUE, refinable = TRUE, extend = FALSE)
    mode <- list(mode_u = log(4.2), sd_u = 0.02, tag = "log")
    pts <- .hyper_propose_at_mode(spec, c(1, 3.91, 15.3, 60), mode)
    # Nearest the mode first, one SD apart, running on past 2 SDs on a side
    # until the box beyond the last step reads under the bound: to 3 below,
    # where the declared node at 3.58 SDs closes the gap, and to 6 above,
    # where the next node is 65 SDs out.
    k <- (log(pts) - log(4.2)) / 0.02
    expect_equal(k, c(0, -1, 1, -2, 2, -3, 3, 4, 5, 6), tolerance = 1e-9)
    # A node within half an SD of a point already reads the density there.
    near <- list(mode_u = log(3.9), sd_u = 0.1, tag = "log")
    expect_false(any(abs(log(.hyper_propose_at_mode(
        spec, c(1, 3.91, 15.3, 60), near)) - log(3.91)) <= 0.05))
    # A stated axis keeps them inside its declared span.
    edge <- list(mode_u = log(1.05), sd_u = 0.1, tag = "log")
    expect_true(all(.hyper_propose_at_mode(spec, c(1, 3.91, 15.3, 60), edge) >= 1))
    # An axis already laid at the placement's 1.25 SDs around the mode reads the
    # density there: every point has a node within half an SD, so none is
    # proposed and the pass bisects as it would without a mode.
    placed <- exp(log(3.9) + c(-2.5, -1.25, 0, 1.25, 2.5) * 0.1)
    expect_length(.hyper_propose_at_mode(spec, placed, near), 0L)
    # No mode, or no spread, proposes nothing.
    expect_length(.hyper_propose_at_mode(spec, c(1, 60), NULL), 0L)
    expect_length(.hyper_propose_at_mode(
        spec, c(1, 60), list(mode_u = 1, sd_u = 0, tag = "log")), 0L)
})

test_that("a transported mode leaves out an axis the mode-find did not resolve", {
    mode <- list(mode_u = c(log(3), log(0.2), 0.5),
                 cov_u = diag(c(0.01, 25, 0)),
                 tags = c("log", "log", "log"),
                 names = c("sigma", "alpha", "phi_pos"))
    ax <- .nl_outer_mode_axes(mode)
    # sd 0.1 is kept as measured, well under the placement floor; sd 5 is past
    # the placement ceiling, and a held axis has no spread at all.
    expect_identical(names(ax), "sigma")
    expect_equal(ax$sigma$sd_u, 0.1)
    expect_null(.nl_outer_mode_axes(NULL))
})

# The axis mean on the log scale, read off the weights a consistency pass left.
.cp_log_mean <- function(out, specs, axis) {
    lq <- .hyper_log_quad_weights(out$theta_grid, specs,
                                  refining = out$refining_axis)
    lw <- out$log_marginal + lq
    w <- exp(lw - max(lw))
    sum(w * log(out$theta_grid[, axis])) / sum(w)
}

# A two-axis grid shaped like the full 25 km Calluna fit's outer posterior: the
# copy scale laid around its mode at the placement's SD floor, the dispersion
# axis pinned at declared nodes 36 posterior SDs from its mode.
.cp_calluna_like <- function(offset = 0, beta = 0) {
    a0 <- 0.274; sa <- 0.0464; p0 <- 3.25; sp <- 0.0053
    p_true <- p0 * exp(offset * sp)
    alpha <- a0 * exp(c(-2, -1, 0, 1, 2) * 1.25 * 0.15)
    phi <- c(1, 3.91, 15.3, 60)
    tg <- as.matrix(expand.grid(alpha = alpha, phi_pos = phi))
    # `beta` couples the copy scale to the dispersion: log alpha given log phi
    # moves by `beta` per unit of log phi off its mode.
    lp <- function(g) {
        dp <- log(g[, "phi_pos"]) - log(p_true)
        -0.5 * ((log(g[, "alpha"]) - log(a0) - beta * dp) / sa)^2 -
            0.5 * (dp / sp)^2
    }
    specs <- list(
        hyper_axis_spec("alpha", grid = alpha, log_scale = TRUE,
                        refinable = TRUE, extend = TRUE),
        hyper_axis_spec("phi_pos", grid = phi, log_scale = TRUE,
                        refinable = TRUE, extend = FALSE))
    modes <- list(alpha   = list(mode_u = log(a0),
                                 sd_u = sqrt(sa^2 + beta^2 * sp^2), tag = "log"),
                  phi_pos = list(mode_u = log(p0), sd_u = sp, tag = "log"))
    list(tg = tg, lp = lp, specs = specs, modes = modes, p0 = p_true, sp = sp,
         a0 = a0)
}

# The axis SD on the log scale, read off the weights a consistency pass left.
.cp_log_sd <- function(out, specs, axis) {
    lq <- .hyper_log_quad_weights(out$theta_grid, specs,
                                  refining = out$refining_axis)
    lw <- out$log_marginal + lq
    w <- exp(lw - max(lw)); w <- w / sum(w)
    x <- log(out$theta_grid[, axis])
    sqrt(sum(w * x^2) - sum(w * x)^2)
}

test_that("the axis farthest from its mode is refined first, and a fibre it empties is held", {
    f <- .cp_calluna_like()
    expect_identical(.hyper_consistency_order(c("alpha", "phi_pos"), f$tg, f$modes),
                     c("phi_pos", "alpha"))
    expect_identical(.hyper_consistency_order(c("alpha", "phi_pos"), f$tg, NULL),
                     c("alpha", "phi_pos"))
    calls <- 0L
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE) {
        calls <<- calls + 1L
        list(log_marginal = f$lp(new_cells))
    }
    out <- .hyper_consistency_pass(f$tg, f$lp(f$tg), NULL, rep("", nrow(f$tg)),
                                   f$specs, kernel_fn, axis_modes = f$modes)
    # The dispersion slice moves the whole posterior into its own fibre, so a
    # copy-scale slice in any base row would re-tile rows holding none of it.
    expect_identical(calls, 1L)
    expect_identical(out$info$axes, "phi_pos")
    expect_identical(out$info$held, "alpha")
    expect_false(any(out$refining_axis == "consistency_alpha"))
    expect_gte(out$info$ess_after, .nl_diag("axis_sd_ess"))
    # The ladder keeps the read on the mode: five points alone left the
    # outermost owning half a 36-SD gap and the mean 1.2 SDs high.
    expect_lt(abs(.cp_log_mean(out, f$specs, "phi_pos") - log(f$p0)),
              0.1 * f$sp)
})

test_that("points laid from a mode off the peak are closed where they are read", {
    # On the Calluna fit the dispersion axis peaked 1.1 SDs above the mode the
    # mode-find reached, and the point closing that side, laid for a density
    # centred on the mode, held a quarter of the axis.
    f <- .cp_calluna_like(offset = 1.1)
    calls <- 0L
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE) {
        calls <<- calls + 1L
        list(log_marginal = f$lp(new_cells))
    }
    out <- .hyper_consistency_pass(f$tg, f$lp(f$tg), NULL, rep("", nrow(f$tg)),
                                   f$specs, kernel_fn, axis_modes = f$modes)
    # The ladder, then one round closing the side the density sits against.
    expect_identical(calls, 2L)
    expect_lt(abs(.cp_log_mean(out, f$specs, "phi_pos") - log(f$p0)),
              0.1 * f$sp)
    expect_lt(abs(.cp_log_sd(out, f$specs, "phi_pos") / f$sp - 1), 0.1)
})

test_that("a slice is laid through the row of the joint mode", {
    # The copy scale correlates with the pinned dispersion, so at the declared
    # dispersion node 36 SDs off its best level is one grid step from its joint
    # mode, and that is where the base grid's heaviest cell sits.
    f <- .cp_calluna_like(beta = (1.25 * 0.15) / log(3.91 / 3.25))
    base_best <- f$tg[which.max(f$lp(f$tg)), "alpha"]
    expect_gt(abs(log(base_best / f$a0)), 0.15)
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE)
        list(log_marginal = f$lp(new_cells))
    out <- .hyper_consistency_pass(f$tg, f$lp(f$tg), NULL, rep("", nrow(f$tg)),
                                   f$specs, kernel_fn, axis_modes = f$modes)
    top <- out$theta_grid[which.max(out$log_marginal), ]
    expect_equal(unname(top[["alpha"]]), f$a0, tolerance = 1e-12)
    expect_lt(abs(.cp_log_mean(out, f$specs, "phi_pos") - log(f$p0)),
              0.1 * f$sp)
})

test_that("a collapsed axis with a known mode is resolved without bisecting to it", {
    grid <- c(1, 3.91, 15.3, 60)
    spec <- hyper_axis_spec("phi_pos", grid = grid, log_scale = TRUE,
                            refinable = TRUE, extend = FALSE)
    lp <- function(v) -0.5 * ((log(v) - log(4.2)) / 0.05)^2
    calls <- 0L
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE) {
        calls <<- calls + 1L
        list(log_marginal = lp(new_cells[, "phi_pos"]))
    }
    tg <- matrix(grid, ncol = 1L, dimnames = list(NULL, "phi_pos"))
    run <- function(modes) .hyper_consistency_pass(
        tg, lp(grid), NULL, rep("", 4L), list(spec), kernel_fn,
        axis_modes = modes)
    at_mode <- run(list(phi_pos = list(mode_u = log(4.2), sd_u = 0.05,
                                       tag = "log")))
    # The declared node 1.43 SDs below the mode stands in for the ladder's
    # point at 1, which leaves the spacing uneven and the ESS a hair under the
    # floor, so one bisection round follows; a ladder laid into an open gap is
    # resolved in the one call (the Calluna-shaped fixture above).
    expect_lte(calls, 2L)
    expect_gte(at_mode$info$ess_after, .nl_diag("axis_sd_ess"))
    expect_lt(abs(.cp_log_mean(at_mode, list(spec), "phi_pos") - log(4.2)),
              0.1 * 0.05)
    # Without the mode the pass bisects towards it and spends its node cap
    # without reaching the floor.
    calls <- 0L
    blind <- run(NULL)
    expect_gt(calls, 1L)
    expect_lt(blind$info$ess_after, .nl_diag("axis_sd_ess"))
})

test_that("an axis the grid left on one node reports the mode's Gaussian", {
    tg <- as.matrix(expand.grid(sigma = exp(1.1 + c(-1, 0, 1) * 0.19),
                                alpha = c(0, 0.27, 0.33)))
    w <- rep(0, nrow(tg)); w[tg[, "sigma"] == exp(1.1) & tg[, "alpha"] == 0.27] <- 1
    res <- list(theta_grid = tg, weights = w,
                theta_sd = c(sigma = 0.015, alpha = 0.02),
                theta_sd_ess = c(sigma = 1, alpha = 3.5),
                theta_sd_source = c(sigma = "stencil", alpha = "weighted"),
                theta_median = c(sigma = exp(1.1), alpha = 0.27),
                theta_ci_lo = c(sigma = 2.74, alpha = 0.25),
                theta_ci_hi = c(sigma = 3.27, alpha = 0.30),
                outer_mode_u = c(1.1, log(0.27)),
                outer_mode_cov_u = diag(c(0.0477, 0.0464)^2),
                outer_mode_axis_tags = c("log", "log"),
                outer_mode_axis_names = c("sigma", "alpha"))
    out <- .nl_mode_read_unresolved(res)
    z <- stats::qnorm(0.975)
    expect_identical(unname(out$theta_sd_source), c("mode", "weighted"))
    expect_equal(out$theta_sd[["sigma"]], exp(1.1) * sinh(0.0477))
    expect_equal(out$theta_ci_lo[["sigma"]], exp(1.1 - z * 0.0477))
    expect_equal(out$theta_ci_hi[["sigma"]], exp(1.1 + z * 0.0477))
    expect_equal(out$theta_median[["sigma"]], exp(1.1))
    # A resolved axis keeps the grid's read.
    expect_identical(out$theta_sd[["alpha"]], 0.02)
    expect_identical(out$theta_ci_lo[["alpha"]], 0.25)
    # A declared point mass the weights carry is the grid's atom split.
    res$theta_sd_ess[["alpha"]] <- 1
    res$weights[tg[, "alpha"] == 0][1] <- 0.2
    out <- .nl_mode_read_unresolved(res)
    expect_identical(out$theta_sd_source[["alpha"]], "weighted")
    # No mode, nothing to read.
    res$outer_mode_u <- NULL
    expect_identical(.nl_mode_read_unresolved(res), res)
})
