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
    # Nearest the mode first: five one SD apart, then the points closing the
    # gaps the outermost node on each side would otherwise read across.
    k <- (log(pts) - log(4.2)) / 0.02
    expect_equal(k[1:5], c(0, -1, 1, -2, 2))
    expect_true(all(abs(k[-(1:5)]) > 2))
    expect_identical(order(abs(k)), seq_along(k))
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
.cp_calluna_like <- function() {
    a0 <- 0.274; sa <- 0.0464; p0 <- 3.25; sp <- 0.0053
    alpha <- a0 * exp(c(-2, -1, 0, 1, 2) * 1.25 * 0.15)
    phi <- c(1, 3.91, 15.3, 60)
    tg <- as.matrix(expand.grid(alpha = alpha, phi_pos = phi))
    lp <- function(g) -0.5 * ((log(g[, "alpha"]) - log(a0)) / sa)^2 -
        0.5 * ((log(g[, "phi_pos"]) - log(p0)) / sp)^2
    specs <- list(
        hyper_axis_spec("alpha", grid = alpha, log_scale = TRUE,
                        refinable = TRUE, extend = TRUE),
        hyper_axis_spec("phi_pos", grid = phi, log_scale = TRUE,
                        refinable = TRUE, extend = FALSE))
    modes <- list(alpha   = list(mode_u = log(a0), sd_u = sa, tag = "log"),
                  phi_pos = list(mode_u = log(p0), sd_u = sp, tag = "log"))
    list(tg = tg, lp = lp, specs = specs, modes = modes, p0 = p0, sp = sp)
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
    # The points closing the gaps keep the read on the mode: without them the
    # outermost point owned half a 36-SD gap and the mean sat 1.2 SDs high.
    expect_lt(abs(.cp_log_mean(out, f$specs, "phi_pos") - log(f$p0)),
              0.1 * f$sp)
})

test_that("a collapsed axis with a known mode is resolved in one round", {
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
    expect_identical(calls, 1L)
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
