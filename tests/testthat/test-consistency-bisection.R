# The consistency pass places its points by where an axis marginal's mass
# sits, and iterates until the axis is resolved (gcol33/tulpa#858).
#
# The case the issue measured: a copy axis whose marginal holds two thirds of
# its mass on two adjacent declared nodes 2.34x apart. Points placed at a
# fraction of the modal parabola around the mean land inside that one gap, the
# ESS clears the floor, and nothing re-checks the result.

log_axis_spec <- function(lev, atom = FALSE) {
    hyper_axis_spec("alpha", grid = lev, log_scale = TRUE,
                    bounds = c(0, Inf), refinable = TRUE,
                    atom_mass = if (atom) 0.5)
}

test_that("the heavy gap is bisected on the integration coordinate, first", {
    lev <- c(0.1, 0.234, 0.548, 1.28, 3)
    p <- c(0.05, 0.12, 0.33, 0.33, 0.17)
    pts <- tulpa:::.hyper_propose_mass_bisection(log_axis_spec(lev), lev, log(p))
    # 0.548 / 1.28 carries 0.66: bisected in log, and returned first.
    expect_equal(pts[1L], sqrt(0.548 * 1.28))
    # Every gap whose two nodes carry at least 1 / 3 is bisected; the light
    # outer gap is not.
    expect_equal(sort(pts), sort(c(sqrt(0.548 * 1.28), sqrt(0.234 * 0.548),
                                   sqrt(1.28 * 3))))
})

test_that("a marginal peaking on an outermost level is not bisected", {
    # The span truncates it; bisecting towards the edge would only shrink the
    # edge level's box, one halving after another.
    lev <- c(0.5, 1, 1.5)
    spec <- hyper_axis_spec("alpha", grid = lev, log_scale = TRUE,
                            bounds = c(0, Inf), refinable = TRUE,
                            extend = FALSE)
    expect_length(tulpa:::.hyper_propose_mass_bisection(spec, lev,
                                                        c(0, -2, -5)), 0L)
    expect_length(tulpa:::.hyper_propose_mass_bisection(spec, lev,
                                                        c(-5, -2, 0)), 0L)

    target <- function(a) -8 * log(a)
    tg <- matrix(lev, ncol = 1L, dimnames = list(NULL, "alpha"))
    called <- 0L
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE,
                          screen = NULL) {
        called <<- called + 1L
        list(log_marginal = target(new_cells[, "alpha"]), extras = NULL)
    }
    out <- tulpa:::.hyper_consistency_pass(
        theta_grid = tg, log_marginal = target(lev), extras = NULL,
        refining_axis = rep("", 3L), specs = list(spec), kernel_fn = kernel_fn)
    expect_identical(out$n_added, 0L)
    expect_identical(called, 0L)
})

test_that("a point mass neither bounds a gap nor enters the shares", {
    lev <- c(0, 0.1, 0.3, 1)
    # The atom holds most of the mass; the continuum's own shares decide.
    lm <- log(c(0.9, 0.01, 0.06, 0.03))
    pts <- tulpa:::.hyper_propose_mass_bisection(log_axis_spec(lev, atom = TRUE),
                                                 lev, lm)
    expect_true(all(pts > 0.1))
    expect_equal(pts[1L], sqrt(0.3 * 1))
})

test_that("the node budget stops a marginal no bisection can resolve", {
    lev <- exp(seq(log(0.1), log(3), length.out = 5L))
    target <- function(a) -0.5 * ((log(a) - log(0.8)) / 0.002)^2
    specs <- list(log_axis_spec(lev))
    tg <- matrix(lev, ncol = 1L, dimnames = list(NULL, "alpha"))
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE,
                          screen = NULL) {
        list(log_marginal = target(new_cells[, "alpha"]), extras = NULL)
    }
    out <- tulpa:::.hyper_consistency_pass(
        theta_grid = tg, log_marginal = target(lev), extras = NULL,
        refining_axis = rep("", nrow(tg)), specs = specs, kernel_fn = kernel_fn,
        max_nodes = 3L)
    expect_identical(out$n_added, 3L)
    expect_identical(out$info$n_added, 3L)
})

skip_on_cran_budget()

test_that("the heaviest gap is bisected even when no gap reaches the share", {
    skip_on_cran_budget()
    lev <- exp(0:9)
    p <- rep(0.1, 10L); p[6L] <- 0.12
    pts <- tulpa:::.hyper_propose_mass_bisection(log_axis_spec(lev), lev, log(p),
                                                 min_ess = 2)
    expect_length(pts, 1L)
    expect_true(pts > lev[5L] && pts < lev[7L])
})

test_that("the pass re-reads the ESS and bisects until the axis is resolved", {
    skip_on_cran_budget()
    lev <- exp(seq(log(0.1), log(3), length.out = 5L))
    target <- function(a) -0.5 * ((log(a) - log(0.8)) / 0.18)^2
    specs <- list(log_axis_spec(lev))
    tg <- matrix(lev, ncol = 1L, dimnames = list(NULL, "alpha"))
    rounds <- 0L
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE,
                          screen = NULL) {
        rounds <<- rounds + 1L
        list(log_marginal = target(new_cells[, "alpha"]), extras = NULL)
    }
    out <- tulpa:::.hyper_consistency_pass(
        theta_grid = tg, log_marginal = target(lev), extras = NULL,
        refining_axis = rep("", nrow(tg)), specs = specs, kernel_fn = kernel_fn)
    floor <- tulpa:::.nl_diag("axis_sd_ess")
    expect_lt(out$info$ess_before, floor)
    expect_gte(out$info$ess_after, floor)
    expect_gt(rounds, 1L)
    expect_lte(out$n_added, tulpa:::.nl_diag("axis_refine_nodes"))
    # Every added node lies between two nodes the axis already had.
    added <- out$theta_grid[-seq_along(lev), "alpha"]
    expect_true(all(added > min(lev) & added < max(lev)))
    expect_true(all(out$refining_axis[-seq_along(lev)] == "alpha"))
})

# A right-skewed axis: a Gaussian core of SD 0.3 in log alpha and an
# exponential shelf above it, decaying at one nat per unit. Laid at 1.25 core
# SDs about the mode, the read places more than `edge_tail_mass` past the
# upper end.
skew_target <- function(a) {
    u <- log(a)
    log(0.9 * stats::dnorm(u, 0, 0.3) + 0.1 * exp(-pmax(u, 0)) * (u > 0))
}

skew_pass <- function(max_nodes = tulpa:::.nl_diag("axis_refine_nodes"),
                      ext_max = tulpa:::.nl_diag("edge_extend_max")) {
    lev <- exp(seq(-0.75, 0.75, by = 0.375))
    tg <- matrix(lev, ncol = 1L, dimnames = list(NULL, "alpha"))
    kernel_fn <- function(new_cells, warm_start = NULL, store_extras = FALSE,
                          screen = NULL) {
        list(log_marginal = skew_target(new_cells[, "alpha"]), extras = NULL)
    }
    out <- tulpa:::.hyper_consistency_pass(
        theta_grid = tg, log_marginal = skew_target(lev), extras = NULL,
        refining_axis = rep("", nrow(tg)), specs = list(log_axis_spec(lev)),
        kernel_fn = kernel_fn, max_nodes = max_nodes, ext_max = ext_max,
        axis_modes = list(alpha = list(mode_u = 0, sd_u = 0.3, tag = "log")))
    lm <- skew_target(out$theta_grid[, "alpha"])
    lq <- tulpa:::.hyper_log_quad_weights(out$theta_grid, list(log_axis_spec(lev)),
                                          refining = out$refining_axis,
                                          log_marginal = lm)
    out$tails <- tulpa:::.hyper_axis_tail_mass(
        out$theta_grid, lm, lm + lq, out$refining_axis, "alpha",
        log_axis_spec(lev))
    out$lev <- lev
    out
}

test_that("an axis laid at its mode still has its ends followed out", {
    out <- skew_pass()
    u <- log(out$theta_grid[, "alpha"])
    expect_gt(max(u), 0.75 + 1)
    expect_lte(out$tails[2L], tulpa:::.nl_diag("edge_tail_mass"))
    # Laid at its mode, the axis is never bisected inside its declared span.
    inside <- u[-seq_along(out$lev)]
    expect_false(any(inside > -0.75 & inside < 0.75))
})

test_that("levels past an end are bounded by the tail rule, not the level budget", {
    out <- skew_pass(max_nodes = 1L)
    above <- sum(log(out$theta_grid[, "alpha"]) > 0.75 + 1e-9)
    expect_gt(above, 1L)
    expect_lte(out$tails[2L], tulpa:::.nl_diag("edge_tail_mass"))
    capped <- skew_pass(max_nodes = 1L, ext_max = 2L)
    expect_lte(sum(log(capped$theta_grid[, "alpha"]) > 0.75 + 1e-9), 2L + 1L)
})

test_that("a level's side is read against the continuum span", {
    lev <- c(0, exp(c(-1, 0, 1)))
    sp <- log_axis_spec(lev, atom = TRUE)
    expect_identical(tulpa:::.hyper_level_side(c(exp(-2), 1.5, exp(2)), lev, sp),
                     c(1L, 0L, 2L))
})
