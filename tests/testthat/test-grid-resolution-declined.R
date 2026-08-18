# An outer axis whose resolution could not be scored (gcol33/tulpa#401).
#
# `h / sd` is the regime variable the within-cell construction's position
# sensitivity is governed by (gcol33/tulpa#357). The SD side comes from a
# 3-point parabola at the axis's modal cell, which withholds a number on five
# distinguishable conditions -- and one of them, the mode sitting on an END node,
# is not a missing measurement at all: it says the grid does not contain that
# axis's own posterior mode.
#
# Conflating the five into a bare NA let the whole-grid verdict be read off the
# axes that happened to score, so a fit could report `resolved` while an axis it
# never looked at held the modal mass at an endpoint, and the note it emitted
# named a DIFFERENT, healthy axis and told the reader to add nodes there.

# Two axes on a tensor grid: `A` interior and finely resolved, `B` rising
# monotonically to its top node.
.gres_two_axis <- function(n = 9L, sd_a = 0.6, slope_b = 3) {
    va <- seq(-1, 1, length.out = n)
    vb <- seq(0, 2, length.out = n)
    g  <- as.matrix(expand.grid(A = va, B = vb))
    list(tg = g, lm = -0.5 * (g[, "A"] / sd_a)^2 + slope_b * g[, "B"])
}

.gres_fit <- function(rs, railed = character(0)) {
    list(outer_grid_h_over_sd = rs$h_over_sd,
         outer_grid_cell_width = rs$h,
         outer_grid_axis_sd = rs$sd,
         outer_grid_resolution_declined = rs$declined,
         outer_grid_railed_axes = railed)
}

test_that("each condition that withholds an axis SD names itself", {
    v <- c(0.5, 1, 2, 4, 8)
    # Mode interior: a number, and no reason.
    ok <- .nl_laplace_at_mode_sd_axis(v, c(-3, -1, 0, -1, -3))
    expect_true(is.finite(ok))
    expect_identical(.nl_axis_sd_reason(ok), NA_character_)

    # Mode on either end node.
    for (lm in list(c(0, -1, -2, -3, -4), c(-4, -3, -2, -1, 0))) {
        d <- .nl_laplace_at_mode_sd_axis(v, lm)
        expect_true(is.na(d))
        expect_identical(.nl_axis_sd_reason(d), "mode_at_edge")
    }

    # Fewer than three nodes: no parabola exists.
    d <- .nl_laplace_at_mode_sd_axis(c(1, 2), c(-1, 0))
    expect_identical(.nl_axis_sd_reason(d), "too_few_nodes")

    # Concave UP at the mode: a curvature the Gaussian read cannot use.
    d <- .nl_laplace_at_mode_sd_axis(v, c(-3, -1, 0, -1, -3), log_axis = FALSE)
    expect_true(is.finite(d))
    d <- .nl_laplace_at_mode_sd_axis(c(1, 2, 3), c(-1, 0, -1) * -1 + c(0, 0, 0))
    expect_identical(.nl_axis_sd_reason(d), "mode_at_edge")

    # A coordinate the transform sends to non-finite values.
    d <- .nl_laplace_at_mode_sd_axis(c(0, 1, 2), c(-1, 0, -1), log_axis = TRUE)
    expect_identical(.nl_axis_sd_reason(d), "coord_not_finite")

    # Every reason emitted is in the closed vocabulary.
    expect_true(all(.NL_AXIS_SD_REASONS %in% .NL_AXIS_RESOLUTION_REASONS))
})

test_that("the reason survives the axis-resolution read, per axis", {
    d  <- .gres_two_axis()
    rs <- .nl_axis_resolution(d$tg, d$lm, rep("", nrow(d$tg)), NULL)

    expect_true(is.finite(rs$h_over_sd[["A"]]))
    expect_true(is.na(rs$h_over_sd[["B"]]))
    expect_identical(rs$declined[["A"]], NA_character_)
    expect_identical(rs$declined[["B"]], "mode_at_edge")
    # The names line up with the axes rather than travelling positionally.
    expect_identical(names(rs$declined), colnames(d$tg))
    # `h` is still measured on the declined axis -- it is the SD that is missing,
    # and the spacing is a property of the design alone.
    expect_true(is.finite(rs$h[["B"]]))
})

test_that("a whole-grid verdict is withheld when an axis went unscored", {
    d  <- .gres_two_axis()
    rs <- .nl_axis_resolution(d$tg, d$lm, rep("", nrow(d$tg)), NULL)
    r  <- .tulpa_grid_resolution(.gres_fit(rs))

    # A on its own is comfortably inside the resolved band, so reading the
    # verdict off the scored axes alone is what returned TRUE here.
    expect_lt(rs$h_over_sd[["A"]], .nl_diag("grid_resolved"))
    expect_false(r$resolved)
    expect_identical(r$unscored, "B")
    expect_identical(r$declined[["B"]], "mode_at_edge")
    expect_identical(r$n_scored, 1L)
    expect_identical(r$n_axes, 2L)
    # `coarsest` still names the coarsest SCORED axis; it is no longer the whole
    # of what the reader is told.
    expect_identical(r$coarsest, "A")
})

test_that("the note names the unscored axis before the coarsest scored one", {
    d  <- .gres_two_axis()
    rs <- .nl_axis_resolution(d$tg, d$lm, rep("", nrow(d$tg)), NULL)
    n  <- .tulpa_grid_resolution_note(.tulpa_grid_resolution(
        .gres_fit(rs, railed = "B:upper")))

    expect_true(length(n) >= 2L)
    expect_match(n[1L], "could not be scored on B \\(mode_at_edge\\)")
    expect_match(n[1L], "1 of 2 axes")
    expect_match(n[2L], "does not contain its own posterior mode on B:upper")
    # And it does not tell the reader to add nodes to the healthy axis: A is
    # inside the resolved band, so the coarse-axis line does not fire at all.
    expect_false(any(grepl("coarser than its own posterior", n)))

    # With no rail recorded the rail line is absent, and the unscored line is not.
    n2 <- .tulpa_grid_resolution_note(.tulpa_grid_resolution(.gres_fit(rs)))
    expect_false(any(grepl("does not contain its own posterior mode", n2)))
    expect_match(n2[1L], "could not be scored")
})

test_that("a fully resolved grid still reports nothing", {
    va <- seq(-1, 1, length.out = 9L)
    g  <- as.matrix(expand.grid(A = va, B = va))
    lm <- -0.5 * ((g[, "A"] / 0.6)^2 + (g[, "B"] / 0.6)^2)
    rs <- .nl_axis_resolution(g, lm, rep("", nrow(g)), NULL)
    r  <- .tulpa_grid_resolution(.gres_fit(rs))

    expect_true(all(is.finite(rs$h_over_sd)))
    expect_true(all(is.na(rs$declined)))
    expect_true(r$resolved)
    expect_identical(r$n_scored, r$n_axes)
    expect_null(.tulpa_grid_resolution_note(r))
})

test_that("a grid where NOTHING scored says so instead of returning nothing", {
    # Both axes rise to their top node: no axis carries a ratio at all. The old
    # `!any(is.finite(r))` early return made this indistinguishable from a fit
    # that records no resolution -- the emptiest possible grid reporting the
    # cleanest possible result.
    va <- seq(0, 2, length.out = 9L)
    g  <- as.matrix(expand.grid(A = va, B = va))
    lm <- 3 * g[, "A"] + 3 * g[, "B"]
    rs <- .nl_axis_resolution(g, lm, rep("", nrow(g)), NULL)
    r  <- .tulpa_grid_resolution(.gres_fit(rs))

    expect_false(is.null(r))
    expect_identical(r$n_scored, 0L)
    expect_setequal(r$unscored, c("A", "B"))
    expect_true(all(r$declined == "mode_at_edge"))
    expect_false(r$resolved)
    expect_true(is.na(r$max))
    expect_true(is.na(r$coarsest))
    expect_match(.tulpa_grid_resolution_note(r)[1L], "could not be scored")
})

test_that("a fit carrying no resolution at all is still NULL", {
    expect_null(.tulpa_grid_resolution(list()))
    expect_null(.tulpa_grid_resolution(list(outer_grid_h_over_sd = numeric(0))))
})

test_that("the provenance attach stamps the rail report unconditionally", {
    # The rail report's only attach points used to be inside the registry
    # rescue, which does not run on a grid the caller pinned -- so the one
    # placement the engine leaves alone by construction was the one that never
    # said so. It is now stamped where the resolution is, which every nested
    # path passes through.
    vals <- c(0.2, 0.5, 0.8, 0.95)
    w    <- c(0.001, 0.02, 0.35, 0.63)
    tg   <- matrix(vals, ncol = 1L, dimnames = list(NULL, "tau"))
    res  <- list(theta_grid = tg, theta_names = "tau", weights = w,
                 log_marginal = log(w), integration = "grid",
                 weight_kind = "density", refining_axis = rep("", length(vals)))
    qs <- .nl_axis_quantiles(tg, res$log_marginal, res$refining_axis,
                             weights = w, support = "density")
    out <- .nl_attach_interval_provenance(res, qs, tg, NULL)

    expect_identical(out$outer_grid_railed_axes, "tau:upper")
    expect_identical(out$outer_grid_resolution_declined[["tau"]], "mode_at_edge")

    # An already-stamped report is not recomputed over: a rescue that ran keeps
    # the value it recorded, including an empty one.
    kept <- .nl_attach_interval_provenance(
        c(res, list(outer_grid_railed_axes = character(0))), qs, tg, NULL)
    expect_identical(kept$outer_grid_railed_axes, character(0))
})

test_that("a real caller-pinned fit carries both fields", {
    skip_on_cran()
    set.seed(4L)
    n_s <- 20L
    nbr <- lapply(seq_len(n_s),
                  function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
    nn  <- vapply(nbr, length, integer(1))
    n   <- 600L
    idx <- sample.int(n_s, n, TRUE)
    f   <- cumsum(rnorm(n_s)); f <- f - mean(f)
    x   <- rnorm(n)
    y   <- rnorm(n, 0.3 + 0.8 * x + f[idx], 0.5)

    fit <- suppressWarnings(tulpa_nested_laplace(
        y = y, n_trials = rep(1L, n), X = cbind(1, x),
        prior = list(type = "icar", n_spatial_units = n_s,
                     adj_row_ptr = as.integer(c(0L, cumsum(nn))),
                     adj_col_idx = as.integer(unlist(nbr)) - 1L,
                     n_neighbors = as.integer(nn),
                     tau_grid = 1 / c(0.4, 0.8, 1.6, 3.2)^2,
                     spatial_idx = idx),
        family = "gaussian", phi = 0.25,
        control = list(max_iter = 40L, tol = 1e-6, diagnose_k = FALSE,
                       diagnose_skew = FALSE)))

    expect_false(is.null(fit$outer_grid_railed_axes))
    expect_true(is.character(fit$outer_grid_railed_axes))
    expect_false(is.null(fit$outer_grid_resolution_declined))
    expect_identical(names(fit$outer_grid_resolution_declined),
                     names(fit$outer_grid_h_over_sd))
    # An axis carries a ratio or a reason, never neither and never both.
    scored <- is.finite(fit$outer_grid_h_over_sd)
    expect_true(all(is.na(fit$outer_grid_resolution_declined[scored])))
    expect_true(all(!is.na(fit$outer_grid_resolution_declined[!scored])))

    r <- .tulpa_grid_resolution(fit)
    expect_identical(r$railed, fit$outer_grid_railed_axes)
})
