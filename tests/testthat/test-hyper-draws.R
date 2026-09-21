# The hyperparameter half of a nested-Laplace posterior draw (gcol33/tulpa#823).
#
# The arbiter is not "the draws look continuous". It is that the continuization
# is the cell-conditional of the read the fit already reports, so the draws'
# own quantiles reproduce `theta_ci_lo` / `theta_median` / `theta_ci_hi` --
# numbers produced by a completely separate code path (`.nl_box_quantile()` /
# `.nl_wtd_quantile()` inverting a CDF, against `runif()` inside a box). If the
# jitter used a different geometry from the read, those two disagree.
#
# Section 4 is the issue's own instrument, on a fixture whose posterior is
# EXACT: the grid's `log_marginal` is the analytic log-likelihood plus prior, so
# nothing about the inner Laplace is under test and the exact posterior is
# available as a reference arm.

# A nested-Laplace-shaped fit whose outer grid is exactly the supplied axes,
# carrying the engine's own cell measure and weights.
hd_fit <- function(axes, log_marginal, within = "box_uniform",
                   integration = "grid", type = "bym2") {
  tg <- as.matrix(expand.grid(axes))
  lm <- log_marginal(tg)
  specs <- tulpa:::.joint_axis_specs_from_grid(tg)
  lq <- tulpa:::.hyper_log_quad_weights(tg, specs)
  res <- list(theta_grid = tg, log_marginal = lm, log_quad = lq,
              integration = integration, prior = list(type = type),
              blocks = NULL, axis_offsets = NULL)
  res$weights <- tulpa:::.nl_normalise_weights_safe(lm, "outer grid",
                                                    log_quad = lq)
  res <- tulpa:::.nl_posterior_moments(res, type, within = within)
  res$within_cell_requested <- within
  structure(res, class = c("tulpa_nested_laplace", "list", "tulpa_fit"))
}

hd_two_axis <- function(within = "box_uniform") {
  hd_fit(
    list(tau = exp(seq(log(0.2), log(3), length.out = 7)),
         rho = seq(0.1, 0.9, length.out = 5)),
    function(tg) -0.5 * (log(tg[, "tau"]) - log(0.8))^2 / 0.3^2 -
                  0.5 * (tg[, "rho"] - 0.55)^2 / 0.2^2,
    within = within)
}


# ---- 1. The draws ARE the reported marginal -------------------------------

test_that("jittered hyperparameter draws reproduce the fit's own quantiles", {
  skip_on_cran()
  for (wc in c("box_uniform", "chord")) {
    fit <- hd_two_axis(wc)
    set.seed(823)
    th <- tulpa_hyper_draws(fit, n = 4e5L)

    expect_equal(dim(th), c(4e5L, 2L))
    expect_identical(colnames(th), c("tau", "rho"))
    expect_identical(unname(attr(th, "within_cell")), rep(wc, 2L))
    expect_true(all(is.na(attr(th, "within_cell_declined"))))

    for (ax in colnames(th)) {
      q <- unname(stats::quantile(th[, ax], c(0.025, 0.5, 0.975)))
      rep_q <- c(fit$theta_ci_lo[[ax]], fit$theta_median[[ax]],
                 fit$theta_ci_hi[[ax]])
      # The read inverts a CDF; the draws sample it. At 4e5 draws the Monte
      # Carlo error on a quantile of an axis whose support is ~1 wide is ~1e-3.
      expect_lt(max(abs(q - rep_q)), 5e-3)
    }
  }
})

test_that("the atom is gone: draws take more values than the grid has nodes", {
  fit <- hd_two_axis()
  set.seed(1)
  th <- tulpa_hyper_draws(fit, n = 5000L)
  # 7 x 5 nodes; every draw is its own value.
  expect_equal(length(unique(th[, "tau"])), 5000L)
  expect_equal(length(unique(th[, "rho"])), 5000L)
  # And each draw sits inside its own cell's box, never outside the axis's
  # extended support.
  ed <- tulpa:::.nl_box_edges(sort(unique(fit$theta_grid[, "tau"])), "positive")
  expect_true(all(th[, "tau"] >= ed[1L] & th[, "tau"] <= ed[length(ed)]))
})


# ---- 2. The geometry is the read's own --------------------------------------

test_that("a box draw stays in the box its own cell owns", {
  fit <- hd_two_axis()
  set.seed(2)
  cells <- rep(seq_len(nrow(fit$theta_grid)), each = 40L)
  th <- tulpa_hyper_draws(fit, cells = cells)
  expect_identical(attr(th, "cells"), as.integer(cells))
  for (ax in c("tau", "rho")) {
    uv <- sort(unique(fit$theta_grid[, ax]))
    dom <- if (identical(ax, "tau")) "positive" else "unit"
    e <- tulpa:::.nl_box_edges(uv, dom)
    k <- match(fit$theta_grid[cells, ax], uv)
    expect_true(all(th[, ax] >= e[k] & th[, ax] <= e[k + 1L]))
  }
})

test_that("a chord draw stays between its cell's two neighbours", {
  fit <- hd_two_axis("chord")
  set.seed(3)
  cells <- rep(seq_len(nrow(fit$theta_grid)), each = 40L)
  th <- tulpa_hyper_draws(fit, cells = cells, within = "chord")
  for (ax in c("tau", "rho")) {
    uv <- sort(unique(fit$theta_grid[, ax]))
    dom <- if (identical(ax, "tau")) "positive" else "unit"
    e <- tulpa:::.nl_cell_edges(uv, dom)
    n <- length(uv)
    lo <- c(e[1L], uv[-n]); hi <- c(uv[-1L], e[2L])
    k <- match(fit$theta_grid[cells, ax], uv)
    expect_true(all(th[, ax] >= lo[k] & th[, ax] <= hi[k]))
  }
})

test_that("the outer half-cell is reachable, and only under an extend support", {
  fit <- hd_two_axis()
  top <- max(fit$theta_grid[, "tau"])
  set.seed(4)
  cells <- which(fit$theta_grid[, "tau"] == top)
  th <- tulpa_hyper_draws(fit, cells = rep(cells, each = 200L))
  expect_true(any(th[, "tau"] > top))

  # A `sample` support clamps, so the outermost cell's outward half stays on
  # its coordinate and no draw passes the extreme node.
  samp <- fit
  samp$integration <- "sample"
  set.seed(4)
  ths <- tulpa_hyper_draws(samp, cells = rep(cells, each = 200L))
  expect_true(all(ths[, "tau"] <= top))
  expect_identical(unname(attr(ths, "within_cell")), rep("chord", 2L))
  expect_identical(unname(attr(ths, "within_cell_declined")),
                   rep("support_sample", 2L))
})


# ---- 3. Declines, and the draws attribute -----------------------------------

test_that("a moment-rule support declines rather than inventing a box", {
  fit <- hd_two_axis()
  fit$integration <- "ccd"
  set.seed(5)
  cells <- rep(seq_len(nrow(fit$theta_grid)), each = 3L)
  th <- tulpa_hyper_draws(fit, cells = cells)
  # A CCD design's node positions carry no mass, so there is no within-cell
  # density to sample: the node value is returned unchanged.
  expect_equal(unname(th[, "tau"]), unname(fit$theta_grid[cells, "tau"]))
  expect_true(all(is.na(attr(th, "within_cell"))))
  expect_identical(unname(attr(th, "within_cell_declined")),
                   rep("support_moment_rule", 2L))
})

test_that("a single-node axis is left alone and says so", {
  fit <- hd_fit(list(tau = exp(seq(log(0.2), log(3), length.out = 5)),
                     rho = 0.5),
                function(tg) -0.5 * (log(tg[, "tau"]))^2)
  set.seed(6)
  th <- tulpa_hyper_draws(fit, n = 200L)
  expect_true(all(th[, "rho"] == 0.5))
  expect_true(is.na(attr(th, "within_cell")[["rho"]]))
  expect_identical(attr(th, "within_cell_declined")[["rho"]], "single_node")
  expect_identical(attr(th, "within_cell")[["tau"]], "box_uniform")
})

test_that("tulpa_posterior_draws() carries the hyperparameter half", {
  fit <- hd_two_axis()
  p <- 2L
  mu <- cbind(seq_len(nrow(fit$theta_grid)) / 10, 1)
  fit$grid_modes <- lapply(seq_len(nrow(mu)), function(k) mu[k, ])
  fit$grid_hessians <- rep(list(diag(p)), nrow(mu))
  fit$modes <- mu
  fit$n_fixed <- p
  fit$fixed_names <- c("b1", "b2")

  # The attach must not move the RNG: the latent draws and everything a caller
  # does after them are bit-for-bit what they were before the attribute existed.
  set.seed(7)
  dr <- tulpa_posterior_draws(fit, n = 500L)
  after <- stats::runif(3L)
  set.seed(7)
  bare <- tulpa:::.nl_mixture_draw(
    w = tulpa:::.nested_fixed_moments(fit)$w,
    cell_id = tulpa:::.nested_fixed_moments(fit)$keep, n = 500L, p = 2L,
    draw_cell = function(i, n_i) matrix(stats::rnorm(n_i * 2L), n_i, 2L))
  expect_identical(attr(bare, "cells"), attr(dr, "cells"))
  expect_identical(stats::runif(3L), after)

  th <- attr(dr, "theta")
  expect_equal(dim(th), c(500L, 2L))
  expect_identical(attr(th, "cells"), attr(dr, "cells"))
  # The attribute is what `tulpa_hyper_draws()` returns for the same cells,
  # so a consumer reading either gets one object.
  set.seed(99)
  a <- tulpa_hyper_draws(fit, cells = attr(dr, "cells"))
  set.seed(99)
  b <- tulpa_hyper_draws(fit, cells = attr(dr, "cells"))
  expect_identical(a, b)
  expect_identical(attr(dr, "theta_within_cell"), attr(th, "within_cell"))
})

test_that("bad cells are rejected and a gridless fit returns NULL", {
  fit <- hd_two_axis()
  expect_error(tulpa_hyper_draws(fit, cells = c(1L, nrow(fit$theta_grid) + 1L)),
               "1-based outer-grid cell indices")
  expect_null(tulpa_hyper_draws(list(theta_grid = NULL)))
})


# ---- 4. What the atom costs, against an exact posterior ---------------------

# A scalar-hyperparameter fixture whose grid posterior is EXACT: y_i ~ N(0,
# sigma^2) with log sigma ~ N(0, s^2), the grid's `log_marginal` the analytic
# log-likelihood plus log prior. The outer read is the only thing under test,
# and the exact posterior CDF is the reference arm -- which matters, because a
# weakly identified hyperparameter makes the PIT correlate with the truth under
# the EXACT posterior too (measured: Spearman +0.490 exact against +0.472
# atom at n = 5). A truth-correlated PIT is therefore not on its own evidence
# of this defect; what the atom provably breaks is UNIFORMITY, and the rate at
# which it pins the PIT to exactly 0 or 1 is exactly the rate at which the
# truth falls outside the node range.
hd_sbc <- function(n_obs, prior_sd, n_nodes, n_sim, n_draw = 1500L,
                   seed = 823L) {
  lpost <- function(ls, S) {
    -n_obs * ls - S / (2 * exp(2 * ls)) - 0.5 * (ls / prior_sd)^2
  }
  grid_fit <- function(S) {
    m <- stats::optimize(function(ls) -lpost(ls, S), c(-8, 8))$minimum
    h <- 1e-3
    d2 <- (lpost(m + h, S) - 2 * lpost(m, S) + lpost(m - h, S)) / h^2
    sd <- 1 / sqrt(max(-d2, 1e-8))
    nodes <- m + sd * seq(-2.5, 2.5, length.out = n_nodes)
    hd_fit(list(sigma = exp(nodes)), function(tg) lpost(log(tg[, "sigma"]), S),
           type = "icar")
  }
  exact <- function(S, t) {
    g <- seq(-12, 12, length.out = 20001)
    lp <- lpost(g, S); d <- exp(lp - max(lp))
    stats::approx(g, cumsum(d) / sum(d), xout = log(t), rule = 2)$y
  }
  set.seed(seed)
  truth <- exp(stats::rnorm(n_sim, 0, prior_sd))
  out <- list(atom = numeric(n_sim), box = numeric(n_sim),
              exact = numeric(n_sim), outside = logical(n_sim))
  for (s in seq_len(n_sim)) {
    y <- stats::rnorm(n_obs, 0, truth[s]); S <- sum(y^2)
    fit <- grid_fit(S)
    cells <- tulpa:::.nl_mixture_cells(fit$weights, seq_along(fit$weights),
                                       n_draw)$row_cells
    nodes <- fit$theta_grid[, 1L]
    out$outside[s] <- truth[s] < min(nodes) || truth[s] > max(nodes)
    out$atom[s]  <- mean(fit$theta_grid[cells, 1L] <= truth[s])
    out$box[s]   <- mean(tulpa_hyper_draws(fit, cells = cells)[, 1L] <=
                           truth[s])
    out$exact[s] <- exact(S, truth[s])
  }
  out
}

test_that("the node atom pins the PIT exactly where the truth leaves the grid", {
  skip_on_cran()
  r <- hd_sbc(n_obs = 20L, prior_sd = 0.5, n_nodes = 5L, n_sim = 150L)
  at_end <- function(p) mean(p <= 1e-3 | p >= 1 - 1e-3)
  # The identity, not an inequality: under the atom read a truth outside the
  # node range has NO draw on its far side, so its PIT is exactly 0 or 1.
  expect_equal(at_end(r$atom), mean(r$outside))
  # The box read's outer half-cell reaches past the outermost node, so it pins
  # no more of the PIT than the exact posterior does.
  expect_lte(at_end(r$box), at_end(r$exact) + 1e-12)
})

test_that("the continuized PIT is uniform where the node atom's is not", {
  skip_if_not_slow()
  r <- hd_sbc(n_obs = 20L, prior_sd = 0.5, n_nodes = 5L, n_sim = 400L)
  ks <- function(p) suppressWarnings(stats::ks.test(p, "punif")$p.value)
  # The atom read is a discrete distribution on the node CDF's steps: 400
  # replicates take ~185 distinct PIT values against 400 for the exact
  # posterior, and its uniformity test is rejected outright.
  expect_lt(length(unique(round(r$atom, 6))), 0.6 * length(r$atom))
  expect_lt(ks(r$atom), 1e-4)
  # The fix restores it, on the same replicates and the same cells.
  expect_gt(length(unique(round(r$box, 6))), 0.8 * length(r$box))
  expect_gt(ks(r$box), 0.05)
  expect_gt(ks(r$exact), 0.05)
})


# ---- 5. Coupling across axes, and what does move a grid-route read ----------

test_that("coupling the axes inside a cell cannot move any axis's marginal", {
  # gcol33/tulpa#853 asks for the two axes of a cell to be continuized JOINTLY,
  # on the reading that independently jittered marginals manufacture variance
  # in a ratio built from them. The within-cell read is affine in its uniform,
  # so every way of tying the axes' uniforms together -- independent, a copula
  # at the cell's own correlation, or one uniform shared outright -- carries
  # the SAME per-axis marginal and differs only in the joint. Coupling is
  # therefore free to choose and cannot repair a marginal.
  #
  # What it does to the ratio was measured rather than assumed
  # (`dev_notes/issue853/RESULTS.md` section 2, 300 replicates against an exact
  # posterior): a copula at the grid's own correlation is indistinguishable
  # from the shipped jitter to three decimals, and a shared uniform reproduces
  # the grid-node atom's own failure, because a ratio of two comonotone draws
  # cancels the jitter. This pins the half of that which is cheap to assert.
  fit <- hd_two_axis()
  set.seed(853)
  # Allocated by WEIGHT, so the draws are the fit's own mixture and their
  # quantiles are comparable to the interval it reports. The identity is
  # structural and the draw count is only Monte Carlo error on it, so CRAN
  # reads it at a size that stays sub-second.
  n_draw <- if (cran_fixture()) 4e4L else 4e5L
  cells <- tulpa:::.nl_mixture_cells(fit$weights, seq_along(fit$weights),
                                     n_draw)$row_cells
  th <- tulpa_hyper_draws(fit, cells = cells)

  u <- stats::runif(length(cells))          # ONE uniform, shared by both axes
  for (ax in c("tau", "rho")) {
    uv <- sort(unique(fit$theta_grid[, ax]))
    e <- tulpa:::.nl_box_edges(uv, if (identical(ax, "tau")) "positive" else "unit")
    k <- match(fit$theta_grid[cells, ax], uv)
    comono <- e[k] + u * (e[k + 1L] - e[k])
    probs <- c(0.025, 0.5, 0.975)
    scale <- diff(range(fit$theta_grid[, ax]))
    rep_q <- c(fit$theta_ci_lo[[ax]], fit$theta_median[[ax]],
               fit$theta_ci_hi[[ax]])
    if (!all(is.finite(rep_q))) next
    # Both constructions reproduce the fit's OWN reported interval, so neither
    # can be the one that fixes it.
    expect_lt(max(abs(unname(stats::quantile(th[, ax], probs)) - rep_q)) / scale,
              0.05)
    expect_lt(max(abs(unname(stats::quantile(comono, probs)) - rep_q)) / scale,
              0.05)
  }
})

test_that("a grid that does not contain its own mode is named as such", {
  # The conditions that DO move with a miscalibrated grid-route read are grid
  # extent -- mass in the outermost cell, and an axis whose nodes do not reach
  # its own posterior mode -- not the within-cell construction
  # (`dev_notes/issue853/RESULTS.md` sections 3 and 4: at cell-width / posterior-SD
  # held near 4, the read runs from tracking the exact posterior to three times
  # its KS as the edge mass goes 0.000 -> 0.803). Both are already recorded on
  # the fit and both are already branches of the note; this holds them there.
  railed <- hd_fit(
    list(tau = exp(seq(log(0.2), log(3), length.out = 7)),
         rho = seq(0.1, 0.9, length.out = 5)),
    # Monotone in tau over the whole axis: the modal mass sits on the top node.
    function(tg) 4 * log(tg[, "tau"]) -
                 0.5 * (tg[, "rho"] - 0.55)^2 / 0.2^2)
  rs <- tulpa:::.tulpa_grid_resolution(railed)
  expect_false(is.null(rs))
  expect_true("tau" %in% c(rs$railed, rs$unscored))
  note <- tulpa:::.tulpa_grid_resolution_note(rs)
  expect_true(any(grepl("tau", note, fixed = TRUE)))

  # The same shape with the mass inside the axis keeps that branch quiet.
  inside <- hd_two_axis()
  rs2 <- tulpa:::.tulpa_grid_resolution(inside)
  expect_length(rs2$railed, 0L)
})


# ---- 6. The occu_cover-shaped joint fit, end to end -------------------------

test_that("a joint fit's draws carry a continuized sigma and alpha", {
  skip_on_cran()
  # The shape the issue was reported on: a binomial occupancy arm, a gaussian
  # cover arm, and a copy coefficient `alpha` -- three coarse outer axes read
  # off the node coordinate by every consumer of the draws.
  fit <- build_icar_joint_fit(sigma_grid = c(0.4, 0.7, 1.0, 1.5),
                              alpha_grid = c(0.2, 0.5, 0.8))
  set.seed(823)
  dr <- tulpa_posterior_draws(fit, idx = 1L, n = 20000L)
  th <- attr(dr, "theta")
  expect_null(attr(dr, "theta_declined"))
  expect_true(is.matrix(th))
  expect_identical(nrow(th), nrow(dr))
  expect_identical(attr(th, "cells"), attr(dr, "cells"))

  tg <- tulpa:::.nl_theta_matrix(fit)
  for (ax in colnames(th)) {
    nodes <- length(unique(tg[attr(dr, "cells"), ax]))
    if (nodes < 2L) next
    # The atom: as many distinct values as the axis has nodes. The fix: one per
    # draw.
    expect_lte(nodes, 5L)
    expect_gt(length(unique(th[, ax])), 0.99 * nrow(th))
    q <- unname(stats::quantile(th[, ax], c(0.025, 0.5, 0.975)))
    rep_q <- c(fit$theta_ci_lo[[ax]], fit$theta_median[[ax]],
               fit$theta_ci_hi[[ax]])
    if (all(is.finite(rep_q))) {
      expect_lt(max(abs(q - rep_q)) / diff(range(tg[, ax])), 0.05)
    }
  }
})


# ---- 7. An axis carrying a declared point mass (gcol33/tulpa#854) -----------
#
# The copy scale is not a continuum. It is the declared "no coupling" model at
# `alpha = 0`, carrying `.TULPA_COPY_ATOM_MASS` whatever the node count, plus a
# log continuum on (0, Inf) -- and three readers already split it there on the
# one rule `.hyper_axis_scale()` states: the measure that integrates the axis
# (`.hyper_axis_measure()`), the support it reports (`.hyper_axis_support()`)
# and the prior that weighs the level (`.hyper_is_atom_level()`). The reporting
# partition was the fourth reader and the only one that did not, because it
# took the axis's support from the outer Pareto-k PROPOSAL's tag (`identity`,
# which a proposal needs in order to reach zero at all).
#
# So the level got a cell, half a node spacing wide, reaching BELOW the axis's
# own support -- and the two failures the issue measured follow from one
# geometry: the level's posterior mass is reproduced by no draw, and draws leave
# the support. The arbiter here is the same one section 1 uses, the read and the
# draws being one construction, applied to the axis where they were two.

hd_copy <- function(within = "box_uniform", lp_alpha = NULL) {
  lp_alpha <- lp_alpha %||% function(a) -0.5 * (a - 0.5)^2 / 0.3^2
  hd_fit(
    list(alpha = c(0, 0.25, 0.5, 1.0),
         sigma = exp(seq(log(0.4), log(1.5), length.out = 4))),
    function(tg) -0.5 * (log(tg[, "sigma"]) - log(0.9))^2 / 0.4^2 +
                  lp_alpha(tg[, "alpha"]),
    within = within)
}

test_that("a declared point mass is named as one, and its continuum is positive", {
  fit <- hd_copy()
  geo <- tulpa:::.nl_axis_geometry(fit)
  nms <- colnames(tulpa:::.nl_theta_matrix(fit))
  # Both axes are log-scale, so both declare the level; only `alpha` carries a
  # grid node on it. The copy axis's PROPOSAL coordinate stays `identity` --
  # that tag has to reach zero -- and is a separate statement from its support.
  expect_identical(geo$domain, rep("positive", length(nms)))
  expect_identical(geo$atom, rep(0, length(nms)))
  expect_identical(
    unname(tulpa:::.joint_axis_tags_raw(list(theta_grid = fit$theta_grid,
                                             prior = list(type = "bym2")))),
    c("identity", "log"))
})

test_that("a level the grid does not carry leaves the read byte-identical", {
  # The no-op half: `sigma` declares the same level and has no node on it, so
  # every read has to return exactly what it returned with nothing declared.
  fit <- hd_copy()
  v <- as.numeric(fit$theta_grid[, "sigma"])
  w <- fit$weights / sum(fit$weights)
  p <- c(0.025, 0.5, 0.975)
  for (wc in c("box_uniform", "chord")) {
    expect_identical(
      tulpa:::.nl_summary_quantile(v, w, p, "positive", "density", wc, 0),
      tulpa:::.nl_summary_quantile(v, w, p, "positive", "density", wc,
                                   NA_real_))
  }
})

test_that("the level keeps its own coordinate, and nothing is drawn below it", {
  for (wc in c("box_uniform", "chord")) {
    fit <- hd_copy(wc)
    av <- as.numeric(fit$theta_grid[, "alpha"])
    w  <- fit$weights / sum(fit$weights)
    mass <- sum(w[av == 0])
    # The fixture puts the level's posterior mass between the 2.5% the interval
    # asks for and the 50% the median does, so the reported bound is the level
    # and the median is above it -- the composition read at a point on either
    # side of the level.
    expect_gt(mass, 0.025)
    expect_lt(mass, 0.5)

    set.seed(854)
    n <- if (cran_fixture()) 2e4L else 2e5L
    a <- as.numeric(tulpa_hyper_draws(fit, n = n)[, "alpha"])
    # Three readings of one geometry. The mass the fit reports on the level is
    # the mass the draws put there; nothing is drawn below the axis's support;
    # and the level is the 2.5% bound rather than a point inside a box that
    # reaches past it.
    expect_lt(abs(mean(a == 0) - mass), 5 / sqrt(n))
    expect_gte(min(a), 0)
    expect_identical(unname(fit$theta_ci_lo[["alpha"]]), 0)
    expect_gt(fit$theta_median[["alpha"]], 0)
    # And the continuum's cells were mirrored in the coordinate it is laid out
    # in, not in the value.
    expect_identical(unname(fit$theta_cell_edge_coord[["alpha"]]), "positive")
    expect_true(is.na(fit$theta_cell_edge_declined[["alpha"]]))
  }
})

test_that("a level holding more than half the posterior IS the median", {
  # The other side of the composition. Under a marginal decreasing across the
  # copy axis the level takes 0.70 of the posterior, so every probability below
  # that reads the level -- including the median, which a read spreading the
  # level over a box would have placed inside that box instead.
  for (wc in c("box_uniform", "chord")) {
    fit <- hd_copy(wc, lp_alpha = function(a) -2 * a)
    av <- as.numeric(fit$theta_grid[, "alpha"])
    w  <- fit$weights / sum(fit$weights)
    expect_gt(sum(w[av == 0]), 0.5)
    expect_identical(unname(fit$theta_ci_lo[["alpha"]]), 0)
    expect_identical(unname(fit$theta_median[["alpha"]]), 0)
    expect_gt(fit$theta_ci_hi[["alpha"]], 0)
    set.seed(854)
    a <- as.numeric(tulpa_hyper_draws(fit, n = 2e4L)[, "alpha"])
    expect_identical(unname(stats::quantile(a, 0.025)), 0)
    expect_identical(unname(stats::quantile(a, 0.5)), 0)
  }
})

test_that("the draws still reproduce the fit's own interval on the level's axis", {
  skip_on_cran()
  # Section 1's arbiter, on the axis that carries a point mass: the read
  # inverts a CDF with an atom in it and the draws sample the same one, so a
  # composition that placed the level differently in either would show up here.
  for (wc in c("box_uniform", "chord")) {
    fit <- hd_copy(wc)
    set.seed(854)
    a <- as.numeric(tulpa_hyper_draws(fit, n = 2e5L)[, "alpha"])
    q <- unname(stats::quantile(a, c(0.025, 0.5, 0.975)))
    rep_q <- c(fit$theta_ci_lo[["alpha"]], fit$theta_median[["alpha"]],
               fit$theta_ci_hi[["alpha"]])
    expect_lt(max(abs(q - rep_q)), 5e-3)
  }
})

test_that("the resolution of an axis with a point mass is its continuum's", {
  # `h` is a cell width and the level owns no cell, so the ratio is read off
  # the continuum alone -- in log, the coordinate the continuum is laid out and
  # integrated in (`.hyper_axis_measure()`). Keeping the level in measured the
  # spacing of a partition the axis is not integrated on.
  fit <- hd_copy()
  h <- fit$outer_grid_cell_width[["alpha"]]
  pos <- sort(unique(as.numeric(fit$theta_grid[, "alpha"])))
  pos <- pos[pos > 0]
  expect_equal(h, stats::median(diff(log(pos))))
  expect_true(is.na(fit$outer_grid_resolution_declined[["alpha"]]))
})
