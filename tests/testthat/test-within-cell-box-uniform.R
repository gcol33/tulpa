# The WITHIN-CELL construction a reported hyperparameter interval is read with
# (gcol33/tulpa#357).
#
# The outer grid's weights say how much mass each cell holds. They do not say
# how it is spread INSIDE the cell, and a quantile needs both. The shipped
# `chord` read puts the cumulative MID-mass at each cell COORDINATE and
# interpolates between coordinates; `box_uniform` puts the cumulative FULL mass
# at each cell EDGE and interpolates between edges -- the same masses over the
# same boxes with the knots moved half a cell, which is a whole order of
# convergence (measured 1.04 against 2.00, `dev_notes/issue353/RESULTS.md` 2.3).
#
# What is pinned here: that the default did not move; that the construction is
# selectable through `control` and reported back per axis; that its boxes tile
# the axis in the axis's OWN coordinate so a bounded axis cannot leave its
# support; that every decline is a recorded reason and a fall back to the chord
# read rather than an error; and that the shipped engine reproduces the arm
# `dev_notes/issue337/` scored.

PROBS_WC <- c(0.005, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.995)

# --------------------------------------------------------- the default is fixed

test_that("the default construction leaves every reported number identical", {
  v <- exp(seq(log(0.2), log(1.5), length.out = 5))
  w <- c(0.30, 0.25, 0.20, 0.15, 0.10)
  for (sup in .NL_SUPPORT_KINDS) {
    for (dm in c("positive", NA_character_)) {
      expect_identical(.nl_summary_quantile(v, w, PROBS_WC, dm, sup),
                       .nl_summary_quantile(v, w, PROBS_WC, dm, sup, "chord"))
    }
  }
  # `.nl_summary_quantile()` is `.nl_summary_quantile_read()`'s `$q` -- one
  # dispatcher, so the vector caller and the provenance caller cannot drift.
  r <- .nl_summary_quantile_read(v, w, PROBS_WC, "positive", "density")
  expect_identical(r$q, .nl_summary_quantile(v, w, PROBS_WC, "positive",
                                             "density"))
  expect_identical(r$within, "chord")
  expect_true(is.na(r$declined))
  # And the engine's own default is the shipped read.
  expect_identical(.nl_diag("within_cell"), "chord")
  expect_identical(.nl_within_cell_mode(NULL), "chord")
  expect_identical(.NL_WITHIN_CELL[1L], "chord")
})

test_that("`.nl_cell_edges()` is unchanged by the partition extraction", {
  # `.nl_cell_partition()` is what the box read needs: the coordinate the outer
  # edges were mirrored in is not recoverable from the edges, because the guard
  # falls back when a mapped edge leaves the support. `.nl_cell_edges()` is now
  # its `$edges` and must be the same function it was.
  cases <- list(exp(seq(log(0.2), log(1.5), length.out = 5)),
                c(-2, 0, 2), c(0.05, 0.3, 0.6, 0.9, 0.9962, 0.99972),
                c(-0.9, -0.2, 0.3, 0.8), c(0.5, 1.5, 2.5), 3)
  for (v in cases) {
    for (dm in c(NA_character_, "positive", "unit", "correlation",
                 "unbounded")) {
      p <- .nl_cell_partition(sort(v), dm)
      expect_identical(.nl_cell_edges(sort(v), dm), p$edges)
      # The reported coordinate really is the one the edges came out of.
      if (length(v) > 1L) {
        expect_equal(p$tr$to(p$edges),
                     local({
                       u <- p$tr$to(sort(v)); n <- length(u)
                       c(u[1L] - 0.5 * (u[2L] - u[1L]),
                         u[n] + 0.5 * (u[n] - u[n - 1L]))
                     }))
      }
    }
  }
})

# --------------------------------------------------------------- the boxes tile

test_that("the boxes tile the axis and carry the shipped masses exactly", {
  v <- exp(seq(log(0.2), log(1.5), length.out = 6))
  w <- c(0.05, 0.15, 0.3, 0.25, 0.15, 0.10)
  e <- .nl_box_edges(v, "positive")
  expect_length(e, length(v) + 1L)
  expect_false(is.unsorted(e, strictly = TRUE))
  # Edge k is BOTH the upper edge of cell k and the lower edge of cell k+1 --
  # one number serving both, so no mass leaves a cell it was integrated for and
  # none is counted twice. Each coordinate sits inside its own box.
  expect_true(all(v > e[-length(e)] & v < e[-1L]))
  # And the outer two are the cell partition's own.
  expect_equal(c(e[1L], e[length(e)]), .nl_cell_edges(v, "positive"))
  # The CDF at edge k is the cumulative shipped mass through cell k.
  cw <- cumsum(w / sum(w))
  q <- .nl_summary_quantile(v, w, cw[-length(cw)], "positive", "density",
                            "box_uniform")
  expect_equal(unname(q), e[seq_len(length(v) - 1L) + 1L])
  # Interior edges are midpoints in the coordinate the partition lives in.
  expect_equal(e[3L], exp(mean(log(v[2:3]))))
})

test_that("the partition comes from the grid, the masses from the weights", {
  # gcol33/tulpa#337's own rule is "keep the masses, tile the axis", and the two
  # halves come from different places. A cell whose integration weight underflows
  # to exactly 0 still SITS on the axis, and its coordinate is what fixes its
  # neighbour's box edge; filtering the coordinates by weight would shrink that
  # neighbour's box to nothing and collapse the read onto the chord one. Measured
  # on the coarsest rung of `dev_notes/issue357/coarse357b.R` -- 2 cells at 400
  # groups -- the softmax underflows one of the two on 43 of 150 seeds, so this
  # is the difference between measuring the construction and measuring nothing.
  #
  # The chord read filters both together, correctly: its knots ARE the
  # positive-weight coordinates. This one's knots are edges.
  v <- c(0.3309751, 0.9064126)
  w <- c(0, 1)
  h <- 0.5 * diff(log(v))
  bx <- c(exp(mean(log(v))), exp(log(v[2L]) + h))   # cell 2's own box
  p <- c(0.025, 0.5, 0.975)
  q <- .nl_summary_quantile(v, w, p, "positive", "density", "box_uniform")
  expect_equal(q, bx[1L] + p * diff(bx))
  expect_false(isTRUE(all.equal(
    q, .nl_summary_quantile(v, w, p, "positive", "density", "chord"))))
  # It is a box read, not a decline.
  expect_true(is.na(.nl_summary_quantile_read(v, w, p, "positive", "density",
                                              "box_uniform")$declined))

  # An interior empty box is a FLAT segment of the CDF, not a merged one: the
  # quantile is located on the cumulative mass and evaluated inside the box it
  # lands in, so an empty box is stepped over rather than interpolated across.
  v4 <- exp(seq(log(0.2), log(1.5), length.out = 4))
  e4 <- .nl_box_edges(v4, "positive")
  q4 <- .nl_summary_quantile(v4, c(0.5, 0, 0, 0.5), c(0.25, 0.5, 0.75),
                             "positive", "density", "box_uniform")
  expect_gte(q4[1L], e4[1L]); expect_lte(q4[1L], e4[2L])
  expect_equal(q4[2L], e4[4L])
  expect_gte(q4[3L], e4[4L]); expect_lte(q4[3L], e4[5L])
  # Interpolating across the two empty boxes would put q(0.75) below e4[4].
  expect_gt(q4[3L], e4[3L])
})

test_that("a bounded axis's boxes stay inside its support", {
  # gcol33/tulpa#361 extended `bym2_rho` to six nodes including 0.999, so a
  # `unit` axis with a node that close to its boundary is live. Mirroring or
  # bisecting such an axis in `log` -- which is what a coordinate guessed from
  # `all(v > 0)` does -- puts an edge above 1, and a BYM2 mixing weight is
  # singular there (gcol33/tulpa#369).
  v <- c(0.05, 0.3, 0.6, 0.9, 0.9962, 0.99972)
  w <- c(0.05, 0.10, 0.20, 0.30, 0.20, 0.15)
  e <- .nl_box_edges(v, "unit")
  expect_true(all(e > 0 & e < 1))
  q <- .nl_summary_quantile(v, w, c(0.001, 0.5, 0.999), "unit", "density",
                            "box_uniform")
  expect_true(all(q > 0 & q < 1))
  # Without the domain the log guess is taken and does leave the support, which
  # is the fallback's own behaviour and not a claim about `unit`.
  expect_gt(.nl_box_edges(v, NA_character_)[length(v) + 1L], 1)
  r <- c(-0.9, -0.2, 0.3, 0.8)
  expect_true(all(abs(.nl_box_edges(r, "correlation")) < 1))
})

# ------------------------------------------------------ declines, never errors

test_that("every decline records a reason and falls back to the chord read", {
  v <- exp(seq(log(0.2), log(1.5), length.out = 5))
  w <- c(0.30, 0.25, 0.20, 0.15, 0.10)
  # A support whose node set is not a tiling partition declines by NAME, so a
  # caller can tell "this kind does not admit it" from "this node set failed".
  for (sup in c("moment_rule", "mixed", "sample")) {
    r <- .nl_summary_quantile_read(v, w, PROBS_WC, "positive", sup,
                                   "box_uniform")
    expect_identical(r$within, "chord")
    expect_identical(r$declined, paste0("support_", sup))
    expect_identical(r$q, .nl_summary_quantile(v, w, PROBS_WC, "positive", sup))
  }
  # One node has no box; the chord read's answer (the value itself) stands.
  r1 <- .nl_summary_quantile_read(3, 1, PROBS_WC, "positive", "density",
                                  "box_uniform")
  expect_identical(r1$declined, "single_node")
  expect_identical(r1$q, rep(3, length(PROBS_WC)))
  # No usable node at all is NA, as it is under either construction.
  r0 <- .nl_summary_quantile_read(c(1, 2), c(0, 0), PROBS_WC, "positive",
                                  "density", "box_uniform")
  expect_identical(r0$declined, "no_usable_node")
  expect_true(all(is.na(r0$q)))
  # `slice_masses()` in `dev_notes/issue337/recon.R` STOPS when the boxes do not
  # tile, because mass would leave the cell it was integrated for. A reported
  # interval cannot take that behaviour, so the engine declines instead
  # (gcol33/tulpa#293). The reachable case is an axis reaching the top of the
  # double range, where the mirrored outer edge is not finite in any coordinate.
  vd <- c(1, 1e300, .Machine$double.xmax)
  expect_null(.nl_box_edges(vd, "positive"))
  # A spacing below the coordinate's own resolution collapses a box to zero
  # width and is refused at the edge builder, but it does not reach the read:
  # `.nl_axis_atoms()` has already merged coordinates that are equal as doubles,
  # so the partition the read sees never has a zero-width interior box.
  expect_null(.nl_box_edges(c(1, 1, 2), "positive"))
  expect_false(is.null(.nl_box_edges(
    .nl_axis_atoms(c(1, 1, 2), c(1, 1, 1))$v, "positive")))
  rd <- .nl_summary_quantile_read(vd, c(1, 1, 1), PROBS_WC,
                                  "positive", "density", "box_uniform")
  expect_identical(rd$declined, "boxes_do_not_tile")
  expect_identical(rd$within, "chord")
  expect_identical(rd$q, .nl_summary_quantile(vd, c(1, 1, 1), PROBS_WC,
                                              "positive", "density"))
  # `.nl_cell_edges()` (the chord read's) does NOT decline there -- it returns
  # the non-finite edge its own fallback chain ends on, which is what the read
  # has always done. The box read needs a whole finite partition, so it is
  # stricter, and that is a decline rather than a change to the chord read.
  expect_false(all(is.finite(.nl_cell_edges(vd, "positive"))))
  # An unknown construction is refused rather than translated.
  expect_error(.nl_summary_quantile(v, w, 0.5, "positive", "density", "boxes"))
  expect_error(.nl_within_cell_mode("uniform"))
})

# ------------------------------------------------------ the per-axis provenance

test_that("the axis read says which construction produced each interval", {
  v1 <- exp(seq(log(0.2), log(1.5), length.out = 4))
  v2 <- c(0.5, 1.5)
  tg <- as.matrix(expand.grid(sigma = v1, tau = v2))
  lm <- c(0, -0.4, -3, -6, -0.2, -0.6, -3.2, -6.2)
  qc <- .nl_axis_quantiles(tg, lm)
  expect_identical(unname(qc$within), c("chord", "chord"))
  expect_true(all(is.na(qc$within_declined)))
  qb <- .nl_axis_quantiles(tg, lm, within = "box_uniform")
  expect_identical(unname(qb$within), c("box_uniform", "box_uniform"))
  expect_false(isTRUE(all.equal(unname(qb$ci_lo), unname(qc$ci_lo))))
  # A single-node axis falls back on its OWN, without taking the fit with it.
  tg1 <- cbind(sigma = tg[, "sigma"], fixed = rep(2, nrow(tg)))
  q1 <- .nl_axis_quantiles(tg1, lm, within = "box_uniform")
  expect_identical(unname(q1$within), c("box_uniform", "chord"))
  expect_identical(unname(q1$within_declined), c(NA_character_, "single_node"))
})

test_that("the per-axis resolution is reported in the axis's own coordinate", {
  # `h / sd` is the regime variable for the position sensitivity a within-cell
  # reconstruction has, so a fit reports it rather than leaving the reader with
  # no way to tell a resolved axis from an unresolved one.
  v <- exp(seq(log(0.2), log(1.5), length.out = 9))
  u <- log(v)
  # A discretized Gaussian in u, so the answer is known.
  sd_true <- 0.35
  lm <- -0.5 * ((u - mean(u)) / sd_true)^2
  tg <- matrix(v, ncol = 1L, dimnames = list(NULL, "sigma"))
  rs <- .nl_axis_resolution(tg, lm, domains = list("positive"))
  expect_equal(unname(rs$h[["sigma"]]), diff(u)[1L])
  expect_equal(unname(rs$sd[["sigma"]]), sd_true, tolerance = 1e-8)
  expect_equal(unname(rs$h_over_sd[["sigma"]]), diff(u)[1L] / sd_true,
               tolerance = 1e-8)
  # The SD is the one `.nl_laplace_at_mode_sd_axis()` reports in that same
  # coordinate; on a positive axis the linear-scale value is the delta map of
  # it, and the ratio would be a different (and meaningless) number.
  expect_false(isTRUE(all.equal(
    unname(rs$sd[["sigma"]]),
    .nl_laplace_at_mode_sd_axis(v, lm))))
  # An axis whose mode sits on its own boundary has no 3-point parabola, so the
  # ratio is withheld rather than guessed.
  lm_rail <- seq(0, 5, length.out = 9)
  expect_true(is.na(.nl_axis_resolution(tg, lm_rail,
                                        domains = list("positive"))$sd[[1L]]))
})

# ----------------------------------------------------------- through a real fit

test_that("a fit selects the construction, reports it, and stays byte-identical by default", {
  skip_on_cran()
  skip_if_fast()
  set.seed(357L)
  G <- 40L; npg <- 30L; N <- G * npg
  g <- sample.int(G, N, replace = TRUE)
  x <- rnorm(N); X <- cbind(1, x)
  y <- as.numeric(X %*% c(-0.2, 0.7)) + rnorm(G, 0, 0.7)[g] +
       rnorm(N, 0, sqrt(0.5))
  sg <- exp(seq(log(0.2), log(1.5), length.out = 9L))
  ctrl <- list(max_iter = 100L, tol = 1e-8, n_threads = 1L, diagnose_k = FALSE,
               diagnose_skew = FALSE, integration = "grid", local_ccd = NULL,
               auto_recenter = FALSE)
  fit <- function(wc) suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, n_trials = rep(1L, N), X = X,
                              family = "gaussian", phi = sqrt(0.5))),
    prior = list(list(type = "iid", obs_idx = list(g), n_units = G,
                      sigma_grid = sg)),
    control = if (is.null(wc)) ctrl else c(ctrl, list(within_cell = wc))))

  f0 <- fit(NULL)
  fc <- fit("chord")
  fb <- fit("box_uniform")

  # Asking for the default explicitly changes nothing.
  expect_identical(f0$theta_median, fc$theta_median)
  expect_identical(f0$theta_ci_lo, fc$theta_ci_lo)
  expect_identical(f0$theta_ci_hi, fc$theta_ci_hi)
  # The construction is reported, and every path stamps the node-set kind now,
  # not only the multi-block driver (gcol33/tulpa#357).
  expect_identical(f0$within_cell_requested, "chord")
  expect_identical(fb$within_cell_requested, "box_uniform")
  expect_identical(unname(f0$theta_within_cell), "chord")
  expect_identical(unname(fb$theta_within_cell), "box_uniform")
  expect_identical(f0$theta_interval_read, "density")
  expect_equal(f0$theta_interval_design_mass, 0)
  # The moments are the SAME under either construction: they are sums over the
  # cells, and this changes only where inside a cell the mass sits.
  expect_identical(f0$theta_mean, fb$theta_mean)
  expect_identical(f0$theta_sd, fb$theta_sd)
  # And the interval is not.
  expect_false(isTRUE(all.equal(unname(f0$theta_ci_lo),
                                unname(fb$theta_ci_lo))))
  expect_lt(unname(fb$theta_ci_hi - fb$theta_ci_lo),
            unname(f0$theta_ci_hi - f0$theta_ci_lo))
  # The fit knows how coarse its own grid is.
  expect_true(is.finite(f0$outer_grid_h_over_sd[[1L]]))
  expect_equal(unname(f0$outer_grid_cell_width[[1L]]), diff(log(sg))[1L])

  # The shipped read IS the arm `dev_notes/issue337/recon.R` scored: cell masses
  # over the cells' own boxes, the CDF read by the chord between edges. Rebuilt
  # here from the fit's own grid, independently of the engine's code path.
  w <- exp(f0$log_marginal - max(f0$log_marginal)); w <- w / sum(w)
  v <- as.numeric(f0$theta_grid); o <- order(v); v <- v[o]; w <- w[o]
  u <- log(v); n <- length(u)
  mid <- (u[-1L] + u[-n]) / 2
  e <- exp(c(u[1L] - (mid[1L] - u[1L]), mid, u[n] + (u[n] - mid[n - 1L])))
  Fc <- c(0, cumsum(w))
  ref <- stats::approx(Fc, e, xout = c(0.025, 0.5, 0.975), ties = "ordered",
                       rule = 2)$y
  expect_equal(unname(c(fb$theta_ci_lo, fb$theta_median, fb$theta_ci_hi)),
               ref, tolerance = 1e-12)

  # `diagnostics()` reads the construction back and says what it means.
  d <- diagnostics(fb)
  expect_identical(attr(d, "within_cell_requested"), "box_uniform")
  expect_identical(attr(d, "interval_read"), "density")
  expect_true(any(grepl("box_uniform", attr(d, "interval_read_note"))))
  expect_true(is.finite(attr(d, "grid_h_over_sd_max")))
  # This fixture is in the unresolved regime -- as every configuration of the
  # 34-row census is -- so the resolution note fires and names the axis.
  expect_gt(attr(d, "grid_h_over_sd_max"), .nl_diag("grid_resolved"))
  expect_true(grepl(attr(d, "grid_coarsest_axis"),
                    attr(d, "grid_resolution_note"), fixed = TRUE))
})

test_that("an unknown within_cell is refused at the front door", {
  skip_on_cran()
  skip_if_fast()
  set.seed(358L)
  N <- 200L
  X <- cbind(1, rnorm(N))
  g <- sample.int(10L, N, replace = TRUE)
  y <- as.numeric(X %*% c(0, 1)) + rnorm(N)
  expect_error(suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, n_trials = rep(1L, N), X = X,
                              family = "gaussian", phi = 1)),
    prior = list(list(type = "iid", obs_idx = list(g), n_units = 10L,
                      sigma_grid = c(0.5, 1, 2))),
    control = list(diagnose_k = FALSE, diagnose_skew = FALSE,
                   within_cell = "quadratic"))))
})
