# The free mass predictor, measured against the correction it predicts
# (gcol33/tulpa#328).
#
# `log_mass_ratio` (gcol33/tulpa#323) compares a refined cell's 25-node mass
# against its own coarse atom and exists only after the 24 node solves are
# spent, so it verifies a refinement rather than triggering one. The local
# quadratic's box integral (gcol33/tulpa#326) is the same comparison off the
# base grid's own neighbours, defined on every interior cell before a node is
# placed. Whether the second can rank the refinement budget in place of the
# cell's own integration weight is what is measured here.
#
# Three properties decide it, and the file measures all three: whether the
# predictor knows which way a cell's mass will move (sign agreement), whether it
# orders the cells the same way (rank agreement), and how much of the realized
# correction the leading fraction of its ordering captures against the ordering
# already shipped (decision concentration). The third is where it fails, and it
# fails because the per-cell log surplus and the mass that surplus moves are
# different orderings of the same cells.
#
# Everything below is post-processing of numbers a refinement already recorded,
# on analytic outer targets whose per-axis quantiles are known in closed form,
# so no inner solve is involved anywhere.

# An equicorrelated Gaussian outer target, or a Gaussian copula with Gamma
# marginals at the same correlation. The second is skewed within the cell, which
# is where the local quadratic stops accounting for the cell's mass and where
# `misfit` fires.
.mps_target <- function(rho, reach, m, d = 4L, gamma_shape = NULL) {
  R <- matrix(rho, d, d); diag(R) <- 1
  if (is.null(gamma_shape)) {
    Q <- solve(R)
    lf <- function(TH) -0.5 * rowSums((TH %*% Q) * TH)
    rng <- c(-reach, reach)
    qx <- stats::qnorm(c(0.025, 0.5, 0.975))
  } else {
    Qi <- solve(R) - diag(d)
    lf <- function(TH) {
      TH <- pmax(TH, 1e-12)
      z <- stats::qnorm(stats::pgamma(TH, shape = gamma_shape))
      z[!is.finite(z)] <- sign(z[!is.finite(z)]) * 8
      rowSums(stats::dgamma(TH, shape = gamma_shape, log = TRUE)) -
        0.5 * rowSums((z %*% Qi) * z)
    }
    rng <- stats::qgamma(stats::pnorm(c(-reach, reach)), shape = gamma_shape)
    qx <- stats::qgamma(c(0.025, 0.5, 0.975), shape = gamma_shape)
  }
  lv <- seq(rng[1L], rng[2L], length.out = m)
  ll <- stats::setNames(replicate(d, lv, simplify = FALSE), paste0("a", seq_len(d)))
  g <- as.matrix(expand.grid(ll, KEEP.OUT.ATTRS = FALSE))
  colnames(g) <- names(ll)
  list(grid = g, lm = lf(g), exact = qx,
       tags = stats::setNames(rep("identity", d), colnames(g)),
       eval = function(theta_mat, warm)
         list(log_marginal = lf(theta_mat), modes = NULL))
}

.mps_refine <- function(tg, max_cells = 40L, skew_max = Inf)
  tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                  colnames(tg$grid), tg$tags, tg$eval,
                                  max_cells = max_cells, skew_max = skew_max)

# One refinement's cells as a frame: the predictor, the realization, the cell's
# own base-grid weight, and the two readings of the realized correction.
#
#   real_a = |log(M_1 / M_0)|                 every cell counted equally
#   real_b = w_c |exp(log_mass_ratio) - 1|    the mass the correction moves,
#                                             i.e. the cell's contribution to
#                                             the integral
.mps_cells <- function(tg, ...) {
  rf <- .mps_refine(tg, ...)
  if (is.null(rf) || rf$info$n_cells_refined < 2L) return(NULL)
  i <- rf$info
  w0 <- tulpa:::.joint_integration_weights(tg$lm, NULL)
  d <- data.frame(share = w0[i$cells], pred = i$log_box_ratio,
                  real = i$log_mass_ratio, misfit = i$misfit,
                  mode_gain = i$mode_gain)
  d <- d[is.finite(d$pred) & is.finite(d$real) & is.finite(d$share) &
           d$share > 0, ]
  if (nrow(d) < 2L) return(NULL)
  d$real_a <- abs(d$real)
  d$real_b <- d$share * abs(expm1(d$real))
  d$pred_w <- d$share * abs(expm1(d$pred))
  d
}

# The share of a fit's total realized correction captured by the leading
# `frac` of its cells under one ordering. `frac` of 1 is 1 by construction, so
# what the curve reports is concentration and not coverage.
.mps_capture <- function(d, key, tgt, frac) {
  k <- max(1L, ceiling(frac * nrow(d)))
  tot <- sum(d[[tgt]])
  if (!is.finite(tot) || tot <= 0) return(NA_real_)
  sum(d[[tgt]][order(d[[key]], decreasing = TRUE)[seq_len(k)]]) / tot
}

# A small deterministic set spanning the skewness ladder the gate keys on.
.mps_set <- function(m = 5L) {
  cfg <- expand.grid(rho = c(0, 0.8), reach = c(2, 3),
                     shape = c(NA, 2, 8, 32))
  out <- list()
  for (i in seq_len(nrow(cfg))) {
    gs <- if (is.na(cfg$shape[i])) NULL else cfg$shape[i]
    tg <- .mps_target(cfg$rho[i], cfg$reach[i], m, gamma_shape = gs)
    d <- .mps_cells(tg)
    if (!is.null(d)) { d$target <- i; out[[length(out) + 1L]] <- d }
  }
  do.call(rbind, out)
}

.mps_mean_capture <- function(s, key, tgt, frac)
  mean(vapply(split(s, s$target), .mps_capture, numeric(1),
              key = key, tgt = tgt, frac = frac), na.rm = TRUE)

# --------------------------------------------------------------------------- #
# Sign and rank: does the free ratio know which way, and in what order?        #
# --------------------------------------------------------------------------- #

test_that("the predictor agrees with the realized surplus in sign and in rank", {
  skip_on_cran()
  s <- .mps_set()
  # The regime first: enough cells, and both signs present, or the agreement
  # rate below would be a statement about one of them.
  expect_gte(nrow(s), 300L)
  expect_gt(sum(s$real < 0), 20L)
  expect_gt(sum(s$real > 0), 20L)

  # Sign agreement. Measured over the wider sweep this set is a slice of (2509
  # cells, 124 fits and targets): 0.9518 overall, 0.9821 on the real four-axis
  # joint fits, and 0.7824 on the Gamma(2) targets, which are the most skewed
  # rung of the ladder. It is asymmetric -- 0.9717 where the cloud read MORE
  # mass than the atom and 0.8624 where it read less.
  expect_gt(mean(sign(s$pred) == sign(s$real)), 0.90)

  # Rank agreement. Pooled over the wider sweep: Spearman 0.940, Pearson 0.892.
  expect_gt(stats::cor(s$pred, s$real, method = "spearman"), 0.85)
  expect_gt(stats::cor(s$pred, s$real), 0.80)

  # It is not CALIBRATED to the realization and does not have to be: the box
  # integral covers the whole cell while the cloud it predicts is clamped to a
  # shrunk design box, so the predictor overstates. The wider sweep's slope is
  # 0.470; a ranking reads the order, not the value.
  b <- stats::coef(stats::lm(real ~ pred, data = s))[2L]
  expect_gt(b, 0.1)
  expect_lt(b, 0.9)
})

test_that("the predictor's error is not the regime misfit characterises", {
  skip_on_cran()
  s <- .mps_set()
  s$resid <- stats::residuals(stats::lm(real ~ pred, data = s))
  # The regime: the set spans the gate's band on both sides.
  thr <- tulpa:::.nl_diag("gamma3_ok")
  expect_gt(sum(s$misfit >= thr), 20L)
  expect_gt(sum(s$misfit < thr), 20L)
  # Where the local quadratic stops holding the predictor does lose some order
  # -- over the wider sweep Spearman 0.968 below the band against 0.860 above,
  # and sign agreement 0.966 against 0.914 -- but the two are not the same axis:
  # per cell, `misfit` carries no information about the size of the predictor's
  # error (correlation -0.015 pearson, -0.021 spearman over 2509 cells), and the
  # residual MAGNITUDE is if anything smaller on the high-misfit side (0.787
  # against 0.914). A cell can carry either failure with the other clean, which
  # is the orthogonality `misfit` and the mass scores were separated on.
  expect_lt(abs(stats::cor(s$misfit, abs(s$resid), method = "spearman")), 0.35)
})

# --------------------------------------------------------------------------- #
# Decision concentration: the reading that decides the criterion               #
# --------------------------------------------------------------------------- #

test_that("the two readings of the realized correction order the cells oppositely", {
  skip_on_cran()
  s <- .mps_set()
  # `real_a` counts every cell once; `real_b` is the mass the correction
  # actually moves. They are not two views of one ordering: over the wider
  # sweep the per-fit Spearman between them is -0.700 (median) and their top
  # deciles share 0.089 of their cells.
  rho <- vapply(split(s, s$target), function(z)
    stats::cor(z$real_a, z$real_b, method = "spearman"), numeric(1))
  expect_lt(stats::median(rho), 0)
  # And the difference is mass: the cells the log surplus puts first hold almost
  # none of the grid's weight. Measured over the wider sweep, the top decile
  # holds 0.045 of the weight under the log surplus against 0.262 under the mass
  # moved, and 0.003 under the bare predictor.
  top <- function(z, key) {
    k <- max(1L, ceiling(0.1 * nrow(z)))
    sum(z$share[order(z[[key]], decreasing = TRUE)[seq_len(k)]])
  }
  wa <- mean(vapply(split(s, s$target), top, numeric(1), key = "real_a"))
  wb <- mean(vapply(split(s, s$target), top, numeric(1), key = "real_b"))
  expect_lt(wa, wb)
})

test_that("the incumbent weight ranking already captures the mass the correction moves", {
  skip_on_cran()
  s <- .mps_set()
  cap <- function(key, tgt, frac) .mps_mean_capture(s, key, tgt, frac)

  # The decision-relevant curve: cells ordered by a rule, share of the fit's
  # total |M_1 - M_0| captured. Measured over the wider sweep at the leading
  # 10% of cells (117 fits with at least 8 refined cells): oracle 0.8231, the
  # incumbent `w_c` 0.8173, `w_c` times the predicted ratio 0.7669, the bare
  # ratio 0.0070, random 0.1170.
  for (fr in c(0.10, 0.25, 0.50)) {
    oracle <- cap("real_b", "real_b", fr)
    inc <- cap("share", "real_b", fr)
    pred <- cap("pred", "real_b", fr)
    predw <- cap("pred_w", "real_b", fr)
    # The incumbent is at the ceiling, so there is nothing for the predictor to
    # win: it is within a few thousandths of an ordering that knows the answer.
    expect_gt(inc, 0.95 * oracle)
    # The bare ratio is not merely worse than the incumbent, it is worse than
    # random: it selects precisely the cells that hold no mass.
    expect_lt(pred, fr)
    expect_lt(pred, inc)
    # Weighting it by the cell's mass recovers most of that, and stops there:
    # the weight factor is doing the work and the predicted ratio, overstating
    # by a factor of two, is a perturbation on it. Over the wider sweep it sits
    # below the incumbent at every fraction (0.7669 / 0.8968 / 0.9529 against
    # 0.8173 / 0.9108 / 0.9537); on this slice it is within 0.004 either way.
    # Neither ordering improves on the other, which is the finding.
    expect_lt(predw - inc, 0.02)
    expect_lt(predw, oracle)
  }

  # On the OTHER reading the predictor is the better rule by as much, which is
  # why the choice between the two readings is the whole of the decision.
  # Measured over the wider sweep at 10%: oracle 0.2379, predictor 0.1838,
  # incumbent 0.1174, random 0.1170.
  expect_gt(cap("pred", "real_a", 0.10), cap("share", "real_a", 0.10))
  expect_gt(cap("pred", "real_a", 0.25), cap("share", "real_a", 0.25))
})

# --------------------------------------------------------------------------- #
# mode_gain, and what the box truncation accounts for                          #
# --------------------------------------------------------------------------- #

# `0.5 sum_j g_j^2 / a_j` over the concave axes of the same stencil the box
# integral reads: the unbounded diagonal quadratic gain, i.e. what the box ratio
# would be if the cell's box were the whole line and the determinant factor were
# dropped.
.mps_stencil_gain <- function(st) {
  a <- -st$d2
  ok <- a > 0
  if (!any(ok)) return(NA_real_)
  0.5 * sum(st$g[ok]^2 / a[ok])
}

test_that("the truncation, not the quadratic, is what separates mode_gain from the box ratio", {
  skip_on_cran()
  tg <- .mps_target(rho = 0.8, reach = 3, m = 5L)
  rf <- .mps_refine(tg)
  i <- rf$info
  expect_gte(i$n_cells_refined, 8L)

  latent <- seq_len(ncol(tg$grid))
  nb <- tulpa:::.joint_local_ccd_neighbors(tg$grid, tg$grid, latent)
  sg <- vapply(i$cells, function(c) {
    st <- tulpa:::.joint_local_ccd_cell_stencil(c, tg$grid, tg$lm, nb$up, nb$dn)
    if (is.null(st)) NA_real_ else .mps_stencil_gain(st)
  }, numeric(1))
  ok <- is.finite(sg) & is.finite(i$mode_gain) & is.finite(i$log_box_ratio)
  expect_gte(sum(ok), 8L)

  # `mode_gain` comes from the design's least-squares FULL Hessian in whitened
  # coordinates and the stencil gain from a DIAGONAL finite difference in u, and
  # they are the same quantity: over the wider sweep their median ratio is
  # 1.068. So the gap between `mode_gain` and the box ratio is not a
  # disagreement about the quadratic.
  expect_equal(stats::median(i$mode_gain[ok] / sg[ok]), 1, tolerance = 0.25)
  # It is the truncation. Integrating the same quadratic over the CELL instead
  # of the line removes a median 6.198 nats over the wider sweep, 0.867 of the
  # unbounded gain, on every one of the 2072 cells carrying both.
  expect_true(all(i$log_box_ratio[ok] < sg[ok]))
  expect_gt(stats::median(1 - i$log_box_ratio[ok] / sg[ok]), 0.3)
})

test_that("the truncated form is the better predictor of the realized surplus", {
  skip_on_cran()
  s <- .mps_set()
  s <- s[is.finite(s$mode_gain), ]
  expect_gte(nrow(s), 200L)
  # Measured over the wider sweep, Spearman against the realized surplus: the
  # box ratio 0.978, the stencil's unbounded gain 0.904, `mode_gain` 0.812. The
  # unbounded form is the same displacement without the cell's own extent, and
  # the cell's own extent is what the surplus is a property of.
  expect_gt(stats::cor(s$pred, s$real, method = "spearman"),
            stats::cor(s$mode_gain, s$real, method = "spearman"))
})

# --------------------------------------------------------------------------- #
# The read, scored the same way: the criterion is left as it is                #
# --------------------------------------------------------------------------- #

# The same refinement run under an alternative candidate ranking. The selection
# ranks on `w[cands]`, so substituting the ranking key for `w` IS the
# alternative rule, and the greedy non-adjacency walk stays the shipped one
# rather than being written down a second time.
.mps_ranked_refine <- function(tg, key, max_cells) {
  orig <- tulpa:::.joint_local_ccd_select
  alt <- function(cands, w, up, dn, max_cells) orig(cands, key, up, dn, max_cells)
  testthat::local_mocked_bindings(.joint_local_ccd_select = alt,
                                  .package = "tulpa")
  .mps_refine(tg, max_cells = max_cells)
}

# Mean absolute error of a refined grid's per-axis 95% endpoints against the
# target's exact quantiles.
.mps_endpoint_err <- function(rf, tg) {
  w <- tulpa:::.joint_integration_weights(rf$log_marginal, rf$dnode)
  Q <- t(vapply(seq_len(ncol(rf$joint_grid)), function(j) {
    v <- as.numeric(rf$joint_grid[, j])
    use <- is.finite(v) & is.finite(w) & w > 0
    tulpa:::.nl_wtd_quantile(v[use], w[use] / sum(w[use]),
                             c(0.025, 0.5, 0.975), "clamp")
  }, numeric(3L)))
  ex <- matrix(rep(tg$exact, each = nrow(Q)), nrow = nrow(Q))
  c(endpoints = mean(abs(Q[, c(1L, 3L)] - ex[, c(1L, 3L)])),
    widths = mean(abs((Q[, 3L] - Q[, 1L]) - (ex[, 3L] - ex[, 1L]))))
}

test_that("ranking the budget by the predictor does not improve the read", {
  skip_if_not_slow()
  # The budget has to BIND for the ranking to mean anything: a five-level tensor
  # in four axes has 81 interior candidates and the budget here is 4 and 8.
  acc <- list()
  for (shape in list(NULL, 2, 8, 32)) for (rho in c(0, 0.5, 0.8))
    for (reach in if (is.null(shape)) c(2, 3, 4) else c(1.5, 2, 3))
      for (mc in c(4L, 8L)) {
        tg <- .mps_target(rho, reach, 5L, gamma_shape = shape)
        w0 <- tulpa:::.joint_integration_weights(tg$lm, NULL)
        bm <- tulpa:::.joint_local_ccd_box_mass(tg$grid, tg$lm,
                                                colnames(tg$grid), tg$tags)
        lbr <- ifelse(bm$computed, bm$log_box_ratio, NA_real_)
        keys <- list(
          raw = ifelse(is.na(lbr), -Inf, lbr),
          move = ifelse(is.na(lbr), -Inf, log(w0) + log(abs(expm1(lbr)))))
        base <- .mps_refine(tg, max_cells = mc)
        if (is.null(base) || base$info$n_cells_refined == 0L) next
        e <- list(weight = .mps_endpoint_err(base, tg))
        for (nm in names(keys)) {
          r <- .mps_ranked_refine(tg, keys[[nm]], mc)
          if (is.null(r)) next
          e[[nm]] <- .mps_endpoint_err(r, tg)
        }
        if (length(e) != 3L) next
        acc[[length(acc) + 1L]] <- data.frame(
          ep_weight = e$weight[["endpoints"]], ep_raw = e$raw[["endpoints"]],
          ep_move = e$move[["endpoints"]],
          wd_weight = e$weight[["widths"]], wd_raw = e$raw[["widths"]],
          wd_move = e$move[["widths"]])
      }
  s <- do.call(rbind, acc)
  expect_gte(nrow(s), 40L)

  # Measured over 468 configurations of the same sweep at three grid
  # resolutions: summed endpoint error 377.07 ranking by the cell's mass against
  # 380.71 by predicted mass movement and 374.96 by the bare ratio, and summed
  # width error 675.02 / 681.38 / 699.37, against 419.98 / 823.53 for not
  # refining at all. The rankings differ on 190 of the 468 and the spread
  # between them is a fortieth of what refining buys, in both directions
  # depending on the part of the read. Nothing here separates them, which is the
  # result: the shipped criterion is left as it is.
  rng <- function(p) {
    v <- c(sum(s[[paste0(p, "_weight")]]), sum(s[[paste0(p, "_raw")]]),
           sum(s[[paste0(p, "_move")]]))
    (max(v) - min(v)) / min(v)
  }
  expect_lt(rng("ep"), 0.15)
  expect_lt(rng("wd"), 0.15)
})
