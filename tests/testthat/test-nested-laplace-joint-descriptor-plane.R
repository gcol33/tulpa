# Does a cell's position in the free `(R_M, R_L)` descriptor plane say WHICH
# correction that cell wants (gcol33/tulpa#333)?
#
# Both descriptors are post-processing of the same three-point stencil, so both
# are available at zero inner solves:
#
#   R_M = |log(M_c^(Q) / M_c^(0))|   the mass the midpoint atom drops
#                                    (gcol33/tulpa#326)
#   R_L = ||ubar||                   the place it drops, per axis standardized
#                                    by the half-width the atom moves toward
#                                    (gcol33/tulpa#327)
#
# `R_L` is reported here as the L2 norm of those per-axis shares; the largest
# share (`bary_shift`, what the refinement records) and the A-metric norm
# `sqrt(sum_j a_j ubar_j^2)` are carried alongside so a conclusion is never a
# property of one of the three.
#
# What is scored is a weighted QUANTILE of the whole grid, and a quantile is not
# a sum over its atoms: moving one atom changes which OTHER atom sits at the
# 2.5% / 50% / 97.5% crossing. So a cell's best correction is measured by a
# ONE-CELL INTERVENTION -- take the shipped grid, change cell c alone, re-read
# the whole grid through `.nl_axis_quantiles()` via the gcol33/tulpa#322 harness,
# and score the result against a fine-grid reference fit of the same model.
# Comparing a cell's own contribution against the reference instead would
# discard exactly the nonlinearity the read is made of.
#
# Cell SELECTION is out of scope and settled: gcol33/tulpa#328 measured it and
# the header of `R/nested_laplace_joint_ccd_local.R` derives why the incumbent
# weight ranking is already the mass-movement ranking. This file is about which
# correction a cell already selected should get.
#
# Two things this file cannot see, both measured by gcol33/tulpa#331 and both
# bounding what a favourable plane here would have licensed. A one-cell
# intervention cannot see an AGGREGATE effect: the gradient points toward the
# peak in every cell, so the whole-grid barycentre moves every atom inward at
# once and the atom set contracts, which moving one atom does not do -- that
# rule covers `sigma_1` on 128 of 200 seeds at four levels against the shipped
# 200 of 200. And the distance to a fine-grid read, which is the metric every
# plane below is coloured by, does not track inferential quality on that
# fixture: the location rule is nearer the converged width (0.3576 against the
# shipped 0.4194) and still loses 72 seeds, because undershooting it costs and
# overshooting it does not. Nothing here is evidence that a correction should
# ship. Those figures are the flat prior's, from `dev_notes/issue331/coverage331.R`
# (the table in test-nested-laplace-recovery.R); under the proper prior the same
# sweep gives 128 of 200, 0.3546 against 0.4240, and 72 seeds.
#
# Every figure quoted below is measured by `dev_notes/issue333/plane333.R` and
# `analyse333.R` and is written flat / proper where both are given. "The test
# sweep" is `.dp_sweep()` itself: eight seeds at spread 3, 280 cells from 24
# fits. "The three-spread sweep" is the same units over spreads 2, 3 and 4, 840
# cells from 72 fits, and its binomial arm is the same layout on
# `ogd_fixture_sim(family = "binomial")`.

# Both descriptors of every cell that has a centred stencil, off the engine's
# own per-cell routines rather than a second reading of the same quadratic.
.dp_descr <- function(d) {
  n <- nrow(d$joint_grid); dd <- ncol(d$joint_grid)
  U <- matrix(0, n, dd)
  for (j in seq_len(dd)) {
    U[, j] <- tulpa:::.joint_pareto_fwd(d$axis_tags[j], d$joint_grid[, j])
  }
  nb <- tulpa:::.joint_local_ccd_neighbors(U, d$joint_grid, seq_len(dd))
  out <- data.frame(cell = seq_len(n), R_M = NA_real_, R_L = NA_real_,
                    R_L_max = NA_real_, R_L_A = NA_real_)
  for (c in seq_len(n)) {
    st <- tulpa:::.joint_local_ccd_cell_stencil(c, U, d$log_marginal, nb$up, nb$dn)
    if (is.null(st)) next
    bm <- tulpa:::.joint_local_ccd_cell_box_mass(st)
    bc <- tulpa:::.joint_local_ccd_cell_bary(st)
    sh <- ifelse(bc$u_bar >= 0, bc$u_bar / st$half_hi, -bc$u_bar / st$half_lo)
    sh[!is.finite(sh)] <- 0
    out$R_M[c]     <- abs(bm$log_box_ratio)
    out$R_L[c]     <- sqrt(sum(sh^2))
    out$R_L_max[c] <- bc$bary_shift
    out$R_L_A[c]   <- sqrt(sum(pmax(-st$d2, 0) * bc$u_bar^2))
  }
  out[!is.na(out$R_M), , drop = FALSE]
}

# The two whole-grid candidates, each in the form its kind of rule enters the
# harness in: the box multiplier as per-cell design weight, the barycentre as a
# perturbed coordinate matrix. `outer_grid_one_cell()` restricts either to one
# cell.
.dp_candidates <- function(d) {
  bm <- tulpa:::.joint_local_ccd_box_mass(d$joint_grid, d$log_marginal,
                                          d$axis_names, d$axis_tags)
  bc <- tulpa:::.joint_local_ccd_barycentre(d$joint_grid, d$log_marginal,
                                            d$axis_names, d$axis_tags)
  base <- if (is.null(d$dnode)) rep(1, nrow(d$joint_grid)) else d$dnode
  list(dnode = base * exp(bm$log_box_ratio), joint_grid = bc$joint_grid,
       weights = outer_grid_weights(d, dnode = base * exp(bm$log_box_ratio)))
}

# One dump's cells, each intervened on alone. `dL_*` is the improvement in the
# reported read against the reference (positive = the intervention moved the
# read toward it) and `res_*` whether the intervention moved the read at all by
# more than this grid resolves.
.dp_cells <- function(d, ref) {
  cand <- .dp_candidates(d)
  shipped <- outer_grid_rebuild(d)
  L0 <- outer_grid_read_diff(ref, shipped)
  fl <- outer_grid_noise_floor(d)
  ds <- .dp_descr(d)
  rows <- lapply(ds$cell, function(c) {
    arms <- list(
      mass = outer_grid_one_cell(d, c, dnode = cand$dnode),
      loc  = outer_grid_one_cell(d, c, joint_grid = cand$joint_grid),
      both = outer_grid_one_cell(d, c, dnode = cand$dnode,
                                 joint_grid = cand$joint_grid))
    row <- list(cell = c)
    for (nm in names(arms)) {
      rd <- outer_grid_rebuild(d, arms[[nm]]$weights, arms[[nm]]$joint_grid)
      e  <- outer_grid_read_diff(ref, rd)
      mv <- outer_grid_read_diff(shipped, rd)
      for (p in names(OGD_PARTS)) {
        row[[paste0("dL_", nm, "_", p)]]  <- L0[[p]] - e[[p]]
        row[[paste0("res_", nm, "_", p)]] <- isTRUE(mv[[p]] > fl[[p]])
      }
    }
    as.data.frame(row)
  })
  ds$w <- d$weights[ds$cell]
  cells <- cbind(ds, do.call(rbind, rows)[, -1L, drop = FALSE])
  whole <- lapply(names(OGD_PARTS), function(p) {
    e <- function(w, g) outer_grid_read_diff(ref, outer_grid_rebuild(d, w, g))[[p]]
    data.frame(part = p, shipped = L0[[p]], floor = fl[[p]],
               mass = e(cand$weights, NULL), loc = e(NULL, cand$joint_grid),
               both = e(cand$weights, cand$joint_grid))
  })
  list(cells = cells, whole = do.call(rbind, whole))
}

# Which single-cell intervention this cell's read prefers. Only interventions
# this grid can resolve are eligible: an unresolved difference is not a colour.
# `none` is the eligible-but-nothing-helps cell, which the partition the issue
# proposes has its own quadrant for.
.dp_winner <- function(df, p) {
  nms <- c("mass", "loc", "both")
  dL  <- as.matrix(df[, paste0("dL_", nms, "_", p)])
  res <- as.matrix(df[, paste0("res_", nms, "_", p)])
  vapply(seq_len(nrow(df)), function(i) {
    k <- which(res[i, ])
    if (!length(k)) return("unresolved")
    b <- k[which.max(dL[i, k])]
    if (dL[i, b] <= 0) "none" else nms[b]
  }, character(1))
}

# The four regimes of the issue's table, on a within-resolution median split of
# whichever location descriptor is being read.
.dp_quad <- function(df, rl = "R_L") {
  m <- df$R_M > stats::median(df$R_M)
  l <- df[[rl]] > stats::median(df[[rl]])
  ifelse(m & l, "M+L+", ifelse(m & !l, "M+L-", ifelse(!m & l, "M-L+", "M-L-")))
}

# How much a per-cell rule reading the plane could buy over ignoring it: the
# accuracy of the best label per quadrant against the accuracy of the single
# best label overall. Zero means the winner is the same in every quadrant, i.e.
# the plane position selects nothing.
.dp_quad_gain <- function(df, p, rl = "R_L") {
  s <- df[df[[paste0("win_", p)]] != "unresolved", , drop = FALSE]
  if (nrow(s) < 20L) return(NULL)
  tb <- table(.dp_quad(s, rl), s[[paste0("win_", p)]])
  majority <- max(table(s[[paste0("win_", p)]])) / nrow(s)
  list(n = nrow(s), majority = majority, table = tb,
       label = names(which.max(table(s[[paste0("win_", p)]]))),
       quadrant = sum(apply(tb, 1L, max)) / sum(tb),
       gain = sum(apply(tb, 1L, max)) / sum(tb) - majority)
}

# The experiment is about the outer grid's own resolution of the marginal
# likelihood, so every fit states a flat outer prior.
.DP_HYPERPRIOR <- "flat"

# One seed of the fixture at one grid spread: a four- and a five-level base,
# each intervened on cell by cell and scored against a twelve-level (1728-cell)
# reference of the same data.
.dp_sweep_unit <- function(seed, spread = 3, family = "gaussian",
                           hyperprior = .DP_HYPERPRIOR,
                           within_cell = formals(ogd_fixture_fit)$within_cell) {
  sim <- ogd_fixture_sim(c(0.8, 0.5, 0.3), seed, family = family)
  fit <- function(lv) ogd_fixture_fit(sim, lv, spread, within_cell = within_cell,
                                      hyperprior = hyperprior)
  ref <- outer_grid_rebuild(outer_grid_dump(fit(12L)))
  lapply(c(4L, 5L), function(lv) {
    r <- .dp_cells(outer_grid_dump(fit(lv)), ref)
    tag <- function(df) {
      df$seed <- seed; df$spread <- spread; df$levels <- lv; df$family <- family
      df
    }
    list(cells = tag(r$cells), whole = tag(r$whole))
  })
}

# Units from `.dp_sweep_unit()` joined, with each cell's winning intervention
# per part.
.dp_combine <- function(units, n_fits) {
  parts <- unlist(units, recursive = FALSE)
  D <- do.call(rbind, lapply(parts, `[[`, "cells"))
  for (p in names(OGD_PARTS)) D[[paste0("win_", p)]] <- .dp_winner(D, p)
  list(cells = D, whole = do.call(rbind, lapply(parts, `[[`, "whole")),
       n_fits = n_fits)
}

# The sweep this file asserts on: eight seeds at spread 3, 3 x 8 gaussian fits,
# computed once. Every read below is post-processing of them.
# `dev_notes/issue333/plane333.R` runs the same units over three spreads and a
# binomial arm in parallel.
.dp_cache <- new.env(parent = emptyenv())
.dp_sweep <- function(seeds = 1:8, spreads = 3, family = "gaussian",
                      hyperprior = .DP_HYPERPRIOR) {
  key <- paste(max(seeds), paste(spreads, collapse = "_"), family, hyperprior)
  if (!is.null(.dp_cache[[key]])) return(.dp_cache[[key]])
  grid <- expand.grid(seed = seeds, spread = spreads)
  units <- lapply(seq_len(nrow(grid)), function(i)
    .dp_sweep_unit(grid$seed[i], grid$spread[i], family, hyperprior))
  out <- .dp_combine(units, 3L * nrow(grid))
  .dp_cache[[key]] <- out
  out
}

# --------------------------------------------------------------------------- #
# The one-cell intervention                                                   #
# --------------------------------------------------------------------------- #

test_that("a one-cell candidate differs from the fit's own state in one cell", {
  skip_on_cran()
  d <- outer_grid_dump(ogd_fixture_fit(ogd_fixture_sim(c(0.8, 0.5, 0.3), 1L), 4L,
                                       hyperprior = .DP_HYPERPRIOR))
  cand <- .dp_candidates(d)
  ds <- .dp_descr(d)
  c1 <- ds$cell[which.max(ds$R_M)]

  oc <- outer_grid_one_cell(d, c1, dnode = cand$dnode, joint_grid = cand$joint_grid)
  # The coordinates differ from the dump's in exactly the intervened row, and
  # from the whole-grid candidate's in every OTHER moved row.
  moved <- which(rowSums(abs(oc$joint_grid - d$joint_grid)) > 0)
  expect_identical(moved, c1)
  expect_equal(oc$joint_grid[c1, ], cand$joint_grid[c1, ], tolerance = 0)
  expect_gt(sum(rowSums(abs(cand$joint_grid - d$joint_grid)) > 0), 1L)

  # The weights are the fit's own softmax with one design weight replaced, so
  # the ratio between any two UNintervened cells is untouched: a one-cell
  # intervention redistributes the normalisation and nothing else.
  r_ship <- d$weights[-c1] / sum(d$weights[-c1])
  r_oc   <- oc$weights[-c1] / sum(oc$weights[-c1])
  expect_equal(r_oc, r_ship, tolerance = 1e-12)
  expect_false(isTRUE(all.equal(oc$weights[c1], d$weights[c1])))

  # Intervening on a cell the rule declines to move leaves the read exactly as
  # the fit shipped it, which is what makes a difference attributable to the
  # cell rather than to the harness.
  still <- setdiff(seq_len(nrow(d$joint_grid)), ds$cell)[1L]
  q0 <- outer_grid_rebuild(d)
  q1 <- outer_grid_rebuild(
    d, NULL, outer_grid_one_cell(d, still, joint_grid = cand$joint_grid)$joint_grid)
  expect_equal(unlist(q1), unlist(q0), tolerance = 1e-14)

  expect_error(outer_grid_one_cell(d, 0L, dnode = cand$dnode), "cell index")
  expect_error(outer_grid_one_cell(d, 1L, dnode = cand$dnode[-1L]), "length")
  expect_error(outer_grid_one_cell(d, 1L, joint_grid = d$joint_grid[-1L, ]), "cell")
})

# --------------------------------------------------------------------------- #
# Item 3: how much does the box truncation separate the two descriptors?      #
# --------------------------------------------------------------------------- #

test_that("the two descriptors separate, but they are far from independent", {
  skip_on_cran()
  D <- .dp_sweep()$cells

  # Measured over the three-spread sweep: Spearman(R_M, R_L) 0.9225 / 0.9224
  # pooled, 0.9211 / 0.9206 at four levels and 0.9251 / 0.9248 at five, per-fit
  # median 0.9286 / 0.9524. A binomial arm on the same layout gives 0.9017 /
  # 0.9094. On the test sweep: 0.8540 / 0.8544 pooled, 0.8765 / 0.8785 and
  # 0.8512 / 0.8518, per-fit median 0.8855 / 0.8929. So the truncation does
  # separate them -- 0.2738 / 0.2702 of cells sit off the quartile diagonal
  # (0.4000 on the test sweep under both) -- but the second dimension is thin,
  # and it is thin in a specific way: `R_L` is the share of its own half-cell an
  # atom moves and so saturates below `sqrt(d)`, while `R_M` is a log mass ratio
  # in nats and does not. Over the three-spread sweep `R_M` spans 0.0085 to
  # 44.3781 (0.0165 to 44.8453 proper) while `R_L` spans 0.0776 to 1.5950
  # (0.0787 to 1.5966) against a bound of 1.7321, and no cell reaches 0.95 of
  # it; on the test sweep `R_M` reaches 24.7168 and `R_L` 1.5250.
  for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv, ]
    rho <- stats::cor(s$R_M, s$R_L, method = "spearman")
    expect_gt(rho, 0.7)
    expect_lt(rho, 0.98)
  }
  expect_gt(max(D$R_M), 10 * max(D$R_L))
  expect_lt(max(D$R_L), sqrt(3))
  expect_lt(max(D$R_L_max), 1)

  # In the A metric -- the unbounded limit, where both descriptors reduce to
  # functions of `A^-1 g` -- the separation narrows within each resolution
  # (0.9558 / 0.9552 at four levels and 0.9307 / 0.9332 at five over the
  # three-spread sweep; 0.9555 / 0.9523 and 0.9151 / 0.9173 on the test sweep,
  # 0.8791 / 0.8778 pooled). Pooled over the three spreads it does not (0.9139 /
  # 0.9158 against the truncated 0.9225 / 0.9224): what the A metric removes is
  # the within-resolution part of the separation, the truncation, and pooling
  # grids of different steepness adds a separation of its own.
  expect_gt(stats::cor(D$R_M, D$R_L_A, method = "spearman"),
            stats::cor(D$R_M, D$R_L, method = "spearman"))

  qm <- cut(D$R_M, stats::quantile(D$R_M, 0:4 / 4), include.lowest = TRUE)
  ql <- cut(D$R_L, stats::quantile(D$R_L, 0:4 / 4), include.lowest = TRUE)
  expect_gt(1 - sum(diag(table(qm, ql))) / nrow(D), 0.25)
})

test_that("the two base resolutions occupy the same region of the plane", {
  skip_on_cran()
  # The issue's mechanism for gcol33/tulpa#327's resolution-dependent ranking is
  # that the four- and five-level grids put their cells in DIFFERENT regions of
  # the plane. They do not. Over the three-spread sweep the median `R_M` is
  # 5.2490 / 5.4958 at four levels against 5.0401 / 5.1759 at five (Mann-Whitney
  # p 0.9219 / 0.9002) and the median `R_L` 1.0994 / 1.1014 against 1.0982 /
  # 1.1015 (p 0.2408 / 0.2577, a shift under 0.2%), and the quadrant occupancy
  # (M+L+ / M+L- / M-L+ / M-L-) is 44.3 / 5.7 / 5.7 / 44.3 percent at four
  # levels under both priors against 44.9 / 5.1 / 5.1 / 44.9 (45.2 / 4.8 / 4.8 /
  # 45.2) at five -- every quadrant holds the same share at both resolutions to
  # within 0.9 points. The binomial arm agrees (p 0.6710 / 0.7432 on `R_M`,
  # 0.3476 / 0.4225 on `R_L`), and so does the test sweep (p 0.9111 / 0.9265 and
  # 0.2993 / 0.3059, quadrant shares within 0.7 points).
  D <- .dp_sweep()$cells
  d4 <- D[D$levels == 4L, ]; d5 <- D[D$levels == 5L, ]
  expect_lt(abs(stats::median(d4$R_M) / stats::median(d5$R_M) - 1), 0.25)
  expect_lt(abs(stats::median(d4$R_L) / stats::median(d5$R_L) - 1), 0.15)

  # Where the resolutions genuinely differ is how much WEIGHT they put on the
  # steep cells, not where the cells sit: the integration-weighted mean `R_M` is
  # 2.4486 / 2.5309 at four levels against 0.7656 / 0.7682 at five over the
  # three-spread sweep, and 2.2745 / 2.3143 against 0.6773 / 0.6996 on the test
  # sweep, a 3.2- to 3.4-fold gap on a population whose unweighted medians
  # agree. That is a one-dimensional
  # statement about steepness, not a region of a two-dimensional plane.
  wm <- function(s, v) sum(s[[v]] * s$w) / sum(s$w)
  expect_gt(wm(d4, "R_M"), 2 * wm(d5, "R_M"))
  expect_gt(wm(d4, "R_L"), wm(d5, "R_L"))
})

# --------------------------------------------------------------------------- #
# The three colour planes                                                     #
# --------------------------------------------------------------------------- #

test_that("a one-cell intervention on the interval is resolvable on the coarse grid only", {
  skip_on_cran()
  # gcol33/tulpa#328 found the four-level grid resolves a whole RANKING change
  # on the interval rarely: under the `chord` read in none of 24 configurations
  # on the endpoints or the widths while resolving the median in 16 of 24 (15
  # proper), under the shipped read the endpoints in 2 / 1, the widths in 2 / 5
  # and the median in 7 / 8 of 24 (`dev_notes/issue_328/measure_fit_ranking.R`).
  # A one-cell intervention reaches the interval far more often on that grid,
  # because a four-level base has only eight interior cells and moving one of
  # eight is not a small perturbation: over the test sweep 29.7% / 39.1% /
  # 90.6% of four-level cells (endpoints / widths / median) have at least one
  # resolvable intervention against 0.0% / 0.5% / 69.9% of five-level ones
  # (proper: 32.8% / 34.4% / 90.6% against 0.0% / 0.5% / 69.0%; three-spread
  # sweep 27.1% / 31.3% / 80.7% against 0.9% / 1.1% / 71.0% flat). So the
  # interval is read at four levels only, and the median at both. Under `chord`
  # the five-level median is resolvable in 16.2% / 12.5% of test-sweep cells and
  # 22.7% of three-spread cells (flat), which is where a coarse-grid-only reading
  # of the median comes from.
  D <- .dp_sweep()$cells
  res <- function(s, p) mean(s[[paste0("res_mass_", p)]] |
                             s[[paste0("res_loc_", p)]] |
                             s[[paste0("res_both_", p)]])
  d4 <- D[D$levels == 4L, ]; d5 <- D[D$levels == 5L, ]
  expect_gt(res(d4, "endpoints"), 0.2)
  expect_gt(res(d4, "median"), 0.8)
  expect_lt(res(d5, "endpoints"), 0.2)
  expect_gt(res(d5, "median"), res(d5, "endpoints"))
})

test_that("the plane position does not select which correction a cell wants", {
  skip_on_cran()
  D <- .dp_sweep()$cells

  # The colour is PER PART: the three parts of the read disagreed on the winner
  # for the whole grid in gcol33/tulpa#327 and they disagree per cell here too.
  # Over the sweep the best label per quadrant buys nothing over the single best
  # label overall: the gain is exactly +0.0000 in 4 of the 9 scored
  # combinations of the test sweep (three `R_L` variants x the
  # part-by-resolution strata with at least 20 resolved cells: the widths and
  # the median at four levels, the median at five; the four-level endpoints
  # resolve 19 cells), 7 of 12 under the proper prior, where they resolve 21.
  #
  # `gain` is scored IN SAMPLE -- the per-quadrant argmax is read off the same
  # cells -- so its null is not 0 and it grows as the stratum shrinks and as the
  # labels even out: on `widths` at four levels, four quadrants over 25 resolved
  # cells, picking the best label in each bin buys 0.0746 from noise alone on
  # average and the null's 95th percentile is 0.1600. A fixed ceiling cannot
  # separate that from signal at one n and be meaningful at another, so each
  # gain is scored against ITS OWN permutation null (quadrant labels shuffled,
  # winners held). Measured: the non-zero gains are on the four-level widths --
  # 0.1600 / 0.0800 / 0.0800 under `R_L` / `R_L_max` / `R_L_A` at p = 0.13 /
  # 0.51 / 0.21 -- and the four-level median, 0.0345 / 0.0517 under `R_L_max` /
  # `R_L_A` at p = 0.17 / 0.067, flat; under the proper prior the smallest p is
  # 0.008 (the four-level median under `R_L_A`, gain 0.0690). The three-spread
  # sweep has one stratum where the plane does select: the four-level median,
  # gain 0.0968 / 0.0710 / 0.1355 flat and 0.1220 / 0.1220 / 0.1890 proper, each
  # at permutation p < 0.001; its other scored strata are zero or
  # non-significant but one (0.0022 on the five-level median under `R_L_A`,
  # proper), and the binomial arm scores zero on all six.
  #
  # What the plane does carry is a shift in the MIX -- over the three-spread
  # sweep the median plane's contingency is significant at both resolutions
  # (chi-square p 3.1e-04 / 2.4e-05 at four levels, Cramer V 0.2577 / 0.2750;
  # p 8.1e-04 / 1.0e-03 at five, V 0.1435 / 0.1424) -- but at five levels the
  # argmax is `none` in every quadrant, which is what a per-cell rule would have
  # to move. At four levels it is `none` in the low-`R_M`, low-`R_L` quadrant
  # and `both` in the other three under either prior, which is the selecting
  # stratum above. The test sweep shows no significant shift in the mix
  # (p 0.576 / 0.167 at four levels and 0.512 / 0.070 at five).
  gain_at <- function(q, wins) {
    tb <- table(q, wins)
    sum(apply(tb, 1L, max)) / sum(tb) - max(table(wins)) / length(wins)
  }
  set.seed(327L)
  n_perm <- 999L
  p_perm <- numeric(0)
  for (p in names(OGD_PARTS)) for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv, ]
    s <- s[s[[paste0("win_", p)]] != "unresolved", , drop = FALSE]
    if (nrow(s) < 20L) next
    wins <- s[[paste0("win_", p)]]
    for (rl in c("R_L", "R_L_max", "R_L_A")) {
      q <- .dp_quad(s, rl)
      obs <- gain_at(q, wins)
      null <- replicate(n_perm, gain_at(sample(q), wins))
      p_perm <- c(p_perm, mean(null >= obs))
    }
  }
  # Bonferroni over the scored combinations at a family alpha of 0.05. A plane
  # that genuinely selected the correction would clear this on the strata where
  # the grid resolves the question, not miss it on every one.
  expect_gte(length(p_perm), 9L)
  expect_gt(min(p_perm), 0.05 / length(p_perm))

  # And the winner is dominated by one label wherever the grid resolves the
  # question at all. At four levels that label is `both` on the endpoints
  # (1.000 / 0.952 of resolved cells over the test sweep) and the median
  # (0.586 / 0.586); on the widths it is `mass` (0.48) flat and `both` (0.591)
  # proper. At five levels the interval has at most one resolved cell per part
  # and the median's label is `none` (0.775 / 0.758).
  s4 <- D[D$levels == 4L & D$win_endpoints != "unresolved", ]
  expect_gt(mean(s4$win_endpoints == "both"), 0.7)
})

test_that("the loc-versus-mass preference does not partition the plane", {
  skip_on_cran()
  # The partition the issue proposes is directional: large `R_L` and small `R_M`
  # should want the location correction, large `R_M` and small `R_L` the mass
  # one. The direct test of that is whether a cell's relative position between
  # the two descriptors predicts which of the two single corrections helps more.
  #
  # It does so only weakly, and not with one sign. Over the test sweep, among
  # cells where either single correction is resolved, Spearman(rank R_L - rank
  # R_M, dL_loc - dL_mass) is -0.1246 / +0.1174 on the endpoints (17 / 17 cells,
  # p 0.63 / 0.65), -0.3234 / -0.3257 on the widths (18 / 11 cells, p 0.19 /
  # 0.33) and +0.2965 / +0.2544 on the median (55 / 57 cells, p 0.028 / 0.056)
  # at four levels, and +0.0681 / +0.0519 on the median at five (150 / 146
  # cells, p 0.41 / 0.53). The strongest, the four-level median, runs the way
  # the partition proposes, and the widths run the other way. The rank of `R_L`
  # with `R_M` partialled out leaves a residual signal that changes sign too
  # (-0.165, -0.231, +0.245, +0.106 flat over the same four strata). Under the
  # `chord` read with the flat prior the three-spread sweep's five-level median
  # runs AGAINST the partition, -0.1235 (p 0.15), against +0.1163 at the
  # four-level median; under the proper prior the two are -0.1451 (p 0.092) and
  # +0.1592. The binomial arm gives -0.1283 / -0.1158 on the five-level median
  # (p 0.079 / 0.082) under the shipped read.
  D <- .dp_sweep()$cells
  rho <- function(s, p) {
    s <- s[s[[paste0("res_mass_", p)]] | s[[paste0("res_loc_", p)]], ]
    if (nrow(s) < 25L) return(NA_real_)
    stats::cor(rank(s$R_L) - rank(s$R_M),
               s[[paste0("dL_loc_", p)]] - s[[paste0("dL_mass_", p)]],
               method = "spearman")
  }
  got <- unlist(lapply(c(4L, 5L), function(lv)
    vapply(names(OGD_PARTS), function(p) rho(D[D$levels == lv, ], p), numeric(1))))
  got <- got[is.finite(got)]
  # Measured under the read the engine ships (`ogd_fixture_fit()` states it
  # rather than inheriting it -- gcol33/tulpa#599): +0.297 on the four-level
  # median and +0.068 on the five-level median, with the other four
  # part-by-resolution cells carrying fewer than 25 scorable rows. No cell is a
  # usable rule on its own, and the typical one carries next to no signal.
  # Under the proper prior the same two cells give +0.254 and +0.052, a median
  # of 0.153.
  expect_gte(length(got), 2L)
  expect_lt(max(abs(got)), 0.6)
  expect_lt(stats::median(abs(got)), 0.2)
  # Which way the small ones lean moves with the within-cell read: the row
  # filter is the per-part floor, and the floor is read-dependent. Under `chord`
  # two cells clear the row count and the pair spans zero (+0.179 on the
  # four-level median, -0.032 on the five-level one); under the shipped read
  # both lean positive. What survives both reads is the magnitude, which is the
  # claim: a per-cell label that correlates with the preferred correction at
  # |rho| < 0.4 does not partition the grid.
})

# --------------------------------------------------------------------------- #
# Why a per-cell label could not be composed into a grid rule anyway           #
# --------------------------------------------------------------------------- #

test_that("one-cell improvements do not add up to the whole-grid improvement", {
  skip_on_cran()
  # The design correction this file is built on, measured. A weighted quantile
  # is not a sum over its atoms, so the improvements of the individual one-cell
  # interventions have no reason to compose -- and they do not. Over the test
  # sweep, averaged over seeds, the sum of the per-cell `both` improvements is
  # 6.26x / 16.65x the whole-grid `both` improvement on the endpoints at four /
  # five levels (proper 6.32x / 16.68x), 10.89x / 14.29x on the widths (11.26x /
  # 14.30x; at five levels both are losses, -0.3357 against -0.0202 and -1.0144
  # against -0.0710 flat), and 4.35x (4.21x) at four levels on the median, where
  # at five levels the two carry OPPOSITE SIGNS (-0.3889 summed against +0.0140
  # whole-grid flat, -0.3851 against +0.0132 proper). The three-spread sweep
  # repeats the five-level median sign split (-0.3956 against +0.0075 flat), and
  # its binomial arm carries opposite signs on the five-level interval, -21.2x
  # on the endpoints and -6.4x on the widths flat (-241.5x and -14.6x
  # proper).
  #
  # So a clean per-cell partition, had there been one, still could not have been
  # applied cell by cell and read off as a grid rule.
  z <- .dp_sweep()
  D <- z$cells; W <- z$whole
  ratio <- function(p, lv) {
    per <- vapply(split(D[D$levels == lv, ], D$seed[D$levels == lv]),
                  function(s) sum(s[[paste0("dL_both_", p)]]), numeric(1))
    w <- W[W$part == p & W$levels == lv, ]
    c(per = mean(per), whole = mean(w$shipped - w$both))
  }
  e4 <- ratio("endpoints", 4L)
  expect_gt(e4[["per"]], 4 * e4[["whole"]])
  w4 <- ratio("widths", 4L)
  expect_gt(w4[["per"]], 4 * w4[["whole"]])
  # The five-level median is the sign disagreement: correcting every cell helps
  # the reported median while the individual corrections, summed, do not.
  m5 <- ratio("median", 5L)
  expect_lt(m5[["per"]], 0)
  expect_gt(m5[["whole"]], 0)
})
