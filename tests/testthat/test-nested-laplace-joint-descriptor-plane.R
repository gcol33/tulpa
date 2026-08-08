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
# plane below is coloured by, does not track inferential quality on this
# fixture: the location rule is nearer the converged width (0.3576 against the
# shipped 0.4194) and still loses 72 seeds, because undershooting it costs and
# overshooting it does not. Nothing here is evidence that a correction should
# ship.

.dp_sim <- function(sd_true, seed, G = 30L, N = 600L) {
  set.seed(seed)
  grp <- lapply(seq_along(sd_true), function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, stats::rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in seq_along(sd_true)) {
    eta <- eta + stats::rnorm(G, 0, sd_true[k])[grp[[k]]]
  }
  list(y = eta + stats::rnorm(N, 0, 0.5), X = X, grp = grp, N = N, G = G,
       sd_true = sd_true)
}

.dp_fit <- function(sim, levels, spread = 3) {
  prior <- lapply(seq_along(sim$grp), function(k) {
    s <- sim$sd_true[k]
    list(type = "iid", obs_idx = list(sim$grp[[k]]), n_units = sim$G,
         sigma_grid = exp(seq(log(s / spread), log(s * spread),
                              length.out = levels)))
  })
  suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = sim$y, n_trials = rep(1L, sim$N), X = sim$X,
                              family = "gaussian", phi = 0.25)),
    prior = prior,
    control = list(n_threads = 1L, diagnose_k = FALSE, max_iter = 100L,
                   tol = 1e-8, integration = "grid")))
}

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

# Eight seeds of the same fixture at two base resolutions against a 1728-cell
# reference, computed once. Every fit is a joint fit of the same model; the
# 3 x 8 of them are the whole cost of this file and every read below is
# post-processing of them.
.dp_cache <- new.env(parent = emptyenv())
.dp_sweep <- function(seeds = 1:8) {
  key <- paste0("s", max(seeds))
  if (!is.null(.dp_cache[[key]])) return(.dp_cache[[key]])
  cells <- list(); whole <- list()
  for (sd_ in seeds) {
    sim <- .dp_sim(c(0.8, 0.5, 0.3), sd_)
    ref <- outer_grid_rebuild(outer_grid_dump(.dp_fit(sim, 12L)))
    for (lv in c(4L, 5L)) {
      r <- .dp_cells(outer_grid_dump(.dp_fit(sim, lv)), ref)
      r$cells$seed <- sd_; r$cells$levels <- lv
      r$whole$seed <- sd_; r$whole$levels <- lv
      cells[[length(cells) + 1L]] <- r$cells
      whole[[length(whole) + 1L]] <- r$whole
    }
  }
  D <- do.call(rbind, cells)
  for (p in names(OGD_PARTS)) D[[paste0("win_", p)]] <- .dp_winner(D, p)
  out <- list(cells = D, whole = do.call(rbind, whole), n_fits = 3L * length(seeds))
  .dp_cache[[key]] <- out
  out
}

# --------------------------------------------------------------------------- #
# The one-cell intervention                                                   #
# --------------------------------------------------------------------------- #

test_that("a one-cell candidate differs from the fit's own state in one cell", {
  skip_on_cran()
  d <- outer_grid_dump(.dp_fit(.dp_sim(c(0.8, 0.5, 0.3), 1L), 4L))
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

  # Measured over 1680 cells from 144 fits (8 seeds x 3 grid spreads, four- and
  # five-level bases against a twelve-level reference): Spearman(R_M, R_L)
  # 0.8496 pooled, 0.8822 at four levels and 0.8505 at five, per-fit median
  # 0.9048. A binomial arm on the same layout (840 cells, 72 fits) gives 0.9072.
  # So the truncation does separate them -- 0.3881 of cells sit off the
  # quartile diagonal -- but the second dimension is thin, and it is thin in a
  # specific way: `R_L` is the share of its own half-cell an atom moves and so
  # saturates below `sqrt(d)`, while `R_M` is a log mass ratio in nats and does
  # not. Over that sweep `R_M` spans 0.0208 to 78.2587 while `R_L` spans 0.2261
  # to 1.6336 against a bound of 1.7321, and no cell reaches 0.99 of it.
  for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv, ]
    rho <- stats::cor(s$R_M, s$R_L, method = "spearman")
    expect_gt(rho, 0.7)
    expect_lt(rho, 0.98)
  }
  expect_gt(max(D$R_M), 30)
  expect_lt(max(D$R_L), sqrt(3))
  expect_lt(max(D$R_L_max), 1)

  # In the A metric -- the unbounded limit, where both descriptors reduce to
  # functions of `A^-1 g` -- the separation all but disappears (0.9495 pooled
  # over the same sweep, 0.9830 at four levels), so what separates them IS the
  # truncation and nothing else.
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
  # the plane. They do not. Over the 144-fit sweep the median `R_M` is 11.7853
  # at four levels against 10.2645 at five (Mann-Whitney p 0.1424) and the
  # median `R_L` 1.1694 against 1.2105 (p 0.0012 on a 3.5% shift), and the
  # quadrant occupancy is 43.2 / 2.6 / 12.8 / 41.4 percent against 41.0 / 10.3 /
  # 7.3 / 41.5 -- the two populated quadrants hold the same share at both
  # resolutions and only the two sparse off-diagonal ones trade. The binomial
  # arm agrees (p 0.4213 / 0.5318).
  D <- .dp_sweep()$cells
  d4 <- D[D$levels == 4L, ]; d5 <- D[D$levels == 5L, ]
  expect_lt(abs(stats::median(d4$R_M) / stats::median(d5$R_M) - 1), 0.25)
  expect_lt(abs(stats::median(d4$R_L) / stats::median(d5$R_L) - 1), 0.15)

  # Where the resolutions genuinely differ is how much WEIGHT they put on the
  # steep cells, not where the cells sit: the integration-weighted mean `R_M` is
  # 3.3417 at four levels against 1.0483 at five, a threefold gap on a
  # population whose unweighted medians agree. That is a one-dimensional
  # statement about steepness, not a region of a two-dimensional plane.
  wm <- function(s, v) sum(s[[v]] * s$w) / sum(s$w)
  expect_gt(wm(d4, "R_M"), 2 * wm(d5, "R_M"))
  expect_gt(wm(d4, "R_L"), wm(d5, "R_L"))
})

# --------------------------------------------------------------------------- #
# The three colour planes                                                     #
# --------------------------------------------------------------------------- #

test_that("a one-cell intervention is resolvable on the coarse grid only", {
  skip_on_cran()
  # gcol33/tulpa#328 found the four-level grid could not resolve a whole RANKING
  # change on the endpoints or the widths while resolving the median in 14 of
  # 24. A one-cell intervention is the opposite way round on that grid, because
  # a four-level base has only eight interior cells and moving one of eight is
  # not a small perturbation: over the 144-fit sweep 64.8% / 64.1% / 94.5% of
  # four-level cells have at least one resolvable intervention against 7.6% /
  # 7.6% / 33.3% of five-level ones. So the plane is read where it can be read,
  # and the five-level interval is not read at all.
  D <- .dp_sweep()$cells
  res <- function(s, p) mean(s[[paste0("res_mass_", p)]] |
                             s[[paste0("res_loc_", p)]] |
                             s[[paste0("res_both_", p)]])
  d4 <- D[D$levels == 4L, ]; d5 <- D[D$levels == 5L, ]
  expect_gt(res(d4, "endpoints"), 0.4)
  expect_gt(res(d4, "median"), 0.8)
  expect_lt(res(d5, "endpoints"), 0.2)
  expect_gt(res(d5, "median"), res(d5, "endpoints"))
})

test_that("the plane position does not select which correction a cell wants", {
  skip_on_cran()
  D <- .dp_sweep()$cells

  # The colour is PER PART: the three parts of the read disagreed on the winner
  # for the whole grid in gcol33/tulpa#327 and they disagree per cell here too.
  # Measured over the 144-fit sweep, the best label per quadrant buys nothing
  # over the single best label overall: the gain is exactly +0.0000 in 23 of the
  # 30 scored combinations (three `R_L` variants x two resolvability gates x the
  # parts with enough resolved cells) and never exceeds +0.0256. A 3 x 3 tercile
  # partition instead of the 2 x 2 median split does not help either (largest
  # gain +0.0408, on 98 cells).
  #
  # What the plane does carry is a shift in the MIX -- the median plane's
  # contingency is significant at both resolutions (p < 1e-4, Cramer V 0.2078 /
  # 0.1882) -- but the argmax is the same in every quadrant, which is what a
  # per-cell rule would have to move.
  for (p in names(OGD_PARTS)) for (lv in c(4L, 5L)) {
    s <- D[D$levels == lv, ]
    for (rl in c("R_L", "R_L_max", "R_L_A")) {
      g <- .dp_quad_gain(s, p, rl)
      if (is.null(g)) next
      expect_lt(g$gain, 0.10)
    }
  }

  # And the winner is dominated by one label wherever the grid resolves the
  # question at all. At four levels that label is `both` on all three parts
  # (0.932 / 0.919 / 0.785 of resolved cells over the sweep); at five levels it
  # is `both` on the endpoints, `loc` on the widths and `none` on the median.
  s4 <- D[D$levels == 4L & D$win_endpoints != "unresolved", ]
  expect_gt(mean(s4$win_endpoints == "both"), 0.7)
})

test_that("the loc-versus-mass preference runs against the proposed partition", {
  skip_on_cran()
  # The partition the issue proposes is directional: large `R_L` and small `R_M`
  # should want the location correction, large `R_M` and small `R_L` the mass
  # one. The direct test of that is whether a cell's relative position between
  # the two descriptors predicts which of the two single corrections helps more.
  #
  # It does not, and where the association is strongest it has the WRONG SIGN.
  # Over the 144-fit sweep, Spearman(rank R_L - rank R_M, dL_loc - dL_mass) is
  # +0.1835 / +0.0283 on the endpoints, -0.0595 / +0.1303 on the widths and
  # +0.1230 / -0.3178 on the median at four / five levels. The largest of them
  # is the five-level median at -0.3178 (p < 1e-4): cells with relatively more
  # location displacement prefer the MASS correction there. The binomial arm
  # reproduces that sign and size (-0.3057, p 0.0275). Partialling `R_M` out
  # leaves the residual `R_L` signal with no stable sign either (+0.031, +0.063,
  # -0.280, +0.012, +0.144, -0.269 over the six part-by-resolution cells).
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
  expect_gt(length(got), 2L)
  # No cell of the six is a usable rule on its own, and the set does not agree
  # on a direction.
  expect_lt(max(abs(got)), 0.6)
  expect_true(any(got < 0))
})

# --------------------------------------------------------------------------- #
# Why a per-cell label could not be composed into a grid rule anyway           #
# --------------------------------------------------------------------------- #

test_that("one-cell improvements do not add up to the whole-grid improvement", {
  skip_on_cran()
  # The design correction this file is built on, measured. A weighted quantile
  # is not a sum over its atoms, so the improvements of the individual one-cell
  # interventions have no reason to compose -- and they do not. Over the 144-fit
  # sweep the sum of the per-cell `both` improvements exceeds the whole-grid
  # `both` improvement by 8.48x / 33.11x on the endpoints, 9.15x / 68.58x on the
  # widths, and 6.23x at four levels on the median, where at five levels the two
  # carry OPPOSITE SIGNS (-0.0548 summed against +0.0040 whole-grid). The
  # binomial arm reaches -136x and -668x on the five-level interval.
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
