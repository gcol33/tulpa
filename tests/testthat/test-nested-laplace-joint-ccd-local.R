# Local CCD refinement of the joint multi-block outer grid (gcol33/tulpa#64):
# total-weight conservation, coarse-grid moment recovery, mutually-non-adjacent
# cell selection, and the decline gates. Pure-R with a synthetic closed-form
# evaluator (no inner solve), so these are tier-1 structural checks.

# Diagonal-Gaussian outer log-marginal in identity coordinates.
.lccd_true_lm <- function(theta, mu, s) {
  z <- sweep(theta, 2L, mu, "-")
  -0.5 * rowSums(sweep(z^2, 2L, s^2, "/"))
}

# Tensor base grid + its closed-form log-marginal.
.lccd_grid <- function(levels_list, mu, s) {
  g <- as.matrix(expand.grid(levels_list, KEEP.OUT.ATTRS = FALSE))
  colnames(g) <- names(levels_list)
  list(grid = g, lm = .lccd_true_lm(g, mu, s))
}

# eval_nodes contract: the same closed form, ignoring the warm start.
.lccd_eval <- function(mu, s) {
  function(theta_mat, warm) {
    list(log_marginal = .lccd_true_lm(theta_mat, mu, s), modes = NULL)
  }
}

.lccd_wmean <- function(x, w) sum(w * x) / sum(w)
.lccd_wsd <- function(x, w) {
  m <- .lccd_wmean(x, w)
  sqrt(sum(w * (x - m)^2) / sum(w))
}

test_that("ccd_weights are a partition of unity (the conservation footing)", {
  for (d in 2:5) {
    ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
    expect_equal(sum(ccd_weights(ccd)), 1, tolerance = 1e-12)
  }
})

test_that("local CCD refinement conserves the total outer design weight", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  lv <- list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2))
  gd <- .lccd_grid(lv, mu, s)
  out <- tulpa:::.joint_local_ccd_refine(
    joint_grid = gd$grid, log_marginal = gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = .lccd_eval(mu, s), max_cells = 4L)
  expect_false(is.null(out))
  # Uniform tensor base: sum(dnode) == n_cells. Each refined cell is replaced by
  # a block whose design weights sum back to the cell's, so the total is fixed.
  expect_equal(sum(out$dnode), nrow(gd$grid), tolerance = 1e-9)
  expect_true(out$info$n_cells_refined >= 1L)
  expect_true(out$info$n_nodes_added > 0L)
})

test_that("local CCD repopulates a coarse peak cell, recovering collapsed SD", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  lv <- list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2))
  gd <- .lccd_grid(lv, mu, s)
  tags <- c(a = "identity", b = "identity")

  w0 <- tulpa:::.joint_integration_weights(gd$lm, NULL)
  sd_coarse <- .lccd_wsd(gd$grid[, "a"], w0)

  # Dense reference on the same support: the moment a fine grid would report.
  fine <- seq(-1, 2, length.out = 61L)
  gdf <- .lccd_grid(list(a = fine, b = fine), mu, s)
  wf  <- tulpa:::.joint_integration_weights(gdf$lm, NULL)
  sd_dense <- .lccd_wsd(gdf$grid[, "a"], wf)

  out <- tulpa:::.joint_local_ccd_refine(
    joint_grid = gd$grid, log_marginal = gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = tags,
    eval_nodes = .lccd_eval(mu, s), max_cells = 4L)
  w1 <- tulpa:::.joint_integration_weights(out$log_marginal, out$dnode)
  sd_ref <- .lccd_wsd(out$joint_grid[, "a"], w1)

  expect_gt(sd_ref, sd_coarse)                                   # collapsed SD recovers
  expect_lt(abs(sd_ref - sd_dense), abs(sd_coarse - sd_dense))   # toward the dense answer
  expect_equal(.lccd_wmean(out$joint_grid[, "a"], w1), 0.5,
               tolerance = 1e-6)                                 # symmetric design keeps the mean
})

test_that("greedy selection picks mutually non-adjacent cells", {
  # Five cells in a chain on one axis (1-2-3-4-5).
  up <- matrix(c(2L, 3L, 4L, 5L, NA), ncol = 1L)
  dn <- matrix(c(NA, 1L, 2L, 3L, 4L), ncol = 1L)
  w  <- c(0.10, 0.90, 0.20, 0.80, 0.15)   # peaks at 2 and 4, both adjacent to 3
  sel <- tulpa:::.joint_local_ccd_select(cands = 1:5, w = w, up = up, dn = dn,
                                         max_cells = 5L)
  expect_true(all(c(2L, 4L) %in% sel))
  expect_false(3L %in% sel)               # excluded: adjacent to both chosen
  for (i in sel) for (k in sel) if (i != k) {
    expect_false(k %in% c(up[i, ], dn[i, ]))
  }
})

test_that("local CCD declines on NULL tags, the engage gate, and a monotone field", {
  expect_false(tulpa:::.joint_local_ccd_engage(3L))
  expect_true(tulpa:::.joint_local_ccd_engage(4L))

  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)

  expect_null(tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, NULL, NULL, c("a", "b"), tags = NULL,
    eval_nodes = .lccd_eval(mu, s)))

  lm_mono <- gd$grid[, "a"] + gd$grid[, "b"]       # no interior peak
  expect_null(tulpa:::.joint_local_ccd_refine(
    gd$grid, lm_mono, NULL, NULL, c("a", "b"),
    tags = c(a = "identity", b = "identity"),
    eval_nodes = .lccd_eval(mu, s)))
})

.lccd_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s), function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn  <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# A 4-axis multi-block joint fit (two coupled ICAR fields -> sigma1, alpha1,
# sigma2, alpha2): the d >= 4 regime where local CCD engages. Coarse,
# truth-centred grids on a sharp (large-N) posterior put the peak on the interior
# middle cell, so the design fits its Voronoi box.
.lccd_sim_joint <- function(seed = 1L, N = 3000L, n_s = 30L) {
  set.seed(seed)
  adj <- .lccd_chain_adj(n_s)
  f1 <- cumsum(rnorm(n_s)); f1 <- (f1 - mean(f1)) / sd(f1)
  f2 <- cumsum(rnorm(n_s)); f2 <- (f2 - mean(f2)) / sd(f2)
  s1 <- sample.int(n_s, N, TRUE); s2 <- sample.int(n_s, N, TRUE)
  X1 <- cbind(1, rnorm(N)); X2 <- cbind(1, rnorm(N))
  eta1 <- X1 %*% c(0.2, 0.5) + f1[s1] + f2[s1]
  eta2 <- X2 %*% c(-0.1, 0.3) + 1.3 * f1[s2] + 0.7 * f2[s2]
  arm <- function(y, X) list(y = as.numeric(y), n_trials = rep(1L, N), X = X,
                             re_idx = rep(0, N), n_re_groups = 0L, sigma_re = 1,
                             family = "gaussian", phi = 1)
  blk <- function(sg) list(type = "icar", n_spatial_units = adj$n_spatial_units,
                           adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                           n_neighbors = adj$n_neighbors, sigma_grid = sg,
                           spatial_idx = list(s1, s2))
  list(responses = list(occ = arm(rnorm(N, eta1, 0.3), X1),
                        pos = arm(rnorm(N, eta2, 0.3), X2)),
       prior = list(blk(c(0.4, 1.0, 2.2)), blk(c(0.4, 1.0, 2.2))),
       copy  = list(list(arm = "pos", block = 1L, alpha_grid = c(0.3, 1.3, 2.6)),
                    list(arm = "pos", block = 2L, alpha_grid = c(0.1, 0.7, 1.5))),
       alpha_true = c(1.3, 0.7))
}

test_that("local CCD engages on a 4-axis multi-block fit and conserves the integral", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE, integration = "grid")

  fit  <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy, control = ctrl))
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L)))))

  # The unrefined tensor fit reports no refinement; the local-CCD fit engages.
  expect_null(fit$local_ccd_info)
  expect_false(is.null(fitl$local_ccd_info))
  expect_gte(fitl$local_ccd_info$n_cells_refined, 1L)
  expect_gt(fitl$local_ccd_info$n_nodes_added, 0L)
  expect_gt(length(fitl$log_marginal), length(fit$log_marginal))   # nodes appended

  # Integration weight is conserved and the posterior stays well-formed.
  expect_true(abs(sum(fitl$weights) - 1) < 1e-8)
  expect_true(all(is.finite(fitl$theta_mean)))

  # Refinement does not degrade recovery: both copy scales stay near truth and
  # near the unrefined fit (refinement re-estimates mass, it does not move it).
  a_base <- vapply(fit$block_moments,  function(b) b$mean[["alpha"]], numeric(1))
  a_ref  <- vapply(fitl$block_moments, function(b) b$mean[["alpha"]], numeric(1))
  expect_lt(max(abs(a_ref - sim$alpha_true)), 0.5)
  expect_lt(max(abs(a_ref - a_base)), 0.15)
})

test_that("k_refine validates and accepts the ccd rung", {
  expect_error(
    tulpa_nested_laplace_joint(list(), list(), control = list(k_refine = "bogus")),
    "k_refine")
  expect_error(
    tulpa_nested_laplace_joint(list(), list(),
                               control = list(local_ccd = "yes")),
    "local_ccd")
})

test_that("k_refine='ccd' escalation drives local CCD and reports an honest verdict", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = list(max_iter = 60L, tol = 1e-6, integration = "grid",
                   var_of_means_consistency = FALSE,
                   k_quality = "ok", k_refine = "ccd",
                   k_samples = 200L, k_max_rounds = 1L)))

  expect_identical(fit$k_quality_requested, "ok")
  expect_true(is.logical(fit$k_quality_reached))         # never NA for an "ok" target
  expect_true(fit$k_quality_rounds >= 0L && fit$k_quality_rounds <= 1L)
  # A round only runs when local CCD actually refined; the verdict never silently
  # downgrades, and the integral stays well-formed throughout.
  if (fit$k_quality_rounds > 0L) expect_false(is.null(fit$local_ccd_info))
  expect_true(abs(sum(fit$weights) - 1) < 1e-8)
  expect_true(all(is.finite(fit$theta_mean)))
})

test_that("local CCD carries inner modes through the splice", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)
  n  <- nrow(gd$grid); n_x <- 3L
  modes <- matrix(seq_len(n * n_x), n, n_x)
  eval_with_modes <- function(theta_mat, warm) {
    list(log_marginal = .lccd_true_lm(theta_mat, mu, s),
         modes = matrix(0, nrow(theta_mat), n_x))
  }
  out <- tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, modes = modes, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = eval_with_modes, max_cells = 4L)
  expect_false(is.null(out))
  expect_equal(nrow(out$modes), nrow(out$joint_grid))
  expect_equal(ncol(out$modes), n_x)
  expect_false(anyNA(out$modes))
})

test_that("local CCD keeps the fixed-effect retention aligned with the refined grid", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE, integration = "grid")
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L)))))

  expect_false(is.null(fitl$local_ccd_info))
  # The node solves carry their own block, so the refined grid reports instead
  # of declining "local_ccd_refined" (gcol33/tulpa#307).
  expect_true(is.na(fitl$grid_fixed_declined))
  expect_length(fitl$grid_hessians, length(fitl$weights))
  expect_length(fitl$grid_modes,    length(fitl$weights))
  for (k in seq_along(fitl$weights)) {
    expect_identical(dim(fitl$grid_hessians[[k]]),
                     c(fitl$n_fixed, fitl$n_fixed))
  }
  expect_true(all(is.finite(vcov(fitl))))
  expect_true(all(is.finite(confint(fitl))))
})

test_that("local CCD splices per-cell covariance blocks alongside the modes", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)
  n  <- nrow(gd$grid)
  blocks <- lapply(seq_len(n), function(k) diag(k, 2L))
  eval_with_blocks <- function(theta_mat, warm) {
    list(log_marginal = .lccd_true_lm(theta_mat, mu, s), modes = NULL,
         cov_blocks = replicate(nrow(theta_mat), diag(-1, 2L), simplify = FALSE))
  }
  out <- tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = eval_with_blocks, max_cells = 4L, cov_blocks = blocks)
  expect_false(is.null(out))
  expect_length(out$cov_blocks, nrow(out$joint_grid))
  expect_true(all(vapply(out$cov_blocks, is.matrix, logical(1))))
})

# --------------------------------------------------------------------------- #
# Design scale on a correlated outer posterior (gcol33/tulpa#316)              #
# --------------------------------------------------------------------------- #
#
# The reference is analytic: an equicorrelated Gaussian outer target on identity
# axes, whose axis marginals are standard normal whatever the correlation is, so
# the exact 95% endpoints are qnorm(0.025) / qnorm(0.975) and the reference
# carries no numerical noise of its own. The target is quadratic in u, so the
# central-difference stencil is exact there too.

.lccd_corr <- function(d = 4L, rho = 0.8, m = 3L, reach = 4, offset = 0) {
  R <- matrix(rho, d, d); diag(R) <- 1
  Q <- solve(R)
  lv <- seq(-reach, reach, length.out = m) + offset * (2 * reach / (m - 1))
  ll <- stats::setNames(replicate(d, lv, simplify = FALSE), paste0("a", seq_len(d)))
  g <- as.matrix(expand.grid(ll, KEEP.OUT.ATTRS = FALSE))
  colnames(g) <- names(ll)
  logf <- function(TH) -0.5 * rowSums((TH %*% Q) * TH)
  list(R = R, Q = Q, grid = g, lm = logf(g),
       tags = stats::setNames(rep("identity", d), colnames(g)),
       eval = function(theta_mat, warm) list(log_marginal = logf(theta_mat),
                                             modes = NULL))
}

# The mean per-axis 95% interval the summary would report off a refined grid.
.lccd_mean_width <- function(rf) {
  w <- tulpa:::.joint_integration_weights(rf$log_marginal, rf$dnode)
  mean(vapply(seq_len(ncol(rf$joint_grid)), function(j) {
    v <- as.numeric(rf$joint_grid[, j])
    use <- is.finite(v) & is.finite(w) & w > 0
    q <- tulpa:::.nl_wtd_quantile(v[use], w[use] / sum(w[use]),
                                  c(0.025, 0.975), outside = "clamp")
    q[2L] - q[1L]
  }, numeric(1)))
}

test_that("the cell's design scale is the marginal spread, not the conditional one", {
  tg <- .lccd_corr(d = 4L, rho = 0.8)
  nb <- tulpa:::.joint_local_ccd_neighbors(tg$grid, tg$grid, seq_len(4L))
  ctr <- which(rowSums(abs(tg$grid)) == 0)
  cc <- tulpa:::.joint_local_ccd_cell_curv(ctr, tg$grid, tg$lm, nb$up, nb$dn)

  # The target is exactly quadratic, so both stencils are exact.
  expect_equal(cc$sd, 1 / sqrt(diag(tg$Q)), tolerance = 1e-12)
  expect_false(is.null(cc$sd_marginal))
  expect_equal(cc$sd_marginal, sqrt(diag(tg$R)), tolerance = 1e-10)
  # At rho = 0.8 in four dimensions the marginal spread is about twice the
  # conditional one, which is the whole of gcol33/tulpa#316.
  expect_equal(cc$sd_marginal / cc$sd, sqrt(diag(tg$Q) * diag(tg$R)),
               tolerance = 1e-10)
  expect_gt(min(cc$sd_marginal / cc$sd), 1.9)
})

test_that("a dominant refined cell reports the marginal spread it stands in for", {
  exact <- diff(stats::qnorm(c(0.025, 0.975)))
  for (rho in c(0.7, 0.8, 0.9)) {
    tg <- .lccd_corr(d = 4L, rho = rho)
    rf <- tulpa:::.joint_local_ccd_refine(
      tg$grid, tg$lm, NULL, NULL, colnames(tg$grid), tg$tags, tg$eval,
      max_cells = 4L)
    expect_false(is.null(rf))

    # The regime, asserted before the width is read: this cell carries
    # essentially the whole posterior, so the neighbours the cloud is clamped
    # against hold none of the truncated tail.
    w <- tulpa:::.joint_integration_weights(rf$log_marginal, rf$dnode)
    expect_gt(sum(w[rf$weight_kind == "design"]), 0.99)

    # Measured from the conditional scale this replaces: 0.675 / 0.554 / 0.394
    # of the exact width at these three correlations.
    expect_equal(.lccd_mean_width(rf) / exact, 1, tolerance = 0.08)
  }
})

test_that("the design scale is unchanged where the outer axes are uncorrelated", {
  tg <- .lccd_corr(d = 4L, rho = 0)
  nb <- tulpa:::.joint_local_ccd_neighbors(tg$grid, tg$grid, seq_len(4L))
  ctr <- which(rowSums(abs(tg$grid)) == 0)
  cc <- tulpa:::.joint_local_ccd_cell_curv(ctr, tg$grid, tg$lm, nb$up, nb$dn)
  expect_equal(cc$sd_marginal, cc$sd, tolerance = 1e-10)

  rf <- tulpa:::.joint_local_ccd_refine(
    tg$grid, tg$lm, NULL, NULL, colnames(tg$grid), tg$tags, tg$eval,
    max_cells = 4L)
  expect_equal(.lccd_mean_width(rf) / diff(stats::qnorm(c(0.025, 0.975))),
               1, tolerance = 0.02)
})

test_that("a grid with no corner stencil keeps the conditional scale", {
  # A grid a previous pass already spliced nodes into has no complete corner
  # stencil; the cell then reports its conditional scale and no marginal one,
  # which is the fallback the refinement runs on.
  tg <- .lccd_corr(d = 4L, rho = 0.8)
  keep <- which(!(tg$grid[, 1L] > 0 & tg$grid[, 2L] > 0))
  g2 <- tg$grid[keep, , drop = FALSE]
  nb <- tulpa:::.joint_local_ccd_neighbors(g2, g2, seq_len(4L))
  ctr <- which(rowSums(abs(g2)) == 0)
  cc <- tulpa:::.joint_local_ccd_cell_curv(ctr, g2, tg$lm[keep], nb$up, nb$dn)
  expect_false(is.null(cc))
  expect_null(cc$sd_marginal)
  expect_equal(cc$sd, 1 / sqrt(diag(tg$Q)), tolerance = 1e-12)
})

# --------------------------------------------------------------------------- #
# What share of the base grid each refined cell already held (gcol33/tulpa#316) #
# --------------------------------------------------------------------------- #

test_that("each refined cell reports the base-grid share it held", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)
  out <- tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = .lccd_eval(mu, s), max_cells = 4L)
  expect_false(is.null(out))
  # One share per refined cell, and each is that cell's own weight on the grid
  # the refinement started from.
  w0 <- tulpa:::.joint_integration_weights(gd$lm, NULL)
  expect_length(out$info$cell_share, out$info$n_cells_refined)
  expect_equal(out$info$cell_share, w0[out$info$cells], tolerance = 1e-12)
})

test_that("a fit reports the base share separately from the design mass", {
  skip_on_cran()
  # The two are different numbers: the replacement nodes sit nearer the peak
  # than the cell's own coordinate did, so refining a cell RAISES the share of
  # the posterior the refined region holds. `design_mass` is the share after,
  # `cell_share` the share before, and the regime gcol33/tulpa#316 describes is
  # the one where either approaches 1.
  sim  <- .lccd_sim_joint()
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
                   var_of_means_consistency = FALSE, integration = "grid",
                   local_ccd = list(max_cells = 4L))))
  info <- fitl$local_ccd_info
  expect_false(is.null(info))
  expect_length(info$cell_share, info$n_cells_refined)
  expect_true(all(info$cell_share > 0 & info$cell_share < 1))
  expect_lte(sum(info$cell_share), 1 + 1e-8)
  # The refinement concentrates: the design-weighted share after exceeds the
  # cells' own share before.
  expect_gt(info$design_mass, sum(info$cell_share))
})

# --------------------------------------------------------------------------- #
# Per-cell weight kind (gcol33/tulpa#311)                                      #
# --------------------------------------------------------------------------- #

test_that("the refined grid labels which cells carry a design weight", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)
  out <- tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = .lccd_eval(mu, s), max_cells = 4L)
  expect_false(is.null(out))
  expect_length(out$weight_kind, nrow(out$joint_grid))
  expect_setequal(unique(out$weight_kind), c("mass", "design"))
  # One label per row, and the design rows are exactly each refined cell's whole
  # replacement cloud: the nodes added plus the centre that reuses the cell's
  # own solve, which carries the design's centre weight rather than the cell's
  # mass.
  n_design <- out$info$n_nodes_added + out$info$n_cells_refined
  expect_identical(sum(out$weight_kind == "design"), n_design)
  expect_identical(out$info$n_design_nodes, n_design)
  expect_identical(sum(out$weight_kind == "mass"),
                   out$info$n_cells_before - out$info$n_cells_refined)
  # The mass-labelled rows are the carried-over base cells, unchanged.
  keep <- setdiff(seq_len(nrow(gd$grid)), out$info$cells)
  expect_equal(out$joint_grid[out$weight_kind == "mass", , drop = FALSE],
               gd$grid[keep, , drop = FALSE])
})

test_that("a fit reports its weight kind per cell and the design mass share", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE, integration = "grid")
  fit  <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy, control = ctrl))
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L)))))

  # An unrefined tensor grid is mass-weighted throughout; the refined one is the
  # case that carries both, which is what `integration` alone cannot say.
  expect_identical(unique(fit$weight_kind), "mass")
  expect_length(fit$weight_kind, length(fit$weights))
  expect_length(fitl$weight_kind, length(fitl$weights))
  expect_setequal(unique(fitl$weight_kind), c("mass", "design"))
  expect_identical(sum(fitl$weight_kind == "design"),
                   fitl$local_ccd_info$n_design_nodes)

  # `design_mass` is the share of the posterior sitting on those cells: the part
  # of the support whose cumulative weight is not a CDF.
  expect_equal(fitl$local_ccd_info$design_mass,
               sum(fitl$weights[fitl$weight_kind == "design"]))
  expect_gt(fitl$local_ccd_info$design_mass, 0)
  expect_lt(fitl$local_ccd_info$design_mass, 1)

  expect_identical(tulpa:::.nl_node_support(fit$integration), "density")
})

test_that("a globally CCD-integrated fit is design-weighted throughout", {
  skip_on_cran()
  # Where the support IS homogeneous the tag and `.nl_node_support(integration)`
  # agree. This needs a fit the CCD integrator actually engages on: the local-CCD
  # fixture above has a ridged outer log-posterior, on which the mode-find
  # declines back to the tensor grid. Crossed iid blocks give it curvature.
  set.seed(4242)
  G <- 30L; N <- 600L
  grp <- lapply(1:3, function(k) sample.int(G, N, replace = TRUE))
  X <- cbind(1, rnorm(N))
  eta <- as.numeric(X %*% c(0.2, 0.6))
  for (k in 1:3) eta <- eta + rnorm(G, 0, c(0.8, 0.5, 0.3)[k])[grp[[k]]]
  y <- eta + rnorm(N, 0, 0.5)
  sg <- exp(seq(log(0.1), log(2), length.out = 7))
  fitc <- suppressWarnings(tulpa_nested_laplace_joint(
    responses = list(a = list(y = y, n_trials = rep(1L, N), X = X,
                              family = "gaussian", phi = 0.25)),
    prior = lapply(grp, function(g)
      list(type = "iid", obs_idx = list(g), n_units = G, sigma_grid = sg)),
    control = list(integration = "ccd", n_threads = 1L, diagnose_k = FALSE,
                   max_iter = 100L, tol = 1e-8)))
  # Asserted first: on a declined CCD the tag check below would pass vacuously
  # for the wrong reason.
  expect_identical(fitc$integration, "ccd")
  expect_identical(unique(fitc$weight_kind), "design")
  expect_length(fitc$weight_kind, length(fitc$weights))
  expect_identical(tulpa:::.nl_node_support(fitc$integration), "moment_rule")
  # Nothing was declined here, which is what makes the tag check above the
  # engaged case rather than the fallback (gcol33/tulpa#315).
  expect_true(is.na(fitc$integration_declined))
})

test_that("this fixture's CCD decline is on the fit, not only in a message", {
  skip_on_cran()
  # The fixture's own outer log-posterior is ridged, so a requested CCD falls
  # back here. Asserted in that order: the fallback has to have happened before
  # the reason means anything.
  sim <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE)
  fit <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(integration = "ccd"))))
  expect_identical(fit$integration_requested, "ccd")
  expect_false(identical(fit$integration, "ccd"))
  expect_identical(fit$integration_declined, "modefind_ridge")

  # The same reason the verbose channel reports, so the field and the message
  # cannot drift apart.
  expect_message(
    suppressWarnings(tulpa_nested_laplace_joint(
      sim$responses, sim$prior, copy = sim$copy,
      control = c(ctrl, list(integration = "ccd", verbose = TRUE)))),
    "flat / ridged")

  # The same fit under integration = "grid" asked for nothing and declined
  # nothing, which is the state the reason field has to be distinguishable from.
  fit_g <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(integration = "grid"))))
  expect_identical(fit_g$integration, fit$integration)
  expect_true(is.na(fit_g$integration_declined))
})

# --------------------------------------------------------------------------- #
# What the per-axis interval was read off (gcol33/tulpa#317)                   #
# --------------------------------------------------------------------------- #

test_that("a heterogeneous weight kind names the support mixed", {
  # `integration` describes a homogeneous support, and a locally refined grid
  # reports "grid", so the integrator name alone reads it as a density one.
  expect_identical(tulpa:::.nl_node_support("grid"), "density")
  expect_identical(tulpa:::.nl_node_support("ccd"), "moment_rule")
  expect_identical(tulpa:::.nl_node_support("grid", rep("mass", 4L)), "density")
  expect_identical(tulpa:::.nl_node_support("ccd", rep("design", 4L)),
                   "moment_rule")
  expect_identical(tulpa:::.nl_node_support("grid", c("mass", "design", "mass")),
                   "mixed")
})

test_that("the interval provenance carries the design share", {
  w <- c(0.4, 0.1, 0.3, 0.2)
  k <- c("mass", "design", "design", "mass")
  p <- tulpa:::.nl_interval_provenance("grid", k, w)
  expect_identical(p$read, "mixed")
  expect_equal(p$design_mass, 0.4)
  # A homogeneous support has no share to report: none of it is a moment rule
  # read as a CDF on a density grid, all of it is on a CCD.
  expect_equal(
    tulpa:::.nl_interval_provenance("grid", rep("mass", 4L), w)$design_mass, 0)
  expect_equal(
    tulpa:::.nl_interval_provenance("ccd", rep("design", 4L), w)$design_mass, 1)
})

test_that("naming the support mixed leaves the reported interval unchanged", {
  # The measurement behind that choice is in `.nl_summary_quantile`: against a
  # converged tensor reference the weighted quantile beat every alternative read
  # in both design_mass regimes, so the support name is what changes here, not
  # the number.
  set.seed(11L)
  v <- sort(stats::rnorm(40L))
  w <- stats::runif(40L); w <- w / sum(w)
  probs <- c(0.025, 0.5, 0.975)
  expect_identical(
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "mixed"),
    tulpa:::.nl_wtd_quantile(v, w, probs, outside = "clamp"))
  # And it is NOT the moment read, which is what a design-weighted support takes.
  expect_false(isTRUE(all.equal(
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "mixed"),
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "moment_rule"))))
})

test_that("a locally refined fit says what its interval was read off", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE, integration = "grid")
  fit  <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy, control = ctrl))
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L)))))

  expect_identical(fit$theta_interval_read, "density")
  expect_equal(fit$theta_interval_design_mass, 0)
  expect_identical(fitl$theta_interval_read, "mixed")
  expect_equal(fitl$theta_interval_design_mass,
               fitl$local_ccd_info$design_mass)
  expect_gt(fitl$theta_interval_design_mass, 0)
  expect_lt(fitl$theta_interval_design_mass, 1)

  # The intervals themselves are what the density read gives, to the last bit:
  # the mixed tag records the support, it does not switch the construction.
  qs <- tulpa:::.nl_axis_quantiles(fitl$theta_grid, fitl$log_marginal,
                                   fitl$refining_axis, weights = fitl$weights,
                                   support = "density")
  expect_equal(fitl$theta_ci_lo, qs$ci_lo)
  expect_equal(fitl$theta_ci_hi, qs$ci_hi)
  expect_equal(fitl$theta_median, qs$median)

  # And the diagnostic layer surfaces the pair rather than leaving it on the fit.
  ir <- tulpa:::.tulpa_interval_read(fitl)
  expect_identical(ir$read, "mixed")
  expect_equal(ir$design_mass, fitl$local_ccd_info$design_mass)
  expect_match(tulpa:::.tulpa_interval_read_note(ir), "MIXED support")
  expect_null(tulpa:::.tulpa_interval_read_note(tulpa:::.tulpa_interval_read(fit)))
})

# --------------------------------------------------------------------------- #
# The measured case for the mixed read (gcol33/tulpa#317)                      #
# --------------------------------------------------------------------------- #
#
# Two analytic outer targets whose per-axis quantiles are known in closed form:
# an equicorrelated Gaussian (qnorm) and a Gaussian copula with Gamma marginals
# (qgamma), which is correlated the same way and skewed. The second one is what
# separates a read that is right from a read that has the right FAMILY: the #308
# moment read moment-matches a Gaussian on an `unbounded` domain, so on the
# Gaussian target it is the exact family and wins by construction.
#
# Alternatives scored beside the shipped mixed read: collapsing each refined
# cell's design block to one atom at its own weighted mean (which removes the
# design's extent), the moment read, and not refining at all.

.lccd_gate_target <- function(rho, reach, m, d = 4L, gamma_shape = NULL) {
  R <- matrix(rho, d, d); diag(R) <- 1
  if (is.null(gamma_shape)) {
    Q  <- solve(R)
    lf <- function(TH) -0.5 * rowSums((TH %*% Q) * TH)
    rng <- c(-reach, reach)
    qx  <- stats::qnorm(c(0.025, 0.5, 0.975))
  } else {
    Qi <- solve(R) - diag(d)
    lf <- function(TH) {
      TH <- pmax(TH, 1e-12)
      z  <- stats::qnorm(stats::pgamma(TH, shape = gamma_shape))
      z[!is.finite(z)] <- sign(z[!is.finite(z)]) * 8
      rowSums(stats::dgamma(TH, shape = gamma_shape, log = TRUE)) -
        0.5 * rowSums((z %*% Qi) * z)
    }
    # `reach` is the tail in standard-normal units mapped through the marginal's
    # own quantile function, so a coarse grid straddles the mode on both targets.
    rng <- stats::qgamma(stats::pnorm(c(-reach, reach)), shape = gamma_shape)
    qx  <- stats::qgamma(c(0.025, 0.5, 0.975), shape = gamma_shape)
  }
  lv <- seq(rng[1L], rng[2L], length.out = m)
  ll <- stats::setNames(replicate(d, lv, simplify = FALSE), paste0("a", seq_len(d)))
  g  <- as.matrix(expand.grid(ll, KEEP.OUT.ATTRS = FALSE))
  colnames(g) <- names(ll)
  list(grid = g, lm = lf(g), exact = qx,
       tags = stats::setNames(rep("identity", d), colnames(g)),
       eval = function(theta_mat, warm)
         list(log_marginal = lf(theta_mat), modes = NULL))
}

# Mean absolute endpoint error of each read against the exact quantiles, on one
# refined grid. NULL when the refinement declines there.
.lccd_gate_reads <- function(tg, max_cells = 4L) {
  rf <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                        colnames(tg$grid), tg$tags, tg$eval,
                                        max_cells)
  if (is.null(rf)) return(NULL)
  w     <- tulpa:::.joint_integration_weights(rf$log_marginal, rf$dnode)
  probs <- c(0.025, 0.5, 0.975)
  # Which refined cell each design row belongs to: the refinement appends one
  # contiguous block per cell, every block the same CCD size.
  nc   <- rf$info$n_cells_refined
  blk  <- sum(rf$weight_kind == "design") %/% nc
  cell <- integer(length(w))
  cell[rf$weight_kind == "design"] <- rep(seq_len(nc), each = blk)

  per_axis <- function(f) t(vapply(seq_len(ncol(rf$joint_grid)), function(j) {
    v   <- as.numeric(rf$joint_grid[, j])
    use <- is.finite(v) & is.finite(w) & w > 0
    f(v[use], w[use] / sum(w[use]), cell[use])
  }, numeric(3L)))

  mixed  <- per_axis(function(v, ww, cc)
    tulpa:::.nl_summary_quantile(v, ww, probs, "unbounded", "mixed"))
  moment <- per_axis(function(v, ww, cc)
    tulpa:::.nl_moment_quantile(v, ww, probs, "unbounded"))
  collapsed <- per_axis(function(v, ww, cc) {
    keep <- cc == 0L
    vv <- v[keep]; wv <- ww[keep]
    for (k in setdiff(unique(cc), 0L)) {
      ix  <- which(cc == k)
      tot <- sum(ww[ix])
      vv  <- c(vv, sum(ww[ix] * v[ix]) / tot)
      wv  <- c(wv, tot)
    }
    tulpa:::.nl_wtd_quantile(vv, wv, probs, outside = "clamp")
  })
  w0 <- tulpa:::.joint_integration_weights(tg$lm, NULL)
  base <- t(vapply(seq_len(ncol(tg$grid)), function(j) {
    v   <- as.numeric(tg$grid[, j])
    use <- is.finite(v) & is.finite(w0) & w0 > 0
    tulpa:::.nl_wtd_quantile(v[use], w0[use] / sum(w0[use]), probs, "clamp")
  }, numeric(3L)))

  ex <- matrix(rep(tg$exact, each = nrow(mixed)), nrow = nrow(mixed))
  e  <- function(M) mean(abs(M - ex))
  list(design_mass = sum(w[rf$weight_kind == "design"]),
       mixed = e(mixed), moment = e(moment), collapsed = e(collapsed),
       base = e(base))
}

.lccd_gate_sweep <- function(gamma_shape = NULL,
                             reaches = c(2, 3, 4, 6), ms = c(3L, 5L, 7L)) {
  acc <- list()
  for (rho in c(0, 0.5, 0.8)) for (reach in reaches) for (m in ms) {
    r <- .lccd_gate_reads(.lccd_gate_target(rho, reach, m,
                                            gamma_shape = gamma_shape))
    if (is.null(r)) next
    acc[[length(acc) + 1L]] <- as.data.frame(r)
  }
  do.call(rbind, acc)
}

test_that("the mixed read beats the alternatives on both sides of design_mass", {
  skip_if_not_slow()
  s <- .lccd_gate_sweep()
  hi <- s[s$design_mass >= 0.5, ]
  lo <- s[s$design_mass <  0.5, ]
  # The regime is asserted before the errors, so neither half can pass by being
  # empty.
  expect_gte(nrow(hi), 15L)
  expect_gte(nrow(lo), 15L)

  # Summed absolute endpoint error against the exact quantiles. Measured:
  # design_mass >= 0.5 (19 configurations) mixed 3.18024, collapsed 15.72792,
  # unrefined 17.80607; design_mass < 0.5 (17) mixed 2.69889, collapsed 3.92249,
  # unrefined 3.83155.
  for (part in list(hi, lo)) {
    tot <- colSums(part[, c("mixed", "moment", "collapsed", "base")])
    expect_lt(tot[["mixed"]], tot[["collapsed"]])
    expect_lt(tot[["mixed"]], tot[["base"]])
  }
  expect_lt(sum(hi$mixed), 3.5)
  expect_lt(sum(lo$mixed), 3.0)

  # The moment read wins on THIS target because a Gaussian moment match is its
  # exact family here, which is why the win does not carry (see below).
  expect_lt(sum(s$moment), sum(s$mixed))
})

test_that("the moment read's advantage is the family, not the support", {
  skip_if_not_slow()
  # Same correlation, same refinement, skewed marginals. Measured over 27
  # configurations: mixed 26.24674 against moment 39.93958.
  s <- .lccd_gate_sweep(gamma_shape = 2, reaches = c(1.5, 2, 3))
  expect_gte(nrow(s), 20L)
  expect_lt(sum(s$mixed), sum(s$moment))
  expect_lt(sum(s$mixed), 29)
})
