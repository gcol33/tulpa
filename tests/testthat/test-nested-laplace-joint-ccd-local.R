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
  # This fixture's own candidate cell scores 0.7458 on the local-quadratic
  # misfit, so the engagement gate declines it at the default `skew_max`
  # (gcol33/tulpa#318, and the test below that pins that). The splice, the weight
  # conservation and the recovery this block checks are the machinery underneath
  # the gate, so it is opened here rather than fitting a different model.
  fitl <- suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L, skew_max = Inf)))))

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
                   local_ccd = list(max_cells = 4L, skew_max = Inf))))
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
    control = c(ctrl, list(local_ccd = list(max_cells = 4L, skew_max = Inf)))))

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
                              family = "gaussian", phi = 0.0625)),
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
  # Compared against the density dispatch itself, not against a hard-coded
  # `outside` policy: the invariant is that the two supports take ONE `outside`
  # construction, so it has to keep holding when that construction changes
  # (gcol33/tulpa#353 moved it from "clamp" to "extend").
  #
  # `within` is the orthogonal field, and there the two supports part company:
  # a refined grid's replacement clouds sit inside one base cell, so a Voronoi
  # partition of its node set is not the design's own boxes and `mixed` declines
  # the box-uniform default (gcol33/tulpa#357). Comparing at the chord read is
  # what isolates the `outside` invariant from that decline.
  expect_identical(
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "mixed"),
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "density", "chord"))
  # The decline is recorded rather than silent, and it is what makes the two
  # reads differ at the shipped default.
  mixed_read <- tulpa:::.nl_summary_quantile_read(v, w, probs, "unbounded", "mixed")
  expect_identical(mixed_read$within, "chord")
  expect_identical(mixed_read$declined, "support_mixed")
  expect_false(isTRUE(all.equal(
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "mixed"),
    tulpa:::.nl_summary_quantile(v, w, probs, "unbounded", "density"))))
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
    control = c(ctrl, list(local_ccd = list(max_cells = 4L, skew_max = Inf)))))

  expect_identical(fit$theta_interval_read, "density")
  expect_equal(fit$theta_interval_design_mass, 0)
  expect_identical(fitl$theta_interval_read, "mixed")
  expect_equal(fitl$theta_interval_design_mass,
               fitl$local_ccd_info$design_mass)
  expect_gt(fitl$theta_interval_design_mass, 0)
  expect_lt(fitl$theta_interval_design_mass, 1)

  # The intervals themselves are what the density read gives at the SAME
  # within-cell construction, to the last bit: the mixed tag records the
  # support, it does not switch the `outside` policy. It does switch the
  # within-cell one -- a refined grid's replacement clouds sit inside a base
  # cell, so mixed declines the box-uniform default and reads chord
  # (gcol33/tulpa#357) -- so the reference is built at chord to isolate the
  # invariant this test is about.
  qs <- tulpa:::.nl_axis_quantiles(fitl$theta_grid, fitl$log_marginal,
                                   fitl$refining_axis, weights = fitl$weights,
                                   support = "density", within = "chord")
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
  # A grid whose every candidate cell was put back as a mass atom
  # (gcol33/tulpa#318) carries no design block, so there is nothing to group.
  nc   <- rf$info$n_cells_refined
  cell <- integer(length(w))
  if (nc > 0L) {
    blk <- sum(rf$weight_kind == "design") %/% nc
    cell[rf$weight_kind == "design"] <- rep(seq_len(nc), each = blk)
  }

  per_axis <- function(f) t(vapply(seq_len(ncol(rf$joint_grid)), function(j) {
    v   <- as.numeric(rf$joint_grid[, j])
    use <- is.finite(v) & is.finite(w) & w > 0
    f(v[use], w[use] / sum(w[use]), cell[use])
  }, numeric(3L)))

  # All three arms are read off the same dispatch at the same within-cell
  # construction, so what varies between them is the collapse construction and
  # nothing else. `mixed` declines the box-uniform default and falls back to the
  # chord read, so the comparators ask for chord too; naming a policy inline
  # here is what let them drift apart the last time the construction moved.
  dens   <- function(v, ww)
    tulpa:::.nl_summary_quantile(v, ww, probs, "unbounded", "density",
                                 within = "chord")
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
    dens(vv, wv)
  })
  w0 <- tulpa:::.joint_integration_weights(tg$lm, NULL)
  base <- t(vapply(seq_len(ncol(tg$grid)), function(j) {
    v   <- as.numeric(tg$grid[, j])
    use <- is.finite(v) & is.finite(w0) & w0 > 0
    dens(v[use], w0[use] / sum(w0[use]))
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
  # design_mass >= 0.5 (19 configurations) mixed 3.46157, collapsed 16.29057,
  # unrefined 18.05801; design_mass < 0.5 (17) mixed 3.14443, collapsed 4.79910,
  # unrefined 4.48729.
  #
  # The mixed totals rose from 3.18024 / 2.69889 when the mixed support moved to
  # `outside = "extend"` alongside the density one, and the comparator totals
  # from 15.72792 / 17.80607 / 3.92249 / 3.83155 when they were aligned onto the
  # same dispatch. The ordering below holds under either policy applied to all
  # three arms; it is the margin, not the verdict, that moves.
  for (part in list(hi, lo)) {
    tot <- colSums(part[, c("mixed", "moment", "collapsed", "base")])
    expect_lt(tot[["mixed"]], tot[["collapsed"]])
    expect_lt(tot[["mixed"]], tot[["base"]])
  }
  expect_lt(sum(hi$mixed), 3.8)
  expect_lt(sum(lo$mixed), 3.5)

  # The moment read wins on THIS target because a Gaussian moment match is its
  # exact family here, which is why the win does not carry (see below).
  expect_lt(sum(s$moment), sum(s$mixed))
})

test_that("the moment read's advantage is the family, not the support", {
  skip_if_not_slow()
  # Same correlation, same refinement, skewed marginals. Measured over 27
  # configurations: mixed 28.52885 against moment 39.85826. The mixed total is
  # off the GATED refinement (gcol33/tulpa#318); unconditional refinement on this
  # target gives 26.24674.
  s <- .lccd_gate_sweep(gamma_shape = 2, reaches = c(1.5, 2, 3))
  expect_gte(nrow(s), 20L)
  expect_lt(sum(s$mixed), sum(s$moment))
  expect_lt(sum(s$mixed), 29)
})

# --------------------------------------------------------------------------- #
# The engagement gate: refine only where the design's quadratic holds (#318)   #
# --------------------------------------------------------------------------- #

test_that("the local-quadratic misfit is zero on a quadratic and linear in the cubic", {
  d <- 4L
  ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
  u_c <- rep(0, d)
  sd  <- rep(1, d)
  quad <- -0.5 * rowSums(ccd$z^2)
  # A central composite design identifies a full quadratic exactly, so nothing is
  # left over: the score cannot fire where the outer target is Gaussian in the
  # transformed coordinate, whatever the correlation or the grid spacing.
  expect_equal(tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, quad, sd)$misfit, 0,
               tolerance = 1e-12)
  # The least-squares residual is linear in the response, so a cubic term of
  # twice the size scores exactly twice as high.
  cub <- ccd$z[, 1L]^3
  g1 <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, quad + cub / 6, sd)$misfit
  g2 <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, quad + cub / 3, sd)$misfit
  expect_gt(g1, 0)
  expect_equal(g2, 2 * g1, tolerance = 1e-10)
  # And it is invariant to the quadratic part it is measured against.
  expect_equal(
    tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, 3 - 2 * quad + cub / 6, sd)$misfit,
    g1, tolerance = 1e-10)
})

test_that("the whitened gradient reads an off-centre cell the misfit cannot see", {
  d <- 4L
  ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
  u_c <- rep(0, d)
  sd  <- rep(1, d)
  # A quadratic centred ON the cell: the design represents it exactly and the
  # cell's coordinate is its peak, so both readings are zero.
  at <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, -0.5 * rowSums(ccd$z^2), sd)
  expect_equal(at$misfit, 0, tolerance = 1e-12)
  expect_equal(at$offset, 0, tolerance = 1e-10)
  # The same quadratic, peaked at `m` instead. It is still exactly quadratic in
  # the whitened coordinate, so the design still represents it exactly and the
  # misfit stays at zero however far off-centre the peak sits; the whitened
  # gradient is `m` and its norm is what the offset reports.
  for (m in list(c(0.4, 0, 0, 0), c(0.9, -0.6, 0.3, 0), c(-2, 1.5, -1, 0.5))) {
    zc <- sweep(ccd$z, 2L, m, FUN = "-")
    fit <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, -0.5 * rowSums(zc^2), sd)
    expect_equal(fit$misfit, 0, tolerance = 1e-10)
    expect_equal(fit$offset, sqrt(sum(m^2)), tolerance = 1e-8)
  }
  # And it is standardized: the same displacement read on a cell whose marginal
  # spread is `s` per axis reports `|p| / s`, not `|p|`.
  s <- c(0.5, 2, 1, 4)
  p <- c(0.45, -1.2, 0.3, 2)
  u_nodes <- sweep(ccd$z, 2L, s, FUN = "*")
  lm_nodes <- -0.5 * rowSums(sweep(sweep(u_nodes, 2L, p, FUN = "-"),
                                   2L, s, FUN = "/")^2)
  scaled <- tulpa:::.joint_local_ccd_misfit(u_nodes, u_c, lm_nodes, s)
  expect_equal(scaled$misfit, 0, tolerance = 1e-10)
  expect_equal(scaled$offset, sqrt(sum((p / s)^2)), tolerance = 1e-8)
})

test_that("the offset declines with the fit rather than reporting an unestimated one", {
  d <- 4L
  ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
  u_c <- rep(0, d)
  sd  <- rep(1, d)
  quad <- -0.5 * rowSums(ccd$z^2)
  # A non-finite response identifies nothing: both readings decline together.
  bad <- quad; bad[1L] <- NA_real_
  na <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, bad, sd)
  expect_true(is.na(na$misfit))
  expect_true(is.na(na$offset))
  # A design flat on one axis is rank deficient there: the residual is still a
  # residual, but that axis's linear coefficient was never estimated, so the
  # norm over the gradient declines instead of standing in for it.
  z <- ccd$z; z[, d] <- 0
  flat <- tulpa:::.joint_local_ccd_misfit(z, u_c, -0.5 * rowSums(z^2), sd)
  expect_true(is.finite(flat$misfit))
  expect_true(is.na(flat$offset))
})

test_that("a cell whose local quadratic does not hold is put back as a mass atom", {
  # Gaussian copula with Gamma(2) marginals: correlated the same way as the
  # quadratic target, skewed within the cell.
  tg <- .lccd_gate_target(rho = 0.8, reach = 1.5, m = 3L, gamma_shape = 2)
  off <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L, skew_max = Inf)
  on  <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L)
  # The regime is established first: unconditional refinement does place a cloud
  # here, and its score is above the band.
  expect_gt(off$info$n_cells_refined, 0L)
  expect_gt(max(off$info$misfit), tulpa:::.nl_diag("gamma3_ok"))
  # So the gate declines it, and says so.
  expect_gt(on$info$n_cells_declined, 0L)
  expect_equal(on$info$n_cells_refined,
               off$info$n_cells_refined - on$info$n_cells_declined)
  expect_true(all(on$info$misfit_declined >= on$info$skew_max))
  expect_equal(on$info$skew_max, tulpa:::.nl_diag("gamma3_ok"))
  # A declined cell still reports how far off-centre it sat: the offset is
  # carried per cell on both sides of the gate (gcol33/tulpa#321).
  expect_length(on$info$offset_declined, on$info$n_cells_declined)
  expect_true(all(is.finite(on$info$offset_declined)))
  expect_true(all(on$info$offset_declined >= 0))
  expect_length(off$info$offset, off$info$n_cells_refined)
  expect_true(all(is.finite(off$info$offset)))
  # Every cell declined here, so the grid is the base grid again: same rows, no
  # design weights, and nothing downstream reads it as a mixed support.
  expect_equal(on$info$n_design_nodes, 0L)
  expect_equal(on$joint_grid, tg$grid)
  expect_equal(on$log_marginal, tg$lm)
  expect_true(all(on$weight_kind == "mass"))
  expect_identical(tulpa:::.nl_node_support("grid", on$weight_kind), "density")
})

test_that("the gate leaves a quadratic outer target's refinement untouched", {
  tg <- .lccd_gate_target(rho = 0.8, reach = 3, m = 3L)
  off <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L, skew_max = Inf)
  on  <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L)
  expect_gt(off$info$n_cells_refined, 0L)
  expect_equal(on$info$n_cells_declined, 0L)
  expect_equal(max(on$info$misfit), 0, tolerance = 1e-10)
  # Zero misfit on a quadratic outer target says nothing about where in the cell
  # its peak sat, and the offset is the number that does.
  expect_length(on$info$offset, on$info$n_cells_refined)
  expect_length(on$info$offset_declined, 0L)
  expect_true(all(is.finite(on$info$offset)))
  expect_equal(on$joint_grid, off$joint_grid)
  expect_equal(on$log_marginal, off$log_marginal)
  expect_equal(on$dnode, off$dnode)
})

# Gated against unconditional refinement on one target, scored the same way
# `.lccd_gate_reads` scores the reads: mean absolute endpoint error of the
# weighted quantile against the exact per-axis quantiles.
.lccd_skew_gate_sweep <- function(gamma_shape = NULL, reaches, ms = c(3L, 5L, 7L)) {
  probs <- c(0.025, 0.5, 0.975)
  one <- function(tg, skew_max) {
    rf <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                          colnames(tg$grid), tg$tags, tg$eval,
                                          max_cells = 4L, skew_max = skew_max)
    if (is.null(rf)) return(NULL)
    w <- tulpa:::.joint_integration_weights(rf$log_marginal, rf$dnode)
    Q <- t(vapply(seq_len(ncol(rf$joint_grid)), function(j) {
      v <- as.numeric(rf$joint_grid[, j])
      use <- is.finite(v) & is.finite(w) & w > 0
      tulpa:::.nl_wtd_quantile(v[use], w[use] / sum(w[use]), probs, "clamp")
    }, numeric(3L)))
    ex <- matrix(rep(tg$exact, each = nrow(Q)), nrow = nrow(Q))
    list(err = mean(abs(Q - ex)), declined = rf$info$n_cells_declined,
         kept = rf$info$n_cells_refined)
  }
  acc <- list()
  for (rho in c(0, 0.5, 0.8)) for (reach in reaches) for (m in ms) {
    tg <- .lccd_gate_target(rho, reach, m, gamma_shape = gamma_shape)
    a <- one(tg, tulpa:::.nl_diag("gamma3_ok"))
    b <- one(tg, Inf)
    if (is.null(a) || is.null(b)) next
    acc[[length(acc) + 1L]] <- data.frame(gated = a$err, unconditional = b$err,
                                          declined = a$declined, kept = a$kept)
  }
  do.call(rbind, acc)
}

test_that("the gate declines this fixture's own cell, and the read moves toward truth", {
  skip_on_cran()
  sim  <- .lccd_sim_joint()
  ctrl <- list(max_iter = 60L, tol = 1e-6, diagnose_k = FALSE,
               var_of_means_consistency = FALSE, integration = "grid")
  # Pinned to the chord within-cell read, because the reference below WAS one:
  # at a converged m = 13 the chord read reproduces `ref_w` to five decimals on
  # the two axes whose 95% bound lies inside the node set, and the shipped
  # box-uniform default does not (gcol33/tulpa#399). Scoring the REFINEMENT GATE
  # against it therefore has to be done in the read that produced it, or the
  # within-cell construction -- an orthogonal choice (gcol33/tulpa#357) -- enters
  # the number. At the default: wid(on) = 0.4706, wid(off) = 0.7313. The ordering
  # this test asserts holds either way.
  #
  # That gap is NOT the box partition being mis-sized on a three-level axis.
  # Measured on the reference's own axes with only the resolution moving, the
  # error against the converged limit falls 0.2867 / 0.1490 / 0.0894 / 0.0509 /
  # 0.0258 at m = 3 / 5 / 7 / 9 / 11 for box-uniform, against 0.8541 / 0.2000 /
  # 0.1267 / 0.1237 / 0.0451 for chord -- monotone, and closer at every rung
  # (`dev_notes/issue399/RESULTS399.md`). What this fixture's own pinned axes
  # differ in is their EXTENT, not their resolution.
  lc <- function(sm) suppressWarnings(tulpa_nested_laplace_joint(
    sim$responses, sim$prior, copy = sim$copy,
    control = c(ctrl, list(local_ccd = list(max_cells = 4L, skew_max = sm),
                           within_cell = "chord"))))
  off <- lc(Inf)
  on  <- lc(tulpa:::.nl_diag("gamma3_ok"))

  # The regime first: this fixture's candidate cell holds essentially none of the
  # base grid's weight and scores 0.7458 on the misfit, so the gate declines it
  # and the fit falls back to the plain tensor read.
  expect_equal(off$local_ccd_info$n_cells_refined, 1L)
  expect_equal(off$local_ccd_info$misfit, 0.7458, tolerance = 1e-3)
  expect_lt(off$local_ccd_info$cell_share, 1e-3)
  expect_equal(on$local_ccd_info$n_cells_refined, 0L)
  expect_equal(on$local_ccd_info$n_cells_declined, 1L)
  expect_identical(on$theta_interval_read, "density")
  expect_equal(on$theta_interval_design_mass, 0)

  # And it is a correction, not only a decline. The converged m = 13 reference of
  # the SAME simulation (dev_notes/issue_316, 28561 cells, refinement off) has
  # per-axis 95% widths 0.81646 / 1.49922 / 0.81826 / 1.22962. Measured mean
  # absolute width error against it: 0.7749 with the cloud, 0.3198 without. (The
  # two grids do not share axis RANGES with the reference, so a clamped endpoint
  # is not on the same support; the widths are the comparable part.)
  ref_w <- c(0.81646, 1.49922, 0.81826, 1.22962)
  wid <- function(f) mean(abs((f$theta_ci_hi - f$theta_ci_lo) - ref_w))
  expect_lt(wid(on), wid(off))
  expect_lt(wid(on), 0.4)
  expect_gt(wid(off), 0.7)
})

test_that("the refinement gate holds on both sides of the regime it keys on", {
  skip_if_not_slow()
  # The Gaussian side FIRST: the gate must not spend the win it is protecting.
  # The score is identically zero on a quadratic target, so no cell is declined
  # and the two arms are bit-identical. Measured over 36 configurations:
  # 5.87914 either way, against 21.63762 for not refining at all.
  gs <- .lccd_skew_gate_sweep(reaches = c(2, 3, 4, 6))
  expect_gte(nrow(gs), 30L)
  expect_equal(sum(gs$declined), 0L)
  expect_gt(sum(gs$kept), 0L)
  expect_equal(sum(gs$gated), sum(gs$unconditional), tolerance = 0)
  expect_lt(sum(gs$gated), 6.5)

  # The skewed side: a Gaussian copula with Gamma(2) marginals, same
  # correlation, same refinement. Measured over 27 configurations: gated
  # 24.59641 against unconditional 26.24674. The gate reduces the harm on this
  # target, it does not remove it -- not refining at all scores 23.18738 here,
  # and the threshold that would recover that much regresses on the two least
  # skewed families of the ladder it was measured on (see `.NL_DIAG`).
  ss <- .lccd_skew_gate_sweep(gamma_shape = 2, reaches = c(1.5, 2, 3))
  expect_gte(nrow(ss), 20L)
  expect_gt(sum(ss$declined), 0L)
  expect_lt(sum(ss$gated), sum(ss$unconditional))
  expect_lt(sum(ss$gated), 25.5)

  # And at the far end of the skewness ladder, where the marginal is nearly
  # Gaussian again, the gate must not cost anything either. Gamma(32) has
  # marginal skewness 0.354. Measured over 27 configurations: gated 35.99392,
  # unconditional 36.93487, unrefined 42.59338 -- refining is a win here and the
  # gate keeps it.
  ws <- .lccd_skew_gate_sweep(gamma_shape = 32, reaches = c(1.5, 2, 3))
  expect_gte(nrow(ws), 20L)
  expect_lte(sum(ws$gated), sum(ws$unconditional))
  expect_lt(sum(ws$gated), 37)
})

# --------------------------------------------------------------------------- #
# The centring score in the cell's own curvature units (gcol33/tulpa#324)      #
# --------------------------------------------------------------------------- #

# A quadratic response in the whitened coordinate, built from a KNOWN gradient
# and Hessian on the design's own convention: b0 + g'z + 0.5 z'Hz, whose
# quadratic coefficients are H_jj / 2 on the diagonal and H_jk off it.
.lccd_quad_resp <- function(Z, g, H, b0 = 0) {
  b0 + drop(Z %*% g) + 0.5 * rowSums((Z %*% H) * Z)
}

test_that("mode_gain recovers 0.5 g' (-H)^-1 g from the design's own coefficients", {
  d <- 4L
  ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
  u_c <- rep(0, d)
  g <- c(0.45, -1.2, 0.3, 0.8)

  # Diagonal first, then a Hessian with genuine cross terms: the second is what
  # exercises the factor-2-on-the-diagonal convention, since a diagonal fit reads
  # the same whether the off-diagonal columns were mapped correctly or not.
  A_diag <- diag(c(2, 0.5, 1.25, 3))
  A_full <- matrix(c( 2.0,  0.6, -0.3,  0.1,
                      0.6,  1.4,  0.2, -0.4,
                     -0.3,  0.2,  1.1,  0.5,
                      0.1, -0.4,  0.5,  1.7), d, d)
  for (A in list(A_diag, A_full)) {
    H <- -A
    exact <- 0.5 * drop(t(g) %*% solve(A) %*% g)
    fit <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c,
                                           .lccd_quad_resp(ccd$z, g, H, 1.7),
                                           rep(1, d))
    # The design identifies the quadratic exactly, so the shape score is zero and
    # the two centring readings are the exact ones.
    expect_equal(fit$misfit, 0, tolerance = 1e-10)
    expect_equal(fit$offset, sqrt(sum(g^2)), tolerance = 1e-8)
    expect_equal(fit$mode_gain, exact, tolerance = 1e-8)
    # It is a log-density gain in nats: the fitted peak sits at (-H)^-1 g and the
    # quadratic's value there exceeds its value at the cell by exactly that.
    zstar <- solve(A, g)
    expect_equal(.lccd_quad_resp(matrix(zstar, 1L), g, H) -
                   .lccd_quad_resp(matrix(0, 1L, d), g, H),
                 exact, tolerance = 1e-10)
  }

  # The same whitened quadratic read on a cell whose per-axis marginal spreads
  # are not 1: the whitening is what puts the gain in curvature units, so both
  # readings are unchanged while the physical node positions are not.
  s <- c(0.5, 2, 1, 4)
  H <- -A_full
  exact <- 0.5 * drop(t(g) %*% solve(A_full) %*% g)
  u_nodes <- sweep(ccd$z, 2L, s, FUN = "*")
  scaled <- tulpa:::.joint_local_ccd_misfit(u_nodes, u_c,
                                            .lccd_quad_resp(ccd$z, g, H), s)
  expect_equal(scaled$misfit, 0, tolerance = 1e-10)
  expect_equal(scaled$offset, sqrt(sum(g^2)), tolerance = 1e-8)
  expect_equal(scaled$mode_gain, exact, tolerance = 1e-8)

  # And it is the reading `offset` is not: two cells with the SAME gradient norm
  # and curvature an order of magnitude apart are displaced by very different
  # amounts, which only the scaled form says.
  sharp <- tulpa:::.joint_local_ccd_misfit(
    ccd$z, u_c, .lccd_quad_resp(ccd$z, g, -10 * A_full), rep(1, d))
  expect_equal(sharp$offset, scaled$offset, tolerance = 1e-8)
  expect_equal(sharp$mode_gain, exact / 10, tolerance = 1e-8)
})

test_that("mode_gain declines where the fitted quadratic has no interior peak", {
  d <- 4L
  ccd <- ccd_grid(d, f_0 = sqrt(d) * 1.1)
  u_c <- rep(0, d)
  sd  <- rep(1, d)
  g <- c(0.3, 0.2, -0.1, 0.4)
  # A saddle: concave on three axes, convex on the fourth. The design represents
  # it exactly, so the shape score and the unscaled displacement are both
  # readable and only the curvature-scaled one declines.
  y <- .lccd_quad_resp(ccd$z, g, diag(c(-1, -2, -1.5, 1)))
  fit <- tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, y, sd)
  expect_equal(fit$misfit, 0, tolerance = 1e-10)
  expect_equal(fit$offset, sqrt(sum(g^2)), tolerance = 1e-8)
  expect_true(is.na(fit$mode_gain))

  # And it declines with the fit: a non-finite response identifies nothing, and a
  # design flat on one axis aliases that axis's coefficients.
  bad <- y; bad[1L] <- NA_real_
  expect_true(is.na(tulpa:::.joint_local_ccd_misfit(ccd$z, u_c, bad, sd)$mode_gain))
  z <- ccd$z; z[, d] <- 0
  flat <- tulpa:::.joint_local_ccd_misfit(z, u_c, -0.5 * rowSums(z^2), sd)
  expect_true(is.finite(flat$misfit))
  expect_true(is.na(flat$mode_gain))
})

test_that("a refined cell carries mode_gain beside the offset it scales", {
  tg <- .lccd_gate_target(rho = 0.8, reach = 3, m = 3L)
  rf <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                        colnames(tg$grid), tg$tags, tg$eval,
                                        max_cells = 4L)
  expect_gt(rf$info$n_cells_refined, 0L)
  expect_length(rf$info$mode_gain, rf$info$n_cells_refined)
  expect_true(all(is.finite(rf$info$mode_gain)))
  expect_true(all(rf$info$mode_gain >= 0))
  # This target's refined cell sits ON its own peak, so both centring readings
  # are zero and the gain is the tighter statement of it.
  expect_lt(max(rf$info$offset), 1e-6)
  expect_lt(max(rf$info$mode_gain), 1e-10)
})

# --------------------------------------------------------------------------- #
# The coarse-vs-refined mass ratio (gcol33/tulpa#323)                          #
# --------------------------------------------------------------------------- #

test_that("each refined cell reports its coarse-vs-refined mass comparison", {
  mu <- c(0.5, 0.5); s <- c(0.3, 0.3)
  gd <- .lccd_grid(list(a = c(-1, 0.5, 2), b = c(-1, 0.5, 2)), mu, s)
  out <- tulpa:::.joint_local_ccd_refine(
    gd$grid, gd$lm, modes = NULL, dnode = NULL,
    latent_axes = c("a", "b"), tags = c(a = "identity", b = "identity"),
    eval_nodes = .lccd_eval(mu, s), max_cells = 4L)
  expect_false(is.null(out))
  info <- out$info
  nc <- info$n_cells_refined
  expect_gt(nc, 0L)
  for (nm in c("log_mass_ratio", "log_mass_coarse", "log_mass_refined",
               "max_node_weight"))
    expect_length(info[[nm]], nc)

  # The ratio is the difference of the two masses it is formed from, so the pair
  # is auditable rather than a derived scalar standing alone.
  expect_equal(info$log_mass_ratio, info$log_mass_refined - info$log_mass_coarse,
               tolerance = 1e-12)
  # The coarse estimate is the base grid's own atom, Delta_c exp(ell_c), and
  # Delta_c is 1 on a uniform tensor base.
  expect_equal(info$log_mass_coarse, gd$lm[info$cells], tolerance = 1e-12)

  # The refined estimate is recomputed off the returned grid rather than trusted:
  # each refined cell's replacement block is one contiguous run of design rows
  # carrying Delta_c * delta_j, so its mass is logSumExp(log dnode + lm).
  design <- which(out$weight_kind == "design")
  blk <- length(design) %/% nc
  for (k in seq_len(nc)) {
    ix  <- design[(k - 1L) * blk + seq_len(blk)]
    lv  <- log(out$dnode[ix]) + out$log_marginal[ix]
    lse <- tulpa:::.tulpa_logsumexp(lv)
    expect_equal(info$log_mass_refined[k], lse, tolerance = 1e-12)
    expect_equal(info$max_node_weight[k], max(exp(lv - lse)), tolerance = 1e-12)
  }
  expect_true(all(info$max_node_weight > 0 & info$max_node_weight <= 1))
  # This fixture's refined cell sits on the peak, so every node it is replaced by
  # sits below the atom it replaced and the cloud reads less mass than the atom.
  expect_true(all(info$log_mass_ratio < 0))
})

test_that("a declined cell reports the mass comparison its nodes already paid for", {
  tg <- .lccd_gate_target(rho = 0.8, reach = 1.5, m = 3L, gamma_shape = 2)
  on <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                        colnames(tg$grid), tg$tags, tg$eval,
                                        max_cells = 4L)
  nd <- on$info$n_cells_declined
  expect_gt(nd, 0L)
  expect_equal(on$info$n_cells_refined, 0L)
  for (nm in c("log_mass_ratio_declined", "log_mass_coarse_declined",
               "log_mass_refined_declined", "max_node_weight_declined",
               "mode_gain_declined"))
    expect_length(on$info[[nm]], nd)
  expect_true(all(is.finite(on$info$log_mass_ratio_declined)))
  expect_equal(on$info$log_mass_ratio_declined,
               on$info$log_mass_refined_declined -
                 on$info$log_mass_coarse_declined, tolerance = 1e-12)
  expect_equal(on$info$log_mass_coarse_declined, tg$lm[on$info$cells_declined],
               tolerance = 1e-12)
  expect_true(all(on$info$max_node_weight_declined > 0 &
                    on$info$max_node_weight_declined <= 1))
  # Every cell declined here, so the kept side is empty rather than absent.
  expect_length(on$info$log_mass_ratio, 0L)
  expect_length(on$info$mode_gain, 0L)
})

test_that("the per-axis decline tally is carried on both sides of the gate", {
  # The two free readings each carry the vocabulary their own estimator can
  # decline under, per cell and in cell order, on the kept side and the declined
  # side alike: a cell the gate put back as a mass atom is as legible as one it
  # kept (gcol33/tulpa#334).
  tg <- .lccd_gate_target(rho = 0.8, reach = 1.5, m = 3L, gamma_shape = 2)
  on  <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L)
  off <- tulpa:::.joint_local_ccd_refine(tg$grid, tg$lm, NULL, NULL,
                                         colnames(tg$grid), tg$tags, tg$eval,
                                         max_cells = 4L, skew_max = Inf)
  expect_gt(on$info$n_cells_declined, 0L)
  expect_equal(on$info$n_cells_refined, 0L)
  expect_gt(off$info$n_cells_refined, 0L)

  vocab <- list(box_axis_reasons  = tulpa:::.LCCD_BOX_REASONS,
                bary_axis_reasons = tulpa:::.LCCD_BARY_REASONS)
  for (nm in names(vocab)) {
    kept <- off$info[[nm]]
    decl <- on$info[[paste0(nm, "_declined")]]
    for (m in list(kept, decl)) {
      expect_true(is.matrix(m))
      expect_true(is.integer(m))
      expect_identical(colnames(m), vocab[[nm]])
    }
    expect_identical(nrow(kept), off$info$n_cells_refined)
    expect_identical(nrow(decl), on$info$n_cells_declined)
    # Every cell reaching the cloud passed `.joint_local_ccd_cell_curv()`, so it
    # has a centred stencil and `boundary` cannot fire here; on this fixture the
    # remaining gates do not fire either, so both tallies are clean and the
    # count of declined axes they break down is zero.
    expect_true(all(kept == 0L))
    expect_true(all(decl == 0L))
    # An empty side is a zero-row matrix carrying the vocabulary rather than an
    # absent entry, so a reader takes the column names off either side.
    empty <- on$info[[nm]]
    expect_identical(dim(empty), c(0L, length(vocab[[nm]])))
    expect_identical(colnames(empty), vocab[[nm]])
  }
  # The tally is a per-axis breakdown of the same cells the scalar readings
  # beside it describe, so its rows track them.
  expect_identical(nrow(on$info$box_axis_reasons_declined),
                   length(on$info$log_box_ratio_declined))
  expect_identical(nrow(off$info$bary_axis_reasons),
                   length(off$info$bary_shift))
})

test_that("the mass ratio is stable where one node dominates the cell", {
  # A node sitting several nats above the cell's own coordinate is the regime the
  # ratio is read in, and the one a naive sum of exponentials loses.
  delta <- c(0.1, 0.3, 0.6)
  lm_all <- c(0, 4.44, -900)
  m <- tulpa:::.joint_local_ccd_mass(delta, lm_all, 2)
  expect_equal(m$log_mass_ratio,
               log(0.1 + 0.3 * exp(4.44) + 0.6 * exp(-900)), tolerance = 1e-12)
  expect_equal(m$log_mass_coarse, log(2) + 0, tolerance = 1e-12)
  expect_equal(m$log_mass_refined - m$log_mass_coarse, m$log_mass_ratio,
               tolerance = 1e-12)
  expect_equal(m$max_node_weight,
               0.3 * exp(4.44) / (0.1 + 0.3 * exp(4.44) + 0.6 * exp(-900)),
               tolerance = 1e-12)

  # Shifting every log-marginal by a constant leaves the ratio and the share
  # untouched and moves both masses by that constant, which is what makes the
  # pair readable on a common scale.
  m2 <- tulpa:::.joint_local_ccd_mass(delta, lm_all + 1200, 2)
  expect_equal(m2$log_mass_ratio, m$log_mass_ratio, tolerance = 1e-12)
  expect_equal(m2$max_node_weight, m$max_node_weight, tolerance = 1e-12)
  expect_equal(m2$log_mass_refined - m$log_mass_refined, 1200, tolerance = 1e-9)

  # A non-finite node declines the whole reading rather than reporting part of it.
  bad <- tulpa:::.joint_local_ccd_mass(delta, c(0, NA_real_, 1), 2)
  expect_true(all(vapply(bad, is.na, logical(1))))
})
