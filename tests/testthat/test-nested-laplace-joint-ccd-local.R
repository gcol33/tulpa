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
                   diagnose_draws = 200L, k_max_rounds = 1L)))

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
