# The batched (multi-response) joint nested-Laplace driver against the
# single-species fit at the same outer grid.
#
# `tulpa_nl_joint_batch()` runs B per-species Newton solves that share one
# design, one sparsity pattern and one fused cell-coupling scatter. What the
# sharing is allowed to buy is bandwidth: each species' trajectory is the
# trajectory of its own independent single-species fit, so the whole stack is
# verifiable against `tulpa_nl_joint_single()`, which marshals the same
# `responses` / `prior` through `.tulpa_nl_joint_marshal()` and therefore lands
# on a byte-identical outer grid.
#
# Two fixtures, both driven by a coupled spec because the batched entry accepts
# only families whose every data arm is cell-coupled:
#
#   * `test_separable_bernoulli` -- single arm, one row per cell. Exercises the
#     within-arm scatter, the per-species warm-start chain across grid cells,
#     and both scatter policies: the driver picks dense or sparse purely by
#     n_x >= SPARSE_THRESHOLD, so a large field routes the batch through
#     SparseScatterPolicy and the oracle through the single-species sparse path
#     (itself pinned to the single-species dense path in
#     test-nested-laplace-joint-sparse-equivalence.R).
#
#   * `test_occupancy_mixture` -- two arms, genuinely non-separable at a cell
#     with no detection. Exercises the cross-arm slot cache and the no-data arm
#     (`y_batch[[k]] = NULL`), which is the shape the occupancy arm carries.
#
# The equivalence is asserted twice, in two blocks: once as numerical agreement,
# and once as exact agreement, which is the stronger claim the driver's header
# makes. A red in the exact block alone locates the overstatement in the header
# rather than in the fit.

.nlb_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  n_neighbors <- vapply(nbr, length, integer(1))
  list(adj_row_ptr     = as.integer(c(0L, cumsum(n_neighbors))),
       adj_col_idx     = as.integer(unlist(nbr)) - 1L,
       n_neighbors     = as.integer(n_neighbors),
       n_spatial_units = n_s)
}

# --------------------------------------------------------------------------- #
# Fixture 1: single-arm coupled Bernoulli, B species over one shared design    #
# --------------------------------------------------------------------------- #

.nlb_bern_sim <- function(seed, n_batch, n_s, n_per_unit) {
  set.seed(seed)
  N <- n_s * n_per_unit
  spatial_idx <- as.integer(rep(seq_len(n_s), each = n_per_unit))
  x <- stats::rnorm(N)
  y <- vapply(seq_len(n_batch), function(s) {
    field <- stats::rnorm(n_s, 0, 0.6)
    as.numeric(stats::rbinom(
      N, 1L, stats::plogis(0.2 * s + 0.5 * x + field[spatial_idx])))
  }, numeric(N))
  list(n_s = n_s, N = N, n_batch = n_batch,
       X = cbind(intercept = 1, x = x),
       spatial_idx = spatial_idx,
       y = matrix(y, nrow = N, ncol = n_batch),
       adj = .nlb_chain_adj(n_s))
}

.nlb_bern_arm <- function(sim, y) {
  list(y = as.numeric(y), n_trials = rep(1L, sim$N), X = sim$X,
       spatial_idx = sim$spatial_idx, family = "binomial", phi = 1,
       coupled = TRUE, cell_obs_map = seq_len(sim$N))
}

# The low-level joint entries read the per-arm index vectors off the BLOCK
# spec, one element per arm, where the tulpa() front door reads them off the
# arm. One arm here, so a length-1 list.
.nlb_bern_prior <- function(sim, sigma_grid) {
  c(list(type = "icar", sigma_grid = sigma_grid,
         spatial_idx = list(sim$spatial_idx)), sim$adj)
}

.nlb_bern_batch <- function(sim, sigma_grid, phi_batch = NULL, y_batch = NULL) {
  if (is.null(phi_batch)) phi_batch <- matrix(1, nrow = 1L, ncol = sim$n_batch)
  if (is.null(y_batch))   y_batch   <- list(sim$y)
  tulpa_nl_joint_batch(
    responses = list(occ = .nlb_bern_arm(sim, sim$y[, 1L])),
    prior     = .nlb_bern_prior(sim, sigma_grid),
    n_batch   = sim$n_batch,
    y_batch   = y_batch,
    phi_batch = phi_batch,
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_separable_bernoulli",
    store_Q   = FALSE)
}

.nlb_bern_single <- function(sim, s, sigma_grid) {
  tulpa_nl_joint_single(
    responses = list(occ = .nlb_bern_arm(sim, sim$y[, s])),
    prior     = .nlb_bern_prior(sim, sigma_grid),
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_separable_bernoulli",
    store_Q   = FALSE)
}

# --------------------------------------------------------------------------- #
# Fixture 2: the two-arm occupancy mixture, B species over one shared design   #
# --------------------------------------------------------------------------- #

# One `coupled_occ_data()` draw per species on a single cell / visit layout, so
# the design the batch shares is the design each oracle fit sees.
.nlb_occ_sim <- function(seed, n_batch, n_cells, n_visits) {
  d <- lapply(seq_len(n_batch),
              function(s) coupled_occ_data(seed + s, n_cells, n_visits,
                                           b_occ = 0.4, b_det = -0.3))
  list(n_batch = n_batch, n_cells = n_cells, n_visits = n_visits, d = d,
       y_det = matrix(vapply(d, function(x) x$y_det,
                             numeric(n_cells * n_visits)),
                      nrow = n_cells * n_visits, ncol = n_batch))
}

.nlb_occ_prior <- function(sim, sigma_grid) {
  n_v <- sim$n_cells * sim$n_visits
  # obs_idx names the block's latent unit for each observation, 1-based.
  list(list(type = "iid", n_units = 1L, sigma_grid = sigma_grid,
            obs_idx = list(rep(1L, sim$n_cells), rep(1L, n_v))))
}

.nlb_occ_batch <- function(sim, sigma_grid) {
  tulpa_nl_joint_batch(
    responses = coupled_occ_arms(sim$d[[1L]]),
    prior     = .nlb_occ_prior(sim, sigma_grid),
    n_batch   = sim$n_batch,
    # The occupancy arm carries no response of its own: the cell's detection
    # history lives on the detection arm, so its y_batch slot is NULL.
    y_batch   = list(NULL, sim$y_det),
    phi_batch = matrix(1, nrow = 2L, ncol = sim$n_batch),
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_occupancy_mixture",
    store_Q   = FALSE)
}

.nlb_occ_single <- function(sim, s, sigma_grid) {
  tulpa_nl_joint_single(
    responses = coupled_occ_arms(sim$d[[s]]),
    prior     = .nlb_occ_prior(sim, sigma_grid),
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_occupancy_mixture",
    store_Q   = FALSE)
}

# --------------------------------------------------------------------------- #
# Shared assertions                                                            #
# --------------------------------------------------------------------------- #

# The oracle is the raw kernel solve; the batched driver folds the regularizing
# hyperprior into each species' log-marginal on top of it, so the oracle's
# log-marginal is compared with that fold added.
.nlb_single_folded <- function(sp, single) {
  as.numeric(single$log_marginal) + as.numeric(sp$log_hyperprior)
}

.nlb_expect_numeric_match <- function(sp, single, label,
                                      lm_tol = 1e-8, mode_tol = 1e-7) {
  expect_equal(as.numeric(sp$log_marginal), .nlb_single_folded(sp, single),
               tolerance = lm_tol, info = paste(label, "log_marginal"))
  expect_equal(dim(sp$modes), dim(single$modes),
               info = paste(label, "mode shape"))
  expect_equal(as.numeric(sp$modes), as.numeric(single$modes),
               tolerance = mode_tol, info = paste(label, "modes"))
}

# The header's claim: the fused scatter changes the cost of a species' solve and
# nothing else, so every number and the iteration count itself are reproduced.
.nlb_expect_exact_match <- function(sp, single, label) {
  expect_identical(as.numeric(sp$log_marginal),
                   .nlb_single_folded(sp, single),
                   info = paste(label, "log_marginal is bit-identical"))
  expect_identical(as.numeric(sp$modes), as.numeric(single$modes),
                   info = paste(label, "modes are bit-identical"))
  expect_identical(as.integer(sp$n_iter), as.integer(single$n_iter),
                   info = paste(label, "inner Newton took the same steps"))
}

# --------------------------------------------------------------------------- #
# (1) Dense scatter policy: B species against B single-species fits            #
# --------------------------------------------------------------------------- #

test_that("batched dense path reproduces the single-species fit for every species", {
  skip_on_cran()
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  sim <- .nlb_bern_sim(seed = 4021L, n_batch = 3L, n_s = 20L, n_per_unit = 4L)
  sigma_grid <- c(0.5, 1.0)
  res <- .nlb_bern_batch(sim, sigma_grid)

  expect_length(res$per_species, sim$n_batch)
  for (s in seq_len(sim$n_batch)) {
    .nlb_expect_numeric_match(res$per_species[[s]],
                              .nlb_bern_single(sim, s, sigma_grid),
                              paste0("dense species ", s))
  }
})

test_that("batched dense path is bit-identical to the single-species fit", {
  skip_on_cran()
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  sim <- .nlb_bern_sim(seed = 4021L, n_batch = 3L, n_s = 20L, n_per_unit = 4L)
  sigma_grid <- c(0.5, 1.0)
  res <- .nlb_bern_batch(sim, sigma_grid)

  for (s in seq_len(sim$n_batch)) {
    .nlb_expect_exact_match(res$per_species[[s]],
                            .nlb_bern_single(sim, s, sigma_grid),
                            paste0("dense species ", s))
  }
})

# --------------------------------------------------------------------------- #
# (2) Sparse scatter policy: the same comparison above SPARSE_THRESHOLD        #
# --------------------------------------------------------------------------- #

test_that("batched sparse path reproduces the single-species fit for every species", {
  skip_on_cran()
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  # n_x = 2 fixed effects + 240 field units, past the 200 the driver dispatches
  # on, so the batch runs through SparseScatterPolicy and the oracle through the
  # single-species sparse Newton.
  sim <- .nlb_bern_sim(seed = 4022L, n_batch = 2L, n_s = 240L, n_per_unit = 2L)
  sigma_grid <- c(0.7, 1.2)
  res <- .nlb_bern_batch(sim, sigma_grid)

  expect_length(res$per_species, sim$n_batch)
  for (s in seq_len(sim$n_batch)) {
    single <- .nlb_bern_single(sim, s, sigma_grid)
    .nlb_expect_numeric_match(res$per_species[[s]], single,
                              paste0("sparse species ", s))
    .nlb_expect_exact_match(res$per_species[[s]], single,
                            paste0("sparse species ", s))
  }
})

# --------------------------------------------------------------------------- #
# (3) n_batch = 1                                                              #
# --------------------------------------------------------------------------- #

test_that("n_batch = 1 reproduces the single-species path exactly", {
  skip_on_cran()
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  sim <- .nlb_bern_sim(seed = 4023L, n_batch = 1L, n_s = 16L, n_per_unit = 5L)
  sigma_grid <- c(0.6, 1.1)
  res <- .nlb_bern_batch(sim, sigma_grid)
  single <- .nlb_bern_single(sim, 1L, sigma_grid)

  expect_length(res$per_species, 1L)
  .nlb_expect_numeric_match(res$per_species[[1L]], single, "B = 1")
  .nlb_expect_exact_match(res$per_species[[1L]], single, "B = 1")
})

# --------------------------------------------------------------------------- #
# (4) Cross-arm coupling and a no-data arm                                     #
# --------------------------------------------------------------------------- #

test_that("batched cross-arm scatter reproduces the single-species coupled fit", {
  skip_on_cran()
  coupled_occ_register()

  sim <- .nlb_occ_sim(seed = 7100L, n_batch = 3L, n_cells = 40L, n_visits = 3L)
  sigma_grid <- c(0.8, 1.2)
  res <- .nlb_occ_batch(sim, sigma_grid)

  expect_length(res$per_species, sim$n_batch)
  for (s in seq_len(sim$n_batch)) {
    single <- .nlb_occ_single(sim, s, sigma_grid)
    .nlb_expect_numeric_match(res$per_species[[s]], single,
                              paste0("coupled species ", s))
    .nlb_expect_exact_match(res$per_species[[s]], single,
                            paste0("coupled species ", s))
  }
})

test_that("every species of the coupled fixture reaches the non-separable branch", {
  # The fixture is an arbiter for the cross-arm slot cache only while some cell
  # stays in the all-undetected branch, so that is read off the data.
  sim <- .nlb_occ_sim(seed = 7100L, n_batch = 3L, n_cells = 40L, n_visits = 3L)
  for (s in seq_len(sim$n_batch)) {
    expect_gt(sum(sim$d[[s]]$n_seen == 0L), 0L)
  }
})

# --------------------------------------------------------------------------- #
# (5) Per-species convergence reaches the caller                               #
# --------------------------------------------------------------------------- #

test_that("the batched driver reports per-species per-cell convergence", {
  skip_on_cran()
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  sim <- .nlb_bern_sim(seed = 4024L, n_batch = 2L, n_s = 18L, n_per_unit = 4L)
  sigma_grid <- c(0.5, 1.0)
  res <- .nlb_bern_batch(sim, sigma_grid)

  for (s in seq_len(sim$n_batch)) {
    sp <- res$per_species[[s]]
    single <- .nlb_bern_single(sim, s, sigma_grid)
    expect_true("converged" %in% names(sp))
    expect_type(sp$converged, "logical")
    # One flag per outer cell, and the entry lays more cells than the sigma
    # axis has nodes, so the oracle's own grid is what gives the length.
    expect_length(sp$converged, length(single$log_marginal))
    expect_true(all(sp$converged))
    expect_identical(sp$converged, as.logical(single$converged))
  }
})

# --------------------------------------------------------------------------- #
# (6) Input contracts the batched entry owns                                   #
# --------------------------------------------------------------------------- #

test_that("the batched entry validates the per-species response and dispersion shapes", {
  cpp_register_test_separable_bernoulli_coupling()
  skip_if_not(cpp_cell_coupling_registry_has("test_separable_bernoulli"),
              "test_separable_bernoulli coupling spec not registered")

  sim <- .nlb_bern_sim(seed = 4025L, n_batch = 2L, n_s = 8L, n_per_unit = 3L)
  sigma_grid <- 1.0

  expect_error(
    .nlb_bern_batch(sim, sigma_grid,
                    phi_batch = matrix(1, nrow = 1L, ncol = sim$n_batch + 1L)),
    "phi_batch")
  expect_error(
    .nlb_bern_batch(sim, sigma_grid,
                    y_batch = list(sim$y[, 1L, drop = FALSE])),
    "y_batch")
})

test_that("the batched entry refuses a family whose arms are not cell-coupled", {
  sim <- .nlb_bern_sim(seed = 4026L, n_batch = 2L, n_s = 8L, n_per_unit = 3L)
  expect_error(
    tulpa_nl_joint_batch(
      responses = list(occ = .nlb_bern_arm(sim, sim$y[, 1L])),
      prior     = .nlb_bern_prior(sim, 1.0),
      n_batch   = sim$n_batch,
      y_batch   = list(sim$y),
      phi_batch = matrix(1, nrow = 1L, ncol = sim$n_batch),
      cell_coupling = "separable"),
    "cell-coupling spec")
})

# --------------------------------------------------------------------------- #
# (5) The weights: the multi-block driver's, on the same grid                  #
# --------------------------------------------------------------------------- #

# Each species' weights must be the ones the multi-block driver reports when it
# fits that species alone on the same fixed grid: the hyperprior folded into the
# log-marginal and the cell measure applied, not a softmax of the kernel's
# log-marginal. The sigma nodes are unevenly spaced on log sigma, so the cell
# measure is not constant and a plain softmax cannot match.
test_that("batched weights are the multi-block driver's weights on the same grid", {
  skip_on_cran()
  coupled_occ_register()

  sim <- .nlb_occ_sim(seed = 7102L, n_batch = 2L, n_cells = 40L, n_visits = 3L)
  sigma_grid <- c(0.4, 0.6, 1.5)
  res <- .nlb_occ_batch(sim, sigma_grid)

  for (s in seq_len(sim$n_batch)) {
    sp <- res$per_species[[s]]
    looped <- suppressWarnings(tulpa_nested_laplace_joint(
      responses = coupled_occ_arms(sim$d[[s]]),
      prior     = .nlb_occ_prior(sim, sigma_grid),
      cell_coupling = "test_occupancy_mixture",
      control = list(max_iter = 60L, tol = 1e-8, integration = "grid",
                     prune = FALSE, diagnose_k = FALSE, k_quality = "none",
                     adaptive_grid = FALSE, var_of_means_consistency = FALSE,
                     auto_recenter = FALSE, store_Q = FALSE)))
    key <- function(tg) apply(signif(as.matrix(tg), 12L), 1L, paste,
                              collapse = "|")
    idx <- match(key(res$theta_grid), key(looped$theta_grid))
    expect_false(anyNA(idx), info = paste("species", s, "grid cells"))
    expect_equal(as.numeric(sp$weights), as.numeric(looped$weights[idx]),
                 tolerance = 1e-8, info = paste("species", s, "weights"))
    expect_equal(as.numeric(sp$log_quad), as.numeric(looped$log_quad[idx]),
                 tolerance = 1e-12, info = paste("species", s, "log_quad"))

    lm <- as.numeric(sp$log_marginal)
    soft <- exp(lm - max(lm)); soft <- soft / sum(soft)
    expect_gt(max(abs(soft - as.numeric(sp$weights))), 1e-6)
  }
})

# --------------------------------------------------------------------------- #
# (6) A dispersion axis carried per species                                    #
# --------------------------------------------------------------------------- #

# Two coupled gaussian arms (`test_weighted_gaussian`, whose density reads the
# per-species dispersion), B species, each arm integrating its dispersion as an
# outer axis crossed onto the field grid. Every species sits at its OWN nodes
# over one shared cell layout, and the two arms carry different node counts, so
# a cell / species / arm slip in the per-cell dispersion table reads another
# value rather than the same one.
.nlb_wg_sim <- function(seed, n_batch, n_s, n_per_unit) {
  set.seed(seed)
  N <- n_s * n_per_unit
  spatial_idx <- as.integer(rep(seq_len(n_s), each = n_per_unit))
  x <- stats::rnorm(N)
  draw <- function(mult) {
    matrix(vapply(seq_len(n_batch), function(s) {
      field <- stats::rnorm(n_s, 0, 0.5)
      as.numeric(mult * (0.2 * s) + 0.5 * x + field[spatial_idx] +
                   stats::rnorm(N, 0, 0.3 + 0.2 * s))
    }, numeric(N)), nrow = N, ncol = n_batch)
  }
  list(n_s = n_s, N = N, n_batch = n_batch,
       X = cbind(intercept = 1, x = x), spatial_idx = spatial_idx,
       y_a = draw(1), y_b = draw(-1), adj = .nlb_chain_adj(n_s))
}

.nlb_wg_arms <- function(sim, s) {
  arm <- function(y) list(y = as.numeric(y), n_trials = rep(1L, sim$N),
                          X = sim$X, spatial_idx = sim$spatial_idx,
                          family = "gaussian", phi = 1, coupled = TRUE,
                          cell_obs_map = seq_len(sim$N))
  list(a = arm(sim$y_a[, s]), b = arm(sim$y_b[, s]))
}

.nlb_wg_prior <- function(sim, tau_grid) {
  list(c(list(type = "icar", tau_grid = tau_grid,
              spatial_idx = list(sim$spatial_idx, sim$spatial_idx)), sim$adj))
}

# Species s's dispersion nodes (residual variances): arm a three nodes, arm b
# two, both scaled per species.
.nlb_wg_phi_grid <- function(s) {
  list(a = c(0.10, 0.25, 0.60) * (1 + 0.5 * s),
       b = c(0.20, 0.55) * (1 + 0.3 * s))
}

.nlb_wg_batch <- function(sim, tau_grid) {
  cpp_register_test_weighted_gaussian_coupling(c(0L, 1L))
  tulpa_nl_joint_batch(
    responses = .nlb_wg_arms(sim, 1L),
    prior     = .nlb_wg_prior(sim, tau_grid),
    n_batch   = sim$n_batch,
    y_batch   = list(sim$y_a, sim$y_b),
    phi_batch = matrix(1, nrow = 2L, ncol = sim$n_batch),
    phi_grid_batch = lapply(seq_len(sim$n_batch), .nlb_wg_phi_grid),
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_weighted_gaussian",
    store_Q   = FALSE)
}

.nlb_wg_single <- function(sim, s, tau_grid) {
  cpp_register_test_weighted_gaussian_coupling(c(0L, 1L))
  tulpa_nl_joint_single(
    responses = .nlb_wg_arms(sim, s),
    prior     = .nlb_wg_prior(sim, tau_grid),
    phi_grid  = .nlb_wg_phi_grid(s),
    max_iter  = 60L, tol = 1e-8,
    cell_coupling = "test_weighted_gaussian",
    store_Q   = FALSE)
}

test_that("a per-species dispersion axis reproduces each species' single fit exactly", {
  skip_on_cran()
  sim <- .nlb_wg_sim(seed = 7201L, n_batch = 3L, n_s = 10L, n_per_unit = 3L)
  tau_grid <- c(0.6, 1.6, 6.0)
  res <- .nlb_wg_batch(sim, tau_grid)

  n_cells <- length(tau_grid) * 3L * 2L
  expect_length(res$per_species, sim$n_batch)
  for (s in seq_len(sim$n_batch)) {
    sp <- res$per_species[[s]]
    single <- .nlb_wg_single(sim, s, tau_grid)
    expect_length(sp$log_marginal, n_cells)
    # The species' grid carries its own dispersion nodes, and no other's.
    pg <- .nlb_wg_phi_grid(s)
    expect_setequal(unique(sp$theta_grid[, "phi_a"]), pg$a)
    expect_setequal(unique(sp$theta_grid[, "phi_b"]), pg$b)
    .nlb_expect_numeric_match(sp, single, paste0("phi-axis species ", s))
    .nlb_expect_exact_match(sp, single, paste0("phi-axis species ", s))
  }
  # The axis is live: species differing only in their dispersion nodes would
  # otherwise coincide cell for cell.
  expect_false(isTRUE(all.equal(res$per_species[[1L]]$log_marginal,
                                res$per_species[[2L]]$log_marginal)))
})

test_that("per-species dispersion-axis weights are the multi-block driver's weights", {
  skip_on_cran()
  sim <- .nlb_wg_sim(seed = 7202L, n_batch = 2L, n_s = 10L, n_per_unit = 3L)
  tau_grid <- c(0.45, 2.8, 6.2)
  res <- .nlb_wg_batch(sim, tau_grid)

  key <- function(tg) apply(signif(as.matrix(tg), 12L), 1L, paste,
                            collapse = "|")
  for (s in seq_len(sim$n_batch)) {
    sp <- res$per_species[[s]]
    looped <- suppressWarnings(tulpa_nested_laplace_joint(
      responses = .nlb_wg_arms(sim, s),
      prior     = .nlb_wg_prior(sim, tau_grid),
      phi_grid  = .nlb_wg_phi_grid(s),
      cell_coupling = "test_weighted_gaussian",
      control = list(max_iter = 60L, tol = 1e-8, integration = "grid",
                     prune = FALSE, diagnose_k = FALSE, k_quality = "none",
                     adaptive_grid = FALSE, var_of_means_consistency = FALSE,
                     auto_recenter = FALSE, store_Q = FALSE)))
    expect_identical(colnames(sp$theta_grid), colnames(looped$theta_grid))
    idx <- match(key(sp$theta_grid), key(looped$theta_grid))
    expect_false(anyNA(idx), info = paste("species", s, "grid cells"))
    expect_equal(as.numeric(sp$log_marginal),
                 as.numeric(looped$log_marginal[idx]),
                 tolerance = 1e-8, info = paste("species", s, "log_marginal"))
    expect_equal(as.numeric(sp$weights), as.numeric(looped$weights[idx]),
                 tolerance = 1e-8, info = paste("species", s, "weights"))
    expect_equal(as.numeric(sp$log_quad), as.numeric(looped$log_quad[idx]),
                 tolerance = 1e-12, info = paste("species", s, "log_quad"))
  }
})

test_that("the batched entry refuses per-species dispersion axes of different shape", {
  sim <- .nlb_wg_sim(seed = 7203L, n_batch = 2L, n_s = 6L, n_per_unit = 2L)
  cpp_register_test_weighted_gaussian_coupling(c(0L, 1L))
  expect_error(
    tulpa_nl_joint_batch(
      responses = .nlb_wg_arms(sim, 1L),
      prior     = .nlb_wg_prior(sim, c(1.0, 4.0)),
      n_batch   = sim$n_batch,
      y_batch   = list(sim$y_a, sim$y_b),
      phi_batch = matrix(1, nrow = 2L, ncol = sim$n_batch),
      phi_grid_batch = list(list(a = c(0.2, 0.5)), list(a = c(0.2, 0.5, 0.9))),
      cell_coupling = "test_weighted_gaussian", store_Q = FALSE),
    "same node count")
})
