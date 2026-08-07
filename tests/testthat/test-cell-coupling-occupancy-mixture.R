# The engine's own genuinely coupled likelihood (gcol33/tulpa#300): a two-arm
# occupancy mixture registered from src/ as "test_occupancy_mixture".
#
# Until this fixture existed every non-separable `CellCouplingSpec` lived
# downstream in tulpaObs, so the cross-arm scatter, the dense-pair allocation
# and the per-cell derivative contract were exercised only by a consumer. The
# spec here is small enough that its value, gradient and full cross-arm Hessian
# can be differenced against each other in a structural test, and that a direct
# quadrature of the conditional posterior it induces is affordable (that part
# lives in test-inner-skew.R).
#
# What makes it an arbiter rather than a separable model wearing a coupled
# label: a cell with no detection has a nonzero d^2 log p_cell / d eta_occ
# d eta_det, and that is asserted numerically both at an arbitrary eta and at
# the mode of a real fit, never claimed in a comment.

# --------------------------------------------------------------------------- #
# (1) Structural: the per-cell contract -- value, gradient, full Hessian       #
# --------------------------------------------------------------------------- #

# The compiled spec's eta-space derivatives at one cell.
.occ_eval <- function(eta_occ, eta_det, y_det, grad_only = FALSE) {
  cpp_cell_coupling_evaluate(
    "test_occupancy_mixture",
    eta = list(eta_occ, eta_det), y = list(0, y_det),
    family = c("binomial", "binomial"), phi = c(1, 1),
    grad_only = grad_only)
}

# log p_cell written out directly from the mixture definition.
.occ_ref_value <- function(eta_occ, eta_det, y_det) {
  psi <- plogis(eta_occ); p <- plogis(eta_det)
  if (sum(y_det) > 0) {
    log(psi) + sum(y_det * log(p) + (1 - y_det) * log(1 - p))
  } else {
    log(psi * prod(1 - p) + 1 - psi)
  }
}

# The spec's negative Hessian over (eta_occ, eta_det_1, ..., eta_det_J),
# assembled from the diagonal buffers and the dense cross blocks the way the
# kernel's scatter reads them.
.occ_neg_hess <- function(r, n_visits) {
  n <- 1L + n_visits
  H <- matrix(0, n, n)
  diag(H) <- c(r$neg_hess_diag[[1]], r$neg_hess_diag[[2]])
  cross_od <- r$cross_hess[[1]][[2]]
  if (!is.null(cross_od)) {
    H[1L, 2:n] <- cross_od[1L, ]
    H[2:n, 1L] <- cross_od[1L, ]
  }
  cross_dd <- r$cross_hess[[2]][[2]]
  if (!is.null(cross_dd)) H[2:n, 2:n] <- H[2:n, 2:n] + cross_dd
  H
}

# Central difference of the spec's own analytic gradient -- the same surface a
# third-derivative tensor differences one level up.
.occ_fd_neg_hess <- function(eta_occ, eta_det, y_det, h = 1e-6) {
  eta <- c(eta_occ, eta_det)
  n <- length(eta)
  grad_at <- function(e) {
    r <- .occ_eval(e[1L], e[-1L], y_det)
    c(r$grad[[1]], r$grad[[2]])
  }
  H <- matrix(0, n, n)
  for (i in seq_len(n)) {
    ep <- eta; em <- eta
    ep[i] <- ep[i] + h; em[i] <- em[i] - h
    H[i, ] <- -(grad_at(ep) - grad_at(em)) / (2 * h)
  }
  H
}

.occ_fd_grad <- function(eta_occ, eta_det, y_det, h = 1e-6) {
  eta <- c(eta_occ, eta_det)
  vapply(seq_along(eta), function(i) {
    ep <- eta; em <- eta
    ep[i] <- ep[i] + h; em[i] <- em[i] - h
    (.occ_ref_value(ep[1L], ep[-1L], y_det) -
     .occ_ref_value(em[1L], em[-1L], y_det)) / (2 * h)
  }, numeric(1))
}

# (eta_occ, eta_det, y_det) triples spanning both branches of the mixture and
# both tails of the occupancy logit.
.occ_cases <- list(
  list(a = 0.4,  v = c(-0.3, 0.6, 0.1), y = c(0, 0, 0), coupled = TRUE),
  list(a = -1.2, v = c(-2.0, 0.9),      y = c(0, 0),    coupled = TRUE),
  list(a = 2.1,  v = c(1.4, -0.7, 0.2, 0.9), y = c(0, 0, 0, 0), coupled = TRUE),
  list(a = 0.4,  v = c(-0.3, 0.6, 0.1), y = c(0, 1, 0), coupled = FALSE),
  list(a = -0.8, v = c(0.5, 0.5),       y = c(1, 1),    coupled = FALSE)
)

test_that("the coupled occupancy fixture registers under its own name", {
  coupled_occ_register()
  expect_true(cpp_cell_coupling_registry_has("test_occupancy_mixture"))
  r <- .occ_eval(0.2, c(0.1, -0.4), c(0, 0))
  expect_equal(r$arm_ids, c(0L, 1L))
})

test_that("evaluate_cell reproduces the mixture density on both branches", {
  coupled_occ_register()
  for (cs in .occ_cases) {
    r <- .occ_eval(cs$a, cs$v, cs$y)
    expect_equal(r$value, .occ_ref_value(cs$a, cs$v, cs$y), tolerance = 1e-12)
  }
})

test_that("the analytic gradient differences the cell log-density", {
  coupled_occ_register()
  for (cs in .occ_cases) {
    r <- .occ_eval(cs$a, cs$v, cs$y)
    g <- c(r$grad[[1]], r$grad[[2]])
    expect_equal(g, .occ_fd_grad(cs$a, cs$v, cs$y), tolerance = 1e-7)
  }
})

test_that("the full cross-arm Hessian differences the analytic gradient", {
  coupled_occ_register()
  for (cs in .occ_cases) {
    r <- .occ_eval(cs$a, cs$v, cs$y)
    H <- .occ_neg_hess(r, length(cs$v))
    expect_equal(H, .occ_fd_neg_hess(cs$a, cs$v, cs$y), tolerance = 1e-6)
    # Symmetric, as a Hessian must be -- the (det, det) block is written on
    # both sides of its diagonal so a direct probe sees the whole matrix.
    expect_equal(H, t(H), tolerance = 1e-14)
  }
})

test_that("an undetected cell couples the arms and a detected cell does not", {
  coupled_occ_register()
  for (cs in .occ_cases) {
    r <- .occ_eval(cs$a, cs$v, cs$y)
    cross_od <- r$cross_hess[[1]][[2]]
    cross_dd <- r$cross_hess[[2]][[2]]
    J <- length(cs$v)
    if (cs$coupled) {
      # Every visit couples to the occupancy state, and (for J > 1) to every
      # other visit: the mixture puts them all inside one logarithm.
      expect_true(all(abs(as.numeric(cross_od)) > 1e-6))
      if (J > 1L) {
        off <- cross_dd[upper.tri(cross_dd)]
        expect_true(all(abs(off) > 1e-8))
      }
    } else {
      # A detected cell factorises exactly: log psi plus a per-visit sum.
      expect_equal(as.numeric(cross_od), rep(0, J))
      expect_equal(as.numeric(cross_dd), rep(0, J * J))
    }
  }
})

test_that("dense_cross_pairs allocates the two blocks the spec writes, and no others", {
  coupled_occ_register()
  r <- .occ_eval(0.3, c(0.2, -0.1, 0.4), c(0, 0, 0))
  # The occupancy arm holds one row per cell, so its self block has no
  # off-diagonal entry and is not allocated. Both blocks the spec fills are.
  expect_null(r$cross_hess[[1]][[1]])
  expect_false(is.null(r$cross_hess[[1]][[2]]))
  expect_false(is.null(r$cross_hess[[2]][[2]]))
  expect_equal(dim(r$cross_hess[[1]][[2]]), c(1L, 3L))
  expect_equal(dim(r$cross_hess[[2]][[2]]), c(3L, 3L))
  # The dense path is taken, not the rank-1 self-cross shortcut: the explicit
  # (det, det) block is what a third-derivative tensor differences.
  expect_equal(r$rank1_coef, c(0, 0))
})

test_that("a grad-only request keeps the gradient exact and skips the curvature", {
  coupled_occ_register()
  for (cs in .occ_cases) {
    full <- .occ_eval(cs$a, cs$v, cs$y)
    lite <- .occ_eval(cs$a, cs$v, cs$y, grad_only = TRUE)
    expect_equal(lite$value, full$value, tolerance = 1e-14)
    expect_equal(lite$grad, full$grad, tolerance = 1e-14)
    expect_equal(as.numeric(lite$neg_hess_diag[[1]]), 0)
    expect_equal(as.numeric(lite$neg_hess_diag[[2]]), rep(0, length(cs$v)))
    expect_equal(as.numeric(lite$cross_hess[[1]][[2]]), rep(0, length(cs$v)))
  }
})

# --------------------------------------------------------------------------- #
# (2) Recovery: an end-to-end joint fit through the coupled path              #
# --------------------------------------------------------------------------- #

test_that("a coupled fit lands on the exact mode of the posterior it claims to solve", {
  skip_on_cran()
  coupled_occ_register()
  # Intercept-only, so the conditional posterior is two-dimensional and its
  # mode can be located independently by a general-purpose optimiser on the
  # same density. A point-recovery check at this size only says the mode is
  # within sampling noise of the truth; this says the coupled inner Newton
  # solves the right problem, to six digits.
  beta_prec <- 0.25
  d <- coupled_occ_data(seed = 311, n_cells = 100L, n_visits = 4L,
                        b_occ = 0.2, b_det = -0.5)
  fit <- tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, beta_prec = beta_prec),
    prior = coupled_occ_flat_prior(d),
    cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 300L, tol = 1e-12, diagnose_k = FALSE))
  md <- as.matrix(fit$modes)[which.max(fit$log_marginal), ]

  lp <- coupled_occ_log_post(d, beta_prec)
  ref <- stats::optim(c(0, 0), function(v) -lp(v[1L], v[2L]),
                      method = "BFGS", control = list(reltol = 1e-14))
  expect_equal(md[1L], ref$par[1L], tolerance = 1e-5)
  expect_equal(md[2L], ref$par[2L], tolerance = 1e-5)

  # And the mode is inside the posterior it came from: within three exact
  # posterior standard deviations of the simulated truth, both arms.
  q <- coupled_occ_quadrature(lp, center = ref$par, half = 12, n_grid = 801L)
  expect_lt(abs(md[1L] - d$b_occ), 3 * q$a[["sd"]])
  expect_lt(abs(md[2L] - d$b_det), 3 * q$b[["sd"]])
})

.occ_chain_adj <- function(n_s) {
  nbr <- lapply(seq_len(n_s),
                function(s) setdiff(c(s - 1L, s + 1L), c(0L, n_s + 1L)))
  nn <- vapply(nbr, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nbr)) - 1L,
       n_neighbors = as.integer(nn), n_spatial_units = n_s)
}

# An occupancy field over cells, entering the occupancy arm only -- the shape a
# real single-season occupancy model has, and the one that forces the cross-arm
# curvature to reach the field's latent dofs through the coupled scatter.
.occ_spatial_fit <- function(d, adj, force_sparse = FALSE) {
  n_v <- d$n_cells * d$n_visits
  prior <- list(type = "icar", n_spatial_units = d$n_cells,
                adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                n_neighbors = adj$n_neighbors,
                sigma_grid = c(0.3, 0.5, 0.8, 1.2),
                spatial_idx = list(seq_len(d$n_cells), rep(0L, n_v)))
  tulpa_nested_laplace_joint(
    responses = coupled_occ_arms(d, spatial_idx = TRUE),
    prior = prior, cell_coupling = "test_occupancy_mixture",
    control = list(max_iter = 200L, tol = 1e-10, diagnose_k = FALSE,
                   force_sparse = force_sparse))
}

test_that("a coupled occupancy fit recovers both intercepts and stays non-separable at its mode", {
  skip_on_cran()
  coupled_occ_register()
  n_cells <- 150L; n_visits <- 5L
  b_occ <- 0.5; b_det <- -0.4
  set.seed(91)
  w <- as.numeric(scale(cumsum(rnorm(n_cells, 0, 0.5)))) * 0.7
  w <- w - mean(w)
  d <- coupled_occ_data(seed = 92, n_cells = n_cells, n_visits = n_visits,
                        b_occ = b_occ, b_det = b_det, field = w)
  adj <- .occ_chain_adj(n_cells)

  fit <- .occ_spatial_fit(d, adj)
  best <- which.max(fit$log_marginal)
  md <- as.matrix(fit$modes)[best, ]

  # Both intercepts come back from a fit that ran entirely through
  # evaluate_cell. An occupancy field over 150 cells competes with the
  # occupancy intercept for the same signal, so that coordinate's posterior is
  # wide (SD around 0.25 at this size, measured across seeds) and the band is
  # set accordingly; the detection intercept is sharper.
  expect_lt(abs(md[1L] - b_occ), 0.6)
  expect_lt(abs(md[2L] - b_det), 0.35)
  expect_true(all(is.finite(fit$log_marginal)))
  expect_true(fit$theta_mean[[1L]] > 0)

  # The cross-arm curvature at the FITTED mode, not at an arbitrary eta: the
  # field offset comes out of the latent vector at the block's own start.
  field <- md[fit$arm_layout$phi_start + seq_len(n_cells)]
  cross_max <- vapply(seq_len(n_cells), function(cc) {
    rows <- ((cc - 1L) * n_visits + 1L):(cc * n_visits)
    r <- .occ_eval(md[1L] + field[cc], rep(md[2L], n_visits), d$y_det[rows])
    max(abs(r$cross_hess[[1]][[2]]))
  }, numeric(1))
  # Cells with no detection are the coupled ones, and they are most of the
  # grid here: 70 of 150, carrying a largest cross entry of 8.9e-2.
  expect_gt(sum(d$n_seen == 0L), 50L)
  expect_gt(max(cross_max), 1e-2)
  expect_equal(sum(cross_max > 1e-6), sum(d$n_seen == 0L))
  # And a detected cell contributes exactly zero cross curvature, so the
  # nonzero entries above are the mixture and not a scatter artefact.
  expect_equal(max(cross_max[d$n_seen > 0L]), 0)
})

test_that("the coupled occupancy fit agrees between the dense and sparse scatters", {
  skip_on_cran()
  coupled_occ_register()
  n_cells <- 60L; n_visits <- 4L
  set.seed(93)
  w <- as.numeric(scale(cumsum(rnorm(n_cells, 0, 0.5)))) * 0.6
  w <- w - mean(w)
  d <- coupled_occ_data(seed = 94, n_cells = n_cells, n_visits = n_visits,
                        b_occ = 0.2, b_det = -0.3, field = w)
  adj <- .occ_chain_adj(n_cells)

  dense  <- .occ_spatial_fit(d, adj, force_sparse = FALSE)
  sparse <- .occ_spatial_fit(d, adj, force_sparse = TRUE)

  expect_equal(sparse$log_marginal, dense$log_marginal, tolerance = 1e-9)
  expect_equal(sparse$modes,        dense$modes,        tolerance = 1e-8)
  expect_equal(sparse$theta_mean,   dense$theta_mean,   tolerance = 1e-8)
  expect_equal(sparse$theta_sd,     dense$theta_sd,     tolerance = 1e-8)
})
