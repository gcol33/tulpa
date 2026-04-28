# test-gp-cg.R
# Verify that the iterative CG / PCG NNGP solvers agree with the direct
# Cholesky reference on log-likelihood and gradients for a small NNGP problem.
#
# The CG solver is wired in behind `spatial_gp(solver = "cg" | "pcg")`. The
# Cholesky path remains the default. This test pins the dispatch contract:
# at the same (w, sigma2, phi), all three solvers must produce numerically
# equivalent log-lik and full gradient vectors (within iterative tolerance).

build_nngp_neighbor_data <- function(coords, nn_k) {
  n_obs <- nrow(coords)
  ord <- order(coords[, 1], coords[, 2])
  coords_ord <- coords[ord, ]

  nn_idx <- matrix(0L, nrow = n_obs, ncol = nn_k)
  nn_dist <- matrix(0, nrow = n_obs, ncol = nn_k)

  for (i in 2:n_obs) {
    dists <- sqrt((coords_ord[1:(i - 1), 1] - coords_ord[i, 1])^2 +
                  (coords_ord[1:(i - 1), 2] - coords_ord[i, 2])^2)
    n_cand <- min(length(dists), nn_k)
    sel <- order(dists)[seq_len(n_cand)]
    nn_idx[i, seq_len(n_cand)] <- sel
    nn_dist[i, seq_len(n_cand)] <- dists[sel]
  }

  # Pairwise distances among the up-to-k neighbors of each obs
  nn_neighbor_dist <- array(0, dim = c(n_obs, nn_k, nn_k))
  for (i in 2:n_obs) {
    actual <- nn_idx[i, ] > 0
    n_act <- sum(actual)
    if (n_act < 2) next
    nb_locs <- coords_ord[nn_idx[i, seq_len(n_act)], , drop = FALSE]
    for (j1 in seq_len(n_act)) {
      for (j2 in seq_len(n_act)) {
        if (j1 == j2) next
        nn_neighbor_dist[i, j1, j2] <- sqrt(sum((nb_locs[j1, ] - nb_locs[j2, ])^2))
      }
    }
  }

  list(
    coords_ord = coords_ord,
    ord = ord,                 # original-index order (for nn_order, 0-based)
    nn_idx = nn_idx,
    nn_dist = nn_dist,
    nn_neighbor_dist = nn_neighbor_dist
  )
}

test_that("CG and Cholesky NNGP solvers agree on log-lik and gradients", {
  set.seed(7)
  n_obs <- 100
  nn_k <- 10L
  coords <- cbind(runif(n_obs), runif(n_obs))
  nb <- build_nngp_neighbor_data(coords, nn_k)

  # Spatial effect drawn from the prior implied by these hyperparameters
  sigma2 <- 0.6
  phi <- 0.25
  w <- rnorm(n_obs, sd = sqrt(sigma2))

  # nn_order: 0-based location indices in NNGP order; nn_order is just 0..N-1
  # since we already sorted coords. nn_order_inv is the inverse.
  nn_order <- seq.int(0L, n_obs - 1L)
  nn_order_inv <- nn_order

  # C++ expects row-major flatten:  index = i*nn*nn + j1*nn + j2.
  # R's array is (n_obs, nn, nn); aperm(., c(3,2,1)) + as.vector() yields
  # the equivalent C-order linearization.
  flat_neighbor_dist <- as.vector(aperm(nb$nn_neighbor_dist, c(3, 2, 1)))

  call_solver <- function(solver) {
    cpp_test_gp_solver_dispatch(
      w = w,
      sigma2 = sigma2,
      phi = phi,
      coords = nb$coords_ord,
      nn_idx = nb$nn_idx,
      nn_dist = nb$nn_dist,
      nn_neighbor_dist = flat_neighbor_dist,
      nn_order = nn_order,
      nn_order_inv = nn_order_inv,
      cov_type = 0L,         # exponential
      solver = solver,
      cg_tol = 1e-10,
      cg_maxiter = 500L
    )
  }

  ref <- call_solver("cholesky")
  cg  <- call_solver("cg")
  pcg <- call_solver("pcg")

  expect_true(is.finite(ref$log_lik))
  expect_true(is.finite(cg$log_lik))
  expect_true(is.finite(pcg$log_lik))

  # Log-likelihood: CG with tight tolerance reproduces Cholesky almost exactly
  expect_equal(cg$log_lik,  ref$log_lik, tolerance = 1e-6)
  expect_equal(pcg$log_lik, ref$log_lik, tolerance = 1e-6)

  # Spatial-effect gradient
  expect_equal(cg$grad_w,  ref$grad_w, tolerance = 1e-5)
  expect_equal(pcg$grad_w, ref$grad_w, tolerance = 1e-5)

  # Hyperparameter gradients
  expect_equal(cg$grad_log_sigma2,  ref$grad_log_sigma2, tolerance = 1e-5)
  expect_equal(pcg$grad_log_sigma2, ref$grad_log_sigma2, tolerance = 1e-5)
  expect_equal(cg$grad_log_phi,  ref$grad_log_phi, tolerance = 1e-5)
  expect_equal(pcg$grad_log_phi, ref$grad_log_phi, tolerance = 1e-5)
})

test_that("spatial_gp accepts solver argument and stores it on the spec", {
  spec_default <- spatial_gp(~ x + y)
  expect_equal(spec_default$solver, "auto")

  spec_chol <- spatial_gp(~ x + y, solver = "cholesky")
  expect_equal(spec_chol$solver, "cholesky")

  spec_cg <- spatial_gp(~ x + y, solver = "cg", cg_tol = 1e-8, cg_maxiter = 200L)
  expect_equal(spec_cg$solver, "cg")
  expect_equal(spec_cg$cg_tol, 1e-8)
  expect_equal(spec_cg$cg_maxiter, 200L)

  expect_error(spatial_gp(~ x + y, solver = "nonsense"))
})
