# Non-centered NNGP transform: hand-derived backward vs finite differences
# (gcol33/tulpa#243).
#
# The funnel fix samples z ~ N(0, I) and reconstructs w = f(z, sigma2, phi),
# wiring the field's likelihood gradient back to (z, log_sigma2, log_phi)
# through nngp_nc_backward inside an arena custom_backward. Those analytic
# hyperparameter gradients had never been validated -- the runtime gradient
# check only fires at the origin (z = 0), where w = 0 for any (sigma2, phi) and
# the log_sigma2 / log_phi likelihood gradients are trivially zero. This checks
# them at a non-trivial z against central differences of the forward, for a
# scalar loss L = sum(a_i w_i) (so dL/dw = a is exactly the backward's input).

nngp_grad_inputs <- function(N = 25L, nn = 8L, seed = 3L) {
  set.seed(seed)
  coords <- cbind(runif(N), runif(N))
  ni <- compute_nngp_neighbors(coords, nn)
  order0 <- as.integer(ni$nn_order - 1L)
  inv <- integer(N); inv[order0 + 1L] <- seq_len(N) - 1L
  nbd <- as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1)))
  list(coords = coords, nn_idx = ni$nn_idx, nn_dist = ni$nn_dist,
       nbd = nbd, order0 = order0, inv = inv, N = N, nn = nn)
}

test_that("nngp_nc_backward matches finite differences (z, log_sigma2, log_phi)", {
  d <- nngp_grad_inputs()
  for (cov_type in 0:1) {                       # exponential, matern 3/2
    for (ls2 in c(log(0.4), log(1.44))) {
      for (lphi in c(log(0.2), log(0.6))) {
        set.seed(7)
        z <- rnorm(d$N)
        a <- rnorm(d$N)                          # dL/dw
        res <- cpp_test_nngp_nc_grad(
          z = z, log_sigma2 = ls2, log_phi = lphi, a = a,
          coords = d$coords, nn_idx = d$nn_idx, nn_dist = d$nn_dist,
          nn_neighbor_dist = d$nbd, nn_order = d$order0, nn_order_inv = d$inv,
          cov_type = cov_type, fd_eps = 1e-6)
        info <- paste("cov:", cov_type, "ls2:", round(ls2, 2),
                      "lphi:", round(lphi, 2))
        expect_equal(res$grad_z, res$grad_z_fd, tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_sigma2, res$grad_log_sigma2_fd,
                     tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_phi, res$grad_log_phi_fd,
                     tolerance = 1e-7, info = info)
      }
    }
  }
})

# SVC shares the transform's engine (hmc_gp_nc.h's nngp_nc_forward/backward,
# generalized to an NNGPNCView) but takes the coords-fallback branch of
# pair_dist() instead of the cached nn_neighbor_dist one GP/multiscale-GP use
# -- SVCData does not cache neighbour-pair distances (see hmc_svc.h). This is
# the other branch cpp_test_nngp_nc_grad above never exercises.
test_that("SVC nngp_nc_backward matches finite differences (coords-fallback pair_dist)", {
  d <- nngp_grad_inputs()
  for (cov_type in 0:1) {
    for (ls2 in c(log(0.4), log(1.44))) {
      for (lphi in c(log(0.2), log(0.6))) {
        set.seed(11)
        z <- rnorm(d$N)
        a <- rnorm(d$N)
        res <- cpp_test_svc_nngp_nc_grad(
          z = z, log_sigma2 = ls2, log_phi = lphi, a = a,
          coords = d$coords, nn_idx = d$nn_idx, nn_dist = d$nn_dist,
          nn_order = d$order0, nn_order_inv = d$inv,
          cov_type = cov_type, fd_eps = 1e-6)
        info <- paste("cov:", cov_type, "ls2:", round(ls2, 2),
                      "lphi:", round(lphi, 2))
        expect_equal(res$grad_z, res$grad_z_fd, tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_sigma2, res$grad_log_sigma2_fd,
                     tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_phi, res$grad_log_phi_fd,
                     tolerance = 1e-7, info = info)
      }
    }
  }
})

# Multiscale GP: each scale is its own NNGPNCView built from
# MultiscaleGPData's per-scale fields (make_msgp_nc_view_local/_regional).
# Both scales share the cached nn_neighbor_dist fast path GP uses -- this
# checks the field construction / index plumbing per scale, not a new branch
# of pair_dist().
test_that("multiscale-GP nngp_nc_backward matches finite differences (both scales)", {
  d <- nngp_grad_inputs()
  for (scale in c("local", "regional")) {
    for (ls2 in c(log(0.4), log(1.44))) {
      for (lphi in c(log(0.2), log(0.6))) {
        set.seed(13)
        z <- rnorm(d$N)
        a <- rnorm(d$N)
        res <- cpp_test_msgp_nngp_nc_grad(
          z = z, log_sigma2 = ls2, log_phi = lphi, a = a,
          coords = d$coords, nn_idx = d$nn_idx, nn_dist = d$nn_dist,
          nn_neighbor_dist = d$nbd, nn_order = d$order0, nn_order_inv = d$inv,
          cov_type = 0L, scale = scale, fd_eps = 1e-6)
        info <- paste("scale:", scale, "ls2:", round(ls2, 2),
                      "lphi:", round(lphi, 2))
        expect_equal(res$grad_z, res$grad_z_fd, tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_sigma2, res$grad_log_sigma2_fd,
                     tolerance = 1e-7, info = info)
        expect_equal(res$grad_log_phi, res$grad_log_phi_fd,
                     tolerance = 1e-7, info = info)
      }
    }
  }
})

# The nugget channel in the sigma2 gradient (gcol33/tulpa#584).
#
# The forward builds C = sigma2 R + jitter I against c = sigma2 r, so neither
# the conditional variance nor the regression weights are homogeneous in
# sigma2 the way they are at jitter = 0:
#
#   dd/d log sigma2     = d_raw - jitter * ||alpha||^2
#   dalpha/d log sigma2 = jitter * C^{-1} alpha
#
# Both terms are proportional to the nugget, which is 1e-8 on the GP view and
# 1e-4 on the SVC one, so dropping them was invisible on the first and reached
# 1e-2 relative on the second. The tolerances above are tight enough that
# either term going missing fails: without them the SVC sweep runs at 8e-5 to
# 1e-2 relative and the GP sweep at 6e-9 to 5e-7.

test_that("the SVC finite-difference reference is converged, so a disagreement is real", {
  d <- nngp_grad_inputs()
  # The configuration the largest #584 disagreement was measured at.
  ls2 <- log(0.4); lphi <- log(0.6)
  set.seed(11)
  z <- rnorm(d$N); a <- rnorm(d$N)
  run <- function(eps) cpp_test_svc_nngp_nc_grad(
    z = z, log_sigma2 = ls2, log_phi = lphi, a = a,
    coords = d$coords, nn_idx = d$nn_idx, nn_dist = d$nn_dist,
    nn_order = d$order0, nn_order_inv = d$inv, cov_type = 1L, fd_eps = eps)
  fd <- vapply(c(1e-4, 1e-5, 1e-6, 1e-7),
               function(e) run(e)$grad_log_sigma2_fd, numeric(1))
  # Four orders of magnitude of step, same value: the central difference is in
  # its converged plateau, so the gradient is what has to match it.
  expect_lt(max(abs(fd - fd[1])) / abs(fd[1]), 1e-6)
  expect_equal(run(1e-6)$grad_log_sigma2, fd[1], tolerance = 1e-7)
})

test_that("the sigma2 gradient tracks the nugget across coordinate arities", {
  # gcol33/tulpa#389: coords_dist sums over every column the matrix carries, so
  # the coords-fallback pair_dist this path takes must agree with a 2-D fixture
  # whose second column is constant. Distances are unchanged, so every gradient
  # is.
  d <- nngp_grad_inputs()
  flat <- cbind(d$coords[, 1], 0)
  ni <- compute_nngp_neighbors(flat, d$nn)
  order0 <- as.integer(ni$nn_order - 1L)
  inv <- integer(d$N); inv[order0 + 1L] <- seq_len(d$N) - 1L
  set.seed(11)
  z <- rnorm(d$N); a <- rnorm(d$N)
  res <- cpp_test_svc_nngp_nc_grad(
    z = z, log_sigma2 = log(0.4), log_phi = log(0.6), a = a,
    coords = flat, nn_idx = ni$nn_idx, nn_dist = ni$nn_dist,
    nn_order = order0, nn_order_inv = inv, cov_type = 1L, fd_eps = 1e-6)
  expect_equal(res$grad_log_sigma2, res$grad_log_sigma2_fd, tolerance = 1e-7)
  expect_equal(res$grad_log_phi, res$grad_log_phi_fd, tolerance = 1e-7)
  expect_equal(res$grad_z, res$grad_z_fd, tolerance = 1e-7)
})
