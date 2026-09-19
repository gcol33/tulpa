# The two SVC parameterizations are one model (gcol33/tulpa#842).
#
# Centered samples the field w directly under tulpa_svc::nngp_log_lik; the
# non-centered one samples z ~ N(0, I) and rebuilds w = f(z, sigma2, phi) with
# nngp_nc_forward. Those are the same posterior only if f is the exact inverse
# of the density's own conditional structure, and they are two separate pieces
# of code: the density factors a neighbour covariance through
# tulpa_nngp::cond_moments, the transform runs its own Eigen Cholesky.
# NNGPNCView already carries the density's conditioning constants (jitter,
# variance floor, floor mode) so the two condition identically, and its comment
# states the consequence -- "a non-centered SVC fit targets a different
# posterior than a centered one" if they drift apart -- but nothing measured it.
# #842 asked whether the centered path's field amplitude reading above the
# non-centered one's is a defect; a drift here is the one way it could have
# been, so the invariant gets a test of its own.
#
# What is asserted. Writing f for the transform and ll for the density, a
# correct pair satisfies the change of variables
#
#   ll(f(z)) + log|det J_f| = -0.5 z'z - c,    c = (N/2) log(2 pi)
#
# and a Gaussian's conditional variances do not depend on the values, so
# log|det J_f| is a constant of (sigma2, phi) alone. Two consequences are
# checkable without exporting the Jacobian's analytic form:
#
#   1. g(z) = ll(f(z)) + 0.5 z'z is CONSTANT in z. This pins the transform's
#      conditional means and variances to the density's, site by site.
#   2. g(z) + log|det J_f| is the SAME constant at every (sigma2, phi, kernel).
#      The determinant is taken numerically off f itself, so this additionally
#      pins how each side scales with the hyperparameters -- a density whose
#      log-determinant term was wrong in sigma2 would still pass (1).

svc_eq_inputs <- function(N = 30L, nn = 8L, seed = 5L) {
  set.seed(seed)
  coords <- cbind(runif(N), runif(N))
  ni <- compute_nngp_neighbors(coords, nn)
  order0 <- if (!is.null(ni$nn_order)) as.integer(ni$nn_order - 1L) else seq_len(N) - 1L
  inv <- integer(N)
  inv[order0 + 1L] <- seq_len(N) - 1L
  list(coords = coords, nn_idx = ni$nn_idx, nn_dist = ni$nn_dist,
       order0 = order0, inv = inv, N = N)
}

# w = f(z) at the given hyperparameters, straight off the shipped transform.
svc_eq_forward <- function(d, z, sigma2, phi, cov_type) {
  cpp_test_svc_nngp_nc_grad(z, log(sigma2), log(phi), rep(0, d$N),
                            d$coords, d$nn_idx, d$nn_dist, d$order0, d$inv,
                            cov_type)$w
}

# ll(f(z)) + 0.5 z'z, the quantity the change of variables makes constant.
svc_eq_g <- function(d, z, sigma2, phi, cov_type) {
  w <- svc_eq_forward(d, z, sigma2, phi, cov_type)
  ll <- cpp_test_svc_nngp_twins(w, sigma2, phi, d$coords, d$nn_idx, d$nn_dist,
                                d$order0, cov_type)[["dbl"]]
  unname(ll + 0.5 * sum(z^2))
}

test_that("the non-centered SVC transform inverts the centered density exactly", {
  d <- svc_eq_inputs()
  zs <- lapply(1:6, function(r) { set.seed(200L + r); rnorm(d$N) })

  for (cov_type in 0:1) {                 # exponential, matern 3/2
    for (sigma2 in c(0.3, 1.0, 4.0)) {
      for (phi in c(0.1, 0.4, 1.5)) {
        g <- vapply(zs, svc_eq_g, 0, d = d, sigma2 = sigma2, phi = phi,
                    cov_type = cov_type)
        lbl <- paste("cov:", cov_type, "sigma2:", sigma2, "phi:", phi)
        # A scale to judge the spread against: the density itself is O(N).
        expect_lt(max(g) - min(g), 1e-8 * max(1, max(abs(g))))
        expect_equal(g, rep(mean(g), length(g)), tolerance = 1e-9, info = lbl)
      }
    }
  }
})

test_that("the SVC transform's Jacobian closes the change of variables", {
  d <- svc_eq_inputs()
  set.seed(77)
  z <- rnorm(d$N)
  eps <- 1e-6

  # log|det J_f| by central differences on the shipped transform.
  logdet_fd <- function(sigma2, phi, cov_type) {
    J <- matrix(0, d$N, d$N)
    for (k in seq_len(d$N)) {
      zp <- z; zp[k] <- zp[k] + eps
      zm <- z; zm[k] <- zm[k] - eps
      J[, k] <- (svc_eq_forward(d, zp, sigma2, phi, cov_type) -
                 svc_eq_forward(d, zm, sigma2, phi, cov_type)) / (2 * eps)
    }
    determinant(J, logarithm = TRUE)$modulus[[1]]
  }

  grid <- expand.grid(sigma2 = c(0.3, 1.0, 4.0), phi = c(0.1, 0.4, 1.5),
                      cov_type = 0:1)
  const <- vapply(seq_len(nrow(grid)), function(i) {
    p <- grid[i, ]
    svc_eq_g(d, z, p$sigma2, p$phi, p$cov_type) +
      logdet_fd(p$sigma2, p$phi, p$cov_type)
  }, 0)

  # -(N/2) log(2 pi) if the density carries the Gaussian normalizer, 0 if it
  # drops it -- either way ONE number for every kernel and hyperparameter pair.
  expect_equal(const, rep(mean(const), length(const)), tolerance = 1e-6)
  expect_true(isTRUE(all.equal(mean(const), 0, tolerance = 1e-6)) ||
              isTRUE(all.equal(mean(const), -0.5 * d$N * log(2 * pi),
                               tolerance = 1e-6)))
})
