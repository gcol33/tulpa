# Each NNGP kernel agrees with its own twin (gcol33/tulpa#142 A3).
#
# Every NNGP conditional log-likelihood exists twice: a hand-written `double`
# version and a templated autodiff one. They are the same function, so the AD
# copy instantiated at T = double must return the hand-written copy's value.
# Where it does not, the value path and the gradient path describe different
# models -- the analytic GP gradients are finite-differenced from the double
# copy.
#
# They did not agree. The GP AD copy added its jitter only to an
# already-degenerate pivot; the GP double copy added it to every diagonal. On
# well-conditioned input the AD copy therefore added none at all.
# hmc_gp_autodiff.h carried a "known heisenbug with autodiff" note. Both twins
# now share one kernel and one set of constants.

nngp_inputs <- function(N = 25L, nn = 5L, seed = 3L) {
  set.seed(seed)
  coords <- cbind(runif(N), runif(N))
  ni <- compute_nngp_neighbors(coords, nn)
  order0 <- if (!is.null(ni$nn_order)) as.integer(ni$nn_order - 1L) else seq_len(N) - 1L
  inv <- integer(N)
  inv[order0 + 1L] <- seq_len(N) - 1L
  # nn_neighbor_dist is [N x nn x nn]; the C++ readers index it row-major.
  nbd <- if (!is.null(ni$nn_neighbor_dist)) {
    as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1)))
  } else {
    numeric(N * nn * nn)
  }
  list(coords = coords, nn_idx = ni$nn_idx, nn_dist = ni$nn_dist,
       nbd = nbd, order0 = order0, inv = inv, N = N, nn = nn)
}

test_that("the GP NNGP autodiff kernel agrees with its double twin", {
  d <- nngp_inputs()
  for (cov_type in 0:1) {            # exponential, matern 3/2
    for (sigma2 in c(0.4, 1.0, 2.5)) {
      for (phi in c(0.15, 0.5, 1.2)) {
        set.seed(11)
        w <- rnorm(d$N)
        got <- cpp_test_gp_nngp_twins(w, sigma2, phi, d$coords, d$nn_idx,
                                      d$nn_dist, d$nbd, d$order0, d$inv,
                                      cov_type)
        expect_equal(unname(got[["ad"]]), unname(got[["dbl"]]),
                     tolerance = 1e-9,
                     info = paste("cov:", cov_type, "sigma2:", sigma2,
                                  "phi:", phi))
      }
    }
  }
})

test_that("the SVC NNGP autodiff kernel agrees with its double twin", {
  # #109 consolidated the double SVC kernel to jitter/var_floor = 1e-6 via a
  # double-only core the autodiff twin could not use, so the twin kept 1e-4 and
  # the two described different models. Both now read tulpa_svc::kSvcJitter /
  # kSvcVarFloor.
  d <- nngp_inputs(seed = 5L)
  for (cov_type in 0:1) {
    for (sigma2 in c(0.4, 1.0, 2.5)) {
      for (phi in c(0.15, 0.5, 1.2)) {
        set.seed(12)
        w <- rnorm(d$N)
        got <- cpp_test_svc_nngp_twins(w, sigma2, phi, d$coords, d$nn_idx,
                                       d$nn_dist, d$order0, cov_type)
        expect_equal(unname(got[["ad"]]), unname(got[["dbl"]]),
                     tolerance = 1e-9,
                     info = paste("cov:", cov_type, "sigma2:", sigma2,
                                  "phi:", phi))
      }
    }
  }
})

test_that("the twins agree on a near-degenerate configuration", {
  # Duplicate coordinates drive the neighbour covariance toward singular, which
  # is where the jitter and the variance floor actually bite -- and where the
  # two copies' policies differed most.
  set.seed(9)
  N <- 12L; nn <- 4L
  coords <- cbind(rep(c(0.2, 0.5, 0.8), each = 4), rep(c(0.2, 0.5, 0.8), each = 4))
  coords <- coords + 1e-9 * matrix(rnorm(N * 2), N, 2)   # near-duplicates
  ni <- compute_nngp_neighbors(coords, nn)
  order0 <- if (!is.null(ni$nn_order)) as.integer(ni$nn_order - 1L) else seq_len(N) - 1L
  inv <- integer(N); inv[order0 + 1L] <- seq_len(N) - 1L
  nbd <- as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1)))
  w <- rnorm(N)

  gp <- cpp_test_gp_nngp_twins(w, 1.0, 0.5, coords, ni$nn_idx, ni$nn_dist,
                               nbd, order0, inv, 0L)
  expect_equal(unname(gp[["ad"]]), unname(gp[["dbl"]]), tolerance = 1e-8)

  svc <- cpp_test_svc_nngp_twins(w, 1.0, 0.5, coords, ni$nn_idx, ni$nn_dist,
                                 order0, 0L)
  expect_equal(unname(svc[["ad"]]), unname(svc[["dbl"]]), tolerance = 1e-8)
})
