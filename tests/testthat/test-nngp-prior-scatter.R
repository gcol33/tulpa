# The NNGP prior scatter against the matrix it claims to build
# (gcol33/tulpa#278).
#
# apply_nngp_full_prior_sparse assembles Lambda = (I - A)' D^-1 (I - A) from the
# (alpha, cv) batch_nngp_scatter returns. Nothing downstream re-derives Lambda,
# so the claim went unchecked; #278 opened on a measured log|H| gap between the
# then-live dense twin and this one, read as one of them being wrong. Assembling
# Lambda independently here settles it: both matched it, and the gap was an
# ABSOLUTE difference on entries of magnitude 1e13. The dense twin is gone (it
# had no callers once blocks_require_sparse() pinned NNGP to the sparse Newton),
# so this file is what holds the survivor to the definition.
#
# Where the 1e13 comes from is asserted at the bottom: nngp_moments_from_chol
# floors cond_var at 1e-10, and a floored node puts 1e10 straight on Lambda's
# diagonal. That conditioning -- not a defective scatter -- is what stalls a
# Newton solve at large nn. gcol33/tulpa#283 tracks it.

# A Vecchia ordering with nearest-predecessor neighbour lists: site i conditions
# on its `nn` closest predecessors in the natural order (nn_order 0-based,
# nn_idx 1-based into that order).
.nngp_scatter_fixture <- function(ng, nn, seed) {
  set.seed(seed)
  coords <- cbind(runif(ng), runif(ng))
  d <- as.matrix(dist(coords))
  nn_idx <- matrix(0L, ng, nn)
  nn_dist <- matrix(0, ng, nn)
  for (i in seq_len(ng)) {
    prev <- seq_len(i - 1L)
    if (!length(prev)) next
    k <- min(nn, length(prev))
    o <- order(d[i, prev])[seq_len(k)]
    nn_idx[i, seq_len(k)] <- as.integer(prev[o])
    nn_dist[i, seq_len(k)] <- d[i, prev[o]]
  }
  list(coords = coords, nn_idx = nn_idx, nn_dist = nn_dist,
       nn_order = as.integer(seq_len(ng) - 1L), ng = ng, nn = nn)
}

# Lambda = (I - A)' D^-1 (I - A), built from the returned coefficients with no
# reference to the scatter's own loops. Row i of A carries alpha[i_nngp, k] at
# the column of neighbour k; A and D are in OBS index space, matching how the
# scatter indexes grad and H.
.nngp_lambda <- function(fx, alpha, cv) {
  A <- matrix(0, fx$ng, fx$ng)
  for (i_nngp in seq_len(fx$ng)) {
    obs_i <- fx$nn_order[i_nngp] + 1L
    for (k in seq_len(fx$nn)) {
      nnidx <- fx$nn_idx[i_nngp, k]
      if (nnidx <= 0L || nnidx > fx$ng) next
      A[obs_i, fx$nn_order[nnidx] + 1L] <- alpha[i_nngp, k]
    }
  }
  ImA <- diag(fx$ng) - A
  t(ImA) %*% diag(1 / cv) %*% ImA
}

.nngp_scatter <- function(fx, w, sigma2 = 0.9, phi_gp = 0.4, cov_type = 0L) {
  tulpa:::cpp_test_nngp_prior_scatter(
    w = w, coords = fx$coords, nn_idx = fx$nn_idx, nn_dist = fx$nn_dist,
    nn_order = fx$nn_order, n_spatial = fx$ng, nn = fx$nn,
    sigma2 = sigma2, phi_gp = phi_gp, cov_type = cov_type)
}

test_that("the NNGP prior scatter reproduces (I-A)' D^-1 (I-A)", {
  skip_on_cran()
  for (ng in c(60L, 150L)) {
    for (nn in c(2L, 3L, 5L, 8L, 10L)) {
      fx <- .nngp_scatter_fixture(ng, nn, 11L + ng)
      set.seed(99)
      r <- .nngp_scatter(fx, rnorm(ng, 0, 0.5))
      L <- .nngp_lambda(fx, r$alpha, r$cv)
      # RELATIVE, deliberately. Lambda's entries span 1e2 to 1e13 across this
      # sweep, so an absolute bound would either pass vacuously at small nn or
      # fail on machine epsilon at large nn -- which is exactly how #278 came
      # to read a correct scatter as a broken one.
      expect_lt(max(abs(r$H - L)) / max(abs(L)), 1e-13)
      # Nothing the scatter writes falls outside the declared pattern.
      expect_equal(r$dropped, 0)
    }
  }
})

test_that("the NNGP prior gradient is -Lambda w", {
  skip_on_cran()
  # Held at neighbour-set sizes where no conditional variance hits its floor,
  # so Lambda stays around 1e2 and the dense matrix-vector product is a usable
  # reference. Past the floor, `Lambda %*% w` loses more digits to cancellation
  # than the scatter's own accumulation does, and the reference stops being one.
  for (nn in c(2L, 3L)) {
    fx <- .nngp_scatter_fixture(60L, nn, 71L)
    set.seed(99)
    w <- rnorm(60L, 0, 0.5)
    r <- .nngp_scatter(fx, w)
    expect_true(all(r$cv > 1e-10 * (1 + 1e-12)))
    g <- as.numeric(-.nngp_lambda(fx, r$alpha, r$cv) %*% w)
    expect_lt(max(abs(r$grad - g)) / max(abs(g)), 1e-10)
  }
})

test_that("the NNGP conditional-variance floor is what inflates Lambda", {
  skip_on_cran()
  # The floor is silent, so this records its reach rather than gating on a
  # threshold nobody chose: at nn = 2 almost nothing is floored and Lambda is
  # O(1e1); by nn = 8 roughly a third of the field is, and Lambda is O(1e13).
  fx2 <- .nngp_scatter_fixture(150L, 2L, 161L)
  fx8 <- .nngp_scatter_fixture(150L, 8L, 161L)
  set.seed(99); w <- rnorm(150L, 0, 0.5)
  r2 <- .nngp_scatter(fx2, w)
  r8 <- .nngp_scatter(fx8, w)
  floored <- function(r) sum(r$cv <= 1e-10 * (1 + 1e-12))

  expect_lt(floored(r2), 5L)
  expect_gt(floored(r8), 30L)
  expect_gt(max(abs(r8$H)) / max(abs(r2$H)), 1e3)
  # And the scatter still reproduces Lambda there -- the magnitude is the
  # prior's, not an error in building it.
  L8 <- .nngp_lambda(fx8, r8$alpha, r8$cv)
  expect_lt(max(abs(r8$H - L8)) / max(abs(L8)), 1e-13)
})
