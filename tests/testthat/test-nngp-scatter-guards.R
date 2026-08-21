# batch_nngp_scatter's entry contract (gcol33/tulpa#425).
#
# nn_order, nn_idx, nn_dist and coords all arrive from R and are used as raw
# array offsets. Phase 1 used to take `obs_idx = nn_order[i]` unchecked and then
# write cond_mean_out[obs_idx] / cond_var_out[obs_idx] -- an out-of-range entry
# is a silent out-of-bounds write on a std::vector, not an error. It also counted
# every positive nn_idx entry in a row and then read the LEADING columns, so a
# zero sentinel sitting before a positive entry made it read nn_order[-1] and
# hand the result to coords_dist as a location index. Phase 3 of the same
# function already checked both.

.guard_fixture <- function(ng = 12L, nn = 3L, seed = 5L) {
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
       nn_order = as.integer(seq_len(ng) - 1L), ng = ng, nn = nn,
       w = rnorm(ng, 0, 0.5))
}

.guard_call <- function(fx, ...) {
  args <- list(w = fx$w, coords = fx$coords, nn_idx = fx$nn_idx,
               nn_dist = fx$nn_dist, nn_order = fx$nn_order,
               n_spatial = fx$ng, nn = fx$nn,
               sigma2 = 0.9, phi_gp = 0.4, cov_type = 0L)
  args[names(list(...))] <- list(...)
  do.call(tulpa:::cpp_test_nngp_prior_scatter, args)
}

test_that("a malformed nn_order is refused rather than written through", {
  fx <- .guard_fixture()
  expect_error(.guard_call(fx, nn_order = fx$nn_order[-1L]),
               "length[(]nn_order[)]")
  bad <- fx$nn_order; bad[4L] <- fx$ng
  expect_error(.guard_call(fx, nn_order = bad), "nn_order[[]4[]]")
  bad[4L] <- -1L
  expect_error(.guard_call(fx, nn_order = bad), "nn_order[[]4[]]")
})

test_that("the neighbour tables and coordinates are checked against n_spatial", {
  fx <- .guard_fixture()
  expect_error(.guard_call(fx, nn_idx = fx$nn_idx[-1L, , drop = FALSE]),
               "nn_idx is")
  expect_error(.guard_call(fx, nn_dist = fx$nn_dist[, -1L, drop = FALSE]),
               "nn_dist is")
  expect_error(.guard_call(fx, coords = fx$coords[-1L, , drop = FALSE]),
               "nrow[(]coords[)]")
  expect_error(.guard_call(fx, sigma2 = 0), "sigma2")
  expect_error(.guard_call(fx, phi_gp = -1), "phi_gp")
})

test_that("an out-of-range nn_idx entry ends the run instead of being indexed", {
  # `nn_idx` is padded, so an entry outside [1, n_spatial] terminates a row's
  # neighbour run exactly as a zero does -- what matters is that it is never
  # used as an offset into nn_order. Both readings agree with the row cleared
  # from that column on.
  fx <- .guard_fixture()
  cleared <- fx$nn_idx; cleared[6L, ] <- 0L
  for (bad_val in c(fx$ng + 1L, -3L)) {
    bad <- fx$nn_idx; bad[6L, 1L] <- bad_val
    r_bad <- .guard_call(fx, nn_idx = bad)
    r_clr <- .guard_call(fx, nn_idx = cleared)
    expect_equal(r_bad$cv, r_clr$cv, info = as.character(bad_val))
    expect_equal(r_bad$alpha, r_clr$alpha, info = as.character(bad_val))
    expect_equal(r_bad$dropped, 0, info = as.character(bad_val))
  }
})

test_that("a row's neighbours are the leading run, not every positive entry", {
  # Row 6 keeps a positive entry in column 2 with column 1 blanked. Counting
  # positives would make n_nb = 1 or 2 and then read column 1, whose value is 0,
  # so nn_order[-1] would index coords. The left-packed scan stops at the blank,
  # which is what phase 3 and apply_nngp_full_prior_sparse already did.
  fx <- .guard_fixture()
  interior <- fx$nn_idx; interior[6L, 1L] <- 0L
  cleared  <- fx$nn_idx; cleared[6L, ]  <- 0L

  r_int <- .guard_call(fx, nn_idx = interior)
  r_clr <- .guard_call(fx, nn_idx = cleared)
  expect_equal(r_int$H, r_clr$H)
  expect_equal(r_int$cv, r_clr$cv)
  expect_equal(r_int$alpha, r_clr$alpha)
  # Location 6 conditions on nothing, so its conditional variance is the
  # marginal sigma2 and it carries no regression weights.
  expect_equal(r_int$cv[6L], 0.9)
  expect_true(all(r_int$alpha[6L, ] == 0))
  # Nothing was written outside the declared pattern on either run.
  expect_equal(r_int$dropped, 0)
})

test_that("a well-formed call is unaffected by the guards", {
  fx <- .guard_fixture()
  r <- .guard_call(fx)
  expect_length(r$cv, fx$ng)
  expect_true(all(is.finite(r$cv)) && all(r$cv > 0))
  expect_equal(dim(r$alpha), c(fx$ng, fx$nn))
  expect_equal(r$dropped, 0)
})

test_that("a one-column coordinate matrix is still a valid NNGP domain", {
  # The guards are about the neighbour tables; the coordinate arity is
  # gcol33/tulpa#389's rule, and coords_dist sums over whatever columns arrive.
  fx <- .guard_fixture()
  fx1 <- fx
  fx1$coords <- fx$coords[, 1L, drop = FALSE]
  d1 <- as.matrix(dist(fx1$coords))
  for (i in seq_len(fx$ng)) {
    for (k in seq_len(fx$nn)) {
      j <- fx$nn_idx[i, k]
      fx1$nn_dist[i, k] <- if (j > 0L) d1[i, j] else 0
    }
  }
  r1 <- .guard_call(fx1)
  # A constant extra column cancels in every squared coordinate difference.
  fx2 <- fx1
  fx2$coords <- cbind(fx1$coords, 1e6)
  r2 <- .guard_call(fx2)
  expect_equal(r1$cv, r2$cv)
  expect_equal(r1$alpha, r2$alpha)
})
