test_that("a 1-D coordinate matrix is 1-D, not two copies of one column", {
  # gcol33/tulpa#389, the R half. Four sampler specs and two prediction paths
  # reshaped their coordinates with `matrix(as.numeric(x), n, 2)`, which does
  # not CHECK the arity, it IMPOSES it: an `n x 1` matrix is recycled so that
  # column 2 equals column 1 -- every location on the diagonal, every distance
  # scaled by sqrt(2) -- and an `n x 3` matrix is truncated to its first two
  # columns. Both are a different geometry accepted in silence.
  x1 <- matrix(c(0, 1, 2, 5), ncol = 1L)
  # The recycling is what the old expression did, and it is not the identity.
  recycled <- matrix(as.numeric(x1), nrow(x1), 2L)
  expect_identical(recycled[, 1L], recycled[, 2L])
  expect_false(identical(ncol(recycled), ncol(x1)))

  # `.coords_plain()` strips attributes and keeps the shape, which is what the
  # dimension-general paths want.
  sc <- scale(x1)
  p  <- .coords_plain(sc)
  expect_identical(dim(p), dim(x1))
  expect_null(attributes(p)$`scaled:center`)
  expect_equal(as.numeric(p), as.numeric(sc), tolerance = 0)
  expect_identical(dim(.coords_plain(matrix(0, 4L, 3L))), c(4L, 3L))

  # `.coords_2col()` refuses rather than reshapes, on BOTH sides of 2.
  expect_error(.coords_2col(x1, "gp()"), "exactly 2 columns")
  expect_error(.coords_2col(x1, "gp()"), "got 1")
  expect_error(.coords_2col(matrix(0, 4L, 3L), "svc()"), "got 3")
  expect_error(.coords_2col(x1, "hsgp() under a sampler mode"), "hsgp\\(\\)")
  ok <- .coords_2col(scale(matrix(c(0, 1, 2, 5, 1, 0, 3, 2), ncol = 2L)), "gp()")
  expect_identical(dim(ok), c(4L, 2L))
  expect_null(dimnames(ok))
})

test_that("the sampler-spec kernels refuse a coordinate matrix they cannot store", {
  # The C++ half of the same guard. `GPData::coords` and its siblings are flat
  # buffers at stride 2, so those fills read column 1 unconditionally -- the
  # out-of-bounds read of gcol33/tulpa#389. They are not made dimension-general
  # (the layout is shared with the samplers and the ABI); they refuse the input
  # instead of misreading it. This is the back door the R guard above does not
  # cover, reachable with a hand-built spec.
  skip_on_cran()
  n <- 6L
  co1 <- matrix(seq(0, 1, length.out = n), ncol = 1L)
  ni  <- matrix(0L, n, 2L); nd <- matrix(0, n, 2L)
  for (i in 2:n) {
    k <- utils::head(order(abs(co1[seq_len(i - 1L)] - co1[i])), 2L)
    ni[i, seq_along(k)] <- k
    nd[i, seq_along(k)] <- abs(co1[k] - co1[i])
  }
  nnd3 <- as.numeric(array(0, c(n, 2L, 2L)))
  ord  <- seq_len(n) - 1L
  expect_error(
    cpp_test_gp_nngp_twins(rep(0, n), 1, 1, co1, ni, nd, nnd3, ord, ord, 0L),
    "exactly 2 columns")
  # A 2-column matrix of the same points is accepted, so the guard is arity and
  # not something else about the fixture.
  expect_silent(
    cpp_test_gp_nngp_twins(rep(0, n), 1, 1, cbind(co1, 0), ni, nd, nnd3,
                           ord, ord, 0L))
})

test_that("an NNGP fit is a function of its data, at any coordinate dimension", {
  # gcol33/tulpa#389. Three NNGP neighbour-covariance loops formed the
  # neighbour-to-neighbour distance by hand over coordinate columns 0 and 1, and
  # read column 1 UNCONDITIONALLY. On an `n x 1` coordinate matrix that offset is
  # `1 * nrow + i`, i.e. n doubles past the end of the allocation, so the
  # neighbour covariance was built from whatever the R heap held behind the
  # matrix. Nothing crashed -- the read lands inside the heap and returns a
  # finite double -- so the fit simply stopped being a function of its data and
  # moved with the process's allocation history. Measured before the fix: the
  # same seeds fitted twice in ONE process at n_threads = 1 differed on 10/20
  # fits at n = 49 with log_marginal moving 4.09, and WHICH sizes broke changed
  # between sessions, which is why the issue's own size table did not reproduce.
  #
  # The sizes straddle the batched-Cholesky dispatch at 50 locations, because
  # that was the first suspect and it was not the cause: n = 49 is on the CPU
  # path and broke, n = 60 is on the GPU path and was clean in the issue's run.
  skip_on_cran()
  fit_lm <- function(co, seed = 11L) {
    n <- nrow(co)
    d1 <- as.matrix(dist(co))
    nn <- 4L
    ni <- matrix(0L, n, nn); nd <- matrix(0, n, nn)
    for (i in seq_len(n)) {
      prev <- seq_len(i - 1L)
      k <- if (length(prev)) utils::head(prev[order(d1[i, prev])], nn) else integer(0)
      if (length(k)) { ni[i, seq_along(k)] <- k; nd[i, seq_along(k)] <- d1[i, k] }
    }
    set.seed(seed)
    idx <- rep(seq_len(n), each = 4L)
    X   <- cbind(1, rnorm(length(idx)))
    y   <- as.numeric(X %*% c(-0.2, 0.7)) + rnorm(length(idx), 0, 0.5)
    f <- suppressWarnings(tulpa_nested_laplace(
      y = y, n_trials = rep(1L, length(idx)), X = X,
      prior = list(type = "nngp", coords = co, nn_idx = ni, nn_dist = nd,
                   n_spatial = n, nn = nn, spatial_idx = idx, cov_type = 0L),
      family = "gaussian", phi = 0.5,
      control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                     progress = FALSE, diagnose_k = FALSE,
                     diagnose_skew = FALSE)))
    f$log_marginal
  }

  for (n in c(30L, 49L, 60L)) {
    co1 <- matrix(seq(0, 3, length.out = n), ncol = 1L)
    a <- fit_lm(co1)
    b <- fit_lm(co1)
    # A deterministic single-threaded computation on identical inputs. Any
    # difference at all is the defect, so this is exact rather than toleranced.
    expect_identical(a, b, info = paste("n =", n))
    expect_true(all(is.finite(a)))

    # THE ARBITER that the out-of-bounds column was what was being read: a
    # CONSTANT second column cancels in `(coords(o1,1) - coords(o2,1))^2`, so it
    # is the same 1-D geometry with the read made in-bounds. Bit-for-bit, and at
    # two different constants, so this cannot pass by the column being ignored.
    expect_identical(fit_lm(cbind(co1, 0)), a, info = paste("n =", n))
    expect_identical(fit_lm(cbind(co1, 1e6)), a, info = paste("n =", n))

    # And 1-D is genuinely 1-D: the silent reshape put every location on the
    # diagonal, which scales every distance by sqrt(2) and is a different model.
    # If this ever matches, the arity is being imposed again somewhere.
    expect_false(identical(fit_lm(cbind(co1, co1)), a))
  }
})

test_that("the nested-Laplace NNGP kernel reads three coordinate columns", {
  # The coordinate DIMENSION is whatever the caller supplied. A 3-column fit is
  # not merely accepted, it has to be the fit that geometry implies: the same
  # points embedded with a constant third column must reproduce the 2-D answer
  # exactly, and a non-constant third column must not.
  skip_on_cran()
  n  <- 24L
  set.seed(4L)
  co2 <- cbind(runif(n), runif(n))
  fit_lm <- function(co) {
    d1 <- as.matrix(dist(co)); nn <- 4L
    ni <- matrix(0L, n, nn); nd <- matrix(0, n, nn)
    for (i in seq_len(n)) {
      prev <- seq_len(i - 1L)
      k <- if (length(prev)) utils::head(prev[order(d1[i, prev])], nn) else integer(0)
      if (length(k)) { ni[i, seq_along(k)] <- k; nd[i, seq_along(k)] <- d1[i, k] }
    }
    set.seed(5L)
    idx <- rep(seq_len(n), each = 4L)
    X   <- cbind(1, rnorm(length(idx)))
    y   <- as.numeric(X %*% c(-0.2, 0.7)) + rnorm(length(idx), 0, 0.5)
    suppressWarnings(tulpa_nested_laplace(
      y = y, n_trials = rep(1L, length(idx)), X = X,
      prior = list(type = "nngp", coords = co, nn_idx = ni, nn_dist = nd,
                   n_spatial = n, nn = nn, spatial_idx = idx, cov_type = 0L),
      family = "gaussian", phi = 0.5,
      control = list(max_iter = 100L, tol = 1e-8, n_threads = 1L,
                     progress = FALSE, diagnose_k = FALSE,
                     diagnose_skew = FALSE)))$log_marginal
  }
  base <- fit_lm(co2)
  expect_identical(fit_lm(cbind(co2, 0)), base)
  expect_identical(fit_lm(cbind(co2, 3.5)), base)
  # `nn_idx` / `nn_dist` still describe the 2-D neighbour graph, so a third
  # coordinate that varies changes only the neighbour-to-neighbour covariance --
  # which is exactly the quantity that was reading out of bounds.
  expect_false(identical(fit_lm(cbind(co2, runif(n))), base))
})

test_that("the neighbour graph reads the same dimension the covariance does", {
  # gcol33/tulpa#391. gcol33/tulpa#389 made the neighbour COVARIANCE read every
  # coordinate column; `compute_nngp_neighbors()` still selected neighbours over
  # the first two, so on anything wider than 2 the selection and the covariance
  # would have been computed under different metrics -- and 1-D could not build
  # a graph at all, which is what put the dimension-general kernels out of reach
  # from the front door.
  set.seed(3L); n <- 40L
  co2 <- cbind(runif(n), runif(n))
  g   <- compute_nngp_neighbors(co2, 5L)

  # The 2-D path is the shipped one and must not have moved. Exact, not
  # toleranced: the generalisation reassociates a sum of two squares and nothing
  # else, so any difference at all would be a change of behaviour.
  o_idx <- order(co2[, 1], co2[, 2])
  cs    <- co2[o_idx, , drop = FALSE]
  ref_d <- sqrt((cs[1:9, 1] - cs[10, 1])^2 + (cs[1:9, 2] - cs[10, 2])^2)
  expect_equal(sort(ref_d)[1:5], unname(g$nn_dist[10, ]), tolerance = 0)

  # A constant extra column is the same domain, so it must give the same graph.
  g3c <- compute_nngp_neighbors(cbind(co2, 7), 5L)
  expect_identical(g3c$nn_idx, g$nn_idx)
  expect_equal(g3c$nn_dist, g$nn_dist, tolerance = 0)
  expect_equal(g3c$nn_neighbor_dist, g$nn_neighbor_dist, tolerance = 0)

  # 1-D and 3-D build at all, which they did not before.
  g1 <- compute_nngp_neighbors(matrix(sort(runif(n)), ncol = 1L), 5L)
  g3 <- compute_nngp_neighbors(cbind(runif(n), runif(n), runif(n)), 5L)
  expect_true(all(is.finite(g1$nn_neighbor_dist)))
  expect_true(all(is.finite(g3$nn_neighbor_dist)))
  # A third coordinate that VARIES has to change the graph, or it is being
  # dropped rather than read.
  expect_false(identical(compute_nngp_neighbors(cbind(co2, runif(n)), 5L)$nn_idx,
                         g$nn_idx))
})

test_that("each spatial door admits the dimension its storage can carry", {
  # One parser (`.parse_coord_spec()`) behind three doors that used to hold
  # three verbatim copies of it, with the arity as its one policy argument.
  # NNGP reads any dimension; the HSGP basis and every sampler spec are 2-D by
  # storage layout, so they refuse rather than truncate.
  expect_s3_class(spatial_gp(~x, nn = 5), "tulpa_gp")
  expect_s3_class(spatial_gp(~x + y + z, nn = 5), "tulpa_gp")
  expect_s3_class(spatial_gp(c("a", "b", "c"), nn = 5), "tulpa_gp")
  expect_error(spatial_gp(~x + y + z, approx = "hsgp"), "exactly 2")
  expect_error(spatial_gp(~x + y + z, approx = "hsgp"), "NNGP approximation")
  expect_error(spatial_svc(~x + y + z), "exactly 2")
  expect_error(spatial_multiscale(~x + y + z), "exactly 2")
  # A spec that names no coordinate at all is still refused.
  expect_error(spatial_gp(3), "must be a formula")
  expect_error(spatial_gp(~1), "at least one coordinate")
})
