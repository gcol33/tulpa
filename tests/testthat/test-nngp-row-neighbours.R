# One rule for how many neighbours a row of nn_idx has (gcol33/tulpa#433).
#
# nn_idx is 1-based into the Vecchia ordering with 0 meaning "no neighbour", and
# every builder in the package emits LEFT-PACKED rows: the neighbours occupy the
# leading columns and the tail is zero. Six kernels read that table, and they
# used to derive the count two ways -- the four density kernels counted every
# positive entry in the row, the gradient and the non-centred transform counted
# the leading run. The two agree only on a left-packed row.
#
# On any other row they disagree twice over. The count-every-positive rule
# conditions on a different neighbour set, so the analytic gradient stops being
# the gradient of the density it is paired with; and it then resolves slot j
# against a zero, giving nn_order[0 - 1], an out-of-bounds read on a
# std::vector<int> whose result is used to index coords and w.
#
# tulpa_nngp::nngp_row_neighbours() is now the one scan, and it stops at the
# first entry outside [1, n_order] -- so every column below the count it returns
# resolves inside nn_order by construction, with no per-site bounds check left
# to omit.

.nnrow_fixture <- function(N = 25L, nn = 5L, seed = 3L) {
  set.seed(seed)
  coords <- cbind(runif(N), runif(N))
  ni     <- compute_nngp_neighbors(coords, nn)
  order0 <- if (!is.null(ni$nn_order)) as.integer(ni$nn_order - 1L) else seq_len(N) - 1L
  inv    <- integer(N); inv[order0 + 1L] <- seq_len(N) - 1L
  set.seed(12)
  list(N = N, nn = nn, coords = coords, nn_idx = ni$nn_idx,
       nn_dist = ni$nn_dist, order0 = order0, inv = inv,
       nbd = as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1))),
       w = rnorm(N))
}

.nnrow_svc <- function(f, idx) {
  cpp_test_svc_nngp_twins(f$w, 1.0, 0.5, f$coords, idx, f$nn_dist, f$order0, 0L)
}
.nnrow_gp <- function(f, idx) {
  cpp_test_gp_nngp_twins(f$w, 1.0, 0.5, f$coords, idx, f$nn_dist, f$nbd,
                         f$order0, f$inv, 0L)
}

test_that("a row with an interior zero is read as its leading run", {
  f <- .nnrow_fixture()
  expect_gte(sum(f$nn_idx[10, ] > 0), 5)      # the row starts full

  hole  <- f$nn_idx; hole[10, 3]      <- 0L   # [a, b, 0, d, e]
  trunc <- f$nn_idx; trunc[10, 3:f$nn] <- 0L  # [a, b, 0, 0, 0]

  for (path in c("svc", "gp")) {
    run <- if (path == "svc") .nnrow_svc else .nnrow_gp
    got  <- run(f, hole)
    want <- run(f, trunc)
    expect_equal(unname(got[["dbl"]]), unname(want[["dbl"]]), tolerance = 1e-12,
                 info = paste(path, "double"))
    expect_equal(unname(got[["ad"]]), unname(want[["ad"]]), tolerance = 1e-12,
                 info = paste(path, "autodiff"))
    # ... and the twins still agree with each other on the malformed row, so the
    # density and the differentiated copy read the same neighbour set.
    expect_equal(unname(got[["ad"]]), unname(got[["dbl"]]), tolerance = 1e-9,
                 info = paste(path, "twins on the interior-zero row"))
  }
})

test_that("the leading-run reading is a different answer from counting positives", {
  # The negative control. Without this the test above passes for a fixture in
  # which the two rules happen to coincide.
  f <- .nnrow_fixture()
  hole   <- f$nn_idx; hole[10, 3] <- 0L
  # What the count-every-positive rule would have conditioned on, written as a
  # well-formed row: the four surviving entries packed to the front.
  packed <- f$nn_idx
  packed[10, ] <- c(f$nn_idx[10, 1:2], f$nn_idx[10, 4:f$nn], 0L)

  for (path in c("svc", "gp")) {
    run <- if (path == "svc") .nnrow_svc else .nnrow_gp
    expect_false(isTRUE(all.equal(unname(run(f, hole)[["dbl"]]),
                                  unname(run(f, packed)[["dbl"]]))),
                 info = path)
  }
})

test_that("an out-of-range entry ends the row instead of indexing past nn_order", {
  f <- .nnrow_fixture()
  oor   <- f$nn_idx; oor[10, 3]        <- as.integer(f$N + 7L)
  trunc <- f$nn_idx; trunc[10, 3:f$nn] <- 0L
  for (path in c("svc", "gp")) {
    run <- if (path == "svc") .nnrow_svc else .nnrow_gp
    expect_equal(unname(run(f, oor)[["dbl"]]), unname(run(f, trunc)[["dbl"]]),
                 tolerance = 1e-12, info = path)
  }
  # A negative entry is outside [1, n_order] the same way.
  neg <- f$nn_idx; neg[10, 3] <- -4L
  expect_equal(unname(.nnrow_svc(f, neg)[["dbl"]]),
               unname(.nnrow_svc(f, trunc)[["dbl"]]), tolerance = 1e-12)
})

test_that("the leading rows of the Vecchia ordering are read at their own counts", {
  # Rows 1..nn are short by construction -- location i has at most i - 1
  # predecessors -- so the scan has to return 0, 1, 2, ... there rather than the
  # table's column count. The first row conditions on nothing and takes the
  # marginal density.
  f <- .nnrow_fixture()
  runs <- apply(f$nn_idx, 1, function(r) {
    m <- 0L
    while (m < length(r) && r[m + 1L] >= 1L && r[m + 1L] <= nrow(f$nn_idx)) m <- m + 1L
    m
  })
  expect_identical(runs[1:5], c(0L, 1L, 2L, 3L, 4L))
  expect_true(all(is.finite(c(.nnrow_svc(f, f$nn_idx)[["dbl"]],
                              .nnrow_gp(f, f$nn_idx)[["dbl"]]))))
})
