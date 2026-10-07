# cpp_joint_inner_vcov_blocks: the selected-inversion + parallel per-cell
# constrained inner-covariance extraction for the joint post-grid step
# (gcol33/tulpa#112, #113; gcol33/tulpaObs#93). Recovery against a dense
# reference: solve(Q, E) minus the conditioning-by-kriging correction.

# Dense reference for the constrained block of `idx` under constraint groups
# `A_cols` (1-based latent indices), the law the cheap recipe reproduces.
.ref_inner_block <- function(Q, idx, A_cols) {
  n_x <- nrow(Q)
  C   <- solve(Q)
  E   <- matrix(0, n_x, length(idx))
  for (j in seq_along(idx)) E[idx[j], j] <- 1
  V   <- C %*% E
  blk <- V[idx, , drop = FALSE]
  if (length(A_cols) > 0L) {
    At <- matrix(0, n_x, length(A_cols))
    for (g in seq_along(A_cols)) At[A_cols[[g]], g] <- 1
    W  <- C %*% At
    M  <- t(At) %*% W
    AV <- t(At) %*% V
    blk <- blk - t(AV) %*% solve(M, AV)
  }
  as.matrix(blk)
}

# Build a random joint-like precision: p_d dense (fixed-effect) coordinates
# coupling to one or more banded ICAR-like fields, symmetric PD.
.make_joint_Q <- function(p_d, field_sizes, seed = 1) {
  set.seed(seed)
  n_f <- sum(field_sizes)
  n_x <- p_d + n_f
  Q <- matrix(0, n_x, n_x)
  Q[seq_len(p_d), seq_len(p_d)] <- diag(2, p_d) + 0.1
  off <- p_d
  field_idx <- list()
  for (s in field_sizes) {
    fi <- off + seq_len(s)
    Lap <- matrix(0, s, s)
    for (i in seq_len(s - 1)) {
      Lap[i, i] <- Lap[i, i] + 1; Lap[i + 1, i + 1] <- Lap[i + 1, i + 1] + 1
      Lap[i, i + 1] <- -1; Lap[i + 1, i] <- -1
    }
    Q[fi, fi] <- Lap + diag(0.3, s)
    Qdf <- matrix(stats::rnorm(p_d * s, sd = 0.2), p_d, s)
    Q[seq_len(p_d), fi] <- Qdf
    Q[fi, seq_len(p_d)] <- t(Qdf)
    field_idx[[length(field_idx) + 1L]] <- fi
    off <- off + s
  }
  Q <- (Q + t(Q)) / 2
  # Guarantee PD (the betas x field coupling can otherwise exceed the dense
  # block's diagonal): floor the spectrum well above zero.
  ev <- eigen(Q, symmetric = TRUE, only.values = TRUE)$values
  Q  <- Q + diag(max(0, -min(ev)) + 0.5, n_x)
  list(Q = Q, n_x = n_x, p_d = p_d, field_idx = field_idx)
}

.csc_lower <- function(Q) {
  Qsp <- methods::as(methods::as(Q, "generalMatrix"), "CsparseMatrix")
  Qlt <- Matrix::tril(Qsp)
  list(p = as.integer(Qlt@p), i = as.integer(Qlt@i), x = as.numeric(Qlt@x))
}

test_that("cheap recipe matches the dense reference on the read sub-blocks", {
  jd <- .make_joint_Q(p_d = 4L, field_sizes = 30L, seed = 11)
  fidx   <- jd$field_idx[[1L]]
  idx    <- c(seq_len(jd$p_d), fidx)
  A_cols <- list(fidx)
  ref    <- .ref_inner_block(jd$Q, idx, A_cols)
  csc    <- .csc_lower(jd$Q)

  got <- cpp_joint_inner_vcov_blocks(
    list(csc$p), list(csc$i), list(csc$x), jd$n_x,
    as.integer(idx), jd$p_d, lapply(A_cols, as.integer),
    field_marginal = TRUE, n_threads = 1L)[[1L]]

  pd <- seq_len(jd$p_d); fr <- (jd$p_d + 1L):length(idx)
  expect_equal(got[pd, pd], ref[pd, pd], tolerance = 1e-9)         # betas block
  expect_equal(got[pd, fr], ref[pd, fr], tolerance = 1e-9)         # betas x field
  expect_equal(diag(got)[fr], diag(ref)[fr], tolerance = 1e-9)     # field marginals
  foff <- got[fr, fr]; diag(foff) <- 0
  expect_identical(max(abs(foff)), 0)                              # field off-diag 0
})

test_that("full mode reproduces the entire dense constrained block", {
  jd <- .make_joint_Q(p_d = 3L, field_sizes = 20L, seed = 22)
  fidx   <- jd$field_idx[[1L]]
  idx    <- c(seq_len(jd$p_d), fidx)
  A_cols <- list(fidx)
  ref    <- .ref_inner_block(jd$Q, idx, A_cols)
  csc    <- .csc_lower(jd$Q)

  got <- cpp_joint_inner_vcov_blocks(
    list(csc$p), list(csc$i), list(csc$x), jd$n_x,
    as.integer(idx), length(idx), lapply(A_cols, as.integer),
    field_marginal = FALSE, n_threads = 1L)[[1L]]
  expect_equal(got, ref, tolerance = 1e-9)
})

test_that("two constraint groups (BYM2-style) recover correctly", {
  # Two fields, each with its own sum-to-zero constraint (the BYM2 structured +
  # unstructured pattern: two constraint rows).
  jd <- .make_joint_Q(p_d = 2L, field_sizes = c(15L, 15L), seed = 33)
  f1 <- jd$field_idx[[1L]]; f2 <- jd$field_idx[[2L]]
  idx    <- c(seq_len(jd$p_d), f1, f2)
  A_cols <- list(f1, f2)
  ref    <- .ref_inner_block(jd$Q, idx, A_cols)
  csc    <- .csc_lower(jd$Q)

  got <- cpp_joint_inner_vcov_blocks(
    list(csc$p), list(csc$i), list(csc$x), jd$n_x,
    as.integer(idx), jd$p_d, lapply(A_cols, as.integer),
    field_marginal = TRUE, n_threads = 1L)[[1L]]

  pd <- seq_len(jd$p_d); fr <- (jd$p_d + 1L):length(idx)
  expect_equal(got[pd, pd], ref[pd, pd], tolerance = 1e-9)
  expect_equal(diag(got)[fr], diag(ref)[fr], tolerance = 1e-9)
})

test_that("multi-cell parallel run is identical to serial and per-cell", {
  jd1 <- .make_joint_Q(p_d = 3L, field_sizes = 25L, seed = 44)
  jd2 <- .make_joint_Q(p_d = 3L, field_sizes = 25L, seed = 45)
  fidx   <- jd1$field_idx[[1L]]
  idx    <- c(seq_len(jd1$p_d), fidx)
  A_cols <- list(fidx)
  c1 <- .csc_lower(jd1$Q); c2 <- .csc_lower(jd2$Q)

  args <- list(
    Q_p_per_grid = list(c1$p, c2$p),
    Q_i_per_grid = list(c1$i, c2$i),
    Q_x_per_grid = list(c1$x, c2$x),
    n_x = jd1$n_x, idx = as.integer(idx), n_dense = jd1$p_d,
    A_cols_list = lapply(A_cols, as.integer), field_marginal = TRUE)

  par <- do.call(cpp_joint_inner_vcov_blocks, c(args, list(n_threads = 2L)))
  ser <- do.call(cpp_joint_inner_vcov_blocks, c(args, list(n_threads = 1L)))
  expect_equal(par, ser)                                   # threading invariant
  pd <- seq_len(jd1$p_d)
  ref1 <- .ref_inner_block(jd1$Q, idx, A_cols)
  ref2 <- .ref_inner_block(jd2$Q, idx, A_cols)
  expect_equal(par[[1L]][pd, pd], ref1[pd, pd], tolerance = 1e-9)
  expect_equal(par[[2L]][pd, pd], ref2[pd, pd], tolerance = 1e-9)
})

test_that("a NULL / empty cell yields a NULL block", {
  jd <- .make_joint_Q(p_d = 2L, field_sizes = 10L, seed = 55)
  fidx <- jd$field_idx[[1L]]; idx <- c(seq_len(jd$p_d), fidx)
  csc  <- .csc_lower(jd$Q)
  out <- cpp_joint_inner_vcov_blocks(
    list(csc$p, NULL), list(csc$i, NULL), list(csc$x, NULL),
    jd$n_x, as.integer(idx), jd$p_d, list(as.integer(fidx)),
    field_marginal = TRUE, n_threads = 1L)
  expect_false(is.null(out[[1L]]))
  expect_null(out[[2L]])
})

test_that("an out-of-range idx or constraint index is rejected at the boundary", {
  jd   <- .make_joint_Q(p_d = 2L, field_sizes = 10L, seed = 66)
  fidx <- jd$field_idx[[1L]]
  idx  <- c(seq_len(jd$p_d), fidx)
  csc  <- .csc_lower(jd$Q)
  call_with <- function(idx_use, A_use = list(as.integer(fidx))) {
    cpp_joint_inner_vcov_blocks(
      list(csc$p), list(csc$i), list(csc$x), jd$n_x,
      as.integer(idx_use), jd$p_d, A_use,
      field_marginal = TRUE, n_threads = 1L)
  }
  expect_error(call_with(replace(idx, 1L, 0L)), "idx")
  expect_error(call_with(replace(idx, 1L, jd$n_x + 1L)), "idx")
  expect_error(call_with(replace(idx, 1L, NA_integer_)), "idx")
  expect_error(call_with(seq_len(jd$n_x + 1L)), "idx")
  expect_error(call_with(idx, list(as.integer(c(fidx, jd$n_x + 5L)))), "A_cols")
  expect_error(call_with(idx, list(as.integer(c(fidx, NA)))), "A_cols")
  expect_error(call_with(idx, list(as.integer(c(fidx, 0L)))), "A_cols")
  expect_silent(call_with(idx))
})

test_that("a malformed CSC triple yields a NULL block instead of reading past it", {
  jd   <- .make_joint_Q(p_d = 2L, field_sizes = 10L, seed = 77)
  fidx <- jd$field_idx[[1L]]
  idx  <- c(seq_len(jd$p_d), fidx)
  csc  <- .csc_lower(jd$Q)
  one  <- function(pp, ii, xx) {
    cpp_joint_inner_vcov_blocks(
      list(pp), list(ii), list(xx), jd$n_x, as.integer(idx), jd$p_d,
      list(as.integer(fidx)), field_marginal = TRUE, n_threads = 1L)[[1L]]
  }
  expect_false(is.null(one(csc$p, csc$i, csc$x)))
  expect_null(one(csc$p[-1L], csc$i, csc$x))                  # short pointer array
  expect_null(one(csc$p, csc$i[-1L], csc$x))                  # Qi shorter than Qx
  expect_null(one(csc$p, replace(csc$i, 1L, jd$n_x), csc$x))  # row index past n_x
  expect_null(one(csc$p, replace(csc$i, 1L, -1L), csc$x))     # negative row index
  expect_null(one(replace(csc$p, 2L, csc$p[1L] - 1L), csc$i, csc$x))  # not monotone
})

test_that("a repeated latent index in a constraint group is read as an indicator", {
  # Row g of A is the indicator of group g's indices, so listing an index twice
  # must give the same constrained block as listing it once. The right-hand side
  # W = Q^-1 A' and the Gram matrix M = A W have to use that same A.
  jd   <- .make_joint_Q(p_d = 3L, field_sizes = 12L, seed = 88)
  fidx <- jd$field_idx[[1L]]
  idx  <- c(seq_len(jd$p_d), fidx)
  csc  <- .csc_lower(jd$Q)
  one  <- function(A_use) cpp_joint_inner_vcov_blocks(
    list(csc$p), list(csc$i), list(csc$x), jd$n_x, as.integer(idx),
    length(idx), A_use, field_marginal = FALSE, n_threads = 1L)[[1L]]

  clean <- one(list(as.integer(fidx)))
  duped <- one(list(as.integer(c(fidx, fidx[c(1L, 3L, 3L)]))))
  expect_identical(duped, clean)
  expect_equal(clean, .ref_inner_block(jd$Q, idx, list(fidx)), tolerance = 1e-9)
})

# tulpa_joint_inner_vcov_mixture(): the law-of-total-covariance mixture over the
# grid, accumulated in the engine without one p x p block per cell
# (gcol33/tulpa#947). Reference: the same sum built in R from the per-cell
# blocks cpp_joint_inner_vcov_blocks() returns.
.mixture_fixture <- function(n_cell = 4L, seed = 101) {
  jds  <- lapply(seq_len(n_cell), function(k)
    .make_joint_Q(p_d = 3L, field_sizes = 18L, seed = seed + k))
  fidx <- jds[[1L]]$field_idx[[1L]]
  idx  <- c(seq_len(jds[[1L]]$p_d), fidx)
  cscs <- lapply(jds, function(jd) .csc_lower(jd$Q))
  set.seed(seed)
  list(
    Qp = lapply(cscs, `[[`, "p"), Qi = lapply(cscs, `[[`, "i"),
    Qx = lapply(cscs, `[[`, "x"), n_x = jds[[1L]]$n_x, idx = idx,
    p_d = jds[[1L]]$p_d, A = list(as.integer(fidx)),
    modes = matrix(stats::rnorm(n_cell * length(idx)), n_cell, length(idx)),
    w = stats::runif(n_cell))
}

.mixture_reference <- function(fx, field_marginal = TRUE, n_dense = fx$p_d) {
  blocks <- cpp_joint_inner_vcov_blocks(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, as.integer(fx$idx), n_dense, fx$A,
    field_marginal = field_marginal, n_threads = 1L)
  w <- fx$w / sum(fx$w)
  mbar <- as.numeric(crossprod(w, fx$modes))
  V <- 0
  for (k in seq_along(w)) {
    if (w[k] == 0) next
    Ck <- if (is.null(blocks[[k]])) 0 else blocks[[k]]
    V <- V + w[k] * (Ck + tcrossprod(fx$modes[k, ] - mbar))
  }
  list(V = V, mbar = mbar, blocks = blocks)
}

test_that("the engine mixture equals the law of total covariance over the blocks", {
  fx  <- .mixture_fixture()
  ref <- .mixture_reference(fx)
  got <- tulpa_joint_inner_vcov_mixture(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, fx$p_d, fx$A,
    weights = fx$w, modes = fx$modes)
  expect_equal(got$vcov, ref$V, tolerance = 1e-12)
  expect_equal(got$mean, ref$mbar, tolerance = 1e-14)
  expect_identical(got$failed, integer(0))
  pd <- seq_len(fx$p_d)
  for (k in seq_along(fx$w))
    expect_equal(got$dense_blocks[[k]], ref$blocks[[k]][pd, pd], tolerance = 1e-14)

  full <- tulpa_joint_inner_vcov_mixture(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, length(fx$idx), fx$A,
    weights = fx$w, modes = fx$modes, field_marginal = FALSE)
  ref_full <- .mixture_reference(fx, field_marginal = FALSE,
                                 n_dense = length(fx$idx))
  expect_equal(full$vcov, ref_full$V, tolerance = 1e-12)

  within <- tulpa_joint_inner_vcov_mixture(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, fx$p_d, fx$A, weights = fx$w)
  expect_null(within$mean)
  expect_equal(within$vcov + crossprod(sqrt(fx$w / sum(fx$w)) *
                 sweep(fx$modes, 2L, ref$mbar)), ref$V, tolerance = 1e-12)
})

test_that("the engine mixture does not depend on the thread count", {
  fx <- .mixture_fixture(n_cell = 7L, seed = 202)
  one <- function(nt) tulpa_joint_inner_vcov_mixture(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, fx$p_d, fx$A,
    weights = fx$w, modes = fx$modes, n_threads = nt)
  expect_identical(one(3L), one(1L))
})

test_that("a zero-weight cell is skipped and a missing cell keeps its between term", {
  fx <- .mixture_fixture(n_cell = 4L, seed = 303)
  fx$w[2L] <- 0
  fx$Qp[3L] <- list(NULL)
  ref <- .mixture_reference(fx)
  got <- tulpa_joint_inner_vcov_mixture(
    fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, fx$p_d, fx$A,
    weights = fx$w, modes = fx$modes)
  expect_equal(got$vcov, ref$V, tolerance = 1e-12)
  expect_null(got$dense_blocks[[2L]])
  expect_null(got$dense_blocks[[3L]])
  expect_identical(got$failed, 3L)
})

test_that("the engine mixture refuses weights and modes that do not fit the grid", {
  fx <- .mixture_fixture(n_cell = 3L, seed = 404)
  call_with <- function(w = fx$w, modes = fx$modes)
    tulpa_joint_inner_vcov_mixture(
      fx$Qp, fx$Qi, fx$Qx, fx$n_x, fx$idx, fx$p_d, fx$A,
      weights = w, modes = modes)
  expect_error(call_with(w = fx$w[-1L]), "weights")
  expect_error(call_with(w = c(-1, 1, 1)), "weights")
  expect_error(call_with(w = c(NA, 1, 1)), "weights")
  expect_error(call_with(w = c(0, 0, 0)), "mass")
  expect_error(call_with(modes = fx$modes[, -1L]), "modes")
  expect_error(call_with(modes = replace(fx$modes, 1L, NA)), "modes")
  expect_silent(call_with(modes = replace(fx$modes, 1L, NA) * c(1, 1, 1),
                          w = c(0, 1, 1)))
})
