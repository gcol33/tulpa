# Exact block-Schur sum-to-zero log-determinant + Newton step (gcol33/tulpa#69).
#
# B = A + sum_k coef_k 1_k 1_k', where A is the sparse joint Hessian and each
# pin is a sum-to-zero penalty on a contiguous intrinsic field block. The
# fused batched joint driver and the single-species sparse oracle both route
# their inner step and Laplace log-determinant through s2z_block_schur /
# s2z_log_det_block_schur (the field block is factored once, the rank-1 pins
# fold via the matrix-determinant lemma, the field<->scalar coupling closes
# with a small dense Schur). This locks that path's two outputs against
# independent references:
#   * log|B| vs s2z_log_det_direct (the CHOLMOD full-(A+11') path) AND a
#     dense Eigen LLT factorization of B;
#   * the Newton step B^{-1} grad vs a dense Eigen LLT solve.
# A divergence here is a genuine engine bug: the two paths must be identical.

# Symmetric, diagonally-dominant (hence SPD) A with the s2z layout: a sparse
# (chain) field block, a dense scalar block, and sparse field<->scalar coupling.
.s2z_make_A <- function(n, nf, seed, density = 0.15) {
  set.seed(seed)
  A <- matrix(0, n, n)
  for (i in seq_len(nf - 1L)) A[i, i + 1L] <- A[i + 1L, i] <- -1
  sc <- (nf + 1L):n
  for (i in sc) for (j in sc) if (j >= i) {
    v <- stats::rnorm(1) * 0.3; A[i, j] <- A[j, i] <- v
  }
  for (i in seq_len(nf)) for (j in sc) if (stats::runif(1) < density) {
    v <- stats::rnorm(1) * 0.2; A[i, j] <- A[j, i] <- v
  }
  diag(A) <- rowSums(abs(A)) + stats::runif(n, 1, 2)   # strict diagonal dominance
  A
}

.s2z_run <- function(n, nf, pins, seed, coupling = NULL) {
  A <- .s2z_make_A(n, nf, seed)
  g <- stats::rnorm(n, sd = 0.7)
  tulpa:::cpp_test_s2z_block_schur(
    A            = A,
    pin_start    = as.integer(vapply(pins, `[[`, integer(1), "start")),
    pin_n        = as.integer(vapply(pins, `[[`, integer(1), "n")),
    pin_coef     = as.numeric(vapply(pins, `[[`, numeric(1), "coef")),
    grad         = g,
    pin_coupling = coupling)
}

.rel <- function(a, b) abs(a - b) / max(1, abs(b))

test_that("block-Schur log|B| matches the direct and dense references (single pin)", {
  for (seed in 1:5) {
    r <- .s2z_run(n = 90L, nf = 60L, pins = list(list(start = 0L, n = 60L, coef = 1.0)),
                  seed = seed)
    expect_true(r$ok)
    expect_true(is.finite(r$ld_block_schur))
    expect_lt(.rel(r$ld_block_schur, r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_direct,      r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_block_schur, r$ld_direct), 1e-9)
    expect_lt(.rel(r$ld_block_schur_step, r$ld_dense), 1e-8)
    expect_lt(r$max_dstep, 1e-8)
  }
})

test_that("block-Schur is exact for two pins (two intrinsic fields)", {
  for (seed in 11:14) {
    r <- .s2z_run(n = 130L, nf = 80L,
                  pins = list(list(start = 0L,  n = 40L, coef = 0.5),
                              list(start = 40L, n = 40L, coef = 2.0)),
                  seed = seed)
    expect_true(r$ok)
    expect_lt(.rel(r$ld_block_schur, r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_block_schur, r$ld_direct), 1e-9)
    expect_lt(r$max_dstep, 1e-8)
  }
})

# --- Dense coupling D: the fold is the rank-K `U D U'` -----------------------
# A Kronecker-structured field (multivariate CAR, precision Sigma^-1 (x) Q) has
# the sum-to-zero augmentation Sigma^-1 (x) 11'/J, which couples FIELDS:
# D[(a,c),(b,c)] = Sinv[a,b]/J_c. K independent rank-1 pins cannot express that
# off-diagonal, so the pins carry an optional dense D.

.s2z_two_field_pins <- list(list(start = 0L,  n = 40L, coef = 1.0),
                            list(start = 40L, n = 40L, coef = 1.0))

test_that("an explicit diagonal coupling reproduces the diag(coef) path", {
  # D = diag(coef) is the pre-existing behaviour, so supplying it explicitly has
  # to land on the same numbers -- the generalization is a strict superset.
  pins <- list(list(start = 0L,  n = 40L, coef = 1.3),
               list(start = 40L, n = 40L, coef = 0.7))
  base <- .s2z_run(n = 130L, nf = 80L, pins = pins, seed = 21L)
  cpl  <- .s2z_run(n = 130L, nf = 80L, pins = pins, seed = 21L,
                   coupling = diag(c(1.3, 0.7)))
  expect_equal(cpl$ld_block_schur, base$ld_block_schur, tolerance = 1e-12)
  expect_equal(cpl$ld_direct,      base$ld_direct,      tolerance = 1e-12)
  expect_equal(cpl$ld_dense,       base$ld_dense,       tolerance = 1e-12)
  expect_equal(cpl$max_dstep,      base$max_dstep,      tolerance = 1e-12)
})

test_that("a cross-block coupling matches the dense U D U' reference", {
  Sinv <- matrix(c(2.0, 0.8, 0.8, 1.5), 2L, 2L)   # SPD, genuine off-diagonal
  D    <- Sinv / 40                                # J_c = 40
  for (seed in 31:34) {
    r <- .s2z_run(n = 130L, nf = 80L, pins = .s2z_two_field_pins, seed = seed,
                  coupling = D)
    expect_true(r$ok)
    expect_true(is.finite(r$ld_block_schur))
    expect_lt(.rel(r$ld_block_schur, r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_direct,      r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_block_schur, r$ld_direct), 1e-9)
    expect_lt(.rel(r$ld_block_schur_step, r$ld_dense), 1e-8)
    expect_lt(r$max_dstep, 1e-8)
  }
})

test_that("the off-diagonal of D actually moves the answer", {
  # Guards the two tests above against passing vacuously: if the cross-block
  # term were dropped anywhere, a full D and its diagonal would agree.
  Sinv <- matrix(c(2.0, 0.8, 0.8, 1.5), 2L, 2L)
  D    <- Sinv / 40
  full <- .s2z_run(n = 130L, nf = 80L, pins = .s2z_two_field_pins, seed = 31L,
                   coupling = D)
  diagonly <- .s2z_run(n = 130L, nf = 80L, pins = .s2z_two_field_pins,
                       seed = 31L, coupling = diag(diag(D)))
  expect_gt(abs(full$ld_dense - diagonly$ld_dense), 1e-6)
  expect_gt(abs(full$ld_block_schur - diagonly$ld_block_schur), 1e-6)
})

test_that("block-Schur is exact at the large-field / large-coef sum-to-zero regime", {
  # n_x past the densify boundary, with the strong sum-to-zero pin the intrinsic
  # field actually registers -- the regime where the older matrix-determinant
  # Woodbury log-det cancelled.
  for (coef in c(1.0, 1e3, 1e5)) {
    r <- .s2z_run(n = 320L, nf = 280L,
                  pins = list(list(start = 0L, n = 280L, coef = coef)),
                  seed = 7L)
    expect_true(r$ok)
    expect_lt(.rel(r$ld_block_schur, r$ld_dense),  1e-7)
    expect_lt(.rel(r$ld_block_schur, r$ld_direct), 1e-8)
    expect_lt(r$max_dstep, 1e-7)
  }
})

# --- Non-contiguous pins: a disconnected map's component node sets -----------
# A genuine disconnected map's connected components need not be contiguous in
# the node ordering (islands sorted among the mainland). The rank-1 fold then
# carries an arbitrary node list rather than a [start, start+n) range; it must
# fold through the SAME Woodbury / block-Schur code, exact against the dense
# Eigen reference over the same node sets.

test_that("block-Schur is exact for non-contiguous (interleaved) pins", {
  for (seed in 41:44) {
    n <- 130L; nf <- 80L
    A <- .s2z_make_A(n, nf, seed)
    g <- stats::rnorm(n, sd = 0.7)
    fld  <- 0:(nf - 1L)
    idx1 <- fld[fld %% 2L == 0L]         # even field indices (non-contiguous)
    idx2 <- fld[fld %% 2L == 1L]         # odd  field indices (non-contiguous)
    r <- tulpa:::cpp_test_s2z_block_schur(
      A = A, pin_start = c(0L, 0L), pin_n = c(0L, 0L),
      pin_coef = c(0.5, 2.0), grad = g,
      pin_idx = list(as.integer(idx1), as.integer(idx2)))
    expect_true(r$ok)
    expect_lt(.rel(r$ld_block_schur, r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_direct,      r$ld_dense),  1e-8)
    expect_lt(.rel(r$ld_block_schur, r$ld_direct), 1e-9)
    expect_lt(.rel(r$ld_block_schur_step, r$ld_dense), 1e-8)
    expect_lt(r$max_dstep, 1e-8)
  }
})

test_that("a non-contiguous pin is not the contiguous pin of the same count", {
  # Guards the test above against the idx path silently collapsing to a
  # contiguous [0, n) fold: the same coef and count over interleaved vs
  # contiguous nodes must give a different B (hence a different log|B|).
  n <- 130L; nf <- 80L
  A <- .s2z_make_A(n, nf, 41L)
  g <- stats::rnorm(n, sd = 0.7)
  fld  <- 0:(nf - 1L)
  idx1 <- fld[fld %% 2L == 0L]; idx2 <- fld[fld %% 2L == 1L]
  nc <- tulpa:::cpp_test_s2z_block_schur(
    A = A, pin_start = c(0L, 0L), pin_n = c(0L, 0L), pin_coef = c(0.5, 2.0),
    grad = g, pin_idx = list(as.integer(idx1), as.integer(idx2)))
  cg <- tulpa:::cpp_test_s2z_block_schur(
    A = A, pin_start = c(0L, 40L), pin_n = c(40L, 40L), pin_coef = c(0.5, 2.0),
    grad = g)
  expect_gt(abs(nc$ld_dense - cg$ld_dense), 1e-6)
  expect_gt(abs(nc$ld_block_schur - cg$ld_block_schur), 1e-6)
})
