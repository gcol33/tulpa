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

.s2z_run <- function(n, nf, pins, seed) {
  A <- .s2z_make_A(n, nf, seed)
  g <- stats::rnorm(n, sd = 0.7)
  tulpa:::cpp_test_s2z_block_schur(
    A          = A,
    pin_start  = as.integer(vapply(pins, `[[`, integer(1), "start")),
    pin_n      = as.integer(vapply(pins, `[[`, integer(1), "n")),
    pin_coef   = as.numeric(vapply(pins, `[[`, numeric(1), "coef")),
    grad       = g)
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
