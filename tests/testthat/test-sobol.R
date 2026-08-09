# Sobol' low-discrepancy sequence (src/sobol.h, src/sobol_direction_numbers.h).
#
# The generator returns points i = 1 .. n: the origin X_0 is skipped, because
# every caller maps the unit cube through qnorm and qnorm(0) is -Inf. Two of the
# checks below are stated in terms of that skip rather than around it.

sob <- tulpa:::cpp_sobol_points
sob_max_dim <- tulpa:::cpp_sobol_max_dim()

# ---------------------------------------------------------------------------
# An independent R construction, written from the published recurrence rather
# than from the C++ source, over the first few rows of the Joe & Kuo table
# (columns d, s, a, m_1 .. m_s). Direction numbers are carried as 32 logical
# bits, position p standing for the binary fraction 2^-p, so nothing here goes
# through R's signed 32-bit integer bit operations.
.jk_rows <- list(
  "2" = list(s = 1L, a = 0L, m = c(1L)),
  "3" = list(s = 2L, a = 1L, m = c(1L, 3L)),
  "4" = list(s = 3L, a = 1L, m = c(1L, 3L, 1L)),
  "5" = list(s = 3L, a = 2L, m = c(1L, 1L, 1L)),
  "6" = list(s = 4L, a = 1L, m = c(1L, 1L, 3L, 3L))
)

# Bits of the integer `x`, least significant first, as a logical vector.
.bits_of <- function(x, nbit) {
  out <- logical(nbit)
  for (b in seq_len(nbit)) {
    out[b] <- (x %% 2L) == 1L
    x <- x %/% 2L
  }
  out
}

# Shift a fraction's bit vector right by `s` places: bit at 2^-p moves to
# 2^-(p+s).
.shift_right <- function(v, s) {
  c(logical(s), v[seq_len(length(v) - s)])
}

# Columns v_1 .. v_32 of dimension `dim` as a 32 x 32 logical matrix.
.ref_direction <- function(dim, L = 32L) {
  V <- matrix(FALSE, L, L)
  if (dim == 1L) {
    for (k in seq_len(L)) V[k, k] <- TRUE
    return(V)
  }
  row <- .jk_rows[[as.character(dim)]]
  s <- row$s
  for (k in seq_len(min(s, L))) {
    mb <- .bits_of(row$m[k], k)
    # m_k / 2^k: bit (k - p) of m_k, zero-based, sits at fraction 2^-p.
    for (p in seq_len(k)) V[p, k] <- mb[k - p + 1L]
  }
  if (s < L) {
    ab <- .bits_of(row$a, max(s - 1L, 1L))
    for (k in seq.int(s + 1L, L)) {
      val <- xor(V[, k - s], .shift_right(V[, k - s], s))
      if (s >= 2L) {
        for (t in seq_len(s - 1L)) {
          # a is packed with its leading coefficient in the high bit.
          if (ab[s - t]) val <- xor(val, V[, k - t])
        }
      }
      V[, k] <- val
    }
  }
  V
}

# 1-based position of the rightmost zero bit of j.
.rz <- function(j) {
  k <- 1L
  while (j %% 2L == 1L) {
    j <- j %/% 2L
    k <- k + 1L
  }
  k
}

.ref_points <- function(n, d, L = 32L) {
  out <- matrix(0, n, d)
  pw <- 2^-seq_len(L)
  for (j in seq_len(d)) {
    V <- .ref_direction(j, L)
    x <- logical(L)
    for (i in seq_len(n)) {
      x <- xor(x, V[, .rz(i - 1L)])
      out[i, j] <- sum(pw[x])
    }
  }
  out
}

# ---------------------------------------------------------------------------

test_that("the generator matches an independent R construction of the recurrence", {
  expect_equal(sob(128L, 6L), .ref_points(128L, 6L), tolerance = 0)
})

test_that("points lie strictly inside the unit cube", {
  P <- sob(4096L, 64L)
  expect_true(all(is.finite(P)))
  expect_true(all(P > 0))
  expect_true(all(P < 1))
  expect_false(any(P == 0))
  expect_false(any(P == 1))
  # Mapping through qnorm is the reason the origin is skipped; nothing the
  # generator emits may reach an infinite quantile.
  expect_true(all(is.finite(stats::qnorm(P))))
})

test_that("dimension 1 is the bit-reversed Gray code, and a van der Corput permutation", {
  # X_i = reverse_bits_32(i XOR (i >> 1)) / 2^32, computed here from the bit
  # pattern of i rather than from the generator's own recurrence.
  revbits <- function(v) {
    out <- numeric(length(v))
    for (b in 0:31) out <- out + bitwAnd(bitwShiftR(v, b), 1L) * 2^(31 - b)
    out
  }
  i <- 1:64
  expect_equal(sob(64L, 1L)[, 1],
               revbits(bitwXor(i, bitwShiftR(i, 1L))) / 2^32,
               tolerance = 0)

  # Over a full block of 2^m - 1 points the Gray-code order is a permutation of
  # the van der Corput values, the origin being the one member left out.
  expect_equal(sort(sob(63L, 1L)[, 1]), sort(revbits(1:63) / 2^32),
               tolerance = 0)
})

test_that("each one-dimensional projection is an elementary-interval net", {
  # X_0 .. X_(2^m - 1) put exactly one point in every interval
  # [j / 2^m, (j + 1) / 2^m). The origin is not emitted, so the returned
  # 2^m - 1 points fill intervals 1 .. 2^m - 1 once each and leave interval 0
  # empty. Any error in the direction numbers or in the Gray-code index breaks
  # this immediately.
  for (m in c(4L, 6L, 8L, 10L, 12L)) {
    n <- 2L^m - 1L
    P <- sob(n, 32L)
    cell <- floor(P * 2^m)
    for (j in seq_len(ncol(P))) {
      expect_identical(sort(cell[, j]), as.numeric(seq_len(2L^m - 1L)),
                       info = sprintf("m = %d, dimension %d", m, j))
    }
  }
})

test_that("the sequence is a prefix sequence and is deterministic", {
  long <- sob(1000L, 7L)
  expect_equal(sob(37L, 7L), long[1:37, , drop = FALSE], tolerance = 0)
  expect_equal(sob(1000L, 7L), long, tolerance = 0)
  # A coordinate does not depend on how many dimensions were asked for.
  expect_equal(sob(200L, 3L), sob(200L, 40L)[1:200, 1:3], tolerance = 0)
})

test_that("integration error decays faster than Monte Carlo", {
  # The arbiter for low discrepancy: a smooth integrand over [0, 1]^d whose
  # value is available in closed form,
  #   int exp(-sum u_i^2) du = (sqrt(pi) / 2 * erf(1))^d,
  # integrated over a ladder of n. Regressing log |error| on log n gives about
  # -1/2 for plain Monte Carlo; a genuine low-discrepancy set decays faster.
  # The Monte Carlo arm is an RMSE over replicates at a pinned seed, so the
  # whole comparison is deterministic.
  #
  # The n ladder rises with d because the n^-1 behaviour is asymptotic in n
  # relative to 2^d: at d = 18 the QMC slope over n = 2^6 .. 2^13 is only
  # -0.40, and -0.78 over n = 2^12 .. 2^17.
  testf <- function(U) exp(-rowSums(U^2))
  i1 <- sqrt(pi) / 2 * (2 * stats::pnorm(sqrt(2)) - 1)
  n_rep <- 12L

  cfg <- list(list(d = 1L, pw = 6:13), list(d = 2L, pw = 6:13),
              list(d = 14L, pw = 10:16), list(d = 18L, pw = 12:17))

  for (cc in cfg) {
    d <- cc$d
    ns <- 2^cc$pw
    truth <- i1^d

    err_q <- vapply(ns, function(n) abs(mean(testf(sob(n, d))) - truth), 0)
    set.seed(20260351)
    err_m <- vapply(ns, function(n) {
      e <- vapply(seq_len(n_rep), function(r) {
        mean(testf(matrix(stats::runif(n * d), n, d))) - truth
      }, 0)
      sqrt(mean(e^2))
    }, 0)

    slope_q <- unname(stats::coef(stats::lm(log(err_q) ~ log(ns)))[2])
    slope_m <- unname(stats::coef(stats::lm(log(err_m) ~ log(ns)))[2])

    expect_lt(slope_q, slope_m - 0.15)
    # And the level, not only the rate: at the top of each ladder the Sobol'
    # error is a fraction of the Monte Carlo RMSE.
    expect_lt(err_q[length(err_q)] / err_m[length(err_m)], 0.5)
  }
})

test_that("dimensions outside the table are refused", {
  expect_identical(sob_max_dim, 1024L)
  expect_error(sob(16L, sob_max_dim + 1L), "exceeds the tabulated maximum")
  expect_error(sob(16L, 0L), "d must be positive")
  expect_error(sob(16L, -3L), "d must be positive")
  expect_error(sob(0L, 4L), "n must be positive")
  expect_error(sob(-1L, 4L), "n must be positive")
  # The last tabulated dimension is reachable.
  expect_equal(dim(sob(8L, sob_max_dim)), c(8L, sob_max_dim))
})
