# Diagonal-plus-low-rank mass metric (gcol33/tulpa#597).
#
# M_block = D + U Lambda U', with column g of U the indicator of a group of
# block coordinates. The engine never forms M or its inverse: the inverse is
# Woodbury on a rank x rank inner matrix and the momentum draw is a sum of two
# independent Gaussians. Both are scored here against the DENSE M assembled in
# R from the same groups and inverted with solve() -- an arbiter outside the
# implementation rather than a second copy of it.
#
# Nothing below is Type-IV-specific. The groups are the caller's, so the same
# storage carries a single sum-to-zero direction over an ICAR / RW field
# (one group, the whole block) and the interaction's S + T margins alike.

# Dense M = D + U Lambda U' over the block, from the CSR the C++ side takes.
lr_dense_mass <- function(inv_mass_diag, start, group_ptr, group_idx, lambda) {
  n <- length(inv_mass_diag)
  nb <- n - start
  M <- diag(1 / inv_mass_diag[(start + 1L):n], nb, nb)
  for (g in seq_along(lambda)) {
    len <- group_ptr[g + 1L] - group_ptr[g]
    u <- numeric(nb)
    if (len > 0) u[group_idx[group_ptr[g] + seq_len(len)] + 1L] <- 1
    M <- M + lambda[g] * tcrossprod(u)
  }
  M
}

# What M^-1 p is, coordinate by coordinate: the plain diagonal outside the
# block, the dense solve inside it.
lr_expected_inv <- function(inv_mass_diag, start, M, p) {
  n <- length(inv_mass_diag)
  out <- inv_mass_diag * p
  blk <- (start + 1L):n
  out[blk] <- as.numeric(solve(M, p[blk]))
  out
}

# Three group layouts, each exercising something the storage has to survive.
lr_cases <- function() {
  list(
    # Overlapping groups of different sizes, one coordinate in no group at all,
    # and a nontrivial `start` so the block is not the whole parameter vector.
    irregular = list(
      inv_mass_diag = c(0.7, 2.1, 0.4, 1.3, 0.9, 2.5, 0.6, 1.1, 0.3),
      start = 2L,
      group_ptr = c(0L, 3L, 6L, 8L),
      group_idx = c(0L, 1L, 2L,   2L, 3L, 4L,   0L, 6L),
      lambda = c(12.0, 3.5, 0.8)
    ),
    # One group over the whole block: the shape a soft sum-to-zero on an
    # intrinsic field (ICAR, RW1, RW2) contributes.
    single_sum = list(
      inv_mass_diag = c(1.4, 0.5, 2.2, 0.8, 1.7, 0.6),
      start = 0L,
      group_ptr = c(0L, 6L),
      group_idx = 0:5,
      lambda = 1 / (0.001 * 6)^2
    ),
    # The Type-IV margins at their own precisions: S = 3, T = 2 laid out as
    # s * T + t, and lambdas four orders of magnitude above the diagonal.
    margins = list(
      inv_mass_diag = c(0.9, 1.2, 0.4, 2.0, 0.7, 1.5),
      start = 0L,
      group_ptr = c(0L, 2L, 4L, 6L, 9L, 12L),
      group_idx = c(0L, 1L,  2L, 3L,  4L, 5L,   0L, 2L, 4L,   1L, 3L, 5L),
      lambda = c(rep(1 / (0.001 * 2)^2, 3), rep(1 / (0.001 * 3)^2, 2))
    )
  )
}

test_that("Woodbury reproduces the dense inverse on every group layout", {
  for (nm in names(lr_cases())) {
    cs <- lr_cases()[[nm]]
    n <- length(cs$inv_mass_diag)
    set.seed(11)
    p <- stats::rnorm(n)
    res <- cpp_test_lowrank_mass_apply(
      cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, cs$lambda, p)
    expect_true(res$ok, info = nm)
    expect_identical(res$rank, length(cs$lambda), info = nm)

    M <- lr_dense_mass(cs$inv_mass_diag, cs$start, cs$group_ptr,
                       cs$group_idx, cs$lambda)
    want <- lr_expected_inv(cs$inv_mass_diag, cs$start, M, p)
    expect_equal(res$inv_mass_times_p, want, tolerance = 1e-10, info = nm)
  }
})

test_that("the kinetic energy is the same quadratic form", {
  for (nm in names(lr_cases())) {
    cs <- lr_cases()[[nm]]
    n <- length(cs$inv_mass_diag)
    set.seed(23)
    p <- stats::rnorm(n)
    res <- cpp_test_lowrank_mass_apply(
      cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, cs$lambda, p)
    M <- lr_dense_mass(cs$inv_mass_diag, cs$start, cs$group_ptr,
                       cs$group_idx, cs$lambda)
    want <- 0.5 * sum(p * lr_expected_inv(cs$inv_mass_diag, cs$start, M, p))
    expect_equal(res$kinetic_energy, want, tolerance = 1e-10, info = nm)
  }
})

test_that("the fused drift and inv_mass_times_p agree", {
  # The zero-allocation NUTS loop applies the metric through apply_drift and
  # the rest of the sampler through inv_mass_times_p. Two application sites
  # that can drift apart is what this pins.
  for (nm in names(lr_cases())) {
    cs <- lr_cases()[[nm]]
    n <- length(cs$inv_mass_diag)
    set.seed(37)
    p <- stats::rnorm(n)
    res <- cpp_test_lowrank_mass_apply(
      cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, cs$lambda, p,
      coeff = 0.37)
    expect_equal(res$drift, 0.37 * res$inv_mass_times_p,
                 tolerance = 1e-12, info = nm)
  }
})

test_that("a rank-0 or malformed term is refused rather than approximated", {
  cs <- lr_cases()$irregular
  n <- length(cs$inv_mass_diag)
  p <- rep(1, n)

  # No groups at all.
  expect_false(cpp_test_lowrank_mass_apply(
    cs$inv_mass_diag, cs$start, 0L, integer(), numeric(), p)$ok)
  # A non-positive weight: Lambda^-1 is not defined there.
  bad <- cs$lambda; bad[2] <- 0
  expect_false(cpp_test_lowrank_mass_apply(
    cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, bad, p)$ok)
  # A coordinate index past the end of the block.
  bad_idx <- cs$group_idx; bad_idx[1] <- 99L
  expect_false(cpp_test_lowrank_mass_apply(
    cs$inv_mass_diag, cs$start, cs$group_ptr, bad_idx, cs$lambda, p)$ok)
  # A CSR whose pointer does not close on the index array.
  expect_false(cpp_test_lowrank_mass_apply(
    cs$inv_mass_diag, cs$start, c(0L, 3L, 6L, 7L), cs$group_idx,
    cs$lambda, p)$ok)
})

test_that("the momentum draw has covariance M", {
  # p = D^(1/2) z1 + U Lambda^(1/2) z2 with z1, z2 independent is N(0, M)
  # exactly. Whitening by chol(M) is scale-free, so the Type-IV lambdas -- four
  # orders of magnitude above the diagonal -- are scored on the same footing as
  # the diagonal directions rather than swamping them.
  cs <- lr_cases()$margins
  n <- length(cs$inv_mass_diag)
  M <- lr_dense_mass(cs$inv_mass_diag, cs$start, cs$group_ptr,
                     cs$group_idx, cs$lambda)
  n_draws <- 200000L
  draws <- cpp_test_lowrank_mass_momentum(
    cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, cs$lambda,
    n_draws, seed = 4)
  Z <- draws %*% solve(chol(M))          # rows ~ N(0, I) if the draw is right
  emp <- crossprod(Z) / n_draws
  # A covariance entry at this draw count has Monte Carlo sd ~ 3e-3, so 0.03 is
  # roughly ten of them: a construction error shows as a whole entry, not as a
  # third decimal.
  expect_lt(max(abs(emp - diag(ncol(Z)))), 0.03)
  expect_lt(max(abs(colMeans(draws) / sqrt(diag(M)))), 0.02)
})

test_that("the block is the only part of the metric the term touches", {
  cs <- lr_cases()$irregular
  n <- length(cs$inv_mass_diag)
  set.seed(5)
  p <- stats::rnorm(n)
  res <- cpp_test_lowrank_mass_apply(
    cs$inv_mass_diag, cs$start, cs$group_ptr, cs$group_idx, cs$lambda, p)
  head_idx <- seq_len(cs$start)
  expect_equal(res$inv_mass_times_p[head_idx],
               (cs$inv_mass_diag * p)[head_idx], tolerance = 1e-12)
})

test_that("make_margin_mass_term lays down the row and column sums", {
  S <- 3L; T <- 4L
  lam_row <- 1 / (0.001 * T)^2
  lam_col <- 1 / (0.001 * S)^2
  term <- cpp_test_margin_mass_term(S, T, lam_row, lam_col,
                                    rep(0.5, S * T), start = 7L)
  expect_true(term$ok)
  expect_identical(term$rank, S + T)
  expect_identical(term$start, 7L)
  expect_identical(term$n, S * T)
  expect_equal(term$lambda, c(rep(lam_row, S), rep(lam_col, T)))

  grp <- lapply(seq_len(S + T), function(g) {
    term$group_idx[term$group_ptr[g] + seq_len(term$group_ptr[g + 1L] -
                                                 term$group_ptr[g])]
  })
  # delta is indexed s * T + t, so a row group is one spatial unit's T times
  # and a column group is one time's S spatial units.
  for (s in seq_len(S)) {
    expect_identical(sort(grp[[s]]), as.integer((s - 1L) * T + seq_len(T) - 1L))
  }
  for (tt in seq_len(T)) {
    expect_identical(sort(grp[[S + tt]]),
                     as.integer(seq(tt - 1L, S * T - 1L, by = T)))
  }

  # A block length that does not match S * T is a layout error, not a shorter
  # term: the builder returns a rank-0 term, which never factorizes.
  short <- cpp_test_margin_mass_term(S, T, lam_row, lam_col, rep(0.5, S * T - 1))
  expect_false(short$ok)
  expect_identical(short$rank, 0L)
})
