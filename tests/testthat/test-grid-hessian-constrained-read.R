# The per-cell fixed-effect covariance and field variances a nested fit retains
# (`.nl_attach_grid_hessians()`) are the block of Cov(x | A x = 0) over the
# cell's stored precision Q. The sparse joint Newton folds the sum-to-zero pins
# at solve time and leaves them off the stored Q, so two intrinsic fields
# loading the same rows leave their opposite shift (+c on one level, -c on the
# other) with no curvature at all: Q is singular there, and only the
# constraint identifies it. The read has to be the conditional covariance all
# the same -- B (B' Q B)^-1 B' for B a basis of null(A) -- and not a
# difference of terms of order 1 / round-off.

.gh_csc <- function(Q) {
  L <- methods::as(methods::as(Matrix::tril(Matrix::Matrix(Q, sparse = TRUE)),
                               "generalMatrix"), "CsparseMatrix")
  list(p = L@p, i = L@i, x = L@x)
}

.gh_null_cov <- function(Q, A) {
  B <- MASS::Null(t(A))
  B %*% solve(crossprod(B, Q %*% B), t(B))
}

test_that("the grid read conditions a precision singular along its constraints", {
  set.seed(4)
  n_a <- 5L; n_b <- 4L; n <- 60L
  ia <- sample(n_a, n, replace = TRUE); ib <- sample(n_b, n, replace = TRUE)
  x <- rnorm(n)
  Xt <- cbind(1, x, outer(ia, seq_len(n_a), "==") * 1,
              outer(ib, seq_len(n_b), "==") * 1)
  n_x <- ncol(Xt)
  Ra <- crossprod(diff(diag(n_a))); Rb <- crossprod(diff(diag(n_b)))
  P <- matrix(0, n_x, n_x)
  P[1:2, 1:2] <- diag(1 / 2.5^2, 2)
  P[2 + seq_len(n_a), 2 + seq_len(n_a)] <- 3 * Ra
  P[2 + n_a + seq_len(n_b), 2 + n_a + seq_len(n_b)] <- 2 * Rb
  Q <- crossprod(Xt, runif(n, 0.5, 2) * Xt) + P
  # The opposite shift of the two levels is an exact null direction of Q.
  v <- c(0, 0, rep(1, n_a), rep(-1, n_b))
  expect_lt(max(abs(Q %*% v)), 1e-10)

  constr <- list(2L + seq_len(n_a), 2L + n_a + seq_len(n_b))
  A <- t(vapply(constr, tulpa:::.constraint_row_dense, numeric(n_x),
                n_x = n_x))
  cs <- .gh_csc(Q)
  res <- list(Q_csc_p_per_grid = list(cs$p), Q_csc_i_per_grid = list(cs$i),
              Q_csc_x_per_grid = list(cs$x), Q_csc_n = n_x,
              modes = matrix(0, 1L, n_x), constraint_cols = constr)
  var_cols <- 2L + n_a + seq_len(n_b)
  out <- tulpa:::.nl_attach_grid_hessians(res, 2L, var_cols = var_cols,
                                          units = list())
  ref <- .gh_null_cov(Q, A)
  expect_equal(solve(out$grid_hessians[[1L]]), ref[1:2, 1:2],
               tolerance = 1e-10)
  expect_equal(as.numeric(out$grid_field_var), diag(ref)[var_cols],
               tolerance = 1e-10)
  expect_equal(out$grid_field_cov[[1L]], ref[var_cols, var_cols],
               tolerance = 1e-10)
})

test_that("a constrained cell draw has the conditional covariance", {
  set.seed(5)
  n_a <- 4L; n_b <- 3L; n <- 40L
  ia <- sample(n_a, n, replace = TRUE); ib <- sample(n_b, n, replace = TRUE)
  Xt <- cbind(1, outer(ia, seq_len(n_a), "==") * 1,
              outer(ib, seq_len(n_b), "==") * 1)
  n_x <- ncol(Xt)
  P <- matrix(0, n_x, n_x)
  P[1, 1] <- 1 / 2.5^2
  P[1 + seq_len(n_a), 1 + seq_len(n_a)] <- 2 * crossprod(diff(diag(n_a)))
  P[1 + n_a + seq_len(n_b), 1 + n_a + seq_len(n_b)] <-
    crossprod(diff(diag(n_b)))
  Q <- crossprod(Xt, Xt) + P
  # Singular along the opposite shift of the two levels.
  v <- c(0, rep(1, n_a), rep(-1, n_b))
  expect_lt(max(abs(Q %*% v)), 1e-10)
  constr <- list(1L + seq_len(n_a), 1L + n_a + seq_len(n_b))
  A <- t(vapply(constr, tulpa:::.constraint_row_dense, numeric(n_x),
                n_x = n_x))
  cs <- .gh_csc(Q)
  ref <- .gh_null_cov(Q, A)
  # The draw map on the identity: X X' is the covariance the draws carry.
  X <- tulpa:::cpp_constrained_cell_draw(cs$p, cs$i, cs$x, n_x,
                                         tulpa:::.constraint_rows_for_cpp(constr),
                                         diag(n_x))
  expect_lt(max(abs(tcrossprod(X) - ref)) / max(abs(ref)), 1e-10)
  expect_lt(max(abs(A %*% X)), 1e-10)
  set.seed(1)
  z <- tulpa:::.joint_draw_cell(cs$p, cs$i, cs$x, n_x, constr, 20000L)
  expect_lt(max(abs(A %*% z)), 1e-10)
  ok <- diag(ref) > 1e-8
  expect_lt(max(abs(apply(z, 1, var)[ok] / diag(ref)[ok] - 1)), 0.05)
})

test_that("an unconstrained cell draw has the inverse precision", {
  set.seed(6)
  M <- matrix(rnorm(36), 6, 6)
  Q <- crossprod(M) + diag(6)
  cs <- .gh_csc(Q)
  X <- tulpa:::cpp_constrained_cell_draw(cs$p, cs$i, cs$x, 6L, list(), diag(6))
  expect_lt(max(abs(tcrossprod(X) - solve(Q))), 1e-12)
})
