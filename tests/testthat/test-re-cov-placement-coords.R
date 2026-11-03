# The nested RE-covariance door lays a correlated block's nodes in the log
# marginal SDs and the atanh canonical partial correlations, and maps every node
# back to the log-Cholesky coordinates. The map has to be a bijection and its
# log-Jacobian has to be the log-determinant of its derivative, or every node
# weight carries the error.

.num_jac_place <- function(f, x, h = 1e-6) {
  m <- length(f(x))
  J <- matrix(0, m, length(x))
  for (j in seq_along(x)) {
    xp <- x; xp[j] <- xp[j] + h
    xm <- x; xm[j] <- xm[j] - h
    J[, j] <- (f(xp) - f(xm)) / (2 * h)
  }
  J
}

test_that("the partial-correlation map inverts the log-Cholesky map", {
  set.seed(11)
  for (nc in 2:4) {
    k <- nc * (nc + 1L) / 2L
    for (rep in 1:5) {
      v <- c(rnorm(nc, 0, 0.7), rnorm(k - nc, 0, 1.2))
      th <- tulpa:::.re_cpc_block_to_logchol(v, nc)$theta
      expect_equal(tulpa:::.re_logchol_block_to_cpc(th, nc), v, tolerance = 1e-10)
      # Sigma from the map has the requested marginal SDs.
      L <- tulpa:::.re_logchol_to_L(th, nc)
      expect_equal(0.5 * log(rowSums(L^2)), v[seq_len(nc)], tolerance = 1e-12)
    }
  }
})

test_that("the map's log-Jacobian is the log-determinant of its derivative", {
  set.seed(12)
  for (nc in 2:4) {
    k <- nc * (nc + 1L) / 2L
    for (rep in 1:5) {
      v <- c(rnorm(nc, 0, 0.7), rnorm(k - nc, 0, 1.2))
      J <- .num_jac_place(function(x) tulpa:::.re_cpc_block_to_logchol(x, nc)$theta, v)
      expect_equal(tulpa:::.re_cpc_block_to_logchol(v, nc)$log_jac,
                   as.numeric(determinant(J, logarithm = TRUE)$modulus),
                   tolerance = 1e-6)
    }
  }
})

test_that("a stacked layout maps only its correlated blocks", {
  layout <- list(list(nc = 2L, full = TRUE, k = 3L),
                 list(nc = 1L, full = FALSE, k = 1L),
                 list(nc = 2L, full = FALSE, k = 2L))
  V <- rbind(c(0.1, -0.4, 0.3, 0.2, -0.5, 0.6),
             c(-1, -3, -0.2, 1.1, 0.4, -0.1))
  tr <- tulpa:::.re_cov_place_to_theta(V, layout)
  expect_equal(tr$theta[, 4:6], V[, 4:6])
  for (r in 1:2) {
    b <- tulpa:::.re_cpc_block_to_logchol(V[r, 1:3], 2L)
    expect_equal(tr$theta[r, 1:3], b$theta)
    expect_equal(tr$log_jac[r], b$log_jac)
    expect_equal(tr$log_jac[r], V[r, 2])         # c = 2: log|J| = log sigma_2
    expect_equal(tulpa:::.re_cov_theta_to_place(tr$theta[r, ], layout), V[r, ],
                 tolerance = 1e-10)
  }
  expect_false(tulpa:::.re_cov_has_full_block(layout[2:3]))
})

