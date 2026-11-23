# The joint door's free-covariance placement searches a correlated block in the
# log marginal SDs and the atanh canonical partial correlations, and maps every
# point back to the log-Cholesky coordinates. The map has to be a bijection and
# its log-Jacobian has to be the log-determinant of its derivative, or the
# search's objective carries the error.

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

.cpc_to_logchol <- function(v, nc) {
  b <- tulpa:::.re_cpc_batch_to_logchol(matrix(v, 1L), nc)
  list(theta = as.numeric(b$theta), log_jac = b$log_jac)
}

test_that("the partial-correlation map inverts the log-Cholesky map", {
  set.seed(11)
  for (nc in 2:4) {
    k <- nc * (nc + 1L) / 2L
    for (rep in 1:5) {
      v <- c(rnorm(nc, 0, 0.7), rnorm(k - nc, 0, 1.2))
      th <- .cpc_to_logchol(v, nc)$theta
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
      J <- .num_jac_place(function(x) .cpc_to_logchol(x, nc)$theta, v)
      expect_equal(.cpc_to_logchol(v, nc)$log_jac,
                   as.numeric(determinant(J, logarithm = TRUE)$modulus),
                   tolerance = 1e-6)
    }
  }
})
