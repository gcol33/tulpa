test_that("ST GP densities and coordinates agree with dense Vecchia covariances", {
  S <- 3L; nt <- 2L; n <- S * nt
  coords <- cbind(rep(c(0, .4, .9), each = nt), rep(c(.2, -.1, .5), each = nt))
  times <- rep(c(0, .7), S)
  nn <- n - 1L
  neighbors <- matrix(0L, n, nn)
  ds <- dt <- matrix(0, n, nn)
  for (i in 2:n) {
    j <- seq_len(i - 1L)
    neighbors[i, j] <- j
    ds[i, j] <- sqrt(rowSums((coords[j, , drop = FALSE] - matrix(coords[i, ], length(j), 2, byrow = TRUE))^2))
    dt[i, j] <- abs(times[j] - times[i])
  }
  X <- cbind(1, seq(-.5, .5, length.out = n)); y <- sin(seq_len(n))
  B <- cbind(kronecker(matrix(1, S, 1), diag(nt)) / (.001 * S),
    kronecker(diag(S), matrix(1, nt, 1)) / (.001 * nt))
  cases <- rbind(c(0L, 0L), c(1L, 0L), c(2L, 0L), c(3L, 0L), c(0L, 2L))
  for (case in seq_len(nrow(cases))) for (nc in 0:1) {
    covariance <- cases[case, 1L]; nonsep <- cases[case, 2L]
    spec <- list(type = if (nonsep == 2L) 6L else 5L, temporal_type = 1L,
      cyclic = FALSE, n_spatial = S, n_times = nt, st_flat = seq_len(n),
      parameterization = nc, adj_row_ptr = integer(), adj_col_idx = integer(),
      coords = as.numeric(t(coords)), time_values = times, nn = nn,
      nn_idx = as.integer(t(neighbors)), nn_dist_space = as.numeric(t(ds)),
      nn_dist_time = as.numeric(t(dt)), nn_order = 0:(n - 1L),
      nonsep = nonsep, cov_space = covariance, cov_time = covariance)
    r <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec)
    q <- r$x; tau <- exp(q[3L]); ps <- exp(q[4L]); pt <- exp(q[5L])
    cor <- function(distance, phi) {
      z <- distance / phi
      switch(covariance + 1L, exp(-z), (1 + sqrt(3) * z) * exp(-sqrt(3) * z),
        exp(-.5 * z^2), ifelse(z < 1, 1 - 1.5 * z + .5 * z^3, 0))
    }
    h <- as.matrix(dist(coords)); u <- abs(outer(times, times, "-"))
    C <- if (nonsep == 2L) {
      base <- 1 + (u / pt)^2
      exp(-(h / ps) / sqrt(base)) / (tau * base)
    } else cor(h, ps) * cor(u, pt) / tau
    A <- diag(n); variance <- rep(1 / tau, n)
    for (i in 2:n) {
      j <- seq_len(i - 1L)
      weight <- solve(C[j, j, drop = FALSE] + diag(1e-8, length(j)), C[j, i])
      A[i, j] <- -weight
      variance[i] <- max(1e-10, 1 / tau - sum(C[j, i] * weight))
    }
    inverse <- solve(A)
    Sigma <- inverse %*% diag(variance) %*% t(inverse)
    z <- tail(q, n)
    field <- if (nc == 1L) drop(inverse %*% (sqrt(variance) * z)) else z
    density <- if (nc == 1L) sum(dnorm(z, log = TRUE)) else
      -.5 * (n * log(2 * pi) + as.numeric(determinant(Sigma)$modulus) +
        drop(crossprod(field, solve(Sigma, field))))
    normalizer <- .5 * as.numeric(determinant(diag(ncol(B)) +
      crossprod(B, Sigma %*% B) + diag(1e-8, ncol(B)))$modulus)
    prior <- -.5 * sum(q[1:2]^2) / 2.5^2 + log(-log(.01) / 2) - .5 * q[3L] -
      (-log(.01)) * exp(-.5 * q[3L]) + sum(q[4:5])
    expected <- prior + density - .5 * sum(crossprod(B, field)^2) + normalizer +
      sum(dnorm(y, drop(X %*% q[1:2]) + field, 1, log = TRUE))
    expect_equal(r$raw_lp, expected, tolerance = 1e-9)
    expect_equal(tail(r$stored, n), field, tolerance = 1e-10)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 5e-5)
  }
})