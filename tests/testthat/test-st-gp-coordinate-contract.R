test_that("ST GP densities and coordinates agree with dense Vecchia covariances", {
  S <- 3L; nt <- 2L; n <- S * nt
  coords <- cbind(rep(c(0, .4, .9), each = nt), rep(c(.2, -.1, .5), each = nt))
  times <- rep(c(0, .7), S)
  nn <- n - 1L
  for (order in list(seq_len(n), c(3L, 1L, 5L, 2L, 6L, 4L))) {
  neighbors <- matrix(0L, n, nn)
  ds <- dt <- matrix(0, n, nn)
  for (i in 2:n) {
    j <- seq_len(i - 1L)
    neighbors[i, j] <- j
    ds[i, j] <- sqrt(rowSums((coords[order[j], , drop = FALSE] - matrix(coords[order[i], ], length(j), 2, byrow = TRUE))^2))
    dt[i, j] <- abs(times[order[j]] - times[order[i]])
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
      nn_dist_time = as.numeric(t(dt)), nn_order = as.integer(order - 1L),
      nonsep = nonsep, cov_space = covariance, cov_time = covariance)
    r <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec)
    q <- r$x; tau <- exp(q[3L]); ps <- .01 + 9.99 * plogis(q[4L]); pt <- .01 + 9.99 * plogis(q[5L])
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
      j <- order[seq_len(i - 1L)]; at <- order[i]
      weight <- solve(C[j, j, drop = FALSE] + diag(1e-8, length(j)), C[j, at])
      A[at, j] <- -weight
      variance[at] <- max(1e-10, 1 / tau - sum(C[j, at] * weight))
    }
    inverse <- solve(A)
    Sigma <- inverse %*% diag(variance) %*% t(inverse)
    z <- tail(q, n)
    conditioned <- solve(solve(Sigma) + tcrossprod(B))
    field <- if (nc == 1L) tail(r$stored, n) else z
    density <- if (nc == 1L) sum(dnorm(z, log = TRUE)) else
      -.5 * (n * log(2 * pi) + as.numeric(determinant(Sigma)$modulus) +
        drop(crossprod(field, solve(Sigma, field))))
    normalizer <- if (nc == 1L) 0 else .5 * as.numeric(determinant(diag(ncol(B)) +
      crossprod(B, Sigma %*% B))$modulus)
    prior <- -.5 * sum(q[1:2]^2) / 2.5^2 + log(-log(.01) / 2) - .5 * q[3L] -
      (-log(.01)) * exp(-.5 * q[3L]) + sum(log(plogis(q[4:5])) + log1p(-plogis(q[4:5])))
    penalty <- if (nc == 1L) 0 else -.5 * sum(crossprod(B, field)^2)
    expected <- prior + density + penalty + normalizer +
      sum(dnorm(y, drop(X %*% q[1:2]) + field, 1, log = TRUE))
    expect_equal(r$raw_lp, expected, tolerance = 1e-9)
    if (nc == 0L) expect_equal(tail(r$stored, n), field, tolerance = 1e-10)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 5e-5)
    expect_equal(r$st_normalizer_gradient, r$st_normalizer_numerical, tolerance = 5e-5)
    if (nc == 1L) {
      expect_equal(r$st_inverse, z, tolerance = 1e-8)
      expect_equal(r$st_log_jacobian, .5 * as.numeric(determinant(conditioned)$modulus), tolerance = 1e-8)
      factor <- matrix(0, n, n)
      for (j in seq_len(n)) {
        point <- q
        point[tail(seq_along(q), n)] <- 0
        point[length(q) - n + j] <- 1
        column <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec, point = point)
        factor[, j] <- tail(column$stored, n)
      }
      expect_equal(tcrossprod(factor), conditioned, tolerance = 1e-8)
      expect_equal(drop(crossprod(field, solve(conditioned, field))), sum(z^2), tolerance = 1e-8)
      if (covariance == 1L || nonsep == 2L) for (precision in c(8, 26)) {
        point <- q
        point[3L] <- precision
        point[tail(seq_along(q), n)] <- 0
        stress <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec, point = point)
        error <- abs(stress$st_normalizer_gradient - stress$st_normalizer_numerical) /
          pmax(1, abs(stress$st_normalizer_numerical))
        expect_lt(max(error), 1e-7,
          label = paste("covariance", covariance, "nonsep", nonsep, "precision", precision))
      }
    }
  }
  }
})
