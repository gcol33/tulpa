test_that("HSGP ST coefficients match dense temporal precisions", {
  set.seed(196)
  nt <- 4L; locations <- 6L; m <- 2L; M <- m^2L
  coords <- cbind(rep(runif(locations), each = nt), rep(runif(locations), each = nt))
  time <- rep(seq_len(nt), locations)
  n <- nrow(coords); X <- cbind(1, rnorm(n)); y <- rnorm(n)
  basis <- tulpa:::cpp_hsgp_basis_2d(coords, m, 1.5)
  for (type in 1:3) for (cyclic in if (type == 3L) FALSE else c(FALSE, TRUE)) for (coordinate in 0:1) for (kernel in 0:2) {
    ar1 <- type == 3L
    spec <- list(type = 4L, temporal_type = type, cyclic = cyclic,
      n_spatial = M, n_times = nt, st_flat = rep(1L, n), parameterization = coordinate,
      adj_row_ptr = integer(), adj_col_idx = integer(),
      hsgp_coords = as.numeric(t(coords)), m = m, t_idx = time, kernel = kernel)
    q <- .03 * sin(seq_len(2L + 3L + ar1 + M * nt))
    log_sigma <- 4L + ar1; log_length <- 5L + ar1
    q[log_length] <- -1.4
    r <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec, point = q)
    tau <- exp(q[3L]); sigma2 <- exp(q[log_sigma]); ell <- exp(q[log_length])
    rho <- if (ar1) 2 * plogis(q[4L]) - 1 else 0
    D <- diff(diag(nt), differences = type)
    if (cyclic) {
      D <- diag(nt) - diag(nt)[c(nt, seq_len(nt - 1L)), ]
      if (type == 2L) D <- D %*% D
    }
    if (ar1) {
      D <- diag(nt); D[cbind(2:nt, 1:(nt - 1L))] <- -rho
      D[1L, 1L] <- sqrt(1 - rho^2)
    }
    Qt <- crossprod(D)
    rank <- if (ar1) nt else nt - if (cyclic) 1L else type
    coefficients <- matrix(tail(r$stored, M * nt), M, nt, byrow = TRUE)
    density <- 0
    for (j in seq_len(M)) {
      frequency <- basis$lambda_eig[j]
      spectrum <- max(1e-10, switch(kernel + 1L,
        sigma2 * 2 * pi * ell^2 * exp(-.5 * ell^2 * frequency),
        sigma2 * (18 * pi * sqrt(3)) / ell^3 * (3 / ell^2 + frequency)^(-2.5),
        sigma2 * (250 * pi * sqrt(5)) / ell^5 * (5 / ell^2 + frequency)^(-3.5)))
      precision <- tau / spectrum
      H <- precision * Qt + matrix(1 / (.001 * nt)^2, nt, nt)
      normalizer <- if (ar1) .5 * as.numeric(determinant(H)$modulus) else .5 * rank * log(precision)
      if (coordinate == 0L) {
        density <- density + normalizer - .5 * drop(crossprod(coefficients[j, ], H %*% coefficients[j, ]))
      } else {
        raw <- matrix(tail(q, M * nt), M, nt, byrow = TRUE)
        normal <- nt - if (type == 2L && !cyclic) 1L else 0L
        density <- density - .5 * sum(raw[j, seq_len(normal)]^2)
        F <- matrix(0, nt, nt)
        for (k in seq_len(nt)) {
          point <- q; point[(length(q) - M * nt + 1L):length(q)] <- 0
          point[length(q) - M * nt + (j - 1L) * nt + k] <- 1
          unit <- tulpa:::cpp_test_field_coordinates(y, X, st_spec = spec, point = point)
          F[, k] <- matrix(tail(unit$stored, M * nt), M, nt, byrow = TRUE)[j, ]
        }
        if (normal < nt) {
          linear <- seq_len(nt) - mean(seq_len(nt))
          U <- qr.Q(qr(cbind(linear, diag(nt)[, -nt])))[, -1L, drop = FALSE]
          covariance <- U %*% solve(crossprod(U, H %*% U)) %*% t(U)
          expect_equal(F[, nt], linear, tolerance = 1e-10)
        } else covariance <- solve(H)
        expect_equal(tcrossprod(F[, seq_len(normal), drop = FALSE]), covariance, tolerance = 1e-8)
      }
    }
    rate <- -log(.01)
    prior <- -.5 * sum(q[1:2]^2) / 2.5^2 + log(rate / 2) - .5 * q[3L] - rate * exp(-.5 * q[3L]) +
      log(rate / 2) + .5 * q[log_sigma] - rate * exp(.5 * q[log_sigma]) - .5 * q[log_length]^2
    if (ar1) prior <- prior + log(plogis(q[4L])) + log1p(-plogis(q[4L]))
    effect <- vapply(seq_len(n), function(i) sum(basis$phi_basis[i, ] * coefficients[, time[i]]), numeric(1))
    expected <- prior + density + sum(dnorm(y, drop(X %*% q[1:2]) + effect, 1, log = TRUE))
    expect_equal(r$raw_lp, expected, tolerance = 1e-9)
    if (coordinate == 1L) expect_equal(r$st_inverse, tail(q, M * nt), tolerance = 1e-10)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 5e-5)
  }
})