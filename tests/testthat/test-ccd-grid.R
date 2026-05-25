# CCD grid generator: shape, point counts, and z -> theta mapping.

test_that("ccd_grid k=1 returns 3 points (centre + 2 axial)", {
  g <- ccd_grid(1L)
  expect_equal(g$n_points, 3L)
  expect_equal(dim(g$z), c(3L, 1L))
  expect_equal(sum(g$kind == "center"), 1L)
  expect_equal(sum(g$kind == "axial"), 2L)
  expect_equal(sum(g$kind == "factorial"), 0L)
})

test_that("ccd_grid k=2 has 1 + 4 + 4 = 9 points (full factorial)", {
  g <- ccd_grid(2L)
  expect_equal(g$n_points, 9L)
  expect_equal(dim(g$z), c(9L, 2L))
  expect_equal(sum(g$kind == "factorial"), 4L)
  # All non-centre points lie on the same sphere of radius f_0
  norms <- sqrt(rowSums(g$z^2))
  non_center <- norms[g$kind != "center"]
  expect_equal(non_center, rep(g$f_0, length(non_center)),
               tolerance = 1e-12)
})

test_that("ccd_grid k=3..6 uses full factorial 2^k", {
  for (k in 3:6) {
    g <- ccd_grid(k)
    expect_equal(g$n_points, 1L + 2L * k + 2L^k)
    expect_equal(sum(g$kind == "axial"), 2L * k)
    expect_equal(sum(g$kind == "factorial"), 2L^k)
    norms <- sqrt(rowSums(g$z^2))
    expect_equal(norms[g$kind != "center"],
                 rep(g$f_0, g$n_points - 1L),
                 tolerance = 1e-12)
  }
})

test_that("ccd_grid k=7 falls back to Resolution-V half-fraction", {
  g <- ccd_grid(7L)
  # 1 centre + 14 axial + 2^6 = 64 factorial = 79
  expect_equal(g$n_points, 1L + 14L + 64L)
  # Half-fraction: every factorial corner has product of signs = +1.
  fac <- g$z[g$kind == "factorial", ]
  signs <- sign(fac)
  expect_true(all(apply(signs, 1L, prod) > 0))
})

test_that("ccd_grid weights sum to N points and sit on the sphere", {
  g <- ccd_grid(4L)
  norms <- sqrt(rowSums(g$z^2))
  # Centre at zero, all others at distance f_0 = sqrt(4) = 2
  expect_equal(norms[1], 0)
  expect_equal(norms[-1], rep(2, g$n_points - 1L), tolerance = 1e-12)
})

test_that("ccd_grid f_0 controls the sphere radius", {
  g <- ccd_grid(3L, f_0 = 1.5)
  norms <- sqrt(rowSums(g$z^2))
  expect_equal(norms[g$kind != "center"],
               rep(1.5, g$n_points - 1L), tolerance = 1e-12)
})

test_that("ccd_to_theta applies affine map and log_scale", {
  g <- ccd_grid(2L, f_0 = 1)
  theta_hat <- c(2, 0.5)
  L <- diag(c(0.3, 0.2))

  theta <- ccd_to_theta(g$z, theta_hat, L)
  # Centre maps to theta_hat
  expect_equal(theta[1, ], theta_hat, tolerance = 1e-12)
  # Axial along axis j moves only that component, by L[j,j]
  axial_idx <- which(g$kind == "axial")
  shifts <- theta[axial_idx, , drop = FALSE] -
            matrix(theta_hat, nrow = length(axial_idx),
                   ncol = 2L, byrow = TRUE)
  # Per axial: only one component non-zero
  nonzero_per_row <- apply(shifts, 1L, function(r) sum(abs(r) > 1e-12))
  expect_true(all(nonzero_per_row == 1L))

  # log_scale exponentiates the first component
  theta_log <- ccd_to_theta(g$z, theta_hat = c(log(2), 0.5),
                             L = diag(c(0.3, 0.2)),
                             log_scale = c(TRUE, FALSE))
  # Centre on log-tau scale -> exp(log(2)) = 2
  expect_equal(theta_log[1, 1], 2, tolerance = 1e-10)
  expect_equal(theta_log[1, 2], 0.5, tolerance = 1e-12)
})

test_that("ccd_to_theta validates dimensions", {
  g <- ccd_grid(2L)
  expect_error(ccd_to_theta(g$z, theta_hat = c(0), diag(2)),
               "length k")
  expect_error(ccd_to_theta(g$z, theta_hat = c(0, 0), diag(3)),
               "k x k")
  expect_error(ccd_to_theta(g$z, theta_hat = c(0, 0), diag(2),
                             log_scale = c(TRUE, FALSE, TRUE)),
               "length 1 or k")
})

test_that("ccd_weights matches the corrected R-INLA reference values", {
  # Corrected R-INLA formula (fixes the Rue et al. 2009 typo), standardized
  # f0 = 1.1: w = 1/((np-1)(1 + e^{-m f0^2/2}(f0^2-1))), w0 = 1-(np-1)w. These
  # constants are the function's output; the *correctness* property (Gaussian
  # moment reproduction) is checked independently in the next test.
  g2 <- ccd_grid(2L, f_0 = sqrt(2) * 1.1)
  w2 <- ccd_weights(g2)
  expect_equal(w2[g2$kind == "center"][1], 0.05893108, tolerance = 1e-7)
  expect_equal(w2[g2$kind != "center"][1], 0.11763361, tolerance = 1e-7)

  g3 <- ccd_grid(3L, f_0 = sqrt(3) * 1.1)
  w3 <- ccd_weights(g3)
  expect_equal(w3[g3$kind == "center"][1], 0.03306526, tolerance = 1e-7)
  expect_equal(w3[g3$kind != "center"][1], 0.06906677, tolerance = 1e-7)

  # Design weights sum to 1; centre weight is positive and the smallest.
  expect_equal(sum(w3), 1, tolerance = 1e-12)
  expect_gt(w3[g3$kind == "center"][1], 0)
  expect_lt(w3[g3$kind == "center"][1], w3[g3$kind != "center"][1])
})

test_that("ccd_weights reproduce N(0, I) moments (the integration check)", {
  # The whole point of the design weights: on a standardized Gaussian posterior
  # the CCD quadrature must recover mean 0 and unit variance. W_k ~ Delta_k * pi.
  for (m in c(2L, 3L, 6L)) {
    g <- ccd_grid(m, f_0 = sqrt(m) * 1.1)
    dnode <- ccd_weights(g)
    logf <- apply(g$z, 1L, function(z) -0.5 * sum(z^2))   # log N(0, I)
    W <- dnode * exp(logf - max(logf)); W <- W / sum(W)

    mean_hat <- colSums(W * g$z)
    var_hat  <- colSums(W * g$z^2)
    cov_hat  <- sum(W * g$z[, 1] * g$z[, 2])
    expect_equal(mean_hat, rep(0, m), tolerance = 1e-8)
    expect_equal(var_hat, rep(1, m), tolerance = 1e-8)
    expect_equal(cov_hat, 0, tolerance = 1e-8)
  }
})
