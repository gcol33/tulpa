# Marginal-variance identity for the HSGP squared-exponential approximation
# (gcol33/tulpa#146). The HSGP field variance sum_j S(lambda_j) phi_j(x)^2 must
# reproduce the GP marginal variance sigma^2 at interior points. This holds for
# the 2-D SE spectral density S = sigma^2 (2 pi) ell^2 exp(-0.5 ell^2 lambda) on
# the 2-D basis, and fails for the 1-D form sigma^2 sqrt(2 pi) ell -- the defect
# fixed in #146. The HSGP equivalence tests separately pin the compiled kernel
# to this same 2-D density (C++ fit == the updated R oracle), so together they
# guard both that the engine uses this density and that this density is correct.

test_that("the 2-D SE spectral density reproduces the GP marginal variance", {
  set.seed(1)
  n <- 400
  coords <- cbind(runif(n), runif(n))
  sigma2 <- 0.8
  ell    <- 0.25

  b   <- tulpa:::cpp_hsgp_basis_2d(coords, 15L, 1.5)
  lam <- b$lambda_eig
  Phi <- b$phi_basis

  S_2d <- sigma2 * (2 * pi) * ell^2 * exp(-0.5 * ell^2 * lam)  # correct, D = 2
  S_1d <- sigma2 * sqrt(2 * pi) * ell * exp(-0.5 * ell^2 * lam) # the #146 bug

  var_2d <- rowSums(sweep(Phi^2, 2, S_2d, `*`))
  var_1d <- rowSums(sweep(Phi^2, 2, S_1d, `*`))

  # Interior points only (the truncated basis underestimates variance near the
  # domain boundary).
  interior <- coords[, 1] > 0.2 & coords[, 1] < 0.8 &
              coords[, 2] > 0.2 & coords[, 2] < 0.8

  # The 2-D density recovers sigma^2; the 1-D density is off by ~1/(sqrt(2pi)ell).
  expect_equal(mean(var_2d[interior]), sigma2, tolerance = 0.08)
  expect_gt(mean(var_1d[interior]), 1.3 * sigma2)
})

test_that("the compiled spectral densities are the Fourier duals of their kernels", {
  # Each density integrates over R^2 to (2 pi)^2 sigma^2, so sigma^2 is the
  # marginal variance whatever the kernel.
  for (k in 0:2) {
    f <- function(r) vapply(r, function(rr)
      tulpa:::cpp_hsgp_sqrt_spectral(rr^2, 1.7, 0.3, k)^2, numeric(1)) * r
    expect_equal(integrate(f, 0, Inf, rel.tol = 1e-10)$value / (2 * pi), 1.7,
                 tolerance = 1e-6)
  }
  # And a large basis on a widened domain reproduces each exact kernel at the
  # centre of the domain, to the truncation error of the basis.
  xy <- rbind(cbind(0.5 + (0:10) * 0.03, 0.5), c(0, 0), c(1, 1))
  b <- tulpa:::cpp_hsgp_basis_2d(xy, 50L, 3)
  r <- (0:10) * 0.03 / 0.15
  exact <- list(exp(-r^2 / 2),
                (1 + sqrt(3) * r) * exp(-sqrt(3) * r),
                (1 + sqrt(5) * r + 5 * r^2 / 3) * exp(-sqrt(5) * r))
  tol <- c(1e-10, 0.01, 0.002)
  for (k in 0:2) {
    s <- tulpa:::cpp_hsgp_sqrt_spectral(b$lambda_eig, 1, 0.15, k)
    P <- sweep(b$phi_basis, 2, s, `*`)
    C <- as.numeric(P[1:11, ] %*% P[1, ])
    expect_lt(max(abs(C - exact[[k + 1L]])), tol[k + 1L])
  }
  expect_error(tulpa:::cpp_hsgp_sqrt_spectral(1, 1, 1, 3L), "not one of")
})
