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
