# The compiled BRASIL search (src/brasil.h) against the R reference
# implementation it was ported from (.spde_rational_roots_r, R/brasil.R).
#
# The fractional-SPDE assembler needs a fresh rational approximation for every
# (range, sigma) cell the outer integrator visits -- 1102 of them, essentially
# all distinct, for a single n = 40 fit -- and at R speed that is ~0.55 s
# apiece, which was the entire run time of a fractional fit (gcol33/tulpa#818).
# The R version stays as the oracle; this pins the port to it.
#
# The two differ only in the linear algebra R outsources to LAPACK: the Loewner
# nullspace comes from a Jacobi SVD rather than dgesdd, and the secular roots
# from a companion matrix rather than polyroot. Both are defined up to
# invariances the consumers already have (the barycentric weights up to scale,
# the roots up to order), so the comparison sorts the roots.

test_that("the compiled rational roots match the R oracle", {
  for (order in 1:4) {
    for (beta in c(0.375, 0.625, 0.875, 1.25, 1.75)) {
      for (s in c(0.4, 0.1, 0.01, 1e-3)) {
        cpp <- tulpa:::.spde_rational_roots(order, beta, s)
        ref <- tulpa:::.spde_rational_roots_r(order, beta, s)
        info <- sprintf("order = %d, beta = %g, spectrum_ratio = %g",
                        order, beta, s)
        expect_equal(length(cpp$rb), length(ref$rb), info = info)
        expect_equal(length(cpp$rc), length(ref$rc), info = info)
        expect_equal(sort(cpp$rb), sort(ref$rb), tolerance = 1e-6, info = info)
        expect_equal(sort(cpp$rc), sort(ref$rc), tolerance = 1e-6, info = info)
        expect_equal(cpp$scale,    ref$scale,    tolerance = 1e-6, info = info)
        expect_equal(cpp$m_beta,   ref$m_beta,                     info = info)
        expect_equal(cpp$beta_rem, ref$beta_rem, tolerance = 1e-12, info = info)
        # The minimax error is the quantity the approximation is chosen to
        # minimise, so agreeing on it is the statement that the two searches
        # landed on the same rational, not merely on nearby roots.
        expect_equal(cpp$error, ref$error, tolerance = 1e-6, info = info)
      }
    }
  }
})

test_that("the compiled search equioscillates to its tolerance", {
  # `converged` means the local error maxima agree to `tol` -- the
  # equioscillation property that defines the minimax rational. A search that
  # silently ran out of iterations would return a worse approximation with no
  # signal, so the flag is asserted rather than assumed.
  for (s in c(0.4, 0.05)) {
    rr <- tulpa:::.spde_rational_roots(2L, 1.25, s)
    expect_true(rr$converged, info = sprintf("spectrum_ratio = %g", s))
    expect_lte(rr$deviation, 1e-7)
  }
})

test_that("the approximation the port defines is the R oracle's function", {
  # Roots and weights carry invariances; the approximating FUNCTION does not.
  # This reads both implementations at the same points.
  for (s in c(0.3, 0.02)) {
    x   <- seq(s, 1, length.out = 41)
    cpp <- cpp_brasil_approx(x, s, 1, 3L, -0.75)
    ref <- tulpa:::.brasil(function(t) t^(-0.75), c(s, 1), 3L, tol = 1e-7)
    expect_equal(as.numeric(cpp), tulpa:::.bary_eval(ref$br, x),
                 tolerance = 1e-7, info = sprintf("spectrum_ratio = %g", s))
    # ... and both approximate the function they were asked for.
    expect_lt(max(abs(as.numeric(cpp) - x^(-0.75))) / max(x^(-0.75)), 0.05)
  }
})

test_that("the compiled entry refuses an out-of-domain request", {
  expect_error(tulpa:::.spde_rational_roots(0L, 1.25, 0.1), "order must be")
  expect_error(tulpa:::.spde_rational_roots(2L, 1.25, 0), "spectrum_ratio")
  expect_error(tulpa:::.spde_rational_roots(2L, 1.25, 1), "spectrum_ratio")
})
