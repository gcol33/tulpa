test_that("HSGP covariance and domain commute with coordinate scaling", {
  skip_on_cran_budget()
  set.seed(98)
  xy <- rbind(c(0, 0), c(1, 1), matrix(runif(24), 12, 2))
  m <- 15L; c <- 2; ell <- 0.2
  domain <- tulpa:::cpp_hsgp_domain_2d(xy, c)
  basis <- tulpa:::cpp_hsgp_basis_2d(xy, m, c)
  for (kernel in 0:2) {
    a <- sweep(basis$phi_basis, 2, tulpa:::cpp_hsgp_sqrt_spectral(
      basis$lambda_eig, 1.2, ell, kernel), `*`)
    for (scale in c(1e-3, 1e3)) {
      scaled_domain <- tulpa:::cpp_hsgp_domain_2d(scale * xy, c)
      expect_equal(scaled_domain / scale, domain, tolerance = 1e-12)
      b <- tulpa:::cpp_hsgp_basis_2d(scale * xy, m, c)
      aa <- sweep(b$phi_basis, 2, tulpa:::cpp_hsgp_sqrt_spectral(
        b$lambda_eig, 1.2, scale * ell, kernel), `*`)
      expect_equal(tcrossprod(aa), tcrossprod(a), tolerance = 1e-10)
    }
  }
})

test_that("zero-span coordinate diagnostics have a finite window", {
  skip_on_cran_budget()
  xy <- cbind(lon = seq(0, 1, length.out = 21), lat = 0)
  S <- tulpa:::.hsgp_half_widths(xy)
  expect_equal(S, c(lon = 0.5, lat = 0))
  window <- tulpa:::.hsgp_lengthscale_window(9, 2, S)
  expect_true(all(is.finite(window)))
  expect_equal(unname(window["hi"]), 0)
  v <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp", scale_coords = FALSE,
                               m = 9, c = 2),
                     as.data.frame(xy))
  expect_equal(v$basis_rule$half_width, S)
  expect_equal(tulpa:::.hsgp_lengthscale_window(9, 2, c(0, 0)), c(lo = 0, hi = 0))
})
