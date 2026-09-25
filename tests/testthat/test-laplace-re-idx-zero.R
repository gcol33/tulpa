# `re_list` idx = 0 means "no group": the kernel skips the row, and the R-side
# RE design the Hessian path builds has to give it no column either
# (gcol33/tulpa#883 -- it indexed column 0 and sparseMatrix() refused).

test_that(".re_design_matrix gives an idx = 0 row no RE column", {
  idx <- c(0L, 1L, 2L, 0L, 2L)
  x <- c(0.5, -1, 2, 3, 1)
  Z1 <- .re_design_matrix(list(list(idx = idx, n_groups = 2L, n_coefs = 1L)), 5L)
  expect_equal(as.matrix(Z1),
               cbind(c(0, 1, 0, 0, 0), c(0, 0, 1, 0, 1)))
  Z2 <- .re_design_matrix(list(list(idx = idx, n_groups = 2L, n_coefs = 2L,
                                    Z = cbind(1, x))), 5L)
  expect_equal(dim(Z2), c(5L, 4L))
  expect_equal(unname(rowSums(abs(as.matrix(Z2)))[c(1, 4)]), c(0, 0))
  # group-major: g1 intercept, g1 slope, g2 intercept, g2 slope
  expect_equal(as.matrix(Z2)[3, ], c(0, 0, 1, 2))
})

test_that("tulpa_laplace fits idx = 0 rows with return_hessian = TRUE", {
  skip_on_cran()
  set.seed(1)
  g <- rep(1:20, each = 6)
  x <- rnorm(120)
  y <- rpois(120, exp(0.3 + 0.5 * x + rnorm(20, 0, 0.5)[g]))
  rl <- list(list(idx = replace(g, 1:6, 0L), n_groups = 20L, n_coefs = 1L,
                  sigma = 1))
  a <- tulpa_laplace(y, rep(1L, 120), cbind(1, x), family = "poisson",
                     re_list = rl)
  b <- tulpa_laplace(y, rep(1L, 120), cbind(1, x), family = "poisson",
                     re_list = rl, return_hessian = FALSE)
  expect_equal(a$mode, b$mode, tolerance = 1e-8)
  expect_true(all(is.finite(sqrt(diag(vcov(a))))))
})
