test_that("free-covariance rails name the marginal SD and partial correlation", {
  C <- as.matrix(expand.grid(L11 = c(-1, 0, 1), L21 = c(-1, 0, 1),
                            L22 = c(-7, -4, -1)))
  M <- tulpa:::.hp_logchol_from_coords(C, list(design = "cpc", p = 2L))
  colnames(M) <- paste0("b1.", colnames(C))
  lm <- -.5 * C[, 1L]^2 - .5 * C[, 2L]^2 - 2 * (C[, 3L] + 7)
  res <- list(theta_grid = M, log_marginal = lm)
  expect_identical(tulpa:::.nl_railed_axes(res), "b1.log_sigma_2:lower")
  lm <- -.5 * C[, 1L]^2 + 2 * C[, 2L] - .5 * (C[, 3L] + 4)^2
  res$log_marginal <- lm
  expect_identical(tulpa:::.nl_railed_axes(res), "b1.atanh_pcor_21:upper")
  res$outer_grid_railed_axes <- tulpa:::.nl_railed_axes(res)
  res$outer_grid_axis_declined <- c("b1.L21" = "curvature_not_negative")
  expect_warning(tulpa:::.nl_warn_unplaced_rail(res, "fit"), "atanh_pcor_21")
})

test_that("a log-Cholesky tensor reports rails in its own design", {
  M <- as.matrix(expand.grid(L11 = c(-1, 0, 1), L21 = c(-1, 0, 1),
                            L22 = c(-7, -4, -1)))
  lm <- -.5 * M[, 1L]^2 - .5 * M[, 2L]^2 - 2 * (M[, 3L] + 7)
  colnames(M) <- paste0("b1.", colnames(M))
  expect_identical(tulpa:::.nl_railed_axes(list(theta_grid = M, log_marginal = lm)),
                   "b1.L22:lower")
})
