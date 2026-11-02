# The SPDE precision is built on the lumped mass C-tilde_ii = <psi_i, 1>
# (Lindgren, Rue & Lindstrom 2011), the row sum of the consistent mass matrix.
# Its diagonal is half of that on a triangulation (area/6 against area/3 per
# triangle and vertex), which stretches the correlation length by sqrt(2).

test_that("spatial_spde_custom() takes the lumped mass as the row sum of C", {
  # Two triangles sharing an edge on the unit square; consistent P1 mass.
  tri_mass <- function(area) area / 12 * (matrix(1, 3, 3) + diag(3))
  C <- matrix(0, 4, 4)
  C[c(1, 2, 3), c(1, 2, 3)] <- C[c(1, 2, 3), c(1, 2, 3)] + tri_mass(0.5)
  C[c(2, 3, 4), c(2, 3, 4)] <- C[c(2, 3, 4), c(2, 3, 4)] + tri_mass(0.5)
  G <- diag(4)
  sp <- spatial_spde_custom(C, G, diag(4), prior_range = c(1, 0.5),
                            prior_sigma = c(1, 0.01))
  expect_equal(sp$C0_diag, rowSums(C))
  expect_equal(sum(sp$C0_diag), 1)                 # the domain's area
  # A mass matrix that is already lumped is its own row sum.
  sl <- spatial_spde_custom(diag(rowSums(C)), G, diag(4),
                            prior_range = c(1, 0.5), prior_sigma = c(1, 0.01))
  expect_equal(sl$C0_diag, sp$C0_diag)
})

skip_on_cran()

test_that("spatial_spde() C0 is the row sum of the consistent mass matrix", {
  skip_if_fast()
  set.seed(1)
  xy <- cbind(runif(60), runif(60))
  sp <- spatial_spde(xy, max_edge = c(0.15, 0.4))
  expect_equal(sp$C0_diag, as.numeric(Matrix::rowSums(sp$C)))
})

test_that("SPDE interior correlation and variance follow the Matern target", {
  skip_on_cran()
  # A fine mesh on a domain wide enough that the boundary does not reach the
  # probed interior; the FEM field converges to the Matern (nu = 1)
  # correlation r(d) = kappa d K_1(kappa d), kappa = sqrt(8) / range.
  set.seed(1)
  xy <- cbind(runif(400, -0.5, 1.5), runif(400, -0.5, 1.5))
  sp <- spatial_spde(xy, max_edge = c(0.03, 0.3), cutoff = 0.02)
  v <- sp$mesh$vertices[, 1:2]
  r <- 0.4
  kappa <- sqrt(8) / r
  Q <- tulpa_spde_precision_Q(sp, kappa, 1 / (sqrt(4 * pi) * kappa))
  centre <- which.min((v[, 1] - 0.5)^2 + (v[, 2] - 0.5)^2)
  d <- sqrt((v[, 1] - v[centre, 1])^2 + (v[, 2] - v[centre, 2])^2)
  ring <- function(d0) which(abs(d - d0) < 0.01)
  probe <- c(centre, ring(r / 2), ring(r))
  E <- Matrix::sparseMatrix(i = probe, j = seq_along(probe), x = 1,
                            dims = c(nrow(Q), length(probe)))
  S <- as.matrix(Matrix::solve(Q, E))[probe, , drop = FALSE]
  s <- sqrt(diag(S))
  R <- S[1, ] / (s[1] * s)
  matern <- function(dd) kappa * dd * besselK(kappa * dd, 1)
  n1 <- length(ring(r / 2))
  expect_lt(abs(mean(R[1 + seq_len(n1)]) - matern(r / 2)), 0.035)
  expect_lt(abs(mean(R[-(seq_len(n1 + 1))]) - matern(r)), 0.02)
  # Marginal variance at the unit sigma the tau was calibrated for.
  expect_lt(abs(mean(s^2) - 1), 0.08)
})
