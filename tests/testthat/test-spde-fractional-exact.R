# The fractional-nu SPDE field against its exact discrete reference.
#
# On a mesh with lumped mass C and stiffness G, the fractional Matern field is
# u ~ N(0, tau^-2 L^-alpha C^-1) with L = C^-1 G + kappa^2 and alpha = nu + 1.
# Its covariance is computed exactly by an eigendecomposition of
# C^-1/2 G C^-1/2, so the gaussian marginal of y = X beta + A u + e is a closed
# form: N(0, A Sigma_u A' + X X' / tau_beta + phi I). The rational assembly and
# the precomputed Laplace solve must reproduce it, with the full normalizer, so
# that a fractional fit's evidence is comparable with an integer one's.

.frac_fixture <- function(seed = 3L, n = 90L, range = 0.2, sigma = 0.8) {
  set.seed(seed)
  xy <- cbind(runif(n), runif(n))
  X <- matrix(1, n, 1)
  D <- as.matrix(stats::dist(xy))
  u <- as.numeric(t(chol(sigma^2 * exp(-2 * D / range) + diag(1e-9, n))) %*%
                    rnorm(n))
  phi <- 0.1
  list(xy = xy, X = X, phi = phi, n = n,
       y = as.numeric(0.5 + u + rnorm(n, 0, sqrt(phi))))
}

.frac_spec <- function(fx, nu) {
  spatial_spde(fx$xy, nu = nu, max_edge = c(0.06, 0.3))
}

# Exact gaussian log marginal of the discrete field at (range, sigma), from the
# eigendecomposition of C^-1/2 G C^-1/2 on the nodes carrying mass.
.frac_exact_lm <- function(fx, sp, range, sigma, tau_beta = 1e-4) {
  C0 <- as.numeric(sp$C0_diag)
  keep <- which(C0 > 1e-12)
  C0k <- C0[keep]
  Gk <- as.matrix(sp$G[keep, keep])
  Ak <- as.matrix(sp$A[, keep])
  S <- Gk / sqrt(outer(C0k, C0k))
  e <- eigen(S, symmetric = TRUE)
  nu <- sp$nu
  kappa <- sqrt(8 * nu) / range
  tau2inv <- 4 * pi * nu * kappa^(2 * nu) * sigma^2
  W <- e$vectors / sqrt(C0k)
  Su <- W %*% (t(W) * (kappa^2 + e$values)^(-(nu + 1))) * tau2inv
  V <- Ak %*% Su %*% t(Ak) + fx$X %*% t(fx$X) / tau_beta + diag(fx$phi, fx$n)
  R <- chol(V)
  z <- backsolve(R, fx$y, transpose = TRUE)
  -0.5 * sum(z^2) - sum(log(diag(R))) - 0.5 * fx$n * log(2 * pi)
}

.frac_kernel_lm <- function(fx, sp, range, sigma) {
  if (tulpa:::.spde_nu_is_fractional(sp$nu)) {
    tulpa:::.spde_nested_logmarginal_at(
      spatial = sp, range = range, sigma = sigma, y = fx$y, X = fx$X,
      family = "gaussian", phi = fx$phi, n_trials = rep(1L, fx$n),
      order = tulpa:::.spde_rational_order(sp), max_iter = 100L, tol = 1e-8,
      n_threads = 1L, offset = NULL)$log_marginal
  } else {
    tulpa:::laplace_spde_at(fx$y, rep(1L, fx$n), fx$X, sp, family = "gaussian",
                            phi = fx$phi, range = range, sigma = sigma)$log_marginal
  }
}

test_that("an SPDE spec carries the Gershgorin bound of C^-1 G", {
  skip_on_cran_budget()
  fx <- .frac_fixture()
  sp <- .frac_spec(fx, 0.5)
  keep <- which(sp$C0_diag > 1e-12)
  S <- as.matrix(sp$G[keep, keep]) / sqrt(outer(sp$C0_diag[keep], sp$C0_diag[keep]))
  top <- max(eigen(S, symmetric = TRUE, only.values = TRUE)$values)
  expect_gte(sp$lambda_max_G, top)
  expect_equal(sp$lambda_max_G,
               max(Matrix::rowSums(abs(sp$G[keep, keep])) / sp$C0_diag[keep]))
})

test_that("the integer SPDE gaussian marginal is the exact marginal, normalizer included", {
  skip_on_cran()
  fx <- .frac_fixture()
  sp <- .frac_spec(fx, 1)
  for (r in c(0.1, 0.4, 3)) {
    expect_equal(.frac_kernel_lm(fx, sp, r, 0.8), .frac_exact_lm(fx, sp, r, 0.8),
                 tolerance = 1e-4)
  }
})

test_that("the fractional SPDE gaussian marginal tracks the exact fractional field", {
  skip_on_cran()
  fx <- .frac_fixture()
  for (nu in c(0.5, 1.5)) {
    sp <- .frac_spec(fx, nu)
    for (r in c(0.1, 0.3, 1)) {
      err <- .frac_kernel_lm(fx, sp, r, 0.8) - .frac_exact_lm(fx, sp, r, 0.8)
      expect_lt(abs(err), 0.5)
    }
  }
})

test_that("the fractional marginal orders the ranges as the exact one does", {
  skip_on_cran()
  # The outer mode-find's box reaches 100x the prior range. A marginal that
  # departs from the exact one out there -- rising where the exact one falls --
  # draws the search to the box corner.
  fx <- .frac_fixture()
  sp <- .frac_spec(fx, 0.5)
  ranges <- c(0.05, 0.1, 0.2, 0.5, 1, 3, 10, 100 * sp$prior_range[1])
  k <- vapply(ranges, function(r) .frac_kernel_lm(fx, sp, r, 0.8), numeric(1))
  e <- vapply(ranges, function(r) .frac_exact_lm(fx, sp, r, 0.8), numeric(1))
  expect_true(all(is.finite(k)))
  expect_identical(order(k), order(e))
  expect_lt(max(abs(k - e)), 5)
})

test_that("a nested fractional fit finds an interior mode and lays a CCD design", {
  skip_on_cran()
  fx <- .frac_fixture()
  sp <- .frac_spec(fx, 0.5)
  expect_no_warning(
    f <- fit_spde(fx$y, fx$X, sp, family = "gaussian", phi = fx$phi,
                  control = list(diagnose_k = FALSE)))
  expect_identical(f$nested$method, "ccd")
  expect_gt(f$nested$range_mean, 0.05)
  expect_lt(f$nested$range_mean, 0.8)
})

test_that("a fractional field's interior marginal variance is sigma^2", {
  skip_on_cran()
  # tau is the Matern map of (range, sigma), as on the integer path, so away
  # from the Neumann boundary the field variance is the exact discrete field's
  # up to the rational error, and sigma^2 up to the FEM error.
  fx <- .frac_fixture()
  sp <- .frac_spec(fx, 0.5)
  asm <- tulpa:::.spde_assemble_at(sp, range = 0.2, sigma = 0.8)
  Sigma <- as.matrix(asm$Pr %*% Matrix::solve(asm$Q, Matrix::t(asm$Pr)))
  C0k <- sp$C0_diag[asm$keep]
  S <- as.matrix(sp$G[asm$keep, asm$keep]) / sqrt(outer(C0k, C0k))
  e <- eigen(S, symmetric = TRUE)
  kappa <- 2 / 0.2
  W <- e$vectors / sqrt(C0k)
  exact <- rowSums(W^2 * rep((kappa^2 + e$values)^(-1.5), each = nrow(W))) *
    4 * pi * 0.5 * kappa * 0.8^2
  v <- sp$mesh$vertices[asm$keep, , drop = FALSE]
  inner <- v[, 1] > 0.3 & v[, 1] < 0.7 & v[, 2] > 0.3 & v[, 2] < 0.7
  expect_gt(sum(inner), 10)
  expect_equal(diag(Sigma)[inner], exact[inner], tolerance = 0.01)
  expect_equal(median(diag(Sigma)[inner]), 0.8^2, tolerance = 0.1)
})
