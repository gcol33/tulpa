# MCAR prior assembly: the separable multivariate-CAR block must assemble the
# precision Sigma^-1 (x) Q exactly, with the matching gradient and log-prior
# (incl. the Sigma-dependent normalizer 0.5 (n-1) log|Sigma^-1|) (gcol33/tulpa#89).
# Direct algebra check against kronecker(Sigma^-1, Q) -- the strongest validation
# of the C++ assembly, independent of the fit pipeline.

.mcar_chain_csr <- function(n) {
  adj <- matrix(0L, n, n)
  for (s in seq_len(n - 1L)) { adj[s, s + 1L] <- 1L; adj[s + 1L, s] <- 1L }
  list(adj = adj, csr = adjacency_to_csr_tulpa(adj),
       Q = diag(rowSums(adj)) - adj)
}

.mcar_check <- function(n, p, theta, seed) {
  set.seed(seed)
  g <- .mcar_chain_csr(n)
  x <- rnorm(p * n)
  r <- tulpa:::cpp_test_mcar_prior(theta, p, n, g$csr$row_ptr, g$csr$col_idx,
                                   g$csr$n_neighbors, x)
  Sinv <- r$Sinv
  ref_H <- kronecker(Sinv, g$Q)

  # (1) Hessian == kronecker(Sigma^-1, Q).
  expect_lt(max(abs(r$H - ref_H)), 1e-8)

  # (2) gradient == -ref_H x - per-field sum-to-zero pin.
  pin <- numeric(p * n)
  for (a in seq_len(p)) {
    blk <- ((a - 1L) * n + 1L):(a * n)
    pin[blk] <- sum(x[blk])                   # MCAR_SUM2ZERO_TAU = 1
  }
  ref_grad <- -as.numeric(ref_H %*% x) - pin
  expect_lt(max(abs(r$grad - ref_grad)), 1e-8)

  # (3) log_prior closed form incl. the Sigma-dependent normalizer.
  quad <- as.numeric(t(x) %*% ref_H %*% x)
  pin_q <- sum(vapply(seq_len(p),
                      function(a) sum(x[((a - 1L) * n + 1L):(a * n)])^2,
                      numeric(1)))
  ld_sinv <- -r$log_det_Sigma
  ref_lp <- -0.5 * quad - 0.5 * pin_q +
            0.5 * (n - 1) * ld_sinv - 0.5 * p * (n - 1) * log(2 * pi)
  expect_lt(abs(r$log_prior - ref_lp), 1e-7)

  invisible(r)
}

test_that("MCAR assembles Sigma^-1 (x) Q with correct gradient and log-prior (p=2)", {
  # logL11, L21, logL22 (column-major lower-tri log-Cholesky).
  .mcar_check(n = 8L, p = 2L, theta = c(log(1.2), 0.5, log(0.9)), seed = 1)
  .mcar_check(n = 12L, p = 2L, theta = c(log(0.7), -0.8, log(1.5)), seed = 2)
})

test_that("MCAR assembly is exact for p=3", {
  # col-major lower-tri: logL11, L21, L31, logL22, L32, logL33
  .mcar_check(n = 10L, p = 3L,
              theta = c(log(1.0), 0.3, -0.2, log(0.8), 0.4, log(1.3)),
              seed = 3)
})

test_that("MCAR log|Sigma| matches the log-Cholesky diagonal", {
  g <- .mcar_chain_csr(6L)
  theta <- c(log(1.4), 0.6, log(0.7))
  r <- tulpa:::cpp_test_mcar_prior(theta, 2L, 6L, g$csr$row_ptr, g$csr$col_idx,
                                   g$csr$n_neighbors, rnorm(12))
  # |Sigma| = (L11 L22)^2 -> log|Sigma| = 2(logL11 + logL22).
  expect_lt(abs(r$log_det_Sigma - 2 * (theta[1] + theta[3])), 1e-12)
})
