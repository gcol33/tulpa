# MIID prior assembly (gcol33/tulpa#114): the multivariate-IID block -- the
# Q = I sibling of MCAR -- must assemble the precision Sigma^-1 (x) I_n exactly,
# with the matching gradient (no sum-to-zero pin: P is full rank) and log-prior
# (Sigma-dependent normalizer 0.5 n log|Sigma^-1|, full n not n - 1). Direct
# algebra against kronecker(Sigma^-1, I_n) -- the strongest validation of the
# C++ assembly, independent of the fit pipeline (mirrors test-mcar-prior.R).

.miid_check <- function(n, p, theta, seed) {
  set.seed(seed)
  x <- rnorm(p * n)
  r <- tulpa:::cpp_test_miid_prior(theta, p, n, x)
  Sinv <- r$Sinv
  ref_H <- kronecker(Sinv, diag(n))

  # (1) Hessian == kronecker(Sigma^-1, I_n) -- full rank, no neighbor entries.
  expect_lt(max(abs(r$H - ref_H)), 1e-9)

  # (2) gradient == -ref_H x (no sum-to-zero pin, unlike MCAR).
  ref_grad <- -as.numeric(ref_H %*% x)
  expect_lt(max(abs(r$grad - ref_grad)), 1e-9)

  # (3) log_prior closed form with the full-n Sigma-dependent normalizer.
  quad <- as.numeric(t(x) %*% ref_H %*% x)
  ld_sinv <- -r$log_det_Sigma
  ref_lp <- -0.5 * quad + 0.5 * n * ld_sinv - 0.5 * p * n * log(2 * pi)
  expect_lt(abs(r$log_prior - ref_lp), 1e-7)

  invisible(r)
}

test_that("MIID assembles Sigma^-1 (x) I with correct gradient and log-prior (p=2)", {
  # col-major lower-tri log-Cholesky: logL11, L21, logL22.
  .miid_check(n = 8L,  p = 2L, theta = c(log(1.2),  0.5, log(0.9)), seed = 1)
  .miid_check(n = 12L, p = 2L, theta = c(log(0.7), -0.8, log(1.5)), seed = 2)
})

test_that("MIID assembly is exact for p=3", {
  # col-major lower-tri: logL11, L21, L31, logL22, L32, logL33.
  .miid_check(n = 10L, p = 3L,
              theta = c(log(1.0), 0.3, -0.2, log(0.8), 0.4, log(1.3)),
              seed = 3)
})

test_that("MIID degenerate p=1 is a scalar N(0, sigma^2) precision", {
  # p = 1: Sigma = sigma^2 with sigma = exp(theta), Sigma^-1 = exp(-2 theta).
  n <- 7L
  theta <- log(1.3)
  set.seed(4)
  x <- rnorm(n)
  r <- tulpa:::cpp_test_miid_prior(theta, 1L, n, x)
  sinv <- exp(-2 * theta)
  expect_lt(max(abs(r$H - sinv * diag(n))), 1e-12)
  expect_lt(max(abs(r$grad - (-sinv * x))), 1e-12)
  ref_lp <- -0.5 * sinv * sum(x^2) + 0.5 * n * log(sinv) - 0.5 * n * log(2 * pi)
  expect_lt(abs(r$log_prior - ref_lp), 1e-10)
})

test_that("MIID log|Sigma| matches the log-Cholesky diagonal", {
  theta <- c(log(1.4), 0.6, log(0.7))
  r <- tulpa:::cpp_test_miid_prior(theta, 2L, 6L, rnorm(12))
  # |Sigma| = (L11 L22)^2 -> log|Sigma| = 2(logL11 + logL22).
  expect_lt(abs(r$log_det_Sigma - 2 * (theta[1] + theta[3])), 1e-12)
})
