# Tests for src/lkj_chol_helpers.h
#
# Verifies the tanh-parameterized LKJ-Cholesky machinery used by HMC and the
# fast-path samplers (mclmc, pathfinder, DA, SMC) and the generic Gibbs kernel.

# ---- helpers ----

# Reference R implementation of build_L_from_raw (mirrors the C++ logic).
ref_build_L <- function(raw, n) {
  L <- matrix(0, n, n)
  log_jac <- 0
  idx <- 1
  ok <- TRUE
  for (i in seq_len(n)) {
    row_sum_sq <- 0
    if (i > 1) {
      for (j in seq_len(i - 1)) {
        l_ij <- tanh(raw[idx])
        L[i, j] <- l_ij
        row_sum_sq <- row_sum_sq + l_ij^2
        sech2 <- 1 - l_ij^2
        log_jac <- log_jac + log(max(1e-300, sech2))
        idx <- idx + 1
      }
    }
    diag_sq <- 1 - row_sum_sq
    if (diag_sq < 1e-10) { ok <- FALSE; break }
    L[i, i] <- sqrt(diag_sq)
  }
  list(L = L, log_jac = log_jac, ok = ok)
}

# Reference LKJ density (matches the C++ formula, which already absorbs the
# L -> R Jacobian; sigma prior is handled outside the helper).
ref_lkj_density <- function(L, eta) {
  n <- nrow(L)
  lp <- 0
  for (k in seq_len(n)) {
    lp <- lp + (eta - 1 + (n - k) / 2) * 2 * log(L[k, k])
  }
  for (k in seq_len(n)[-1]) {
    lp <- lp + (n - k + 1) * log(L[k, k])
  }
  # Reconcile R's 1-based indexing vs the C++ 0-based formula:
  # C++:   k = 0..n-1, factor (eta - 1 + (n-k-1)/2)
  # R k1:  k1 = k+1, so n-k-1 = n-k1, and exponent has 2*log(L[k1, k1])
  # Above expression already accounts for this.
  lp
}

# ---- tests ----

test_that("build_L produces unit-norm rows", {
  set.seed(1L)
  for (n in 2:5) {
    n_raw <- n * (n - 1) / 2
    raw <- rnorm(n_raw, 0, 0.5)
    res <- cpp_test_lkj_build_L(raw, n)
    expect_true(res$ok)
    row_sq <- rowSums(res$L^2)
    # Lower-triangular L: each row's squared entries should sum to 1
    expect_equal(unname(row_sq), rep(1, n), tolerance = 1e-12)
    # Diagonal must be > 0
    expect_true(all(diag(res$L) > 0))
    # Upper triangle untouched (zero)
    expect_true(all(res$L[upper.tri(res$L)] == 0))
  }
})

test_that("build_L matches reference R implementation", {
  set.seed(2L)
  n <- 4
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.7)
  cpp <- cpp_test_lkj_build_L(raw, n)
  ref <- ref_build_L(raw, n)
  expect_equal(cpp$L, ref$L, tolerance = 1e-12)
  expect_equal(cpp$log_jac_tanh, ref$log_jac, tolerance = 1e-12)
})

test_that("LKJ density matches reference formula", {
  set.seed(3L)
  for (n in 2:4) {
    n_raw <- n * (n - 1) / 2
    raw <- rnorm(n_raw, 0, 0.3)
    res <- cpp_test_lkj_build_L(raw, n)
    for (eta in c(1.0, 2.0, 4.0)) {
      cpp <- cpp_test_lkj_density(res$L, eta)
      ref <- ref_lkj_density(res$L, eta)
      expect_equal(cpp, ref, tolerance = 1e-12,
                   info = sprintf("n=%d eta=%g", n, eta))
    }
  }
})

test_that("LKJ gradient matches central finite difference", {
  set.seed(4L)
  n <- 4
  eta <- 2.0
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.4)

  # Total prior density as a function of raw (LKJ + L-Jacobian + tanh-Jacobian)
  total_logp <- function(r) {
    res <- cpp_test_lkj_build_L(r, n)
    if (!res$ok) return(NA_real_)
    cpp_test_lkj_density(res$L, eta) + res$log_jac_tanh
  }

  analytical <- cpp_test_lkj_grad(raw, n, eta)

  numerical <- numeric(n_raw)
  h <- 1e-5
  for (i in seq_len(n_raw)) {
    rp <- raw; rp[i] <- rp[i] + h
    rm <- raw; rm[i] <- rm[i] - h
    numerical[i] <- (total_logp(rp) - total_logp(rm)) / (2 * h)
  }

  expect_equal(analytical, numerical, tolerance = 1e-6)
})

test_that("compute_u_eff equals sigma * (z %*% t(L))", {
  set.seed(5L)
  n <- 3; n_groups <- 6
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.5)
  res <- cpp_test_lkj_build_L(raw, n)
  sigma <- c(0.5, 1.0, 1.5)
  z <- matrix(rnorm(n_groups * n), n_groups, n)

  cpp <- cpp_test_compute_u_eff(res$L, sigma, z)

  # Reference: u_eff[g, c] = sigma[c] * (L %*% z[g,])[c]
  # In matrix form: U = (z %*% t(L)) %*% diag(sigma)
  ref <- (z %*% t(res$L)) %*% diag(sigma)

  expect_equal(cpp, ref, tolerance = 1e-12)
})

test_that("correlation_from_L equals L %*% t(L) and is a valid correlation matrix", {
  set.seed(6L)
  n <- 4
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.6)
  res <- cpp_test_lkj_build_L(raw, n)
  R_cpp <- cpp_test_correlation_from_L(res$L)
  R_ref <- res$L %*% t(res$L)
  expect_equal(R_cpp, R_ref, tolerance = 1e-12)

  # Diagonal must be 1, off-diagonal in [-1, 1]
  expect_equal(unname(diag(R_cpp)), rep(1, n), tolerance = 1e-12)
  expect_true(all(abs(R_cpp) <= 1 + 1e-12))
  # Symmetric, positive semi-definite
  expect_equal(R_cpp, t(R_cpp), tolerance = 1e-12)
  expect_true(all(eigen(R_cpp, symmetric = TRUE, only.values = TRUE)$values > -1e-10))
})

test_that("chol_nc_chain_rule grad_log_sigma matches finite difference", {
  set.seed(7L)
  n <- 3; n_groups <- 5
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.4)
  log_sigma <- c(-0.2, 0.1, 0.3)
  z <- matrix(rnorm(n_groups * n), n_groups, n)
  glik <- matrix(rnorm(n_groups * n), n_groups, n)

  res <- cpp_test_lkj_build_L(raw, n)
  sigma <- exp(log_sigma)
  u_eff <- cpp_test_compute_u_eff(res$L, sigma, z)

  cr <- cpp_test_chol_nc_chain_rule(res$L, sigma, z, raw, u_eff, glik)

  # Linear functional of u_eff: f = sum(glik * u_eff)
  # df/d(log_sigma[c]) = sum_g glik[g,c] * u_eff[g,c]  (since d sigma / d log_sigma = sigma)
  ref_grad_log_sigma <- colSums(glik * u_eff)
  expect_equal(cr$grad_log_sigma, ref_grad_log_sigma, tolerance = 1e-12)

  # Numerical check via finite difference on log_sigma
  fwd <- function(ls) {
    sg <- exp(ls)
    u <- cpp_test_compute_u_eff(res$L, sg, z)
    sum(glik * u)
  }
  h <- 1e-6
  fd <- numeric(n)
  for (c in seq_len(n)) {
    lsp <- log_sigma; lsp[c] <- lsp[c] + h
    lsm <- log_sigma; lsm[c] <- lsm[c] - h
    fd[c] <- (fwd(lsp) - fwd(lsm)) / (2 * h)
  }
  expect_equal(cr$grad_log_sigma, fd, tolerance = 1e-6)
})

test_that("chol_nc_chain_rule grad_z matches finite difference", {
  set.seed(8L)
  n <- 3; n_groups <- 4
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.5)
  sigma <- c(0.7, 1.1, 1.3)
  z <- matrix(rnorm(n_groups * n), n_groups, n)
  glik <- matrix(rnorm(n_groups * n), n_groups, n)

  res <- cpp_test_lkj_build_L(raw, n)
  u_eff <- cpp_test_compute_u_eff(res$L, sigma, z)
  cr <- cpp_test_chol_nc_chain_rule(res$L, sigma, z, raw, u_eff, glik)

  fwd <- function(zz) {
    u <- cpp_test_compute_u_eff(res$L, sigma, zz)
    sum(glik * u)
  }
  h <- 1e-6
  fd <- matrix(0, n_groups, n)
  for (g in seq_len(n_groups)) {
    for (k in seq_len(n)) {
      zp <- z; zp[g, k] <- zp[g, k] + h
      zm <- z; zm[g, k] <- zm[g, k] - h
      fd[g, k] <- (fwd(zp) - fwd(zm)) / (2 * h)
    }
  }
  expect_equal(cr$grad_z, fd, tolerance = 1e-6)
})

test_that("chol_nc_chain_rule grad_raw matches finite difference", {
  set.seed(9L)
  n <- 3; n_groups <- 4
  n_raw <- n * (n - 1) / 2
  raw <- rnorm(n_raw, 0, 0.4)
  sigma <- c(0.5, 0.9, 1.4)
  z <- matrix(rnorm(n_groups * n), n_groups, n)
  glik <- matrix(rnorm(n_groups * n), n_groups, n)

  res <- cpp_test_lkj_build_L(raw, n)
  u_eff <- cpp_test_compute_u_eff(res$L, sigma, z)
  cr <- cpp_test_chol_nc_chain_rule(res$L, sigma, z, raw, u_eff, glik)

  fwd <- function(r) {
    rr <- cpp_test_lkj_build_L(r, n)
    if (!rr$ok) return(NA_real_)
    u <- cpp_test_compute_u_eff(rr$L, sigma, z)
    sum(glik * u)
  }
  h <- 1e-6
  fd <- numeric(n_raw)
  for (i in seq_len(n_raw)) {
    rp <- raw; rp[i] <- rp[i] + h
    rm <- raw; rm[i] <- rm[i] - h
    fd[i] <- (fwd(rp) - fwd(rm)) / (2 * h)
  }
  expect_equal(cr$grad_raw, fd, tolerance = 1e-6)
})

test_that("build_L returns ok=FALSE for raw values that violate the row-norm constraint", {
  # Push raw values toward extremes so tanh saturates near +/-1 and the row
  # squared sum exceeds 1 - 1e-10.
  raw <- c(8, 8, 8)  # n = 3, all near saturation
  res <- cpp_test_lkj_build_L(raw, 3)
  expect_false(res$ok)
})
