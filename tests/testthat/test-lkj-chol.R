# Tests for src/lkj_chol_helpers.h
#
# Verifies the tanh-parameterized LKJ-Cholesky machinery used by HMC and the
# fast-path samplers (mclmc, pathfinder, DA, SMC) and the generic Gibbs kernel.

# ---- helpers ----

# Reference R implementation of build_L_from_raw (mirrors the C++ logic):
# z = tanh(raw) are canonical partial correlations and each entry is scaled by
# the room the row has left, so every raw vector maps into the cone.
ref_build_L <- function(raw, n) {
  L <- matrix(0, n, n)
  log_jac <- 0
  idx <- 1
  L[1, 1] <- 1
  if (n >= 2) {
    for (i in 2:n) {
      s <- 1
      for (j in seq_len(i - 1)) {
        z <- tanh(raw[idx])
        L[i, j] <- z * sqrt(s)
        log_jac <- log_jac + log(1 - z^2) + 0.5 * log(s)
        s <- s * (1 - z^2)
        idx <- idx + 1
      }
      L[i, i] <- sqrt(s)
    }
  }
  list(L = L, log_jac = log_jac)
}

# Independent reference: the Stan lkj_corr_cholesky_lpdf, written in its
# textbook 1-based form (unnormalized). The exponent (K - k + 2*eta - 2) on
# log(L[k,k]) is det(R)^(eta-1) with the correlation -> Cholesky Jacobian
# already folded in -- there is NO separate second Jacobian term. This is a
# different algebraic expression from the C++ helper's
# (eta - 1 + (n-k-1)/2)*2 form, so it cross-checks the code rather than copying
# it; the two agree only if the helper carries exactly one Jacobian.
ref_lkj_density <- function(L, eta) {
  n <- nrow(L)
  lp <- 0
  for (k in seq_len(n)) {
    lp <- lp + (n - k + 2 * eta - 2) * log(L[k, k])
  }
  lp
}

# ---- tests ----

test_that("build_L produces unit-norm rows", {
  set.seed(1L)
  for (n in 2:5) {
    n_raw <- n * (n - 1) / 2
    raw <- rnorm(n_raw, 0, 0.5)
    res <- cpp_test_lkj_build_L(raw, n)
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
  expect_equal(cpp$log_jac, ref$log_jac, tolerance = 1e-12)
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

test_that("raw-space density is the LKJ(eta) pushforward (det(R)^(eta-1) * |dR/draw|)", {
  # Ground-truth independent of the helper: the density on the raw parameters
  # that induces R ~ LKJ(eta) is det(R)^(eta-1) times the Jacobian of the free
  # correlations (strict lower triangle of R) with respect to raw. The helper's
  # (LKJ-on-Cholesky) + (tanh Jacobian) must equal this up to an additive
  # constant that depends only on (n, eta). A second, spurious Jacobian in the
  # helper would break the constant-offset property below.
  free_corr <- function(raw, n) {
    res <- cpp_test_lkj_build_L(raw, n)
    R <- res$L %*% t(res$L)
    R[lower.tri(R)]
  }
  target_logp <- function(raw, n, eta) {
    res <- cpp_test_lkj_build_L(raw, n)
    R <- res$L %*% t(res$L)
    J <- numDeriv_jacobian(function(r) free_corr(r, n), raw)
    (eta - 1) * log(det(R)) + log(abs(det(J)))
  }
  # small self-contained central-difference Jacobian (no extra dependency)
  numDeriv_jacobian <- function(f, x, h = 1e-6) {
    m <- length(f(x)); p <- length(x)
    J <- matrix(0, m, p)
    for (j in seq_len(p)) {
      xp <- x; xp[j] <- xp[j] + h
      xm <- x; xm[j] <- xm[j] - h
      J[, j] <- (f(xp) - f(xm)) / (2 * h)
    }
    J
  }
  set.seed(31L)
  for (n in 2:4) {
    for (eta in c(1.0, 2.0, 3.5)) {
      n_raw <- n * (n - 1) / 2
      raws <- lapply(1:5, function(i) rnorm(n_raw, 0, 0.3))
      offsets <- vapply(raws, function(raw) {
        helper <- cpp_test_lkj_density(cpp_test_lkj_build_L(raw, n)$L, eta) +
          cpp_test_lkj_build_L(raw, n)$log_jac
        helper - target_logp(raw, n, eta)
      }, numeric(1))
      # helper and the LKJ pushforward differ only by the normalizing constant
      expect_equal(max(offsets) - min(offsets), 0, tolerance = 1e-5,
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
    cpp_test_lkj_density(res$L, eta) + res$log_jac
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

test_that("every raw vector maps into the cone, including saturating ones", {
  # raw = atanh(0.8) twice on row 3 has row sum of squares 1.28 under a direct
  # tanh map, which is the case that map cannot represent. Here it is an
  # ordinary interior point.
  for (raw in list(c(8, 8, 8), c(atanh(0.8), atanh(0.8), atanh(0.8)),
                   c(-30, 30, -30), rep(0, 3))) {
    res <- cpp_test_lkj_build_L(raw, 3)
    R <- cpp_test_correlation_from_L(res$L)
    expect_equal(unname(diag(R)), rep(1, 3), tolerance = 1e-12)
    expect_true(all(diag(res$L) >= 0))
    ev <- eigen(R, symmetric = TRUE, only.values = TRUE)$values
    expect_true(all(ev > -1e-10))
  }
})

test_that("raw_from_L inverts build_L_from_raw", {
  set.seed(11L)
  for (n in 2:5) {
    n_raw <- n * (n - 1) / 2
    raw <- rnorm(n_raw, 0, 1.2)
    res <- cpp_test_lkj_build_L(raw, n)
    back <- cpp_test_lkj_raw_from_L(res$L)
    expect_equal(back, raw, tolerance = 1e-9)
    # and the round trip through L is exact
    expect_equal(cpp_test_lkj_build_L(back, n)$L, res$L, tolerance = 1e-12)
  }
})

test_that("n = 2 reduces to the direct tanh map", {
  # The scaling factor sqrt(s) is 1 for the single strict-lower entry of a 2x2
  # row, and the extra 0.5*log(s) Jacobian term is 0.5*log(1) = 0. So a random
  # intercept-and-slope term -- the overwhelmingly common correlated fit -- maps
  # as it did before, and a change in such a fit would be a defect rather than
  # the new parameterization.
  #
  # The residual against R's tanh is the engine's shared safe_tanh, which is
  # 2*sigmoid(2x) - 1 and differs from std::tanh in the last bit; that
  # implementation is what the HMC prior already used and is not what this
  # reduction is about.
  set.seed(41L)
  for (raw in as.list(rnorm(12, 0, 1.5))) {
    res <- cpp_test_lkj_build_L(raw, 2L)
    z <- tanh(raw)
    expect_equal(res$L[1, 1], 1, tolerance = 0)
    expect_equal(res$L[2, 1], z, tolerance = 1e-15)
    # sqrt(1 - z^2) is ill-conditioned in z as |z| -> 1, so the last-bit
    # difference in safe_tanh is amplified here; 1e-12 is still four orders
    # inside anything the map itself could change.
    expect_equal(res$L[2, 2], sqrt(1 - z^2), tolerance = 1e-12)
    expect_equal(res$log_jac, log(1 - z^2), tolerance = 1e-12)
  }
})

test_that("a saturating row at n = 3 gives a correlation matrix, not a clamped one", {
  # (atanh(0.8), atanh(0.8)) on row 3 has a squared sum of 1.28 under the direct
  # map, so its diagonal was clamped at sqrt(1e-12) and R = L L' came back with
  # a diagonal entry of 1.28 rather than 1.
  raw <- c(0.3, atanh(0.8), atanh(0.8))
  res <- cpp_test_lkj_build_L(raw, 3L)
  R <- cpp_test_correlation_from_L(res$L)
  expect_equal(unname(diag(R)), rep(1, 3), tolerance = 1e-12)
  expect_true(all(abs(R[lower.tri(R)]) < 1))
  expect_true(all(eigen(R, symmetric = TRUE, only.values = TRUE)$values > 1e-8))
  expect_true(is.finite(res$log_jac))
})
