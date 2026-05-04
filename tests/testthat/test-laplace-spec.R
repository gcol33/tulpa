# LikelihoodSpec-driven Laplace path. Cross-checks the new
# tulpa_laplace_spec_dense entry against the family-enum reference
# (cpp_laplace_fit, family = "gaussian") on a small simulated dataset.
# Both paths should land on the same posterior mode within Newton
# tolerance — that proves the spec contract computes the same answer
# the proven internal Laplace path does.

test_that("spec-driven Laplace matches family-enum reference for Gaussian + iid RE", {
  set.seed(42L)
  N <- 200L
  p <- 3L
  n_re_groups <- 10L
  sigma_re <- 0.5
  phi <- 1.0
  beta_true <- c(0.5, -0.3, 0.8)

  X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
  re_idx <- sample.int(n_re_groups, N, replace = TRUE)
  u_true <- rnorm(n_re_groups, sd = sigma_re)
  eta <- as.numeric(X %*% beta_true) + u_true[re_idx]
  y <- eta + rnorm(N, sd = phi)

  # Reference: family-enum Laplace (binomial path also exercises the same
  # solver — gaussian uses identity link with the same scatter helpers).
  ref <- tulpa:::cpp_laplace_fit(
    y         = y,
    n         = rep(1L, N),
    X         = X,
    re_idx    = as.numeric(re_idx),
    n_re_groups = n_re_groups,
    sigma_re  = sigma_re,
    family    = "gaussian",
    phi       = phi,
    max_iter  = 100L,
    tol       = 1e-10,
    n_threads = 1L
  )

  # Spec-driven Laplace via the new shim. Use sigma_beta = 100 so
  # tau_beta = 1e-4 matches the reference's hardcoded prior precision.
  spec <- tulpa:::cpp_laplace_spec_test_gaussian(
    y           = y,
    X           = X,
    re_idx      = re_idx,
    n_re_groups = n_re_groups,
    sigma_re    = sigma_re,
    sigma_beta  = 100,
    phi         = phi,
    max_iter    = 100L,
    tol         = 1e-10,
    n_threads   = 1L
  )

  expect_true(spec$converged)
  expect_equal(length(spec$mode), p + n_re_groups)

  # Mode agreement: same Newton solver shape, same priors, same likelihood
  # contributions per obs → same fixed point. Tighter than 1e-6 in practice;
  # 1e-5 leaves headroom for compiler reordering across the two TUs.
  expect_equal(spec$mode, ref$mode, tolerance = 1e-5)
})

test_that("spec-driven Laplace works without RE", {
  set.seed(7L)
  N <- 100L
  p <- 2L
  beta_true <- c(1.0, -0.5)
  X <- cbind(1, rnorm(N))
  y <- as.numeric(X %*% beta_true) + rnorm(N, sd = 0.5)

  ref <- tulpa:::cpp_laplace_fit(
    y         = y,
    n         = rep(1L, N),
    X         = X,
    re_idx    = numeric(0),
    n_re_groups = 0L,
    sigma_re  = 1.0,
    family    = "gaussian",
    phi       = 0.5,
    max_iter  = 100L,
    tol       = 1e-10,
    n_threads = 1L
  )

  spec <- tulpa:::cpp_laplace_spec_test_gaussian(
    y           = y,
    X           = X,
    re_idx      = integer(0),
    n_re_groups = 0L,
    sigma_re    = 1.0,
    sigma_beta  = 100,
    phi         = 0.5,
    max_iter    = 100L,
    tol         = 1e-10,
    n_threads   = 1L
  )

  expect_true(spec$converged)
  expect_equal(spec$mode, ref$mode, tolerance = 1e-5)
})

test_that("spec-driven Laplace handles n_processes == 2 with shared iid RE", {
  # Two independent Gaussian processes on the same observations sharing a
  # single iid RE that enters both linear predictors. The mode of (beta_1,
  # beta_2, u) is the joint solution of:
  #   X1' (y1 - X1 beta_1 - u_idx) / phi1^2 - tau_beta * beta_1 = 0
  #   X2' (y2 - X2 beta_2 - u_idx) / phi2^2 - tau_beta * beta_2 = 0
  #   sum_{i: idx == g} ((y1 - eta1)/phi1^2 + (y2 - eta2)/phi2^2) - tau_re u_g = 0
  # This is a linear system; solve it directly in R as the reference.
  set.seed(123L)
  N <- 150L
  p1 <- 2L
  p2 <- 3L
  n_re_groups <- 8L
  sigma_re <- 0.4
  sigma_beta <- 100
  phi1 <- 0.5
  phi2 <- 0.8

  X1 <- cbind(1, rnorm(N))
  X2 <- cbind(1, rnorm(N), rnorm(N))
  re_idx <- sample.int(n_re_groups, N, replace = TRUE)
  u_true <- rnorm(n_re_groups, sd = sigma_re)
  beta1_true <- c(0.5, -0.7)
  beta2_true <- c(-0.2, 0.9, 0.4)

  eta1 <- as.numeric(X1 %*% beta1_true) + u_true[re_idx]
  eta2 <- as.numeric(X2 %*% beta2_true) + u_true[re_idx]
  y1 <- eta1 + rnorm(N, sd = phi1)
  y2 <- eta2 + rnorm(N, sd = phi2)

  spec <- tulpa:::cpp_laplace_spec_test_gaussian2p(
    y1 = y1, y2 = y2, X1 = X1, X2 = X2,
    offset1 = numeric(0), offset2 = numeric(0),
    re_idx = re_idx, n_re_groups = n_re_groups,
    sigma_re = sigma_re, sigma_beta = sigma_beta,
    phi1 = phi1, phi2 = phi2,
    re_into_proc0 = TRUE, re_into_proc1 = TRUE,
    max_iter = 200L, tol = 1e-12, n_threads = 1L
  )
  expect_true(spec$converged)
  expect_equal(length(spec$mode), p1 + p2 + n_re_groups)

  # Closed-form reference: build the Hessian H and gradient g of the
  # negative log-posterior for the latent vector x = (beta_1, beta_2, u),
  # then x* = H^{-1} (-g_at_zero) in one shot (linear-Gaussian model).
  inv1 <- 1 / phi1^2
  inv2 <- 1 / phi2^2
  tau_beta <- 1 / sigma_beta^2
  tau_re   <- 1 / sigma_re^2

  Z <- matrix(0, N, n_re_groups)
  for (i in seq_len(N)) Z[i, re_idx[i]] <- 1

  # Block Hessian (negative log-posterior):
  #  [X1'X1 * inv1 + tau_beta*I, 0,                       X1'Z * inv1;
  #   0,                         X2'X2 * inv2 + tau_beta, X2'Z * inv2;
  #   Z'X1 * inv1,               Z'X2 * inv2,             Z'Z*(inv1+inv2)+tau_re*I]
  H11 <- crossprod(X1) * inv1 + tau_beta * diag(p1)
  H22 <- crossprod(X2) * inv2 + tau_beta * diag(p2)
  H33 <- crossprod(Z)  * (inv1 + inv2) + tau_re * diag(n_re_groups)
  H13 <- crossprod(X1, Z) * inv1
  H23 <- crossprod(X2, Z) * inv2
  H <- rbind(
    cbind(H11, matrix(0, p1, p2), H13),
    cbind(matrix(0, p2, p1), H22, H23),
    cbind(t(H13), t(H23), H33)
  )
  rhs <- c(crossprod(X1, y1) * inv1,
           crossprod(X2, y2) * inv2,
           crossprod(Z, y1) * inv1 + crossprod(Z, y2) * inv2)
  ref_mode <- as.numeric(solve(H, rhs))

  expect_equal(spec$mode, ref_mode, tolerance = 1e-5)
})

test_that("spec-driven Laplace honours per-process offsets", {
  # Adding a constant offset c to one process's eta should shift its beta_0
  # mode by -c (when intercept is the only spanned term that can absorb it).
  # Here we add a small offset to process 1 only, no RE, no shared structure;
  # the closed-form mode follows directly from the OLS fit on (y - offset).
  set.seed(99L)
  N <- 100L
  X1 <- cbind(1, rnorm(N))
  X2 <- cbind(1, rnorm(N))
  beta1_true <- c(0.3, -0.6)
  beta2_true <- c(0.1, 0.4)
  off <- rep(0.7, N)
  phi1 <- 0.5
  phi2 <- 0.5
  sigma_beta <- 100

  y1 <- as.numeric(X1 %*% beta1_true) + off + rnorm(N, sd = phi1)
  y2 <- as.numeric(X2 %*% beta2_true) + rnorm(N, sd = phi2)

  spec <- tulpa:::cpp_laplace_spec_test_gaussian2p(
    y1 = y1, y2 = y2, X1 = X1, X2 = X2,
    offset1 = off, offset2 = numeric(0),
    re_idx = integer(0), n_re_groups = 0L,
    sigma_re = 1.0, sigma_beta = sigma_beta,
    phi1 = phi1, phi2 = phi2,
    max_iter = 200L, tol = 1e-12, n_threads = 1L
  )
  expect_true(spec$converged)

  # Ridge MAP for each process: beta = (X'X + tau_beta * I)^{-1} X' (y - offset)
  tau_beta <- 1 / sigma_beta^2
  rhs1 <- crossprod(X1, y1 - off)
  rhs2 <- crossprod(X2, y2)
  ref1 <- as.numeric(solve(crossprod(X1) + tau_beta * diag(2), rhs1))
  ref2 <- as.numeric(solve(crossprod(X2) + tau_beta * diag(2), rhs2))

  expect_equal(spec$mode[1:2], ref1, tolerance = 1e-6)
  expect_equal(spec$mode[3:4], ref2, tolerance = 1e-6)
})

