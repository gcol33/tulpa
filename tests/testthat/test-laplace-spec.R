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

