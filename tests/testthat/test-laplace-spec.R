# LikelihoodSpec-driven Laplace path. cpp_laplace_spec_test_gaussian drives the
# spec solver with a hand-written Gaussian LikelihoodSpec (a stand-in for a
# downstream package's custom likelihood, NOT the built-in family adapter);
# cross-checking it against cpp_laplace_fit (family = "gaussian", the built-in
# adapter) proves an arbitrary user spec lands on the same posterior mode as the
# shipped family math. The multi-RE blocks below check the spec solver against a
# closed-form Gaussian posterior; the np == 2 fixtures exercise the multi-process
# coupling no single-response export reaches.

test_that("custom Gaussian spec matches the built-in family path for Gaussian + iid RE", {
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

  # Reference: the built-in Gaussian family through cpp_laplace_fit (identity
  # link), itself spec-driven since B2-live -- so this pins a custom user spec
  # against the shipped family closed form.
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

# ============================================================================
# Multi-term + slope RE tests. These exercise the lifted single-term
# contract on the spec-Laplace path. Each test builds a tiny linear-Gaussian
# problem, runs Laplace, and compares the mode against a closed-form
# (X'X + Λ) x = X'y solve in pure R.
# ============================================================================

# Build the full block-diagonal RE precision (sum_t Q_t ⊗ I_{G_t}) given
# a list of (sigma vector, optional L matrix) per term and a per-term
# group count. Needed to construct the closed-form reference Hessian.
make_re_precision_block <- function(re_terms_R, n_groups_per_term) {
  blocks <- lapply(seq_along(re_terms_R), function(t) {
    sig <- re_terms_R[[t]]$sigma
    L_t <- re_terms_R[[t]]$L
    q   <- length(sig)
    if (is.null(L_t) || q == 1) {
      Q_t <- diag(1 / sig^2, q)
    } else {
      Sigma_t <- diag(sig) %*% L_t %*% t(L_t) %*% diag(sig)
      Q_t <- solve(Sigma_t)
    }
    n_g <- n_groups_per_term[t]
    # block-diag (Q_t reused n_g times)
    full <- matrix(0, n_g * q, n_g * q)
    for (g in seq_len(n_g)) {
      idx <- ((g - 1) * q + 1):(g * q)
      full[idx, idx] <- Q_t
    }
    full
  })
  Reduce(function(A, B) {
    nA <- nrow(A); nB <- nrow(B)
    out <- matrix(0, nA + nB, nA + nB)
    out[seq_len(nA), seq_len(nA)] <- A
    out[(nA + 1):(nA + nB), (nA + 1):(nA + nB)] <- B
    out
  }, blocks)
}

# Stack the per-obs Z = [Z_1 | Z_2 | ... | Z_K] design where Z_t is N x
# (G_t * q_t) (intercept + slope columns interleaved per group).
make_re_design <- function(re_terms_R, N) {
  Zs <- lapply(re_terms_R, function(tm) {
    n_g <- tm$n_groups
    q   <- tm$n_coefs
    Zt  <- matrix(0, N, n_g * q)
    for (i in seq_len(N)) {
      g <- tm$group_idx[i]
      base <- (g - 1) * q
      Zt[i, base + 1] <- 1               # intercept
      if (q > 1) {
        for (s in seq_len(q - 1)) {
          Zt[i, base + 1 + s] <- tm$slope_mat[i, s]
        }
      }
    }
    Zt
  })
  do.call(cbind, Zs)
}

test_that("spec-driven Laplace handles two crossed intercept-only terms (1|g1)+(1|g2)", {
  set.seed(202605L)
  N <- 200L
  p <- 3L
  G1 <- 5L
  G2 <- 7L
  sigma1 <- 0.4
  sigma2 <- 0.7
  sigma_beta <- 100
  phi <- 0.6

  X <- cbind(1, matrix(rnorm(N * (p - 1L)), nrow = N))
  g1 <- sample.int(G1, N, replace = TRUE)
  g2 <- sample.int(G2, N, replace = TRUE)
  beta_true <- c(0.5, -0.3, 0.8)
  u1 <- rnorm(G1, sd = sigma1)
  u2 <- rnorm(G2, sd = sigma2)
  y <- as.numeric(X %*% beta_true) + u1[g1] + u2[g2] + rnorm(N, sd = phi)

  re_terms <- list(
    list(group_idx = g1, n_groups = G1, n_coefs = 1L,
         sigma = sigma1, correlated = FALSE),
    list(group_idx = g2, n_groups = G2, n_coefs = 1L,
         sigma = sigma2, correlated = FALSE)
  )
  spec <- tulpa:::cpp_laplace_spec_test_multi_re(
    y = y, X = X, re_terms = re_terms,
    sigma_beta = sigma_beta, phi = phi,
    max_iter = 200L, tol = 1e-12, n_threads = 1L
  )
  expect_true(spec$converged)

  # Closed form: Hessian on (beta, b1, b2) with prior precision Λ
  # Λ_beta = tau_beta I_p; Λ_b1 = (1/sigma1^2) I_{G1}; Λ_b2 = (1/sigma2^2) I_{G2}
  inv_phi2 <- 1 / phi^2
  tau_beta <- 1 / sigma_beta^2
  re_terms_R <- list(
    list(sigma = sigma1, n_groups = G1, n_coefs = 1L, group_idx = g1),
    list(sigma = sigma2, n_groups = G2, n_coefs = 1L, group_idx = g2)
  )
  Z <- make_re_design(re_terms_R, N)
  Lambda_re <- make_re_precision_block(re_terms_R, c(G1, G2))
  H_beta_beta <- inv_phi2 * crossprod(X) + tau_beta * diag(p)
  H_beta_b    <- inv_phi2 * crossprod(X, Z)
  H_b_b       <- inv_phi2 * crossprod(Z) + Lambda_re
  H_full <- rbind(
    cbind(H_beta_beta, H_beta_b),
    cbind(t(H_beta_b), H_b_b)
  )
  rhs <- c(inv_phi2 * crossprod(X, y), inv_phi2 * crossprod(Z, y))
  ref <- as.numeric(solve(H_full, rhs))
  expect_equal(spec$mode, ref, tolerance = 1e-7)
})

test_that("spec-driven Laplace handles uncorrelated random slope (x||g)", {
  set.seed(202606L)
  N <- 200L
  p <- 2L
  G <- 6L
  sigma_int   <- 0.4
  sigma_slope <- 0.25
  sigma_beta  <- 100
  phi <- 0.5

  X <- cbind(1, rnorm(N))
  g <- sample.int(G, N, replace = TRUE)
  x_slope <- rnorm(N)
  beta_true <- c(0.5, -0.2)
  b_int   <- rnorm(G, sd = sigma_int)
  b_slope <- rnorm(G, sd = sigma_slope)
  y <- as.numeric(X %*% beta_true) +
       b_int[g] + b_slope[g] * x_slope +
       rnorm(N, sd = phi)

  re_terms <- list(
    list(group_idx = g, n_groups = G, n_coefs = 2L,
         sigma = c(sigma_int, sigma_slope), correlated = FALSE,
         slope_mat = matrix(x_slope, ncol = 1L))
  )
  spec <- tulpa:::cpp_laplace_spec_test_multi_re(
    y = y, X = X, re_terms = re_terms,
    sigma_beta = sigma_beta, phi = phi,
    max_iter = 200L, tol = 1e-12, n_threads = 1L
  )
  expect_true(spec$converged)

  # Closed-form reference: Z_t has columns [intercept (1 if i in g), slope (x_i if i in g)]
  # interleaved per group as (b_{g, intercept}, b_{g, slope}, ...) for g = 1..G.
  inv_phi2 <- 1 / phi^2
  tau_beta <- 1 / sigma_beta^2
  re_terms_R <- list(
    list(sigma = c(sigma_int, sigma_slope), n_groups = G, n_coefs = 2L,
         group_idx = g, slope_mat = matrix(x_slope, ncol = 1L))
  )
  Z <- make_re_design(re_terms_R, N)
  Lambda_re <- make_re_precision_block(re_terms_R, G)
  H_beta_beta <- inv_phi2 * crossprod(X) + tau_beta * diag(p)
  H_beta_b    <- inv_phi2 * crossprod(X, Z)
  H_b_b       <- inv_phi2 * crossprod(Z) + Lambda_re
  H_full <- rbind(
    cbind(H_beta_beta, H_beta_b),
    cbind(t(H_beta_b), H_b_b)
  )
  rhs <- c(inv_phi2 * crossprod(X, y), inv_phi2 * crossprod(Z, y))
  ref <- as.numeric(solve(H_full, rhs))
  expect_equal(spec$mode, ref, tolerance = 1e-7)
})

test_that("spec-driven Laplace handles correlated random slope (x|g)", {
  set.seed(202607L)
  N <- 200L
  p <- 2L
  G <- 5L
  sigma_int   <- 0.5
  sigma_slope <- 0.4
  sigma_beta  <- 100
  phi <- 0.5

  # tanh-Cholesky parameterization: for q = 2 there is one raw value
  # (the off-diagonal correlation). raw = atanh(rho) so tanh(raw) = rho.
  rho <- 0.5
  raw <- atanh(rho)
  # L = [[1, 0], [rho, sqrt(1 - rho^2)]] (matches build_chol_L in C++)
  L_mat <- matrix(c(1, rho, 0, sqrt(1 - rho^2)), nrow = 2, byrow = FALSE)
  # Use rbind to emphasise the lower-tri layout match with C++ build_chol_L.

  X <- cbind(1, rnorm(N))
  g <- sample.int(G, N, replace = TRUE)
  x_slope <- rnorm(N)
  beta_true <- c(0.5, -0.2)
  # Σ = D L L^T D
  Sigma_b <- diag(c(sigma_int, sigma_slope)) %*% L_mat %*% t(L_mat) %*%
             diag(c(sigma_int, sigma_slope))
  Lc <- chol(Sigma_b)  # only used to simulate; reference uses tanh-Chol
  b_mat <- matrix(rnorm(G * 2), nrow = G) %*% Lc
  y <- as.numeric(X %*% beta_true) +
       b_mat[g, 1] + b_mat[g, 2] * x_slope +
       rnorm(N, sd = phi)

  re_terms <- list(
    list(group_idx = g, n_groups = G, n_coefs = 2L,
         sigma = c(sigma_int, sigma_slope), correlated = TRUE,
         slope_mat = matrix(x_slope, ncol = 1L),
         chol_raw = raw)
  )
  spec <- tulpa:::cpp_laplace_spec_test_multi_re(
    y = y, X = X, re_terms = re_terms,
    sigma_beta = sigma_beta, phi = phi,
    max_iter = 200L, tol = 1e-12, n_threads = 1L
  )
  expect_true(spec$converged)

  inv_phi2 <- 1 / phi^2
  tau_beta <- 1 / sigma_beta^2
  re_terms_R <- list(
    list(sigma = c(sigma_int, sigma_slope), n_groups = G, n_coefs = 2L,
         group_idx = g, slope_mat = matrix(x_slope, ncol = 1L), L = L_mat)
  )
  Z <- make_re_design(re_terms_R, N)
  Lambda_re <- make_re_precision_block(re_terms_R, G)
  H_beta_beta <- inv_phi2 * crossprod(X) + tau_beta * diag(p)
  H_beta_b    <- inv_phi2 * crossprod(X, Z)
  H_b_b       <- inv_phi2 * crossprod(Z) + Lambda_re
  H_full <- rbind(
    cbind(H_beta_beta, H_beta_b),
    cbind(t(H_beta_b), H_b_b)
  )
  rhs <- c(inv_phi2 * crossprod(X, y), inv_phi2 * crossprod(Z, y))
  ref <- as.numeric(solve(H_full, rhs))
  expect_equal(spec$mode, ref, tolerance = 1e-7)
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

