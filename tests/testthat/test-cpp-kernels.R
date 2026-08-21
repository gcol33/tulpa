# The compiled numeric kernels the samplers and the Laplace path are built on,
# each against an independent R computation of the same quantity. The wrappers
# exist so these can be reached from R; unreached, the harness reads from the
# outside as coverage that is not there.

test_that("scalar math kernels match their R closed forms", {
  v <- c(-3.2, 0.4, 7.1, 2.0)
  expect_equal(cpp_test_log_sum_exp(v), log(sum(exp(v))))
  expect_equal(cpp_test_softmax(v), exp(v) / sum(exp(v)))
  expect_equal(cpp_test_inv_logit(c(-4, 0, 2.5)), plogis(c(-4, 0, 2.5)))
  for (a in c(0.5, 3, 12.25)) expect_equal(cpp_test_lgamma(a), lgamma(a))
})

test_that("log_sum_exp does not overflow on a large shift", {
  big <- c(800, 799)
  expect_equal(cpp_test_log_sum_exp(big), 800 + log(1 + exp(-1)))
  expect_true(all(is.finite(cpp_test_softmax(big))))
  expect_equal(sum(cpp_test_softmax(big)), 1)
})

test_that("family log-likelihood kernels match the R densities", {
  y   <- c(0L, 2L, 5L, 1L)
  lam <- c(0.4, 2.1, 4.4, 1.0)
  expect_equal(cpp_test_poisson_loglik(y, lam),
               sum(dpois(y, lam, log = TRUE)))

  n <- c(3L, 4L, 8L, 2L)
  p <- c(0.2, 0.55, 0.71, 0.4)
  expect_equal(cpp_test_binomial_loglik(y, n, p),
               sum(dbinom(y, n, p, log = TRUE)))

  mu <- c(0.7, 2.0, 3.9, 1.1)
  expect_equal(cpp_test_negbin_loglik(y, mu, 2.5),
               sum(dnbinom(y, size = 2.5, mu = mu, log = TRUE)))

  ynum <- c(-0.4, 1.2, 0.3, 2.2)
  expect_equal(cpp_test_normal_loglik(ynum, mu, 1.3),
               sum(dnorm(ynum, mu, 1.3, log = TRUE)))
})

test_that("the negbin kernel answers the mu = 0 boundary rather than 0 * -Inf", {
  expect_equal(cpp_test_negbin_loglik(0L, 0.0, 2.5), 0)
  expect_equal(cpp_test_negbin_loglik(2L, 0.0, 2.5), -Inf)
})

test_that("dense linear algebra kernels match base R", {
  set.seed(11)
  A <- crossprod(matrix(rnorm(16), 4, 4)) + diag(4)
  x <- rnorm(4)

  L <- matrix(cpp_test_cholesky(A), 4, 4)
  expect_equal(L %*% t(L), A, tolerance = 1e-10, ignore_attr = TRUE)

  M <- matrix(rnorm(12), 4, 3)
  b <- rnorm(3)
  expect_equal(cpp_test_matvec(M, b), as.numeric(M %*% b), tolerance = 1e-12)
  expect_equal(cpp_test_linalg_matvec(M, b), as.numeric(M %*% b),
               tolerance = 1e-12)
  expect_equal(cpp_test_linalg_matvec_add(M, b, x), as.numeric(M %*% b) + x,
               tolerance = 1e-12)
  expect_equal(cpp_test_linalg_matvec_transpose(M, x),
               as.numeric(crossprod(M, x)), tolerance = 1e-12)
})

test_that("vector kernels match base R", {
  x <- c(1.5, -2, 0.25, 4)
  y <- c(0.5, 3, -1, 2)
  expect_equal(cpp_test_dot_product(x, y), sum(x * y))
  expect_equal(cpp_test_norm_squared(x), sum(x^2))
  expect_equal(cpp_test_vector_sum(x), sum(x))
  expect_equal(cpp_test_axpy(2.5, x, y), 2.5 * x + y)
  expect_equal(cpp_test_scale(-3, x), -3 * x)
  expect_equal(cpp_test_linalg_log_sum_exp(1.2, -0.4), log(exp(1.2) + exp(-0.4)))
  expect_equal(cpp_test_linalg_log_sum_exp_vec(x), log(sum(exp(x))))
  expect_equal(cpp_test_softmax_inplace(x), exp(x) / sum(exp(x)))
})

test_that("the sparse Laplacian quadratic form matches the dense x L x", {
  # Path graph on 5 nodes, CSR neighbour lists.
  n <- 5L
  nb <- list(2L, c(1L, 3L), c(2L, 4L), c(3L, 5L), 4L)
  row_ptr <- as.integer(c(0, cumsum(lengths(nb))))
  col_idx <- as.integer(unlist(nb) - 1L)
  x <- c(0.3, -1.2, 2.0, 0.5, -0.7)

  A <- matrix(0, n, n)
  for (i in seq_len(n)) A[i, nb[[i]]] <- 1
  L <- diag(rowSums(A)) - A
  expect_equal(cpp_test_sparse_laplacian_quadform(row_ptr, col_idx, x),
               as.numeric(t(x) %*% L %*% x), tolerance = 1e-12)
})

test_that("the two-process linear predictors match X beta on each arm", {
  set.seed(3)
  Xn <- matrix(rnorm(30), 10, 3); bn <- rnorm(3)
  Xd <- matrix(rnorm(20), 10, 2); bd <- rnorm(2)
  r <- cpp_test_compute_linear_predictors(Xn, bn, Xd, bd, 1L)
  expect_equal(r$eta_num, as.numeric(Xn %*% bn), tolerance = 1e-12)
  expect_equal(r$eta_denom, as.numeric(Xd %*% bd), tolerance = 1e-12)
})

test_that("temporal quadratic forms and the AR1 density match their definitions", {
  phi <- c(0.4, -0.2, 1.1, 0.7, -0.5)
  T_n <- length(phi)
  expect_equal(cpp_test_rw1_quadratic_form(phi, FALSE), sum(diff(phi)^2))
  expect_equal(cpp_test_rw1_quadratic_form(phi, TRUE),
               sum(diff(phi)^2) + (phi[1] - phi[T_n])^2)
  expect_equal(cpp_test_rw2_quadratic_form(phi, FALSE),
               sum(diff(phi, differences = 2)^2))

  rho <- 0.6; tau <- 2.0
  quad <- (1 - rho^2) * phi[1]^2 + sum((phi[-1] - rho * phi[-T_n])^2)
  # One rank-T GMRF: |tau Q_AR1| gives 0.5 T (log tau - log 2pi) plus
  # 0.5 log(1 - rho^2). Equivalently, the stationary marginal on phi[1] times
  # the T - 1 Gaussian transitions.
  expect_equal(cpp_test_ar1_log_density(phi, rho, tau),
               0.5 * T_n * (log(tau) - log(2 * pi)) + 0.5 * log(1 - rho^2) -
                 0.5 * tau * quad,
               tolerance = 1e-10)
  expect_equal(cpp_test_ar1_log_density(phi, rho, tau),
               dnorm(phi[1], 0, sqrt(1 / (tau * (1 - rho^2))), log = TRUE) +
                 sum(dnorm(phi[-1], rho * phi[-T_n], sqrt(1 / tau), log = TRUE)),
               tolerance = 1e-10)
})

test_that("the leapfrog integrator conserves the Hamiltonian to second order", {
  # U(q) = 0.5 q'q, K(p) = 0.5 p'p: a harmonic oscillator, where a symplectic
  # integrator has bounded rather than growing energy error.
  q0 <- c(1.0, -0.5); p0 <- c(0.2, 0.9)
  H0 <- cpp_test_hamiltonian(q0, p0)
  expect_equal(H0, 0.5 * sum(q0^2) + 0.5 * sum(p0^2))

  errs <- vapply(c(0.1, 0.05, 0.025), function(eps) {
    st <- cpp_test_leapfrog(q0, p0, eps, as.integer(round(1 / eps)))
    abs(cpp_test_hamiltonian(st$q, st$p) - H0)
  }, numeric(1))
  expect_lt(errs[2], errs[1] / 3)
  expect_lt(errs[3], errs[2] / 3)
})

test_that("the leapfrog integrator is time-reversible", {
  q0 <- c(1.0, -0.5); p0 <- c(0.2, 0.9)
  fwd <- cpp_test_leapfrog(q0, p0, 0.1, 12L)
  back <- cpp_test_leapfrog(fwd$q, -fwd$p, 0.1, 12L)
  expect_equal(back$q, q0, tolerance = 1e-10)
  expect_equal(-back$p, p0, tolerance = 1e-10)
})

test_that("the Polya-Gamma conditional updates are the posteriors they claim", {
  set.seed(7)
  N <- 40L; p <- 2L
  X <- cbind(1, rnorm(N))
  omega <- runif(N, 0.5, 1.5)
  kappa <- rnorm(N)
  re_contrib <- rep(0, N)
  prior_sd <- 3

  # beta | omega is Gaussian with precision X' Omega X + I / sd^2. The update
  # draws, so what is checked is the moments of many draws.
  V <- solve(crossprod(X, omega * X) + diag(1 / prior_sd^2, p))
  m <- as.numeric(V %*% crossprod(X, kappa))
  draws <- replicate(400, cpp_test_pg_update_beta(kappa, omega, X, re_contrib,
                                                  prior_sd))
  expect_equal(dim(draws), c(p, 400L))
  se <- sqrt(diag(V) / 400)
  expect_true(all(abs(rowMeans(draws) - m) < 8 * se))
  expect_true(all(abs(apply(draws, 1, var) / diag(V) - 1) < 0.3))

  g <- rep(1:4, length.out = N)
  re <- cpp_test_pg_update_re(kappa, omega, as.numeric(X %*% m),
                              as.integer(g), 4L, 1.5)
  expect_length(re, 4L)
  expect_true(all(is.finite(re)))

  s <- cpp_test_pg_update_sigma_re(rnorm(20, sd = 2), 2.5)
  expect_length(s, 1L)
  expect_true(s > 0 && is.finite(s))
})

test_that("the row-major 3-D flattener matches an explicit R reshape", {
  d1 <- 2L; d2 <- 3L; d3 <- 4L
  arr <- array(seq_len(d1 * d2 * d3), dim = c(d1, d2, d3))
  flat <- cpp_flatten_3d_rowmajor(arr, d1, d2, d3)
  expect_length(flat, d1 * d2 * d3)
  ref <- numeric(d1 * d2 * d3)
  for (i in seq_len(d1)) for (j in seq_len(d2)) for (k in seq_len(d3)) {
    ref[(i - 1) * d2 * d3 + (j - 1) * d3 + k] <- arr[i, j, k]
  }
  expect_equal(flat, ref)
})

test_that("the Chinese-restaurant-table sampler has the mean it advertises", {
  # The augmentation the negative binomial dispersion update draws its Gibbs
  # count from: the expected table count is sum_k r / (r + k) over k < y.
  y <- 12L; r <- 3.0
  expect_equal(cpp_crt_mean(y, r), sum(r / (r + seq_len(y) - 1)),
               tolerance = 1e-12)
  expect_equal(cpp_crt_mean(0L, r), 0)

  set.seed(19)
  d <- replicate(4000, cpp_sample_crt(y, r))
  expect_true(all(d >= 1 & d <= y))
  expect_equal(mean(d), cpp_crt_mean(y, r), tolerance = 0.15)

  yv <- c(0L, 3L, 12L, 7L)
  s <- cpp_sample_crt_sum(yv, r)
  expect_length(s, 1L)
  expect_true(s >= 0 && s <= sum(yv))
})

test_that("the Laplace sampler draws from the Gaussian its Hessian defines", {
  set.seed(23)
  H <- matrix(c(4, 1, 1, 2), 2, 2)
  mode <- c(-1, 3)
  d <- cpp_laplace_sample(mode, H, 20000L)
  expect_equal(dim(d), c(20000L, 2L))
  expect_equal(colMeans(d), mode, tolerance = 0.05)
  expect_equal(cov(d), solve(H), tolerance = 0.05, ignore_attr = TRUE)
})

test_that("the tgmrf registry reports a non-negative size", {
  n <- cpp_tgmrf_registry_size()
  expect_length(n, 1L)
  expect_true(is.numeric(n) && n >= 0)
})
