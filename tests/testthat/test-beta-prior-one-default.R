# One fixed-effect prior whatever the backend: tulpa() states its default
# (`.TULPA_PRIOR$beta_sd`) to the nested-Laplace kernels as to every other
# route, and a caller's `beta_prior` replaces it there too; a direct engine call
# with none solves under the kernels' own weak default. Checked on the evidence, whose dependence on the prior is exact for a
# Gaussian response: N(0, phi I + X S_b X' + Z C Z'), C the inverse of the
# augmented ICAR precision tau (Q + 1 1' / K).

.bp_fixture <- function(seed = 1) {
  set.seed(seed)
  K <- 20L
  A <- matrix(0, K, K)
  for (i in seq_len(K)) { j <- if (i < K) i + 1L else 1L; A[i, j] <- A[j, i] <- 1 }
  idx <- rep(seq_len(K), each = 4L); n <- length(idx); x <- rnorm(n)
  list(A = A, K = K, idx = idx, n = n, X = cbind(1, x),
       y = 0.3 + 0.5 * x + rnorm(n), Z = outer(idx, seq_len(K), "==") * 1,
       csr = tulpa:::adjacency_to_csr_tulpa(A),
       Qp = solve(diag(rowSums(A)) - A + 1 / K))
}

.bp_exact <- function(fx, Sb, C) {
  S <- diag(fx$n) + fx$X %*% Sb %*% t(fx$X) + fx$Z %*% C %*% t(fx$Z)
  -0.5 * (fx$n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
            sum(fx$y * solve(S, fx$y)))
}

.bp_ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
                prune = FALSE, var_of_means_consistency = FALSE,
                skew_correct = FALSE, progress = FALSE)

.bp_icar <- function(fx, tau) {
  list(type = "icar", spatial_idx = fx$idx, n_spatial_units = fx$K,
       adj_row_ptr = fx$csr$row_ptr, adj_col_idx = fx$csr$col_idx,
       n_neighbors = fx$csr$n_neighbors, tau_grid = tau)
}

test_that("a nested ICAR fit solves under the kernel default or the caller's prior", {
  skip_on_cran()
  fx <- .bp_fixture()
  s0 <- tulpa:::cpp_test_default_sigma_beta()
  fit <- function(beta_prior, prior = .bp_icar(fx, 2)) {
    tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X, prior = prior,
                         family = "gaussian", phi = 1, hyperprior = "flat",
                         beta_prior = beta_prior, control = .bp_ctl)$log_marginal
  }
  expect_equal(fit(NULL), .bp_exact(fx, diag(s0^2, 2), fx$Qp / 2),
               tolerance = 1e-8)
  bp <- list(mean = c(1, -0.5), sd = c(10, 0.7))
  # A non-zero prior mean shifts y's marginal mean by X m.
  fx_m <- fx; fx_m$y <- fx$y - as.numeric(fx$X %*% bp$mean)
  expect_equal(fit(bp), .bp_exact(fx_m, diag(bp$sd^2), fx$Qp / 2),
               tolerance = 1e-8)
  # The multi-block driver reads the same prior.
  iid <- list(type = "iid", obs_idx = rep(1:4, length.out = fx$n),
              n_units = 4L, sigma_grid = 0.6)
  Zr <- outer(iid$obs_idx, 1:4, "==") * 1
  lm_multi <- tulpa_nested_laplace(
    fx$y, rep(1L, fx$n), fx$X, prior = list(.bp_icar(fx, 2), iid),
    family = "gaussian", phi = 1, hyperprior = "flat",
    beta_prior = list(sd = 4), control = .bp_ctl)$log_marginal
  S <- diag(fx$n) + 16 * fx$X %*% t(fx$X) + fx$Z %*% (fx$Qp / 2) %*% t(fx$Z) +
    0.36 * Zr %*% t(Zr)
  expect_equal(lm_multi, -0.5 * (fx$n * log(2 * pi) +
                                   as.numeric(determinant(S)$modulus) +
                                   sum(fx$y * solve(S, fx$y))),
               tolerance = 1e-8)
})

test_that("a field at a vanishing scale has the non-spatial fit's evidence", {
  skip_on_cran()
  # Same formula, same default fixed-effect prior on both routes: the field
  # contributes nothing as tau grows, so the two evidences meet.
  fx <- .bp_fixture(2)
  d <- data.frame(y = fx$y, x = fx$X[, 2], region = fx$idx)
  f0 <- tulpa(y ~ x, d, family = "gaussian", phi = 1, mode = "laplace")
  f1 <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = 1,
              spatial = list(type = "icar", adjacency = fx$A, tau_grid = 1e8),
              mode = "nested_laplace", control = .bp_ctl)
  expect_equal(as.numeric(logLik(f1)), as.numeric(logLik(f0)), tolerance = 1e-6)
  f2 <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = 1,
              spatial = list(type = "icar", adjacency = fx$A, tau_grid = 1e8),
              beta_prior = list(sd = 7), mode = "nested_laplace",
              control = .bp_ctl)
  f0b <- tulpa(y ~ x, d, family = "gaussian", phi = 1, mode = "laplace",
               beta_prior = list(sd = 7))
  expect_equal(as.numeric(logLik(f2)), as.numeric(logLik(f0b)), tolerance = 1e-6)
  expect_equal(f2$beta_prior$sd, 7)
})
