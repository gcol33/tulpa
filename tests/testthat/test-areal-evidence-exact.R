# The log marginal likelihood an areal field reports, against the closed form a
# Gaussian response gives at a fixed outer cell. The ICAR density is the
# augmented one, precision tau * (Q + 1 1' / J), whose normalizer carries the
# graph's generalized determinant |Q|_+ (matrix-tree: J * det of any principal
# (J - 1)-minor); BYM2's structured component carries the same at unit
# precision. With the kernels' default fixed-effect prior N(0, s^2 I)
# (DEFAULT_SIGMA_BETA), the evidence is the density of y under
# N(0, phi I + s^2 X X' + Z C Z'), C the field covariance.

.ae_graph <- function() {
  K <- 30L
  A <- matrix(0, K, K)
  for (i in seq_len(K)) { j <- if (i < K) i + 1L else 1L; A[i, j] <- A[j, i] <- 1 }
  A[1, 15] <- A[15, 1] <- 1; A[5, 22] <- A[22, 5] <- 1
  A
}

.ae_fixture <- function(seed = 1) {
  set.seed(seed)
  A <- .ae_graph(); K <- nrow(A)
  idx <- rep(seq_len(K), each = 4L); n <- length(idx); x <- rnorm(n)
  list(A = A, K = K, idx = idx, n = n, X = cbind(1, x),
       y = 0.3 + 0.5 * x + rnorm(n), Z = outer(idx, seq_len(K), "==") * 1,
       csr = tulpa:::adjacency_to_csr_tulpa(A))
}

.ae_exact <- function(fx, C) {
  S <- diag(fx$n) + fx$X %*% diag(default_sigma_beta()^2, 2) %*% t(fx$X) +
    fx$Z %*% C %*% t(fx$Z)
  -0.5 * (fx$n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
            sum(fx$y * solve(S, fx$y)))
}

.ae_ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
                prune = FALSE, var_of_means_consistency = FALSE,
                skew_correct = FALSE, progress = FALSE)

.ae_block <- function(fx, ...) {
  c(list(spatial_idx = fx$idx, n_spatial_units = fx$K,
         adj_row_ptr = fx$csr$row_ptr, adj_col_idx = fx$csr$col_idx,
         n_neighbors = fx$csr$n_neighbors), list(...))
}

# The field covariance at unit precision: the inverse of the augmented
# precision Q + 1 1' / J, i.e. Q^+ + 1 1' / J -- the level carries its own unit
# variance and the intercept's prior is not touched by the field.
.ae_lap_pinv <- function(A) solve(diag(rowSums(A)) - A + 1 / nrow(A))

test_that("the ICAR normalizer carries the graph's generalized determinant", {
  # Path, cycle and a disconnected pair: |Q|_+ = J * (spanning trees) per
  # component, so a path of J nodes gives J, a cycle J^2, and components add.
  load_cpp_fixture("helpers")
  path <- matrix(0, 5, 5); for (i in 1:4) path[i, i + 1] <- path[i + 1, i] <- 1
  cyc <- path; cyc[1, 5] <- cyc[5, 1] <- 1
  two <- as.matrix(Matrix::bdiag(path, cyc))
  for (A in list(path, cyc, two)) {
    csr <- tulpa:::adjacency_to_csr_tulpa(A)
    n <- nrow(A)
    ev <- eigen(diag(rowSums(A)) - A, symmetric = TRUE, only.values = TRUE)$values
    L <- sum(ev < 1e-9)
    lp <- cpp_test_log_prior_icar(numeric(n), n, 1.0, csr$row_ptr,
                                          csr$col_idx, csr$n_neighbors)
    expect_equal(lp, 0.5 * sum(log(ev[ev > 1e-9])) - 0.5 * n * log(2 * pi),
                 tolerance = 1e-10)
    expect_identical(L, if (identical(A, two)) 2L else 1L)
  }
})

test_that("ICAR and BYM2 log marginals equal the exact Gaussian evidence", {
  skip_on_cran()
  fx <- .ae_fixture()
  Qp <- .ae_lap_pinv(fx$A)
  for (tau in c(1, 10)) {
    f <- tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X,
      prior = .ae_block(fx, type = "icar", tau_grid = tau),
      family = "gaussian", phi = 1, hyperprior = "flat", control = .ae_ctl)
    expect_equal(f$log_marginal, .ae_exact(fx, Qp / tau), tolerance = 1e-6)
  }
  sf <- tulpa:::.bym2_component_scaling(fx$A)$scale_factor
  for (sr in list(c(0.8, 0.5), c(1.5, 0.9))) {
    f <- tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X,
      prior = .ae_block(fx, type = "bym2", scale_factor = sf,
                        sigma_grid = sr[1], rho_grid = sr[2]),
      family = "gaussian", phi = 1, hyperprior = "flat", control = .ae_ctl)
    C <- sr[1]^2 * (sr[2] * sf^2 * Qp + (1 - sr[2]) * diag(fx$K))
    expect_equal(f$log_marginal, .ae_exact(fx, C), tolerance = 1e-6)
  }
})

test_that("the multi-block driver's areal log marginal is the exact evidence", {
  skip_on_cran()
  fx <- .ae_fixture(2)
  Qp <- .ae_lap_pinv(fx$A)
  set.seed(3)
  re <- sample(1:5, fx$n, TRUE)
  Zr <- outer(re, 1:5, "==") * 1
  iid <- list(type = "iid", obs_idx = re, n_units = 5L, sigma_grid = 0.7)
  f <- tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X,
    prior = list(.ae_block(fx, type = "icar", tau_grid = 2), iid),
    family = "gaussian", phi = 1, hyperprior = "flat", control = .ae_ctl)
  S <- diag(fx$n) + fx$X %*% diag(default_sigma_beta()^2, 2) %*% t(fx$X) +
    fx$Z %*% (Qp / 2) %*% t(fx$Z) + 0.49 * Zr %*% t(Zr)
  ex <- -0.5 * (fx$n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
                  sum(fx$y * solve(S, fx$y)))
  expect_equal(f$log_marginal, ex, tolerance = 1e-6)
  sf <- tulpa:::.bym2_component_scaling(fx$A)$scale_factor
  f <- tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X,
    prior = list(.ae_block(fx, type = "bym2", scale_factor = sf,
                           sigma_grid = 0.8, rho_grid = 0.5), iid),
    family = "gaussian", phi = 1, hyperprior = "flat", control = .ae_ctl)
  C <- 0.64 * (0.5 * sf^2 * Qp + 0.5 * diag(fx$K))
  S <- diag(fx$n) + fx$X %*% diag(default_sigma_beta()^2, 2) %*% t(fx$X) +
    fx$Z %*% C %*% t(fx$Z) + 0.49 * Zr %*% t(Zr)
  ex <- -0.5 * (fx$n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
                  sum(fx$y * solve(S, fx$y)))
  expect_equal(f$log_marginal, ex, tolerance = 1e-6)
})

test_that("an ICAR field at a vanishing scale has the non-spatial evidence", {
  skip_on_cran()
  # The field collapses onto zero as tau grows, so the evidence tends to the
  # same model without it, under the same fixed-effect prior.
  fx <- .ae_fixture(4)
  f <- tulpa_nested_laplace(fx$y, rep(1L, fx$n), fx$X,
    prior = .ae_block(fx, type = "icar", tau_grid = 1e7),
    family = "gaussian", phi = 1, hyperprior = "flat", control = .ae_ctl)
  expect_equal(f$log_marginal, .ae_exact(fx, matrix(0, fx$K, fx$K)),
               tolerance = 1e-4)
})

test_that("field_effects() on a nested ICAR / BYM2 fit is the exact posterior", {
  skip_on_cran()
  # One outer cell, Gaussian response: the per-unit field's posterior is the
  # Gaussian conditional of the field given y with the ICAR level constrained
  # to zero (the intercept carries it). x = (beta, phi[, theta]); tulpa()
  # solves under its own fixed-effect default.
  fx <- .ae_fixture(5)
  Lap <- diag(rowSums(fx$A)) - fx$A
  K <- fx$K; p <- 2L
  post_field <- function(Pprior, load, n_icar_start) {
    D <- cbind(fx$X, fx$Z %*% load$eta)
    P <- Pprior + crossprod(D)
    Cx <- solve(P)
    m <- Cx %*% crossprod(D, fx$y)
    a <- numeric(ncol(P)); a[n_icar_start + seq_len(K)] <- 1
    Ca <- Cx %*% a
    m <- m - Ca * sum(a * m) / sum(a * Ca)
    Cx <- Cx - Ca %*% t(Ca) / sum(a * Ca)
    list(mean = as.numeric(t(load$field) %*% m),
         sd = sqrt(diag(t(load$field) %*% Cx %*% load$field)))
  }
  d <- data.frame(y = fx$y, x = fx$X[, 2], region = fx$idx)
  f <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = 1,
             spatial = list(type = "icar", adjacency = fx$A, tau_grid = 3),
             mode = "nested_laplace", control = .ae_ctl)
  fe <- field_effects(f)
  tau_b <- 1 / tulpa:::.tulpa_prior_sd()^2
  Pprior <- as.matrix(Matrix::bdiag(diag(tau_b, p), 3 * (Lap + 1 / K)))
  I_K <- diag(K)
  ex <- post_field(Pprior, list(eta = cbind(I_K), field = rbind(matrix(0, p, K), I_K)), p)
  expect_identical(unique(fe$source), "mixture")
  expect_equal(fe$estimate, ex$mean, tolerance = 1e-6)
  expect_equal(fe$sd, ex$sd, tolerance = 1e-6)

  sf <- tulpa:::.bym2_component_scaling(fx$A)$scale_factor
  f2 <- tulpa(y ~ x + spatial(region), d, family = "gaussian", phi = 1,
              spatial = list(type = "bym2", adjacency = fx$A,
                             sigma_grid = 0.9, rho_grid = 0.6),
              mode = "nested_laplace", control = .ae_ctl)
  fe2 <- field_effects(f2)
  a <- 0.9 * sqrt(0.6) * sf; b <- 0.9 * sqrt(0.4)
  Pprior <- as.matrix(Matrix::bdiag(diag(tau_b, p), Lap + 1 / K, I_K))
  load <- list(eta = cbind(a * I_K, b * I_K),
               field = rbind(matrix(0, p, K), a * I_K, b * I_K))
  ex2 <- post_field(Pprior, load, p)
  expect_identical(unique(fe2$source), "mixture")
  # The engine moves the structured level into the intercept by centring the
  # augmented mode, which differs from conditioning it on the constraint by the
  # level's prior variance against the intercept's, here ~1e-6.
  expect_equal(fe2$estimate, ex2$mean, tolerance = 1e-5)
  expect_equal(fe2$sd, ex2$sd, tolerance = 1e-6)
})
