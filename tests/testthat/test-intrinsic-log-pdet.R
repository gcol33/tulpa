# The generalized log-determinant every intrinsic field density carries
# (src/intrinsic_log_pdet.h), against closed forms and eigenvalues, and the
# temporal random-walk evidence against the exact Gaussian marginal.

.ilp_eig <- function(Q) {
  e <- eigen(Q, symmetric = TRUE, only.values = TRUE)$values
  sum(log(e[e > 1e-9 * max(e)]))
}

.ilp_rw_Q <- function(T, order, cyclic) {
  D <- if (order == 1) diff(diag(T)) else diff(diag(T), differences = 2)
  if (cyclic) {
    D <- if (order == 1) rbind(D, replace(numeric(T), c(T, 1), c(-1, 1)))
         else rbind(D, replace(numeric(T), c(T - 1, T, 1), c(1, -2, 1)),
                    replace(numeric(T), c(T, 1, 2), c(1, -2, 1)))
  }
  crossprod(D)
}

test_that("a random walk's generalized log-determinant has its closed form", {
  load_cpp_fixture("helpers")
  for (T in c(5L, 12L)) {
    expect_equal(cpp_test_rw_log_pdet(T, 1L, FALSE), log(T), tolerance = 1e-10)
    expect_equal(cpp_test_rw_log_pdet(T, 1L, TRUE), 2 * log(T), tolerance = 1e-10)
    expect_equal(cpp_test_rw_log_pdet(T, 2L, FALSE),
                 log(T^2 * (T^2 - 1) / 12), tolerance = 1e-10)
    expect_equal(cpp_test_rw_log_pdet(T, 2L, TRUE), 4 * log(T), tolerance = 1e-10)
    for (o in 1:2) for (cy in c(FALSE, TRUE)) {
      expect_equal(cpp_test_rw_log_pdet(T, o, cy),
                   .ilp_eig(.ilp_rw_Q(T, o, cy)), tolerance = 1e-8)
    }
  }
})

# A covariance conditioned on the sum of its coordinates being zero.
.ilp_cond1 <- function(C) { c1 <- rowSums(C); C - tcrossprod(c1) / sum(c1) }

test_that("a nested RW1 / cyclic RW2 log marginal is the exact Gaussian evidence", {
  skip_on_cran()
  load_cpp_fixture("helpers")
  set.seed(4)
  T <- 15L
  t_idx <- rep(seq_len(T), each = 3L); n <- length(t_idx); x <- rnorm(n)
  y <- 0.2 + 0.4 * x + sin(t_idx / 3) + rnorm(n)
  X <- cbind(1, x); Z <- outer(t_idx, seq_len(T), "==") * 1
  s0 <- cpp_test_default_sigma_beta()
  ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
              prune = FALSE, var_of_means_consistency = FALSE,
              skew_correct = FALSE, progress = FALSE)
  for (case in list(list("rw1", FALSE), list("rw1", TRUE), list("rw2", TRUE))) {
    tau <- 3
    f <- tulpa_nested_laplace(y, rep(1L, n), X,
      prior = list(type = case[[1]], temporal_idx = t_idx, n_times = T,
                   cyclic = case[[2]], tau_grid = tau),
      family = "gaussian", phi = 1, hyperprior = "flat", control = ctl)
    # The walk's augmented precision tau (Q + 1 1' / T), its covariance
    # conditioned on the level row 1'u = 0: the level is a hard constraint and
    # sits in the intercept (gcol33/tulpa#1053).
    C <- .ilp_cond1(solve(.ilp_rw_Q(T, if (case[[1]] == "rw1") 1 else 2,
                                    case[[2]]) + 1 / T) / tau)
    S <- diag(n) + s0^2 * X %*% t(X) + Z %*% C %*% t(Z)
    ex <- -0.5 * (n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
                    sum(y * solve(S, y)))
    expect_equal(f$log_marginal, ex, tolerance = 1e-8)
  }
})

test_that("the multi-block driver's RW1 log marginal is the exact Gaussian evidence", {
  skip_on_cran()
  load_cpp_fixture("helpers")
  set.seed(6)
  T <- 12L
  t_idx <- rep(seq_len(T), each = 3L); n <- length(t_idx); x <- rnorm(n)
  g <- rep(1:3, length.out = n)
  y <- 0.2 + 0.4 * x + cos(t_idx / 2) + rnorm(n)
  X <- cbind(1, x); Z <- outer(t_idx, seq_len(T), "==") * 1
  Zr <- outer(g, 1:3, "==") * 1
  s0 <- cpp_test_default_sigma_beta()
  ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE,
              prune = FALSE, var_of_means_consistency = FALSE,
              skew_correct = FALSE, progress = FALSE)
  f <- tulpa_nested_laplace(y, rep(1L, n), X,
    prior = list(list(type = "rw1", temporal_idx = t_idx, n_times = T,
                      tau_grid = 2),
                 list(type = "iid", obs_idx = g, n_units = 3L, sigma_grid = 0.5)),
    family = "gaussian", phi = 1, hyperprior = "flat", control = ctl)
  C <- .ilp_cond1(solve(.ilp_rw_Q(T, 1, FALSE) + 1 / T) / 2)
  S <- diag(n) + s0^2 * X %*% t(X) + Z %*% C %*% t(Z) + 0.25 * Zr %*% t(Zr)
  ex <- -0.5 * (n * log(2 * pi) + as.numeric(determinant(S)$modulus) +
                  sum(y * solve(S, y)))
  expect_equal(f$log_marginal, ex, tolerance = 1e-8)
})
