# Correlated random slopes (1 + x | g) on the Laplace engine (gcol33/tulpa#28).
# tulpa_laplace() now accepts a per-term covariance via `L` (lower-triangular
# Cholesky, Sigma = L L') or `cov`; the off-diagonal enters both the joint
# Hessian (mode finding, via the C++ kernel) and the marginal fixed-effect SE.

sim_corr <- function(seed, G = 60L, npg = 15L, beta = c(-0.3, 0.7),
                     Sigma = matrix(c(0.9^2, 0.6 * 0.9 * 0.5,
                                      0.6 * 0.9 * 0.5, 0.5^2), 2)) {
  set.seed(seed)
  N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  X <- cbind(1, x)
  Z <- cbind(1, x)                          # (1 + x | g)
  u <- t(t(chol(Sigma)) %*% matrix(rnorm(2 * G), nrow = 2))   # G x 2 true RE
  eta <- as.numeric(X %*% beta) + rowSums(Z * u[grp, ])
  y <- rbinom(N, 1L, plogis(eta))
  list(y = y, X = X, Z = Z, grp = grp, G = G, N = N, beta = beta,
       Sigma = Sigma, u = u)
}

re_term <- function(d, ..., n_coefs = 2L) {
  list(list(idx = d$grp, n_groups = d$G, n_coefs = n_coefs, Z = d$Z, ...))
}

fit_lap <- function(d, re_list) {
  tulpa_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X,
    re_list = re_list, family = "binomial",
    max_iter = 200L, tol = 1e-10, return_hessian = TRUE
  )
}


test_that("L, cov, and a diagonal sigma agree when the covariance is the same", {
  skip_on_cran()
  d <- sim_corr(1L)
  s <- c(0.7, 0.4)
  L0 <- diag(s)                               # zero correlation

  f_sigma <- fit_lap(d, re_term(d, sigma = s))
  f_L     <- fit_lap(d, re_term(d, L = L0))
  f_cov   <- fit_lap(d, re_term(d, cov = L0 %*% t(L0)))

  # Same covariance -> same mode and same marginal fixed-effect Hessian,
  # whichever way it is specified.
  expect_equal(f_L$mode,   f_sigma$mode, tolerance = 1e-7)
  expect_equal(f_cov$mode, f_sigma$mode, tolerance = 1e-7)
  expect_equal(f_L$H_beta,   f_sigma$H_beta, tolerance = 1e-6)
  expect_equal(f_cov$H_beta, f_sigma$H_beta, tolerance = 1e-6)
})


test_that("correlated fit recovers fixed effects and the RE structure", {
  skip_on_cran()
  rec <- t(vapply(1:5, function(sd) {
    d <- sim_corr(100L + sd)
    f <- fit_lap(d, re_term(d, L = t(chol(d$Sigma))))
    re <- f$mode[-(1:2)]
    u_hat <- cbind(re[seq(1, by = 2, length.out = d$G)],
                   re[seq(2, by = 2, length.out = d$G)])
    c(b0 = f$mode[1], b1 = f$mode[2], conv = f$converged,
      cor_int = cor(u_hat[, 1], d$u[, 1]),
      cor_slp = cor(u_hat[, 2], d$u[, 2]))
  }, numeric(5)))

  expect_true(all(rec[, "conv"] == 1))
  expect_equal(mean(rec[, "b1"]), 0.7, tolerance = 0.12)
  expect_lt(abs(mean(rec[, "b0"]) - (-0.3)), 0.15)
  expect_gt(mean(rec[, "cor_int"]), 0.6)
  expect_gt(mean(rec[, "cor_slp"]), 0.5)
})


test_that("the marginal SE uses the off-diagonal precision (tulpa#28)", {
  skip_on_cran()
  Sig <- matrix(c(1.0, 0.8, 0.8, 1.0), 2)        # strong correlation
  d <- sim_corr(7L, Sigma = Sig)

  f_corr <- fit_lap(d, re_term(d, L = t(chol(Sig))))
  f_diag <- fit_lap(d, re_term(d, sigma = sqrt(diag(Sig))))  # drops rho

  # Dropping rho changes the marginal fixed-effect curvature: the two must
  # differ materially, so the off-diagonal genuinely propagates.
  expect_false(isTRUE(all.equal(f_corr$H_beta, f_diag$H_beta, tolerance = 1e-3)))

  # Independent Schur with the full block precision Q = Sigma^{-1}, evaluated
  # at the correlated fit's mode, must reproduce f_corr$H_beta exactly.
  beta_hat <- f_corr$mode[1:2]
  re_hat   <- f_corr$mode[-(1:2)]
  u_hat    <- matrix(re_hat, ncol = 2, byrow = TRUE)        # G x 2
  eta <- as.numeric(d$X %*% beta_hat) + rowSums(d$Z * u_hat[d$grp, ])
  W   <- tulpa:::glmm_weights(eta, "binomial", rep(1L, d$N))

  ii <- rep(seq_len(d$N), each = 2L)
  jj <- rep((d$grp - 1L) * 2L, each = 2L) + rep(1:2, d$N)
  Zs <- Matrix::sparseMatrix(i = ii, j = jj, x = as.numeric(t(d$Z)),
                             dims = c(d$N, 2L * d$G))
  Q    <- solve(Sig)
  Dinv <- Matrix::bdiag(rep(list(Matrix::Matrix(Q, sparse = TRUE)), d$G))
  ZtWZ <- Matrix::crossprod(Zs, W * Zs)
  XtWZ <- crossprod(d$X, as.matrix(W * Zs))
  H_indep <- crossprod(d$X, W * d$X) -
    XtWZ %*% solve(as.matrix(ZtWZ + Dinv)) %*% t(XtWZ)

  expect_equal(unname(f_corr$H_beta), unname(as.matrix(H_indep)),
               tolerance = 1e-5)
})


test_that("a malformed L / cov is rejected", {
  skip_on_cran()
  d <- sim_corr(3L, G = 10L, npg = 10L)
  expect_error(
    fit_lap(d, re_term(d, L = matrix(1, 3, 3))),
    "must be 2 x 2"
  )
  # cov that is not positive definite -> chol fails loudly rather than silently.
  expect_error(
    fit_lap(d, re_term(d, cov = matrix(c(1, 2, 2, 1), 2)))
  )
})
