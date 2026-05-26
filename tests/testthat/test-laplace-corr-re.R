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


# Independent reference for the per-group posterior covariance: assemble the
# full joint precision H at the fit's mode using the SAME working weights the
# kernel uses (glmm_weights), add the RE prior precision per group, the weak
# beta-prior ridge (DEFAULT_TAU_BETA), and the uniform 1e-10 ridge, invert, and
# read off each group's 2x2 diagonal block of solve(H). This is the FULL
# inverse block (fixed effects + other groups marginalized out), which is what
# the kernel returns.
ref_cov_blocks <- function(d, Sigma, fit) {
  DEFAULT_TAU_BETA <- 1e-4   # BetaPrior() default (laplace_re_priors.h)
  UNIFORM_RIDGE    <- 1e-10  # LAPLACE_UNIFORM_RIDGE (laplace_cholesky.h)
  p <- ncol(d$X)

  beta_hat <- fit$mode[seq_len(p)]
  u_hat    <- matrix(fit$mode[-(seq_len(p))], ncol = 2, byrow = TRUE)  # G x 2
  eta <- as.numeric(d$X %*% beta_hat) + rowSums(d$Z * u_hat[d$grp, ])
  W   <- tulpa:::glmm_weights(eta, "binomial", rep(1L, d$N))

  ii <- rep(seq_len(d$N), each = 2L)
  jj <- rep((d$grp - 1L) * 2L, each = 2L) + rep(1:2, d$N)
  Zs <- Matrix::sparseMatrix(i = ii, j = jj, x = as.numeric(t(d$Z)),
                             dims = c(d$N, 2L * d$G))
  Dfull <- cbind(Matrix::Matrix(d$X, sparse = TRUE), Zs)   # N x (p + 2G)

  H <- as.matrix(Matrix::crossprod(Dfull, W * Dfull))
  diag(H)[seq_len(p)] <- diag(H)[seq_len(p)] + DEFAULT_TAU_BETA
  Q <- solve(Sigma)
  for (g in seq_len(d$G)) {
    idx <- p + (g - 1L) * 2L + (1:2)
    H[idx, idx] <- H[idx, idx] + Q
  }
  diag(H) <- diag(H) + UNIFORM_RIDGE
  Cov <- solve(H)

  lapply(seq_len(d$G), function(g) {
    idx <- p + (g - 1L) * 2L + (1:2)
    unname(Cov[idx, idx])
  })
}

test_that("return_re_cov gives per-group posterior covariance (dense path)", {
  skip_on_cran()
  Sig <- matrix(c(0.8, 0.3, 0.3, 0.5), 2)
  d <- sim_corr(11L, G = 40L, npg = 20L, Sigma = Sig)   # n_x = 2 + 80 = 82 (dense)
  f <- tulpa_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X,
    re_list = re_term(d, L = t(chol(Sig))), family = "binomial",
    max_iter = 200L, tol = 1e-10, return_hessian = FALSE, return_re_cov = TRUE
  )

  expect_length(f$cov_blocks, d$G)
  expect_equal(dim(f$cov_blocks[[1]]), c(2L, 2L))
  # each block symmetric positive-definite
  expect_true(all(vapply(f$cov_blocks, function(B) isSymmetric(unname(B)), logical(1))))
  expect_true(all(vapply(f$cov_blocks, function(B) all(eigen(B, TRUE,
                          only.values = TRUE)$values > 0), logical(1))))

  ref <- ref_cov_blocks(d, Sig, f)
  err <- max(vapply(seq_len(d$G),
                    function(g) max(abs(f$cov_blocks[[g]] - ref[[g]])), numeric(1)))
  expect_lt(err, 1e-8)
})

test_that("return_re_cov matches the full-inverse blocks on the sparse path", {
  skip_on_cran()
  Sig <- matrix(c(0.8, 0.3, 0.3, 0.5), 2)
  d <- sim_corr(12L, G = 120L, npg = 8L, Sigma = Sig)   # n_x = 2 + 240 = 242 (sparse)
  f <- tulpa_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X,
    re_list = re_term(d, L = t(chol(Sig))), family = "binomial",
    max_iter = 200L, tol = 1e-10, return_hessian = FALSE, return_re_cov = TRUE
  )

  expect_length(f$cov_blocks, d$G)
  ref <- ref_cov_blocks(d, Sig, f)
  err <- max(vapply(seq_len(d$G),
                    function(g) max(abs(f$cov_blocks[[g]] - ref[[g]])), numeric(1)))
  expect_lt(err, 1e-8)
})

test_that("a single random slope (0 + x | g) uses its slope design, not intercept", {
  skip_on_cran()
  set.seed(77L)
  G <- 60L; npg <- 15L; N <- G * npg
  grp <- rep(seq_len(G), each = npg)
  x <- rnorm(N); X <- cbind(1, x)
  s_true <- 0.8
  u <- rnorm(G, 0, s_true)              # random SLOPE on x, no random intercept
  eta <- as.numeric(X %*% c(-0.3, 0.7)) + x * u[grp]
  y <- rbinom(N, 1L, plogis(eta))

  # n_coefs == 1 with a supplied slope column -> random slope `(0 + x | g)`
  f_slope <- tulpa_laplace(
    y, rep(1L, N), X,
    re_list = list(list(idx = grp, n_groups = G, n_coefs = 1L,
                        Z = matrix(x, ncol = 1L), sigma = s_true)),
    family = "binomial", return_hessian = TRUE, max_iter = 200L, tol = 1e-10)
  # the same term mis-specified as a random intercept (no Z)
  f_int <- tulpa_laplace(
    y, rep(1L, N), X,
    re_list = list(list(idx = grp, n_groups = G, n_coefs = 1L, sigma = s_true)),
    family = "binomial", return_hessian = TRUE, max_iter = 200L, tol = 1e-10)

  # the slope design is honoured: the fits genuinely differ, and the correctly
  # specified slope model fits the slope-structured data better (higher Laplace
  # marginal at the same sigma).
  expect_gt(f_slope$log_marginal, f_int$log_marginal)
  expect_false(isTRUE(all.equal(f_slope$mode, f_int$mode, tolerance = 1e-3)))

  # fixed effects recover under the correct design
  expect_equal(f_slope$mode[1:2], c(-0.3, 0.7), tolerance = 0.3)
})


test_that("return_re_cov is rejected on the spatial path", {
  skip_on_cran()
  d <- sim_corr(3L, G = 10L, npg = 10L)
  expect_error(
    tulpa_laplace(
      y = d$y, n_trials = rep(1L, d$N), X = d$X,
      re_list = re_term(d, L = t(chol(d$Sigma))), family = "binomial",
      spatial = list(type = "spde"), return_re_cov = TRUE
    ),
    "return_re_cov"
  )
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
