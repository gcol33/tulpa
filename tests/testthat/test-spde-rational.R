# Rational-SPDE layer. The BRASIL roots must reproduce the Matern spectral
# density on the FEM spectrum, and the covariance-based assembly built from them
# must reproduce the fractional field's covariance. Verified mode by mode on a
# 1D periodic (circulant) mesh, where the generalized eigenvalues of CiL are
# known in closed form.

skip_on_cran()

spde_mode_eigs <- function(n = 200L, kappa = 7) {
  h <- 1 / n
  k <- 0:(n - 1)
  kappa^2 + (2 - 2 * cos(2 * pi * k / n)) / h^2   # eigenvalues of CiL
}

test_that("rational-SPDE roots reproduce the Matern spectral density l^{-2 beta}", {
  skip_on_cran()
  l <- spde_mode_eigs()
  lf <- l / max(l)                                # L / l_max-normalized axis
  ratio <- min(l) / max(l)
  for (nu in c(0.5, 1.5)) {
    beta <- (nu + 1) / 2
    rr <- tulpa:::.spde_rational_roots(order = 4L, beta = beta,
                                       spectrum_ratio = ratio)
    # Field-mode variance (up to a constant): (b_k / a_k)^2 with
    # b_k = prod(1 - lf rc), a_k = lf^{m_beta-1} prod(1 - lf rb).
    bk <- vapply(lf, function(x) prod(1 - x * rr$rc), numeric(1))
    ak <- lf^(rr$m_beta - 1) * vapply(lf, function(x) prod(1 - x * rr$rb), numeric(1))
    var_model <- (bk / ak)^2
    target <- lf^(-2 * beta)
    # Spectral SHAPE (constant absorbed by tau / sigma) must match closely.
    shape_cor <- cor(log(var_model), log(target))
    expect_gt(shape_cor, 0.999)
  }
})

test_that("rational-SPDE root counts and structure follow the rSPDE convention", {
  skip_on_cran()
  rr <- tulpa:::.spde_rational_roots(order = 3L, beta = 0.75, spectrum_ratio = 1e-3)
  expect_length(rr$rb, 3L)            # balanced (m, m): m denominator factors
  expect_length(rr$rc, 3L)            # m numerator factors
  expect_equal(rr$m_beta, 1L)         # floor(0.75) -> max(1, 0) = 1
  expect_true(is.finite(rr$scale))
  # Roots real (the minimax approximation of x^{-beta} on a positive interval
  # has real interlacing zeros/poles), so the (1 - l r) factors are real.
  expect_true(all(is.finite(rr$rb)) && all(is.finite(rr$rc)))
})

test_that("the assembled components sum to the fractional Matern covariance", {
  skip_on_cran()
  # 1D periodic (circulant) FEM: C0 = h, G = (1/h) circ(2, -1, ..., -1). The
  # field covariance Pr Q^-1 Pr' = sum_k Q_k^-1 shares the Fourier eigenbasis
  # with L = C^-1 G + kappa^2, so its eigenvalues are its first row's FFT and
  # must equal tau^-2 l^-alpha / h, the covariance of the exact fractional
  # field on this mesh, up to the rational approximation error.
  n <- 128L; h <- 1 / n; kappa <- 6
  C0 <- rep(h, n)
  G <- Matrix::bandSparse(n, n, c(-1, 0, 1),
                          list(rep(-1 / h, n - 1), rep(2 / h, n), rep(-1 / h, n - 1)))
  G <- as(G, "CsparseMatrix"); G[1, n] <- -1 / h; G[n, 1] <- -1 / h
  l <- kappa^2 + (2 - 2 * cos(2 * pi * (0:(n - 1)) / n)) / h^2   # eigenvalues of CiL
  tau <- 0.7

  for (nu in c(0.5, 1.5)) {
    asm <- tulpa:::.spde_rational_assemble(C0, G, kappa = kappa, tau = tau,
                                           nu = nu, order = 3L, d = 2,
                                           l_max = kappa^2 + 4 / h^2)
    expect_equal(asm$n_comp, 4L)
    expect_true(Matrix::isSymmetric(asm$Q))
    expect_gt(min(eigen(as.matrix(asm$Q), symmetric = TRUE,
                        only.values = TRUE)$values), 0)
    Sigma <- as.matrix(asm$Pr %*% Matrix::solve(asm$Q, Matrix::t(asm$Pr)))
    var_mode <- Re(fft(Sigma[1, ]))
    target <- l^(-(nu + 1)) / (tau^2 * h)
    # Relative error per Fourier mode, over the modes carrying all but 1e-6 of
    # the field variance, and of the total variance.
    carry <- target > 1e-6 * max(target)
    expect_lt(max(abs(var_mode[carry] / target[carry] - 1)), 0.05)
    expect_lt(abs(sum(var_mode) / sum(target) - 1), 0.005)
  }
})

test_that("higher rational order lowers the approximation error", {
  skip_on_cran()
  errs <- vapply(2:5, function(m)
    tulpa:::.spde_rational_roots(order = m, beta = 0.75,
                                 spectrum_ratio = 1e-2)$error, numeric(1))
  expect_true(all(diff(errs) < 0))
})

test_that("the precomputed C++ SPDE fit recovers a fractional-Matern field", {
  skip_on_cran()
  set.seed(20)
  n <- 100L; h <- 1 / n; kappa <- 8
  C0 <- rep(h, n)
  G <- Matrix::bandSparse(n, n, c(-1, 0, 1),
                          list(rep(-1 / h, n - 1), rep(2 / h, n), rep(-1 / h, n - 1)))
  G <- as(G, "CsparseMatrix"); G[1, n] <- -1 / h; G[n, 1] <- -1 / h

  nu <- 0.5
  asm <- tulpa:::.spde_rational_assemble(C0, G, kappa = kappa, tau = 1,
                                         nu = nu, order = 4L, d = 2,
                                         l_max = kappa^2 + 4 / h^2)
  n_lat <- ncol(asm$Q)
  # Simulate the true field u = Pr x, x ~ N(0, Q^{-1}).
  R <- chol(as.matrix(asm$Q))                    # upper, R' R = Q
  x_true <- backsolve(R, rnorm(n_lat))           # Cov = (R' R)^{-1} = Q^{-1}
  u_true <- as.numeric(asm$Pr %*% x_true)
  u_true <- u_true / sd(u_true)                  # standardize the field scale
  beta0 <- 0.2
  # Gaussian observations (Laplace is exact) at high SNR -> clean field recovery.
  y <- beta0 + u_true + rnorm(n, 0, 0.3)

  # Obs at every mesh node (A = I) -> A_eff = Pr. Pass the FULL precision.
  Qg   <- as(asm$Q, "generalMatrix")
  Prg  <- as(asm$Pr, "CsparseMatrix")
  fit <- tulpa:::cpp_laplace_fit_spde_precomputed(
    y = as.numeric(y), n_trials = rep(1L, n), X = matrix(1, n, 1),
    re_idx = rep(0, n), n_re_groups = 0L, sigma_re = 1.0,
    n_obs = n, n_mesh = n_lat,
    Q_p = Qg@p, Q_i = Qg@i, Q_x = Qg@x,
    Aeff_x = Prg@x, Aeff_i = Prg@i, Aeff_p = Prg@p,
    family = "gaussian", phi = 0.09)

  expect_true(fit$converged)
  expect_lt(fit$n_iter, 100L)
  x_hat <- fit$mode[-1]                           # drop the intercept
  u_hat <- as.numeric(asm$Pr %*% x_hat)
  expect_gt(cor(u_hat, u_true), 0.85)
})
