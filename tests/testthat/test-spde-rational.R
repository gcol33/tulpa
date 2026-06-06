# Rational-SPDE coefficient layer (gcol33/tulpa#71, stage 2; gated, not wired).
# The rSPDE operator-based construction with BRASIL coefficients must reproduce
# the Matern spectral density l^{-2 beta} on the FEM spectrum. Verified mode by
# mode on a 1D periodic (circulant) mesh, where the generalized eigenvalues of
# CiL are known in closed form.

spde_mode_eigs <- function(n = 200L, kappa = 7) {
  h <- 1 / n
  k <- 0:(n - 1)
  kappa^2 + (2 - 2 * cos(2 * pi * k / n)) / h^2   # eigenvalues of CiL
}

test_that("rational-SPDE roots reproduce the Matern spectral density l^{-2 beta}", {
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
  rr <- tulpa:::.spde_rational_roots(order = 3L, beta = 0.75, spectrum_ratio = 1e-3)
  expect_length(rr$rb, 3L)            # balanced (m, m): m denominator factors
  expect_length(rr$rc, 3L)            # m numerator factors
  expect_equal(rr$m_beta, 1L)         # floor(0.75) -> max(1, 0) = 1
  expect_true(is.finite(rr$scale))
  # Roots real (the minimax approximation of x^{-beta} on a positive interval
  # has real interlacing zeros/poles), so the (1 - l r) factors are real.
  expect_true(all(is.finite(rr$rb)) && all(is.finite(rr$rc)))
})

test_that("the assembled precision yields a Matern field covariance", {
  # 1D periodic (circulant) FEM: C0 = h, G = (1/h) circ(2, -1, ..., -1). The
  # implied covariance Pr Q^{-1} Pr' shares the Fourier eigenbasis with L, so its
  # spectrum must track the Matern density l^{-2 beta}.
  n <- 128L; h <- 1 / n; kappa <- 6
  C0 <- rep(h, n)
  G <- Matrix::bandSparse(n, n, c(-1, 0, 1),
                          list(rep(-1 / h, n - 1), rep(2 / h, n), rep(-1 / h, n - 1)))
  G <- as(G, "CsparseMatrix"); G[1, n] <- -1 / h; G[n, 1] <- -1 / h
  l <- kappa^2 + (2 - 2 * cos(2 * pi * (0:(n - 1)) / n)) / h^2   # eigenvalues of CiL

  for (nu in c(0.5, 1.5)) {
    asm <- tulpa:::.spde_rational_assemble(C0, G, kappa = kappa, tau = 1,
                                           nu = nu, order = 4L, d = 2)
    # Q is a valid (PD, symmetric) sparse precision (relative PSD tolerance:
    # the spectrum spans many orders of magnitude).
    expect_true(Matrix::isSymmetric(asm$Q))
    ev <- eigen(as.matrix(asm$Q), symmetric = TRUE, only.values = TRUE)$values
    expect_gt(min(ev), -1e-8 * max(ev))
    # Per-mode field variance from the assembled Pl / Pr directly (no inverse):
    # Pl, Pr are circulant, so their Fourier eigenvalues are the FFT of the first
    # row, and the field-mode variance is |eig_Pr|^2 / (|eig_Pl|^2 / C0). This
    # must track the Matern density l^{-2 beta}.
    ePl <- Mod(fft(as.numeric(asm$Pl[1, ])))
    ePr <- Mod(fft(as.numeric(asm$Pr[1, ])))
    Qk  <- ePl^2 / C0[1]
    var_mode <- ePr^2 / Qk
    beta <- (nu + 2 / 2) / 2
    target <- (l / asm$l_max)^(-2 * beta)
    expect_gt(cor(log(var_mode), log(target)), 0.999)
  }
})

test_that("higher rational order lowers the approximation error", {
  errs <- vapply(2:5, function(m)
    tulpa:::.spde_rational_roots(order = m, beta = 0.75,
                                 spectrum_ratio = 1e-2)$error, numeric(1))
  expect_true(all(diff(errs) < 0))
})
