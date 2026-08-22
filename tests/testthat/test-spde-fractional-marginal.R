# cpp_spde_fractional_logmarginal: the log-marginal for the fractional rSPDE
# path (R/rational_spde.R), evaluated at a mode that came from
# cpp_laplace_fit_spde_precomputed.
#
# Three things it has to get right, all of which used to be written a second
# way here rather than read off the engine (gcol33/tulpa#440):
#
#   * the binomial log-density. Materialising pi = 1 / (1 + exp(-eta)) and then
#     taking log(pi) / log1p(-pi) returns -Inf for any row with y > 0 once |eta|
#     passes about 40, because pi rounds to exactly 0 or 1 in double. That takes
#     the whole (range, sigma) cell to -Inf on a representation choice rather
#     than on the data, and confident eta is routine for an occupancy-style fit.
#   * the lumped-mass inverse. 1 / C0sub[j] on a zero-mass FEM node is Inf, the
#     quadratic form is then NaN, and the R tryCatch wrappers around this call
#     catch conditions rather than non-finite returns. Every other SPDE assembly
#     floors it (spde_zero_mass.h).
#   * the lengths and shapes R hands it. Rcpp's operator[] is unchecked.
#
# The reference below rebuilds the whole marginal in R -- the determinant lemma
# on B, the quadratic form through the Pl matvec, the stable binomial kernel --
# so it is an outside arbiter of the value, not only of its finiteness.

.frac_fixture <- function(seed = 7L, n = 12L, n_sub = 8L,
                          c0_zero = FALSE, big_eta = FALSE) {
  set.seed(seed)
  X  <- cbind(1, round(rnorm(n), 3))
  Pl <- Matrix::Diagonal(n_sub, 1 + runif(n_sub)) +
        Matrix::sparseMatrix(i = 2:n_sub, j = 1:(n_sub - 1),
                             x = rep(0.3, n_sub - 1), dims = c(n_sub, n_sub))
  Pl <- as(Pl, "CsparseMatrix")
  A <- Matrix::sparseMatrix(
    i = rep(1:n, each = 2),
    j = as.integer(((seq_len(2 * n) * 3L) %% n_sub) + 1L),
    x = round(runif(2 * n, 0.2, 1), 3), dims = c(n, n_sub))
  C0 <- round(runif(n_sub, 0.5, 2), 3)
  if (c0_zero) C0[3] <- 0
  list(y = as.numeric(rbinom(n, 3, 0.5)), X = X, A = A, Pl = Pl, C0 = C0,
       beta = if (big_eta) c(45, 0) else c(0.3, -0.4),
       x    = if (big_eta) rep(0, n_sub) else round(rnorm(n_sub, 0, 0.5), 3),
       nt = rep(3L, n), n = n, n_sub = n_sub)
}

.frac_ref <- function(f, tau_beta = 1e-4, wt = NULL) {
  n <- f$n
  if (is.null(wt)) wt <- rep(1, n)
  Mt <- Matrix::solve(f$Pl, Matrix::t(f$A))
  B  <- as.matrix(Matrix::t(Mt) %*% Matrix::Diagonal(x = f$C0) %*% Mt) +
        (f$X %*% t(f$X)) / tau_beta
  B  <- 0.5 * (B + t(B))
  eta <- as.numeric(f$X %*% f$beta + f$A %*% f$x)
  # The engine's kernel: never forms the probability.
  kern <- function(y, nn, e) {
    if (e > 0) y * e - nn * e - nn * log1p(exp(-e)) else y * e - nn * log1p(exp(e))
  }
  pr <- plogis(eta)
  w  <- wt * f$nt * pr * (1 - pr)
  loglik <- sum(wt * (lchoose(f$nt, f$y) + mapply(kern, f$y, f$nt, eta)))
  Plx    <- as.numeric(f$Pl %*% f$x)
  c0inv  <- ifelse(f$C0 > 1e-15, 1 / f$C0, 0)
  quad   <- tau_beta * sum(f$beta^2) + sum(Plx^2 * c0inv)
  sw     <- sqrt(w)
  G      <- diag(n) + (sw %o% sw) * B
  loglik - 0.5 * quad - 0.5 * as.numeric(determinant(G, logarithm = TRUE)$modulus)
}

.frac_call <- function(f, tau_beta = 1e-4, wt = NULL, off = NULL, nt = NULL,
                       X = NULL, A = NULL, Pl = NULL, beta = NULL, x = NULL) {
  cpp_spde_fractional_logmarginal(
    y = f$y, X = if (is.null(X)) f$X else X,
    A_eff = if (is.null(A)) f$A else A,
    Pl = if (is.null(Pl)) f$Pl else Pl, C0sub = f$C0,
    family = "binomial", phi = 1,
    beta_hat = if (is.null(beta)) f$beta else beta,
    x_hat = if (is.null(x)) f$x else x,
    n_trials = if (is.null(nt)) f$nt else nt,
    offset_nullable = off, tau_beta = tau_beta, weights_nullable = wt)
}

test_that("the fractional marginal reproduces an independent R reference", {
  f <- .frac_fixture()
  expect_equal(.frac_call(f), .frac_ref(f), tolerance = 1e-10)
  wv <- c(rep(1, 6), rep(0.5, 6))
  expect_equal(.frac_call(f, wt = wv), .frac_ref(f, wt = wv), tolerance = 1e-10)
})

test_that("a confident binomial fit is finite, where a materialised pi is not", {
  f <- .frac_fixture(big_eta = TRUE)
  eta <- as.numeric(f$X %*% f$beta + f$A %*% f$x)
  expect_gt(min(abs(eta)), 40)          # the fixture is in the regime
  got <- .frac_call(f)
  expect_true(is.finite(got))
  expect_equal(got, .frac_ref(f), tolerance = 1e-10)

  # The negative control: the form this used to carry returns -Inf on every row
  # of this fixture, so the whole cell was discarded.
  pi_ <- 1 / (1 + exp(-eta))
  naive <- sum(lchoose(f$nt, f$y) + f$y * log(pi_) + (f$nt - f$y) * log1p(-pi_))
  expect_false(is.finite(naive))
})

test_that("a zero-mass FEM node contributes 0 rather than making the cell NaN", {
  f <- .frac_fixture(c0_zero = TRUE)
  expect_true(any(f$C0 == 0))
  got <- .frac_call(f)
  expect_true(is.finite(got))
  expect_equal(got, .frac_ref(f), tolerance = 1e-10)

  # Dividing outright is the alternative, and it is NaN, which the R tryCatch
  # wrappers around this call do not catch.
  Plx <- as.numeric(f$Pl %*% f$x)
  expect_false(is.finite(sum(Plx^2 / f$C0)))

  # The floor is SPDE_C0_EPS, not exact zero: a node at 1e-16 is the same orphan
  # and gives the same marginal, where 1 / c0 would put 1e16 into the quadratic
  # form. The node still carries its A_eff weight into eta either way -- only
  # its term in x' Q x drops.
  f_eps <- f
  f_eps$C0[3] <- 1e-16
  expect_equal(.frac_call(f_eps), .frac_call(f), tolerance = 1e-12)
})

test_that("the lengths and shapes R hands it are checked", {
  f <- .frac_fixture()
  expect_error(.frac_call(f, off = rnorm(3)), "offset length")
  expect_error(.frac_call(f, nt = rep(3L, 3)), "n_trials length")
  expect_error(.frac_call(f, wt = rep(1, 5)), "weights length")
  expect_error(.frac_call(f, X = f$X[1:5, , drop = FALSE]), "nrow", fixed = TRUE)
  expect_error(.frac_call(f, A = f$A[, 1:4, drop = FALSE]), "A_eff must be",
               fixed = TRUE)
  expect_error(.frac_call(f, Pl = f$Pl[1:4, 1:4, drop = FALSE]), "Pl must be",
               fixed = TRUE)
  expect_error(.frac_call(f, beta = c(0.1)), "beta_hat", fixed = TRUE)
  expect_error(.frac_call(f, x = rep(0, 3)), "x_hat", fixed = TRUE)
})

test_that("an unsupported family is named rather than silently mis-evaluated", {
  f <- .frac_fixture()
  expect_error(
    cpp_spde_fractional_logmarginal(
      y = f$y, X = f$X, A_eff = f$A, Pl = f$Pl, C0sub = f$C0,
      family = "gamma", phi = 1, beta_hat = f$beta, x_hat = f$x,
      n_trials = f$nt),
    "gaussian, poisson, binomial")
})
