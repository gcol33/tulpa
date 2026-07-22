# Marginal fixed-effect SE for intrinsic areal fields (#196). Before this fix
# ICAR / CAR / BYM2 fell into the dense Laplace Hessian branch that ignores the
# spatial field entirely, so summary()/vcov()/confint() reported SEs computed at
# the wrong linear predictor and without marginalizing the field.
#
# The marginal H_beta must equal the Schur complement of the latent block out of
# the joint penalized-GLM Hessian at the fitted mode. We validate two things
# against an INDEPENDENT penalized-IRLS reference on the same objective:
#   (a) the fitted beta mode matches the reference  -> the reconstructed field
#       precision Q (ICAR: the augmented L + 11'/J; BYM2: the two-block
#       convolution) is exactly the one the kernel penalizes with;
#   (b) fit$H_beta matches the reference Schur       -> the marginalization is
#       the correct field-aware one, not the field-ignoring dense branch.
#
# (a) and (b) are both weak on the AUGMENTATION COEFFICIENT specifically: the
# intercept prior is deliberately weak (sigma_beta = 100), so the field constant
# is driven to ~0 by any pin of order 1 and the mode barely moves; and (b)
# compares the helper against a reference stating the same convention. The
# coefficient therefore gets its own gate below, against the kernel's own
# log-prior rather than against another R statement of it.

rook_adj_se <- function(nr, nc) {
  n <- nr * nc; W <- matrix(0, n, n)
  id <- function(r, c) (c - 1) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
    if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
  }
  W
}

# Reference marginal H_beta by penalized-IRLS on the same objective the ICAR /
# BYM2 kernel maximizes, then a direct Schur of the latent block.
areal_reference <- function(spatial_type, y, ntr, X, W, unit) {
  n_units <- nrow(W); N <- length(unit); p <- ncol(X)
  L <- diag(rowSums(W)) - W
  # ICAR structure plus the sum-to-zero augmentation 1_c 1_c'/J_c
  # (inst/include/tulpa/sum_to_zero.h); this graph is connected, so J_c = n_units.
  Q <- L + matrix(1 / n_units, n_units, n_units)
  Zind <- Matrix::sparseMatrix(i = seq_len(N), j = unit, x = 1,
                               dims = c(N, n_units))
  if (spatial_type == "bym2") {                        # sigma = 1, rho = 0.5, sf = 1
    D <- cbind(sqrt(0.5) * Zind, sqrt(0.5) * Zind)
    Qlat <- Matrix::bdiag(Matrix::Matrix(Q, sparse = TRUE),
                          Matrix::Diagonal(n_units, 1))
  } else {
    D <- Zind
    Qlat <- Matrix::Matrix(Q, sparse = TRUE)
  }
  Dm <- as.matrix(D); nlat <- ncol(Dm)
  M  <- cbind(X, Dm)
  # Mode-find with the kernel's built-in weak beta prior (sigma_beta = 100) so
  # the modes line up; the marginal itself omits that negligible ridge.
  Qjoint <- Matrix::bdiag(Matrix::Diagonal(p, 1 / 100^2), Qlat)
  th <- rep(0, p + nlat)
  for (it in seq_len(200)) {
    eta <- as.numeric(M %*% th); pr <- plogis(eta); Wv <- ntr * pr * (1 - pr)
    g <- as.numeric(crossprod(M, y - ntr * pr)) - as.numeric(Qjoint %*% th)
    H <- crossprod(M, Wv * M) + as.matrix(Qjoint)
    step <- solve(H, g); th <- th + step
    if (max(abs(step)) < 1e-10) break
  }
  eta <- as.numeric(M %*% th); pr <- plogis(eta); Wv <- ntr * pr * (1 - pr)
  XtWX <- crossprod(X, Wv * X)
  XtWD <- crossprod(X, Wv * Dm)
  DtWD <- crossprod(Dm, Wv * Dm) + as.matrix(Qlat)
  list(beta = th[seq_len(p)],
       H_beta = XtWX - XtWD %*% solve(DtWD, t(XtWD)))
}

for (ty in c("icar", "car", "bym2")) {
  test_that(paste0("areal marginal H_beta matches the joint Schur (", ty, ", #196)"), {
    skip_on_cran()
    set.seed(11)
    nr <- 4L; nc <- 4L; n_units <- nr * nc; reps <- 5L
    W <- rook_adj_se(nr, nc)
    unit <- rep(seq_len(n_units), each = reps); N <- length(unit)
    L <- diag(rowSums(W)) - W
    phi_true <- as.numeric(t(chol(solve(L + diag(n_units)))) %*% rnorm(n_units))
    phi_true <- phi_true - mean(phi_true)
    x <- rnorm(N); ntr <- rep(6L, N)
    X <- cbind(1, x)
    y <- rbinom(N, ntr, plogis(-0.3 + 0.8 * x + phi_true[unit]))

    fit <- tulpa_laplace(
      y = y, n_trials = ntr, X = X, family = "binomial",
      spatial = list(type = ty, adjacency = W, spatial_idx = unit,
                     scale_factor = 1.0),
      return_hessian = TRUE)

    expect_false(is.null(fit$H_beta))            # field-aware marginal produced
    expect_true(all(is.finite(fit$H_beta)))

    ref <- areal_reference(ty, y, ntr, X, W, unit)
    # (a) same objective -> same mode (kernel Q == reconstructed Q).
    expect_lt(max(abs(fit$mode[seq_len(ncol(X))] - ref$beta)), 1e-4)
    # (b) marginalization matches the reference Schur.
    expect_lt(max(abs(fit$H_beta - ref$H_beta)) / max(abs(ref$H_beta)), 1e-6)
  })
}


# The field precision the R marginal reconstructs must be the one the C++ kernel
# actually penalizes with, so the gate is the kernel's own exported log-prior
# rather than a second R statement of the convention. log_prior_icar is exactly
# quadratic in the field (its normalizer is constant in x), so
#   Q_ij = -(lp(e_i + e_j) - lp(e_i) - lp(e_j) + lp(0))
# recovers it with no truncation error.
kernel_icar_Q <- function(W) {
  n   <- nrow(W)
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  lp  <- function(x) tulpa:::cpp_test_log_prior_icar(
    x, as.integer(n), 1.0, as.integer(csr$row_ptr), as.integer(csr$col_idx),
    as.integer(csr$n_neighbors))
  e  <- function(k) { v <- numeric(n); if (k > 0L) v[k] <- 1; v }
  l0 <- lp(e(0L))
  l1 <- vapply(seq_len(n), function(i) lp(e(i)), numeric(1))
  Q  <- matrix(0, n, n)
  for (i in seq_len(n)) for (j in seq_len(n)) {
    Q[i, j] <- -(lp(e(i) + e(j)) - l1[i] - l1[j] + l0)
  }
  Q
}

chain_adj <- function(k) {
  W <- matrix(0, k, k)
  if (k > 1L) for (i in seq_len(k - 1L)) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  W
}

bd_adj <- function(...) as.matrix(Matrix::bdiag(
  lapply(list(...), function(a) Matrix::Matrix(a, sparse = TRUE))))

test_that(".icar_precision_Q reproduces the kernel's augmented field precision", {
  skip_on_cran()
  # Interleave the 5+3 components so neither is contiguous in the node ordering.
  perm8 <- c(1L, 6L, 2L, 7L, 3L, 8L, 4L, 5L)
  for (case in list(
    list(lab = "connected chain",       W = chain_adj(8L)),
    list(lab = "4x4 rook grid",         W = rook_adj_se(4L, 4L)),
    # Replicated field (`spatial(by =)`): equal-size disjoint copies.
    list(lab = "two equal chains",      W = bd_adj(chain_adj(5L), chain_adj(5L))),
    # Genuine disconnected map: UNEQUAL components (the equal-split bug pinned
    # {1..4},{5..8} here instead of {1..5},{6..8}) ...
    list(lab = "unequal 5+3 chains",    W = bd_adj(chain_adj(5L), chain_adj(3L))),
    # ... and the same, node order permuted so both components are
    # NON-CONTIGUOUS (the arbitrary node-index path).
    list(lab = "5+3 permuted",
         W = { A <- bd_adj(chain_adj(5L), chain_adj(3L)); A[perm8, perm8] })
  )) {
    Qr <- as.matrix(tulpa:::.icar_precision_Q(list(adjacency = case$W)))
    Qk <- kernel_icar_Q(case$W)
    expect_lt(max(abs(Qr - Qk)), 1e-10, label = case$lab)

    # The augmentation must be the true per-component 1/J_c one, not a unit 11'
    # (which understates every intrinsic-areal intercept SE).
    n <- nrow(case$W)
    Q_unit <- as.matrix(Matrix::Diagonal(n, Matrix::rowSums(case$W)) - case$W) +
      matrix(1, n, n)
    expect_gt(max(abs(Q_unit - Qk)), 0.5, label = case$lab)
  }
})
