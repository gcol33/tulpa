# Marginal fixed-effect SE for intrinsic areal fields (#196). Before this fix
# ICAR / CAR / BYM2 fell into the dense Laplace Hessian branch that ignores the
# spatial field entirely, so summary()/vcov()/confint() reported SEs computed at
# the wrong linear predictor and without marginalizing the field.
#
# The marginal H_beta must equal the Schur complement of the latent block out of
# the joint penalized-GLM Hessian at the fitted mode. We validate two things
# against an INDEPENDENT penalized-IRLS reference on the same objective:
#   (a) the fitted beta mode matches the reference  -> the reconstructed field
#       precision Q (ICAR: L + 11'; BYM2: the two-block convolution) is exactly
#       the one the kernel penalizes with;
#   (b) fit$H_beta matches the reference Schur       -> the marginalization is
#       the correct field-aware one, not the field-ignoring dense branch.

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
  Q <- L + matrix(1, n_units, n_units)                 # ICAR structure + 11'
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
