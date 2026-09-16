# Correctness (not just PD/shape) for the continuous-spatial marginal
# fixed-effect SE (#225). The areal path (icar/car/bym2) is validated against an
# independent penalized-IRLS + Schur reference; the GP/NNGP path was only
# asserted finite/PD. Here we pin the NNGP builder against an EXACT dense-GP
# reference: with k = n-1 neighbours the Vecchia (NNGP) factorization conditions
# each ordered point on ALL predecessors, so it reproduces the full multivariate
# normal and its precision Q_nngp equals K^{-1} exactly. We check
#   (a) fit$mode[beta] matches the dense reference -> the C++ NNGP kernel the fit
#       maximizes is the exact GP;
#   (b) fit$H_beta matches the dense Schur         -> .nngp_precision_Q (the R
#       reconstruction the marginal SE is built from) equals the fitted precision.

# Pre-sorted coords so order(coords) == seq_len(n): nn_order is the identity and
# the obs<->location design Z is I, removing all permutation bookkeeping.
make_full_nngp_spec <- function(coords) {
  n <- nrow(coords); k <- n - 1L
  nn_idx  <- matrix(0L, n, k); nn_dist <- matrix(0, n, k)
  for (i in 2:n) {
    d  <- sqrt((coords[1:(i - 1), 1] - coords[i, 1])^2 +
               (coords[1:(i - 1), 2] - coords[i, 2])^2)
    nc <- min(length(d), k)
    sel <- order(d)[seq_len(nc)]
    nn_idx[i, seq_len(nc)]  <- sel
    nn_dist[i, seq_len(nc)] <- d[sel]
  }
  spec <- spatial_gp(coords = ~ x1 + x2, cov = "exponential", nn = k)
  spec$n_obs <- n; spec$n_spatial <- n; spec$n_unique <- n
  spec$obs_to_loc <- seq_len(n)
  spec$unique_coords <- coords; spec$coords_matrix <- coords
  spec$nn <- k
  spec$neighbor_info <- list(nn_idx = nn_idx, nn_dist = nn_dist,
                             nn_order = seq_len(n), nn_order_inv = seq_len(n))
  spec
}

# Exact dense-GP reference: Q = K^{-1}, K the exponential kernel with the same
# kGpJitter (1e-8) the kernel adds, then a penalized-IRLS mode + Schur of the
# latent block (beta ridge 1/100^2, matching the built-in weak N(0,100^2)).
gp_marginal_reference <- function(y, ntr, X, coords, sigma2, phi) {
  n <- nrow(coords); p <- ncol(X)
  D <- as.matrix(dist(coords))
  K <- sigma2 * exp(-D / phi); diag(K) <- sigma2 + 1e-8
  Q <- solve(K)
  Z <- diag(n); M <- cbind(X, Z)
  Qjoint <- rbind(
    cbind(diag(p) / 100^2, matrix(0, p, n)),
    cbind(matrix(0, n, p), Q))
  th <- rep(0, p + n)
  for (it in seq_len(200)) {
    eta <- as.numeric(M %*% th); pr <- plogis(eta); Wv <- ntr * pr * (1 - pr)
    g <- as.numeric(crossprod(M, y - ntr * pr)) - as.numeric(Qjoint %*% th)
    H <- crossprod(M, Wv * M) + Qjoint
    step <- solve(H, g); th <- th + step
    if (max(abs(step)) < 1e-11) break
  }
  eta <- as.numeric(M %*% th); pr <- plogis(eta); Wv <- ntr * pr * (1 - pr)
  XtWX <- crossprod(X, Wv * X)
  XtWD <- crossprod(X, Wv * Z)
  DtWD <- crossprod(Z, Wv * Z) + Q
  list(beta = th[seq_len(p)], H_beta = XtWX - XtWD %*% solve(DtWD, t(XtWD)))
}

test_that("NNGP marginal H_beta matches the exact dense-GP Schur (#225)", {
  skip_on_cran()
  set.seed(23)
  n <- 12L
  coords <- cbind(runif(n), runif(n))
  coords <- coords[order(coords[, 1], coords[, 2]), , drop = FALSE]  # pre-sorted
  sigma2 <- 0.8; phi <- 0.35

  X <- cbind(1, rnorm(n)); beta_true <- c(-0.3, 0.6)
  ntr <- rep(8L, n)
  y <- rbinom(n, ntr, plogis(as.numeric(X %*% beta_true)))

  spec <- make_full_nngp_spec(coords)
  spec$sigma2_gp <- sigma2; spec$phi_gp <- phi

  fit <- tulpa_laplace(
    y = y, n_trials = ntr, X = X, re_list = list(), family = "binomial",
    spatial = spec, max_iter = 100L, tol = 1e-9, n_threads = 1L,
    return_hessian = TRUE)

  expect_false(is.null(fit$H_beta))
  expect_equal(dim(fit$H_beta), c(ncol(X), ncol(X)))

  ref <- gp_marginal_reference(y, ntr, X, coords, sigma2, phi)
  # (a) same objective -> same mode (kernel Q == exact GP precision).
  expect_lt(max(abs(fit$mode[seq_len(ncol(X))] - ref$beta)), 1e-4)
  # (b) marginalization matches the dense Schur (jitter 1e-8 is the only gap).
  expect_lt(max(abs(fit$H_beta - ref$H_beta)) / max(abs(ref$H_beta)), 1e-4)
})

test_that(".nngp_cov_fn reads the compiled kernel at every CovType code (#798)", {
  # tulpa::CovType: 0 = exponential, 1 = matern32, 2 = gaussian, 4 = matern52.
  # .nngp_cov_fn() used to restate these formulas by hand with the PRE-#686
  # numbering (0/1/2 = exponential/matern32/matern52), so gp_cov_type()'s
  # code 4 for Matern nu = 2.5 was rejected outright and code 2 would have
  # silently evaluated Matern 5/2 where the engine means Gaussian. Pinning
  # against the closed forms keeps the R side from drifting from
  # inst/include/tulpa/cov_kernel.h again.
  d <- 1; sigma2 <- 1; phi <- 1
  x32 <- sqrt(3) * d / phi
  x52 <- sqrt(5) * d / phi
  closed <- c(
    `0` = sigma2 * exp(-d / phi),                                    # exponential
    `1` = sigma2 * (1 + x32) * exp(-x32),                            # matern32
    `2` = sigma2 * exp(-(d / phi)^2),                                # gaussian
    `4` = sigma2 * (1 + x52 + x52 * x52 / 3) * exp(-x52)             # matern52
  )
  for (code in c(0L, 1L, 2L, 4L)) {
    fn <- tulpa:::.nngp_cov_fn(code, sigma2, phi)
    expect_equal(fn(d), unname(closed[[as.character(code)]]), tolerance = 1e-12)
  }
  # matrix input keeps its shape (the NNGP scatter passes a distance matrix).
  D <- matrix(c(0, 1, 1, 0), 2, 2)
  fn52 <- tulpa:::.nngp_cov_fn(4L, sigma2, phi)
  out <- fn52(D)
  expect_equal(dim(out), dim(D))
  expect_equal(diag(out), c(sigma2, sigma2))
})
