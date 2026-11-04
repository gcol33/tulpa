smc_nngp_fixture <- function(name) {
  cfg <- switch(name,
    F1 = c(J = 12, n = 72, s2 = 0.5, range = 0.3, noise = 0.25, seed = 20),
    F2 = c(J = 15, n = 45, s2 = 0.1, range = 0.3, noise = 1, seed = 21),
    F3 = c(J = 20, n = 100, s2 = 1, range = 0.5, noise = 0.1, seed = 22),
    F4 = c(J = 15, n = 60, s2 = 0.5, range = 1.5, noise = 0.25, seed = 23))
  set.seed(cfg[["seed"]])
  J <- as.integer(cfg[["J"]]); n <- as.integer(cfg[["n"]])
  xy <- data.frame(lon = runif(J), lat = runif(J))
  loc <- rep_len(1:J, n); x <- rnorm(n)
  d <- data.frame(lon = xy$lon[loc], lat = xy$lat[loc], x = x)
  sp <- tulpa:::.gp_sampler_spec(tulpa:::validate_gp(
    spatial_gp(~lon + lat, nn = J - 1L), d))
  D <- as.matrix(dist(sp$coords))
  A <- outer(sp$obs_to_loc + 1L, 1:J, "==") * 1
  w <- t(chol(cfg[["s2"]] * exp(-D / cfg[["range"]]) + diag(1e-8, J))) %*% rnorm(J)
  y <- 1 + 0.5 * x + as.numeric(A %*% w) + rnorm(n, 0, sqrt(cfg[["noise"]]))
  stopifnot(length(y) == n, nrow(sp$coords) == J, ncol(A) == J)
  list(y = y, X = cbind(1, x), A = A, D = D, sp = sp, noise = cfg[["noise"]])
}

smc_nngp_log_marginal <- function(f, log_sigma2, log_range) {
  # The nugget is independent of sigma2. Whiten its covariance together with
  # the observation noise and beta, then diagonalize the field covariance
  # once per range. All variance levels share that eigendecomposition.
  n <- length(f$y); J <- nrow(f$D)
  V <- f$noise * diag(n) + 4 * tcrossprod(f$X) + 1e-8 * tcrossprod(f$A)
  R <- chol(V); ViA <- backsolve(R, forwardsolve(t(R), f$A))
  Viy <- backsolve(R, forwardsolve(t(R), f$y))
  G <- crossprod(f$A, ViA); b <- crossprod(f$A, Viy)
  base <- -n * log(2 * pi) / 2 - sum(log(diag(R))) - sum(f$y * Viy) / 2
  U <- chol(G)
  by <- forwardsolve(t(U), b)
  vapply(log_range, function(r) {
    E <- eigen(U %*% exp(-f$D / exp(r)) %*% t(U), symmetric = TRUE)
    stopifnot(all(E$values > 0))
    h2 <- as.numeric(crossprod(E$vectors, by))^2
    a <- outer(exp(log_sigma2), E$values)
    base - rowSums(log1p(a)) / 2 + as.numeric((a / (1 + a)) %*% h2) / 2
  }, numeric(length(log_sigma2)))
}

smc_nngp_exact <- function(f, h = 0.1) {
  ls <- seq(-40, 8, by = h); lr <- seq(-12, 20, by = h)
  rate_s <- -log(f$sp$sigma2_prior_alpha) / f$sp$sigma2_prior_U
  rate_r <- -log(f$sp$phi_prior_alpha) * f$sp$phi_prior_U
  value <- smc_nngp_log_marginal(f, ls, lr) +
    outer(log(rate_s) + ls / 2 - log(2) - rate_s * exp(ls / 2),
          log(rate_r) - lr - rate_r * exp(-lr), "+")
  m <- max(value); w <- exp(value - m); w <- w / sum(w)
  list(logZ = m + log(sum(exp(value - m)) * h^2),
       mean_scale = sum(rowSums(w) * ls), mean_range = sum(colSums(w) * lr),
       p_low = sum(w[ls < -6, ]), p_range = sum(w[, lr > 2]),
       edge = max(w[c(1, nrow(w)), ], w[, c(1, ncol(w))]))
}
