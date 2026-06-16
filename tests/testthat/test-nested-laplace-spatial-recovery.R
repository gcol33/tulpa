# Parameter-recovery + CI-coverage for the nested-Laplace SPATIAL hyperparameters
# on the cpp_nested_laplace_* kernels the FRONT DOOR dispatches (gcol33/tulpa#97).
#
# The existing single-block spatial tests (test-nested-laplace-bym2 / -car-proper
# / -gp / -nested-laplace) assert only that the integrator runs and produces a
# finite, varying log-marginal -- plumbing, not estimator correctness. Recovery
# of the spatial variance / range / mixing existed only on OTHER paths (the
# inline tulpa_spatial_field_fit and the PG-Gibbs sampler). These tests close
# that gap on the deterministic Tier-2 hot path: they fit through
# tulpa_nested_laplace() (which integrates the kernel marginals and exposes
# theta_mean / theta_ci_lo / theta_ci_hi) and check the marginalized
# hyperparameter posterior against a known truth.
#
# What is achievable, by kernel (a single field realization weakly identifies a
# global VARIANCE, so the point estimate of a variance is attenuated at small n
# even when the marginalized CI stays calibrated -- the "marginalize derived
# quantities" discipline):
#   * car_proper -- a proper GMRF with enough units/replicates: clean (tau, rho)
#     point recovery + CI coverage + field recovery (the gold case).
#   * icar / bym2 -- the field-amplitude hyperparameter tracks the simulated
#     amplitude directionally (icar's axis is the precision tau, so it moves
#     INVERSELY with amplitude; bym2's `sigma` moves WITH it).
#   * nngp / hsgp -- one observation per location identifies the variance only
#     weakly, so the check is that the marginalized (sigma2, range) CI COVERS the
#     truth (calibration), not a tight point estimate.

# Heavier multi-seed sweeps run only off-CRAN; a couple of seeds always run.
.nlsr_n_seed <- function() if (isTRUE(as.logical(Sys.getenv("TULPA_SLOW_TESTS", "false")))) 16L else 8L

# ---- self-contained simulators / graph helpers ----------------------------
.grid_adj <- function(nr, nc) {
  n <- nr * nc; nb <- vector("list", n)
  for (i in seq_len(n)) {
    r <- (i - 1) %/% nc + 1; c <- (i - 1) %% nc + 1; e <- integer(0)
    if (r > 1) e <- c(e, i - nc); if (r < nr) e <- c(e, i + nc)
    if (c > 1) e <- c(e, i - 1);  if (c < nc) e <- c(e, i + 1)
    nb[[i]] <- e
  }
  nn <- vapply(nb, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nb)) - 1L,
       n_neighbors = as.integer(nn), n = n)
}
.chain_adj <- function(S) {
  nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
  nn <- vapply(nb, length, integer(1))
  list(adj_row_ptr = as.integer(c(0L, cumsum(nn))),
       adj_col_idx = as.integer(unlist(nb)) - 1L, n_neighbors = as.integer(nn))
}
.csr_to_W <- function(adj, S) {
  W <- matrix(0, S, S)
  for (s in seq_len(S)) {
    a <- adj$adj_row_ptr[s] + 1L; b <- adj$adj_row_ptr[s + 1L]
    if (b >= a) W[s, adj$adj_col_idx[a:b] + 1L] <- 1
  }
  (W + t(W) > 0) * 1
}
.sim_car_field <- function(W, tau, rho, seed) {        # u ~ N(0, (tau(D-rho W))^{-1})
  set.seed(seed)
  Q <- tau * (diag(rowSums(W)) - rho * W)
  u <- backsolve(chol(Q), rnorm(nrow(W))); u - mean(u)
}
.structured_field <- function(S, seed) {              # unit-SD RW1 / chain-ICAR field
  set.seed(seed); f <- cumsum(rnorm(S)); f <- f - mean(f); f / stats::sd(f)
}
.gp_dataset_gauss <- function(n, sigma2, phi_gp, noise_sd, seed) {  # field nearly observed
  set.seed(seed)
  coords <- cbind(runif(n), runif(n))
  K <- sigma2 * exp(-as.matrix(dist(coords)) / phi_gp) + diag(1e-6, n)
  w <- as.numeric(crossprod(chol(K), rnorm(n))); w <- w - mean(w)
  list(y = w + rnorm(n, 0, noise_sd), coords = coords, n = n)
}
.nngp_neighbors <- function(coords, k) {
  n <- nrow(coords); o <- order(coords[, 1], coords[, 2]); co <- coords[o, ]
  ni <- matrix(0L, n, k); nd <- matrix(0, n, k)
  for (i in 2:n) {
    di <- sqrt((co[1:(i - 1), 1] - co[i, 1])^2 + (co[1:(i - 1), 2] - co[i, 2])^2)
    nc <- min(length(di), k); od <- order(di)[1:nc]
    ni[i, seq_len(nc)] <- od; nd[i, seq_len(nc)] <- di[od]
  }
  list(coords_ord = co, nn_idx = ni, nn_dist = nd,
       nn_order_0 = as.integer(o - 1L), n_neighbors = k)
}
.hsgp_basis_2d <- function(coords, m = 5L, cc = 1.5) {
  x <- coords[, 1]; y <- coords[, 2]
  L1 <- max(cc * (max(x) - min(x)) / 2, 0.1); L2 <- max(cc * (max(y) - min(y)) / 2, 0.1)
  xs <- x - (max(x) + min(x)) / 2; ys <- y - (max(y) + min(y)) / 2
  M <- m * m; N <- nrow(coords); pb <- matrix(0, N, M); le <- numeric(M)
  for (j1 in seq_len(m)) for (j2 in seq_len(m)) {
    idx <- (j1 - 1) * m + j2
    le[idx] <- (pi * j1 / (2 * L1))^2 + (pi * j2 / (2 * L2))^2
    pb[, idx] <- (1 / sqrt(L1)) * sin(pi * j1 * (xs + L1) / (2 * L1)) *
                 (1 / sqrt(L2)) * sin(pi * j2 * (ys + L2) / (2 * L2))
  }
  list(phi_basis = pb, lambda_eig = le)
}

# --------------------------------------------------------------------------- #
# CAR_proper: clean (tau, rho) recovery + CI coverage + field recovery         #
# --------------------------------------------------------------------------- #

test_that("nested-Laplace CAR_proper recovers (tau, rho) with CI coverage", {
  skip_if_not_slow()
  S <- 49L; reps <- 16L
  adj <- .grid_adj(7, 7); W <- .csr_to_W(adj, S)
  tau_true <- 2.0; rho_true <- 0.85
  tau_grid <- exp(seq(log(0.5), log(8), length.out = 7))
  rho_grid <- c(0.3, 0.5, 0.7, 0.85, 0.95)

  n_seed <- .nlsr_n_seed()
  tau_hat <- rho_hat <- field_cor <- numeric(n_seed)
  tau_cov <- rho_cov <- logical(n_seed)
  for (s in seq_len(n_seed)) {
    u <- .sim_car_field(W, tau_true, rho_true, seed = 200L + s)
    sidx <- rep(seq_len(S), each = reps); N <- length(sidx)
    set.seed(800L + s); y <- rbinom(N, 1L, plogis(0.2 + u[sidx]))
    prior <- list(type = "car_proper", spatial_idx = sidx, n_spatial_units = S,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, rho_bounds = c(0, 1),
                  tau_grid = tau_grid, rho_car_grid = rho_grid)
    fit <- suppressWarnings(tulpa_nested_laplace(
      as.integer(y), rep(1L, N), matrix(1, N, 1), prior = prior,
      family = "binomial", control = list(diagnose_k = FALSE)))
    tau_hat[s] <- fit$theta_mean[["tau"]]; rho_hat[s] <- fit$theta_mean[["rho"]]
    field_cor[s] <- stats::cor(fit$modes[1, 2:(1 + S)], u)
    tau_cov[s] <- tau_true >= fit$theta_ci_lo[["tau"]] && tau_true <= fit$theta_ci_hi[["tau"]]
    rho_cov[s] <- rho_true >= fit$theta_ci_lo[["rho"]] && rho_true <= fit$theta_ci_hi[["rho"]]
  }
  # rho (the spatial autocorrelation) recovers as a point estimate; tau (a global
  # variance from ONE field realization) is attenuated/noisy as a point estimate
  # but its MARGINALIZED CI stays calibrated -- the "marginalize derived
  # quantities" reality, so tau is gated on coverage, not on the point.
  expect_lt(abs(median(rho_hat) / rho_true - 1), 0.35)
  expect_gte(mean(tau_cov), 0.80)
  expect_gte(mean(rho_cov), 0.70)
  expect_gt(median(field_cor), 0.60)                      # the latent field is recovered
})

# --------------------------------------------------------------------------- #
# ICAR / BYM2: the field-amplitude hyperparameter tracks the truth             #
# --------------------------------------------------------------------------- #

test_that("nested-Laplace ICAR precision moves inversely with the field amplitude", {
  skip_if_not_slow()
  S <- 40L; reps <- 12L; adj <- .chain_adj(S)
  sidx <- rep(seq_len(S), each = reps); N <- length(sidx)
  icar_tau <- function(amp, seed) {                       # icar axis is the precision tau
    phi <- .structured_field(S, seed); set.seed(seed + 1L)
    y <- rbinom(N, 1L, plogis(-0.1 + amp * phi[sidx]))
    prior <- list(type = "icar", spatial_idx = sidx, n_spatial_units = S,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors,
                  sigma_grid = exp(seq(log(0.2), log(2.5), length.out = 7)))
    fit <- suppressWarnings(tulpa_nested_laplace(
      as.integer(y), rep(1L, N), matrix(1, N, 1), prior = prior,
      family = "binomial", control = list(diagnose_k = FALSE)))
    fit$theta_mean[[1]]
  }
  lo <- vapply(1:5, function(s) icar_tau(0.4, 300L + s), numeric(1))  # small amplitude
  hi <- vapply(1:5, function(s) icar_tau(1.7, 400L + s), numeric(1))  # large amplitude
  expect_gt(median(lo), median(hi))                       # less amplitude => MORE precision
})

test_that("nested-Laplace BYM2 sigma tracks the field amplitude", {
  skip_if_not_slow()
  S <- 40L; reps <- 12L; adj <- .chain_adj(S)
  sidx <- rep(seq_len(S), each = reps); N <- length(sidx)
  bym2_sigma <- function(amp, seed) {
    phi <- .structured_field(S, seed); set.seed(seed + 1L)
    y <- rbinom(N, 1L, plogis(-0.2 + amp * phi[sidx]))
    prior <- list(type = "bym2", spatial_idx = sidx, n_spatial_units = S,
                  adj_row_ptr = adj$adj_row_ptr, adj_col_idx = adj$adj_col_idx,
                  n_neighbors = adj$n_neighbors, scale_factor = 1.0,
                  sigma_grid = exp(seq(log(0.3), log(2.5), length.out = 6)),
                  rho_grid = c(0.3, 0.5, 0.7, 0.9))
    fit <- suppressWarnings(tulpa_nested_laplace(
      as.integer(y), rep(1L, N), matrix(1, N, 1), prior = prior,
      family = "binomial", control = list(diagnose_k = FALSE)))
    fit$theta_mean[["sigma"]]
  }
  lo <- vapply(1:5, function(s) bym2_sigma(0.4, 200L + s), numeric(1))
  hi <- vapply(1:5, function(s) bym2_sigma(1.8, 300L + s), numeric(1))
  expect_gt(median(hi), median(lo))
})

# --------------------------------------------------------------------------- #
# NNGP / HSGP: the marginalized (sigma2, range) CI covers the truth            #
# --------------------------------------------------------------------------- #

test_that("nested-Laplace NNGP marginalized CI covers (sigma2, phi_gp)", {
  skip_if_not_slow()
  sigma2_true <- 0.8; phi_true <- 0.35
  s2_grid <- exp(seq(log(0.2), log(2.0), length.out = 6))
  pg_grid <- exp(seq(log(0.12), log(0.9), length.out = 6))
  n_seed <- .nlsr_n_seed()
  s2_cov <- pg_cov <- logical(n_seed)
  for (s in seq_len(n_seed)) {
    d  <- .gp_dataset_gauss(120L, sigma2_true, phi_true, noise_sd = 0.2, seed = 700L + s)
    nb <- .nngp_neighbors(d$coords, 10L)
    prior <- list(type = "nngp", coords = nb$coords_ord, nn_idx = nb$nn_idx,
                  nn_dist = nb$nn_dist, nn_order = nb$nn_order_0 + 1L,
                  n_spatial = d$n, nn = nb$n_neighbors, cov_type = 0L,
                  sigma2_grid = s2_grid, phi_gp_grid = pg_grid)
    fit <- suppressWarnings(tulpa_nested_laplace(
      d$y, rep(1L, d$n), matrix(1, d$n, 1), prior = prior,
      family = "gaussian", phi = 0.2, control = list(diagnose_k = FALSE)))
    s2_cov[s] <- sigma2_true >= fit$theta_ci_lo[["sigma2"]] &&
                 sigma2_true <= fit$theta_ci_hi[["sigma2"]]
    pg_cov[s] <- phi_true >= fit$theta_ci_lo[["phi_gp"]] &&
                 phi_true <= fit$theta_ci_hi[["phi_gp"]]
  }
  expect_gte(mean(s2_cov), 0.60)
  expect_gte(mean(pg_cov), 0.60)
})

test_that("nested-Laplace HSGP marginalized sigma2 CI covers the truth", {
  skip_if_not_slow()
  N <- 90L; sigma2_true <- 0.8; ell <- 0.3
  s2_vals <- exp(seq(log(0.1), log(2.5), length.out = 6))
  ell_vals <- c(0.15, 0.30, 0.50, 0.15, 0.30, 0.50)
  gr <- expand.grid(s2 = s2_vals, ell = c(0.15, 0.30, 0.50))   # paired (equal-length)
  n_seed <- min(.nlsr_n_seed(), 8L)
  s2_cov <- logical(n_seed)
  for (s in seq_len(n_seed)) {
    set.seed(500L + s)
    coords <- cbind(runif(N), runif(N))
    bs <- .hsgp_basis_2d(coords, m = 5L, cc = 1.5)
    sd_w <- sqrt(sigma2_true * exp(-0.5 * ell^2 * bs$lambda_eig))
    f  <- as.numeric(bs$phi_basis %*% (rnorm(ncol(bs$phi_basis)) * sd_w))
    y  <- rbinom(N, 1L, plogis(-0.1 + f))
    prior <- list(type = "hsgp", phi_basis = bs$phi_basis, lambda_eig = bs$lambda_eig,
                  sigma2_grid = gr$s2, lengthscale_grid = gr$ell)
    fit <- suppressWarnings(tulpa_nested_laplace(
      as.integer(y), rep(1L, N), matrix(1, N, 1), prior = prior,
      family = "binomial", control = list(diagnose_k = FALSE)))
    s2_cov[s] <- sigma2_true >= fit$theta_ci_lo[["sigma2"]] &&
                 sigma2_true <= fit$theta_ci_hi[["sigma2"]]
  }
  expect_gte(mean(s2_cov), 0.50)
})
