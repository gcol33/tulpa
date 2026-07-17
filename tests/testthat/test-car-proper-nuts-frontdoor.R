# Front-door Tier-1 exact NUTS for a proper-CAR areal field (gcol33/tulpa#158c).
# tulpa(y ~ x + spatial(region), spatial = list(type = "car_proper", ...),
# mode = "exact") routes through the generic ModelData sampler, which allocates
# log_tau, logit_rho_car and the per-unit field (keyed on spatial_type ==
# CAR_PROPER) and samples them jointly. Q(rho) = D - rho W is full-rank, so
# rho is estimated from the data and the field mean is identified. The
# differentiable log|Q(rho)| uses the precomputed adjacency eigenvalues
# (sum_i log(1 - rho mu_i)), not a per-gradient Cholesky.

# rook_adj() / helpers live in helper-spatial.R.

sim_car_proper_pois <- function(nr = 6L, nc = 6L, reps = 5L,
                                 tau = 1.2, rho = 0.85, seed = 11L) {
  set.seed(seed)
  W <- rook_adj(nr, nc); S <- nr * nc
  Q <- tau * (diag(rowSums(W)) - rho * W)
  L <- chol(Q)
  phi <- as.numeric(backsolve(L, rnorm(S)))   # N(0, Q^{-1})
  phi <- phi - mean(phi)
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  x <- rnorm(N)
  y <- rpois(N, exp(-0.2 + 0.7 * x + phi[unit]))
  list(data = data.frame(y = y, x = x, region = factor(unit)),
       W = W, phi = phi, S = S)
}

test_that("car_proper exact NUTS allocates the rho + field columns (structural)", {
  skip_if_not_slow()
  s <- sim_car_proper_pois(nr = 4L, nc = 4L, reps = 4L)
  fit <- tulpa(y ~ x + spatial(region), data = s$data, family = "poisson",
               spatial = list(type = "car_proper", adjacency = s$W),
               mode = "exact",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  expect_true(all(c("log_tau_spatial", "logit_rho_car") %in% pn))
  expect_equal(sum(grepl("^phi_spatial\\[", pn)), s$S)
  expect_equal(ncol(fit$draws), 4L + s$S)   # 2 fixed + log_tau + logit_rho + field
})

test_that("car_proper exact NUTS matches the nested-Laplace CAR fit", {
  skip_if_not_slow()
  s <- sim_car_proper_pois()
  fit <- tulpa(y ~ x + spatial(region), data = s$data, family = "poisson",
               spatial = list(type = "car_proper", adjacency = s$W),
               mode = "exact",
               control = list(n_iter = 400L, n_warmup = 200L, seed = 3L))
  fcol <- grep("^phi_spatial\\[", colnames(fit$draws))
  f_nuts <- colMeans(fit$draws[, fcol, drop = FALSE])

  # Fixed effects recover; the field tracks the simulated truth.
  expect_lt(abs(unname(coef(fit)["x"]) - 0.7), 0.3)
  expect_gt(cor(f_nuts - mean(f_nuts), s$phi - mean(s$phi)), 0.4)

  # Same model as the nested-Laplace proper-CAR path: the two posterior-mean
  # fields agree closely (a modest per-seed truth correlation is expected for a
  # 36-unit areal field, but NUTS and Laplace must fit the SAME field).
  lap <- tulpa(y ~ x + spatial(region), data = s$data, family = "poisson",
               spatial = list(type = "car_proper", adjacency = s$W),
               mode = "nested_laplace")
  mo <- lap$modes
  f_lap <- if (is.matrix(mo)) colMeans(mo)[(ncol(mo) - s$S + 1):ncol(mo)]
           else tail(as.numeric(mo), s$S)
  expect_gt(cor(f_nuts - mean(f_nuts), f_lap - mean(f_lap)), 0.85)
})
