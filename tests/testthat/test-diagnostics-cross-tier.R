# spatial_range() / temporal_corr() must report the SAME interpretable quantity
# and row label on a nested-Laplace (Tier 2) fit as on a sampler (Tier 1) fit
# (#199). The nested path used to return the raw grid axes (tau, phi_gp, sigma2)
# with raw names; it now maps each axis to the sampler's interpretable quantity
# (sigma = 1/sqrt(tau) or sqrt(sigma2); range = 3 * lengthscale; rho identity),
# marginalizing the DERIVED quantity per grid cell.

rook_ct <- function(nr, nc) {
  n <- nr * nc; W <- matrix(0, n, n); id <- function(r, c) (c - 1) * nr + r
  for (r in seq_len(nr)) for (c in seq_len(nc)) {
    if (r < nr) { W[id(r, c), id(r + 1, c)] <- 1; W[id(r + 1, c), id(r, c)] <- 1 }
    if (c < nc) { W[id(r, c), id(r, c + 1)] <- 1; W[id(r, c + 1), id(r, c)] <- 1 }
  }
  W
}

nested_areal_fit <- function(type) {
  set.seed(9)
  nr <- nc <- 5L; S <- nr * nc; reps <- 4L; W <- rook_ct(nr, nc)
  unit <- rep(seq_len(S), each = reps); N <- length(unit)
  x <- rnorm(N); y <- rbinom(N, 3, plogis(-0.3 + 0.6 * x))
  idx <- tulpa:::.resolve_spatial_idx(factor(unit), S, W, "region")
  csr <- tulpa:::adjacency_to_csr_tulpa(W)
  prior <- list(type = type, spatial_idx = idx, n_spatial_units = S,
                adj_row_ptr = csr$row_ptr, adj_col_idx = csr$col_idx,
                n_neighbors = csr$n_neighbors)
  fit <- tulpa_nested_laplace(y = y, n_trials = rep(3L, N), X = cbind(1, x),
                              prior = prior, family = "binomial")
  fit$prior <- prior
  fit
}

test_that("nested spatial_range reports interpretable sigma / rho, marginalized (#199)", {
  skip_on_cran()
  # icar: single tau axis -> sigma = 1/sqrt(tau).
  f1 <- nested_areal_fit("icar")
  s1 <- spatial_range(f1)
  expect_identical(rownames(s1), "sigma")               # not the raw "tau"
  w1 <- f1$weights / sum(f1$weights); tau1 <- as.numeric(f1$theta_grid)
  expect_equal(s1["sigma", "mean"], sum(w1 * (1 / sqrt(tau1))))

  # car_proper: (tau, rho) -> (sigma, rho); rho passes through, sigma = 1/sqrt(tau).
  f2 <- nested_areal_fit("car_proper")
  s2 <- spatial_range(f2)
  expect_setequal(rownames(s2), c("sigma", "rho"))
  w2 <- f2$weights / sum(f2$weights)
  expect_equal(s2["sigma", "mean"], sum(w2 * (1 / sqrt(f2$theta_grid[, 1]))))
  expect_equal(s2["rho", "mean"],   sum(w2 * f2$theta_grid[, 2]))
})

test_that("the nested transform maps mirror the sampler's interpretable formulas (#199)", {
  sm <- tulpa:::.SPATIAL_HYPER_TRANSFORM
  expect_equal(sm$tau$fn(4), 0.5)         # 1/sqrt(4)  -> sigma
  expect_equal(sm$sigma2$fn(9), 3)        # sqrt(9)    -> sigma
  expect_equal(sm$phi_gp$fn(2), 6)        # 3 * 2      -> range
  expect_equal(sm$lengthscale$fn(2), 6)   # 3 * 2      -> range
  expect_identical(sm$tau$name, "sigma")
  expect_identical(sm$phi_gp$name, "range")

  tmn <- tulpa:::.TEMPORAL_HYPER_TRANSFORM
  expect_identical(tmn$tau$name, "precision")   # temporal reports tau as precision
})

test_that("nested temporal_corr maps its grid axes to the sampler labels (#199)", {
  ft <- structure(
    list(theta_grid = matrix(c(0.5, 1, 2), ncol = 1,
                             dimnames = list(NULL, "tau")),
         weights = c(0.2, 0.5, 0.3), theta_names = "tau",
         prior = list(type = "rw1")),
    class = c("tulpa_nested_laplace", "tulpa_fit"))
  tc <- temporal_corr(ft)
  expect_identical(rownames(tc), "precision")
  expect_equal(tc["precision", "mean"], sum(c(0.2, 0.5, 0.3) * c(0.5, 1, 2)))
})

# gcol33/tulpa#800: spatial_range() / temporal_corr() matched only a fixed
# name table, so HSGP, SVC, SPDE, temporal_multiscale, temporal_tvc and a
# latent(temporal_ar2()) block all errored "No .../ hyperparameters found"
# despite the fit carrying exactly those hyperparameters.
test_that("spatial_range covers hsgp, svc and spde", {
  skip_on_cran()
  set.seed(2)
  L <- cbind(lon = runif(80, 0, 10), lat = runif(80, 0, 10))
  g <- data.frame(L, x = rnorm(80)); g$y <- rpois(80, exp(0.3 + 0.5 * g$x))
  ctl <- list(n_iter = 200, warmup = 100, n_chains = 2, seed = 1)

  fit_hsgp <- tulpa(y ~ x, data = g, family = "poisson", mode = "hmc",
                    spatial = spatial_gp(~ lon + lat, approx = "hsgp"),
                    control = ctl)
  sr_hsgp <- spatial_range(fit_hsgp)
  expect_setequal(rownames(sr_hsgp), c("sigma_hsgp", "range"))
  expect_true(all(is.finite(as.matrix(sr_hsgp))))
  expect_true(sr_hsgp["range", "mean"] > 0)

  fit_spde <- tulpa(y ~ x, data = g, family = "poisson",
                    spatial = spatial_spde(~ lon + lat, data = g))
  sr_spde <- spatial_range(fit_spde)
  expect_setequal(rownames(sr_spde), c("range", "sigma"))
  expect_true(all(is.finite(as.matrix(sr_spde))))

  fit_svc <- tulpa(y ~ x, data = g, family = "poisson", mode = "hmc",
                   spatial = spatial_svc(~ lon + lat, terms = "x"), control = ctl)
  sr_svc <- spatial_range(fit_svc)
  expect_setequal(rownames(sr_svc), c("sigma_svc", "range_svc"))
  expect_true(all(is.finite(as.matrix(sr_svc))))
  # Default approx = "nngp": phi_svc is a direct exponential-kernel range, the
  # same convention log_phi_gp uses, not the HSGP squared-exponential one.
  expect_identical(fit_svc$spatial$approx, "nngp")
})

test_that("temporal_corr covers multiscale, tvc and a latent AR(p) block", {
  skip_on_cran()
  set.seed(3)
  Tm <- data.frame(tidx = rep(1:40, each = 4), x = rnorm(160))
  Tm$y <- rpois(160, exp(0.3 + 0.5 * Tm$x))
  ctl <- list(n_iter = 200, warmup = 100, n_chains = 2, seed = 1)

  fit_ms <- tulpa(y ~ x, data = Tm, family = "poisson", control = ctl,
                  temporal = temporal_multiscale("tidx", trend = "rw2",
                                                 seasonal = 12, short_term = "ar1"))
  tc_ms <- temporal_corr(fit_ms)
  expect_setequal(rownames(tc_ms),
                  c("sigma_trend", "sigma_seasonal", "sigma_short", "rho_short"))
  expect_true(all(is.finite(as.matrix(tc_ms))))
  expect_true(tc_ms["rho_short", "mean"] >= -1 && tc_ms["rho_short", "mean"] <= 1)

  fit_tvc <- tulpa(y ~ x, data = Tm, family = "poisson", mode = "hmc", control = ctl,
                   temporal = temporal_tvc("tidx", terms = "x", structure = "rw1"))
  tc_tvc <- temporal_corr(fit_tvc)
  expect_identical(rownames(tc_tvc), "tau_tvc")
  expect_true(is.finite(tc_tvc["tau_tvc", "mean"]))

  fit_ar2 <- tulpa(y ~ x + latent(temporal_ar2(Tm$tidx)), data = Tm, family = "poisson")
  tc_ar2 <- temporal_corr(fit_ar2)
  expect_setequal(rownames(tc_ar2), c("precision", "psi1", "psi2"))
  expect_true(all(is.finite(as.matrix(tc_ar2))))
  expect_true(all(abs(tc_ar2[c("psi1", "psi2"), "mean"]) <= 1))
})
