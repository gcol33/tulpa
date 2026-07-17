# Front-door Tier-1 exact NUTS for a continuous HSGP (Hilbert-space GP) spatial
# field (gcol33/tulpa#158c). tulpa(spatial = spatial_gp(..., approx = "hsgp"),
# mode = "exact") routes through the generic ModelData sampler, which allocates
# the two HSGP hyperparameters and the m^2 basis coefficients (keyed on
# spatial_type == HSGP) and samples them jointly. The Laplacian basis is built
# in C++ by setup_hsgp_2d, the single source of truth every HSGP path uses.

make_hsgp_pois <- function(n = 90L, seed = 303L) {
  set.seed(seed)
  truef <- function(lon, lat) 1.0 * sin(3 * lon) + 0.8 * cos(2.5 * lat)
  lon <- runif(n); lat <- runif(n)
  d <- data.frame(lon = lon, lat = lat, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + truef(lon, lat)))
  list(d = d, truef = truef, lon = lon, lat = lat)
}

test_that("hsgp exact NUTS allocates the basis columns (structural)", {
  skip_if_not_slow()
  sim <- make_hsgp_pois(n = 40L)
  m <- 5L
  fit <- tulpa(y ~ x, data = sim$d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = m, c = 1.5),
               mode = "exact",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  expect_true(all(c("log_sigma2_hsgp", "log_lengthscale_hsgp") %in% pn))
  expect_equal(sum(grepl("^hsgp_beta\\[", pn)), m * m)
  expect_equal(ncol(fit$draws), 4L + m * m)
})

test_that("hsgp exact NUTS recovers the latent field", {
  skip_if_not_slow()
  sim <- make_hsgp_pois(n = 90L)
  m <- 6L; cc <- 1.5
  sp0 <- spatial_gp(~ lon + lat, approx = "hsgp", m = m, c = cc)
  fit <- tulpa(y ~ x, data = sim$d, family = "poisson", spatial = sp0,
               mode = "exact",
               control = list(n_iter = 350L, n_warmup = 175L, seed = 7L))
  pn <- colnames(fit$draws)
  bcol <- grep("^hsgp_beta\\[", pn)

  # Reconstruct the posterior-mean field f_i = sum_j phi[i,j] sqrt(S_j) beta_j
  # (the same spectral scaling the C++ prior applies), then check recovery.
  spv <- validate_hsgp(sp0, data = sim$d)
  basis <- cpp_hsgp_basis_2d(as.matrix(spv$coords_matrix), m, cc)
  Phi <- basis$phi_basis; lam <- basis$lambda_eig
  fld <- t(apply(fit$draws, 1, function(row) {
    s2 <- exp(row["log_sigma2_hsgp"]); ell <- exp(row["log_lengthscale_hsgp"])
    S  <- s2 * 2 * pi * ell^2 * exp(-0.5 * ell^2 * lam)
    as.numeric(Phi %*% (sqrt(S) * row[bcol]))
  }))
  fm <- colMeans(fld)
  tru <- sim$truef(sim$lon, sim$lat)

  expect_gt(cor(fm - mean(fm), tru - mean(tru)), 0.8)
  expect_lt(abs(unname(coef(fit)["x"]) - 0.5), 0.25)
})
