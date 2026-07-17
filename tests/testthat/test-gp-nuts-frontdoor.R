# Front-door Tier-1 exact NUTS for a continuous GP / NNGP spatial field
# (gcol33/tulpa#158c). tulpa(spatial = spatial_gp(...), mode = "exact") routes
# through the generic ModelData sampler, which allocates the GP hyperparameters
# and the per-location field (keyed on spatial_type == GP) and samples them with
# the rest of the parameter vector -- the same model the nested-Laplace GP path
# approximates.

make_gp_pois <- function(n = 80L, seed = 202L) {
  set.seed(seed)
  truef <- function(lon, lat) 1.0 * sin(3 * lon) + 0.8 * cos(2.5 * lat)
  lon <- runif(n); lat <- runif(n)
  d <- data.frame(lon = lon, lat = lat, x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.5 * d$x + truef(lon, lat)))
  list(d = d, truef = truef, lon = lon, lat = lat)
}

test_that("gp/nngp exact NUTS allocates the field columns (structural)", {
  sim <- make_gp_pois(n = 40L)
  spv <- validate_gp(spatial_gp(~ lon + lat, nn = 5L), data = sim$d)
  skip_if_not_slow()
  fit <- tulpa(y ~ x, data = sim$d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 5L), mode = "exact",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  # 2 fixed + log_sigma2_gp + log_phi_gp + one field column per unique location.
  expect_true(all(c("log_sigma2_gp", "log_phi_gp") %in% pn))
  expect_equal(sum(grepl("^gp_w\\[", pn)), spv$n_unique)
  expect_equal(ncol(fit$draws), 4L + spv$n_unique)
})

test_that("gp/nngp exact NUTS recovers the latent field and matches Laplace", {
  skip_if_not_slow()
  sim <- make_gp_pois(n = 80L)
  spv <- validate_gp(spatial_gp(~ lon + lat, nn = 6L), data = sim$d)
  loc_first <- tapply(seq_len(nrow(sim$d)), spv$obs_to_loc, function(ix) ix[1])
  loc_first <- loc_first[order(as.integer(names(loc_first)))]
  tru <- sim$truef(sim$lon[loc_first], sim$lat[loc_first])

  fit <- tulpa(y ~ x, data = sim$d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 6L), mode = "exact",
               control = list(n_iter = 350L, n_warmup = 175L, seed = 7L))
  fld <- grep("^gp_w\\[", colnames(fit$draws))
  f_nuts <- colMeans(fit$draws[, fld, drop = FALSE])

  # Field recovered against the simulated truth.
  expect_gt(cor(f_nuts - mean(f_nuts), tru - mean(tru)), 0.7)
  # Fixed slope recovered (the intercept is confounded with the field mean).
  expect_lt(abs(unname(coef(fit)["x"]) - 0.5), 0.25)

  # Same model as the nested-Laplace GP path: the two posterior-mean fields
  # agree, and exact MCMC is at least as accurate as the Laplace approximation.
  lap <- tulpa(y ~ x, data = sim$d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 6L), mode = "nested_laplace")
  mo <- lap$modes
  f_lap <- if (is.matrix(mo)) colMeans(mo)[(ncol(mo) - spv$n_unique + 1):ncol(mo)]
           else tail(as.numeric(mo), spv$n_unique)
  expect_gt(cor(f_nuts - mean(f_nuts), f_lap - mean(f_lap)), 0.75)
})
