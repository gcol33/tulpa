# The NUTS store overwrites a non-centered GP / SVC / multiscale-GP block's
# stored slice with the reconstructed field w (src/hmc_nuts_chain_iter_store.h),
# keeping q itself as z for sampling. cpp_tulpa_glmm_eta_draws() re-evaluates a
# stored draw through initialize_generic_state() at the spec's own
# gp_parameterization / svc_parameterization, which for a non-centered spec
# applies the forward transform a SECOND time to a slice that is no longer z
# (gcol33/tulpa#822). .tulpa_eta_draws_sampler() -- posterior_predict() /
# simulate() / WAIC / LOO's eta source for a "hmc" fit -- now reads the stored
# draws back through .stored_draw_field_specs(), which reports the spec as
# centered whenever the fit's own backend is "hmc": the forward transform is
# the identity on an already-centered field, matching what the store wrote. No
# other ModelData kernel (ess / sghmc / sgld / mclmc / smc / vi) applies this
# store-side overwrite, so those fits keep their own spec unchanged.

nc_field_fixture <- function(n = 60L, seed = 1L, box = 5) {
  set.seed(seed)
  lon <- runif(n, 0, box); lat <- runif(n, 0, box)
  D <- as.matrix(dist(cbind(lon, lat)))
  K <- 0.64 * exp(-D / 1.2) + diag(1e-8, n)
  w <- as.numeric(t(chol(K)) %*% rnorm(n))
  x <- rnorm(n)
  y <- rpois(n, exp(0.3 + 0.5 * x + w))
  data.frame(lon = lon, lat = lat, x = x, y = y)
}

nc_ctl <- list(n_iter = 100L, warmup = 60L, n_chains = 1L, seed = 1L)

test_that("eta at a noncentered SVC NUTS fit's stored draws matches stored svc_w", {
  skip_if_not_slow()
  d <- nc_field_fixture(seed = 3L)
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
              control = nc_ctl,
              spatial = spatial_svc(~ lon + lat, nn = 8L,
                                   parameterization = "noncentered"))
  expect_equal(fit$backend, "hmc")

  mi <- fit$model_inputs
  eta <- tulpa:::cpp_tulpa_glmm_eta_draws(
    draws = fit$draws, y = mi$y, n_trials = mi$n_trials, X = mi$X,
    family = mi$family, phi = mi$phi, sigma_beta = mi$sigma_beta,
    offset_nullable = mi$offset, re_spec = mi$re_spec,
    spatial_spec = mi$spatial_spec, temporal_spec = mi$temporal_spec,
    sigma_re_scale = mi$sigma_re_scale, phi2 = mi$phi2,
    # centered: the store already reconstructed w, so this is the read
    # .tulpa_eta_draws_sampler() now performs for a "hmc" fit.
    svc_spec = { s <- mi$svc_spec; s$svc_parameterization <- 0L; s },
    tvc_spec = mi$tvc_spec, zi_spec = mi$zi_spec)

  fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
    t(mi$X[, c("(Intercept)", "x"), drop = FALSE])
  field_from_eta <- eta - fixed

  w_cols <- grep("^svc_w", colnames(fit$draws))
  expect_length(w_cols, nrow(d))
  stored_w <- fit$draws[, w_cols, drop = FALSE]

  expect_equal(unname(field_from_eta), unname(stored_w), tolerance = 1e-8)

  # The front door itself: posterior_predict()'s in-sample eta source agrees
  # with the same stored field, not the double-transformed one.
  ppred_eta <- tulpa:::.tulpa_eta_draws_sampler(fit)
  expect_equal(unname(ppred_eta - fixed), unname(stored_w), tolerance = 1e-8)
})

test_that("eta at a centered SVC NUTS fit's stored draws is unaffected", {
  skip_if_not_slow()
  d <- nc_field_fixture(seed = 3L)
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
              control = nc_ctl,
              spatial = spatial_svc(~ lon + lat, nn = 8L,
                                   parameterization = "centered"))
  ppred_eta <- tulpa:::.tulpa_eta_draws_sampler(fit)
  mi <- fit$model_inputs
  eta_raw <- tulpa:::cpp_tulpa_glmm_eta_draws(
    draws = fit$draws, y = mi$y, n_trials = mi$n_trials, X = mi$X,
    family = mi$family, phi = mi$phi, sigma_beta = mi$sigma_beta,
    offset_nullable = mi$offset, re_spec = mi$re_spec,
    spatial_spec = mi$spatial_spec, temporal_spec = mi$temporal_spec,
    sigma_re_scale = mi$sigma_re_scale, phi2 = mi$phi2,
    svc_spec = mi$svc_spec, tvc_spec = mi$tvc_spec, zi_spec = mi$zi_spec)
  # Centered was never double-transformed, so the correction is a no-op.
  expect_equal(unname(ppred_eta), unname(eta_raw))
})

test_that("eta at a noncentered GP NUTS fit's stored draws matches stored gp_w", {
  skip_if_not_slow()
  d <- nc_field_fixture(seed = 4L)
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "hmc",
              control = nc_ctl,
              spatial = spatial_gp(~ lon + lat, nn = 8L,
                                  parameterization = "noncentered"))
  expect_equal(fit$backend, "hmc")

  fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
    t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
  w_cols <- grep("^gp_w", colnames(fit$draws))
  expect_length(w_cols, nrow(d))
  stored_w <- fit$draws[, w_cols, drop = FALSE]

  ppred_eta <- tulpa:::.tulpa_eta_draws_sampler(fit)
  expect_equal(unname(ppred_eta - fixed), unname(stored_w), tolerance = 1e-8)
})

test_that("a non-hmc sampler backend's stored field is read unchanged", {
  skip_if_not_slow()
  d <- nc_field_fixture(n = 40L, seed = 5L)
  # sghmc carries the same store contract question as hmc but does not run
  # through hmc_nuts_chain_iter_store.h, so its stored slice is whatever the
  # kernel natively samples -- .stored_draw_field_specs() must leave a
  # non-"hmc" fit's spec untouched.
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "sghmc",
              control = list(n_iter = 100L, warmup = 60L, seed = 1L),
              spatial = spatial_svc(~ lon + lat, nn = 6L,
                                   parameterization = "noncentered"))
  expect_false(identical(fit$backend, "hmc"))
  fs <- tulpa:::.stored_draw_field_specs(fit, fit$model_inputs)
  expect_identical(fs$svc_spec$svc_parameterization,
                   fit$model_inputs$svc_spec$svc_parameterization)
})
