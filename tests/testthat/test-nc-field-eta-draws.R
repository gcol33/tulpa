# Every ModelData sampler stores a non-centered GP / SVC / multiscale-GP /
# temporal-GP block's slice as the reconstructed field w
# (src/sampler_field_store.h), keeping its own position as z for sampling.
# cpp_tulpa_glmm_eta_draws() re-evaluates a stored draw through
# initialize_generic_state() at the spec's own parameterization, which for a
# non-centered spec would apply the forward transform a SECOND time to a slice
# that is no longer z (gcol33/tulpa#822). .tulpa_eta_draws_sampler() --
# posterior_predict() / simulate() / WAIC / LOO's eta source -- reads the
# stored draws back through .stored_draw_field_specs(), which reports the spec
# as centered: the forward transform is the identity on an already-centered
# field, matching what the store wrote.

skip_on_cran()

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

# The temporal fields the store reconstructs: a non-centred temporal GP's slice
# is z and its stored draw f = sigma L z; a non-centred multiscale trend's
# slice is its spectral coordinates and its stored draw the walk. Each is read
# back as centred, so the stored field is the one eta is built from.
nc_temporal_fixture <- function(n_t = 20L, reps = 3L, seed = 1L) {
  set.seed(seed)
  f <- cumsum(cumsum(rnorm(n_t, 0, 0.05))); f <- f - mean(f)
  t <- rep(seq_len(n_t), each = reps)
  x <- rnorm(length(t))
  data.frame(t = t, x = x, y = 0.3 + 0.5 * x + f[t] + rnorm(length(t), 0, 0.3))
}

test_that("eta at a non-centred temporal GP NUTS fit's stored draws is the stored field", {
  skip_if_not_slow()
  d <- nc_temporal_fixture()
  fit <- tulpa(y ~ x, data = d, family = "gaussian", phi = 0.09, mode = "hmc",
               control = nc_ctl, temporal = temporal_gp("t", parameterization = "noncentered"))
  expect_identical(fit$model_inputs$temporal_spec$parameterization, "noncentered")
  fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
    t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
  f_cols <- grep("^phi_temporal[[]", colnames(fit$draws))
  stored <- fit$draws[, f_cols, drop = FALSE][, d$t, drop = FALSE]
  expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
               unname(stored), tolerance = 1e-8)
})

test_that("eta at a non-centred multiscale trend's stored draws is the stored walk", {
  skip_if_not_slow()
  d <- nc_temporal_fixture()
  fit <- tulpa(y ~ x, data = d, family = "gaussian", phi = 0.09, mode = "hmc",
               control = nc_ctl,
               temporal = temporal_multiscale("t", trend = "rw2",
                                              short_term = "none", parameterization = "noncentered"))
  expect_identical(fit$model_inputs$temporal_spec$parameterization,
                   "noncentered")
  fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
    t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
  tr <- fit$draws[, grep("^trend[[]", colnames(fit$draws)), drop = FALSE]
  # The stored walk is the centred one the likelihood reads.
  expect_lt(max(abs(rowMeans(tr))), 1e-10)
  expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
               unname(tr[, d$t, drop = FALSE]), tolerance = 1e-8)
})

nc_backend_controls <- list(
  sghmc = list(n_iter = 100L, warmup = 60L, seed = 1L),
  mclmc = list(n_iter = 100L, warmup = 60L, seed = 1L),
  smc   = list(n_particles = 100L, smc_max_moves = 3L, seed = 1L),
  vi    = list(vi_max_iter = 200L, n_draws = 100L, seed = 1L)
)

for (backend in names(nc_backend_controls)) {
  local({
    b <- backend
    test_that(sprintf("backend '%s' stores the non-centered fields, not z", b), {
      skip_if_not_slow()
      d <- nc_field_fixture(n = 40L, seed = 5L)
      ctl <- nc_backend_controls[[b]]

      fit <- tulpa(y ~ x, data = d, family = "poisson", mode = b,
                   control = ctl,
                   spatial = spatial_gp(~ lon + lat, nn = 6L,
                                        parameterization = "noncentered"))
      expect_identical(fit$model_inputs$spatial_spec$gp_parameterization, 1L)
      fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
        t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
      gp_w <- fit$draws[, grep("^gp_w", colnames(fit$draws)), drop = FALSE]
      expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
                   unname(gp_w), tolerance = 1e-8)

      fit <- tulpa(y ~ x, data = d, family = "poisson", mode = b,
                   control = ctl,
                   spatial = spatial_svc(~ lon + lat, nn = 6L,
                                         parameterization = "noncentered"))
      expect_identical(fit$model_inputs$svc_spec$svc_parameterization, 1L)
      svc_w <- fit$draws[, grep("^svc_w", colnames(fit$draws)), drop = FALSE]
      expect_lt(max(abs(rowMeans(svc_w))), 1e-10)
      fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
        t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
      expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
                   unname(svc_w), tolerance = 1e-8)

      fit <- tulpa(y ~ x, data = nc_temporal_fixture(), family = "gaussian",
                   phi = 0.09, mode = b, control = ctl,
                   temporal = temporal_gp("t", parameterization = "noncentered"))
      expect_identical(fit$model_inputs$temporal_spec$parameterization,
                       "noncentered")
      dt <- nc_temporal_fixture()
      fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*%
        t(fit$model_inputs$X[, c("(Intercept)", "x"), drop = FALSE])
      f <- fit$draws[, grep("^phi_temporal[[]", colnames(fit$draws)),
                     drop = FALSE]
      expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
                   unname(f[, dt$t, drop = FALSE]), tolerance = 1e-8)
    })
  })
}
