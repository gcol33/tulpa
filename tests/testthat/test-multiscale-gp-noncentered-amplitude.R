# Multi-scale (local + regional) GP field: front-door NUTS wiring + amplitude
# recovery (gcol33/tulpa#243).
#
# spatial_multiscale() + tulpa(mode = "exact") had no route through
# build_sampler_model_inputs() before this issue -- compute_multiscale_gp_prior
# and its ParamLayout allocation existed, but no R-side stype branch ever set
# SpatialType::MULTISCALE_GP, so the path was unreachable. This is the first
# front door for it, wired alongside the fix so the non-centered transform
# (msgp_parameterization, mirroring gp_parameterization / svc_parameterization)
# has something to actually test end to end.
#
# The two scales share the exact centered field/hyperparameter funnel GP and
# SVC show; msgp_nc_apply.cpp reconstructs each scale's field independently
# via the shared nngp_nc_term_apply.h primitive. This checks amplitude, not
# just correlation, same reasoning as test-gp-noncentered-amplitude.R.

sim_msgp <- function(n = 150L, sigma2_local = 1.0, phi_local = 0.08,
                     sigma2_regional = 1.0, phi_regional = 1.5,
                     a0 = 0.3, seed = 1L, box = 1) {
  set.seed(seed)
  lon <- runif(n, 0, box)
  lat <- runif(n, 0, box)
  D <- as.matrix(dist(cbind(lon, lat)))
  draw_field <- function(sigma2, phi) {
    K <- sigma2 * exp(-D / phi)
    L <- chol(K + diag(1e-8, n))
    as.numeric(t(L) %*% rnorm(n))
  }
  w_local <- draw_field(sigma2_local, phi_local)
  w_regional <- draw_field(sigma2_regional, phi_regional)
  y <- rpois(n, exp(a0 + w_local + w_regional))
  data.frame(lon = lon, lat = lat, y = y,
             w_local_true = w_local, w_regional_true = w_regional)
}

test_that("auto routes a multiscale field to nested Laplace, the sampler as fallback", {
  # A multi-scale field rides the nested-Laplace multi-block driver as two
  # NNGP blocks; the exact ModelData sampler carries it when nested Laplace
  # cannot carry the call (a zero-inflation formula).
  sel <- tulpa:::auto_select_mode(
    family = list(name = "poisson"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "multiscale")
  expect_equal(sel$backend, "nested_laplace")
  sel_zi <- tulpa:::auto_select_mode(
    family = list(name = "poisson"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "multiscale", feat = list(ziformula = TRUE))
  expect_equal(sel_zi$mode, "exact")
  expect_equal(tulpa:::BACKEND_REGISTRY[[sel_zi$backend]]$input, "modeldata")
})

skip_on_cran()

test_that("non-centered multiscale GP NUTS recovers both scales' amplitude", {
  skip_if_not_slow()
  # The per-scale non-centered transform itself is verified directly by
  # test-nngp-nc-grad.R, which finite-differences nngp_nc_backward on both
  # scales' views; this checks it end to end. The ranges are stated in the
  # units the truth is simulated in (scale_coords = FALSE): on standardized
  # coordinates the regional truth (phi = 4 on a 10 x 10 box) falls below
  # range_regional's lower anchor, so the regional prior lengthens it and the
  # local scale absorbs the field on both routes.
  #
  # The amplitude is read off each scale's SD, whose interval has to hold the
  # simulated 1. The sd of the posterior MEAN field is not an amplitude: at
  # 80 locations the regional mean field is shrunk to 0.21-0.40 of the true
  # field's sd under NUTS and to 0.22-0.42 under nested Laplace (seeds 1-3),
  # so it bounds only a re-added z -> w Jacobian from above.
  d <- sim_msgp(n = 80L, sigma2_local = 1.0, phi_local = 1.0,
                sigma2_regional = 1.0, phi_regional = 4.0, seed = 1L, box = 10)

  fit <- tulpa(
    y ~ 1, data = d, family = "poisson",
    spatial = spatial_multiscale(~ lon + lat, nn_local = 8L, nn_regional = 8L,
                                 range_local = c(0.2, 1.8),
                                 range_regional = c(2, 6),
                                 scale_coords = FALSE),
    # "auto" (the spatial_multiscale() default) resolves to non-centered.
    mode = "exact",
    control = list(n_iter = 600L, n_warmup = 300L, seed = 7L))

  pn <- colnames(fit$draws)
  local_hat <- colMeans(fit$draws[, grep("^gp_local\\[", pn), drop = FALSE])
  regional_hat <- colMeans(fit$draws[, grep("^gp_regional\\[", pn), drop = FALSE])

  ratio_local <- sd(local_hat) / sd(d$w_local_true)
  ratio_regional <- sd(regional_hat) / sd(d$w_regional_true)

  for (sc in c("local", "regional")) {
    sig <- exp(fit$draws[, paste0("log_sigma2_gp_", sc)] / 2)
    q <- stats::quantile(sig, c(0.025, 0.975))
    expect_lt(q[[1L]], 1)
    expect_gt(q[[2L]], 1)
  }
  expect_gt(ratio_local, 0.5)
  expect_lt(ratio_local, 2.0)
  expect_lt(ratio_regional, 2.0)
})

test_that("multiscale GP fits under exact NUTS and allocates both scales", {
  skip_if_not_slow()
  d <- sim_msgp(n = 40L, seed = 2L)
  fit <- tulpa(
    y ~ 1, data = d, family = "poisson",
    spatial = spatial_multiscale(~ lon + lat, nn_local = 8L, nn_regional = 8L,
                                 range_local = c(0.01, 1), range_regional = c(1, 10)),
    mode = "exact",
    control = list(n_iter = 120L, n_warmup = 60L, seed = 1L))
  pn <- colnames(fit$draws)
  expect_true(all(c("log_sigma2_gp_local", "log_phi_gp_local",
                    "log_sigma2_gp_regional", "log_phi_gp_regional") %in% pn))
  expect_equal(sum(grepl("^gp_local\\[", pn)), 40L)
  expect_equal(sum(grepl("^gp_regional\\[", pn)), 40L)
})

test_that("a multiscale range starts from its bounds and the chain moves", {
  # Regression for the start-outside-the-support bug (#243) and for the range
  # prior that made it fatal (#244). With bounds excluding phi = 1 -- the
  # origin every coordinate starts at -- the old hard -INFINITY box left the
  # chain starting at -Inf: it never moved, every draw was divergent, and the
  # field came back all zeros with phi pinned at exactly 1. The prior is now
  # proper on (0, inf) and init_bounded_support_params() starts each scale from
  # its own bounds, so the guard is that the chain SAMPLES rather than that it
  # stays boxed (a proper prior may legitimately wander outside the bounds).
  skip_if_not_slow()
  set.seed(3)
  n <- 40L
  d <- data.frame(lon = runif(n, 0, 10), lat = runif(n, 0, 10))
  d$y <- rpois(n, 3)

  fit <- tulpa(
    y ~ 1, data = d, family = "poisson",
    spatial = spatial_multiscale(~ lon + lat, nn_local = 8L, nn_regional = 8L,
                                 # Deliberately excludes phi = 1, the origin
                                 # start, on BOTH scales.
                                 range_local = c(2, 6), range_regional = c(8, 20)),
    mode = "exact",
    control = list(n_iter = 200L, n_warmup = 100L, seed = 1L))

  phi_local <- exp(fit$draws[, "log_phi_gp_local"])
  phi_regional <- exp(fit$draws[, "log_phi_gp_regional"])
  fld <- fit$draws[, grep("^gp_local\\[", colnames(fit$draws)), drop = FALSE]

  # Not frozen at the origin's phi = 1 (the stuck-chain signature), and not
  # frozen at the start either.
  expect_false(all(abs(phi_local - 1) < 1e-8))
  expect_false(all(abs(phi_regional - 1) < 1e-8))
  expect_gt(sd(phi_local), 0)
  expect_gt(sd(phi_regional), 0)
  # The field is sampled, not returned as all-zeros.
  expect_gt(sd(as.numeric(fld)), 0)
  # And the fit is not the 100%-divergent wall behaviour.
  expect_lt(mean(fit$divergent), 0.5)
})
