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

test_that("auto mode routes a multiscale field to the exact ModelData backend", {
  # The multi-scale field has no nested-Laplace kernel and
  # dispatch_laplace_spatial() rejects it, so auto must pick the sampler that
  # fits it rather than a Laplace heuristic whose dispatch errors.
  sel <- tulpa:::auto_select_mode(
    family = list(name = "poisson"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "multiscale")
  expect_equal(sel$mode, "exact")
  expect_equal(tulpa:::BACKEND_REGISTRY[[sel$backend]]$input, "modeldata")
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

test_that("non-centered multiscale GP NUTS recovers both scales' amplitude", {
  skip("blocked on gcol33/tulpa#244 (multiscale range prior)")
  # The per-scale non-centered transform itself is verified directly by
  # test-nngp-nc-grad.R, which finite-differences nngp_nc_backward on both
  # scales' views. What cannot be asserted yet is END-TO-END amplitude
  # recovery, because #244 leaves each scale's range under a Uniform prior
  # behind a hard -INFINITY wall and the sampler spends most of its time
  # against it.
  #
  # Measured on this block at n = 80, 300 iter / 250 warmup, two draws, on a
  # box = 10 domain with phi_local = 1.0 / phi_regional = 4.0 and bounds
  # centred on those truths (the most favourable configuration found -- wide
  # or off-centre bounds are worse, because the in-bounds density is a
  # Uniform on phi whose mean is the box centre):
  #
  #   draw 1  noncentered  83% divergent  sd ratio local 0.96  regional 0.20
  #   draw 1  centered     83% divergent  sd ratio local 0.92  regional 0.38
  #   draw 2  noncentered  88% divergent  sd ratio local 0.72  regional 0.38
  #   draw 2  centered     82% divergent  sd ratio local 0.87  regional 0.15
  #
  # At those divergence rates the centered / non-centered difference changes
  # sign between draws, so an amplitude band here would be asserting noise.
  # Restore this test with #244; the configuration below is the one to use.
  d <- sim_msgp(n = 80L, sigma2_local = 1.0, phi_local = 1.0,
                sigma2_regional = 1.0, phi_regional = 4.0, seed = 1L, box = 10)

  fit <- tulpa(
    y ~ 1, data = d, family = "poisson",
    spatial = spatial_multiscale(~ lon + lat, nn_local = 8L, nn_regional = 8L,
                                 range_local = c(0.2, 1.8),
                                 range_regional = c(2, 6)),
    # "auto" (the spatial_multiscale() default) resolves to non-centered.
    mode = "exact",
    control = list(n_iter = 300L, n_warmup = 250L, seed = 7L))

  pn <- colnames(fit$draws)
  local_hat <- colMeans(fit$draws[, grep("^gp_local\\[", pn), drop = FALSE])
  regional_hat <- colMeans(fit$draws[, grep("^gp_regional\\[", pn), drop = FALSE])

  ratio_local <- sd(local_hat) / sd(d$w_local_true)
  ratio_regional <- sd(regional_hat) / sd(d$w_regional_true)

  # Two-sided per scale: too small flags the funnel, too large flags a
  # re-added z -> w Jacobian, same reasoning as the GP / SVC amplitude tests.
  expect_gt(ratio_local, 0.3)
  expect_lt(ratio_local, 2.0)
  expect_gt(ratio_regional, 0.3)
  expect_lt(ratio_regional, 2.0)
})

test_that("a bounded-support range starts inside its own bounds", {
  # The samplers start every coordinate at the origin, which puts phi = 1.
  # Every PC-range block is proper on (0, inf) so that is a valid start, but
  # the multi-scale block rejects a range outside its bounds with a hard
  # -INFINITY (#244): with bounds excluding 1 the chain started at -Inf, never
  # moved, and returned an all-zero field at a 100% divergence rate.
  # init_bounded_support_params() starts each bounded range at the geometric
  # mean of its own bounds. Asserted on the drawn range rather than on the
  # init vector directly, since that is what a stuck chain would expose.
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
    control = list(n_iter = 40L, n_warmup = 20L, seed = 1L))

  phi_local <- exp(fit$draws[, "log_phi_gp_local"])
  phi_regional <- exp(fit$draws[, "log_phi_gp_regional"])

  # Inside the declared support, and not frozen at the origin's phi = 1.
  expect_true(all(phi_local >= 2 & phi_local <= 6))
  expect_true(all(phi_regional >= 8 & phi_regional <= 20))
})
