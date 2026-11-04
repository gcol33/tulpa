test_that("the PC anchor has the same range meaning on HSGP and SPDE", {
  skip_on_cran_budget()
  U <- 0.7; alpha <- 0.2
  block <- list(type = "hsgp", prior_range = c(U, alpha))
  hp <- tulpa:::.hp_axis_prior("lengthscale", block)$fn
  spde <- tulpa:::.hp_axis_prior("range", list(type = "spde",
                                             prior_range = c(U, alpha)))$fn
  ell <- exp(seq(-4, 3, length.out = 100L))
  expect_equal(hp(ell), spde(2 * ell), tolerance = 1e-12)
  nngp <- tulpa:::.hp_axis_prior("phi_gp", list(type = "nngp",
                                              prior_range = c(U, alpha)))$fn
  expect_equal(nngp(ell), spde(2 * ell), tolerance = 1e-12)
  cdf <- integrate(function(x) exp(hp(x)) / x, 0, U / 2,
                    rel.tol = 1e-9)$value
  expect_equal(cdf, alpha, tolerance = 1e-8)
  expect_equal(tulpa:::.hsgp_prior_median(c(U, alpha)),
                U * log(alpha) / log(0.5) / 2)
  set.seed(120)
  d <- data.frame(lon = runif(30), lat = runif(30), x = rnorm(30))
  v <- validate_hsgp(spatial_gp(~ lon + lat, approx = "hsgp"), d)
  p <- tulpa:::.hsgp_range_prior(v$coords_matrix)
  sampler <- tulpa:::.hsgp_sampler_spec(v)
  expect_equal(sampler$lengthscale_prior_U, p[1L] / 2)
  svc <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x - 1,
                                 approx = "hsgp"), d, model.matrix(~ x, d))
  sampler_svc <- tulpa:::.svc_sampler_spec(svc, model.matrix(~ x, d))
  expect_equal(sampler_svc$phi_prior_U, p[1L] / 2)
  ms <- suppressMessages(validate_gp(spatial_multiscale(
    ~ lon + lat, range_local = c(0.2, 1), range_regional = c(1, 10)), d))
  sampler_ms <- tulpa:::.msgp_sampler_spec(ms)
  expect_equal(sampler_ms$range_local_lower, 0.2 / 2)
  expect_equal(sampler_ms$range_regional_lower, 1 / 2)
  gp <- validate_gp(spatial_gp(~ lon + lat), d)
  prior_gp <- tulpa:::.nl_default_range_prior(gp$unique_coords)
  gp_sampler <- tulpa:::.gp_sampler_spec(gp)
  expect_equal(c(gp_sampler$phi_prior_U, gp_sampler$phi_prior_alpha),
                 prior_gp / c(2, 1))
  local_mocked_bindings(
    cpp_pg_binomial_gibbs_multiscale_gp = function(...) list(...),
    .pg_as_chain = function(raw, ...) raw,
    .package = "tulpa")
  gibbs <- tulpa:::dispatch_gibbs_spatial(
    rep(0L, nrow(d)), rep(1L, nrow(d)), model.matrix(~ x, d),
    rep(0L, nrow(d)), 0L, ms, "binomial", iter = 10L, warmup = 5L)
  expect_equal(c(gibbs$prior_phi_local_lower, gibbs$prior_phi_local_upper,
                 gibbs$prior_phi_regional_lower, gibbs$prior_phi_regional_upper),
                 c(0.1, 0.5, 0.5, 5))
  expect_equal(c(gibbs$phi_local_init, gibbs$phi_regional_init), c(0.3, 2.75))
})

