test_that("Gibbs multiscale range intervals are expressed in physical units", {
  skip_on_cran_budget()
  set.seed(119)
  d <- data.frame(lon = runif(30), lat = runif(30), x = rnorm(30))
  ms <- suppressMessages(validate_gp(spatial_multiscale(
    ~ lon + lat, range_local = c(0.2, 1), range_regional = c(1, 10)), d))
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
