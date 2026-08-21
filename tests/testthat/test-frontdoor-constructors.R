# Structural coverage for the exported spec constructors whose fitting paths
# live in the companion model packages (or are experimental): the
# spatiotemporal, temporal-GP, TVC/RTR, and SVC front doors. Constructors no
# fitter consumes refuse to build; the rest must hold their construction,
# field layout, and validation errors even where no end-to-end tulpa() route
# exists yet.

.fc_adj <- function(n = 6L) {
  adj <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

test_that("spatiotemporal() and spatiotemporal_gp() refuse to build an unfitted spec", {
  sp <- spatial_car(.fc_adj(), level = "group", group_var = "region")
  tm <- temporal_rw1("year")

  for (ty in c("I", "II", "III", "IV", "iid", "separable")) {
    expect_error(spatiotemporal(spatial = sp, temporal = tm, type = ty),
                 "not fitted by any tulpa backend")
  }
  expect_error(spatiotemporal_gp(~ lon + lat, time_var = "year", nn = 10),
               "not fitted by any tulpa backend")
  expect_error(spatiotemporal_gp(c("x", "y"), "t"),
               "not fitted by any tulpa backend")
})

test_that("temporal_gp() validates covariance-specific parameters", {
  tg <- temporal_gp("year")
  expect_s3_class(tg, "tulpa_temporal")
  expect_identical(tg$type, "gp")
  expect_identical(tg$time_var, "year")

  expect_identical(temporal_gp("year", cov = "matern", nu = 2.5)$nu, 2.5)
  expect_null(temporal_gp("year", cov = "exponential")$nu)
  expect_identical(temporal_gp("year", cov = "periodic", period = 12)$period, 12)

  expect_error(temporal_gp(c("a", "b")), "single character")
  expect_error(temporal_gp("year", cov = "matern", nu = -1), "positive")
  expect_error(temporal_gp("year", cov = "periodic"), "period")
})

test_that("temporal_tvc() builds all three terms forms; tvc() rejects non-TVC fits", {
  tv <- temporal_tvc("year", terms = 2L, structure = "rw2")
  expect_s3_class(tv, "tulpa_tvc")
  expect_s3_class(tv, "tulpa_temporal")
  expect_identical(tv$structure, "rw2")
  expect_identical(tv$terms_spec$type, "index")

  expect_identical(temporal_tvc("year", terms = "x")$terms_spec$type, "names")
  expect_identical(temporal_tvc("year", terms = ~ x)$terms_spec$type, "formula")
  expect_error(temporal_tvc(1L), "single character")
  expect_error(temporal_tvc("year", terms = TRUE), "formula, integer")

  fit_stub <- structure(list(), class = c("tulpa_fit", "list"))
  expect_error(tvc(fit_stub), "TVC|tvc")
})

test_that("temporal_rtr() refuses to build a spec no fitter projects", {
  tm <- temporal_rw1("year")
  expect_error(temporal_rtr(tm, restrict_to = ~ x),
               "not fitted by any tulpa backend")
  expect_error(temporal_rtr(list(), ~ x), "not fitted by any tulpa backend")
})

test_that("spatial_svc() builds NNGP and HSGP variants and validates knobs", {
  sv <- spatial_svc(~ lon + lat, terms = 1L)
  expect_identical(sv$coord_vars, c("lon", "lat"))
  expect_identical(sv$terms_spec$type, "index")

  sh <- spatial_svc(c("x", "y"), terms = "slope", approx = "hsgp", m = 8)
  expect_identical(sh$terms_spec$type, "names")

  expect_error(spatial_svc(~ lon, terms = 1L), "exactly 2")
  expect_error(spatial_svc(~ lon + lat, terms = TRUE), "formula, integer")
  expect_error(spatial_svc(~ lon + lat, terms = 1L, nn = 0), "positive")
  expect_error(spatial_svc(~ lon + lat, terms = 1L, approx = "hsgp", m = 2),
               "between 3 and 50")
  expect_error(
    spatial_svc(~ lon + lat, terms = 1L, approx = "hsgp", c_boundary = 0.5),
    ">= 1")
})

test_that("spatiotemporal_effects() rejects a fit that carries no ST field", {
  fit_stub <- structure(list(), class = c("tulpa_fit", "list"))
  expect_error(spatiotemporal_effects(fit_stub), "spatiotemporal|ST|field")
})
