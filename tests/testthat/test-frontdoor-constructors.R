# Structural coverage for the exported spec constructors whose fitting paths
# live in the companion model packages (or are experimental): the
# spatiotemporal, temporal-GP, TVC/RTR, and SVC front doors. These are
# exported surface -- their construction, field layout, and validation
# errors must hold even where no end-to-end tulpa() route exists yet.

.fc_adj <- function(n = 6L) {
  adj <- matrix(0, n, n)
  for (i in seq_len(n - 1L)) adj[i, i + 1L] <- adj[i + 1L, i] <- 1
  adj
}

test_that("spatiotemporal() builds every interaction type and validates inputs", {
  sp <- spatial_car(.fc_adj(), level = "group", group_var = "region")
  tm <- temporal_rw1("year")

  for (ty in c("I", "II", "III", "IV")) {
    st <- spatiotemporal(spatial = sp, temporal = tm, type = ty)
    expect_s3_class(st, "tulpa_spatiotemporal")
    expect_identical(st$type, ty)
  }
  # "iid" normalizes to type I.
  expect_identical(spatiotemporal(sp, tm, type = "iid")$type, "I")

  expect_error(spatiotemporal(spatial = list(), temporal = tm),
               "spatial")
  expect_error(spatiotemporal(spatial = sp, temporal = list()),
               "temporal")
  # separable over CAR x RW warns toward type IV.
  expect_warning(spatiotemporal(sp, tm, type = "separable"),
                 "Separable")
})

test_that("spatiotemporal_gp() parses coords/time and validates nn", {
  st <- spatiotemporal_gp(~ lon + lat, time_var = "year", nn = 10)
  expect_identical(st$type, "st_gp")
  expect_identical(st$coord_vars, c("lon", "lat"))
  expect_identical(st$time_var, "year")
  expect_identical(st$nn, 10L)

  expect_identical(spatiotemporal_gp(c("x", "y"), "t")$coord_vars, c("x", "y"))
  expect_error(spatiotemporal_gp(~ lon, "year"), "exactly 2")
  expect_error(spatiotemporal_gp(1:2, "year"), "formula")
  expect_error(spatiotemporal_gp(~ lon + lat, c("a", "b")), "single character")
  expect_error(spatiotemporal_gp(~ lon + lat, "year", nn = 0), "positive")
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

test_that("temporal_rtr() stamps the RTR modifier and validates its inputs", {
  tm <- temporal_rw1("year")
  rt <- temporal_rtr(tm, restrict_to = ~ x)
  expect_s3_class(rt, "tulpa_rtr")
  expect_s3_class(rt, "tulpa_temporal")
  expect_true(rt$rtr)
  expect_identical(rt$rtr_formula, ~ x)
  expect_output(print(rt), "Restricted Temporal Regression")

  expect_error(temporal_rtr(list(), ~ x), "temporal")
  expect_error(temporal_rtr(tm, "x"), "formula")
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
