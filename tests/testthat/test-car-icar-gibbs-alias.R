# spatial_car() (the exported intrinsic-CAR / ICAR constructor, proper = FALSE)
# returns type = "car". The nested-Laplace path already treats "car" as the
# same field as a bare `list(type = "icar", ...)` (.NL_FRONTDOOR_AREAL,
# .spatial_spec_to_nl_prior()), but dispatch_gibbs_spatial() and auto's
# Gibbs-eligibility check matched only the literal "icar", so mode = "gibbs"
# refused a spatial_car() spec and auto routed it to nested Laplace instead of
# the Polya-Gamma Gibbs sampler the equivalent bare list reached
# (gcol33/tulpa#819). .areal_gibbs_type() (R/spatial_car.R) is the one alias
# both call sites now read.

test_that(".areal_gibbs_type() aliases only 'car', leaving every other type alone", {
  expect_equal(tulpa:::.areal_gibbs_type("car"), "icar")
  for (t in c("icar", "bym2", "rsr", "car_proper", "gp", "nngp",
             "multiscale", "multiscale_gp", "")) {
    expect_equal(tulpa:::.areal_gibbs_type(t), t)
  }
  expect_null(tulpa:::.areal_gibbs_type(NULL))
})

test_that("auto routes a spatial_car() binomial model to Gibbs, matching a bare icar list", {
  sel_car <- tulpa:::auto_select_mode(
    family = list(name = "binomial"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "car")
  sel_icar <- tulpa:::auto_select_mode(
    family = list(name = "binomial"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "icar")
  expect_equal(sel_car$backend, "gibbs")
  expect_equal(sel_car$backend, sel_icar$backend)

  # car_proper is a different (proper CAR) field and stays off the Gibbs route.
  sel_car_proper <- tulpa:::auto_select_mode(
    family = list(name = "binomial"), n_obs = 100L,
    has_spatial = TRUE, has_temporal = FALSE, has_latent = FALSE,
    spatial_type = "car_proper")
  expect_equal(sel_car_proper$backend, "nested_laplace")
})

test_that("mode = 'gibbs' accepts spatial_car() and reproduces the bare-list icar fit", {
  skip_if_not_slow()
  set.seed(4)
  W <- rook_adj(6L, 6L)
  n_units <- 36L
  ar <- data.frame(region = rep(seq_len(n_units), each = 4L), x = rnorm(144))
  set.seed(4)
  ar$y  <- rbinom(144, 10, plogis(0.3 + 0.5 * ar$x))
  ar$yc <- rnbinom(144, mu = exp(0.3 + 0.5 * ar$x), size = 5)
  sc <- spatial_car(W, level = "group", group_var = "region")
  expect_equal(sc$type, "car")
  gctl <- list(n_iter = 60L, warmup = 30L, seed = 1L)

  f_car  <- tulpa(y ~ x + spatial(region), data = ar, family = "binomial",
                  n_trials = 10L, mode = "gibbs", spatial = sc, control = gctl)
  f_icar <- tulpa(y ~ x + spatial(region), data = ar, family = "binomial",
                  n_trials = 10L, mode = "gibbs",
                  spatial = list(type = "icar", adjacency = W), control = gctl)
  expect_equal(f_car$backend, "gibbs")
  expect_equal(unname(as.matrix(f_car$draws)), unname(as.matrix(f_icar$draws)))

  # Previously refused outright ("Spatial Gibbs not wired for type 'car'").
  f_negbin <- tulpa(yc ~ x + spatial(region), data = ar, family = "neg_binomial_2",
                    mode = "gibbs", spatial = sc, control = gctl)
  expect_equal(f_negbin$backend, "gibbs")

  # auto now agrees with the explicit request, matching the bare-list route.
  f_auto <- tulpa(y ~ x + spatial(region), data = ar, family = "binomial",
                  n_trials = 10L, spatial = sc, control = gctl)
  expect_equal(f_auto$backend, "gibbs")
})
