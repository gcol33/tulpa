# spatial_range() / temporal_corr() report every hyperparameter of the field
# they are asked about, and only that field's (gcol33/tulpa#906), in the user's
# units (gcol33/tulpa#907), on every door that fits one (gcol33/tulpa#910).

lattice_adj_hy <- function(nr) {
  S <- nr * nr; A <- matrix(0, S, S); id <- function(i, j) (j - 1L) * nr + i
  for (i in 1:nr) for (j in 1:nr) {
    if (i < nr) A[id(i, j), id(i + 1L, j)] <- A[id(i + 1L, j), id(i, j)] <- 1
    if (j < nr) A[id(i, j), id(i, j + 1L)] <- A[id(i, j + 1L), id(i, j)] <- 1
  }
  A
}

fake_draw_fit <- function(cols, spatial = NULL) {
  set.seed(3)
  dr <- matrix(rnorm(200 * length(cols)), 200, length(cols),
               dimnames = list(NULL, cols))
  structure(list(draws = dr, spatial = spatial), class = c("tulpa_fit", "list"))
}

test_that("sampler BYM2 and proper-CAR fits report both hyperparameters", {
  bym <- spatial_range(fake_draw_fit(c("beta[1]", "log_sigma_spatial",
                                       "logit_rho_bym2")))
  expect_setequal(rownames(bym), c("sigma", "rho"))
  A <- lattice_adj_hy(4L)
  sp <- spatial_car_proper(A, group_var = "region")
  car <- spatial_range(fake_draw_fit(c("log_tau_spatial", "logit_rho_car"), sp))
  expect_setequal(rownames(car), c("sigma", "rho"))
  rb <- sp$rho_bounds
  expect_true(car["rho", "q2.5"] > min(rb) && car["rho", "q97.5"] < max(rb))
})

test_that("a fit that conditioned on the hyperparameters says so", {
  f <- structure(list(draws = NULL, spatial = list(type = "icar"),
                      temporal = list(type = "ar1")),
                 class = c("tulpa_fit", "list"))
  expect_error(spatial_range(f), "conditions on them")
  expect_error(temporal_corr(f), "conditions on them")
  expect_error(spatial_range(structure(list(), class = "tulpa_fit")),
               "Is this a spatial model")
})

test_that("coordinates are standardized by one common factor", {
  set.seed(2)
  xy <- cbind(lon = runif(40, 0, 10), lat = runif(40, 0, 1))
  s <- tulpa:::.scale_coords_isotropic(xy)
  # Every distance shrinks by the same factor, so ratios survive.
  d0 <- as.numeric(dist(xy)); d1 <- as.numeric(dist(s))
  expect_equal(d0 / d1, rep(d0[1] / d1[1], length(d0)), tolerance = 1e-12)
  k <- attr(s, "scaled:scale")
  expect_equal(k, rep(sqrt(mean(apply(xy, 2, var))), 2))
  expect_equal(unname(tulpa:::.unscale_coords(s)), unname(xy), tolerance = 1e-12)
  sp <- validate_gp(spatial_gp(~ lon + lat), as.data.frame(xy))
  expect_equal(tulpa:::.coord_scale(sp), k[1])
  sp0 <- validate_gp(spatial_gp(~ lon + lat, scale_coords = FALSE),
                     as.data.frame(xy))
  expect_identical(tulpa:::.coord_scale(sp0), 1)
})

test_that("a temporal GP's lengthscale support sits on the spread of its times", {
  set.seed(1); tt <- sort(round(runif(12, 0, 100), 1))
  d <- data.frame(t = rep(tt, each = 2))
  spec <- validate_temporal_gp(temporal_gp("t"), d)
  expect_equal(c(spec$phi_prior_lower, spec$phi_prior_upper), c(0.01, 10))
  raw <- validate_temporal_gp(temporal_gp("t", scale_coords = FALSE), d)
  expect_equal(c(raw$phi_prior_lower, raw$phi_prior_upper),
               c(0.01, 10) * sd(d$t))
  # A lengthscale draw reads back in the user's time units either way.
  expect_equal(tulpa:::.gp_phi_from_logit(0, spec),
               (0.01 + 9.99 / 2) * spec$time_scale)
  expect_equal(tulpa:::.gp_phi_from_logit(0, raw), (0.01 + 9.99 / 2) * sd(d$t))
})

test_that("a GP range is reported in data units whether or not coords are scaled", {
  skip_on_cran()
  set.seed(31); n <- 150; lon <- runif(n); lat <- runif(n)
  D <- as.matrix(dist(cbind(lon, lat)))
  w <- as.numeric(t(chol(0.64 * exp(-D / 0.15) + diag(1e-8, n))) %*% rnorm(n))
  d <- data.frame(lon, lat, x = rnorm(n))
  d$y <- 0.5 + 0.8 * d$x + w + rnorm(n, sd = 0.3)
  fit_range <- function(sc) {
    f <- suppressMessages(tulpa(
      y ~ x, data = d, phi = 0.09, mode = "nested_laplace",
      spatial = spatial_gp(~ lon + lat, approx = "nngp", scale_coords = sc),
      control = list(n_threads = 1)))
    spatial_range(f)["range", ]
  }
  r_sc <- fit_range(TRUE); r_raw <- fit_range(FALSE)
  # Both in data units (true practical range 0.45), and the two agree.
  expect_lt(abs(log(r_sc$mean / r_raw$mean)), log(1.6))
  expect_true(r_sc$q2.5 < 0.45 && r_sc$q97.5 > 0.45)
})

test_that("a (1 | g) term is not read as a temporal field", {
  skip_on_cran()
  A <- lattice_adj_hy(5L); S <- nrow(A)
  set.seed(1)
  d <- data.frame(region = rep(1:S, 8), g = factor(sample(1:8, 200, TRUE)),
                  x = rnorm(200))
  u <- rnorm(8, 0, 1.5); d$y <- rpois(200, exp(0.3 + 0.5 * d$x + u[d$g]))
  ft <- tulpa(y ~ x + spatial(region) + (1 | g), data = d, family = "poisson",
              mode = "nested_laplace",
              spatial = spatial_car(A, group_var = "region"),
              control = list(n_threads = 1, progress = FALSE))
  expect_error(temporal_corr(ft), "temporal")
  expect_identical(rownames(spatial_range(ft)), "sigma")
})

test_that("a proper-CAR correlation interval stays on its support", {
  skip_on_cran()
  A <- lattice_adj_hy(8L); n <- nrow(A)
  set.seed(11); ph <- backsolve(chol(2 * (diag(rowSums(A)) - 0.9 * A)), rnorm(n))
  set.seed(10); d <- data.frame(region = rep(1:n, each = 4), x = rnorm(4 * n))
  d$y <- rpois(nrow(d), exp(0.2 + 0.5 * d$x + ph[d$region]))
  f <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
             mode = "nested_laplace",
             spatial = spatial_car_proper(A, group_var = "region"),
             control = list(n_threads = 1))
  s <- spatial_range(f)
  rb <- compute_car_rho_bounds(A)
  expect_true(s["rho", "q2.5"] >= min(rb) && s["rho", "q97.5"] <= max(rb))
})

test_that("fit_st_nested() fits are read by the generic accessors", {
  skip_on_cran()
  A <- lattice_adj_hy(6L); S <- nrow(A); Tn <- 10L
  set.seed(41); phi <- rnorm(S, 0, 0.7); f <- 1.2 * sin(1:Tn)
  d <- expand.grid(region = 1:S, time = 1:Tn); set.seed(43)
  d$x <- rnorm(nrow(d))
  d$y <- rpois(nrow(d), exp(0.3 + 0.5 * d$x + phi[d$region] + f[d$time]))
  ft <- fit_st_nested(d$y, cbind(1, d$x), d$region, A, d$time, Tn,
                      spatial_type = "icar", temporal_type = "ar1",
                      family = "poisson", control = list(n_threads = 1))
  expect_identical(rownames(spatial_range(ft)), "sigma")
  tc <- temporal_corr(ft)
  expect_setequal(rownames(tc), c("precision", "rho_ar1"))
  expect_true(tc["rho_ar1", "q2.5"] > -1 && tc["rho_ar1", "q97.5"] < 1)
  tp <- temporal(ft, summary = TRUE)
  expect_equal(tp$mean, ft$temporal_effects, tolerance = 1e-8)
  expect_gt(cor(tp$mean, f), 0.95)
  expect_true(all(tp$sd > 0))
})
