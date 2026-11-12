test_that("automatic SVC coordinates retain an independent unit Gaussian field prior", {
  skip_on_cran_budget()
  d <- data.frame(lon = c(.05, .2, .4, .6, .8, .95),
    lat = c(.1, .7, .3, .8, .2, .6), x = c(-1, .5, -.3, 1, -.7, .8))
  X <- cbind(1, x = d$x)
  spec <- .svc_sampler_spec(validate_svc(
    spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 3L), d, X), X)
  for (range_prior in c("pc", "uniform")) {
    spec$range_prior <- range_prior
    spec$range_lower <- .1
    spec$range_upper <- 2
    probe <- function(q = NULL) tulpa:::cpp_test_field_coordinates(
      y = .2 + .5 * d$x, X = X, svc_spec = spec, shift = 0, point = q,
      native_point = FALSE)
    initial <- probe()$q
    field <- seq.int(5L, length(initial))
    for (scale in c(-4, 0, 2)) for (range in c(-1, .5)) {
      q <- initial
      q[3:4] <- c(scale, range)
      zero <- probe(q)
      q[field] <- .5 * sin(seq_along(field))
      mixed <- probe(q)
      expect_equal(mixed$x, q, tolerance = 1e-12)
      expect_equal(mixed$jacobian, 0)
      expect_equal(mixed$prior_lp - zero$prior_lp, -.5 * sum(q[field]^2), tolerance = 1e-9)
      expect_equal(mixed$prior_gradient[3:4], zero$prior_gradient[3:4], tolerance = 1e-9)
      expect_equal(mixed$inverse, q, tolerance = 1e-12)
      expect_equal(mixed$gradient, mixed$numerical, tolerance = 1e-5)
      expect_equal(mixed$prior_gradient, mixed$numerical_prior, tolerance = 1e-5)
      expect_true(mixed$verified)
    }
  }
})

test_that("automatic SVC readback returns the stored field at its sites", {
  skip_on_cran_budget()
  d <- data.frame(lon = c(.05, .2, .4, .6, .8, .95),
    lat = c(.1, .7, .3, .8, .2, .6), x = c(-1, .5, -.3, 1, -.7, .8))
  d$y <- .2 + .5 * d$x
  fit <- tulpa(y ~ x, d, family = "gaussian", phi = 1, mode = "exact",
    spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 3L),
    control = list(n_iter = 80L, warmup = 40L, n_chains = 1L, seed = 1172L))
  field <- fit$draws[, grep("^svc_w\\[", colnames(fit$draws)), drop = FALSE]
  expect_equal(unname(svc(fit)$draws[, , 1L]), unname(field), tolerance = 1e-10)
  fixed <- fit$draws[, c("(Intercept)", "x"), drop = FALSE] %*% t(fit$model_inputs$X)
  expect_equal(unname(tulpa:::.tulpa_eta_draws_sampler(fit) - fixed),
               unname(sweep(field, 2L, fit$model_inputs$X[, "x"], "*")), tolerance = 1e-10)
})
