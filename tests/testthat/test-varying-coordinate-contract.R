test_that("SVC prior coordinates pass finite differences and the runtime gate", {
  set.seed(193)
  n <- 18L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  X <- cbind(1, x = d$x)
  y <- rnorm(n)
  for (approx in c("nngp", "hsgp")) {
    specification <- validate_svc(spatial_svc(~ lon + lat, terms = ~ x,
      nn = 3L, approx = approx), d, X)
    svc <- .svc_sampler_spec(specification, X)
    for (prior in if (approx == "nngp") c("pc", "uniform") else c("pc", "lognormal")) {
      svc$range_prior <- prior
      svc$range_lower <- .02
      svc$range_upper <- 2
      for (coordinate in c("centered", "noncentered")) {
        svc$parameterization <- coordinate
        r <- tulpa:::cpp_test_field_coordinates(y, X, svc_spec = svc)
        expect_true(r$verified)
        expect_equal(r$gradient, r$numerical, tolerance = 3e-5)
        expect_equal(r$prior_gradient, r$numerical_prior, tolerance = 3e-5)
        if (prior == "uniform") expect_equal(r$stored[4L], r$x[4L])
      }
    }
  }
})

test_that("bilinear factor scores and their constraint match a dense Gaussian prior", {
  set.seed(194)
  n <- 12L
  y <- rnorm(n)
  X <- matrix(1, n, 1L)
  for (constraint in 0:1) {
    specification <- list(n_factors = 1L, constraint = constraint)
    r <- tulpa:::cpp_test_field_coordinates(y, X, latent_spec = specification)
    q <- r$x
    beta <- q[1L]
    loading <- exp(q[2L])
    raw <- q[-(1:2)]
    scores <- raw - if (constraint == 0L) mean(raw) else raw[1L]
    expected <- -.5 * beta^2 / 2.5^2 + q[2L] - loading - .5 * sum(raw^2) -
      .5 * sum((y - beta - loading * scores)^2) - .5 * n * log(2 * pi)
    expect_equal(r$raw_lp, expected, tolerance = 1e-10)
    expect_true(r$verified)
    expect_equal(r$gradient, r$numerical, tolerance = 2e-5)
  }
})
