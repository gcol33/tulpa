# Print methods of the user-facing classes that carry one (gcol33/tulpa#954).

test_that("tulpa_family prints its name, processes and extra parameters", {
  fam <- tulpa_family(
    name = "gaussian",
    simulate_fn = function(eta, params, n_obs, ...) {
      rnorm(n_obs, eta[[1]], params$sigma_y)
    },
    extra_params = list(sigma_y = prior_half_normal(1))
  )
  out <- capture.output(res <- print(fam))
  expect_identical(res, fam)
  expect_match(out[1], "tulpa family 'gaussian'", fixed = TRUE)
  expect_true(any(grepl("processes: y", out, fixed = TRUE)))
  expect_true(any(grepl("sigma_y ~ Half-Normal(1.00)", out, fixed = TRUE)))
})

test_that("tulpa_hyper_axis_spec prints its span, scale, support and prior", {
  ax <- hyper_axis_spec("alpha", grid = c(0, 0.5, 1), log_scale = TRUE,
                        bounds = c(0, Inf), atom_mass = 0.5)
  out <- capture.output(res <- print(ax))
  expect_identical(res, ax)
  expect_match(out[1], "Hyperparameter axis 'alpha': 3 levels in [0, 1], log scale",
               fixed = TRUE)
  expect_true(any(grepl("point mass at 0: 0.5", out, fixed = TRUE)))
  expect_true(any(grepl("support: (0, Inf)", out, fixed = TRUE)))
  expect_true(any(grepl("prior: flat", out, fixed = TRUE)))

  lin <- hyper_axis_spec("rho", grid = 0.3,
                         log_prior = function(x) stats::dbeta(x, 2, 2, log = TRUE),
                         log_prior_coord = "natural", refinable = TRUE)
  out <- capture.output(print(lin))
  expect_match(out[1], "1 level in [0.3, 0.3], linear scale", fixed = TRUE)
  expect_true(any(grepl("density on the natural coordinate", out, fixed = TRUE)))
  expect_true(any(grepl("refinable", out, fixed = TRUE)))
})

test_that("a temporal AR(p) block prints its order, then the tgmrf body", {
  blk <- temporal_ar(rep(seq_len(12L), 2L), p = 3L)
  out <- capture.output(res <- print(blk))
  expect_identical(res, blk)
  expect_match(out[1], "Temporal AR(3) latent block over 12 time points",
               fixed = TRUE)
  expect_true(any(grepl("<tgmrf>", out, fixed = TRUE)))
})

test_that("tulpa_variogram prints a header and its bins", {
  set.seed(1)
  co <- cbind(runif(30), runif(30))
  vg <- tulpa_variogram(rnorm(30), co, n_bins = 5L)
  out <- capture.output(res <- print(vg))
  expect_identical(res, vg)
  expect_match(out[1], sprintf("Empirical semivariogram: %d distance bins",
                               nrow(vg)), fixed = TRUE)
  expect_true(any(grepl("n_pairs", out, fixed = TRUE)))
})

test_that("the posterior summary tables print a header with the draw count", {
  tab <- data.frame(time_idx = 1:2, time = 1:2, mean = c(0.1, -0.1),
                    sd = c(1, 1), q2.5 = c(-2, -2), q97.5 = c(2, 2))
  cases <- list(
    tulpa_svc_summary = "Spatially varying coefficient posterior summary",
    tulpa_tvc_summary = "Temporally varying coefficient posterior summary",
    tulpa_temporal_summary = "Temporal effect posterior summary"
  )
  for (cls in names(cases)) {
    x <- structure(tab, n_draws = 200L, class = c(cls, "data.frame"))
    out <- capture.output(res <- print(x))
    expect_identical(res, x)
    expect_identical(out[1], paste0(cases[[cls]], ", 200 posterior draws"))
    expect_true(any(grepl("q97.5", out, fixed = TRUE)))
  }

  st <- structure(
    data.frame(s = c(1, 2), t = c(1, 1), mean = c(0, 1), q50 = c(0, 1),
               sd = c(1, 1)),
    n_spatial = 2L, n_times = 1L, n_draws = 50L,
    class = c("tulpa_st_summary", "data.frame"))
  out <- capture.output(print(st))
  expect_identical(out[1], paste0(
    "Spatiotemporal interaction posterior summary: 2 locations x 1 times, ",
    "50 posterior draws"))
})

test_that("each nested-Laplace class prints its own header", {
  testthat::local_mocked_bindings(
    .print_nested_laplace_body = function(x) invisible(x),
    .print_fit_body = function(x) invisible(x),
    .package = "tulpa")
  headers <- list(
    "tulpa nested-Laplace fit" = c("tulpa_nested_laplace", "list"),
    "tulpa joint nested-Laplace fit" =
      c("tulpa_nested_laplace_joint", "tulpa_nested_laplace", "list"),
    "tulpa joint nested-Laplace fit (multi-block)" =
      c("tulpa_nested_laplace_joint_multi", "tulpa_nested_laplace_joint",
        "tulpa_nested_laplace", "list"),
    "tulpa space-time nested-Laplace fit" =
      c("tulpa_st_nested", "tulpa_nested_laplace", "list")
  )
  for (h in names(headers)) {
    x <- structure(list(), class = c(headers[[h]], "tulpa_fit"))
    out <- capture.output(res <- print(x))
    expect_identical(res, x)
    expect_identical(trimws(out[1]), h)
  }
})

test_that("an EP fit is classed tulpa_fit alone", {
  skip_on_cran()
  set.seed(2)
  d <- data.frame(x = rnorm(40))
  d$y <- rbinom(40, 1, plogis(0.3 * d$x))
  fit <- tulpa_ep(y ~ x, data = d, family = "binomial")
  expect_identical(class(fit), "tulpa_fit")
})
