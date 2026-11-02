# plot_map() / plot_map_panel() on a generic spatial fit. Guards against the
# ratio-era regression where plot_map_panel called plot_map(what = "ratio")
# (rejected by match.arg) and the "uncertainty" branch called the undefined
# ratio() accessor (gcol33/tulpa#152).

skip_on_cran()

test_that("plot_map and plot_map_panel build maps from a spatial fit", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")

  set.seed(123)
  n_sites <- 20
  df <- data.frame(
    y = rbinom(n_sites, 20, 0.4),
    elevation = rnorm(n_sites),
    site = factor(seq_len(n_sites)),
    lon = runif(n_sites),
    lat = runif(n_sites)
  )
  adj <- matrix(0, n_sites, n_sites)
  for (i in 1:(n_sites - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
  fit <- tulpa(
    y ~ elevation + spatial(site), data = df, family = "binomial",
    n_trials = rep(20L, n_sites),
    spatial = spatial_car(adj, group_var = "site"), mode = "laplace"
  )
  cc <- df[, c("lon", "lat")]

  p_fit <- plot_map(fit, coords = cc)
  expect_s3_class(p_fit, "ggplot")

  p_unc <- plot_map(fit, what = "uncertainty", coords = cc)
  expect_s3_class(p_unc, "ggplot")

  panel <- plot_map_panel(fit, coords = cc)
  expect_true(inherits(panel, "ggplot") || is.list(panel))

  # A fit without stored coordinates and no `coords` errors clearly rather than
  # failing deep inside a data.frame() row-mismatch.
  expect_error(plot_map(fit), "records no coordinate columns")
})

test_that("plot_map reads the fit's own coordinate columns, not a column named x", {
  skip_on_cran()
  skip_if_not_installed("ggplot2")
  set.seed(5)
  n <- 120L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.4 * d$x + sin(3 * d$lon)))
  fit <- tulpa(y ~ x, data = d, family = "poisson",
               spatial = spatial_gp(~ lon + lat, approx = "hsgp", m = 6))
  g <- expand.grid(lon = seq(0.1, 0.9, length.out = 5),
                   lat = seq(0.1, 0.9, length.out = 4))
  nd <- data.frame(g, x = 0)

  pd <- tulpa:::prepare_map_data_from_fit(fit, "fitted", "mean", nd)
  expect_equal(pd$x, nd$lon)
  expect_equal(pd$y, nd$lat)
  expect_equal(pd$value, predict(fit, newdata = nd, type = "response"))

  # At the training rows the map sits at the coordinates the data gave, the
  # field's standardization undone.
  pt <- tulpa:::prepare_map_data_from_fit(fit, "fitted", "mean", NULL)
  expect_equal(pt$x, d$lon, tolerance = 1e-10)
  expect_equal(pt$y, d$lat, tolerance = 1e-10)

  p <- plot_map(fit, newdata = nd)
  expect_s3_class(p, "ggplot")
  legend <- vapply(p$scales$scales, function(sc) as.character(sc$name)[1L],
                   character(1))
  expect_true("Mean" %in% legend)
  expect_error(plot_map(fit, newdata = nd, summary = "median"),
               "no posterior median")
  expect_error(plot_map(fit, newdata = nd[, c("lon", "x")]), "lat")
})
