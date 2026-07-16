# plot_map() / plot_map_panel() on a generic spatial fit. Guards against the
# ratio-era regression where plot_map_panel called plot_map(what = "ratio")
# (rejected by match.arg) and the "uncertainty" branch called the undefined
# ratio() accessor (gcol33/tulpa#152).

test_that("plot_map and plot_map_panel build maps from a spatial fit", {
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
  expect_error(plot_map(fit), "no point coordinates")
})
