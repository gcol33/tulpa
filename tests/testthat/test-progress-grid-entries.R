# The outer-grid progress reporter reaches every grid door. The SPDE door and
# the single-block registry entries hand their grid to a driver through the
# shared entry bundle (src/nl_entry_inputs.h), so `control$progress*` is read
# there or not at all; the heartbeat file is the observable, written by the
# reporter's final emit whether or not the console line is on.

test_that("the SPDE door takes control$progress and writes the heartbeat file", {
  skip_on_cran()
  set.seed(1); n <- 60
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- rpois(n, exp(0.2 + 0.4 * d$x))
  sp <- spatial_spde(~ lon + lat, data = d, max_edge = c(0.3, 0.6),
                     cutoff = 0.08)
  hb <- tempfile(fileext = ".txt")
  on.exit(unlink(hb), add = TRUE)
  expect_no_error(
    tulpa(y ~ x, data = d, family = "poisson", spatial = sp,
          control = list(n_threads = 1, progress = FALSE, progress.file = hb,
                         method = "ccd", diagnose_k = FALSE)))
  expect_true(file.exists(hb))
  expect_gt(length(readLines(hb, warn = FALSE)), 0L)
})

test_that("a single-block registry fit writes the heartbeat file", {
  skip_on_cran()
  W <- matrix(0, 6, 6)
  for (i in 1:5) W[i, i + 1] <- W[i + 1, i] <- 1
  set.seed(2)
  d <- data.frame(region = rep(1:6, each = 5), x = rnorm(30))
  d$y <- rpois(30, exp(0.3 + 0.4 * d$x + sin(d$region)))
  hb <- tempfile(fileext = ".txt")
  on.exit(unlink(hb), add = TRUE)
  tulpa(y ~ x + spatial(region), data = d, family = "poisson",
        spatial = list(type = "icar", adjacency = W), mode = "nested_laplace",
        control = list(progress = FALSE, progress.file = hb,
                       diagnose_k = FALSE))
  expect_true(file.exists(hb))
})
