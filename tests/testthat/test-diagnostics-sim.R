# Structural coverage for the residual diagnostics in R/diagnostics_sim.R.
# Every block is tier 1: the inputs are plain residual vectors, coordinate
# matrices and simulation matrices, so nothing here fits a model.

test_that("moran_i() and tulpa_variogram() reject coords that do not match the residuals", {
  set.seed(1)
  r  <- rnorm(60)
  co <- cbind(runif(50), runif(50))

  expect_error(moran_i(r, co), "coords has 50 rows but residuals has length 60")
  expect_error(tulpa_variogram(r, co),
               "coords has 50 rows but residuals has length 60")

  # The other direction is just as silent without the check.
  expect_error(moran_i(rnorm(40), co),
               "coords has 50 rows but residuals has length 40")
  expect_error(tulpa_variogram(rnorm(40), co),
               "coords has 50 rows but residuals has length 40")
})

test_that("both spatial diagnostics refuse a degenerate input", {
  expect_error(tulpa_variogram(rnorm(1), cbind(0, 0)), "at least 2 observations")
  expect_error(moran_i(rnorm(1), cbind(0, 0)), "at least 2 observations")

  # Coincident locations leave no distance axis to bin over.
  co <- cbind(rep(0, 10), rep(0, 10))
  expect_error(tulpa_variogram(rnorm(10), co), "coincident")
})

test_that("tulpa_variogram() bins matched residuals and coordinates", {
  set.seed(2)
  n  <- 60L
  co <- cbind(runif(n), runif(n))
  vg <- tulpa_variogram(rnorm(n), co, n_bins = 6L)

  expect_s3_class(vg, "tulpa_variogram")
  expect_identical(names(vg), c("dist", "gamma", "n_pairs"))
  expect_true(all(vg$n_pairs > 0L))
  expect_true(all(is.finite(vg$gamma)))
  # Every pair counted at most once: the bins partition [0, max_dist).
  expect_lte(sum(vg$n_pairs), choose(n, 2))

  # The bins hold the pairs they claim: recompute the first by hand.
  d      <- as.numeric(stats::dist(co))
  breaks <- seq(0, max(d) / 2, length.out = 7L)
  expect_identical(vg$n_pairs[1L], sum(d >= breaks[1L] & d < breaks[2L]))
})

test_that("moran_i() returns an htest under both weight schemes", {
  set.seed(3)
  n  <- 40L
  co <- cbind(runif(n), runif(n))
  r  <- rnorm(n)

  for (w in c("inverse", "knn")) {
    h <- moran_i(r, co, weights = w)
    expect_s3_class(h, "htest")
    expect_true(is.finite(h$statistic[["Moran's I"]]))
    expect_true(h$p.value >= 0 && h$p.value <= 1)
  }
})

test_that("a zero-variance input gives a NaN statistic rather than an error", {
  set.seed(4)
  co <- cbind(runif(20), runif(20))
  expect_true(is.nan(moran_i(rep(1, 20), co)$statistic[["Moran's I"]]))
  expect_true(is.nan(durbin_watson(rep(1, 10))$statistic[["DW"]]))
})

test_that("pit_residuals() reads an n_obs x nsim simulation matrix", {
  set.seed(5)
  n    <- 50L
  nsim <- 200L
  y    <- rpois(n, 4)
  sims <- matrix(rpois(n * nsim, 4), nrow = n, ncol = nsim)

  pit <- pit_residuals(sims, observed = y)
  expect_length(pit, n)
  expect_true(all(pit >= 0 & pit <= 1))

  expect_error(pit_residuals(sims), "observed required")

  # test_uniformity() takes the PIT vector directly.
  h <- test_uniformity(pit)
  expect_s3_class(h, "htest")
  expect_true(h$p.value >= 0 && h$p.value <= 1)
})
