# Known-answer coverage for the generic DHARMa/spdep-equivalent diagnostics
# (#224). These exported statistics previously shipped with no test at all; a
# sign or coefficient slip in the hand-transcribed Moran randomization variance
# or the Durbin-Watson statistic would have returned wrong p-values with every
# plumbing test still green. The closed-form checks below need no reference
# package (they hand-compute the statistic); the reference cross-checks run only
# where spdep / lmtest are installed.

test_that("durbin_watson matches its closed form on a hand-computed vector", {
  x <- c(1, 2, 1, 2, 1)                    # strongly alternating -> DW near 4
  e <- x - mean(x)                         # centred residuals
  dw_expected <- sum(diff(e)^2) / sum(e^2) # = 4 / 1.2 = 10/3
  r1_expected <- 1 - dw_expected / 2       # = -2/3

  res <- durbin_watson(x)
  expect_equal(as.numeric(res$statistic), 10 / 3, tolerance = 1e-12)
  expect_equal(as.numeric(res$parameter), -2 / 3, tolerance = 1e-12)

  # Alternating residuals are negative autocorrelation: the one-sided "greater"
  # (positive autocorrelation) alternative should return a large p-value.
  expect_gt(durbin_watson(x, alternative = "greater")$p.value, 0.9)
})

test_that("moran_i matches its closed form on a 3-point line", {
  # coords 0,1,2 on a line; residuals (1, 0, -1) (already mean-zero). With
  # inverse-distance weights W = [[0,1,.5],[1,0,1],[.5,1,0]] this hand-computes
  # to I = (N/S0) * (dx' W dx / ss) = (3/5) * (-1 / 2) = -0.3, and the
  # randomization expectation EI = -1/(N-1) = -0.5.
  x <- c(1, 0, -1)
  coords <- matrix(c(0, 1, 2), ncol = 1L)
  res <- moran_i(x, coords, weights = "inverse")
  expect_equal(as.numeric(res$statistic), -0.3, tolerance = 1e-12)
  expect_equal(as.numeric(res$parameter), -0.5, tolerance = 1e-12)
})

test_that("moran_i sign tracks the spatial pattern", {
  set.seed(3)
  coords <- as.matrix(expand.grid(x = 1:6, y = 1:6))
  # Smooth gradient -> strong positive autocorrelation.
  grad <- coords[, 1] + coords[, 2]
  expect_gt(as.numeric(moran_i(grad, coords, weights = "inverse")$statistic), 0.2)
  # Checkerboard -> negative autocorrelation under nearest-neighbour weights.
  checker <- ifelse((coords[, 1] + coords[, 2]) %% 2 == 0, 1, -1)
  expect_lt(as.numeric(moran_i(checker, coords, weights = "knn", k = 4L)$statistic), 0)
})

test_that("moran_i agrees with spdep::moran where available", {
  skip_if_not_installed("spdep")
  set.seed(5)
  coords <- as.matrix(expand.grid(x = 1:6, y = 1:6))
  x <- rnorm(nrow(coords)) + 0.3 * (coords[, 1] + coords[, 2])

  D <- as.matrix(dist(coords)); Wm <- 1 / D; diag(Wm) <- 0; Wm[!is.finite(Wm)] <- 0
  # Keep the raw inverse-distance weights unstandardized, matching moran_i
  # (S0 = sum(W)); a row-standardized listw would change the statistic.
  lw <- suppressWarnings(spdep::mat2listw(Wm, style = "M"))
  ref <- spdep::moran(x, lw, n = length(x), S0 = spdep::Szero(lw))

  got <- moran_i(x, coords, weights = "inverse")
  expect_equal(as.numeric(got$statistic), ref$I, tolerance = 1e-8)
})

test_that("durbin_watson statistic agrees with lmtest::dwtest where available", {
  skip_if_not_installed("lmtest")
  set.seed(9)
  n <- 40L
  e <- as.numeric(arima.sim(list(ar = 0.6), n = n))
  ref <- lmtest::dwtest(e ~ 1)             # intercept-only: residuals are e - mean
  got <- durbin_watson(e)
  expect_equal(as.numeric(got$statistic), as.numeric(ref$statistic), tolerance = 1e-8)
})
