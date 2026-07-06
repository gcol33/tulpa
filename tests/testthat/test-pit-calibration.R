# Calibration of tulpa_pit(): under a correctly specified predictive CDF the PIT
# values are Uniform(0, 1) (Dawid 1984; Gneiting et al. 2007). test-pit-cpp.R
# proves only the C++-vs-R port equivalence -- it says nothing about whether the
# returned values are calibrated. These tests check uniformity under a correct
# model AND confirm the check has power (a misspecified CDF is rejected), on both
# the continuous (jitter) and the interval-randomized (cdf_lower) branches.

# Kolmogorov-Smirnov distance of a sample from Uniform(0, 1). Deterministic given
# the sample, so it isolates the calibration question from the KS p-value's own
# sampling noise.
.ks_unif_D <- function(u) {
  u <- sort(u)
  n <- length(u)
  ecdf_hi <- seq_len(n) / n
  ecdf_lo <- (seq_len(n) - 1) / n
  max(max(ecdf_hi - u), max(u - ecdf_lo))
}

test_that("continuous PIT is Uniform(0,1) under a correct model, non-uniform under a wrong one", {
  set.seed(101)
  N <- 4000L; S <- 1L
  mu <- rnorm(N, 0, 1); sigma <- 0.7
  y  <- rnorm(N, mu, sigma)

  # Correct model: predictive CDF F(y_i) = Phi((y_i - mu_i) / sigma). One draw
  # row is enough -- tulpa_pit averages over draws, and there is no posterior
  # uncertainty in this idealized correct-model probe.
  cdf_ok <- matrix(pnorm(y, mu, sigma), nrow = S, ncol = N, byrow = TRUE)
  set.seed(1); pit_ok <- tulpa_pit(cdf_ok)
  D_ok <- .ks_unif_D(pit_ok)

  # Misspecified: predictive SD too small by 2x -> PIT piles up near 0 and 1.
  cdf_bad <- matrix(pnorm(y, mu, sigma / 2), nrow = S, ncol = N, byrow = TRUE)
  set.seed(1); pit_bad <- tulpa_pit(cdf_bad)
  D_bad <- .ks_unif_D(pit_bad)

  # Under H0 the KS distance for N = 4000 sits well below the 0.05 critical
  # value (1.358 / sqrt(N) ~= 0.0215); a 2x-too-tight CDF blows far past it.
  expect_lt(D_ok, 0.025)
  expect_gt(D_bad, 0.10)
  expect_lt(abs(mean(pit_ok) - 0.5), 0.02)
})

test_that("interval-randomized PIT (cdf_lower) is Uniform(0,1) under a correct Poisson model", {
  set.seed(202)
  N <- 5000L; S <- 1L
  lambda <- runif(N, 1, 8)
  y <- rpois(N, lambda)

  # Randomized PIT for discrete data (Czado et al. 2009): U in [F(y-1), F(y)].
  Fl <- matrix(ppois(y - 1L, lambda), nrow = S, ncol = N, byrow = TRUE)
  Fu <- matrix(ppois(y,      lambda), nrow = S, ncol = N, byrow = TRUE)

  set.seed(7); pit <- tulpa_pit(Fu, cdf_lower = Fl)
  D <- .ks_unif_D(pit)
  expect_lt(D, 0.025)
  expect_lt(abs(mean(pit) - 0.5), 0.02)

  # A wrong lambda (inflated 1.6x) breaks the discrete calibration.
  Fl_bad <- matrix(ppois(y - 1L, lambda * 1.6), nrow = S, ncol = N, byrow = TRUE)
  Fu_bad <- matrix(ppois(y,      lambda * 1.6), nrow = S, ncol = N, byrow = TRUE)
  set.seed(7); pit_bad <- tulpa_pit(Fu_bad, cdf_lower = Fl_bad)
  expect_gt(.ks_unif_D(pit_bad), 0.10)
})
