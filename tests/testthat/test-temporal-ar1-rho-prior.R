# AR1 rho prior (gcol33/tulpa#209): temporal_ar1(rho_prior=) places a Beta(a, b)
# prior on u = (rho + 1)/2, honoured on both the sampler and nested-Laplace paths.

test_that("temporal_ar1 validates rho_prior is a Beta prior", {
  expect_s3_class(temporal_ar1("t", rho_prior = prior_beta(5, 2)), "tulpa_temporal")
  expect_null(temporal_ar1("t")$rho_prior)
  expect_error(temporal_ar1("t", rho_prior = prior_normal(0, 1)), "Beta prior")
  expect_error(temporal_ar1("t", rho_prior = 0.5), "Beta prior")
})

test_that(".ar1_rho_beta_ab maps the prior to (a, b) shape params", {
  expect_identical(.ar1_rho_beta_ab(NULL), c(1, 1))           # default = Uniform
  expect_identical(.ar1_rho_beta_ab(prior_beta(5, 2)), c(5, 2))
})

test_that("the nested-Laplace AR1 rho axis carries the normalised Beta prior", {
  rho <- c(0.0, 0.5, 0.9)
  dens <- function(p) .hp_axis_default("rho", list(type = "ar1", rho_prior = p))$fn

  # Beta(1, 1) is the uniform on (-1, 1): density 1/2 everywhere, and a NULL
  # prior on the block is the same default.
  expect_equal(dens(prior_beta(1, 1))(rho), rep(log(0.5), 3), tolerance = 1e-14)
  expect_equal(dens(NULL)(rho), rep(log(0.5), 3), tolerance = 1e-14)

  # Beta(a, b) on u = (rho + 1)/2 carried to rho: the Beta log-density with its
  # normalising constant, plus log(1/2).
  u <- 0.5 * (rho + 1)
  expect_equal(dens(prior_beta(10, 1))(rho),
               9 * log(u) - lbeta(10, 1) + log(0.5), tolerance = 1e-12)
})

test_that("informative AR1 rho prior shifts the nested posterior toward the prior", {
  skip_on_cran()
  skip_if_not(nzchar(Sys.getenv("TULPA_SLOW_TESTS")), "slow recovery test")
  set.seed(11)
  Tn <- 60
  rho_true <- 0.3
  z <- numeric(Tn); z[1] <- rnorm(1)
  for (t in 2:Tn) z[t] <- rho_true * z[t - 1] + rnorm(1, 0, 0.4)
  z <- z - mean(z)
  x <- rnorm(Tn)
  y <- rpois(Tn, exp(0.2 + 0.3 * x + z))
  d <- data.frame(y = y, x = x, t = seq_len(Tn))

  # `temporal_ar1()` builds a spec for the `temporal =` argument; it is not an
  # inline formula term, and writing it as one reached model.frame() with a
  # list. The claim being tested is about the prior, not the call form.
  fit_flat <- tulpa(y ~ x, data = d, family = "poisson",
                    temporal = temporal_ar1(~ t))
  # A prior concentrated on strong positive autocorrelation should pull the rho
  # posterior above the flat-prior fit on this short, weakly-identified series.
  fit_hi <- tulpa(y ~ x, data = d, family = "poisson",
                  temporal = temporal_ar1(~ t, rho_prior = prior_beta(12, 2)))
  rho_flat <- temporal_corr(fit_flat)["rho_ar1", "mean"]
  rho_hi   <- temporal_corr(fit_hi)["rho_ar1", "mean"]
  expect_true(is.finite(rho_flat) && is.finite(rho_hi))
  expect_gt(rho_hi, rho_flat)
})
