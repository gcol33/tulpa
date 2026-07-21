# Random-effect covariance summary.
#
# The property under test is not just "returns numbers" but that it reports
# WHERE the numbers came from. A Laplace fit conditions on sigma_re -- often the
# default of 1, which the user never chose -- and presenting that beside an
# empirical-Bayes estimate as though they were the same kind of quantity is the
# failure mode this accessor exists to avoid.

vc_data <- function(seed = 1L, G = 30L, per = 12L, sigma = 0.8) {
  set.seed(seed)
  n <- G * per
  g <- rep(seq_len(G), each = per)
  x <- rnorm(n)
  b <- rnorm(G, 0, sigma)
  data.frame(y = rpois(n, exp(0.3 + 0.5 * x + b[g])), x = x, g = factor(g))
}


test_that("an empirical-Bayes fit reports its estimate as estimated", {
  skip_on_cran()
  d <- vc_data()
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
  vc <- VarCorr(fit)

  expect_s3_class(vc, "data.frame")
  expect_equal(nrow(vc), 1L)
  expect_equal(vc$term, "g")
  expect_equal(vc$coef, "(Intercept)")
  expect_equal(vc$source, "estimated")
  # The reported SD is the fit's own estimate, not a re-derivation of it.
  expect_equal(vc$sd, unname(fit$map$sigma), tolerance = 1e-10)
  # And it is in the right neighbourhood of truth, so the wiring is not simply
  # self-consistent about a wrong number.
  expect_lt(abs(vc$sd - 0.8), 0.25)
})


test_that("a conditioning fit is labelled conditioned, not estimated", {
  d <- vc_data()
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
               mode = "laplace", sigma_re = 0.5)
  vc <- VarCorr(fit)

  expect_equal(vc$source, "conditioned")
  # The value is the input echoed back. That is exactly why it carries a label:
  # without one it is indistinguishable from a fitted 0.5.
  expect_equal(vc$sd, 0.5, tolerance = 1e-8)
})


test_that("a sampler fit reports a posterior-mean SD", {
  skip_on_cran()
  d <- vc_data()
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "hmc",
               control = list(n_iter = 1200L, warmup = 600L, n_chains = 2L,
                              seed = 4L))
  vc <- VarCorr(fit)

  expect_equal(vc$source, "sampled")
  expect_lt(abs(vc$sd - 0.8), 0.3)

  # Averaged on the SD scale, not the log scale: E[exp(z)] != exp(E[z]), and
  # the log-scale answer is the smaller one for any non-degenerate posterior.
  dm <- tulpa:::.re_draws_mat(fit)
  ls <- dm[, grep("^log_sigma_re", colnames(dm)), drop = FALSE]
  expect_equal(vc$sd, unname(mean(exp(ls[, 1]))), tolerance = 1e-10)
  expect_gt(mean(exp(ls[, 1])), exp(mean(ls[, 1])))
})


test_that("several terms each get a row, in layout order", {
  skip_on_cran()
  set.seed(3)
  G <- 25L; H <- 12L; per <- 14L; n <- G * per
  g <- rep(seq_len(G), each = per)
  h <- sample(seq_len(H), n, replace = TRUE)
  x <- rnorm(n)
  y <- rpois(n, exp(0.3 + 0.4 * x + rnorm(G, 0, 0.9)[g] + rnorm(H, 0, 0.4)[h]))
  d <- data.frame(y = y, x = x, g = factor(g), h = factor(h))

  fit <- tulpa(y ~ x + (1 | g) + (1 | h), data = d, family = "poisson",
               mode = "eb")
  vc <- VarCorr(fit)

  expect_equal(nrow(vc), 2L)
  expect_equal(vc$term, c("g", "h"))
  expect_true(all(vc$sd > 0))
  # The larger simulated component must come back larger; a mis-ordered or
  # recycled assignment would not preserve that.
  expect_gt(vc$sd[1], vc$sd[2])
})


test_that("the covariance matrices ride along with a correlation attribute", {
  skip_on_cran()
  d <- vc_data()
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb")
  covs <- attr(VarCorr(fit), "cov")

  expect_type(covs, "list")
  expect_named(covs, "g")
  expect_equal(dim(covs$g), c(1L, 1L))
  # A single-coefficient block correlates with itself and nothing else.
  expect_equal(as.numeric(attr(covs$g, "correlation")), 1, tolerance = 1e-10)
  expect_equal(sqrt(as.numeric(covs$g)), VarCorr(fit)$sd, tolerance = 1e-10)
})


test_that("a fit with no random effects returns an empty table", {
  set.seed(2)
  d <- data.frame(y = rpois(80, 2), x = rnorm(80))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  vc <- VarCorr(fit)

  expect_s3_class(vc, "data.frame")
  expect_equal(nrow(vc), 0L)
  # Empty but well-formed, so downstream code can bind it without special-casing.
  expect_named(vc, c("term", "coef", "sd", "source"))
})
