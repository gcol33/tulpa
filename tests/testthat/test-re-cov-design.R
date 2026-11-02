# The default outer design of the random-effect covariance integrator follows
# the number of covariance coordinates (`.re_cov_outer_design()`): a tensor
# grid up to two, the CCD above. A scale's posterior on its log coordinate
# reaches several SDs of the mode's Gaussian into its left tail, where a 3- or
# 9-node CCD places no node.

test_that(".re_cov_outer_design picks the layout by coordinate count", {
  d1 <- tulpa:::.re_cov_outer_design(1L)
  d2 <- tulpa:::.re_cov_outer_design(2L)
  d3 <- tulpa:::.re_cov_outer_design(3L)
  expect_identical(c(d1$integration, d2$integration, d3$integration),
                   c("grid", "grid", "ccd"))
  expect_identical(c(d1$n_per_axis, d2$n_per_axis),
                   tulpa:::.RE_COV_DESIGN$tensor_n)
  expect_identical(tulpa:::.re_cov_outer_design(3L, list(integration = "grid"))$n_per_axis,
                   tulpa:::.RE_COV_DESIGN$grid_n)
  expect_identical(tulpa:::.re_cov_outer_design(1L, list(n_per_axis = 7L))$n_per_axis, 7L)
  expect_error(tulpa:::.re_cov_outer_design(1L, list(integration = "tensor")))
})

skip_on_cran()

test_that("a (1 | g) scale's default interval matches a dense reference", {
  set.seed(20261102)
  G <- 40L; npg <- 8L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  u <- rnorm(G, sd = 0.8)
  y <- rbinom(N, 1, plogis(-0.4 + 0.9 * x + u[g]))
  d <- data.frame(y = y, x = x, g = factor(g))
  fit <- function(ctl) tulpa(y ~ x + (1 | g), data = d, family = "binomial",
                             mode = "structured",
                             control = c(list(re_cov = "nested",
                                              diagnose_k = FALSE), ctl))
  def <- summary(fit(list()))$hyperparameters
  ref <- summary(fit(list(integration = "grid", n_per_axis = 201L,
                          span = 8)))$hyperparameters
  # The 3-node CCD read this lower bound as 0.495 against the reference 0.358.
  expect_lt(abs(def$ci_lo[1] - ref$ci_lo[1]), 0.005)
  expect_lt(abs(def$ci_hi[1] - ref$ci_hi[1]), 0.02)
  expect_lt(abs(def$median[1] - ref$median[1]), 0.002)
  expect_lt(abs(def$sd[1] / ref$sd[1] - 1), 0.01)
})
