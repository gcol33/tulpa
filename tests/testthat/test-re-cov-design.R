# The default outer design of the random-effect covariance integrator follows
# the number of covariance coordinates (`.nl_outer_design()`): a tensor
# grid up to three, the CCD above. A scale's posterior on its log coordinate
# reaches several SDs of the mode's Gaussian into its left tail, where a 3-, 9-
# or 15-node CCD places no node.

test_that(".nl_outer_design picks the layout by coordinate count", {
  d1 <- tulpa:::.nl_outer_design(1L)
  d2 <- tulpa:::.nl_outer_design(2L)
  d3 <- tulpa:::.nl_outer_design(3L)
  d4 <- tulpa:::.nl_outer_design(4L)
  expect_identical(c(d1$integration, d2$integration, d3$integration,
                     d4$integration),
                   c("grid", "grid", "grid", "ccd"))
  expect_identical(c(d1$n_per_axis, d2$n_per_axis, d3$n_per_axis),
                   tulpa:::.NL_OUTER_DESIGN$tensor_n)
  expect_identical(tulpa:::.nl_outer_design(4L, list(integration = "grid"))$n_per_axis,
                   tulpa:::.NL_OUTER_DESIGN$grid_n)
  expect_identical(tulpa:::.nl_outer_design(3L, list(integration = "ccd"))$integration,
                   "ccd")
  expect_identical(tulpa:::.nl_outer_design(1L, list(n_per_axis = 7L))$n_per_axis, 7L)
  expect_error(tulpa:::.nl_outer_design(1L, list(integration = "tensor")))
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

test_that("a correlated random slope's default interval matches a 15^3 tensor", {
  skip_if_not_slow()
  # (1 + x | g), k = 3 log-Cholesky coordinates, 30 binary groups x 10.
  set.seed(102)
  G <- 30L; npg <- 10L; N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  Sig <- matrix(c(0.8^2, 0.5 * 0.8 * 0.6, 0.5 * 0.8 * 0.6, 0.6^2), 2)
  u <- t(t(chol(Sig)) %*% matrix(rnorm(2 * G), nrow = 2))
  y <- rbinom(N, 1L, plogis(-0.3 + 0.7 * x + u[g, 1] + x * u[g, 2]))
  X <- cbind(1, x)
  re_term <- list(idx = g, n_groups = G, n_coefs = 2L, Z = cbind(1, x))
  fit <- function(ctl) tulpa_re_cov_nested(y, rep(1L, N), X, re_term,
                                           family = "binomial",
                                           control = c(list(diagnose_k = FALSE),
                                                       ctl))$posterior
  ref <- fit(list(integration = "grid", n_per_axis = 15L, span = 4.5))
  def <- fit(list())
  ccd <- fit(list(integration = "ccd"))
  par <- c("sigma_1", "sigma_2", "rho_12")
  r <- ref[match(par, ref$parameter), ]
  d <- def[match(par, def$parameter), ]
  cc <- ccd[match(par, ccd$parameter), ]
  # Reference upper bounds (1.270, 1.240, 0.803); the CCD's (1.244, 1.429,
  # 0.733), the 9^3 default's (1.366, 1.200, 0.824).
  expect_true(all(abs(d$ci_hi / r$ci_hi - 1) < 0.1))
  expect_true(all(abs(d$ci_lo - r$ci_lo) < 0.06))
  expect_gt(max(abs(cc$ci_hi / r$ci_hi - 1)), 0.1)
})
