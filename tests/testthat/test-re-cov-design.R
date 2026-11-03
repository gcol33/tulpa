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

test_that("a tensor grows a level past an end that still carries mass", {
  h <- 1
  ax <- seq(-2, 2, by = h)
  z <- as.matrix(expand.grid(ax, ax))
  dimnames(z) <- NULL
  # Axis 1 has a flat lower half, axis 2 is a Gaussian of SD 0.5: only axis
  # 1's lower end grows, by one level across all five levels of axis 2.
  lw <- ifelse(z[, 1] < 0, 0, -2 * z[, 1]^2) - 2 * z[, 2]^2
  new <- tulpa:::.nl_tensor_edge_cells(z, lw, h)
  expect_equal(nrow(new$z), length(ax))
  expect_true(all(new$z[, 1] == -3))
  expect_setequal(new$z[, 2], ax)
  # Laid back onto the tensor, it is still a full tensor of uniform spacing.
  zz <- rbind(z, new$z)
  expect_equal(nrow(unique(zz)), 6L * 5L)
  # Only the rows holding all but 1e-3 of the edge level's mass are solved:
  # the two outer rows of axis 2 carry 2.6e-4 of it each.
  expect_setequal(new$z[new$solve, 2], c(-1, 0, 1))
  # A tensor whose ends hold no mass is left as it is.
  ax <- seq(-6, 6, by = h)
  z <- as.matrix(expand.grid(ax, ax))
  expect_null(tulpa:::.nl_tensor_edge_cells(z, -0.5 * rowSums(z^2), h))
})

skip_on_cran()

test_that("a slope SD's left tail past the 4.5-SD span is integrated", {
  # (1 + x | g), 50 binary groups x 6. The log slope SD's posterior has a left
  # tail heavier than the Gaussian at the mode, and the 9^3 tensor laid at
  # +/- 4.5 SDs cut it: the 2.5% bound read 0.0196 against 0.0116 from a dense
  # (log sigma_1, log sigma_2, atanh rho) tensor (0.05 / 0.15 / 0.15 apart,
  # about a million cells trimmed to all but 1e-9 of each marginal).
  set.seed(4)
  gg <- rep(1:50, each = 6); xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1); b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  rt <- list(idx = gg, n_groups = 50L, n_coefs = 2L, Z = cbind(1, xx),
             correlated = TRUE)
  fit <- suppressWarnings(tulpa_re_cov_nested(
    yy, NULL, cbind(1, xx), rt, family = "binomial",
    control = list(n_threads = 2L, diagnose_k = FALSE)))
  expect_gt(fit$tensor_extended, 0L)
  expect_gt(fit$n_grid, 9L^3L)
  p <- fit$posterior
  lo <- p$ci_lo[p$parameter == "sigma_2"]
  expect_lt(abs(log(lo / 0.0116)), 0.25)
  hi <- p$ci_hi[p$parameter == "sigma_2"]
  expect_lt(abs(log(hi / 0.880)), 0.1)
})

test_that("a within-cell coupling whose zeroed precision is near singular is read", {
  # On this grown tensor the within-cell copula zeroes a pair whose conditional
  # sign opposes the posterior's, and the precision that leaves, clipped at the
  # eigenvalue floor, is too ill-conditioned for a Cholesky factorization.
  set.seed(29)
  gg <- rep(1:50, each = 6); xx <- rnorm(300)
  b0 <- rnorm(50, 0, 1); b1 <- rnorm(50, 0, 0.7)
  yy <- rbinom(300, 1, plogis(-0.3 + 0.8 * xx + b0[gg] + b1[gg] * xx))
  rt <- list(idx = gg, n_groups = 50L, n_coefs = 2L, Z = cbind(1, xx),
             correlated = TRUE)
  fit <- suppressWarnings(tulpa_re_cov_nested(
    yy, NULL, cbind(1, xx), rt, family = "binomial",
    control = list(n_threads = 2L, diagnose_k = FALSE)))
  p <- fit$posterior
  expect_true(all(is.finite(c(p$ci_lo, p$median, p$ci_hi))))
  expect_true(all(p$ci_lo <= p$median & p$median <= p$ci_hi))
})

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
