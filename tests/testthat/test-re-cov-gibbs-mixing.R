# The fixed-effect update of the Metropolis-within-Gibbs sweep targets
# beta | b, Sigma, y, so its random-walk proposal takes the shape of that
# conditional. A proposal shaped by the marginal (which adds the random effects'
# spread along the directions they alias) tunes one scale for the block to the
# widest of those directions and leaves the others' steps short.

.rcg_ess <- function(x) {
  x <- as.numeric(x) - mean(x); n <- length(x)
  ac <- stats::acf(x, lag.max = min(n - 1, 1000), plot = FALSE)$acf[-1]
  k <- which(ac[-1] + ac[-length(ac)] < 0)[1]
  if (is.na(k)) k <- length(ac)
  n / (1 + 2 * sum(ac[seq_len(k)]))
}

skip_on_cran()

test_that("the fixed-effect proposal reads the conditional precision", {
  set.seed(1)
  G <- 60L; npg <- 12L; N <- G * npg
  grp <- rep(seq_len(G), each = npg); x <- rnorm(N)
  y <- rpois(N, exp(0.2 + 0.5 * x + rnorm(G, 0, 0.7)[grp]))
  d <- data.frame(y = y, x = x, g = factor(grp))
  f <- suppressWarnings(tulpa(y ~ x + (1 | g), data = d, family = "poisson",
         mode = "laplace", control = list(re_cov = "gibbs", n_iter = 4000L,
                                          warmup = 1000L, seed = 1L)))
  # 4000 kept sweeps: the slope's effective size is 289 at this seed under the
  # conditional shape and 119 under a marginal-shaped proposal.
  expect_gt(.rcg_ess(f$draws[, "x"]), 200)
  expect_gt(.rcg_ess(f$draws[, "(Intercept)"]), 1000)
})

test_that("the Newton-step move mixes an auto-routed (1 | g) binomial at the default length", {
  set.seed(2)
  n <- 300; x <- rnorm(n)
  d1 <- data.frame(y = rbinom(n, 1, plogis(-0.3 + x)), x = x,
                   g = factor(sample(1:3, n, TRUE)))
  expect_no_warning(fa <- tulpa(y ~ x + (1 | g), d1, family = "binomial"))
  expect_identical(fa$backend, "re_cov_gibbs")
  # 2000 kept sweeps. With the random walk alone the slope's effective size sat
  # at 144-308 over 40 simulated data sets; with the Newton-step move, 754-1504.
  expect_gt(.rcg_ess(fa$draws[, "x"]), 600)
  expect_gt(fa$accept$beta_newton, 0.5)
})
