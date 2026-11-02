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
