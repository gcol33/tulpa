# SoftAbs divergence-retry invariance (gcol33/tulpa#189)
#
# The post-warmup SoftAbs retry (src/hmc_nuts_chain_iter_nuts.h) is a
# state-dependent kernel mixture: on a divergent NUTS trajectory it re-runs a
# fresh trajectory under a frozen Hessian-based metric and takes that proposal
# instead. Selecting the transition kernel conditional on the first kernel's
# divergence can, in principle, fail to leave the target invariant. These tests
# exercise the exact production run_hmc_chain_cpp retry path on Neal's funnel --
# the canonical divergence-generating target with a known marginal
# v ~ N(0, gamma^2) -- and check that toggling the retry on does not shift the
# posterior (the "equal calibrated posteriors" claim in the issue).
#
# cpp_test_funnel_nuts(riemannian = 1) forces the retry on; riemannian = 0 is
# the plain NUTS reference. Both target the identical funnel log-density; the
# funnel is encoded as an extra-parameter-only model (one process, zero
# fixed-effect columns) so the engine adds no beta prior.

test_that("funnel entry point returns a well-formed funnel fit", {
  skip_on_cran()  # tier 2: a single short fit, plumbing + shape

  fit <- tulpa:::cpp_test_funnel_nuts(
    K = 6L, gamma = 3.0,
    n_iter = 400L, n_warmup = 200L,
    max_treedepth = 8L, adapt_delta = 0.8,
    seed = 1L, riemannian = 0L
  )

  expect_equal(fit$n_params, 7L)                 # v + x_1..x_6
  expect_equal(dim(fit$draws), c(200L, 7L))
  expect_equal(colnames(fit$draws),
               c("v", paste0("x[", 1:6, "]")))
  expect_true(all(is.finite(fit$draws)))
  expect_length(fit$divergent, 200L)
  expect_equal(fit$n_divergent, sum(fit$divergent))
  expect_equal(fit$riemannian, 0L)

  # The retry toggle is honored (both settings run and return the same shape).
  fit_on <- tulpa:::cpp_test_funnel_nuts(
    K = 6L, gamma = 3.0,
    n_iter = 400L, n_warmup = 200L,
    max_treedepth = 8L, adapt_delta = 0.8,
    seed = 1L, riemannian = 1L
  )
  expect_equal(fit_on$riemannian, 1L)
  expect_equal(dim(fit_on$draws), c(200L, 7L))
  expect_true(all(is.finite(fit_on$draws)))
})

test_that("SoftAbs retry preserves the funnel posterior (retry on == off)", {
  skip_if_not_slow()  # tier 3: multi-seed MCMC invariance / recovery

  # Neal's funnel: v ~ N(0, 3),  x_i | v ~ N(0, exp(v/2)).  gamma = 3, K = 9 is
  # the classic divergence-generating setting for diagonal-metric NUTS.
  seeds <- 1:10
  gamma <- 3.0
  K <- 9L

  fit_seed <- function(s, riemannian) {
    tulpa:::cpp_test_funnel_nuts(
      K = K, gamma = gamma,
      n_iter = 4000L, n_warmup = 1200L,
      max_treedepth = 10L, adapt_delta = 0.8,
      seed = s, riemannian = riemannian
    )
  }

  rows <- lapply(seeds, function(s) {
    off <- fit_seed(s, 0L)
    on  <- fit_seed(s, 1L)
    data.frame(
      div_off  = off$n_divergent,          div_on  = on$n_divergent,
      mean_off = mean(off$draws[, "v"]),   mean_on = mean(on$draws[, "v"]),
      sd_off   = sd(off$draws[, "v"]),     sd_on   = sd(on$draws[, "v"]),
      q10_off  = unname(quantile(off$draws[, "v"], 0.10)),
      q10_on   = unname(quantile(on$draws[, "v"], 0.10)),
      q90_off  = unname(quantile(off$draws[, "v"], 0.90)),
      q90_on   = unname(quantile(on$draws[, "v"], 0.90))
    )
  })
  df <- do.call(rbind, rows)

  # --- Non-vacuity: the retry path is actually exercised. -------------------
  # Plain NUTS must diverge on the funnel (otherwise the retry never fires and
  # the comparison is meaningless), and forcing the retry on must rescue the
  # bulk of those divergences (proving the frozen-metric re-run engaged).
  expect_gt(sum(df$div_off), 0)
  expect_lt(sum(df$div_on), sum(df$div_off))

  # --- Invariance: turning the retry on must not shift the v-marginal. ------
  # Per-seed differences are noisy (single-chain MCMC), so the invariance claim
  # is tested on the seed-averaged paired difference, which concentrates at 0
  # if the retry kernel leaves the target invariant. Tolerances are wide enough
  # to absorb Monte Carlo noise (each se ~ 0.08-0.18 over 10 seeds) yet tight
  # enough to catch a real posterior shift.
  expect_lt(abs(mean(df$mean_on - df$mean_off)), 0.5)
  expect_lt(abs(mean(df$sd_on   - df$sd_off)),   0.35)
  expect_lt(abs(mean(df$q10_on  - df$q10_off)),  0.6)
  expect_lt(abs(mean(df$q90_on  - df$q90_off)),  0.5)

  # --- Recovery sanity (loose): both settings recover the same funnel. ------
  # Diagonal-metric NUTS undersamples the funnel neck, so sd(v) sits below the
  # true 3 for BOTH on and off; the point of #189 is that the retry shares that
  # bias rather than adding its own. Assert both hover near the same values.
  expect_lt(abs(mean(df$mean_off)), 1.5)
  expect_lt(abs(mean(df$mean_on)),  1.5)
  expect_true(mean(df$sd_off) > 1.6 && mean(df$sd_off) < 3.4)
  expect_true(mean(df$sd_on)  > 1.6 && mean(df$sd_on)  < 3.4)
})
