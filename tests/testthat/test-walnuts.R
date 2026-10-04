# WALNUTS transition (src/hmc_walnuts.cpp, gcol33/tulpa#937)
#
# cpp_test_funnel_nuts() drives the production chain driver on Neal's funnel,
# v ~ N(0, gamma^2), x_i | v ~ N(0, exp(v / 2)), or with neck = FALSE on the
# Gaussian that keeps the v-marginal and drops the varying curvature. Both
# targets have a known v-marginal, so every check below is against the target
# itself rather than against another sampler.

funnel_v <- function(seed, walnuts = TRUE, neck = TRUE, n_iter = 10000L,
                     n_warmup = 1500L, max_step_halvings = 10L,
                     max_error = 0.5) {
  tulpa:::cpp_test_funnel_nuts(
    K = 9L, gamma = 3, n_iter = n_iter, n_warmup = n_warmup,
    max_treedepth = 10L, adapt_delta = 0.8, seed = seed,
    walnuts = walnuts, max_step_halvings = max_step_halvings,
    max_error = max_error, neck = neck
  )
}

test_that("the funnel entry returns a well-formed fit under either transition", {
  skip_on_cran()  # tier 2: short fits, plumbing + shape

  for (w in c(FALSE, TRUE)) {
    fit <- tulpa:::cpp_test_funnel_nuts(
      K = 6L, gamma = 3, n_iter = 400L, n_warmup = 200L,
      max_treedepth = 8L, seed = 1L, walnuts = w
    )
    expect_equal(fit$walnuts, w)
    expect_equal(fit$n_params, 7L)
    expect_equal(dim(fit$draws), c(200L, 7L))
    expect_equal(colnames(fit$draws), c("v", paste0("x[", 1:6, "]")))
    expect_true(all(is.finite(fit$draws)))
    expect_equal(fit$n_divergent, sum(fit$divergent))
    expect_true(all(fit$treedepth >= 0L & fit$treedepth <= 8L))
    expect_true(all(fit$n_leapfrog >= 1L))
  }
})

test_that("WALNUTS samples a Gaussian exactly while subdividing every macro step", {
  skip_on_cran()  # tier 2: one fit against a known target

  # max_error = 0.05 makes most macro steps subdivide, so the reversibility
  # check runs on most leaves; the default-tolerance fit takes about 6.6
  # gradients an iteration.
  fit <- funnel_v(1L, neck = FALSE, max_error = 0.05)
  v <- fit$draws[, "v"]
  expect_gt(mean(fit$n_leapfrog), 15)
  expect_lt(abs(mean(v)), 0.2)
  expect_lt(abs(sd(v) - 3), 0.15)
  expect_lt(abs(quantile(v, 0.1, names = FALSE) - 3 * qnorm(0.1)), 0.3)
  expect_lt(abs(quantile(v, 0.9, names = FALSE) - 3 * qnorm(0.9)), 0.3)
  expect_lt(abs(sd(fit$draws[, "x[1]"]) - 1), 0.08)
})

test_that("WALNUTS recovers the funnel's v-marginal, where NUTS does not", {
  skip_if_not_slow()  # tier 3: multi-seed recovery against the known target

  # At 100k iterations over 20 seeds WALNUTS reads mean 0.02 (se 0.03),
  # sd 2.98 (se 0.02), q10 -3.78; NUTS on the same chains reads 0.73 / 2.37 /
  # -2.15, so the sd bound below separates the two.
  seeds <- 1:10
  rows <- lapply(seeds, function(s) {
    w <- funnel_v(s, n_iter = 50000L)$draws[, "v"]
    data.frame(mean = mean(w), sd = sd(w),
               q10 = quantile(w, 0.1, names = FALSE),
               q90 = quantile(w, 0.9, names = FALSE))
  })
  df <- do.call(rbind, rows)

  expect_lt(abs(mean(df$mean)), 0.25)
  expect_lt(abs(mean(df$sd) - 3), 0.15)
  expect_lt(abs(mean(df$q10) - 3 * qnorm(0.1)), 0.3)
  expect_lt(abs(mean(df$q90) - 3 * qnorm(0.9)), 0.3)
  expect_lt(max(abs(df$mean)), 1)

  nuts_sd <- vapply(seeds, function(s) {
    sd(funnel_v(s, walnuts = FALSE, n_iter = 50000L)$draws[, "v"])
  }, numeric(1))
  expect_gt(abs(mean(nuts_sd) - 3), 0.15)
})
