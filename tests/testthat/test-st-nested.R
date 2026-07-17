# Additive spatiotemporal nested-Laplace driver fit_st_nested() (#158): wires the
# cpp_nested_laplace_st_* kernels (previously reachable only from consumer
# packages) through an exported fitter. Correctness anchor: the recovered
# spatial and temporal field posterior means track the simulated fields, and the
# fixed effects recover.

sim_st <- function(seed = 7L, n_s = 20L, n_t = 10L, N = 600L) {
  set.seed(seed)
  adj <- matrix(0, n_s, n_s)
  for (i in 1:(n_s - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
  us <- as.numeric(scale(cumsum(rnorm(n_s)))) * 0.9
  vt <- as.numeric(scale(cumsum(rnorm(n_t)))) * 0.8
  s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE); x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.2 + 0.6 * x + us[s] + vt[tt]))
  list(y = y, X = cbind(1, x), s = s, tt = tt, adj = adj, n_t = n_t,
       us = us, vt = vt)
}

test_that("fit_st_nested recovers the spatial and temporal fields (icar x rw1)", {
  skip_on_cran()
  d <- sim_st()
  fit <- fit_st_nested(d$y, d$X, d$s, d$adj, d$tt, d$n_t,
                       spatial_type = "icar", temporal_type = "rw1",
                       family = "binomial")
  expect_s3_class(fit, "tulpa_nested_laplace")
  # Fixed effects recover.
  expect_lt(abs(coef(fit)[2] - 0.6), 0.2)
  # Field posterior means track the truth (centred; the field level is
  # confounded with the intercept).
  cs <- cor(fit$spatial_effects - mean(fit$spatial_effects),
            d$us - mean(d$us))
  ct <- cor(fit$temporal_effects - mean(fit$temporal_effects),
            d$vt - mean(d$vt))
  expect_gt(cs, 0.85)
  expect_gt(ct, 0.80)
  # Weights are a proper posterior over the (tau_spatial, tau_temporal, rho) grid.
  expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
  expect_true(all(is.finite(fit$log_marginal)))
})

test_that("fit_st_nested supports ar1 temporal and the generic accessors", {
  skip_on_cran()
  d <- sim_st(seed = 3L)
  fit <- fit_st_nested(d$y, d$X, d$s, d$adj, d$tt, d$n_t,
                       spatial_type = "icar", temporal_type = "ar1",
                       family = "binomial")
  expect_length(coef(fit), 2L)
  expect_false(is.null(vcov(fit)))
  expect_length(fit$temporal_effects, d$n_t)
  # rho is a real integrated axis.
  expect_true("rho" %in% fit$theta_names)
})

test_that("fit_st_nested validates its indices", {
  d <- sim_st(N = 40L, n_s = 10L, n_t = 5L)
  expect_error(
    fit_st_nested(d$y, d$X, d$s + 100L, d$adj, d$tt, d$n_t),
    "spatial_idx"
  )
  expect_error(
    fit_st_nested(d$y, d$X, d$s, d$adj, d$tt[-1], d$n_t),
    "must have length"
  )
})
