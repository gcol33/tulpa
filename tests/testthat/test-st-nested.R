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

test_that("fit_st_nested stamps family/n_trials/phi so the observation-level accessors work (#777)", {
  skip_on_cran()
  d <- sim_st(seed = 5L)
  # Fully named design: the blank-column-name crash is a separate defect (#780).
  colnames(d$X) <- c("(Intercept)", "x")
  fit <- fit_st_nested(d$y, d$X, d$s, d$adj, d$tt, d$n_t,
                       spatial_type = "icar", temporal_type = "rw1",
                       family = "binomial")
  expect_identical(fit$family, "binomial")
  expect_identical(fit$n_trials, rep(1L, length(d$y)))
  expect_identical(fit$phi, 1.0)

  fv <- fitted(fit)
  expect_length(fv, length(d$y))
  expect_true(all(is.finite(fv)))
  rv <- residuals(fit)
  expect_length(rv, length(d$y))
  expect_true(all(is.finite(rv)))

  pp <- posterior_predict(fit, ndraws = 5)
  expect_equal(dim(pp), c(5L, length(d$y)))
  sim <- simulate(fit, nsim = 2)
  expect_equal(dim(sim), c(length(d$y), 2L))
  expect_error(test_dispersion(fit), NA)
})

# bym2's kernel (cpp_nested_laplace_st_bym2) integrates a (sigma_spatial,
# rho_spatial) grid + a `scale_factor` computed from the adjacency, not the
# `tau_spatial_grid` every other spatial type shares -- fit_st_nested() used
# to build the icar-shaped argument list for every kernel, so this cell was
# unreachable ("unused argument (tau_spatial_grid = ...)") on all three
# temporal types (gcol33/tulpa#776).
sim_st_bym2 <- function(seed = 11L, n_s = 20L, n_t = 10L, N = 600L, rho = 0.7) {
  set.seed(seed)
  adj <- matrix(0, n_s, n_s)
  for (i in 1:(n_s - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
  sf <- compute_bym2_scale(adj)
  phi   <- as.numeric(scale(cumsum(rnorm(n_s))))
  theta <- rnorm(n_s)
  us <- 0.9 * sf * (sqrt(rho) * phi + sqrt(1 - rho) * theta)
  vt <- as.numeric(scale(cumsum(rnorm(n_t)))) * 0.8
  s <- sample(n_s, N, TRUE); tt <- sample(n_t, N, TRUE); x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.2 + 0.6 * x + us[s] + vt[tt]))
  list(y = y, X = cbind(1, x), s = s, tt = tt, adj = adj, n_t = n_t,
       us = us, vt = vt)
}

test_that("fit_st_nested reaches bym2 x every temporal type (gcol33/tulpa#776)", {
  skip_on_cran()
  for (tty in c("ar1", "rw1", "rw2")) {
    d <- sim_st_bym2(seed = switch(tty, ar1 = 11L, rw1 = 12L, rw2 = 13L))
    fit <- fit_st_nested(d$y, d$X, d$s, d$adj, d$tt, d$n_t,
                         spatial_type = "bym2", temporal_type = tty,
                         family = "binomial")
    expect_s3_class(fit, "tulpa_nested_laplace")
    expect_identical(fit$spatial_type, "bym2")
    expect_lt(abs(coef(fit)[2] - 0.6), 0.25)
    expect_length(fit$spatial_effects, 20L)
    expect_length(fit$temporal_effects, d$n_t)
    cs <- cor(fit$spatial_effects - mean(fit$spatial_effects),
              d$us - mean(d$us))
    ct <- cor(fit$temporal_effects - mean(fit$temporal_effects),
              d$vt - mean(d$vt))
    expect_gt(cs, 0.6)
    expect_gt(ct, 0.6)
    expect_equal(sum(fit$weights), 1, tolerance = 1e-8)
    expect_true(all(is.finite(fit$log_marginal)))
    expect_true(all(c("sigma_spatial", "rho_spatial") %in% fit$theta_names))
    if (identical(tty, "ar1")) expect_true("rho" %in% fit$theta_names)
  }
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
