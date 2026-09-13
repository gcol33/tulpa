# The in-sample linear predictor behind WAIC / LOO / posterior_predict()
# (gcol33/tulpa#721), and the log evidence logLik() reports for an outer grid
# (gcol33/tulpa#722).

make_trend_data <- function(seed, n_times = 24L, n = 720L) {
  set.seed(seed)
  time  <- sample.int(n_times, n, replace = TRUE)
  trend <- cumsum(rnorm(n_times, 0, 0.35))
  trend <- trend - mean(trend)
  x     <- rnorm(n)
  list(df = data.frame(y = rbinom(n, 1, plogis(0.3 + 0.7 * x + trend[time])),
                       x = x, time = time),
       trend = trend, time = time)
}

test_that("a nested fit's eta draws carry its temporal field", {
  skip_on_cran()
  d   <- make_trend_data(20260529)
  fit <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_rw1("time"))
  expect_identical(tulpa:::.tulpa_linpred_source(fit), "grid_mixture")

  eta  <- tulpa:::.tulpa_eta_draws(fit, synth_seed = 285603L)
  expect_identical(dim(eta), c(400L, nrow(d$df)))
  expect_identical(eta, tulpa:::.tulpa_eta_draws(fit, synth_seed = 285603L))

  # The draws' moments are the grid mixture's, to Monte Carlo error, and what
  # the mean adds to X beta is the trend: it varies over time and tracks the
  # simulated walk.
  em <- as.numeric(crossprod(fit$weights, fit$fitted_eta))
  ev <- as.numeric(crossprod(fit$weights, fit$fitted_eta_var + fit$fitted_eta^2)) -
    em^2
  expect_lt(max(abs(colMeans(eta) - em) / sqrt(ev / nrow(eta))), 5)
  by_t <- tapply(em - as.numeric(fit$model_matrix %*% coef(fit)), d$time, mean)
  expect_gt(stats::sd(by_t), 0.2)
  expect_gt(stats::cor(as.numeric(by_t), d$trend), 0.8)

  # WAIC then ranks the model that carries the trend first.
  m_nt <- tulpa(y ~ x, data = d$df, family = "binomial", mode = "laplace")
  cmp  <- compare_models(no_temporal = m_nt, temporal = fit, criterion = "waic")
  expect_identical(cmp$model[1L], "temporal")
  expect_gt(-cmp$delta[2L], 5)
})

test_that("a sampler fit's eta draws are the engine's linear predictor, field included", {
  skip_on_cran()
  d <- make_trend_data(7, n_times = 12L, n = 240L)
  h <- tulpa(y ~ x, data = d$df, family = "binomial",
             temporal = temporal_rw1("time"), mode = "hmc",
             control = list(n_iter = 60L, warmup = 40L, n_chains = 1L, seed = 3L))
  expect_identical(tulpa:::.tulpa_linpred_source(h), "sampler_model")
  eta <- tulpa:::.tulpa_eta_draws(h)

  # Independent assembly from the draw columns: the walk enters eta centred,
  # because its global level is aliased with the intercept.
  D  <- h$draws
  ph <- D[, grep("^phi_temporal\\[", colnames(D)), drop = FALSE]
  hand <- D[, c("(Intercept)", "x")] %*% t(h$model_matrix) +
    (ph - rowMeans(ph))[, d$time]
  expect_equal(unname(eta), unname(hand), tolerance = 1e-12)
})

test_that("an areal nested fit's eta draws carry its field", {
  skip_on_cran()
  set.seed(11)
  S <- 16L
  W <- matrix(0, S, S)
  for (i in 1:4) for (j in 1:4) {
    k <- (i - 1L) * 4L + j
    if (j < 4L) { W[k, k + 1L] <- 1; W[k + 1L, k] <- 1 }
    if (i < 4L) { W[k, k + 4L] <- 1; W[k + 4L, k] <- 1 }
  }
  u <- as.numeric(scale(rep(c(-1, 1), each = 8L)))
  site <- sample.int(S, 480L, replace = TRUE)
  x <- rnorm(480L)
  dd <- data.frame(y = rpois(480L, exp(0.2 + 0.4 * x + 0.8 * u[site])),
                   x = x, site = site)
  fit <- tulpa(y ~ x + spatial(site), data = dd, family = "poisson",
               spatial = list(type = "icar", adjacency = W),
               mode = "nested_laplace")
  em <- colMeans(tulpa:::.tulpa_eta_draws(fit, synth_seed = 1L))
  by_s <- tapply(em - as.numeric(fit$model_matrix %*% coef(fit)), site, mean)
  expect_gt(stats::cor(as.numeric(by_s), u[as.integer(names(by_s))]), 0.8)
})

test_that("the outer evidence normalises the prior the weights define", {
  lm <- c(-10, -11, -12, -13)
  lq <- log(c(0.1, 0.2, 0.3, 0.4))
  # Flat hyperprior, a normalised measure: the plain cell-mass-weighted sum.
  expect_equal(tulpa:::.nl_outer_log_evidence(lm, lq),
               log(sum(exp(lm + lq))), tolerance = 1e-14)
  # Rescaling the measure or the folded density leaves the value alone.
  expect_equal(tulpa:::.nl_outer_log_evidence(lm, lq + 3),
               tulpa:::.nl_outer_log_evidence(lm, lq), tolerance = 1e-13)
  lh <- c(-1, -0.5, -0.2, -2)
  expect_equal(tulpa:::.nl_outer_log_evidence(lm + lh, lq, lh),
               tulpa:::.nl_outer_log_evidence(lm + lh + 5, lq, lh + 5),
               tolerance = 1e-13)
  # A folded density is normalised over the cells it was folded on.
  expect_equal(tulpa:::.nl_outer_log_evidence(lm + lh, lq, lh),
               log(sum(exp(lm + lh + lq))) - log(sum(exp(lh + lq))),
               tolerance = 1e-14)
})

test_that("a failed cell is conditioned out; a pruned cell keeps its prior mass", {
  lm <- c(-10, -11, -12, -13)
  lq <- log(rep(0.25, 4))
  ref <- tulpa:::.nl_outer_log_evidence(lm[1:3], lq[1:3])
  expect_equal(tulpa:::.nl_outer_log_evidence(c(lm[1:3], NaN), lq), ref)
  expect_equal(tulpa:::.nl_outer_log_evidence(c(lm[1:3], Inf), lq), ref)
  expect_equal(tulpa:::.nl_outer_log_evidence(c(lm[1:3], -Inf), lq),
               log(sum(exp(lm[1:3] + lq[1:3]))), tolerance = 1e-14)
  expect_true(is.na(tulpa:::.nl_outer_log_evidence(c(NaN, NaN), c(0, 0))))
})

test_that("logLik() on a single-axis grid does not move with its node count", {
  # The node set tiles the same log-tau support at every K, so the evidence is
  # one integral read at increasing resolution. The unweighted log-sum-exp
  # grows by about log(K) over the same range.
  skip_on_cran()
  d <- make_trend_data(20260529)
  lo <- log(0.5); hi <- log(200)
  ev <- vapply(c(9L, 33L, 129L), function(K) {
    tg <- exp(lo + (hi - lo) / K * (seq_len(K) - 0.5))
    f <- tulpa_nested_laplace(
      y = d$df$y, n_trials = rep(1L, nrow(d$df)), X = cbind(1, d$df$x),
      prior = list(type = "rw1", temporal_idx = as.integer(d$time),
                   n_times = 24L, tau_grid = tg),
      family = "binomial",
      control = list(diagnose_k = FALSE, diagnose_skew = FALSE))
    expect_equal(as.numeric(f$axis_support$tau), c(0.5, 200), tolerance = 1e-12)
    expect_length(f$log_quad, K)
    as.numeric(logLik(f))
  }, numeric(1))
  expect_lt(diff(range(ev)), 1e-3)

  # Arbiter: a trapezoid in log tau over the same support, on a dense grid of
  # inner marginals, written without the engine's cell measure.
  tg <- exp(seq(lo, hi, length.out = 401L))
  f <- tulpa_nested_laplace(
    y = d$df$y, n_trials = rep(1L, nrow(d$df)), X = cbind(1, d$df$x),
    prior = list(type = "rw1", temporal_idx = as.integer(d$time),
                 n_times = 24L, tau_grid = tg),
    family = "binomial",
    control = list(diagnose_k = FALSE, diagnose_skew = FALSE))
  u <- log(tg); v <- f$log_marginal; m <- max(v)
  trap <- m + log(sum(diff(u) * (exp(v[-1L] - m) + exp(v[-401L] - m)) / 2)) -
    log(hi - lo)
  expect_equal(ev[3L], trap, tolerance = 1e-4)
})

test_that("a two-axis grid's evidence converges in its node count at a fixed support", {
  skip_on_cran()
  set.seed(5)
  S <- 20L
  adj <- .pgp_chain_adj(S)
  u <- sin(seq_len(S) / 3)
  site <- rep(seq_len(S), each = 15L)
  x <- rnorm(length(site))
  y <- rpois(length(site), exp(0.2 + 0.4 * x + u[site]))
  # Both axes tile a support that does not move with K: sigma over [0.05, 5] in
  # log, rho over [0.1, 0.9], inside its (0, 1) domain so no closure applies.
  lse <- function(v) { m <- max(v); m + log(sum(exp(v - m))) }
  old <- numeric(2)
  ev <- vapply(c(12L, 24L), function(K) {
    sg <- exp(log(0.05) + (log(5) - log(0.05)) / K * (seq_len(K) - 0.5))
    rg <- 0.1 + 0.8 * (seq_len(K) - 0.5) / K
    gr <- expand.grid(sigma = sg, rho = rg)
    f <- tulpa_nested_laplace(
      y = y, n_trials = rep(1L, length(y)), X = cbind(1, x),
      prior = list(type = "bym2", spatial_idx = as.integer(site),
                   n_spatial_units = S, adj_row_ptr = adj$adj_row_ptr,
                   adj_col_idx = adj$adj_col_idx, n_neighbors = adj$n_neighbors,
                   scale_factor = 1, sigma_grid = gr$sigma, rho_grid = gr$rho),
      family = "poisson",
      control = list(diagnose_k = FALSE, diagnose_skew = FALSE,
                     auto_recenter = FALSE))
    old[K %/% 12L] <<- lse(f$log_marginal)
    as.numeric(logLik(f))
  }, numeric(1))
  expect_lt(abs(diff(ev)), 0.01)
  expect_gt(abs(diff(old)), 1)
})

test_that("a fit with a per-cell vector and no recorded measure declines", {
  fit <- structure(list(log_marginal = c(-10, -11, -12), n_fixed = 2L, N = 100L),
                   class = "tulpa_fit")
  ll <- logLik(fit)
  expect_true(is.na(as.numeric(ll)))
  expect_identical(attr(ll, "declined"), "outer_measure_not_recorded")
})
