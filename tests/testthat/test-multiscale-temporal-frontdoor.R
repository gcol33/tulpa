# Front-door multiscale temporal (gcol33/tulpa#241 follow-up).
#
# tulpa(y ~ x, temporal = temporal_multiscale("t", ...), mode = "hmc") routes
# through the generic ModelData sampler: build_sampler_model_inputs() parses the
# validated spec onto MultiscaleTemporalData and sets has_multiscale_temporal,
# compute_param_layout() allocates the trend / seasonal / short-term arms plus
# their scales, and the generic log-posterior adds ms_temporal_eta.
#
# That flag was previously never set anywhere in tulpa -- only consumer packages
# reached the block through the ModelData ABI -- so the whole multiscale stack
# was unreachable from this package and its behaviour could only be asserted by
# calling the log-prior kernel directly. These exercise it through the actual
# construction and fitting path instead.
#
# The nested-Laplace path fits the same spec as one block per component
# (test-multiscale-nested.R).

make_ms_binom <- function(n_t = 36L, reps = 8L, seed = 3L) {
  set.seed(seed)
  trend <- cumsum(rnorm(n_t, 0, 0.25)); trend <- trend - mean(trend)
  time <- rep(seq_len(n_t), each = reps); N <- length(time)
  x <- rnorm(N)
  ntr <- rep(8L, N)
  y <- rbinom(N, ntr, plogis(0.4 - 0.6 * x + trend[time]))
  list(d = data.frame(y = y, x = x, t = time, ntrials = ntr),
       trend = trend, n_t = n_t, b1 = -0.6)
}

test_that("multiscale temporal reaches the nested-Laplace path as blocks", {
  s <- make_ms_binom(n_t = 6L, reps = 4L)
  spec <- validate_temporal_multiscale(
    temporal_multiscale("t", trend = "rw1", short_term = "none"), s$d)
  blocks <- tulpa:::.temporal_spec_to_nl_blocks(spec)
  expect_length(blocks, 1L)
  expect_identical(blocks[[1L]]$type, "rw1")
  expect_identical(blocks[[1L]]$name, "trend")
})

# The trend spec build_sampler_model_inputs() reads, one observation per time
# point so eta is the field itself.
ms_trend_spec <- function(n_t, trend, par = "noncentered") {
  list(type = "multiscale", time_index = seq_len(n_t), n_times = n_t,
       trend = trend, trend_parameterization = par, seasonal = NULL,
       short_term = "none", shared = TRUE,
       sigma2_trend_prior_U = 1, sigma2_trend_prior_alpha = 0.01)
}

ms_trend_eta <- function(n_t, trend, rows) {
  spec <- ms_trend_spec(n_t, trend)
  X <- matrix(1, n_t, 1, dimnames = list(NULL, "(Intercept)"))
  args <- list(y = rnorm(n_t), n_trials = rep(1L, n_t), X = X,
               family = "gaussian", phi = 1, temporal_spec = spec)
  pn <- do.call(tulpa:::cpp_tulpa_glmm_layout, args)$param_names
  D <- matrix(0, length(rows), length(pn), dimnames = list(NULL, pn))
  for (r in seq_along(rows)) D[r, names(rows[[r]])] <- rows[[r]]
  do.call(tulpa:::cpp_tulpa_glmm_eta_draws, c(list(draws = D), args))
}

test_that("the non-centred trend is the walk's prior covariance in its spectral basis", {
  # A unit coordinate j puts the j-th basis column b_j into eta, and
  # sum_j b_j b_j' is the walk's covariance on the complement of its null
  # space: the pseudo-inverse of the structure matrix R = D'D, at sigma = 1.
  for (trend in c("rw1", "rw2")) {
    n_t <- 9L; k <- if (trend == "rw1") 1L else 2L
    unit <- lapply(seq_len(n_t - k), function(j) {
      v <- c(0, 1)
      names(v) <- c("log_sigma2_trend", sprintf("trend[%d]", k + j))
      v
    })
    B <- t(ms_trend_eta(n_t, trend, unit))
    Dk <- diff(diag(n_t), differences = k)
    R <- crossprod(Dk)
    e <- eigen(R, symmetric = TRUE)
    pos <- e$values > 1e-9
    R_pinv <- e$vectors[, pos] %*% (t(e$vectors[, pos]) / e$values[pos])
    expect_equal(B %*% t(B), R_pinv, tolerance = 1e-8)
    expect_equal(colSums(B), rep(0, n_t - k), tolerance = 1e-10)

    # The scale multiplies the field: sigma = 2 doubles every column.
    two <- lapply(unit, function(v) { v["log_sigma2_trend"] <- 2 * log(2); v })
    expect_equal(t(ms_trend_eta(n_t, trend, two)), 2 * B, tolerance = 1e-10)

    # The level coordinate enters no field value.
    lvl <- list(c(log_sigma2_trend = 0, "trend[1]" = 3))
    expect_equal(max(abs(ms_trend_eta(n_t, trend, lvl))), 0, tolerance = 1e-12)
  }
  # An RW2's slope coordinate is the unit centred linear direction, at any sigma.
  sl <- ms_trend_eta(7L, "rw2", list(c(log_sigma2_trend = -3, "trend[2]" = 1)))
  lin <- seq_len(7L) - 4
  expect_equal(as.numeric(sl), lin / sqrt(sum(lin^2)), tolerance = 1e-10)
})

test_that("the trend parameterization is an argument, non-centred by default", {
  expect_identical(temporal_multiscale("t")$parameterization, "noncentered")
  expect_identical(
    temporal_multiscale("t", parameterization = "centered")$parameterization,
    "centered")
  expect_error(temporal_multiscale("t", parameterization = "cp"),
               "should be one of")
})

test_that("a per-group multiscale field is refused where the sampler reads it", {
  spec <- ms_trend_spec(6L, "rw1")
  spec$n_groups <- 2L
  spec$group_index <- rep(1:2, 3)
  X <- matrix(1, 6, 1, dimnames = list(NULL, "(Intercept)"))
  expect_error(
    tulpa:::cpp_tulpa_glmm_layout(y = rnorm(6), n_trials = rep(1L, 6), X = X,
                                  family = "gaussian", temporal_spec = spec),
    "one walk per component")
  expect_error(temporal_multiscale("t", group_var = "g"), "unused argument")
})

skip_on_cran_budget()

test_that("an unsupported multiscale component is refused by name", {
  skip_on_cran_budget()
  d <- data.frame(t = rep(1:5, 2))
  expect_error(temporal_multiscale("t", trend = "rw3"), "should be one of")
  expect_error(temporal_multiscale("t", short_term = "arma"), "should be one of")
  # Every component switched off is refused at construction, not at fit time.
  expect_error(temporal_multiscale("t", trend = "none", short_term = "none"),
               "At least one temporal component")
})

skip_on_cran()

test_that("multiscale temporal allocates and names its arms (structural)", {
  skip_if_not_slow()
  s <- make_ms_binom(n_t = 12L, reps = 4L)
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1", seasonal = 4,
                                              short_term = "ar1"),
               mode = "hmc",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 1L,
                              verbose = FALSE))
  pn <- colnames(fit$draws)
  # One scale per active arm, the AR1 correlation, and one value per node:
  # trend over n_times, seasonal over the period, short-term over n_times.
  expect_true(all(c("log_sigma2_trend", "log_sigma2_seasonal",
                    "log_sigma2_short", "logit_rho_short") %in% pn))
  expect_equal(sum(grepl("^trend\\[", pn)), 12L)
  expect_equal(sum(grepl("^seasonal\\[", pn)), 4L)
  expect_equal(sum(grepl("^short_term\\[", pn)), 12L)
  # 2 fixed + 4 hyper + 12 + 4 + 12.
  expect_equal(ncol(fit$draws), 2L + 4L + 12L + 4L + 12L)
})

test_that("a switched-off arm allocates nothing", {
  skip_if_not_slow()
  s <- make_ms_binom(n_t = 10L, reps = 4L)
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1",
                                              short_term = "none"),
               mode = "hmc",
               control = list(n_iter = 120L, n_warmup = 60L, seed = 2L,
                              verbose = FALSE))
  pn <- colnames(fit$draws)
  expect_true("log_sigma2_trend" %in% pn)
  expect_false(any(grepl("^seasonal\\[|^short_term\\[", pn)))
  expect_false(any(c("log_sigma2_seasonal", "log_sigma2_short",
                     "logit_rho_short") %in% pn))
  expect_equal(ncol(fit$draws), 2L + 1L + 10L)
})

test_that("multiscale temporal recovers the trend through the front door", {
  skip_if_not_slow()
  s <- make_ms_binom()
  fit <- tulpa(y ~ x, data = s$d, family = "binomial", n_trials = s$d$ntrials,
               temporal = temporal_multiscale("t", trend = "rw1",
                                              short_term = "none"),
               mode = "hmc",
               control = list(n_iter = 1500L, n_warmup = 750L, seed = 5L,
                              verbose = FALSE))
  dr <- as.matrix(fit$draws)
  tcol <- grep("^trend\\[", colnames(dr))
  expect_equal(length(tcol), s$n_t)

  # The trend shape is the identified quantity: an RW1 level is confounded with
  # the intercept, so the constant is not asserted.
  th <- colMeans(dr[, tcol, drop = FALSE])
  expect_gt(cor(th, s$trend), 0.85)

  # The covariate slope is identified separately from the walk.
  expect_lt(abs(unname(coef(fit)["x"]) - s$b1), 0.2)
})

# The exact posterior of log(sigma2_trend) for a gaussian response with known
# noise variance and an RW2 trend: conditional on the scale the response is
# Gaussian with the fixed effects, the walk's innovations and its flat slope
# integrated in closed form (the slope's flat prior is the limit of a wide
# Gaussian, which leaves the profile terms below), on a dense grid in the
# scale. The level is centred out of eta on every route, so it does not enter.
ms_rw2_exact <- function(d, n_t, phi, sigma_beta = 2.5, U = 1, alpha = 0.01) {
  N <- nrow(d)
  X <- cbind(1, d$x)
  A <- matrix(0, N, n_t); A[cbind(seq_len(N), d$t)] <- 1
  C <- diag(n_t) - 1 / n_t
  L <- matrix(0, n_t, n_t - 2)
  for (j in seq_len(n_t - 2)) {
    f <- numeric(n_t); f[j + 2] <- 1
    if (j + 3 <= n_t) for (t in (j + 3):n_t) f[t] <- 2 * f[t - 1] - f[t - 2]
    L[, j] <- f
  }
  M <- A %*% C %*% L
  a <- as.numeric(A %*% C %*% (seq_len(n_t) - 1))
  lam <- -log(alpha) / U
  th <- seq(-14, 2, length.out = 801)
  lp <- vapply(th, function(t) {
    S <- sigma_beta^2 * tcrossprod(X) + exp(t) * tcrossprod(M) + diag(phi, N)
    R <- chol(S)
    Siy <- backsolve(R, forwardsolve(t(R), d$y))
    Sia <- backsolve(R, forwardsolve(t(R), a))
    q <- sum(a * Sia)
    -sum(log(diag(R))) - 0.5 * sum(d$y * Siy) - 0.5 * log(q) +
      0.5 * sum(a * Siy)^2 / q +
      log(lam) - lam * exp(t / 2) - log(2) + 0.5 * t
  }, 0)
  w <- exp(lp - max(lp)); w <- w / sum(w)
  m <- sum(w * th)
  list(mean = m, sd = sqrt(sum(w * th^2) - m^2),
       q = stats::approx(cumsum(w), th, xout = c(0.05, 0.5, 0.95),
                         ties = "ordered")$y)
}

test_that("the non-centred RW2 trend samples the exact scale posterior", {
  skip_if_not_slow()
  set.seed(1)
  n_t <- 30L; phi <- 0.25
  tr <- cumsum(cumsum(rnorm(n_t, 0, 0.03))); tr <- tr - mean(tr)
  t <- rep(seq_len(n_t), each = 3L)
  x <- rnorm(length(t))
  d <- data.frame(t = t, x = x,
                  y = 0.3 + 0.5 * x + tr[t] + rnorm(length(t), 0, sqrt(phi)))
  ex <- ms_rw2_exact(d, n_t, phi)
  fit <- suppressWarnings(tulpa(
    y ~ x, data = d, family = "gaussian", phi = phi, mode = "hmc",
    temporal = temporal_multiscale("t", trend = "rw2", short_term = "none"),
    control = list(n_iter = 3000L, n_warmup = 1000L, n_chains = 4L, seed = 1L,
                   verbose = FALSE)))
  ls2 <- fit$draws[, "log_sigma2_trend"]
  # Measured at 4000 x 4 on three such data sets: 0.5-1.0% divergent, bulk ESS
  # 1140-2300, every quantile within 0.13 posterior SD of the exact one. The
  # centred trend reached ESS 320-510 and the innovation (cumulative-sum)
  # non-centred one 2.5-7.2% divergent.
  expect_lt(mean(fit$divergent), 0.03)
  expect_gt(posterior::ess_bulk(matrix(ls2, ncol = 4L)), 400)
  expect_lt(abs(mean(ls2) - ex$mean), 0.25 * ex$sd)
  expect_lt(max(abs(stats::quantile(ls2, c(0.05, 0.5, 0.95)) - ex$q)),
            0.3 * ex$sd)
})

test_that("a smooth RW2 trend on a binomial response samples without a funnel", {
  skip_if_not_slow()
  set.seed(11)
  n_t <- 40L
  tr <- cumsum(cumsum(rnorm(n_t, 0, 0.02))); tr <- tr - mean(tr)
  t <- rep(seq_len(n_t), each = 6L)
  x <- rnorm(length(t))
  d <- data.frame(t = t, x = x, ntr = 10L,
                  y = rbinom(length(t), 10L, plogis(-0.2 + 0.5 * x + tr[t])))
  fit <- suppressWarnings(tulpa(
    y ~ x, data = d, family = "binomial", n_trials = d$ntr, mode = "hmc",
    temporal = temporal_multiscale("t", trend = "rw2", short_term = "iid"),
    control = list(n_iter = 2000L, n_warmup = 1000L, n_chains = 4L, seed = 7L,
                   verbose = FALSE)))
  # The centred trend on these data:
  # 6.7% divergent, 20% in the lowest quartile of log(sigma2_trend), bulk ESS
  # 59; the spectral non-centred one 0.4% and 505.
  ls2 <- fit$draws[, "log_sigma2_trend"]
  low <- ls2 <= stats::quantile(ls2, 0.25)
  expect_lt(mean(fit$divergent), 0.02)
  expect_lt(mean(fit$divergent[low]), 0.02)
  expect_gt(posterior::ess_bulk(matrix(ls2, ncol = 4L)), 250)
  tr_hat <- colMeans(fit$draws[, grep("^trend[[]", colnames(fit$draws))])
  expect_gt(cor(tr_hat, tr), 0.95)
})
