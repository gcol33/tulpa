# Soft sum-to-zero identification of intrinsic fields (gcol33/tulpa#241).
#
# The constant that pins an intrinsic field's constant null direction is a
# PRECISION, derived from the reference idiom sd(sum phi) = kappa * n
# (Morris et al. 2019 / brms / the Stan ICAR case study). Passing kappa itself
# where a precision is meant leaves the direction free at sd 1/sqrt(kappa) and
# the field mean aliases with the intercept -- the defect #241 records.

test_that("s2z_precision implements sd(sum) = kappa * n", {
  kappa <- 0.001
  for (n in c(1L, 5L, 36L, 400L)) {
    expect_equal(tulpa:::cpp_test_s2z_precision(n, kappa),
                 1 / (kappa * n)^2, tolerance = 1e-9)
  }
  # It is a precision, not a standard deviation: larger fields are pinned at a
  # LOWER precision, because the sum they pin is longer.
  expect_lt(tulpa:::cpp_test_s2z_precision(400L, kappa),
            tulpa:::cpp_test_s2z_precision(36L, kappa))
  # The field MEAN is held at sd = kappa regardless of n, which is the point of
  # scaling with n: precision on the mean = n^2 * precision on the sum.
  for (n in c(5L, 36L, 400L)) {
    expect_equal(1 / sqrt(n^2 * tulpa:::cpp_test_s2z_precision(n, kappa)), kappa,
                 tolerance = 1e-9)
  }
})

test_that("the RW sum-to-zero penalty uses the derived precision", {
  phi <- c(0.4, -0.1, 0.3, -0.2, 0.9)
  expected <- -0.5 * tulpa:::cpp_test_s2z_precision(length(phi), 0.001) * sum(phi)^2
  expect_equal(tulpa:::cpp_test_sum_to_zero_penalty(phi), expected, tolerance = 1e-9)
  # A centred field is unpenalised however large the constraint precision is.
  expect_equal(tulpa:::cpp_test_sum_to_zero_penalty(phi - mean(phi)), 0, tolerance = 1e-12)
})

test_that("multiscale temporal pins its intrinsic trend and seasonal arms", {
  # trend (RW1/RW2) and seasonal (cyclic RW1) are both intrinsic and both land
  # on the same linear predictor, so each needs its own pin. The multiscale
  # block is only populated by consumer packages, so this asserts the log-prior
  # directly rather than through a fit.
  set.seed(4)
  n <- 24L; per <- 12L
  trend <- rnorm(n); seasonal <- rnorm(per); short <- rnorm(n)

  lp <- function(tr, se) {
    tulpa:::cpp_test_multiscale_temporal_log_lik(
      tr, se, short,
      sigma2_trend = 1, sigma2_seasonal = 1, sigma2_short = 1, rho_short = 0.5,
      trend_type = "rw1", seasonal_period = per, short_term_type = "ar1")
  }

  # Shifting an intrinsic arm by a constant leaves its RW quadratic untouched,
  # so any change in the log-prior is the augmentation doing its job. Without it
  # these differences are exactly 0 and the level is free.
  expect_gt(abs(lp(trend + 1, seasonal) - lp(trend, seasonal)), 1)
  expect_gt(abs(lp(trend, seasonal + 1) - lp(trend, seasonal)), 1)

  # The augmentation is Q -> Q + 11'/n, entering the quadratic at tau/n = 1/n
  # here (both arms are at sigma2 = 1), on each arm's own length.
  d_trend <- lp(trend + 1, seasonal) - lp(trend, seasonal)
  expect_equal(
    d_trend,
    -0.5 * ((sum(trend) + n)^2 - sum(trend)^2) / n,
    tolerance = 1e-6)

  d_seas <- lp(trend, seasonal + 1) - lp(trend, seasonal)
  expect_equal(
    d_seas,
    -0.5 * ((sum(seasonal) + per)^2 - sum(seasonal)^2) / per,
    tolerance = 1e-6)

  # It scales with the arm's own variance, which the soft pin it replaced did
  # not: at sigma2 = 4 the same shift moves the log-prior a quarter as far.
  lp4 <- function(tr, se) {
    tulpa:::cpp_test_multiscale_temporal_log_lik(
      tr, se, short,
      sigma2_trend = 4, sigma2_seasonal = 1, sigma2_short = 1, rho_short = 0.5,
      trend_type = "rw1", seasonal_period = per, short_term_type = "ar1")
  }
  expect_equal((lp4(trend + 1, seasonal) - lp4(trend, seasonal)) / d_trend,
               0.25, tolerance = 1e-6)

  # The proper short-term arm identifies its own level and is not pinned.
  base <- lp(trend, seasonal)
  expect_true(is.finite(base))
})

test_that("ICAR pins the field level instead of aliasing it into the intercept", {
  skip_on_cran()
  skip_if_not_slow()

  set.seed(11)
  nr <- 6L; nc <- 6L; reps <- 12L; S <- nr * nc
  b0 <- 0.5; b1 <- 0.8
  W <- rook_adj(nr, nc)
  rc <- expand.grid(r = seq_len(nr), c = seq_len(nc))
  phi <- sin(rc$r / 1.4) + cos(rc$c / 1.7) + rnorm(S, 0, 0.3)
  phi <- phi - mean(phi)                      # truth carries no level

  region <- rep(seq_len(S), each = reps)
  x <- rnorm(S * reps)
  ntrials <- rep(20L, S * reps)
  y <- rbinom(length(region), ntrials, plogis(b0 + b1 * x + phi[region]))
  d <- data.frame(y = y, x = x, ntrials = ntrials,
                  region = factor(region, levels = seq_len(S)))

  fit <- tulpa(
    y ~ x + spatial(region), data = d, family = "binomial",
    n_trials = d$ntrials,
    spatial = list(type = "icar", adjacency = W),
    mode = "hmc",
    control = list(n_iter = 1200L, warmup = 600L, seed = 11L, verbose = FALSE)
  )

  dr <- as.matrix(fit$draws)
  icept <- dr[, grep("Intercept|^beta\\[1\\]|^beta_1$", colnames(dr))[1]]
  phi_cols <- grep("^phi", colnames(dr))
  phi_sum <- rowSums(dr[, phi_cols, drop = FALSE])

  # The constant direction is not penalised to a fixed width any more: under the
  # augmented Q_aug = Q + sum_c 1_c 1_c'/J_c it carries the field's OWN
  # precision, so the per-component mean has precision tau*J and the field sum
  # sits at sd = sqrt(J / tau). Its width no longer has to be small, because the
  # field is centred on its way into eta and the direction never reaches the
  # likelihood -- what the assertions below check.
  #
  # This discriminates every regime: the soft pin this replaced held the sum at
  # kappa*J = 0.036, dropping the augmentation entirely leaves it far wider than
  # sqrt(J/tau), and only the augmented prior lands on it.
  tau_draws <- exp(dr[, grep("log_tau_spatial", colnames(dr))[1]])
  expect_lt(abs(sd(phi_sum) / sqrt(S / mean(tau_draws)) - 1), 0.5)

  # With the level pinned, the intercept is identified: it no longer trades
  # one-for-one against the field mean, and its posterior is tight on truth.
  # Pre-fix this correlation was -0.996 and the intercept sd was 0.31.
  expect_lt(abs(cor(phi_sum, icept)), 0.5)
  expect_lt(sd(icept), 0.1)
  expect_lt(abs(mean(icept) - b0), 0.1)

  # Pinning the constant must not shrink the field pattern (the penalty's
  # Hessian is rank-1 on the constant eigenspace, orthogonal to deviations).
  expect_gt(cor(colMeans(dr[, phi_cols, drop = FALSE]), phi), 0.9)
})
