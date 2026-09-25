# A GP-evolving temporally-varying coefficient (gcol33/tulpa#847).
#
# `rw1` / `rw2` / `ar1` read the time index as a position on a grid, so
# consecutive instants are one step apart whatever the data says. `gp` is the
# continuous-time structure: the coefficient is a Gaussian process over the
# distinct time VALUES. It is not a duplicate of `temporal_gp()`, which is a GP
# over time entering eta additively (`eta_i += f(t_i)`); here the GP IS the
# coefficient (`eta_i += x_i w(t_i)`).
#
# Four claims:
#   1. THE SURFACE -- the constructor takes a kernel and refuses the choices
#      that have no closed form, and a factor time has no spacing to measure.
#   2. THE LAYOUT -- the field samples an amplitude and a lengthscale per
#      coefficient, under names that say which coordinate they are on.
#   3. THE DERIVATIVE -- the engine's own runtime check (an FD comparison
#      against the active gradient, `ensure_gradient_verified()`) passes on
#      every kernel, so no fit falls back to numerical gradients.
#   4. THE POINT -- on irregularly-spaced instants the GP recovers the
#      trajectory measurably better than the `rw1` that assumes the spacing
#      away.

# Irregular instants: exponential gaps, so the lag between consecutive times
# varies by an order of magnitude and a grid reading of the index is wrong.
.tvcgp_sim <- function(seed = 4L, n_t = 24L, reps = 12L) {
  set.seed(seed)
  tt <- sort(cumsum(stats::rexp(n_t, rate = 1 / 4)))
  tt <- (tt - min(tt)) / diff(range(tt)) * 100
  w  <- 0.9 * sin(2 * pi * tt / 70) + 0.3 * cos(2 * pi * tt / 130)
  w  <- w - mean(w)
  day <- rep(tt, each = reps)
  idx <- rep(seq_len(n_t), each = reps)
  x   <- stats::rnorm(length(day))
  d   <- data.frame(day = day, x = x,
                    y = stats::rpois(length(day), exp(0.4 + (0.8 + w[idx]) * x)))
  list(d = d, w = w, n_t = n_t, N = nrow(d))
}

.tvcgp_fit <- function(sim, temporal, n_iter = 20L, n_chains = 1L, seed = 3L) {
  tulpa(y ~ x, data = sim$d, family = "poisson", temporal = temporal,
        mode = "exact",
        control = list(n_iter = n_iter, n_warmup = as.integer(n_iter / 2),
                       seed = seed, n_chains = n_chains))
}

# TRUE when the fit fell back to numerical gradients, which is what the engine
# does when its runtime finite-difference check disagrees with the active
# gradient. Captured in one handler: `suppressWarnings()` INSIDE a
# `withCallingHandlers()` muffles before the outer handler ever runs.
.tvcgp_grad_fallback <- function(expr) {
  msgs <- character(0)
  withCallingHandlers(
    force(expr),
    warning = function(w) {
      msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning")
    })
  any(grepl("Gradient mismatch", msgs))
}


test_that("the constructor takes a kernel and refuses what has no closed form", {
  s <- temporal_tvc("day", terms = ~ x - 1, structure = "gp")
  expect_equal(s$structure, "gp")
  expect_equal(s$cov, "exponential")

  # Matern is closed-form at nu in {1/2, 3/2, 5/2}; anything between needs a
  # Bessel function of the second kind (gcol33/tulpa#288 was those being
  # accepted and then silently run as exponential).
  expect_error(
    temporal_tvc("day", structure = "gp", cov = "matern", nu = 1.2),
    "0.5, 1.5 or 2.5")
  expect_error(
    temporal_tvc("day", structure = "gp", cov = "periodic"),
    "period.*must be a positive number")
  for (nu in c(0.5, 1.5, 2.5)) {
    expect_s3_class(temporal_tvc("day", structure = "gp", cov = "matern",
                                 nu = nu), "tulpa_tvc")
  }
  # The same check the sibling door makes, so the two cannot come to accept
  # different kernels.
  expect_error(temporal_gp("day", cov = "matern", nu = 1.2), "0.5, 1.5 or 2.5")

  # `rw1` / `rw2` / `ar1` carry no kernel and ignore the arguments.
  expect_null(temporal_tvc("day", structure = "rw1", cov = "matern")$cov)
})


test_that("a GP coefficient needs a time variable with spacing", {
  sim <- .tvcgp_sim(n_t = 6L, reps = 4L)
  d <- sim$d
  d$day <- factor(d$day)
  expect_error(
    tulpa(y ~ x, data = d, family = "poisson",
          temporal = temporal_tvc("day", terms = ~ x - 1, structure = "gp"),
          mode = "exact",
          control = list(n_iter = 4L, n_warmup = 2L, seed = 1L)),
    "needs a numeric time variable")
})


test_that("the field samples an amplitude and a lengthscale per coefficient", {
  skip_on_cran()
  sim <- .tvcgp_sim(n_t = 8L, reps = 4L)
  fit <- .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1, structure = "gp"))
  pn <- colnames(fit$draws)

  expect_true("log_sigma2_tvc_gp[1]" %in% pn)
  expect_true("logit_phi_tvc_gp[1]" %in% pn)
  # The discrete structures' coordinates are NOT allocated: a `log_tau_tvc[j]`
  # holding a log-variance would be a parameter whose name does not say what
  # it is.
  expect_false(any(grepl("^log_tau_tvc\\[", pn)))
  expect_false(any(grepl("^logit_rho_tvc\\[", pn)))
  expect_equal(sum(grepl("^tvc_w\\[", pn)), sim$n_t)
  expect_equal(ncol(fit$draws), 2L + 2L + sim$n_t)   # 2 fixed + 2 hyper + field

  # Proper field, so the level is REMOVED by centring rather than supplied by
  # an augmentation -- the stored draws are centred to match.
  W <- fit$draws[, grep("^tvc_w\\[", pn), drop = FALSE]
  expect_lt(max(abs(rowMeans(W))), 1e-8)
})


test_that("the runtime gradient check passes on every kernel", {
  skip_on_cran()
  # The engine verifies the active gradient against a numerical one before
  # sampling and falls back to numerical gradients on a disagreement. A new
  # density that trips it is a wrong derivative or a floor binding at the
  # starting point; neither should be shipped silently.
  #
  # The lengthscale starts at 0.2 * sd(time) rather than at the midpoint of the
  # (0.01, 10) support. On standardized time the midpoint is a lengthscale five
  # times the data's own spread, which makes the dense T x T covariance
  # numerically rank-one and binds its Cholesky jitter: measured deviation then
  # ordered by kernel smoothness (Matern 5/2 2.4e-03, Gaussian 1.18e-02), the
  # signature of a floor rather than of a derivative.
  sim <- .tvcgp_sim(n_t = 12L, reps = 6L)
  kernels <- list(
    list(cov = "exponential", nu = 1.5, period = NULL),   # O(T) Markov
    list(cov = "matern",      nu = 0.5, period = NULL),   # the same kernel
    list(cov = "matern",      nu = 1.5, period = NULL),   # dense
    list(cov = "matern",      nu = 2.5, period = NULL),   # dense
    list(cov = "gaussian",    nu = 1.5, period = NULL),   # dense, smoothest
    list(cov = "periodic",    nu = 1.5, period = 30)      # dense
  )
  for (k in kernels) {
    fell_back <- .tvcgp_grad_fallback(
      .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1, structure = "gp",
                                   cov = k$cov, nu = k$nu, period = k$period)))
    expect_false(fell_back,
                 label = paste0("gradient fallback for cov = ", k$cov,
                                ", nu = ", k$nu))
  }
})


test_that("on irregular instants the GP beats the grid the rw1 assumes", {
  skip_if_not_slow()
  # The claim the structure exists to make. `rw1` penalises first differences of
  # consecutive INDICES, so a ten-fold gap and a unit gap are penalised alike;
  # the GP measures the lag. Both fits see the same data, the same terms and the
  # same budget.
  sim <- .tvcgp_sim()
  score <- function(fit) {
    W <- fit$draws[, grep("^tvc_w\\[", colnames(fit$draws)), drop = FALSE]
    what <- colMeans(W)
    list(cor = stats::cor(what, sim$w),
         rmse = sqrt(mean((what - sim$w)^2)),
         sd = stats::sd(what))
  }
  f_gp <- .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1, structure = "gp",
                                       cov = "matern", nu = 2.5),
                     n_iter = 900L, n_chains = 2L, seed = 11L)
  f_rw <- .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1, structure = "rw1"),
                     n_iter = 900L, n_chains = 2L, seed = 11L)
  s_gp <- score(f_gp)
  s_rw <- score(f_rw)

  # Measured at this fixture: gp cor 0.999 / rmse 0.042, rw1 cor 0.989 /
  # rmse 0.114. The gates sit well inside both.
  expect_gt(s_gp$cor, 0.97)
  expect_lt(s_gp$rmse, 0.07)
  expect_lt(s_gp$rmse, 0.6 * s_rw$rmse)
  # Amplitude, not only shape: a field recovered at the wrong scale can still
  # correlate perfectly.
  expect_gt(s_gp$sd / stats::sd(sim$w), 0.7)
  expect_lt(s_gp$sd / stats::sd(sim$w), 1.4)
})


test_that("temporal_corr() reports the GP coefficient's own hyperparameters", {
  skip_on_cran()
  sim <- .tvcgp_sim(n_t = 10L, reps = 5L)
  fit <- .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1, structure = "gp"),
                    n_iter = 60L)
  tc <- temporal_corr(fit)
  expect_true("sigma_tvc_gp" %in% rownames(tc))
  expect_true("lengthscale_tvc_gp" %in% rownames(tc))
  # An amplitude is a standard deviation, not the log-variance sampled, and a
  # lengthscale is on the time axis, not the unit position of the logit that
  # samples it: the support is (0.01, 10) in the kernel's standardized units,
  # reported in the user's own time units (gcol33/tulpa#907).
  expect_gt(tc["sigma_tvc_gp", "mean"], 0)
  ell <- tc["lengthscale_tvc_gp", "mean"]
  ts <- fit$temporal$time_scale
  expect_gt(ell, fit$temporal$phi_prior_lower * ts)
  expect_lt(ell, fit$temporal$phi_prior_upper * ts)
  # A discrete-structure fit reports the precision instead, and neither row.
  fit_rw <- .tvcgp_fit(sim, temporal_tvc("day", terms = ~ x - 1,
                                         structure = "rw1"), n_iter = 60L)
  tr <- temporal_corr(fit_rw)
  expect_true("tau_tvc" %in% rownames(tr))
  expect_false("lengthscale_tvc_gp" %in% rownames(tr))
})
