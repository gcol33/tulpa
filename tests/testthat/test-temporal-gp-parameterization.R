# The centered and non-centered temporal GP are reparameterizations of one
# posterior, so their densities differ by exactly the forward transform's
# log-determinant (gcol33/tulpa#499).
#
# What that pins is the conditional variance the two branches use. The
# non-centered branch reaches it through the transform's scale
# a_t = sigma sqrt(1 - rho_t^2) and the centered branch through
# cond_var_t = sigma^2 (1 - rho_t^2). Flooring DIFFERENT quantities --
# sigma^2 max(1 - rho^2, eps) against max(sigma^2 (1 - rho^2), eps) -- leaves
# the two agreeing wherever the floor is slack and disagreeing by orders of
# magnitude wherever it binds, which is any long lengthscale on a fine time
# grid. The identity below is what no single determinant can absorb.

# log|det A| for the AR1 state-space transform f = A z, plus the (2 pi)
# normalizers the centered density carries and the non-centered one does not.
# Written here rather than read off the engine: the arbiter has to be
# independent of the code under test.
tgp_log_det_shift <- function(times, n_groups, sigma2, phi) {
  T_times <- length(times)
  dt <- diff(times)
  rho <- exp(-dt / phi)
  omr2 <- pmax(1 - rho^2, 1e-10)          # kAr1StationaryFloor
  a <- sqrt(sigma2) * sqrt(omr2)
  n_groups * (0.5 * T_times * log(2 * pi) + 0.5 * log(sigma2) + sum(log(a)))
}

tgp_pair <- function(times, z, log_sigma2, logit_phi, n_groups = 1L,
                     phi_lower = 0.01, phi_upper = 10) {
  nc <- cpp_test_temporal_gp_density(times, n_groups, z, log_sigma2, logit_phi,
                                     parameterization = 1L,
                                     phi_lower = phi_lower,
                                     phi_upper = phi_upper)
  ce <- cpp_test_temporal_gp_density(times, n_groups, nc$field, log_sigma2,
                                     logit_phi, parameterization = 0L,
                                     phi_lower = phi_lower,
                                     phi_upper = phi_upper)
  list(nc = nc, centered = ce)
}

test_that("the two parameterizations differ by the transform's log-determinant", {
  set.seed(4L)
  times <- sort(runif(8L, 0, 5))
  z <- rnorm(8L)
  p <- tgp_pair(times, z, log_sigma2 = log(0.7), logit_phi = 0.4)

  shift <- tgp_log_det_shift(times, 1L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-10)
})

test_that("the identity holds where the (1 - rho^2) floor binds", {
  # dt / phi = 1e-11, so 1 - rho^2 ~ 2e-11 is below the 1e-10 floor at every
  # step. sigma^2 = 1e-4 is small enough that flooring the product instead of
  # the correlation factor would place the conditional variance six orders of
  # magnitude away.
  times <- c(0, 1e-5, 2e-5, 3e-5)
  z <- c(0.3, -0.8, 0.15, 0.6)
  # bounded_from_logit at logit 0 sits at the midpoint of (lower, upper).
  p <- tgp_pair(times, z, log_sigma2 = log(1e-4), logit_phi = 0,
                phi_lower = 1, phi_upper = 2e6)

  expect_gt(p$nc$phi, 1e5)
  rho <- exp(-diff(times) / p$nc$phi)
  expect_lt(max(1 - rho^2), 1e-10)        # the floor is genuinely binding

  shift <- tgp_log_det_shift(times, 1L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-9)
})

test_that("the identity holds over several independent groups", {
  set.seed(9L)
  times <- sort(runif(5L, 0, 3))
  z <- rnorm(15L)
  p <- tgp_pair(times, z, log_sigma2 = log(2.1), logit_phi = -0.6,
                n_groups = 3L)

  shift <- tgp_log_det_shift(times, 3L, p$nc$sigma2, p$nc$phi)
  expect_equal(p$nc$log_post - p$centered$log_post, shift, tolerance = 1e-10)
})


test_that("the lengthscale starts below the data's own spread, on both doors", {
  skip_on_cran()
  # A bounded lengthscale used to start at the midpoint of its (0.01, 10)
  # support. Both GP doors standardize their time values, so that is a
  # lengthscale five times the data's spread: the dense T x T covariance is
  # then numerically rank-one and its Cholesky jitter binds, and the runtime
  # gradient check deviates on the lengthscale by an amount ordered by kernel
  # smoothness -- a floor binding, not a wrong derivative (gcol33/tulpa#851).
  # It starts at 0.2 * sd(time) now, and no arm falls back.
  set.seed(4)
  n_t <- 20L; reps <- 8L
  tt <- sort(cumsum(stats::rexp(n_t, rate = 1 / 4)))
  tt <- (tt - min(tt)) / diff(range(tt)) * 100
  f <- as.numeric(scale(sin(2 * pi * tt / 70)))
  day <- rep(tt, each = reps); idx <- rep(seq_len(n_t), each = reps)
  x <- stats::rnorm(length(day))
  d <- data.frame(day = day, x = x,
                  y = stats::rpois(length(day), exp(0.4 + 0.8 * x + f[idx])))

  fell_back <- function(temporal) {
    msgs <- character(0)
    withCallingHandlers(
      tulpa(y ~ x, data = d, family = "poisson", temporal = temporal,
            mode = "exact",
            control = list(n_iter = 20L, n_warmup = 10L, seed = 3L,
                           n_chains = 1L)),
      warning = function(w) {
        msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning")
      })
    any(grepl("Gradient mismatch", msgs))
  }

  for (cv in list(c("matern", "2.5"), c("gaussian", "1.5"))) {
    for (par in c("noncentered", "centered")) {
      expect_false(
        fell_back(temporal_gp("day", cov = cv[1], nu = as.numeric(cv[2]),
                              parameterization = par)),
        label = paste0("gradient fallback for ", cv[1], " ", par))
    }
  }
})
