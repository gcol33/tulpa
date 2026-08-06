# temporal_gp() reaches a fitter, and its covariance choice selects a kernel
# (gcol33/tulpa#287, gcol33/tulpa#288).
#
# #287: the constructor was exported and documented with a tulpa() example, but
# tulpa() rejected type = "gp" and nothing populated TemporalGPData -- there was
# no fitter to reach.
#
# #288: cov / nu / period were carried on the spec and read by nothing. The live
# density hardcoded exp(-dt/phi), so every kernel ran as exponential with no
# error or warning. A test asserting the four kernels agree would have passed
# then and hidden it, so these assert they DISAGREE.

sim_tgp <- function(seed = 11L, n_t = 20L, N = 200L) {
  set.seed(seed)
  times <- sort(runif(n_t, 0, 10))
  f <- as.numeric(scale(sin(times * 0.8))) * 0.9
  ti <- sample(n_t, N, TRUE)
  x <- rnorm(N)
  y <- rbinom(N, 1, plogis(0.2 + 0.7 * x + f[ti]))
  list(df = data.frame(y = y, x = x, tvar = times[ti]),
       f = f, n_t = n_t, times = times)
}

fit_tgp <- function(d, cov = "exponential", nu = NULL, period = NULL,
                    parameterization = "noncentered", n_iter = 400L,
                    seed = 3L) {
  args <- list(time_var = "tvar", cov = cov, parameterization = parameterization)
  if (!is.null(nu)) args$nu <- nu
  if (!is.null(period)) args$period <- period
  tulpa(y ~ x, data = d$df, family = "binomial",
        temporal = do.call(temporal_gp, args), mode = "hmc",
        control = list(n_iter = n_iter, warmup = n_iter %/% 2, n_chains = 1,
                       seed = seed, verbose = FALSE))
}

test_that("temporal_gp() fits through tulpa() and names its hyperparameters", {
  skip_on_cran()
  d <- sim_tgp()
  fit <- fit_tgp(d)

  expect_s3_class(fit, "tulpa_fit")
  # The two GP hyperparameters are addressable by name, not as param[k].
  expect_true("log_sigma2_temporal_gp" %in% names(fit$means))
  expect_true("logit_phi_temporal_gp" %in% names(fit$means))
  expect_false(any(grepl("^param\\[", names(fit$means))))
  # One field value per DISTINCT time, not per observation.
  expect_length(grep("^phi_temporal\\[", names(fit$means)), d$n_t)
  expect_true(all(is.finite(fit$means)))
})

test_that("temporal_gp() recovers the fixed effect and the latent curve", {
  skip_on_cran()
  d <- sim_tgp()
  # Centered, so the stored phi_temporal IS the field rather than its z.
  fit <- fit_tgp(d, parameterization = "centered", n_iter = 1200L)

  expect_lt(abs(coef(fit)[["x"]] - 0.7), 0.35)

  f_hat <- fit$means[grep("^phi_temporal\\[", names(fit$means))]
  expect_gt(cor(f_hat - mean(f_hat), d$f - mean(d$f)), 0.7)
})

test_that("each covariance runs its own kernel, not the exponential", {
  skip_on_cran()
  d <- sim_tgp()
  # Same data, same seed, same iterations -- only cov differs. Before #288 all
  # four ran exp(-dt/phi) and these were identical.
  fits <- list(
    exponential = fit_tgp(d, cov = "exponential"),
    matern32    = fit_tgp(d, cov = "matern", nu = 1.5),
    matern52    = fit_tgp(d, cov = "matern", nu = 2.5),
    gaussian    = fit_tgp(d, cov = "gaussian"),
    periodic    = fit_tgp(d, cov = "periodic", period = 3)
  )
  field <- function(f) unname(f$means[grep("^phi_temporal\\[", names(f$means))])

  nms <- names(fits)
  for (i in seq_along(nms)) {
    for (j in seq_len(i - 1L)) {
      expect_false(
        isTRUE(all.equal(field(fits[[i]]), field(fits[[j]]))),
        info = paste(nms[i], "vs", nms[j])
      )
    }
  }
  for (nm in nms) expect_true(all(is.finite(fits[[nm]]$means)), info = nm)
})

test_that("Matern nu = 0.5 is the exponential kernel", {
  skip_on_cran()
  # Not a coincidence to be papered over: Matern at nu = 1/2 IS the exponential
  # covariance, so it takes the same O(T) Markov path and must give the same
  # answer to the bit. If this ever diverges, the two kernels have drifted.
  d <- sim_tgp()
  a <- fit_tgp(d, cov = "exponential")
  b <- fit_tgp(d, cov = "matern", nu = 0.5)
  expect_equal(unname(a$means), unname(b$means), tolerance = 0)
})

test_that("the periodic kernel tracks its period", {
  skip_on_cran()
  d <- sim_tgp()
  a <- fit_tgp(d, cov = "periodic", period = 2)
  b <- fit_tgp(d, cov = "periodic", period = 5)
  fa <- unname(a$means[grep("^phi_temporal\\[", names(a$means))])
  fb <- unname(b$means[grep("^phi_temporal\\[", names(b$means))])
  expect_false(isTRUE(all.equal(fa, fb)))
})

test_that("a Matern smoothness with no closed form is rejected", {
  for (nu in c(0.5, 1.5, 2.5)) {
    expect_s3_class(temporal_gp("t", cov = "matern", nu = nu),
                    "tulpa_temporal_gp")
  }
  for (nu in c(0.75, 1.0, 2.0, 3.0)) {
    expect_error(temporal_gp("t", cov = "matern", nu = nu),
                 "0.5, 1.5 or 2.5")
  }
  # nu is still required to be positive first
  expect_error(temporal_gp("t", cov = "matern", nu = -1), "positive number")
})

test_that("the periodic kernel requires a period", {
  expect_error(temporal_gp("t", cov = "periodic"), "period")
  expect_error(temporal_gp("t", cov = "periodic", period = 0), "period")
  expect_s3_class(temporal_gp("t", cov = "periodic", period = 12),
                  "tulpa_temporal_gp")
})

test_that("a temporal GP cannot yet ride alongside another field", {
  d <- sim_tgp(N = 60L, n_t = 8L)
  d$df$g <- rep(1:4, length.out = nrow(d$df))
  adj <- matrix(0L, 4, 4)
  for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1L
  expect_error(
    tulpa(y ~ x + spatial(g), data = d$df, family = "binomial",
          temporal = temporal_gp("tvar"), mode = "hmc",
          spatial = spatial_car(adj, group_var = "g"),
          control = list(n_iter = 40, warmup = 20, n_chains = 1)),
    "temporal GP"
  )
})
