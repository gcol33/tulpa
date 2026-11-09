# A one-trial binomial response weakly identifies the varying slope's field.
# The marginal standard deviation (SD) has a penalized-complexity (PC) prior
# with P(sigma > 1) = 0.5 and median one.
# In centered coordinates, the conditional field precision scales as 1/sigma2;
# the non-centered coordinates have a unit normal prior at every field scale.
# This coupling can affect mixing even when both parameterizations target the
# same posterior, as checked by test-svc-parameterization-equivalence.R.
#
# The field-spread bands describe the pinned Bernoulli data and sampler budget.
# The Gaussian fixtures below check intervals against a simulated marginal SD.
# Variance convergence is checked separately with 1000 retained draws per chain.
# Short-budget mixing cost is bulk effective sample size (ESS) divided by
# tulpa_profile()'s gradient count, including warmup and step-size search,
# for the field's sampled log variance.

skip_on_cran()

sim_svc_bernoulli <- function(n = 150L, sigma2 = 1.0, phi = 0.30,
                              a0 = 0.0, x_sd = 0.6, seed = 1L) {
  set.seed(seed)
  lon <- runif(n)
  lat <- runif(n)
  D <- as.matrix(dist(cbind(lon, lat)))
  K <- sigma2 * exp(-D / phi)
  L <- chol(K + diag(1e-8, n))
  w <- as.numeric(t(L) %*% rnorm(n))
  x <- rnorm(n, sd = x_sd)
  y <- rbinom(n, 1, plogis(a0 + (1 + w) * x))
  data.frame(lon = lon, lat = lat, x = x, y = y, w_true = w)
}

.SVC_AMP_ITER <- 500L
.SVC_AMP_WARMUP <- 400L
.SVC_AMP_LONG_ITER <- 2500L
.SVC_AMP_LONG_WARMUP <- 1500L

.svc_amp_fits <- new.env(parent = emptyenv())

fit_svc_amp <- function(d, parameterization = "noncentered",
                        n_iter = .SVC_AMP_ITER, n_warmup = .SVC_AMP_WARMUP) {
  tulpa(y ~ x, data = d, family = "binomial",
        spatial = spatial_svc(~ lon + lat, terms = ~ x - 1, nn = 10L,
                              parameterization = parameterization,
                              sigma_prior_U = 1, sigma_prior_alpha = 0.5),
        mode = "exact",
        control = list(n_iter = n_iter, n_warmup = n_warmup, seed = 7L))
}

svc_amp_fit <- function(parameterization) {
  if (is.null(.svc_amp_fits[[parameterization]])) {
    d <- sim_svc_bernoulli(n = 150L, seed = 1L)
    profile <- tulpa_profile(fit_svc_amp(d, parameterization))
    .svc_amp_fits[[parameterization]] <-
      list(d = d, fit = attr(profile, "value"), profile = profile)
  }
  .svc_amp_fits[[parameterization]]
}

svc_sd_ratio <- function(fit, w_true) {
  wcol <- grep("^svc_w\\[", colnames(fit$draws))
  sd(colMeans(fit$draws[, wcol, drop = FALSE])) / sd(w_true)
}

svc_sigma2_rhat <- function(fit) {
  d <- diagnostics(fit, pars = "log_sigma2_svc[1]", measures = "rhat")
  d$rhat[[1L]]
}

svc_sigma2_ess_per_gradient <- function(arm) {
  d <- diagnostics(arm$fit, pars = "log_sigma2_svc[1]", measures = "ess_bulk")
  gradients <- arm$profile$calls[arm$profile$phase == "gradient"]
  stopifnot(length(gradients) == 1L, is.finite(gradients), gradients > 0)
  d$ess_bulk[[1L]] / gradients
}

test_that("non-centered SVC NUTS retains field spread on the pinned Bernoulli data", {
  skip_if_not_slow()
  nc <- svc_amp_fit("noncentered")   # the default
  d <- nc$d
  fit <- nc$fit

  expect_equal(length(fit$divergent) / length(unique(fit$chain_id)),
               .SVC_AMP_ITER - .SVC_AMP_WARMUP)

  ratio <- svc_sd_ratio(fit, d$w_true)
  expect_gt(ratio, 0.55)
  expect_lt(ratio, 1.8)
  expect_lte(mean(fit$divergent), 0.05)
})

test_that("non-centered SVC NUTS mixes in the field's variance", {
  skip_if_not_slow()
  d <- sim_svc_bernoulli(n = 150L, seed = 1L)
  fit <- fit_svc_amp(d, n_iter = .SVC_AMP_LONG_ITER,
                   n_warmup = .SVC_AMP_LONG_WARMUP)
  expect_equal(length(fit$divergent) / length(unique(fit$chain_id)), 1000L)
  expect_lt(svc_sigma2_rhat(fit), 1.05)
})

test_that("centered SVC NUTS retains field spread on the pinned Bernoulli data", {
  skip_if_not_slow()
  ce <- svc_amp_fit("centered")
  d <- ce$d
  fit <- ce$fit

  expect_equal(length(fit$divergent) / length(unique(fit$chain_id)),
               .SVC_AMP_ITER - .SVC_AMP_WARMUP)

  ratio <- svc_sd_ratio(fit, d$w_true)
  expect_gt(ratio, 0.55)
  expect_lt(ratio, 2.2)
  expect_lte(mean(fit$divergent), 0.05)
})

test_that("non-centered SVC NUTS has higher variance ESS per gradient on weak data", {
  skip_if_not_slow()
  eff_c <- svc_sigma2_ess_per_gradient(svc_amp_fit("centered"))
  eff_nc <- svc_sigma2_ess_per_gradient(svc_amp_fit("noncentered"))
  expect_gt(eff_nc, eff_c,
            label = sprintf("sigma2 bulk ESS per gradient: non-centered %.6g, centered %.6g",
                            eff_nc, eff_c))
})

svc_sd_recovery_data <- function(approx) {
  d <- sim_svc_bernoulli(sigma2 = 0.25, seed = 1L)
  if (approx == "hsgp") {
    cm <- as.matrix(d[, c("lon", "lat")])
    dom <- cpp_hsgp_domain_2d(cm, 1.5)
    basis <- cpp_hsgp_basis_2d_domain(cm, 8L, dom)
    set.seed(55)
    d$w_true <- as.numeric(basis$phi_basis %*%
      (cpp_hsgp_sqrt_spectral(basis$lambda_eig, 0.25, 0.3,
        tulpa:::.hsgp_kernel_code("matern32")) * rnorm(64L)))
  }
  set.seed(54)
  d$y <- 0.2 + (0.8 + d$w_true) * d$x + rnorm(nrow(d), sd = 0.15)
  d
}

test_that("SVC marginal SD intervals cover a simulated amplitude on both bases", {
  skip_if_not_slow()
  for (approx in c("nngp", "hsgp")) {
    d <- svc_sd_recovery_data(approx)
    fit <- tulpa(y ~ x, d, family = "gaussian", phi = 0.15^2,
      spatial = spatial_svc(~lon+lat, terms = ~x-1, approx = approx,
        nn = 10L, m = 8L, c_boundary = 1.5, scale_coords = FALSE),
      mode = "exact", control = list(n_iter = 1500L, n_warmup = 750L,
                                     n_chains = 2L, seed = 7L))
    sigma <- exp(0.5 * fit$draws[, "log_sigma2_svc[1]"])
    ci <- quantile(sigma, c(.025, .975))
    expect_lt(ci[1], 0.5, label = approx)
    expect_gt(ci[2], 0.5, label = approx)
    expect_lte(mean(fit$divergent), 0.05, label = approx)
    expect_lt(svc_sigma2_rhat(fit), 1.05, label = approx)
    pred <- posterior_predict(fit, ndraws = 12L, seed = 18L)
    expect_equal(dim(pred), c(12L, nrow(d)))
    expect_equal(as.matrix(simulate(fit, nsim = 12L, seed = 18L)),
      t(pred), ignore_attr = TRUE)
  }
})
