# The linear predictor of a continuous-spatial-field fit at the training rows
# and at new locations (R/spatial_field_predict.R): one map from the fit's
# latent mixture to eta, read by predict(), posterior_linpred() and
# posterior_predict().

.sfp_data <- function(seed, n = 160L, m = 40L) {
  set.seed(seed)
  lon <- runif(n + m); lat <- runif(n + m)
  K <- exp(-as.matrix(dist(cbind(lon, lat))) / 0.25) * 0.6
  w <- as.numeric(t(chol(K + diag(1e-8, n + m))) %*% rnorm(n + m))
  x <- rnorm(n + m)
  eta <- 0.4 + 0.5 * x + w
  d <- data.frame(lon, lat, x, eta, y = rpois(n + m, exp(eta)),
                  g = factor(sample(1:8, n + m, TRUE)))
  list(obs = d[seq_len(n), ], out = d[n + seq_len(m), ])
}

test_that("each outer cell's mode maps to the eta the inner solve assembled", {
  skip_on_cran()
  dd <- .sfp_data(31)
  specs <- list(spatial_gp(~ lon + lat, nn = 10L),
                spatial_gp(~ lon + lat, approx = "hsgp", m = 6))
  for (sp in specs) {
    for (f in list(y ~ x, y ~ x + (1 | g))) {
      fit <- tulpa(f, data = dd$obs, family = "poisson", spatial = sp,
                   control = list(progress = FALSE))
      src <- tulpa:::.field_source(fit)
      expect_false(is.null(src))
      on <- which(fit$weights > 0)
      E <- tulpa:::.field_eta_rows(
        fit, src, tulpa:::.tulpa_designs(fit, NULL, "test"), NULL,
        src$modes[on, , drop = FALSE], on, draw = FALSE,
        des = fit$re_design, sel = tulpa:::.re_form_select(fit, NULL))
      expect_lt(max(abs(E - fit$fitted_eta[on, ])), 1e-10)
    }
  }
})

test_that("predict(), posterior_linpred() and posterior_predict() share the field at newdata", {
  skip_on_cran()
  dd <- .sfp_data(32)
  mesh <- spatial_spde(~ lon + lat, data = dd$obs, max_edge = c(0.1, 0.4),
                       cutoff = 0.03)
  specs <- list(spde = mesh, nngp = spatial_gp(~ lon + lat, nn = 10L),
                hsgp = spatial_gp(~ lon + lat, approx = "hsgp", m = 6))
  for (nm in names(specs)) {
    fit <- tulpa(y ~ x, data = dd$obs, family = "poisson", spatial = specs[[nm]])
    p   <- predict(fit, newdata = dd$out, se.fit = TRUE)
    pf  <- predict(fit, newdata = dd$out, include_field = FALSE)
    lp  <- posterior_linpred(fit, newdata = dd$out, summary = TRUE,
                             ndraws = 4000, seed = 1)
    # The same posterior, read twice: means within Monte Carlo error of the
    # SD, and the same SD.
    expect_lt(max(abs(p$fit - lp$estimate) / p$se.fit), 0.15, label = nm)
    expect_lt(max(abs(lp$sd / p$se.fit - 1)), 0.1, label = nm)
    # The field is in the draws: their mean is not the fixed-effect prediction.
    expect_gt(max(abs(lp$estimate - pf)), 0.2, label = nm)
    expect_gt(cor(lp$estimate, dd$out$eta), cor(pf, dd$out$eta), label = nm)
    # Counts drawn at newdata follow the field.
    yr <- posterior_predict(fit, newdata = dd$out, ndraws = 400, seed = 2)
    expect_gt(cor(log1p(colMeans(yr)), p$fit), 0.9, label = nm)
    # At the training coordinates as newdata the SPDE projector and the HSGP
    # basis are the fitted ones, so the point prediction is the in-sample one.
    if (nm != "nngp") {
      expect_equal(predict(fit, newdata = dd$obs), predict(fit),
                   tolerance = 1e-8, label = nm)
    }
  }
})

test_that("the NNGP field at a new location carries its conditional variance", {
  skip_on_cran()
  dd <- .sfp_data(33)
  fit <- tulpa(y ~ x, data = dd$obs, family = "poisson",
               spatial = spatial_gp(~ lon + lat, nn = 10L, scale_coords = FALSE),
               control = list(progress = FALSE))
  src <- tulpa:::.field_source(fit)
  on <- which(fit$weights > 0)
  w <- fit$weights[on] / sum(fit$weights[on])
  s2 <- tulpa:::.field_theta(src$theta, "sigma2")[on]
  # Far from every fitted location the predictive is the prior: the weighted
  # mixture of N(0, sigma2_k) over the cells.
  far <- data.frame(lon = 50, lat = 50, x = 0)
  eta_far <- as.numeric(posterior_linpred(fit, newdata = far, ndraws = 8000,
                                          seed = 3))
  # The draws' spread is the prior's plus the intercept's posterior variance.
  expect_equal(stats::var(eta_far), sum(w * s2) + vcov(fit)[1, 1],
               tolerance = 0.12)
  # On a fitted location the field is the fitted one: the draw's spread is
  # the posterior's, far below the prior variance.
  on_fit <- dd$obs[1, c("lon", "lat", "x")]
  p_on <- predict(fit, newdata = on_fit, se.fit = TRUE)
  expect_lt(p_on$se.fit^2, 0.5 * sum(w * s2))
})

test_that("held-out 95% intervals cover the simulated log mean", {
  skip_if_not_slow()
  hits <- list(nngp = NULL, spde = NULL)
  for (s in 41:43) {
    dd <- .sfp_data(s)
    mesh <- spatial_spde(~ lon + lat, data = dd$obs, max_edge = c(0.05, 0.3),
                         cutoff = 0.02)
    fits <- list(nngp = tulpa(y ~ x, data = dd$obs, family = "poisson",
                              spatial = spatial_gp(~ lon + lat),
                              control = list(progress = FALSE)),
                 spde = tulpa(y ~ x, data = dd$obs, family = "poisson",
                              spatial = mesh))
    for (nm in names(fits)) {
      p <- predict(fits[[nm]], newdata = dd$out, se.fit = TRUE)
      hits[[nm]] <- c(hits[[nm]], p$lower <= dd$out$eta & dd$out$eta <= p$upper)
    }
  }
  for (nm in names(hits)) expect_gt(mean(hits[[nm]]), 0.88, label = nm)
})

test_that("a fit at fixed SPDE hyperparameters is one cell, solved again exactly", {
  skip_on_cran()
  dd <- .sfp_data(34)
  mesh <- spatial_spde(~ lon + lat, data = dd$obs, max_edge = c(0.1, 0.4),
                       cutoff = 0.03)
  fit <- tulpa(y ~ x, data = dd$obs, family = "poisson", spatial = mesh,
               mode = "laplace")
  src <- tulpa:::.field_source(fit)
  expect_equal(nrow(src$modes), 1L)
  sol <- tulpa:::.field_cell_solve(fit, src, 1L)
  expect_lt(max(abs(sol$modes - src$modes)), 1e-5)
  p  <- predict(fit, newdata = dd$out, se.fit = TRUE)
  lp <- posterior_linpred(fit, newdata = dd$out, summary = TRUE, ndraws = 4000,
                          seed = 1)
  expect_lt(max(abs(p$fit - lp$estimate) / p$se.fit), 0.15)
})
