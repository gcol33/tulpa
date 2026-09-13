# Which quantity logLik() reports and when AIC / BIC apply (gcol33/tulpa#723),
# and the offset on the nested-Laplace route (gcol33/tulpa#726).

test_that("the quantity follows what the fit estimated, not the vector length", {
  skip_on_cran()
  set.seed(3)
  n <- 300L
  time <- sample.int(15L, n, replace = TRUE)
  x <- rnorm(n)
  g <- factor(sample.int(10L, n, replace = TRUE))
  db <- data.frame(y = rbinom(n, 1, plogis(0.2 + 0.6 * x + sin(time / 2))),
                   x = x, time = time, g = g)

  glm_fit <- tulpa(y ~ x, data = db, family = "binomial", mode = "laplace")
  nested  <- tulpa(y ~ x, data = db, family = "binomial",
                   temporal = temporal_rw1("time"))
  re_fixed <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                    mode = "laplace", sigma_re = 0.5)
  eb_fit <- tulpa(y ~ x + (1 | g), data = db, family = "binomial", mode = "eb")

  # Nothing to integrate, something integrated, something supplied: all three
  # are the evidence of the model as specified.
  expect_identical(attr(logLik(glm_fit), "quantity"), "log_evidence")
  expect_identical(attr(logLik(nested), "quantity"), "log_evidence")
  expect_identical(attr(logLik(re_fixed), "quantity"), "log_evidence")
  expect_length(attr(logLik(nested), "conditioned_on"), 0L)

  # An estimate from the same data is what makes the value conditional.
  l_eb <- logLik(eb_fit)
  expect_identical(attr(l_eb, "quantity"), "log_marginal_likelihood")
  expect_identical(attr(l_eb, "conditioned_on"), "re_covariance")

  # So the GLM and the nested model are ranked, and the EB fit is refused.
  cmp <- compare_models(glm = glm_fit, temporal = nested, criterion = "loglik")
  expect_identical(nrow(cmp), 2L)
  expect_error(compare_models(glm = glm_fit, eb = eb_fit, criterion = "loglik"),
               "conditional on re_covariance")
})

test_that("AIC and BIC refuse anything but a maximised log-likelihood", {
  fit <- structure(
    list(log_prob = rep(-100, 5L), means = stats::setNames(rnorm(6L), letters[1:6]),
         N = 50L),
    class = "tulpa_fit")
  expect_error(stats::AIC(fit), "maximised log-likelihood")
  expect_error(stats::BIC(fit), "log posterior mean")

  ml <- structure(list(log_marginal = -120, n_params = 3L, n_fixed = 2L,
                       N = 80L, backend = "agq"),
                  class = "tulpa_fit")
  l <- logLik(ml)
  expect_identical(attr(l, "quantity"), "log_likelihood")
  expect_identical(attr(l, "df"), 3L)
  expect_equal(stats::AIC(ml), 2 * 120 + 2 * 3)
  expect_equal(stats::BIC(ml), 2 * 120 + log(80) * 3)
  expect_error(stats::AIC(ml, fit), "maximised log-likelihood")
})

test_that("a nested evidence records why it declined on a CCD design", {
  fit <- structure(list(log_marginal = c(-10, -11), log_evidence = NA_real_,
                        log_evidence_declined = "moment_rule_design",
                        n_fixed = 2L, N = 50L),
                   class = "tulpa_fit")
  l <- logLik(fit)
  expect_true(is.na(as.numeric(l)))
  expect_identical(attr(l, "declined"), "moment_rule_design")
})

test_that("an offset() reaches the nested-Laplace inner solve", {
  skip_on_cran()
  set.seed(1)
  nT <- 20L; n <- 400L
  time <- sample.int(nT, n, replace = TRUE)
  trend <- cumsum(rnorm(nT, 0, 0.3)); trend <- trend - mean(trend)
  x <- rnorm(n)
  off <- log(runif(n, 1, 3))
  df <- data.frame(y = rpois(n, exp(0.2 + 0.5 * x + trend[time] + off)),
                   x = x, time = time, off = off, zero = 0)

  f_off  <- tulpa(y ~ x + offset(off), data = df, family = "poisson",
                  temporal = temporal_rw1("time"))
  f_none <- tulpa(y ~ x, data = df, family = "poisson",
                  temporal = temporal_rw1("time"))
  f_zero <- tulpa(y ~ x + offset(zero), data = df, family = "poisson",
                  temporal = temporal_rw1("time"))

  # The offset moves the intercept by about its own mean and nothing else.
  expect_equal(coef(f_zero), coef(f_none), tolerance = 1e-10)
  expect_equal(f_zero$log_marginal, f_none$log_marginal, tolerance = 1e-10)
  expect_lt(abs(coef(f_off)[["(Intercept)"]] - 0.2), 0.25)
  expect_gt(coef(f_none)[["(Intercept)"]] - coef(f_off)[["(Intercept)"]], 0.4)

  # The per-cell linear predictor carries it.
  k <- which.max(f_off$weights)
  expect_equal(f_off$fitted_eta[k, ] - f_off$fitted_eta[k, 1L],
               {
                 e <- as.numeric(f_off$modes[k, 1:2] %*% t(cbind(1, x))) + off +
                   f_off$modes[k, 2L + time]
                 e - e[1L]
               }, tolerance = 1e-8)
})

test_that("a constant offset shifts the intercept on every nested driver", {
  # A Poisson log-link offset c is an intercept shift: the mode moves by -c in
  # the intercept and nowhere else, up to the weak fixed-effect ridge.
  skip_on_cran()
  set.seed(2)
  S <- 12L; n <- 360L
  adj <- .pgp_chain_adj(S)
  site <- sample.int(S, n, replace = TRUE)
  time <- sample.int(8L, n, replace = TRUE)
  x <- rnorm(n)
  y <- rpois(n, exp(0.3 + 0.3 * x + sin(site / 2)))
  icar <- list(type = "icar", spatial_idx = as.integer(site),
               n_spatial_units = S, adj_row_ptr = adj$adj_row_ptr,
               adj_col_idx = adj$adj_col_idx, n_neighbors = adj$n_neighbors,
               tau_grid = c(0.5, 2, 8))
  rw1 <- list(type = "rw1", temporal_idx = as.integer(time), n_times = 8L,
              tau_grid = c(1, 10))
  ctl <- list(diagnose_k = FALSE, diagnose_skew = FALSE, auto_recenter = FALSE)
  shift <- function(prior) {
    f0 <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                               family = "poisson", control = ctl)
    f1 <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
                               family = "poisson", offset = rep(0.7, n),
                               control = ctl)
    c(intercept = coef(f1)[[1L]] - coef(f0)[[1L]],
      slope = coef(f1)[[2L]] - coef(f0)[[2L]])
  }
  for (prior in list(icar, list(icar, rw1))) {
    d <- shift(prior)
    expect_equal(d[["intercept"]], -0.7, tolerance = 1e-4)
    expect_lt(abs(d[["slope"]]), 1e-5)
  }
  expect_error(tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = icar,
                                    family = "poisson", offset = rep(0, n - 1L),
                                    control = ctl),
               "offset")

  # The joint driver (HSGP), reached through the front door.
  dd <- data.frame(y = y, x = x, lon = site / S, lat = (site %% 3) / 3,
                   c0 = 0, c1 = 0.7)
  g0 <- tulpa(y ~ x + offset(c0), data = dd, family = "poisson",
              spatial = spatial_gp(coords = ~ lon + lat, approx = "hsgp"),
              mode = "nested_laplace")
  g1 <- tulpa(y ~ x + offset(c1), data = dd, family = "poisson",
              spatial = spatial_gp(coords = ~ lon + lat, approx = "hsgp"),
              mode = "nested_laplace")
  expect_equal(coef(g1)[[1L]] - coef(g0)[[1L]], -0.7, tolerance = 1e-3)
  expect_equal(g1$fitted_eta, g0$fitted_eta, tolerance = 1e-5)
})
