# Contract + recovery tests for the generic tulpa_fit S3 methods. Guards the
# fix that made coef/summary/confint/vcov/tidy/ranef work on the Laplace tier
# (which carries $mode/$H_beta, not $draws) and report named fixed effects.

make_re_data <- function(seed, n = 400L, b0 = 0.5, b1 = 1.2,
                         sigma_re = 0.6, sigma_y = 0.7, n_groups = 12L) {
  set.seed(seed)
  x <- rnorm(n)
  g <- factor(sample(seq_len(n_groups), n, replace = TRUE))
  u <- rnorm(n_groups, 0, sigma_re)
  list(
    df = data.frame(
      y = b0 + b1 * x + u[g] + rnorm(n, 0, sigma_y),
      x = x, g = g
    ),
    b0 = b0, b1 = b1, n_groups = n_groups, u = u
  )
}

test_that("Laplace fit supports the full S3 method contract", {
  d <- make_re_data(42)
  fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "gaussian",
               mode = "laplace", sigma_re = 0.6, phi = 0.7)

  cf <- coef(fit)
  expect_named(cf, c("(Intercept)", "x"))
  expect_length(cf, 2L)

  sm <- summary(fit)
  expect_equal(nrow(sm), 2L)
  expect_true(all(is.finite(sm$std.error)))
  expect_equal(rownames(sm), c("(Intercept)", "x"))

  ci <- confint(fit)
  expect_equal(dim(ci), c(2L, 2L))
  expect_true(all(ci[, 1] < ci[, 2]))

  V <- vcov(fit)
  expect_equal(dim(V), c(2L, 2L))
  expect_equal(rownames(V), c("(Intercept)", "x"))
  expect_true(all(diag(V) > 0))

  td <- tidy(fit)
  expect_setequal(names(td),
                  c("term", "estimate", "std.error", "conf.low", "conf.high"))

  re <- ranef(fit)
  expect_equal(nrow(re), d$n_groups)
  expect_true(all(grepl("^g\\[", re$term)))

  gl <- glance(fit)
  expect_equal(gl$n_fixed, 2L)
  expect_true(isTRUE(gl$converged))
})

test_that("Laplace fit recovers the fixed-effect slope", {
  # The intercept absorbs the sample mean of the random intercepts, so test the
  # slope (clean) and average the intercept over several RE realizations.
  slopes <- numeric(5); intercepts <- numeric(5)
  for (i in seq_len(5)) {
    d <- make_re_data(100 + i)
    fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "gaussian",
                 mode = "laplace", sigma_re = 0.6, phi = 0.7)
    cf <- coef(fit)
    slopes[i] <- cf[["x"]]
    intercepts[i] <- cf[["(Intercept)"]] - mean(d$u)  # remove RE-mean offset
  }
  expect_lt(abs(mean(slopes) - 1.2), 0.1)
  expect_lt(abs(mean(intercepts) - 0.5), 0.15)
})

test_that("Laplace credible intervals cover the truth at roughly nominal rate", {
  hits_slope <- 0L
  n_rep <- 20L
  for (i in seq_len(n_rep)) {
    d <- make_re_data(200 + i)
    fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "gaussian",
                 mode = "laplace", sigma_re = 0.6, phi = 0.7)
    ci <- confint(fit)["x", ]
    if (ci[1] <= 1.2 && 1.2 <= ci[2]) hits_slope <- hits_slope + 1L
  }
  # 95% nominal; allow slack for finite reps and conditioning on sigma_re.
  expect_gte(hits_slope, 16L)
})

test_that("predict and fitted work on a Laplace fit", {
  d <- make_re_data(11)
  fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "gaussian",
               mode = "laplace", sigma_re = 0.6, phi = 0.7)

  # In-sample fitted values
  fv <- fitted(fit)
  expect_length(fv, nrow(d$df))

  # Predict at new covariates, link scale, with intervals
  nd <- data.frame(x = c(-1, 0, 1))
  p_link <- predict(fit, newdata = nd)
  expect_length(p_link, 3L)
  # gaussian: response == link
  expect_equal(predict(fit, newdata = nd, type = "response"), p_link)

  ci <- predict(fit, newdata = nd, se.fit = TRUE, type = "link")
  expect_setequal(names(ci), c("fit", "se.fit", "lower", "upper"))
  expect_true(all(ci$lower < ci$upper))
  # prediction tracks the slope
  expect_gt(p_link[3], p_link[1])
})

test_that("predict response scale respects the family range (binomial)", {
  d <- make_re_data(13)
  eta <- -0.3 + 1.0 * d$df$x + d$u[d$df$g]
  d$df$y <- rbinom(nrow(d$df), 1, plogis(eta))
  fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "binomial",
               mode = "laplace", sigma_re = 0.6)
  nd <- data.frame(x = seq(-2, 2, by = 1))
  pr <- predict(fit, newdata = nd, type = "response", se.fit = TRUE)
  expect_true(all(pr$fit > 0 & pr$fit < 1))
  expect_true(all(pr$lower >= 0 & pr$upper <= 1))
})

test_that("mode = 'auto' selects an R-reachable backend on a plain model", {
  d <- make_re_data(23)
  # Plain (no spatial/latent/temporal) small model: auto must not resolve to a
  # C-ABI-only backend that errors at dispatch.
  fit <- tulpa(y ~ x, data = d$df, family = "gaussian", mode = "auto", phi = 0.8,
               control = list(n_iter = 800L, warmup = 400L))
  expect_s3_class(fit, "tulpa_fit")
  expect_true(backend_is_reachable(fit$backend))
  expect_named(coef(fit), c("(Intercept)", "x"))
})

test_that("Gibbs fit (draws in $beta/$re) supports the contract", {
  d <- make_re_data(17)
  eta <- -0.3 + 1.0 * d$df$x + d$u[d$df$g]
  d$df$y <- rbinom(nrow(d$df), 1, plogis(eta))
  fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "binomial",
               mode = "gibbs", control = list(iter = 800L, warmup = 400L))

  cf <- coef(fit)
  expect_named(cf, c("(Intercept)", "x"))
  expect_length(cf, 2L)
  expect_equal(nrow(summary(fit)), 2L)
  expect_true(all(is.finite(summary(fit)$std.error)))
  expect_equal(dim(vcov(fit)), c(2L, 2L))
  expect_gt(nrow(ranef(fit)), 0L)
  # predict reuses coef()/vcov(), so it must work here too
  expect_length(predict(fit, newdata = data.frame(x = c(-1, 1))), 2L)
})

test_that("sampler fit reports named fixed effects and separate random effects", {
  d <- make_re_data(7)
  eta <- -0.3 + 1.0 * d$df$x + d$u[d$df$g]
  d$df$y <- rbinom(nrow(d$df), 1, plogis(eta))
  fit <- tulpa(y ~ x + (1 | g), data = d$df, family = "binomial",
               mode = "mala", sigma_re = 0.6,
               control = list(n_iter = 1500L, warmup = 500L))

  cf <- coef(fit)
  expect_named(cf, c("(Intercept)", "x"))
  expect_length(cf, 2L)             # fixed effects only

  expect_equal(nrow(summary(fit)), 2L)
  expect_equal(nrow(ranef(fit)), d$n_groups)
  expect_setequal(names(tidy(fit)),
                  c("term", "estimate", "std.error", "conf.low", "conf.high"))
})
