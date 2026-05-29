# Temporal RW1 routed through tulpa(): an RW1 random walk is an intrinsic CAR on
# a 1D chain, so it integrates through the nested-Laplace areal (ICAR) path. These
# tests guard the front-door wiring and the fixed-effect recovery.

make_temporal_data <- function(seed, Tt = 24L, m = 25L, b1 = 0.9) {
  set.seed(seed)
  time <- rep(seq_len(Tt), each = m)
  ftrue <- as.numeric(scale(sin(2 * pi * seq_len(Tt) / Tt)))
  x <- rnorm(Tt * m)
  list(df = data.frame(x = x, time = time), time = time, ftrue = ftrue, b1 = b1)
}

test_that("temporal_rw1 routes through tulpa() and recovers the fixed slope", {
  d <- make_temporal_data(11)
  d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
  fit <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_rw1("time"), mode = "auto")

  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "nested_laplace")
  expect_match(fit$selection_reason, "temporal")

  cf <- coef(fit)
  expect_named(cf, c("(Intercept)", "x"))
  expect_lt(abs(cf[["x"]] - d$b1), 0.15)
})

test_that("temporal_rw1 works for a gaussian response", {
  d <- make_temporal_data(12)
  d$df$y <- -0.2 + d$b1 * d$df$x + d$ftrue[d$time] + rnorm(nrow(d$df), 0, 0.7)
  fit <- tulpa(y ~ x, data = d$df, family = "gaussian",
               temporal = temporal_rw1("time"), mode = "auto", phi = 0.7)
  expect_lt(abs(coef(fit)[["x"]] - d$b1), 0.15)
})

test_that("nested temporal fit reports a marginalized fixed-effect SE", {
  d <- make_temporal_data(14)
  d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
  fit <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_rw1("time"), mode = "auto")

  sm <- summary(fit)
  expect_true(is.finite(sm["x", "std.error"]))   # slope SE from grid marginalization
  ci <- confint(fit)["x", ]
  expect_lt(ci[1], d$b1)
  expect_gt(ci[2], d$b1)

  # se.fit prediction now works (uses the marginalized vcov)
  pr <- predict(fit, newdata = data.frame(x = c(-1, 1)), se.fit = TRUE)
  expect_true(all(is.finite(pr$se.fit)))
})

test_that("nested-fit slope CI covers the truth at roughly nominal rate", {
  hits <- 0L
  for (i in seq_len(15)) {
    d <- make_temporal_data(200 + i)
    d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
    fit <- tulpa(y ~ x, data = d$df, family = "binomial",
                 temporal = temporal_rw1("time"), mode = "auto")
    ci <- confint(fit)["x", ]
    if (ci[1] <= d$b1 && d$b1 <= ci[2]) hits <- hits + 1L
  }
  expect_gte(hits, 12L)
})

test_that("logLik and compare_models return one scalar row per nested fit", {
  d <- make_temporal_data(15)
  d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
  f_t <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_rw1("time"), mode = "auto")
  f_0 <- tulpa(y ~ x, data = d$df, family = "binomial", mode = "laplace")

  expect_length(as.numeric(logLik(f_t)), 1L)   # integrated, not per-grid
  expect_true(is.finite(as.numeric(logLik(f_t))))
  cmp <- compare_models(temporal = f_t, plain = f_0, criterion = "loglik")
  expect_equal(nrow(cmp), 2L)                   # one row per model, not per grid
})

test_that("unsupported temporal configurations error clearly", {
  d <- make_temporal_data(13)
  d$df$y <- rbinom(nrow(d$df), 1, 0.5)

  expect_error(
    tulpa(y ~ x, data = d$df, family = "binomial", temporal = temporal_rw2("time")),
    "Only temporal_rw1"
  )
  expect_error(
    tulpa(y ~ x, data = d$df, family = "binomial",
          temporal = temporal_rw1("time", group_var = "x")),
    "panel"
  )
  K <- 6L; W <- matrix(0, K, K)
  for (i in seq_len(K - 1L)) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  d$df$region <- factor(sample(seq_len(K), nrow(d$df), replace = TRUE))
  expect_error(
    tulpa(y ~ x + spatial(region), data = d$df, family = "binomial",
          temporal = temporal_rw1("time"), spatial = list(type = "icar", adjacency = W)),
    "temporal field together with a spatial field"
  )
})
