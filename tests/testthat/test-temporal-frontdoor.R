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
  skip_on_cran()
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
  skip_on_cran()
  d <- make_temporal_data(12)
  d$df$y <- -0.2 + d$b1 * d$df$x + d$ftrue[d$time] + rnorm(nrow(d$df), 0, 0.7)
  fit <- tulpa(y ~ x, data = d$df, family = "gaussian",
               temporal = temporal_rw1("time"), mode = "auto", phi = 0.7)
  expect_lt(abs(coef(fit)[["x"]] - d$b1), 0.15)
})

test_that("nested temporal fit reports a marginalized fixed-effect SE", {
  skip_on_cran()
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
  skip_if_not_slow()
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
  skip_on_cran()
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

# RW2 / AR1 route through the native nested-Laplace temporal kernel (issue #73).
test_that("temporal_rw2 routes through tulpa() and recovers the fixed slope", {
  skip_on_cran()
  d <- make_temporal_data(21)
  d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
  fit <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_rw2("time"), mode = "auto")
  expect_equal(fit$backend, "nested_laplace")
  expect_match(fit$selection_reason, "rw2")
  expect_lt(abs(coef(fit)[["x"]] - d$b1), 0.15)
})

test_that("temporal_ar1 routes through tulpa() and recovers the fixed slope", {
  skip_on_cran()
  d <- make_temporal_data(31)
  d$df$y <- rbinom(nrow(d$df), 1, plogis(-0.2 + d$b1 * d$df$x + d$ftrue[d$time]))
  fit <- tulpa(y ~ x, data = d$df, family = "binomial",
               temporal = temporal_ar1("time"), mode = "auto")
  expect_equal(fit$backend, "nested_laplace")
  expect_match(fit$selection_reason, "ar1")
  expect_lt(abs(coef(fit)[["x"]] - d$b1), 0.15)
})

# Additive areal spatial + temporal (space-time) routes through the joint
# multi-block nested-Laplace path (issue #73).
test_that("areal spatial + temporal rw1 routes through the joint nested path", {
  skip_on_cran()
  set.seed(7)
  K <- 12L; Tt <- 10L; m <- 6L
  W <- matrix(0, K, K)
  for (i in seq_len(K - 1L)) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  Qs <- diag(rowSums(W)) - W
  eig <- eigen(Qs, symmetric = TRUE)
  phi_s <- as.numeric(eig$vectors[, 1:(K - 1)] %*%
                      (rnorm(K - 1) / sqrt(eig$values[1:(K - 1)])))
  phi_s <- phi_s - mean(phi_s)
  ftrue <- as.numeric(scale(sin(2 * pi * seq_len(Tt) / Tt)))
  grid <- expand.grid(region = seq_len(K), time = seq_len(Tt))
  grid <- grid[rep(seq_len(nrow(grid)), each = m), ]
  grid$x <- rnorm(nrow(grid))
  grid$y <- rbinom(nrow(grid), 1,
                   plogis(-0.2 + 0.9 * grid$x + 0.7 * phi_s[grid$region] +
                          ftrue[grid$time]))
  # The 9 x 9 (tau_spatial x tau_temporal) Cartesian grid warns it is > 50 cells
  # (CCD is a follow-up); not what this test checks.
  fit <- suppressWarnings(
    tulpa(y ~ x + spatial(region), data = grid, family = "binomial",
          spatial = list(type = "icar", adjacency = W),
          temporal = temporal_rw1("time"), mode = "auto"))
  expect_equal(fit$backend, "nested_laplace")
  expect_match(fit$selection_reason, "spatial field \\+ temporal")
  expect_lt(abs(coef(fit)[["x"]] - 0.9), 0.2)
})

# Panel (grouped) temporal: a separate walk per group sharing one tau (issue #73).
test_that("grouped (panel) temporal routes through tulpa()", {
  skip_on_cran()
  set.seed(101)
  G <- 4L; Tt <- 16L; m <- 6L
  shapes <- sapply(seq_len(G), function(g)
    as.numeric(scale(sin(2 * pi * seq_len(Tt) / Tt + g))) * 0.9)
  grid <- expand.grid(time = seq_len(Tt), grp = seq_len(G))
  grid <- grid[rep(seq_len(nrow(grid)), each = m), ]
  grid$grp <- factor(grid$grp); grid$x <- rnorm(nrow(grid))
  grid$y <- rbinom(nrow(grid), 1,
                   plogis(-0.1 + 0.8 * grid$x +
                          shapes[cbind(grid$time, as.integer(grid$grp))]))
  fit <- tulpa(y ~ x, data = grid, family = "binomial",
               temporal = temporal_rw1("time", group_var = "grp"), mode = "auto")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "nested_laplace")
  expect_lt(abs(coef(fit)[["x"]] - 0.8), 0.2)
})

test_that("panel-fit slope CI covers the truth at roughly nominal rate", {
  skip_on_cran()
  b1 <- 0.8; hits <- 0L; nseed <- 12L
  for (i in seq_len(nseed)) {
    set.seed(500 + i)
    G <- 4L; Tt <- 16L; m <- 8L
    shapes <- sapply(seq_len(G), function(g)
      as.numeric(scale(sin(2 * pi * seq_len(Tt) / Tt + g))) * 0.9)
    grid <- expand.grid(time = seq_len(Tt), grp = seq_len(G))
    grid <- grid[rep(seq_len(nrow(grid)), each = m), ]
    grid$grp <- factor(grid$grp); grid$x <- rnorm(nrow(grid))
    grid$y <- rbinom(nrow(grid), 1,
                     plogis(-0.1 + b1 * grid$x +
                            shapes[cbind(grid$time, as.integer(grid$grp))]))
    fit <- tulpa(y ~ x, data = grid, family = "binomial",
                 temporal = temporal_rw1("time", group_var = "grp"), mode = "auto")
    ci <- confint(fit)["x", ]
    if (ci[1] <= b1 && b1 <= ci[2]) hits <- hits + 1L
  }
  expect_gte(hits, 9L)   # ~nominal 95% coverage over 12 panel datasets
})

test_that("grouped temporal with one group is identical to the ungrouped fit", {
  skip_on_cran()
  # n_groups == 1 is the degenerate panel case: the flattened node index reduces
  # to the time index and the grouped C++ path must reproduce the single walk
  # bit-for-bit. Guards the panel generalisation against drift.
  set.seed(303)
  Tt <- 18L; m <- 8L
  df <- data.frame(time = rep(seq_len(Tt), each = m), grp = factor(1L))
  ftrue <- as.numeric(scale(sin(2 * pi * seq_len(Tt) / Tt)))
  df$x <- rnorm(nrow(df))
  df$y <- rbinom(nrow(df), 1, plogis(-0.2 + 0.9 * df$x + ftrue[df$time]))
  f_plain   <- tulpa(y ~ x, data = df, family = "binomial",
                     temporal = temporal_rw1("time"), mode = "auto")
  f_grouped <- tulpa(y ~ x, data = df, family = "binomial",
                     temporal = temporal_rw1("time", group_var = "grp"),
                     mode = "auto")
  expect_equal(as.numeric(logLik(f_grouped)), as.numeric(logLik(f_plain)),
               tolerance = 1e-10)
  expect_equal(unname(coef(f_grouped)), unname(coef(f_plain)), tolerance = 1e-8)
})

test_that("unsupported temporal configurations error clearly", {
  d <- make_temporal_data(13)
  d$df$y <- rbinom(nrow(d$df), 1, 0.5)

  K <- 6L; W <- matrix(0, K, K)
  for (i in seq_len(K - 1L)) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  d$df$region <- factor(sample(seq_len(K), nrow(d$df), replace = TRUE))

  # Panel temporal cannot ride alongside a spatial field (the joint path has no
  # per-group temporal layout yet).
  expect_error(
    tulpa(y ~ x + spatial(region), data = d$df, family = "binomial",
          spatial = list(type = "icar", adjacency = W),
          temporal = temporal_rw1("time", group_var = "region")),
    "panel"
  )
  # A continuous spatial field cannot host a temporal block through the front
  # door yet (only areal space-time is wired).
  d$df$lon <- rnorm(nrow(d$df)); d$df$lat <- rnorm(nrow(d$df))
  expect_error(
    tulpa(y ~ x, data = d$df, family = "binomial",
          spatial = spatial_gp(~ lon + lat),
          temporal = temporal_rw1("time")),
    "areal"
  )
})
