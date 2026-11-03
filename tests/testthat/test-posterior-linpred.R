# posterior_linpred() is the in-sample linear predictor with every fitted
# component, on the link scale: on a nested-Laplace fit the exact grid mixture
# of the per-cell Gaussians, elsewhere a summary of the draws.

test_that("posterior_linpred() reads the nested grid mixture with the fields", {
  skip_if_fast()
  set.seed(1)
  S <- 9L; n_t <- 6L
  xy <- expand.grid(col = 1:3, row = 1:3)
  W <- 1 * (as.matrix(stats::dist(xy, method = "manhattan")) == 1)
  u <- as.numeric(scale(xy$col + xy$row)) * 0.7
  v <- as.numeric(scale(sin(seq_len(n_t)))) * 0.5
  d <- expand.grid(region = seq_len(S), time = seq_len(n_t), rep = 1:3)
  d$x <- stats::rnorm(nrow(d))
  d$y <- stats::rpois(nrow(d), exp(1 + 0.3 * d$x + u[d$region] + v[d$time]))
  fit <- suppressWarnings(tulpa(y ~ x + spatial(region), data = d,
    family = "poisson", spatial = list(type = "icar", adjacency = W),
    temporal = temporal_rw1("time"), mode = "auto",
    control = list(n_threads = 1L, progress = FALSE)))
  pl <- posterior_linpred(fit, summary = TRUE)
  expect_identical(nrow(pl), nrow(d))
  expect_true(all(pl$source == "mixture"))
  w <- fit$weights; keep <- w > 0; w <- w[keep] / sum(w[keep])
  M <- fit$fitted_eta[keep, , drop = FALSE]
  V <- fit$fitted_eta_var[keep, , drop = FALSE]
  m <- as.numeric(crossprod(w, M))
  expect_equal(pl$estimate, m, tolerance = 1e-10)
  expect_equal(pl$sd, sqrt(as.numeric(crossprod(w, V + M^2)) - m^2),
               tolerance = 1e-10)
  # The fields are in it, as they are in predict(); without them the
  # prediction is the population level.
  expect_equal(predict(fit, type = "link"), m, tolerance = 1e-10)
  pop <- predict(fit, type = "link", include_field = FALSE)
  expect_gt(stats::sd(pl$estimate - pop), 0.3)
  expect_gt(stats::cor(pl$estimate - pop, u[d$region] + v[d$time]), 0.9)
  dr <- posterior_linpred(fit, ndraws = 50, seed = 3)
  expect_identical(dim(dr), c(50L, nrow(d)))
  expect_null(attr(dr, "cells"))
})

test_that("posterior_linpred() summarizes draws on a conditional fit", {
  set.seed(2)
  d <- data.frame(y = stats::rpois(60, 3), x = stats::rnorm(60))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  pl <- posterior_linpred(fit, summary = TRUE, ndraws = 200, seed = 1)
  expect_true(all(pl$source == "sampled"))
  expect_equal(pl$estimate, predict(fit, type = "link"), tolerance = 0.05)
  expect_true(all(pl$conf.low < pl$estimate & pl$estimate < pl$conf.high))
})
