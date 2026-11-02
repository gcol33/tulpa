# fitted() / residuals() read the in-sample linear predictor at one of two
# levels: "conditional" (default, every fitted component) or "population"
# (random effects and latent fields at zero). The structure diagnostics read the
# conditional residuals, the ones the posterior predictive replicates are
# checked against.

.fe_grid_data <- function(seed = 1) {
  set.seed(seed)
  side <- 6L; K <- side^2
  cells <- expand.grid(col = 1:side, row = 1:side)
  W <- matrix(0, K, K)
  for (i in 1:K) for (j in 1:K) {
    if (abs(cells$col[i] - cells$col[j]) + abs(cells$row[i] - cells$row[j]) == 1)
      W[i, j] <- 1
  }
  field <- as.numeric(scale(sin(cells$col / 2) + cos(cells$row / 2))) * 0.6
  n <- 600
  region <- sample(1:K, n, replace = TRUE)
  x <- rnorm(n)
  y <- rpois(n, exp(0.5 + 0.4 * x + field[region]))
  list(d = data.frame(y = y, x = x, region = factor(region)), W = W,
       coords = cbind(cells$col[region], cells$row[region]))
}

test_that("population level is X beta + offset; conditional adds the field", {
  skip_on_cran()
  s <- .fe_grid_data()
  fit <- tulpa(y ~ x + spatial(region), data = s$d, family = "poisson",
               spatial = list(type = "icar", adjacency = s$W),
               mode = "nested_laplace", control = list(n_threads = 1))
  b <- coef(fit)
  expect_equal(log(fitted(fit, effects = "population")),
               as.numeric(b[1] + b[2] * s$d$x), tolerance = 1e-10)
  # On the grid the conditional predictor is the weight-averaged per-cell eta.
  on <- fit$weights > 0
  w <- fit$weights[on] / sum(fit$weights[on])
  expect_equal(log(fitted(fit)),
               as.numeric(colSums(w * fit$fitted_eta[on, , drop = FALSE])),
               tolerance = 1e-10)
  for (e in c("conditional", "population")) {
    expect_equal(residuals(fit, type = "response", effects = e),
                 s$d$y - fitted(fit, effects = e), tolerance = 1e-10)
  }
  # The field the model fitted is absent from the conditional residuals and
  # present in the population ones.
  i_pop  <- moran_i(fit, s$coords, weights = "knn", k = 8,
                    effects = "population")$statistic
  i_cond <- moran_i(fit, s$coords, weights = "knn", k = 8)$statistic
  expect_gt(i_pop, 0.3)
  expect_lt(abs(i_cond), 0.1)
  vg_c <- tulpa_variogram(fit, s$coords, n_bins = 6)
  vg_p <- tulpa_variogram(fit, s$coords, n_bins = 6, effects = "population")
  expect_lt(diff(range(vg_c$gamma)) / mean(vg_c$gamma),
            diff(range(vg_p$gamma)) / mean(vg_p$gamma))
  pdf(NULL); on.exit(grDevices::dev.off(), add = TRUE)
  cm <- check_model(fit, coords = s$coords, nsim = 50)
  expect_true(all(abs(cm$moran$I) < 0.1))
})

test_that("conditional Laplace RE fit adds the modal group effects", {
  skip_on_cran()
  set.seed(3)
  g <- rep(1:12, each = 20); x <- rnorm(length(g)); bg <- rnorm(12, 0, 0.8)
  d <- data.frame(y = rpois(length(g), exp(0.2 + 0.5 * x + bg[g])), x = x,
                  g = factor(g))
  fit <- tulpa(y ~ x + (1 | g), data = d, family = "poisson",
               mode = "laplace", sigma_re = 0.8)
  b <- coef(fit)
  u <- fit$mode[fit$n_fixed + seq_len(12)]
  expect_equal(log(fitted(fit)), as.numeric(b[1] + b[2] * x + u[g]),
               tolerance = 1e-8)
  expect_equal(log(fitted(fit, effects = "population")),
               as.numeric(b[1] + b[2] * x), tolerance = 1e-10)
})

test_that("conditional re_cov_gibbs fit adds the posterior-mean group effects", {
  skip_on_cran()
  set.seed(2)
  g <- rep(1:15, each = 16); x <- rnorm(length(g)); bg <- rnorm(15)
  d <- data.frame(y = rbinom(length(g), 1, plogis(-0.3 + 0.8 * x + bg[g])),
                  x = x, g = factor(g))
  fit <- suppressWarnings(tulpa(
    y ~ x + (1 | g), data = d, family = "binomial", mode = "laplace",
    control = list(re_cov = "gibbs", n_iter = 400, warmup = 200, seed = 1)))
  expect_identical(fit$backend, "re_cov_gibbs")
  bm <- colMeans(fit$beta_draws)
  um <- colMeans(fit$re)
  expect_equal(qlogis(fitted(fit)), as.numeric(bm[1] + bm[2] * x + um[g]),
               tolerance = 1e-8)
  expect_equal(qlogis(fitted(fit, effects = "population")),
               as.numeric(bm[1] + bm[2] * x), tolerance = 1e-8)
})

test_that("conditional sampler fit is the mean of the engine's eta draws", {
  skip_on_cran()
  s <- .fe_grid_data(2)
  fit <- suppressWarnings(tulpa(
    y ~ x + spatial(region), data = s$d, family = "poisson",
    spatial = list(type = "icar", adjacency = s$W), mode = "hmc",
    control = list(n_iter = 200, warmup = 100, n_chains = 2, seed = 1)))
  eta <- tulpa:::.tulpa_eta_draws(fit)
  expect_equal(log(fitted(fit)), as.numeric(colMeans(eta)), tolerance = 1e-10)
  i_cond <- moran_i(fit, s$coords, weights = "knn", k = 8)$statistic
  expect_lt(abs(i_cond), 0.1)
})

test_that("the diagnostics take residual vectors and other models unchanged", {
  set.seed(1)
  coords <- cbind(runif(40), runif(40))
  r <- rnorm(40)
  expect_equal(moran_i(r, coords)$statistic,
               moran_i(r, coords, effects = "population")$statistic)
  lm_fit <- stats::lm(r ~ 1)
  expect_equal(unname(durbin_watson(lm_fit)$statistic),
               unname(durbin_watson(stats::residuals(lm_fit))$statistic))
  expect_error(moran_i(list(a = 1), coords), "fitted model or a numeric vector")
})
