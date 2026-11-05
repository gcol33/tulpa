test_that("continuous field samplers carry physical range divergence on the fit", {
  skip_if_not_slow()
  set.seed(427)
  n <- 16L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- .3 + .2 * d$x + rnorm(n, sd = .4)
  specs <- list(
    spatial_gp(~lon + lat, nn = 5L, scale_coords = FALSE),
    spatial_gp(~lon + lat, approx = "hsgp", m = 3L, c = 2, scale_coords = FALSE),
    spatial_svc(~lon + lat, terms = ~x - 1, nn = 5L, scale_coords = FALSE),
    spatial_svc(~lon + lat, terms = ~x - 1, approx = "hsgp", m = 3L,
      c_boundary = 2, scale_coords = FALSE),
    spatial_multiscale(~lon + lat, nn_local = 5L, nn_regional = 5L,
      range_local = c(.1, .5), range_regional = c(.5, 2), scale_coords = FALSE))
  for (sp in specs) {
    fit <- suppressWarnings(tulpa(y ~ x, data = d, family = "gaussian", phi = .16,
      spatial = sp, mode = "hmc", control = list(n_iter = 180L, n_warmup = 90L,
        n_chains = 1L, seed = 19L)))
    s <- spatial_range(fit)
    range <- startsWith(rownames(s), "range")
    expect_true(any(range))
    expect_true(all(s$mean[range] == Inf))
    expect_true(all(s$sd[range] == Inf))
    expect_true(all(is.finite(s$q2.5[range]) & is.finite(s$q97.5[range])))
    expect_true(length(fit$moment_existence) > 0L)
    expect_true(all(vapply(fit$moment_existence, function(x) !x$mean_exists, TRUE)))
    expect_true(all(is.finite(unlist(fit$theta_log_mean))))
  }
})

test_that("bounded Polya-Gamma range samplers retain their existing moments", {
  skip_if_not_slow()
  set.seed(431)
  n <- 24L
  d <- data.frame(lon = runif(n), lat = runif(n), x = rnorm(n))
  d$y <- rbinom(n, 1L, plogis(.3 + .2 * d$x))
  gp <- spatial_gp(~lon + lat, nn = 5L, scale_coords = FALSE)
  specs <- list(gp, spatial_rsr(gp, restrict_to = ~x),
    spatial_multiscale(~lon + lat, nn_local = 5L, nn_regional = 5L,
      range_local = c(.1, .5), range_regional = c(.5, 2), scale_coords = FALSE))
  for (sp in specs) {
    fit <- suppressWarnings(tulpa(y ~ x, data = d, family = "binomial", spatial = sp,
      mode = "gibbs", control = list(n_iter = 250L, warmup = 100L, seed = 23L)))
    s <- spatial_range(fit)
    r <- startsWith(rownames(s), "range")
    expect_true(all(is.finite(as.matrix(s[r, ]))))
    expect_true(all(vapply(fit$moment_existence, function(x)
      isTRUE(x$bounded) && isTRUE(x$mean_exists) && isTRUE(x$sd_exists), TRUE)))
    for (axis in names(fit$moment_existence)) {
      bounds <- fit$moment_existence[[axis]]$bounds
      values <- exp(fit$draws[, axis])
      expect_true(all(values >= bounds[1] & values <= bounds[2]))
    }
    if (sp$type == "gp") {
      values <- 3 * exp(fit$draws[, "log_phi_gp"])
      expect_equal(s["range", "mean"], mean(values), tolerance = 1e-12)
      expect_equal(s["range", "sd"], sd(values), tolerance = 1e-12)
    }
  }
})

test_that("joint SPDE sampling keeps its existing mean and infinite range SD", {
  skip_if_not_slow()
  C <- Matrix::Diagonal(4)
  G <- Matrix::Matrix(matrix(c(1, -1, 0, 0, -1, 2, -1, 0,
                              0, -1, 2, -1, 0, 0, -1, 1), 4), sparse = TRUE)
  A <- Matrix::sparseMatrix(i = 1:20, j = rep(1:4, 5), x = 1)
  sp <- spatial_spde_custom(C, G, A, prior_range = c(.5, .5), prior_sigma = c(1, .1))
  set.seed(433)
  X <- cbind(1, x = rnorm(20))
  y <- .3 + .2 * X[, 2] + rep(c(-.3, .1, .1, .1), 5) + rnorm(20, sd = .3)
  fit <- tulpa_nuts_spde(y, X, sp, family = "gaussian", joint = TRUE,
    control = list(n_iter = 240L, n_warmup = 120L, seed = 29L))
  s <- spatial_range(fit)
  expect_true(is.finite(s["range", "mean"]))
  expect_identical(s["range", "sd"], Inf)
  expect_identical(fit$range_summary[["sd"]], Inf)
  expect_true(fit$moment_existence$log_kappa$mean_exists)
  expect_false(fit$moment_existence$log_kappa$sd_exists)
})
