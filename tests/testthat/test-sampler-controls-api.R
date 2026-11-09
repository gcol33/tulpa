test_that("sampler controls cross the C API and recover a conjugate posterior", {
  skip_if_not_slow()
  x <- seq(-1, 1, length.out = 80)
  X <- cbind(1, x)
  set.seed(180)
  y <- drop(X %*% c(.4, -.7)) + rnorm(length(x))
  covariance <- solve(crossprod(X) + diag(1 / 2.5^2, ncol(X)))
  mean <- drop(covariance %*% crossprod(X, y))
  reference <- vapply(seq_len(ncol(X)), function(j)
    qnorm(c(.1, .5, .9), mean[j], sqrt(covariance[j, j])), numeric(3))
  for (seed in c(18L, 181L, 182L)) {
    for (metric in c("auto", "diag", "dense", "block_diag")) {
      for (algorithm in c("nuts", "hmc", "walnuts")) {
        fit <- tulpa:::cpp_test_sampler_controls(y, X,
          L = if (algorithm == "hmc") 7L else 0L,
          metric = metric, walnuts = algorithm == "walnuts", seed = seed)
        expect_true(fit$identical)
        expect_true(fit$gradient_verified)
        expect_equal(fit$sampler, switch(algorithm, nuts = "nuts", hmc = "HMC",
                                         walnuts = "WALNUTS"))
        draws <- fit$draws[[1]]
        scale <- sqrt(diag(covariance))
        expect_lt(max(abs(colMeans(draws) - mean) / scale), .25)
        q <- apply(draws, 2, quantile, probs = c(.1, .5, .9))
        message(sprintf('CONTROL seed=%d metric=%s sampler=%s mean_ref_sd=%.6f quantile_ref_sd=%.6f',
          seed, metric, algorithm, max(abs(colMeans(draws) - mean) / scale),
          max(abs(sweep(q - reference, 2, scale, "/")))))
        expect_lt(max(abs(sweep(q - reference, 2, scale, "/"))), .35)
        expect_true(all(is.finite(fit$energy[[1]])))
        if (algorithm == "hmc") expect_true(all(fit$n_leapfrog[[1]] == 7L))
      }
    }
  }
})

test_that("multiple chains preserve sampler controls and diagnostics", {
  skip_if_not_slow()
  X <- matrix(1, 40, 1)
  y <- seq(-1, 1, length.out = 40)
  for (algorithm in c("nuts", "hmc", "walnuts")) {
    fit <- tulpa:::cpp_test_sampler_controls(y, X,
      L = if (algorithm == "hmc") 5L else 0L, metric = "dense",
      walnuts = algorithm == "walnuts", n_chains = 2L,
      n_iter = 600L, n_warmup = 300L)
    expect_true(fit$identical)
    expect_length(fit$draws, 2L)
    expect_equal(fit$sampler, rep(switch(algorithm, nuts = "nuts", hmc = "HMC",
                                        walnuts = "WALNUTS"), 2))
    expect_true(all(vapply(fit$energy, function(x) all(is.finite(x)), logical(1))))
    if (algorithm == "hmc") expect_true(all(unlist(fit$n_leapfrog) == 5L))
  }
})
