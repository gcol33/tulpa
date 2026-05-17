# Tests for the third-cumulant (gamma) extension of rubins_pool().

test_that("rubins_pool() pools gamma via law of total cumulants", {
  set.seed(1)
  K <- 50L
  p <- 2L
  mu_k    <- matrix(rnorm(K * p, mean = c(0.2, -0.5), sd = 0.1),
                    nrow = K, byrow = TRUE)
  sigma_k <- matrix(runif(K * p, 0.3, 0.6), nrow = K)
  gamma_k <- matrix(runif(K * p, -0.4, 0.4), nrow = K)

  draws <- lapply(seq_len(K), function(k) {
    list(only = list(
      beta  = mu_k[k, ],
      se    = sigma_k[k, ],
      gamma = gamma_k[k, ]
    ))
  })

  pooled <- rubins_pool(draws)
  expect_true("gamma"  %in% names(pooled$only))
  expect_true("kappa3" %in% names(pooled$only))
  expect_length(pooled$only$gamma,  p)
  expect_length(pooled$only$kappa3, p)

  # Recompute the formula by hand and check agreement.
  for (j in seq_len(p)) {
    mu_pooled    <- mean(mu_k[, j])
    V_within     <- mean(sigma_k[, j]^2)
    V_between    <- var(mu_k[, j])
    V_total      <- V_within + (1 + 1 / K) * V_between
    sigma_pooled <- sqrt(V_total)
    kappa3 <- mean(sigma_k[, j]^3 * gamma_k[, j]) +
      3 * mean((mu_k[, j] - mu_pooled) * sigma_k[, j]^2) +
      mean((mu_k[, j] - mu_pooled)^3)

    expect_equal(pooled$only$kappa3[j], kappa3, tolerance = 1e-12)
    expect_equal(pooled$only$gamma[j], kappa3 / sigma_pooled^3,
                 tolerance = 1e-12)
  }
})

test_that("rubins_pool() skips gamma path when any draw lacks gamma", {
  set.seed(2)
  K <- 5L; p <- 2L
  draws <- lapply(seq_len(K), function(k) {
    list(only = list(beta = rnorm(p), se = runif(p, 0.2, 0.5),
                     gamma = rnorm(p, sd = 0.2)))
  })
  draws[[3]]$only$gamma <- NULL  # one draw missing gamma

  pooled <- rubins_pool(draws)
  expect_false("gamma"  %in% names(pooled$only))
  expect_false("kappa3" %in% names(pooled$only))
  expect_true("mean" %in% names(pooled$only))
  expect_true("se"   %in% names(pooled$only))
})

test_that("rubins_pool() reduces to existing path when no gamma supplied", {
  set.seed(3)
  K <- 4L; p <- 2L
  draws <- lapply(seq_len(K), function(k) {
    list(only = list(beta = rnorm(p), se = runif(p, 0.2, 0.5)))
  })
  pooled <- rubins_pool(draws)
  expect_named(pooled$only,
               c("mean", "se", "V_within", "V_between", "V_total", "K"),
               ignore.order = TRUE)
})

test_that("rubins_pool() recovers zero skewness from symmetric inputs", {
  set.seed(4)
  K <- 200L; p <- 1L
  mu_k    <- rnorm(K, mean = 0, sd = 0.05)
  sigma_k <- rep(0.5, K)
  gamma_k <- rep(0, K)

  draws <- lapply(seq_len(K), function(k) {
    list(only = list(beta = mu_k[k], se = sigma_k[k], gamma = gamma_k[k]))
  })
  pooled <- rubins_pool(draws)
  # No within-draw skewness and (approximately) symmetric between-draw mu
  # distribution should leave kappa3 ~ 0.
  expect_lt(abs(pooled$only$kappa3), 5e-4)
})
