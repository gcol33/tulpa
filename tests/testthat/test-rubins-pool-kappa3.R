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
    # The mixture's own variance (1/K between-draw moment), not V_total.
    V_mix        <- mean(sigma_k[, j]^2) + mean((mu_k[, j] - mu_pooled)^2)
    kappa3 <- mean(sigma_k[, j]^3 * gamma_k[, j]) +
      3 * mean((mu_k[, j] - mu_pooled) * sigma_k[, j]^2) +
      mean((mu_k[, j] - mu_pooled)^3)

    expect_equal(pooled$only$kappa3[j], kappa3, tolerance = 1e-12)
    expect_equal(pooled$only$gamma[j], kappa3 / V_mix^1.5,
                 tolerance = 1e-12)
    expect_equal(pooled$only$V_total[j], V_total, tolerance = 1e-12)
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

test_that("rubins_pool() gamma is the skewness of the draw mixture (#888)", {
  # Equal-weight mixture of N(0, 1), N(0, 1), N(3, 1): mean 1, variance
  # 1 + 2 = 3, third central moment 2, so skewness 2 / 3^1.5 = 0.3849.
  # Standardizing by Rubin's V_total = 1 + (4/3) * 3 = 5 read 0.179.
  dr <- lapply(c(0, 0, 3), function(m) list(a = list(beta = m, se = 1, gamma = 0)))
  expect_equal(rubins_pool(dr)$a$gamma, 2 / 3^1.5, tolerance = 1e-12)

  # Against a Monte Carlo draw from a K = 5 mixture of skew-normals.
  set.seed(8)
  mu <- c(-0.5, 0, 0.2, 1.5, 3); sg <- c(1, 0.5, 0.8, 1.2, 0.7)
  al <- c(3, -2, 0, 4, 1)            # skew-normal shapes
  sn_mom <- function(xi, om, a) {    # skew-normal mean / sd / skewness
    b <- a / sqrt(1 + a^2) * sqrt(2 / pi)
    c(mean = xi + om * b, sd = om * sqrt(1 - b^2),
      gamma = (4 - pi) / 2 * b^3 / (1 - b^2)^1.5)
  }
  mom <- unname(vapply(seq_along(mu), function(k) sn_mom(mu[k], sg[k], al[k]),
                       numeric(3)))
  dr <- lapply(seq_along(mu), function(k)
    list(a = list(beta = mom[1, k], se = mom[2, k], gamma = mom[3, k])))
  n <- 4e5; k <- sample.int(5, n, replace = TRUE)
  d <- al[k] / sqrt(1 + al[k]^2)
  x <- mu[k] + sg[k] * (d * abs(rnorm(n)) + sqrt(1 - d^2) * rnorm(n))
  mc <- mean((x - mean(x))^3) / mean((x - mean(x))^2)^1.5
  expect_equal(rubins_pool(dr)$a$gamma, mc, tolerance = 0.02)
})

test_that("rubins_pool() refuses mismatched draws instead of recycling (#888)", {
  expect_error(
    rubins_pool(list(list(m = list(beta = 1, se = 1)),
                     list(m = list(beta = c(1, 2), se = c(1, 1))))),
    "common length")
  expect_error(
    rubins_pool(list(list(m = list(beta = 1, se = 1)),
                     list(m = list(beta = 2)))),
    "numeric `se`")
  expect_error(
    rubins_pool(list(list(m = list(beta = 1)), list(m = list(beta = 2)))),
    "numeric `se`")
  expect_error(
    rubins_pool(list(list(m = list(beta = 1, se = 1)),
                     list(m = list(beta = 2, se = c(1, 1))))),
    "common length")
  expect_error(rubins_pool(list(1, 2)), "`draws`")
})

test_that("rubins_pool() pools a submodel draw 1 lacks (#888)", {
  dr <- list(list(a = list(beta = 1, se = 1)),
             list(a = list(beta = 2, se = 1), b = list(beta = 5, se = 1)),
             list(a = list(beta = 3, se = 1), b = list(beta = 6, se = 1)))
  out <- rubins_pool(dr)
  expect_named(out, c("a", "b"))
  expect_equal(out$b$mean, 5.5)
  expect_equal(out$b$K, 2L)
  expect_equal(out$a$K, 3L)

  dr[[3]]$b <- NULL
  expect_warning(out <- rubins_pool(dr), "'b' appears in only 1 draw")
  expect_named(out, "a")
})
