test_that("the registry importance design matches the exact additive-variance posterior", {
  skip_on_cran()
  set.seed(131L)
  n <- 16L
  y <- .3 + rnorm(n, sd = .2) + rnorm(n, sd = .8) + rnorm(n, sd = .3)
  block <- list(type = "iid", n_units = n, obs_idx = seq_len(n))
  a <- block; a$name <- "local"
  b <- block; b$name <- "regional"
  fit <- tulpa_nested_laplace(y, rep(1L, n), matrix(1, n, 1),
    prior = list(a, b), family = "gaussian", phi = .09,
    control = list(progress = FALSE, integration = "qmc", qmc_nodes = 8192L,
                   diagnose_skew = FALSE))
  u <- seq(-10, 2, by = .02)
  g <- expand.grid(a = u, b = u)
  v <- .09 + exp(2 * g$a) + exp(2 * g$b)
  rss <- sum((y - mean(y))^2)
  lp <- -.5 * ((n - 1) * log(v) + log(v + n * 2.5^2) + rss / v +
                n * mean(y)^2 / (v + n * 2.5^2) + n * log(2 * pi))
  anchor <- tulpa:::.nl_scale_anchor()
  lambda <- -log(anchor[2L]) / anchor[1L]
  lp <- lp + 2 * log(lambda) + g$a + g$b - lambda * (exp(g$a) + exp(g$b))
  w <- exp(lp - max(lp)); w <- matrix(w / sum(w), length(u))
  dense <- t(vapply(list(rowSums(w), colSums(w)), function(m)
    approx(c(0, cumsum(m)), c(u - .01, max(u) + .01),
           c(.025, .5, .975), ties = "ordered")$y, numeric(3)))
  got <- log(cbind(fit$theta_ci_lo, fit$theta_median, fit$theta_ci_hi))
  expect_identical(fit$integration, "qmc")
  expect_identical(fit$theta_interval_read, "sample")
  expect_lt(max(abs(got - dense)), .1)
  h <- .02
  log_z <- max(lp) + log(sum(exp(lp - max(lp)))) + 2 * log(h)
  expect_lt(abs(fit$log_evidence - log_z), .015)
  expect_equal(length(fit$grid_modes), nrow(fit$theta_grid))
  expect_equal(length(fit$grid_hessians), nrow(fit$theta_grid))
})

test_that("a multi-scale Gaussian field resolves both variance arms against dense integration", {
  skip_if_not_slow()
  set.seed(1L)
  n <- 60L
  lon <- runif(n, 0, 10); lat <- runif(n, 0, 10)
  D <- as.matrix(dist(cbind(lon, lat)))
  draw <- function(s, r) as.numeric(t(chol(s^2 * exp(-D / r) + diag(1e-9, n))) %*% rnorm(n))
  y <- .3 + draw(.63, .5) + draw(1, 3) + rnorm(n, sd = sqrt(.1))
  d <- data.frame(lon, lat, y)
  fit <- suppressWarnings(tulpa(y ~ 1, data = d, family = "gaussian", phi = .1,
    spatial = spatial_multiscale(~lon + lat, nn_local = 59L, nn_regional = 59L,
      range_local = c(.2, 1.5), range_regional = c(1.5, 10), scale_coords = FALSE),
    mode = "nested_laplace", control = list(progress = FALSE)))
  exact <- rbind(c(-5.355663379053, -.660969779930, .195844608825),
                 c(-2.467203848841, -.935123254317, 1.065092774484),
                 c(-6.387328253070, -.791950189684, .789880274331),
                 c(-.523917839438, .961086902352, 3.880699007175))
  sd <- c(1.428668226221, .905404086889, 1.818775063455, 1.149732198638)
  got <- log(cbind(fit$theta_ci_lo, fit$theta_median, fit$theta_ci_hi))
  expect_lt(max(abs((got - exact) / sd)), .1)
  expect_lt(abs(fit$log_evidence + 83.3996422521), .015)
  expect_true(fit$outer_qmc$converged)
})
