gaussian_field_reference <- function(y, group, Q, beta_sd = 2.5, U = 3, alpha = .01) {
  S <- nrow(Q)
  P <- diag(S) - matrix(1 / S, S, S)
  augmented <- Q + matrix(1 / S, S, S)
  basis <- t(chol(solve(augmented)))
  field_design <- P[group, , drop = FALSE] %*% basis
  rate <- -log(alpha) / U
  conditional <- function(log_tau) {
    design <- cbind(1, exp(-.5 * log_tau) * field_design)
    rhs <- drop(crossprod(design, y))
    H <- crossprod(design) + diag(c(1 / beta_sd^2, rep(1, S)))
    covariance <- solve(H)
    mean <- drop(covariance %*% rhs)
    list(log_weight = -.5 * as.numeric(determinant(H, logarithm = TRUE)$modulus) +
      .5 * sum(rhs * mean) - .5 * log_tau - rate * exp(-.5 * log_tau),
      mean = mean[1], variance = covariance[1, 1])
  }
  log_weight <- function(z) vapply(z, function(x) conditional(x)$log_weight, numeric(1))
  peak <- optimize(log_weight, c(-15, 50), maximum = TRUE)$objective
  weight <- function(z) exp(log_weight(z) - peak)
  integral <- function(f, lower = -15, upper = 50)
    integrate(f, lower, upper, rel.tol = 1e-8, subdivisions = 500L)$value
  norm <- integral(weight)
  z_mean <- integral(function(z) z * weight(z)) / norm
  z_sd <- sqrt(integral(function(z) (z - z_mean)^2 * weight(z)) / norm)
  b_mean <- integral(function(z) weight(z) * vapply(z,
    function(x) conditional(x)$mean, numeric(1))) / norm
  b_sd <- sqrt(integral(function(z) weight(z) * vapply(z,
    function(x) { c <- conditional(x); c$variance + (c$mean - b_mean)^2 }, numeric(1))) / norm)
  quantile <- function(p, variable) uniroot(function(x) {
    cdf <- if (variable == "log_tau") integral(weight, upper = x) else
      integral(function(z) weight(z) * vapply(z, function(a) {
        c <- conditional(a); pnorm(x, c$mean, sqrt(c$variance))
      }, numeric(1)))
    cdf / norm - p
  }, if (variable == "log_tau") c(-15, 50) else c(-10, 10), tol = 1e-7)$root
  list(mean = c(b_mean, z_mean), sd = c(b_sd, z_sd),
    quantiles = sapply(c("beta", "log_tau"), function(v)
      vapply(c(.1, .5, .9), quantile, numeric(1), variable = v)),
    tail_mass = integral(weight, 50, Inf) / norm)
}

test_that("automatic ICAR and RW1 fields recover dense Gaussian integration", {
  skip_if_not_slow()
  set.seed(1802)
  S <- 4L
  group <- rep(seq_len(S), each = 20L)
  d <- data.frame(unit = group, time = group,
    y = .4 + c(-.7, -.3, .3, .7)[group] + rnorm(length(group)))
  adjacency <- matrix(0, S, S)
  for (s in seq_len(S - 1L)) adjacency[s, s + 1L] <- adjacency[s + 1L, s] <- 1
  reference <- gaussian_field_reference(d$y, group, diag(rowSums(adjacency)) - adjacency)
  expect_lt(reference$tail_mass, 1e-8)
  for (field in c("icar", "rw1")) for (seed in c(18L, 181L, 182L)) {
    args <- list(formula = y ~ 1, data = d, family = "gaussian", phi = 1,
      mode = "hmc", control = list(n_iter = 6000L, warmup = 2000L,
        n_chains = 1L, seed = seed, verbose = FALSE, adapt_delta = .9))
    if (field == "icar") {
      args$formula <- y ~ 1 + spatial(unit)
      args$spatial <- spatial_car(adjacency, group_var = "unit")
    }
    else args$temporal <- temporal_rw1("time")
    fit <- do.call(tulpa, args)
    columns <- c(which(colnames(fit$draws) == "(Intercept)"),
                 grep("^log_tau", colnames(fit$draws)))
    expect_length(columns, 2L)
    draws <- fit$draws[, columns, drop = FALSE]
    mean_gap <- max(abs(colMeans(draws) - reference$mean) / reference$sd)
    quantile_gap <- max(abs(sweep(apply(draws, 2, quantile, c(.1, .5, .9)) -
      reference$quantiles, 2, reference$sd, "/")))
    message(sprintf('FIELD field=%s seed=%d mean_ref_sd=%.6f quantile_ref_sd=%.6f',
      field, seed, mean_gap, quantile_gap))
    expect_lt(mean_gap, .25)
    expect_lt(quantile_gap, .45)
  }
})
