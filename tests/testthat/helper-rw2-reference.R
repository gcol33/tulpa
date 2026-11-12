.dense_tvc_rw2_reference <- function(y, x, time, n_times, n_grid = 801L,
                                     sigma_beta = 2.5, U = 1, alpha = .01) {
  stopifnot(length(y) == length(x), length(y) == length(time), n_times >= 3L,
            all(time %in% seq_len(n_times)))
  difference <- diff(diag(n_times), differences = 2L)
  spectral <- eigen(crossprod(difference), symmetric = TRUE)
  rank <- n_times - 2L
  linear <- seq_len(n_times) - (n_times + 1) / 2
  linear <- linear / sqrt(sum(linear^2))
  basis <- cbind(linear, spectral$vectors[, seq_len(rank), drop = FALSE])
  design <- cbind(1, x, x * basis[time, , drop = FALSE])
  p <- ncol(design)
  information <- crossprod(design)
  diag(information)[1:2] <- diag(information)[1:2] + 1 / sigma_beta^2
  score <- drop(crossprod(design, y))
  slope <- cbind(0, 1, basis)
  log_tau <- seq(-30, 50, length.out = n_grid)
  log_weight <- numeric(n_grid)
  means <- variances <- matrix(NA_real_, n_grid, n_times)
  beta_means <- matrix(NA_real_, n_grid, 2L)
  lambda <- -log(alpha) / U
  for (k in seq_along(log_tau)) {
    precision <- information
    penalized <- seq.int(4L, p)
    diag(precision)[penalized] <- diag(precision)[penalized] +
      exp(log_tau[k]) * spectral$values[seq_len(rank)]
    root <- chol(precision)
    covariance <- chol2inv(root)
    mean <- drop(covariance %*% score)
    log_weight[k] <- .5 * rank * log_tau[k] - sum(log(diag(root))) +
      .5 * sum(score * mean) + log(lambda / 2) - .5 * log_tau[k] -
      lambda * exp(-.5 * log_tau[k])
    means[k, ] <- drop(slope %*% mean)
    variances[k, ] <- rowSums((slope %*% covariance) * slope)
    beta_means[k, ] <- mean[1:2]
  }
  weight <- exp(log_weight - max(log_weight))
  weight[c(1L, n_grid)] <- weight[c(1L, n_grid)] / 2
  weight <- weight / sum(weight)
  average <- drop(crossprod(weight, means))
  variance <- drop(crossprod(weight, variances + means^2)) - average^2
  interval <- vapply(seq_len(n_times), function(j) {
    sd <- sqrt(variances[, j])
    bounds <- c(min(means[, j] - 10 * sd), max(means[, j] + 10 * sd))
    vapply(c(.025, .975), function(probability) {
      uniroot(function(value) sum(weight * pnorm(value, means[, j], sd)) - probability,
              interval = bounds, tol = 1e-10)$root
    }, numeric(1))
  }, numeric(2))
  list(slope_mean = average, slope_sd = sqrt(variance),
    slope_lower = interval[1L, ], slope_upper = interval[2L, ],
    beta_mean = drop(crossprod(weight, beta_means)),
    log_tau_mean = sum(weight * log_tau),
    log_tau_sd = sqrt(sum(weight * log_tau^2) - sum(weight * log_tau)^2),
    log_tau = log_tau, weight = weight)
}
