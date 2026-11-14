fused_fixture <- function(family, n = 31L, phi = 2.3, zi = FALSE, chains = 1L) {
  set.seed(1183 + n)
  X <- cbind(1, rnorm(n, 0, .3), runif(n, -.2, .2))
  trials <- rep(17L, n)
  y <- switch(family,
    gaussian = rnorm(n), gamma = exp(rnorm(n, 0, .3)), beta = runif(n, .1, .9),
    binomial = rbinom(n, trials, .4), beta_binomial = rbinom(n, trials, .4),
    truncated_poisson = rpois(n, 3) + 1,
    truncated_neg_binomial_2 = rpois(n, 3) + 1, rpois(n, 3))
  if (zi) y[seq.int(1L, n, 3L)] <- 0
  groups <- rep(seq_len(5L), length.out = n)
  width <- ncol(X) * (1L + zi) + max(groups)
  coefficients <- matrix(rnorm(width * chains, 0, .2), width, chains)
  list(family = family, y = as.double(y), trials = trials, X = X,
       coefficients = coefficients, phi = phi, zi = zi,
       offset = rnorm(n, 0, .1), weights = runif(n, .5, 2), groups = groups)
}

fused_run <- function(fixture, choice, ...) {
  do.call(tulpa:::cpp_device_family, c(fixture, list(choice = choice), list(...)))
}

