# Each temporal_gp() kernel reaches the density the samplers evaluate. The
# reference is the multivariate normal of the kernel's covariance, written out
# here independently of the C++.

kernel_r <- function(d, phi, kernel, period = 1) {
  switch(kernel,
    exponential = exp(-d / phi),
    matern32 = {
      u <- sqrt(3) * d / phi
      (1 + u) * exp(-u)
    },
    matern52 = {
      u <- sqrt(5) * d / phi
      (1 + u + u^2 / 3) * exp(-u)
    },
    gaussian = exp(-0.5 * (d / phi)^2),
    periodic = exp(-2 * sin(pi * d / period)^2 / phi^2)
  )
}

tgp_bundle <- function(times, cov, nu = 1.5, period = 1, param = "centered") {
  list(n_times = length(times), n_groups = 1L, shared = TRUE,
       time_values = times, cov_type = cov, nu = nu, period = period,
       gp_sigma2_prior_U = 1, gp_sigma2_prior_alpha = 0.01,
       gp_phi_prior_lower = 0.01, gp_phi_prior_upper = 10,
       gp_parameterization = param)
}

KERNELS <- list(
  exponential = list(cov = "exponential", nu = 1.5),
  matern32 = list(cov = "matern", nu = 1.5),
  matern52 = list(cov = "matern", nu = 2.5),
  gaussian = list(cov = "gaussian", nu = 1.5),
  periodic = list(cov = "periodic", nu = 1.5)
)

times <- c(0, 0.7, 1.1, 2.4, 3.0, 4.6, 5.2)
sigma2 <- 0.8
phi <- 1.3
period <- 2.5
logit_phi <- qlogis((phi - 0.01) / (10 - 0.01))

# The correlation the sampler factors: the exponential kernel goes through its
# exact AR(1) recursion, the others carry a 1e-6 nugget.
corr_r <- function(kernel) {
  R <- kernel_r(abs(outer(times, times, "-")), phi, kernel, period)
  if (kernel == "exponential") R else R + diag(1e-6, length(times))
}

# PC prior on sigma and uniform prior on phi, on (log sigma2, logit phi)
hyper_log_prior <- function(kernel) {
  rate <- -log(0.01) / 1
  sigma <- sqrt(sigma2)
  log(rate) - rate * sigma - log(2 * sigma) + log(sigma2) +
    log(phi - 0.01) + log(10 - phi) - log(10 - 0.01) +
    if (kernel == "gaussian") log(sqrt(2)) else 0
}

test_that("each kernel's centred density is the normal of its covariance", {
  load_cpp_fixture("test_temporal_gp_prior")
  set.seed(1)
  f <- rnorm(length(times))
  for (k in names(KERNELS)) {
    p <- tgp_bundle(times, KERNELS[[k]]$cov, KERNELS[[k]]$nu, period)
    K <- sigma2 * corr_r(k)
    ref <- -0.5 * (length(f) * log(2 * pi) +
                   as.numeric(determinant(K)$modulus) + sum(f * solve(K, f)))
    got <- cpp_temporal_gp_log_prior(f, log(sigma2), logit_phi, p)
    expect_equal(got$log_prior, hyper_log_prior(k) + ref, tolerance = 1e-8, label = k)
    expect_equal(got$f, f)
  }
})

test_that("the non-centred block is N(0, I) and maps to sigma L z", {
  load_cpp_fixture("test_temporal_gp_prior")
  set.seed(2)
  z <- rnorm(length(times))
  for (k in names(KERNELS)) {
    p <- tgp_bundle(times, KERNELS[[k]]$cov, KERNELS[[k]]$nu, period,
                    param = "noncentered")
    got <- cpp_temporal_gp_log_prior(z, log(sigma2), logit_phi, p)
    expect_equal(got$log_prior, hyper_log_prior(k) - 0.5 * sum(z^2),
                 tolerance = 1e-10, label = k)
    expect_equal(got$f, as.vector(sqrt(sigma2) * t(chol(corr_r(k))) %*% z),
                 tolerance = 1e-8, label = k)
  }
})

test_that("the draws' effects come from the sampler's kernel", {
  load_cpp_fixture("test_temporal_gp_prior")
  set.seed(3)
  z <- matrix(rnorm(3 * length(times)), 3)
  p <- tgp_bundle(times, "matern", 2.5, param = "noncentered")
  f <- cpp_temporal_gp_field(z, rep(log(sigma2), 3),
                                          rep(logit_phi, 3), p)
  L <- t(chol(corr_r("matern52")))
  expect_equal(f, t(sqrt(sigma2) * L %*% t(z)), tolerance = 1e-8)
})

