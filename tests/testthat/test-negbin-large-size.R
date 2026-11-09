# Negative binomial densities, scores and curvatures at a large size, where
# lgamma(y + phi) - lgamma(phi) and the polygamma differences cancel when they
# are formed as differences of two large numbers.

rising_sum <- function(k, y, size) {
  if (y == 0) return(0)
  sum((-1)^k * factorial(k) / (size + 0:(y - 1))^(k + 1))
}

test_that("polygamma rising differences match their finite sums at every size", {
  for (k in 0:3) for (y in c(0, 1, 5, 33, 40, 200, 5000))
    for (size in c(1e-3, 0.7, 9.5, 50, 1e3, 1e8, 1e17)) {
      expect_equal(tulpa:::.polygamma_rising_difference(k, y, size),
                   rising_sum(k, y, size), tolerance = 1e-10,
                   info = sprintf("k = %d, y = %g, size = %g", k, y, size))
    }
  expect_equal(tulpa:::.log_rising_factorial(c(0, 4, 40), 5),
               lgamma(c(0, 4, 40) + 5) - lgamma(5), tolerance = 1e-12)
})

test_that("every negative binomial type 2 density reaches the Poisson limit", {
  y <- c(0, 3, 17)
  mu <- c(0.4, 2, 12)
  # log Gamma(y + phi) - log Gamma(phi) summed term by term, which is exact for a
  # count; stats::dnbinom() itself is off by 8.6e-10 at phi = 1e8, y = 3.
  nb_exact <- function(y, mu, phi) {
    rising <- vapply(y, function(v) sum(log(phi + seq_len(v) - 1)), numeric(1))
    s <- log1p(mu / phi)
    rising - lgamma(y + 1) - phi * s + y * (log(mu) - log(phi) - s)
  }
  expect_equal(nb_exact(y, mu, 1e3), stats::dnbinom(y, size = 1e3, mu = mu, log = TRUE),
               tolerance = 1e-12)
  expect_equal(nb_exact(y, mu, 1e17), stats::dpois(y, mu, log = TRUE),
               tolerance = 1e-12)
  for (phi in c(1e3, 1e8, 1e12, 1e17)) {
    ref <- nb_exact(y, mu, phi)
    expect_equal(tulpa:::.FAMILY_OPS$neg_binomial_2$loglik(log(mu), y, 1, phi),
                 ref, tolerance = 1e-10, info = format(phi))
    expect_equal(tulpa:::.FAMILY_MU$neg_binomial_2$loglik_mu(y, mu, phi, 1),
                 ref, tolerance = 1e-10, info = format(phi))
    for (i in seq_along(y)) {
      expect_equal(unname(tulpa:::cpp_family_terms(y[i], 1L, log(mu[i]),
                                                   "neg_binomial_2", phi)[1L]),
                   ref[i], tolerance = 1e-10, info = paste(phi, y[i]))
      expect_equal(unname(tulpa:::cpp_glmm_elt_terms("negbin", log(mu[i]), y[i],
                                                     1, phi)[1L]),
                   ref[i], tolerance = 1e-10, info = paste(phi, y[i]))
    }
    pos <- y > 0
    p0 <- exp(-phi * log1p(mu[pos] / phi))
    expect_equal(tulpa:::.FAMILY_OPS$truncated_neg_binomial_2$loglik(
                   log(mu[pos]), y[pos], 1, phi),
                 ref[pos] - log1p(-p0), tolerance = 1e-10, info = format(phi))
  }
})

test_that("the type 1 density, score and curvature reach the Poisson limit", {
  y <- c(0, 3, 17)
  mu <- c(0.4, 2, 12)
  for (phi in c(1e-4, 1e-10, 1e-15)) {
    r <- mu / phi
    ref <- if (phi >= 1e-4) {
      stats::dnbinom(y, size = r, prob = 1 / (1 + phi), log = TRUE)
    } else {
      stats::dpois(y, mu, log = TRUE)
    }
    expect_equal(tulpa:::.FAMILY_OPS$neg_binomial_1$loglik(log(mu), y, 1, phi),
                 ref, tolerance = max(1e-10, 1e3 * phi), info = format(phi))
    score <- vapply(seq_along(y), function(i)
      r[i] * (rising_sum(0, y[i], r[i]) - log1p(phi)), numeric(1))
    expect_equal(tulpa:::.FAMILY_OPS$neg_binomial_1$score(log(mu), y, 1, phi),
                 score, tolerance = 1e-8, info = format(phi))
    obs <- vapply(seq_along(y), function(i)
      -score[i] - r[i]^2 * rising_sum(1, y[i], r[i]), numeric(1))
    expect_equal(tulpa:::.FAMILY_OPS$neg_binomial_1$obs_weight(log(mu), y, 1, phi),
                 obs, tolerance = 1e-8, info = format(phi))
    for (i in seq_along(y)) {
      terms <- tulpa:::cpp_family_obs_terms(y[i], 1L, log(mu[i]),
                                            "neg_binomial_1", phi)
      expect_equal(unname(terms[1L]), score[i], tolerance = 1e-8,
                   info = paste(phi, y[i]))
      expect_equal(unname(terms[2L]), obs[i], tolerance = 1e-8,
                   info = paste(phi, y[i]))
    }
  }
})

test_that("the type 2 size score and the CRT mean are exact at a large size", {
  y <- c(0, 3, 17)
  mu <- c(0.4, 2, 12)
  for (phi in c(50, 1e4, 1e6)) {
    ref <- vapply(seq_along(y), function(i)
      rising_sum(0, y[i], phi) - log1p(mu[i] / phi) + (mu[i] - y[i]) / (phi + mu[i]),
      numeric(1))
    expect_equal(tulpa:::.FAMILY_DPHI$neg_binomial_2$dloglik(log(mu), y, 1, phi),
                 ref, tolerance = 1e-8, info = format(phi))
  }
  for (r in c(0.3, 40, 1e9, 1e16)) for (yy in c(0L, 1L, 8L, 120L)) {
    expect_equal(tulpa:::cpp_crt_mean(yy, r), r * rising_sum(0, yy, r),
                 tolerance = 1e-10, info = paste(r, yy))
  }
})
