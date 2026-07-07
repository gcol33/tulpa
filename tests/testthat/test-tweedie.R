# Tweedie compound Poisson-gamma family (gcol33/tulpa C11), on the phi2
# channel (power p). Density correctness is anchored self-contained: the zero
# mass plus the integral of the positive density must be 1, the sampler's
# moments must match (mu, phi mu^p), and the score must differentiate the
# series log-density. Cross-checked against tweedie::dtweedie when installed.

test_that("tweedie log-density normalizes and matches its moments", {
  phi <- 1.2; p <- 1.5; eta <- 0.7
  mu  <- exp(eta)

  # P(y = 0) + integral of the positive density == 1.
  p0 <- exp(tulpa:::family_loglik(eta, 0, "tweedie", phi = phi, phi2 = p))
  dens <- function(y) {
    vapply(y, function(yy)
      exp(tulpa:::family_loglik(eta, yy, "tweedie", phi = phi, phi2 = p)),
      numeric(1))
  }
  ipos <- stats::integrate(dens, 1e-7, 60, rel.tol = 1e-8)$value
  expect_equal(p0 + ipos, 1, tolerance = 1e-4)

  # Mean and variance of the density: integral y f(y), integral y^2 f(y).
  m1 <- stats::integrate(function(y) y * dens(y), 1e-7, 60,
                         rel.tol = 1e-8)$value
  m2 <- stats::integrate(function(y) y^2 * dens(y), 1e-7, 60,
                         rel.tol = 1e-8)$value
  expect_equal(m1, mu, tolerance = 1e-3)
  expect_equal(m2 - m1^2, phi * mu^p, tolerance = 1e-2)

  # Sampler moments (including the zero mass).
  set.seed(97)
  ys <- tulpa:::family_sample(rep(eta, 60000), "tweedie", phi = phi, phi2 = p)
  expect_equal(mean(ys == 0), p0, tolerance = 0.01)
  expect_equal(mean(ys), mu, tolerance = 0.05)
  expect_equal(stats::var(ys), phi * mu^p, tolerance = 0.15)

  # Score == numerical derivative of the series log-density.
  y <- c(0, 0.4, 2.5); h <- 1e-6
  num <- (tulpa:::family_loglik(eta + h, y, "tweedie", phi = phi, phi2 = p) -
          tulpa:::family_loglik(eta - h, y, "tweedie", phi = phi, phi2 = p)) /
         (2 * h)
  expect_equal(tulpa:::family_score_eta(rep(eta, 3), y, "tweedie",
                                        phi = phi, phi2 = p),
               num, tolerance = 1e-5)
})

test_that("tweedie matches the reference tweedie package density", {
  skip_if_not_installed("tweedie")
  for (p in c(1.2, 1.5, 1.8)) {
    eta <- c(-0.5, 0.3, 1.2); phi <- 0.9
    y   <- c(0, 0.7, 3.1)
    ours <- tulpa:::family_loglik(eta, y, "tweedie", phi = phi, phi2 = p)
    ref  <- log(tweedie::dtweedie(y, mu = exp(eta), phi = phi, power = p))
    expect_equal(ours, ref, tolerance = 1e-8)
  }
})

test_that("C++ tweedie kernel matches the R registry", {
  # The Laplace kernel's series lives in laplace_family_link.h; pin it against
  # the R implementation through a laplace fit vs an optim reference.
  skip_on_cran()
  set.seed(99)
  n <- 300L
  x <- rnorm(n)
  eta <- 0.5 + 0.6 * x
  mu <- exp(eta); phi <- 1.0; p <- 1.6
  lam <- mu^(2 - p) / (phi * (2 - p))
  a <- (2 - p) / (p - 1); b <- mu^(1 - p) / (phi * (p - 1))
  nev <- rpois(n, lam)
  y <- ifelse(nev > 0, rgamma(n, shape = nev * a, rate = b), 0)
  d <- data.frame(y = y, x = x)

  fit <- tulpa(y ~ x, data = d, family = "tweedie", mode = "laplace",
               phi = phi, phi2 = p)

  X <- cbind(1, x)
  nlp <- function(bb) {
    -sum(tulpa:::family_loglik(as.numeric(X %*% bb), y, "tweedie",
                               phi = phi, phi2 = p)) + sum(bb^2) / (2 * 100^2)
  }
  ref <- stats::optim(c(0, 0), nlp, method = "BFGS")$par
  expect_equal(unname(coef(fit)), ref, tolerance = 1e-3)
  expect_lt(max(abs(ref - c(0.5, 0.6))), 0.15)
})

test_that("tweedie validation fires", {
  expect_error(tulpa:::.tweedie_power(NULL), "requires")
  expect_error(tulpa:::.tweedie_power(2.5), "strictly in \\(1, 2\\)")
  d <- data.frame(x = rnorm(20), y = rgamma(20, 2))
  expect_error(
    tulpa(y ~ x, data = d, family = "tweedie", mode = "laplace", phi = 1),
    "requires")
  # y < 0 is outside the support.
  expect_identical(
    tulpa:::family_loglik(0, -1, "tweedie", phi = 1, phi2 = 1.5), -Inf)
})
