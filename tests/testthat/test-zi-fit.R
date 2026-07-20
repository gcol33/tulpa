# End-to-end zero-inflation through the compiled Laplace kernel.
#
# The density-level math is validated in test-family-zi.R; this file checks the
# wiring -- that `ziformula` reaches the kernel, that the second process is
# assembled and labelled correctly, and that the estimates recover known truth.
# Without recovery, a mixture that silently fit the count model alone would pass
# every structural check here.

sim_zip <- function(n, beta, gamma, seed) {
  set.seed(seed)
  x <- stats::rnorm(n)
  z <- stats::rnorm(n)
  eta <- beta[1] + beta[2] * x
  lz  <- gamma[1] + gamma[2] * z
  y <- stats::rpois(n, exp(eta))
  y[stats::runif(n) < stats::plogis(lz)] <- 0
  data.frame(y = y, x = x, z = z)
}


test_that("ziformula reaches the kernel and is labelled in the fit", {
  d <- sim_zip(400, c(1.2, 0.5), c(-0.4, 0.8), seed = 1)
  fit <- tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ z,
               mode = "laplace")

  expect_equal(fit$n_fixed, 4L)
  expect_equal(fit$fixed_names,
               c("(Intercept)", "x", "zi_(Intercept)", "zi_z"))
  expect_length(coef(fit), 4L)
  expect_true(all(is.finite(coef(fit))))
  # The zi block must be genuinely estimated, not left at its start value.
  expect_false(isTRUE(all.equal(unname(coef(fit)[3:4]), c(0, 0))))
})


test_that("a zero-inflated Poisson recovers its generating parameters", {
  beta <- c(1.2, 0.5); gamma <- c(-0.4, 0.8)
  est <- t(vapply(1:12, function(s) {
    d <- sim_zip(3000, beta, gamma, seed = 100 + s)
    coef(tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ z,
               mode = "laplace"))
  }, numeric(4)))

  truth <- c(beta, gamma)
  for (j in 1:4) {
    expect_equal(mean(est[, j]), truth[j], tolerance = 0.06)
  }
})


test_that("ignoring zero inflation biases the count intercept downward", {
  # The check that a silently-dropped mixture cannot pass: fitting the same
  # data without ziformula must show the structural zeros pulling the count
  # intercept below the truth, while the ZI fit stays on it.
  d <- sim_zip(4000, c(1.2, 0.5), c(0.2, 0.0), seed = 7)
  plain <- coef(tulpa(y ~ x, data = d, family = "poisson", mode = "laplace"))
  zi    <- coef(tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ 1,
                      mode = "laplace"))

  expect_lt(plain[["(Intercept)"]], 1.2 - 0.2)
  expect_equal(unname(zi[["(Intercept)"]]), 1.2, tolerance = 0.06)
})


test_that("a zero-inflated negative binomial recovers its parameters", {
  beta <- c(1.0, 0.4); gamma <- c(-0.3, 0.0); phi <- 3
  est <- t(vapply(1:10, function(s) {
    set.seed(500 + s)
    n <- 3000
    x <- stats::rnorm(n)
    y <- stats::rnbinom(n, size = phi, mu = exp(beta[1] + beta[2] * x))
    y[stats::runif(n) < stats::plogis(gamma[1])] <- 0
    d <- data.frame(y = y, x = x)
    coef(tulpa(y ~ x, data = d, family = "neg_binomial_2", phi = phi,
               ziformula = ~ 1, mode = "laplace"))
  }, numeric(3)))

  truth <- c(beta, gamma[1])
  for (j in 1:3) expect_equal(mean(est[, j]), truth[j], tolerance = 0.07)
})


test_that("zero inflation composes with a random intercept", {
  # Averaged over seeds: the per-fit SD of the zi intercept here is ~0.07, so a
  # single fit lands within ~0.15 of truth routinely and a one-seed assertion
  # would either be flaky or too loose to detect real bias.
  n_g <- 40; n_per <- 30; sig <- 0.5
  one <- function(seed) {
    set.seed(seed)
    n <- n_g * n_per
    g <- rep(seq_len(n_g), each = n_per)
    u <- stats::rnorm(n_g, 0, sig)
    x <- stats::rnorm(n)
    y <- stats::rpois(n, exp(0.9 + 0.4 * x + u[g]))
    y[stats::runif(n) < stats::plogis(-0.5)] <- 0
    d <- data.frame(y = y, x = x, g = factor(g))
    tulpa(y ~ x + (1 | g), data = d, family = "poisson",
          ziformula = ~ 1, sigma_re = sig, mode = "laplace")
  }

  fit <- one(42)
  # The RE block must still be present, and carried by the count process only.
  expect_equal(nrow(ranef(fit)), n_g)
  expect_equal(fit$n_fixed, 3L)

  est <- t(vapply(1:12, function(s) {
    cf <- coef(one(200 + s))
    c(cf[["x"]], cf[["zi_(Intercept)"]])
  }, numeric(2)))
  expect_equal(mean(est[, 1]), 0.4, tolerance = 0.05)
  expect_equal(mean(est[, 2]), -0.5, tolerance = 0.08)
})


test_that("the sampler path carries zero inflation through logit_zi", {
  # The sampler backends reach the mixture through a different channel than the
  # Laplace path (the `logit_zi` callback argument rather than process 1), so
  # agreement between the two is the check that both channels are live: a
  # sampler that silently dropped the mixture would sit at the plain-Poisson
  # estimate, which the second assertion pins away from.
  skip_on_cran()
  d <- sim_zip(2500, c(1.1, 0.45), c(-0.2, 0.0), seed = 21)

  lap <- coef(tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ 1,
                    mode = "laplace"))
  smp <- tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ 1,
               mode = "hmc",
               control = list(n_iter = 700, n_warmup = 350, n_chains = 2,
                              seed = 4))
  cs <- coef(smp)

  expect_equal(unname(cs[["(Intercept)"]]), unname(lap[["(Intercept)"]]),
               tolerance = 0.08)
  expect_equal(unname(cs[["x"]]), unname(lap[["x"]]), tolerance = 0.08)
  # And away from the estimate a mixture-blind fit would give.
  plain <- coef(tulpa(y ~ x, data = d, family = "poisson", mode = "laplace"))
  expect_gt(abs(unname(cs[["(Intercept)"]]) - unname(plain[["(Intercept)"]])),
            0.1)
})


test_that("the front door refuses zero inflation it cannot fit", {
  d <- data.frame(y = c(0, 1, 2, 3), x = c(1, 2, 3, 4), z = c(1, 0, 1, 0))
  # Continuous family: no atom at zero.
  expect_error(tulpa(y ~ x, data = d, family = "gaussian", ziformula = ~ 1),
               "no probability atom at zero")
  # Count family with an R-only kernel.
  expect_error(tulpa(y ~ x, data = d, family = "beta_binomial",
                     n_trials = rep(5, 4), ziformula = ~ 1),
               "no compiled kernel")
  # Random effects in the zi predictor are not carried.
  expect_error(tulpa(y ~ x, data = d, family = "poisson",
                     ziformula = ~ (1 | z)),
               "fixed effects only")
  # A backend that would ignore the mixture entirely.
  expect_error(tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ 1,
                     mode = "gibbs"),
               "not carried by backend")
})
