# test-tulpa-entry.R
# Integration contract for the unified tulpa() entry: formula -> bundle ->
# backend selection -> per-backend arg assembly -> dispatch. Verifies routing,
# the fail-loud reachability contract, and the documented coverage boundaries.

make_pois_re <- function(seed = 1, ng = 8L, per = 10L) {
  set.seed(seed)
  g <- rep(seq_len(ng), each = per)
  x <- rnorm(ng * per)
  eta <- 0.3 + 0.5 * x + rnorm(ng, 0, 0.6)[g]
  data.frame(y = rpois(ng * per, exp(eta)), x = x, g = factor(g))
}

test_that("tulpa() routes a random-intercept GLMM through the Laplace path", {
  d <- make_pois_re()
  fit <- tulpa(y ~ x + (1 | g), d, family = "poisson",
               mode = "laplace", sigma_re = 0.6)
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "laplace")
  expect_equal(fit$inference_tier, 2L)
  expect_identical(fit$family, "poisson")
  expect_true(inherits(fit$formula, "formula"))
})

test_that("tulpa() routes through a sampler (logpost) backend", {
  d <- make_pois_re()
  fit <- tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "mala",
               sigma_re = 0.6, control = list(n_iter = 400L, warmup = 200L))
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "mala")
  expect_equal(fit$inference_tier, 1L)
})

test_that("tulpa() fails loudly on a C-ABI-only backend", {
  d <- make_pois_re()
  err <- expect_error(tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "ess"))
  expect_match(conditionMessage(err), "no R-level fitter")
  expect_match(conditionMessage(err), "tulpa_run_ess_sampler")
})

test_that("correlated random slopes: logpost path works, Laplace path integrates Sigma", {
  d <- make_pois_re(seed = 2)
  # logpost path handles slopes (builder is general)
  fit <- tulpa(y ~ x + (1 + x | g), d, family = "poisson", mode = "mala",
               sigma_re = 0.5, control = list(n_iter = 300L, warmup = 150L))
  expect_equal(fit$backend, "mala")
  # Laplace path now routes a single correlated `(1 + x | g)` term to the
  # nested-Laplace Sigma integrator (no scalar sigma_re to condition on).
  fit2 <- tulpa(y ~ x + (1 + x | g), d, family = "poisson", mode = "laplace",
                control = list(seed = 1L, n_draws = 300L))
  expect_equal(fit2$backend, "re_cov_nested")
  expect_equal(fit2$inference_tier, 2L)
  expect_setequal(fit2$posterior$parameter,
                  c("sigma_1", "sigma_2", "rho_12",
                    "Sigma_11", "Sigma_12", "Sigma_22"))
  # the exact debias is reachable via control$re_cov
  fit3 <- tulpa(y ~ x + (1 + x | g), d, family = "poisson", mode = "laplace",
                control = list(re_cov = "gibbs", n_iter = 400L, n_burnin = 200L,
                               seed = 1L))
  expect_equal(fit3$backend, "re_cov_gibbs")
})

test_that("tulpa() validates family and defaults sigma_re with a message", {
  d <- make_pois_re()
  expect_error(tulpa(y ~ x + (1 | g), d, family = "weibull"), "Unknown family")
  expect_message(
    tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "laplace"),
    "sigma_re = 1"
  )
})

test_that("tulpa() handles a no-RE model on the design path", {
  set.seed(5)
  d <- data.frame(y = rpois(120, exp(0.4 + 0.3 * rnorm(120))), x = rnorm(120))
  fit <- tulpa(y ~ x, d, family = "poisson", mode = "laplace")
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "laplace")
})

test_that("tulpa() routes through imh_laplace (Laplace proposal + MH)", {
  d <- make_pois_re()
  fit <- tulpa(y ~ x + (1 | g), d, family = "poisson", mode = "imh_laplace",
               sigma_re = 0.6, control = list(n_iter = 400L, warmup = 200L))
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "imh_laplace")
  expect_equal(fit$inference_tier, 1L)
  expect_true(is.finite(fit$mean_accept))
})

make_binom_re <- function(seed = 3, ng = 8L, per = 12L, nt = 10L) {
  set.seed(seed)
  g <- rep(seq_len(ng), each = per)
  x <- rnorm(ng * per)
  eta <- -0.2 + 0.6 * x + rnorm(ng, 0, 0.5)[g]
  list(
    d = data.frame(y = rbinom(ng * per, nt, plogis(eta)), x = x, g = factor(g)),
    nt = rep(nt, ng * per)
  )
}

test_that("tulpa() routes a binomial random-intercept model through Gibbs", {
  s <- make_binom_re()
  fit <- tulpa(y ~ x + (1 | g), s$d, family = "binomial", mode = "gibbs",
               n_trials = s$nt, control = list(iter = 400L, warmup = 200L))
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$backend, "gibbs")
  expect_equal(fit$inference_tier, 1L)
})

test_that("Gibbs rejects unsupported families and non-intercept RE structure", {
  s <- make_binom_re()
  # gaussian not supported by the PG sampler
  err1 <- expect_error(
    tulpa(y ~ x + (1 | g), s$d, family = "gaussian", mode = "gibbs")
  )
  expect_match(conditionMessage(err1), "binomial.*neg_binomial_2")
  # random slope: more than a single intercept term
  err2 <- expect_error(
    tulpa(y ~ x + (1 + x | g), s$d, family = "binomial", mode = "gibbs",
          n_trials = s$nt)
  )
  expect_match(conditionMessage(err2), "one random-intercept")
})
