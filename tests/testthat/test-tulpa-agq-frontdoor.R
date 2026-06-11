# Front-door routing of adaptive Gauss-Hermite quadrature. A single
# random-intercept `(1 | g)` model with mode = "agq" reaches agq_fit() through
# tulpa(); the front door must map the bundle + control knobs onto agq_fit()'s
# signature so the result equals a direct call.

make_agq_re_data <- function(seed = 401L, G = 40L, npg = 8L,
                             beta = c(0.4, 0.8), sigma = 0.7) {
  set.seed(seed)
  N <- G * npg
  g <- factor(rep(seq_len(G), each = npg))
  x <- rnorm(N)
  u <- rnorm(G, 0, sigma)
  eta <- beta[1] + beta[2] * x + u[as.integer(g)]
  data.frame(y = rbinom(N, 1L, plogis(eta)), x = x, g = g)
}


test_that("agq is registered, reachable, and a valid front-door backend", {
  expect_true("agq" %in% ALL_BACKENDS)
  expect_true(backend_is_reachable("agq"))
  expect_equal(get_backend_tier("agq")$tier, 2L)
})


test_that("mode='agq' routes a (1 | g) model to agq_fit and matches a direct call", {
  skip_on_cran()
  d <- make_agq_re_data()

  fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial",
               mode = "agq", control = list(n_quad = 7L))

  expect_equal(fit$backend, "agq")
  expect_equal(fit$inference_tier, 2L)
  expect_s3_class(fit, "tulpa_fit")
  expect_equal(fit$n_quad, 7L)

  # Front-door fit equals a direct agq_fit() on the same design (deterministic
  # BFGS from the same init): proves the bundle -> arg mapping is faithful.
  direct <- agq_fit(d$y, cbind(1, d$x), as.integer(d$g),
                    family = "binomial", n_quad = 7L)
  expect_equal(unname(fit$means), unname(direct$means), tolerance = 1e-8)
  expect_equal(fit$sigma_re, direct$sigma_re, tolerance = 1e-8)
  expect_equal(fit$log_marginal, direct$log_marginal, tolerance = 1e-8)

  # Generic accessors report the two fixed effects with real SEs from $cov.
  expect_length(coef(fit), 2L)
  expect_equal(dim(vcov(fit)), c(2L, 2L))
  expect_equal(nrow(confint(fit)), 2L)
  expect_true(all(is.finite(summary(fit)$std.error)))
})


test_that("n_quad control threads through (n_quad=1 front door == Laplace)", {
  skip_on_cran()
  d <- make_agq_re_data(seed = 402L, G = 25L, npg = 6L)

  fit1 <- tulpa(y ~ x + (1 | g), data = d, family = "binomial",
                mode = "agq", control = list(n_quad = 1L))
  direct1 <- agq_fit(d$y, cbind(1, d$x), as.integer(d$g),
                     family = "binomial", n_quad = 1L)

  expect_equal(fit1$n_quad, 1L)
  expect_equal(unname(fit1$means), unname(direct1$means), tolerance = 1e-8)
})


test_that("gaussian agq maps phi to the residual sd (sigma_eps = sqrt(phi))", {
  skip_on_cran()
  set.seed(403L)
  G <- 30L; npg <- 8L; N <- G * npg
  g <- factor(rep(seq_len(G), each = npg))
  x <- rnorm(N)
  u <- rnorm(G, 0, 0.6)
  eta <- 0.3 + 0.7 * x + u[as.integer(g)]
  sig_eps <- 0.5
  d <- data.frame(y = eta + rnorm(N, 0, sig_eps), x = x, g = g)

  fit <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian",
               mode = "agq", phi = sig_eps^2, control = list(n_quad = 7L))
  direct <- agq_fit(d$y, cbind(1, d$x), as.integer(d$g),
                    family = "gaussian", n_quad = 7L, sigma_eps = sig_eps)

  expect_equal(fit$backend, "agq")
  expect_equal(unname(fit$means), unname(direct$means), tolerance = 1e-8)
})


test_that("mode='agq' rejects random slopes, multiple terms, and bad families", {
  d <- make_agq_re_data(G = 20L, npg = 6L)
  d$g2 <- factor(rep(seq_len(5L), length.out = nrow(d)))

  expect_error(
    tulpa(y ~ x + (1 + x | g), data = d, family = "binomial", mode = "agq"),
    "exactly one random-intercept")
  expect_error(
    tulpa(y ~ x + (1 | g) + (1 | g2), data = d, family = "binomial",
          mode = "agq"),
    "exactly one random-intercept")
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "neg_binomial_2", mode = "agq"),
    "AGQ supports family")
})


test_that("mode='agq' rejects a fixed-effect prior (marginal-likelihood fit)", {
  d <- make_agq_re_data(G = 20L, npg = 6L)
  expect_error(
    tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "agq",
          beta_prior = list(mean = 0, sd = 2.5)),
    "not supported on the AGQ path")
})
