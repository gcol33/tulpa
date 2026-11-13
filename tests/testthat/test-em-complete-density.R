test_that("complete-data Gaussian density retains likelihood and prior normalizers", {
  blocks <- list(a = list(y = c(.2, 1), X = matrix(1, 2, 1), family = "gaussian",
                          phi = .4, offset = c(.1, -.2), weights = c(1, 0)))
  bp <- list(mean = .7, sd = .8)
  density <- .em_complete_density(blocks, beta_prior = bp)
  theta <- -.3
  expected <- dnorm(.2, theta + .1, sqrt(.4), log = TRUE) +
    dnorm(theta, bp$mean, bp$sd, log = TRUE)
  expect_equal(density(theta), expected, tolerance = 1e-13)
  marginal <- integrate(function(t) vapply(t, function(v) exp(density(v)), 0),
                         -Inf, Inf, rel.tol = 1e-10)$value
  expect_equal(marginal, dnorm(.2, bp$mean + .1, sqrt(.4 + bp$sd^2)),
               tolerance = 1e-9)
  blocks$a$beta_prior <- list(mean = -1, sd = .3)
  expect_equal(.em_complete_density(blocks, bp)(theta),
                dnorm(.2, theta + .1, sqrt(.4), log = TRUE) +
                  dnorm(theta, -1, .3, log = TRUE), tolerance = 1e-13)
})

test_that("complete-data count density uses the native family constants", {
  theta <- .8
  cases <- list(
    list(family = "poisson", y = c(0, 3), phi = 1,
         expected = dpois(c(0, 3), exp(theta), log = TRUE)),
    list(family = "binomial", y = c(0, 3), n_trials = 5L, phi = 1,
         expected = dbinom(c(0, 3), 5, plogis(theta), log = TRUE)),
    list(family = "negbin", y = c(0, 3), phi = 1.3,
         expected = dnbinom(c(0, 3), size = 1.3, mu = exp(theta), log = TRUE)),
    list(family = "gamma", y = c(.4, 1.1), phi = 1.3,
         expected = dgamma(c(.4, 1.1), shape = 1.3, rate = 1.3 / exp(theta), log = TRUE)),
    list(family = "beta", y = c(.2, .8), phi = 3.2,
         expected = dbeta(c(.2, .8), 3.2 * plogis(theta),
                           3.2 * plogis(-theta), log = TRUE)))
  for (case in cases) {
    expected <- sum(case$expected) + dnorm(theta, 0, 2.5, log = TRUE)
    case$expected <- NULL
    case$X <- matrix(1, 2, 1)
    expect_equal(.em_complete_density(list(a = case))(theta), expected,
                 tolerance = 1e-12)
  }
})

test_that("complete-data density uses full correlated and crossed RE priors", {
  L <- matrix(c(.8, .3, 0, .6), 2, 2)
  block <- list(y = c(0, 2, 1), X = matrix(1, 3, 1), family = "poisson",
                re_list = list(
                  list(idx = c(1L, 2L, 0L), n_groups = 2L, n_coefs = 2L,
                       Z = cbind(1, c(-1, .4, 2)), L = L),
                  list(idx = c(1L, 1L, 2L), n_groups = 2L, sigma = .5)))
  theta <- c(.2, .1, -.3, -.2, .4, .1, -.1)
  eta <- c(.2 + .1 + .3 + .1, .2 - .2 + .4 * .4 + .1, .2 - .1)
  u <- matrix(theta[2:5], 2, 2)
  expected <- sum(dpois(block$y, exp(eta), log = TRUE)) +
    dnorm(theta[1], 0, 2.5, log = TRUE) +
    sum(dnorm(theta[6:7], 0, .5, log = TRUE)) -
    2 * (log(2 * pi) + sum(log(diag(L)))) - sum(solve(L, u)^2) / 2
  expect_equal(.em_complete_density(list(a = block))(theta), expected,
               tolerance = 1e-12)
  block$re_list[[1]]$L <- NULL
  block$re_list[[1]]$cov <- tcrossprod(L)
  expect_equal(.em_complete_density(list(a = block))(theta), expected,
               tolerance = 1e-12)
})

test_that("hard draws with no active observations retain the proper prior", {
  b <- list(y = c(0, 1), X = matrix(1, 2, 1), family = "poisson", weights = c(0, 0))
  expect_equal(.em_complete_density(list(a = b))(.3), dnorm(.3, 0, 2.5, log = TRUE))
  b$weights <- c(.5, 1)
  expect_error(.em_complete_density(list(a = b)), "hard 0/1")
  b$weights <- NULL
  expect_error(.em_complete_density(list(a = b), list(sd = Inf)), "proper finite")
  expect_identical(.em_complete_density(list(a = b))(Inf), -Inf)
})

test_that("complete-data blocks sum to the observed ZIP density and its exact E-step", {
  skip_on_cran_budget()
  source(test_path("fixtures", "em_exact_zip.R"), local = TRUE)
  fixture <- .em_zip_fixture()
  zero <- which(fixture$y == 0)
  patterns <- as.matrix(expand.grid(rep(list(0:1), length(zero))))
  theta <- c(.3, -.4, -.2, .5, -.1)
  log_joint <- vapply(seq_len(nrow(patterns)), function(i) {
    z <- rep(1L, length(fixture$y))
    z[zero] <- patterns[i, ]
    .em_complete_density(fixture$encode(z), fixture$beta_prior)(theta)
  }, 0)
  log_marginal <- .tulpa_logsumexp(log_joint)
  expect_equal(log_marginal, fixture$log_marginal(theta), tolerance = 1e-12)
  layout <- .em_complete_layout(fixture$encode(rep(1L, length(fixture$y))))
  weights <- fixture$e_step(.em_fits_at(theta, layout))$weights
  conditional <- apply(patterns, 1L, function(z) prod(dbinom(z, 1, weights[zero])))
  expect_equal(exp(log_joint - log_marginal), conditional, tolerance = 1e-12)
})

test_that("shared MH ratio matches the conjugate posterior ratio", {
  y <- .7
  variance <- 1 / (1 / .8^2 + 1 / .4)
  mu <- variance * (y / .4 + .2 / .8^2)
  proposal <- .imh_laplace_proposal(mu, matrix(1 / variance, 1, 1))
  evaluate <- function(t) list(log_prob = dnorm(y, t, sqrt(.4), log = TRUE) +
                                dnorm(t, .2, .8, log = TRUE), auxiliary = t)
  current <- -.4
  candidate <- .8
  step <- .imh_laplace_transition(current, evaluate(current), proposal, evaluate,
                                   proposed = candidate)
  log_q <- function(t) dt((t - mu) / sqrt(variance), df = .imh_proposal("df"),
                           log = TRUE) - log(sqrt(variance))
  expected <- dnorm(candidate, mu, sqrt(variance), log = TRUE) -
    dnorm(current, mu, sqrt(variance), log = TRUE) + log_q(current) - log_q(candidate)
  expect_equal(step$log_alpha, expected, tolerance = 1e-13)
  state <- evaluate(current)
  rejected <- .imh_laplace_transition(current, state, proposal,
                function(t) list(log_prob = -Inf, auxiliary = 999), proposed = candidate)
  expect_false(rejected$accepted)
  expect_identical(rejected$state, state)
  expect_identical(rejected$theta, current)
})

test_that("exact controls reject deterministic parameter updates and invalid settings", {
  e <- function(fits, ...) list(weights = .5)
  m <- function(z, ...) list(a = list(y = z, X = matrix(1, 1, 1), family = "gaussian"))
  expect_error(tulpa_em_laplace(e, m, correction = "gibbs", m_step_extra = identity),
                "deterministic")
  expect_error(tulpa_em_mc(e, m, correction = "gibbs", log_complete = 3), "log_complete")
  expect_error(.em_gibbs_control(3, list(warmup = 3)), "warmup")
  expect_error(.em_gibbs_control(3, list(mh_steps = 1.5)), "mh_steps")
  expect_error(.em_gibbs_control(3, list(thin = 0)), "thin")
  expect_error(.em_gibbs_control(3, list(scale = Inf)), "scale")
  expect_error(.em_gibbs_control(3, list(unknown = 1)), "accepts")
  expect_error(.em_gibbs_control(3, list(scale = 1, scale = 2)), "accepts")
  expect_error(.em_gibbs_control(3.1, list()), "n_gibbs")
  expect_error(.draw_z_default(c(-.1, .2)), "finite Bernoulli probabilities")
  expect_error(.draw_z_default(c(.1, Inf)), "finite Bernoulli probabilities")
  expect_error(tulpa_em_mc(e, m, log_complete = identity), "require correction")
  b <- m(1)
  b$a$prior <- list(type = "iid")
  expect_error(.em_complete_layout(b), "full conditional parameter proposal")
})

test_that("Gibbs stores paired states and re-evaluates theta after each Z draw", {
  skip_on_cran()
  initial <- list(a = list(mode = 0, beta = 0))
  seen <- list()
  z_count <- 0L
  e <- function(fits, ...) {
    seen[[length(seen) + 1L]] <<- fits$a$mode
    z_count <<- z_count + 1L
    list(weights = (-1)^z_count)
  }
  m <- function(z, ...) list(a = list(y = z, X = matrix(1, 1, 1), family = "gaussian"))
  evaluated <- list()
  density <- function(fits, z, ...) {
    evaluated[[length(evaluated) + 1L]] <<- c(fits$a$mode, z)
    dnorm(z, fits$a$mode, 1, log = TRUE) + dnorm(fits$a$mode, 0, 2.5, log = TRUE)
  }
  set.seed(21)
  fit <- .gibbs_correction(initial, e, m, identity, n_gibbs = 6L, verbose = FALSE,
                            log_complete = density,
                            correction_control = list(warmup = 0L, mh_steps = 2L))
  expect_equal(length(evaluated), 18L)
  expect_equal(seen[-1L], as.list(unname(fit$draws[1:5, 1])))
  expected <- vapply(1:6, function(i) {
    dnorm(fit$latent_draws[[i]], fit$draws[i, 1], 1, log = TRUE) +
      dnorm(fit$draws[i, 1], 0, 2.5, log = TRUE)
  }, 0)
  expect_equal(fit$log_prob, expected)
  expect_identical(fit$inference_tier, 1L)
  expect_identical(fit$draws_kind, "chain")
  expect_match(fit$target, "joint posterior")
  expect_true(any(abs(diff(fit$draws[, 1])) > 0))
  expect_equal(vapply(evaluated[seq(1, 18, 3)], `[`, 0, 1),
                c(0, fit$draws[1:5, 1]), ignore_attr = TRUE)
})

test_that("MCEM and EM share the exact correction and preserve Tier 2 summaries", {
  skip_on_cran()
  e <- function(fits, ...) list(weights = 1)
  es <- function(fits, n_mc, ...) rep(list(1L), n_mc)
  m <- function(z, ...) list(a = list(y = c(.2, .8), X = matrix(1, 2, 1),
                                     family = "gaussian"))
  set.seed(25)
  a <- tulpa_em_laplace(e, m, correction = "gibbs", n_gibbs = 6L,
                         max_iter = 1L, n_louis = 0L, verbose = FALSE)
  set.seed(25)
  b <- tulpa_em_mc(es, m, correction = "gibbs", n_gibbs = 6L,
                    n_mc = 2L, max_iter = 1L, verbose = FALSE)
  expect_identical(a$draws, b$draws)
  expect_identical(a$latent_draws, b$latent_draws)
  expect_identical(b$inference_tier, 1L)
  expect_identical(b$pooled_target, "Rubin-pooled complete-data Laplace estimates")
  expect_length(b$em_draws, 2L)
  expect_null(a$pooled)
  expect_equal(nrow(b$draws), 3L)
  expect_equal(b$complete_density_evaluations, 12L)
  expect_equal(b$vcov, cov(b$draws))
  expect_equal(b$se$a, b$sds)
  expect_equal(b$means, colMeans(b$draws))
  expect_null(a$louis)
  plain <- tulpa_em_mc(es, m, n_mc = 2L, max_iter = 1L, verbose = FALSE)
  expect_identical(plain$inference_tier, 2L)
  expect_identical(plain$target, b$pooled_target)
})
