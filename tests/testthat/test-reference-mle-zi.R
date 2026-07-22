# Zero-inflation and hurdle mixtures against external MLE implementations.
#
# The mixture in R/family_zi.R adds a second linear predictor and two branches
# to the log-density, so it is checked against pscl::zeroinfl (the same ZI
# mixture, by ML) and against glmmTMB's truncated family with a `ziformula`
# (the same hurdle two-part model). Both the count-side and the zero-side
# coefficients are compared -- a sign error in the logit linkage moves only the
# latter.

test_that("zero-inflated poisson reproduces pscl::zeroinfl", {
  skip_on_cran()
  skip_if_not_installed("pscl")
  set.seed(42)
  N <- 800L
  x <- rnorm(N); z <- rnorm(N)
  pi_zi <- plogis(-0.5 + 0.8 * z)
  y <- ifelse(rbinom(N, 1, pi_zi) == 1, 0L, rpois(N, exp(1.5 + 0.6 * x)))
  d <- data.frame(y = y, x = x, z = z)

  ref <- pscl::zeroinfl(y ~ x | z, data = d, dist = "poisson")
  co  <- summary(ref)$coefficients

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "poisson", ziformula = ~ z,
          mode = "laplace",
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD),
          zi_prior = list(sd = REF_DIFFUSE_SD)))

  expect_identical(names(coef(fit)),
                   c("(Intercept)", "x", "zi_(Intercept)", "zi_z"))
  expect_agrees_with_mle(
    coef(fit),
    c(co$count[, "Estimate"],   co$zero[, "Estimate"]),
    c(co$count[, "Std. Error"], co$zero[, "Std. Error"]),
    label = "zi_poisson")
})


test_that("zero-inflated neg_binomial_2 reproduces pscl::zeroinfl", {
  skip_on_cran()
  skip_if_not_installed("pscl")
  skip_if_not_installed("MASS")
  set.seed(2026)
  N <- 800L
  x <- rnorm(N); z <- rnorm(N)
  pi_zi <- plogis(-1.0 + 0.5 * z)
  y <- ifelse(rbinom(N, 1, pi_zi) == 1, 0L,
              MASS::rnegbin(N, mu = exp(2.0 + 0.5 * x), theta = 5))
  d <- data.frame(y = y, x = x, z = z)

  ref <- pscl::zeroinfl(y ~ x | z, data = d, dist = "negbin")
  co  <- summary(ref)$coefficients
  # pscl appends a Log(theta) row to the count table; the regression
  # coefficients are the rows above it.
  n_count <- nrow(co$count) - 1L

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "neg_binomial_2", ziformula = ~ z,
          mode = "laplace", phi = ref$theta,
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD),
          zi_prior = list(sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(
    coef(fit),
    c(co$count[seq_len(n_count), "Estimate"],   co$zero[, "Estimate"]),
    c(co$count[seq_len(n_count), "Std. Error"], co$zero[, "Std. Error"]),
    label = "zi_neg_binomial_2")
})


test_that("hurdle poisson reproduces glmmTMB truncated_poisson + ziformula", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  set.seed(45)
  N <- 800L
  x <- rnorm(N); z <- rnorm(N)
  pi_hurdle <- plogis(-0.4 + 0.7 * z)
  y <- ifelse(runif(N) < pi_hurdle, 0L,
              ref_sim_truncated_poisson(exp(0.7 + 0.4 * x)))
  d <- data.frame(y = y, x = x, z = z)
  expect_true(any(d$y == 0))
  expect_true(any(d$y > 0))

  ref <- glmmTMB::glmmTMB(y ~ x, ziformula = ~ z, data = d,
                          family = glmmTMB::truncated_poisson())
  co  <- summary(ref)$coefficients

  # The mixture over a zero-truncated base is the hurdle model: the base puts
  # no mass at zero, so the y = 0 branch collapses to log(pi) and the two parts
  # separate. It is reached by pairing `ziformula` with a truncated family
  # rather than by a family of its own.
  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "truncated_poisson", ziformula = ~ z,
          mode = "laplace",
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD),
          zi_prior = list(sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(
    coef(fit),
    c(co$cond[, "Estimate"],   co$zi[, "Estimate"]),
    c(co$cond[, "Std. Error"], co$zi[, "Std. Error"]),
    label = "hurdle_poisson")
})


test_that("hurdle neg_binomial_2 reproduces glmmTMB truncated_nbinom2 + ziformula", {
  skip_on_cran()
  skip_if_not_installed("glmmTMB")
  set.seed(46)
  N <- 800L
  x <- rnorm(N); z <- rnorm(N)
  pi_hurdle <- plogis(-0.3 + 0.6 * z)
  y <- ifelse(runif(N) < pi_hurdle, 0L,
              ref_sim_truncated_nb2(exp(0.5 + 0.4 * x), 2))
  d <- data.frame(y = y, x = x, z = z)

  ref   <- glmmTMB::glmmTMB(y ~ x, ziformula = ~ z, data = d,
                            family = glmmTMB::truncated_nbinom2())
  co    <- summary(ref)$coefficients
  theta <- exp(glmmTMB::fixef(ref)$disp)

  fit <- suppressMessages(
    tulpa(y ~ x, data = d, family = "truncated_neg_binomial_2",
          ziformula = ~ z, mode = "laplace", phi = theta,
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD),
          zi_prior = list(sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(
    coef(fit),
    c(co$cond[, "Estimate"],   co$zi[, "Estimate"]),
    c(co$cond[, "Std. Error"], co$zi[, "Std. Error"]),
    label = "hurdle_neg_binomial_2")
})
