# Random-effect structures against lme4.
#
# lme4 maximizes the same Laplace marginal likelihood over the variance
# components, so an empirical-Bayes tulpa fit with the dispersion free is
# fitting the identical objective and both halves of its answer are checkable:
# the fixed effects against lmer/glmer's, and the variance components against
# VarCorr(). REML = FALSE throughout, since the empirical-Bayes maximization is
# over the ML marginal.
#
# `estimate_phi = TRUE` is what makes the Gaussian comparisons meaningful: with
# the residual variance held at its default the outer maximization absorbs the
# mismatch into the random-effect scales, and the variance components are no
# longer comparable to lme4's.

lme4_sd <- function(ref, grp) {
  vc <- as.data.frame(lme4::VarCorr(ref))
  vc$sdcor[vc$grp == grp & is.na(vc$var2)]
}


test_that("gaussian (1 | g) reproduces lme4::lmer", {
  skip_on_cran()
  skip_if_not_installed("lme4")
  set.seed(11)
  G <- 12L; n_per <- 18L; N <- G * n_per
  g <- factor(rep(LETTERS[seq_len(G)], each = n_per))
  x <- rnorm(N)
  u <- rnorm(G, 0, 0.7)
  d <- data.frame(y = 1.0 + 0.5 * x + u[as.integer(g)] + rnorm(N, 0, 0.5),
                  x = x, g = g)

  ref    <- lme4::lmer(y ~ x + (1 | g), data = d, REML = FALSE)
  ref_b  <- lme4::fixef(ref)
  ref_se <- sqrt(diag(as.matrix(stats::vcov(ref))))

  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = TRUE))

  expect_true(fit$phi_estimated)
  expect_agrees_with_mle(coef(fit), ref_b, ref_se, tol = 0.05,
                         label = "lmer (1|g)")
  expect_lt(abs(sqrt(fit$phi) - stats::sigma(ref)) / stats::sigma(ref), 0.02)

  vc <- VarCorr(fit)
  expect_lt(abs(vc$sd[vc$term == "g"] - lme4_sd(ref, "g")) /
              lme4_sd(ref, "g"), 0.15)
})


test_that("gaussian nested (1 | g1/g2) reproduces lme4::lmer", {
  skip_on_cran()
  skip_if_not_installed("lme4")
  set.seed(42)
  n_g1 <- 12L; n_g2 <- 6L; n_per <- 4L
  d <- expand.grid(g2 = factor(seq_len(n_g2)), g1 = factor(seq_len(n_g1)),
                   rep = seq_len(n_per))
  d$x <- rnorm(nrow(d))
  g12 <- interaction(d$g1, d$g2, drop = TRUE)
  b1  <- rnorm(n_g1, sd = 0.7)
  b12 <- rnorm(nlevels(g12), sd = 0.4)
  d$y <- rnorm(nrow(d),
               1 + 0.5 * d$x + b1[as.integer(d$g1)] + b12[as.integer(g12)],
               0.5)

  ref    <- lme4::lmer(y ~ x + (1 | g1/g2), data = d, REML = FALSE)
  ref_b  <- lme4::fixef(ref)
  ref_se <- sqrt(diag(as.matrix(stats::vcov(ref))))

  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g1/g2), data = d, family = "gaussian", mode = "eb",
          estimate_phi = TRUE))

  # (1 | g1/g2) expands to (1 | g1) + (1 | g1:g2); lme4 labels the inner term
  # the other way round, as g2:g1.
  expect_identical(VarCorr(fit)$term, c("g1", "g1:g2"))
  expect_agrees_with_mle(coef(fit), ref_b, ref_se, tol = 0.05,
                         label = "lmer (1|g1/g2)")
  expect_lt(abs(sqrt(fit$phi) - stats::sigma(ref)) / stats::sigma(ref), 0.02)

  vc <- VarCorr(fit)
  expect_lt(abs(vc$sd[vc$term == "g1"]    - lme4_sd(ref, "g1"))    /
              lme4_sd(ref, "g1"), 0.15)
  expect_lt(abs(vc$sd[vc$term == "g1:g2"] - lme4_sd(ref, "g2:g1")) /
              lme4_sd(ref, "g2:g1"), 0.20)
})


test_that("gaussian correlated slopes (x | g) reproduce lme4::lmer", {
  skip_on_cran()
  skip_if_not_installed("lme4")
  set.seed(11)
  G <- 20L; n_per <- 25L; N <- G * n_per
  g <- factor(rep(LETTERS[seq_len(G)], each = n_per))
  x <- rnorm(N)
  Sigma <- matrix(c(0.49, 0.14, 0.14, 0.16), 2L, 2L)
  b <- t(chol(Sigma)) %*% matrix(rnorm(2L * G), nrow = 2L)
  d <- data.frame(
    y = 1 + 0.5 * x + b[1L, as.integer(g)] + b[2L, as.integer(g)] * x +
      rnorm(N, 0, 0.5),
    x = x, g = g)

  ref    <- lme4::lmer(y ~ x + (x | g), data = d, REML = FALSE)
  ref_b  <- lme4::fixef(ref)
  ref_se <- sqrt(diag(as.matrix(stats::vcov(ref))))
  vc_ref <- as.data.frame(lme4::VarCorr(ref))

  fit <- suppressMessages(
    tulpa(y ~ x + (x | g), data = d, family = "gaussian", mode = "eb",
          estimate_phi = TRUE))

  expect_agrees_with_mle(coef(fit), ref_b, ref_se, tol = 0.05,
                         label = "lmer (x|g)")
  expect_lt(abs(sqrt(fit$phi) - stats::sigma(ref)) / stats::sigma(ref), 0.02)

  # A correlated block reports its scales and the correlation between them;
  # `map` carries the full covariance the outer maximization landed on.
  ref_sd  <- vc_ref$sdcor[vc_ref$grp == "g" & is.na(vc_ref$var2)]
  ref_rho <- vc_ref$sdcor[vc_ref$grp == "g" & !is.na(vc_ref$var2)]
  expect_lt(max(abs(fit$map$sigma - ref_sd) / ref_sd), 0.15)
  expect_lt(abs(fit$map$rho - ref_rho), 0.15)
})


test_that("poisson (1 | g) reproduces lme4::glmer at its variance component", {
  skip_on_cran()
  skip_if_not_installed("lme4")
  set.seed(21)
  G <- 12L; n_per <- 18L; N <- G * n_per
  g <- factor(rep(LETTERS[seq_len(G)], each = n_per))
  x <- rnorm(N)
  u <- rnorm(G, 0, 0.5)
  d <- data.frame(y = rpois(N, exp(0.5 + 0.4 * x + u[as.integer(g)])),
                  x = x, g = g)

  ref    <- suppressMessages(
    lme4::glmer(y ~ x + (1 | g), data = d, family = stats::poisson()))
  ref_b  <- lme4::fixef(ref)
  ref_se <- sqrt(diag(as.matrix(stats::vcov(ref))))

  # glmer's default nAGQ = 1 is the same Laplace approximation to the marginal
  # likelihood, so conditioning on its variance component compares the two
  # inner solves directly. The gap that remains is tulpa's hyperprior on the
  # random-effect scale, which glmer does not carry.
  fit <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "laplace",
          sigma_re = lme4_sd(ref, "g"),
          beta_prior = list(mean = 0, sd = REF_DIFFUSE_SD)))

  expect_agrees_with_mle(coef(fit), ref_b, ref_se, tol = 0.15,
                         label = "glmer (1|g)")

  fit_eb <- suppressMessages(
    tulpa(y ~ x + (1 | g), data = d, family = "poisson", mode = "eb"))
  expect_agrees_with_mle(coef(fit_eb), ref_b, ref_se, tol = 0.15,
                         label = "glmer (1|g) eb")
  vc <- VarCorr(fit_eb)
  expect_lt(abs(vc$sd[vc$term == "g"] - lme4_sd(ref, "g")) /
              lme4_sd(ref, "g"), 0.25)
})
