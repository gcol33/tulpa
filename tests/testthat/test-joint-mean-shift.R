# The joint door's fixed effects are the mixture over cells of each cell's
# posterior mean to leading order, mode_k + delta_k with
# delta_k = (1/2) Sigma_k A' (l3 * s_k) -- the same shift
# tulpa_re_cov_nested() centres its nodes with -- so coef(), the draws and
# confint() describe one centre.

.jms_data <- function(seed = 102L, G = 30L, npg = 8L) {
  set.seed(seed)
  N <- G * npg
  g <- rep(seq_len(G), each = npg)
  x <- rnorm(N)
  S <- matrix(c(0.64, 0.24, 0.24, 0.36), 2)
  u <- t(t(chol(S)) %*% matrix(rnorm(2 * G), 2))
  eta <- -0.3 + 0.7 * x + u[g, 1] + u[g, 2] * x
  data.frame(y = rbinom(N, 1L, plogis(eta)), x = x, g = factor(g))
}

test_that("a joint fit's coefficients are the mixture of the cells' shifted modes", {
  skip_if_fast()
  d <- .jms_data()
  f <- tulpa(y ~ x + (1 + x | g), d, family = "binomial",
             mode = "nested_laplace_joint", control = list(diagnose_k = FALSE))
  expect_true(is.na(f$grid_mean_shift_declined))
  p <- 2L
  expect_true(tulpa:::.nl_skew_correction(f, p)$centre_in_components)
  el <- f$skew_correction$eligible
  expect_true(all(el))
  w <- f$weights / sum(f$weights)
  mean_shifted <- colSums(w * (f$modes[, seq_len(p)] +
                                 f$grid_mean_shift[, seq_len(p)]))
  expect_equal(unname(coef(f)), unname(mean_shifted), tolerance = 1e-8)
  # The covariance is the law of total variance over the shifted components.
  mom <- tulpa:::.nested_fixed_moments(f)
  expect_equal(unname(colSums(mom$w * mom$mu)), unname(mean_shifted),
               tolerance = 1e-8)
  expect_equal(unname(vcov(f)), unname(mom$cov), tolerance = 1e-10)
  # The shift is not negligible on a binary logit with eight rows per group.
  mode_mix <- colSums(w * f$modes[, seq_len(p)])
  se <- sqrt(diag(vcov(f)))
  expect_gt(max(abs(coef(f) - mode_mix) / se), 0.1)
})

test_that("the joint door and tulpa_re_cov_nested() report one centre", {
  skip_if_fast()
  d <- .jms_data()
  ctl <- list(diagnose_k = FALSE)
  fj <- tulpa(y ~ x + (1 + x | g), d, family = "binomial",
              mode = "nested_laplace_joint", control = ctl)
  fr <- tulpa(y ~ x + (1 + x | g), d, family = "binomial",
              mode = "structured", control = ctl)
  expect_identical(fr$backend, "re_cov_nested")
  se <- sqrt(diag(vcov(fr)))
  expect_lt(max(abs(coef(fj) - coef(fr)) / se), 0.1)
})

test_that("skew_correct = FALSE keeps the mode mixture; a gaussian has no shift", {
  skip_if_fast()
  d <- .jms_data()
  off <- tulpa(y ~ x + (1 + x | g), d, family = "binomial",
               mode = "nested_laplace_joint",
               control = list(diagnose_k = FALSE, skew_correct = FALSE))
  expect_null(off$grid_mean_shift)
  w <- off$weights / sum(off$weights)
  expect_equal(unname(coef(off)), unname(colSums(w * off$modes[, 1:2])),
               tolerance = 1e-8)

  set.seed(8)
  G <- 20L; g <- rep(seq_len(G), each = 8L); x <- rnorm(G * 8L)
  dg <- data.frame(y = 0.3 + 0.5 * x + rnorm(G, 0, 0.6)[g] + rnorm(G * 8L, 0, 0.5),
                   x = x, g = factor(g))
  on_g  <- tulpa(y ~ x + (1 | g), dg, family = "gaussian", mode = "structured",
                 control = list(diagnose_k = FALSE))
  off_g <- tulpa(y ~ x + (1 | g), dg, family = "gaussian", mode = "structured",
                 control = list(diagnose_k = FALSE, skew_correct = FALSE))
  expect_equal(max(abs(on_g$grid_mean_shift)), 0)
  expect_equal(coef(on_g), coef(off_g), tolerance = 1e-12)
})
