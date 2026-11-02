# What the front door reports about a fit and about a call it cannot make:
# an unconverged mode-find, the dispersion of a fixed-effect gaussian model,
# the spread of Laplace group effects, sampler draw labels, and the errors
# that name the door that works.

test_that("a formula variable missing from data is named", {
  d <- data.frame(y = rnorm(10), x = rnorm(10))
  expect_error(tulpa(y ~ x + zz_not_there, d, mode = "laplace", phi = 1),
               "'zz_not_there'.*`data` has: y, x")
})

test_that("a variable from the formula's environment still resolves", {
  skip_on_cran()
  d <- data.frame(y = rnorm(30), x = rnorm(30))
  w_env <- rnorm(30)
  f <- tulpa(y ~ x + w_env, d, mode = "laplace", phi = 1)
  expect_true("w_env" %in% names(coef(f)))
})

skip_on_cran()

.rep_glmm_data <- function(seed = 2L, n = 300L) {
  set.seed(seed)
  x <- rnorm(n)
  g <- factor(sample(1:3, n, TRUE))
  data.frame(y = rbinom(n, 1, plogis(-0.3 + x)), x = x, g = g)
}

test_that("a Laplace mode-find that stops early warns and is reported", {
  skip_on_cran()
  d1 <- .rep_glmm_data()
  expect_warning(
    f <- tulpa(y ~ x + (1 | g), d1, family = "binomial", mode = "laplace",
               sigma_re = 0.5, control = list(max_iter = 1)),
    "mode-find stopped before converging")
  expect_identical(f$convergence$kind, "mode")
  expect_true(any(grepl("mode-find stopped", capture.output(print(f)))))
  expect_true(any(grepl("mode-find stopped",
                        capture.output(print(summary(f))))))
  expect_message(diagnostics(f), "mode-find stopped")
  expect_no_warning(tulpa(y ~ x + (1 | g), d1, family = "binomial",
                          mode = "laplace", sigma_re = 0.5))
})

test_that("mode = 'eb' estimates the dispersion of a fixed-effect gaussian", {
  skip_on_cran()
  set.seed(2)
  n <- 300; x <- rnorm(n)
  d <- data.frame(y = 1 + 2 * x + rnorm(n, sd = 3), x = x)
  X <- cbind(1, x)
  # Under a flat fixed-effect prior the marginal integrates beta under a flat
  # measure, and its maximizer is the REML residual variance RSS / (n - p).
  f_flat <- tulpa_eb(y = d$y, X = X, re_terms = list(), family = "gaussian",
                     estimate_phi = TRUE, beta_prior = list(mean = 0, sd = Inf))
  expect_equal(f_flat$phi, summary(stats::lm(y ~ x, d))$sigma^2,
               tolerance = 1e-5)
  # Through tulpa() the default N(0, 2.5^2) prior is on beta; the exact
  # marginal is then N(0, phi I + 2.5^2 X X') and its maximizer over phi is
  # found here by a dense 1-D search.
  f <- tulpa(y ~ x, d, mode = "eb", estimate_phi = TRUE)
  lml <- function(lphi) {
    S <- exp(lphi) * diag(n) + 2.5^2 * tcrossprod(X)
    R <- chol(S)
    -sum(log(diag(R))) - 0.5 * sum(backsolve(R, d$y, transpose = TRUE)^2)
  }
  o <- stats::optimize(lml, log(c(1, 50)), maximum = TRUE, tol = 1e-10)
  expect_equal(f$phi, exp(o$maximum), tolerance = 1e-4)
  expect_true(f$phi_estimated)
  vc <- VarCorr(f)
  expect_identical(vc$term, "Residual")
  expect_identical(vc$source, "estimated")
  expect_equal(vc$sd, sqrt(f$phi))
  expect_error(tulpa_eb(y = d$y, X = X, re_terms = list(), family = "gaussian"),
               "no random-effect terms and no dispersion")
})

test_that("ranef() on a Laplace fit reports the exact gaussian group spread", {
  skip_on_cran()
  set.seed(3)
  n <- 120; G <- 6
  x <- rnorm(n); g <- factor(sample(seq_len(G), n, TRUE))
  d <- data.frame(y = 0.5 + x + rnorm(G, sd = 0.7)[g] + rnorm(n), x = x, g = g)
  phi <- 1; s_re <- 0.7
  f <- tulpa(y ~ x + (1 | g), d, mode = "laplace", phi = phi, sigma_re = s_re)
  r <- ranef(f)
  # A gaussian model at fixed (phi, sigma_re) has an exactly Gaussian latent
  # posterior with precision A'A / phi + blockdiag(I / 2.5^2, I / sigma_re^2).
  A <- cbind(1, x, stats::model.matrix(~ 0 + g, d))
  Q <- crossprod(A) / phi + diag(c(rep(1 / 2.5^2, 2), rep(1 / s_re^2, G)))
  V <- chol2inv(chol(Q))
  m <- V %*% crossprod(A, d$y) / phi
  expect_identical(unique(r$source), "laplace")
  expect_equal(r$estimate, as.numeric(m[-(1:2)]), tolerance = 1e-6)
  expect_equal(r$sd, sqrt(diag(V)[-(1:2)]), tolerance = 1e-6)
  expect_equal(r$conf.high - r$estimate, stats::qnorm(0.975) * r$sd)
})

test_that("sampler draws carry the parameter names", {
  skip_on_cran()
  d1 <- .rep_glmm_data()
  fi <- tulpa(y ~ x + (1 | g), d1, family = "binomial", mode = "imh_laplace",
              sigma_re = 0.5, control = list(n_iter = 300, warmup = 100,
                                             seed = 1))
  s <- posterior_sample(fi)
  expect_identical(colnames(s), c("(Intercept)", "x", "g[1]", "g[2]", "g[3]"))
})

test_that("nested_laplace on random effects alone names mode = 'structured'", {
  skip_on_cran()
  d1 <- .rep_glmm_data()
  expect_error(tulpa(y ~ x + (1 | g), d1, family = "binomial",
                     mode = "nested_laplace"),
               "mode = 'structured'")
})
