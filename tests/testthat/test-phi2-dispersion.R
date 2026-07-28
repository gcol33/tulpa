# Second dispersion channel phi2 (gcol33/tulpa C7): configurable Student-t df
# through the registry, the Laplace kernel, and the samplers; the lognormal
# family registered with the variance convention; and the gaussian phi
# convention unified across backends (tulpa()'s phi = residual VARIANCE
# everywhere -- the SD-parameterized kernels receive sqrt(phi) at the
# boundary, so mode = 'laplace' and the R-side H_beta now describe the same
# model). The conversion resolves the base family, so a `<family>_<link>`
# spelling of a normal family converts like its base (gcol33/tulpa#256), and
# phi2 is threaded into the random-effect covariance paths' inner solve rather
# than defaulted there (gcol33/tulpa#257).

test_that("t-family registry ops honor phi2 and match dt()", {
  eta <- c(-0.5, 0.2, 1.4)
  y   <- c(0.3, -1.1, 2.0)
  s   <- 0.7

  for (nu in c(3, 10, 25)) {
    ll <- tulpa:::family_loglik(eta, y, "t", phi = s, phi2 = nu)
    expect_equal(ll, stats::dt((y - eta) / s, df = nu, log = TRUE) - log(s),
                 tolerance = 1e-12)
  }
  # Default df = 4 when phi2 is NULL.
  expect_equal(tulpa:::family_loglik(eta, y, "t", phi = s),
               stats::dt((y - eta) / s, df = 4, log = TRUE) - log(s),
               tolerance = 1e-12)
  # Score matches a numerical derivative at phi2 = 7.
  h <- 1e-6
  num <- (tulpa:::family_loglik(eta + h, y, "t", phi = s, phi2 = 7) -
          tulpa:::family_loglik(eta - h, y, "t", phi = s, phi2 = 7)) / (2 * h)
  expect_equal(tulpa:::family_score_eta(eta, y, "t", phi = s, phi2 = 7), num,
               tolerance = 1e-6)
  # Variance: phi^2 nu / (nu - 2).
  expect_equal(tulpa:::family_variance(eta, "t", phi = s, phi2 = 10),
               rep(s^2 * 10 / 8, 3))
})

test_that("phi2 is rejected for families without a second dispersion", {
  expect_error(tulpa:::family_loglik(0, 1, "gaussian", phi = 1, phi2 = 5),
               "no second dispersion")
  d <- data.frame(x = rnorm(30)); d$y <- rnorm(30)
  expect_error(
    tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace", phi2 = 5),
    "no second dispersion")
  d$y2 <- rbinom(30, 1, 0.5)
  expect_error(
    tulpa(y2 ~ x, data = d, family = "binomial", mode = "laplace", phi2 = -1),
    "phi2")
})

test_that("Laplace t fit with phi2 matches an optim reference and differs from df 4", {
  skip_on_cran()
  set.seed(81)
  n <- 200L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(1, -0.6)) + 0.8 * rt(n, df = 10)
  d <- data.frame(y = y, x = X[, 2])

  fit10 <- tulpa(y ~ x, data = d, family = "t", mode = "laplace",
                 phi = 0.8, phi2 = 10)
  fit4  <- tulpa(y ~ x, data = d, family = "t", mode = "laplace", phi = 0.8)

  # Reference: penalized MAP of the same posterior (weak builtin prior
  # beta ~ N(0, 100^2)) via optim.
  nlp <- function(b, nu) {
    -sum(stats::dt((y - X %*% b) / 0.8, df = nu, log = TRUE) - log(0.8)) +
      sum(b^2) / (2 * 100^2)
  }
  ref10 <- stats::optim(c(0, 0), nlp, nu = 10, method = "BFGS")$par
  expect_equal(unname(coef(fit10)), ref10, tolerance = 1e-4)
  expect_false(isTRUE(all.equal(unname(coef(fit10)), unname(coef(fit4)),
                                tolerance = 1e-6)))
})

test_that("gaussian phi means the residual variance on every backend", {
  skip_on_cran()
  set.seed(83)
  n <- 60L
  X <- cbind(1, rnorm(n))
  y <- as.numeric(X %*% c(1, -0.5)) + rnorm(n, 0, 2)   # variance 4
  d <- data.frame(y = y, x = X[, 2])

  # Laplace: a strong prior makes the likelihood/prior balance identify the
  # convention. MAP must be the variance-convention penalized WLS.
  fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
               phi = 4, beta_prior = list(mean = 0, sd = 1))
  map_var <- solve(crossprod(X) / 4 + diag(1, 2), crossprod(X, y) / 4)
  expect_equal(unname(coef(fit)), as.numeric(map_var), tolerance = 1e-5)

  # ModelData NUTS: the posterior SD must match the variance convention.
  fit2 <- tulpa_sample_glmm(y, NULL, X, family = "gaussian", backend = "hmc",
                            phi = 2,   # the DIRECT door takes the residual SD
                            control = list(n_iter = 3000L, warmup = 1000L,
                                           n_chains = 2L, seed = 1L))
  sd_ref <- sqrt(diag(solve(crossprod(X) / 4)))
  expect_equal(unname(apply(fit2$draws[, 1:2], 2, sd)), sd_ref,
               tolerance = 0.15)

  # And through the front door (which converts variance -> SD for the kernel):
  fit3 <- tulpa(y ~ x, data = d, family = "gaussian", mode = "hmc", phi = 4,
                control = list(n_iter = 3000L, warmup = 1000L,
                               n_chains = 2L, seed = 1L))
  expect_equal(unname(apply(.fixed_draws_mat(fit3), 2, sd)), sd_ref,
               tolerance = 0.15)
})

test_that("lognormal is registered with the variance convention", {
  set.seed(85)
  eta <- c(0.2, 1.0); y <- c(1.5, 4.0); v <- 0.36

  expect_equal(tulpa:::family_loglik(eta, y, "lognormal", phi = v),
               stats::dlnorm(y, meanlog = eta, sdlog = sqrt(v), log = TRUE),
               tolerance = 1e-12)
  expect_equal(tulpa:::family_mean(eta, "lognormal", phi = v),
               exp(eta + v / 2))
  # Sample moments.
  ys <- tulpa:::family_sample(rep(0.5, 40000), "lognormal", phi = 0.25)
  expect_equal(mean(ys), exp(0.5 + 0.125), tolerance = 0.03)
  expect_equal(stats::var(ys),
               (exp(0.25) - 1) * exp(1 + 0.25), tolerance = 0.1)
})

test_that("lognormal fits through the front door", {
  skip_on_cran()
  set.seed(87)
  n <- 300L
  d <- data.frame(x = rnorm(n))
  d$y <- exp(1 + 0.5 * d$x + rnorm(n, 0, 0.6))   # log-scale variance 0.36

  fit <- tulpa(y ~ x, data = d, family = "lognormal", mode = "laplace",
               phi = 0.36)
  ref <- unname(coef(stats::lm(log(y) ~ x, data = d)))
  expect_equal(unname(coef(fit)), ref, tolerance = 1e-3)

  # And on the sampler tier.
  fit2 <- tulpa(y ~ x, data = d, family = "lognormal", mode = "hmc",
                phi = 0.36, control = list(n_iter = 2000L, warmup = 1000L,
                                           n_chains = 2L, seed = 2L))
  expect_equal(unname(coef(fit2)), ref, tolerance = 0.05)
})

test_that("the variance/SD convention follows the base family, not the spelling", {
  # gcol33/tulpa#256: the conversion tested the family name for exact equality,
  # so every `<family>_<link>` spelling of a normal family reached the kernel
  # with the variance where the SD belongs.
  for (f in c("gaussian", "gaussian_identity", "gaussian_log",
              "gaussian_inverse", "lognormal", "lognormal_log")) {
    expect_true(tulpa:::.phi_is_variance(f), info = f)
  }
  for (f in c("gamma", "gamma_log", "poisson", "binomial", "t",
              "neg_binomial_2", "beta_logit")) {
    expect_false(tulpa:::.phi_is_variance(f), info = f)
  }
  # The two directions are inverses on a variance family and identities
  # elsewhere.
  expect_equal(tulpa:::.phi_to_kernel("gaussian_log", 0.64), 0.8)
  expect_equal(tulpa:::.phi_to_registry("gaussian_log", 0.8), 0.64)
  expect_equal(tulpa:::.phi_to_kernel("gamma_log", 0.64), 0.64)
  expect_equal(tulpa:::.phi_to_registry("gamma_log", 0.64), 0.64)
})

test_that("a suffixed normal family is fitted at the dispersion asked for", {
  skip_on_cran()
  set.seed(256)
  G <- 10L; per <- 8L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x   <- rnorm(n)
  b   <- rnorm(G, 0, 0.6)
  y   <- rlnorm(n, log(exp(0.4 + 0.3 * x + b[grp])), 0.5)   # strictly positive
  X   <- cbind(1, x)
  re  <- list(list(idx = grp, n_groups = G, n_coefs = 1L, sigma = 0.6))
  v   <- 0.8                                  # the residual VARIANCE requested

  # Two spellings of the same family/link pair are one model.
  fit_bare <- tulpa_laplace(y, rep(1L, n), X, re, family = "gaussian",
                            phi = v)
  fit_suff <- tulpa_laplace(y, rep(1L, n), X, re, family = "gaussian_identity",
                            phi = v)
  expect_equal(fit_suff$mode, fit_bare$mode, tolerance = 1e-12)
  expect_equal(fit_suff$log_marginal, fit_bare$log_marginal, tolerance = 1e-12)

  # gaussian_log is where the two conventions stop coinciding: dW/deta is not
  # identically zero there, so a wrong phi no longer cancels out of the mode.
  # The reference goes straight to the kernel, which takes the SD.
  fit_log <- tulpa_laplace(y, rep(1L, n), X, re, family = "gaussian_log",
                           phi = v)
  kern_at <- function(p) {
    tulpa:::cpp_laplace_fit_multi_re(
      y = as.numeric(y), n = rep(1L, n), X = X,
      re_idx_list = list(as.integer(grp)), re_ngroups = G,
      re_sigma_list = list(0.6), family = "gaussian_log", phi = p,
      max_iter = 100L, tol = 1e-6, n_threads = 1L)
  }
  expect_equal(fit_log$mode, kern_at(sqrt(v))$mode, tolerance = 1e-10)
  # And is genuinely distinguishable from the unconverted value, so the check
  # above is not satisfied by both.
  expect_false(isTRUE(all.equal(fit_log$mode, kern_at(v)$mode,
                                tolerance = 1e-6)))
})

test_that("phi2 reaches the EB and nested random-effect covariance paths", {
  skip_on_cran()
  # gcol33/tulpa#257: the outer core hard-coded phi2 = NA_real_, so `t` fitted
  # at the compiled default df whatever the caller asked for.
  set.seed(257)
  G <- 12L; per <- 10L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x   <- rnorm(n)
  u   <- rnorm(G, 0, 0.7)
  y   <- 0.5 + 0.8 * x + u[grp] + 0.9 * rt(n, df = 6)
  X   <- cbind(1, x)
  re  <- list(idx = grp, n_groups = G, n_coefs = 1L)

  eb_def <- tulpa_eb(y, NULL, X, re, family = "t", phi = 1.0)
  eb_8   <- tulpa_eb(y, NULL, X, re, family = "t", phi = 1.0, phi2 = 8)
  # A different df is a different model: the objective and the estimate move.
  expect_false(isTRUE(all.equal(eb_def$log_marginal, eb_8$log_marginal,
                                tolerance = 1e-8)))
  expect_false(isTRUE(all.equal(eb_def$map$sigma, eb_8$map$sigma,
                                tolerance = 1e-8)))
  # Supplying the compiled default reproduces it exactly, which is what pins
  # the threading as faithful rather than merely different.
  eb_4 <- tulpa_eb(y, NULL, X, re, family = "t", phi = 1.0, phi2 = 4)
  expect_equal(eb_4$log_marginal, eb_def$log_marginal, tolerance = 1e-12)

  nl_def <- tulpa_re_cov_nested(y, NULL, X, re, family = "t", phi = 1.0,
                                control = list(diagnose_k = FALSE,
                                               n_draws = 200L, seed = 1L))
  nl_8   <- tulpa_re_cov_nested(y, NULL, X, re, family = "t", phi = 1.0,
                                phi2 = 8,
                                control = list(diagnose_k = FALSE,
                                               n_draws = 200L, seed = 1L))
  expect_false(isTRUE(all.equal(as.numeric(nl_def$Sigma_mean),
                                as.numeric(nl_8$Sigma_mean),
                                tolerance = 1e-8)))
})

test_that("the covariance paths require tweedie's power and refuse a spurious phi2", {
  skip_on_cran()
  set.seed(2571)
  G <- 12L; per <- 10L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  x   <- rnorm(n)
  u   <- rnorm(G, 0, 0.5)
  y   <- rpois(n, exp(0.2 + 0.4 * x + u[grp])) * rgamma(n, 2, 2)
  X   <- cbind(1, x)
  re  <- list(idx = grp, n_groups = G, n_coefs = 1L)

  # Without a power the fit used to die as a generic inner-solve failure; it
  # now names the argument that had no way to arrive.
  expect_error(tulpa_eb(y, NULL, X, re, family = "tweedie", phi = 1.0),
               "requires `phi2`")
  expect_error(tulpa_re_cov_nested(y, NULL, X, re, family = "tweedie",
                                   phi = 1.0),
               "requires `phi2`")
  expect_error(tulpa_eb(y, NULL, X, re, family = "tweedie", phi = 1.0,
                        phi2 = 2.5),
               "strictly in \\(1, 2\\)")
  # With one it fits.
  ft <- tulpa_eb(y, NULL, X, re, family = "tweedie", phi = 1.0, phi2 = 1.5)
  expect_true(is.finite(ft$log_marginal))

  # A family with no second dispersion errors rather than ignoring it.
  expect_error(tulpa_eb(y, NULL, X, re, family = "poisson", phi2 = 3),
               "no second dispersion")
  expect_error(tulpa_re_cov_nested(y, NULL, X, re, family = "poisson",
                                   phi2 = 3),
               "no second dispersion")
})

test_that("tulpa() forwards phi2 to the covariance backends and refuses it elsewhere", {
  skip_on_cran()
  set.seed(2572)
  G <- 12L; per <- 10L; n <- G * per
  grp <- rep(seq_len(G), each = per)
  d <- data.frame(x = rnorm(n), g = factor(rep(seq_len(G), each = per)))
  u <- rnorm(G, 0, 0.7)
  d$y <- 0.5 + 0.8 * d$x + u[grp] + 0.9 * rt(n, df = 6)

  f_def <- tulpa(y ~ x + (1 | g), data = d, family = "t", phi = 1.0,
                 mode = "eb")
  f_8   <- tulpa(y ~ x + (1 | g), data = d, family = "t", phi = 1.0, phi2 = 8,
                 mode = "eb")
  expect_false(isTRUE(all.equal(f_def$log_marginal, f_8$log_marginal,
                                tolerance = 1e-8)))

  # The backend list in the refusal is derived, so it names eb / re_cov_nested
  # as available rather than restating the pre-#257 set.
  expect_true(all(c("laplace", "eb", "re_cov_nested") %in%
                    tulpa:::.phi2_backends()))
  expect_false("re_cov_gibbs" %in% tulpa:::.phi2_backends())
  expect_error(tulpa(y ~ x + (1 | g), data = d, family = "t", phi = 1.0,
                     phi2 = 8, mode = "gibbs"),
               "not supported by backend")
})

test_that("posterior_predict honors the t df through phi2", {
  skip_on_cran()
  set.seed(89)
  n <- 300L
  d <- data.frame(x = rnorm(n))
  d$y <- 1 + 0.5 * d$x + 0.8 * rt(n, df = 20)
  fit <- tulpa(y ~ x, data = d, family = "t", mode = "laplace",
               phi = 0.8, phi2 = 20)

  yrep <- posterior_predict(fit, ndraws = 400L, seed = 3)
  # Residual variance of replicates ~ phi^2 nu/(nu-2) = 0.64 * 20/18 = 0.711.
  res_var <- mean(apply(yrep, 1, function(r) stats::var(r - fitted(fit))))
  expect_equal(res_var, 0.8^2 * 20 / 18, tolerance = 0.15)
})
