# Censored and truncated Gaussian responses through tulpa(): `y | cens(lower,
# upper)` and `y | trunc(lb, ub)` (gcol33/tulpa#1092). The response term is read
# off the formula's parse tree, the bounds are evaluated against the data, and
# the fit runs on the one-arm joint nested-Laplace door, whose arm carries the
# bounds to the interval_gaussian / truncated_gaussian kernels.

.rb_kernel <- function(y, a, b, eta, sd, family) {
  tulpa:::cpp_bound_gaussian_loglik(y, a, b, eta, sd, family)
}

test_that("the response term is read off the parse tree", {
  rb <- tulpa:::.parse_response_bounds(quote(y | cens(lo, hi)))
  expect_identical(rb$kind, "cens")
  expect_identical(rb$family, "interval_gaussian")
  expect_identical(rb$response_expr, quote(y))
  expect_identical(rb$lower, quote(lo))
  expect_identical(rb$upper, quote(hi))

  # Named, reordered, and an omitted side of trunc() is open.
  tr <- tulpa:::.parse_response_bounds(quote(log(y) | trunc(ub = top)))
  expect_identical(tr$response_expr, quote(log(y)))
  expect_identical(tr$lower, -Inf)
  expect_identical(tr$upper, quote(top))
  # An expression bound stays a parse tree, evaluated later against the data.
  ex <- tulpa:::.parse_response_bounds(quote(y | cens(lower = pmin(a, b), b)))
  expect_identical(ex$lower, quote(pmin(a, b)))

  expect_null(tulpa:::.parse_response_bounds(quote(y)))
  expect_null(tulpa:::.parse_response_bounds(quote(cbind(s, f))))
  expect_error(tulpa:::.parse_response_bounds(quote(y | weights(w))),
               "cens\\(lower, upper\\) or trunc")
  expect_error(tulpa:::.parse_response_bounds(quote(y | cens(lo))),
               "needs both")

  pf <- tulpa_parse_formula(y | cens(lo, hi) ~ x + (1 | g))
  expect_identical(pf$response, "y")
  expect_identical(pf$response_bounds$kind, "cens")
  expect_identical(all.vars(pf$fixed_formula), c("y", "x"))
  expect_identical(pf$n_re_terms, 1L)
})

test_that("bounds are evaluated against the data, NA an open side", {
  d <- data.frame(y = c(1, NA, 3, 0.5), lo = c(1, NA, 2, NA),
                  hi = c(1, 0, NA, 0.7), x = 1:4)
  md <- tulpa_build_model_data(tulpa_parse_formula(y | cens(lo, hi) ~ x), d)
  b <- md$response_bounds
  expect_equal(b$lower, c(1, -Inf, 2, -Inf))
  expect_equal(b$upper, c(1, 0, Inf, 0.7))
  # A censored row with no recorded value keeps a finite placeholder, so the
  # kernel does not read it as a missing response.
  expect_true(all(is.finite(b$y_kernel)))

  # A scalar bound recycles; y outside its truncation range is refused.
  md2 <- tulpa_build_model_data(tulpa_parse_formula(y ~ x), d)
  expect_null(md2$response_bounds)
  dd <- data.frame(y = c(0.2, 0.5), x = 1:2)
  mt <- tulpa_build_model_data(tulpa_parse_formula(y | trunc(0, 1) ~ x), dd)
  expect_equal(mt$response_bounds$lower, c(0, 0))
  expect_error(tulpa_build_model_data(
    tulpa_parse_formula(y | trunc(0.3, 1) ~ x), dd), "outside its truncation")
  expect_error(tulpa_build_model_data(
    tulpa_parse_formula(y | cens(1, 0) ~ x), dd), "empty interval")
  expect_error(tulpa_build_model_data(
    tulpa_parse_formula(y | trunc(1, 1) ~ x), dd), "empty truncation")
})

test_that("the interval kernel reads exact, censored and interval rows", {
  eta <- c(0.3, 0.3, 0.3, 0.3, 0.3); sd <- 0.7
  lo <- c(1.1, -Inf, 0.5, -0.4, -Inf)
  hi <- c(1.1, 0.2, Inf, 0.9, Inf)
  ll <- .rb_kernel(rep(0, 5), lo, hi, eta, sd, "interval_gaussian")
  expect_equal(ll[1], stats::dnorm(1.1, 0.3, sd, log = TRUE), tolerance = 1e-12)
  expect_equal(ll[2], stats::pnorm(0.2, 0.3, sd, log.p = TRUE), tolerance = 1e-12)
  expect_equal(ll[3], stats::pnorm(0.5, 0.3, sd, lower.tail = FALSE,
                                   log.p = TRUE), tolerance = 1e-12)
  expect_equal(ll[4], log(stats::pnorm(0.9, 0.3, sd) - stats::pnorm(-0.4, 0.3, sd)),
               tolerance = 1e-12)
  expect_equal(ll[5], 0)
})

test_that("the truncated kernel divides by the retained interval mass", {
  y <- c(0.4, 0.4, 0.4, -0.2); eta <- c(0.1, 0.1, 1.5, -2)
  a <- c(-Inf, 0, -1, -0.5); b <- c(Inf, 1, 0.6, Inf); sd <- 0.6
  ll <- .rb_kernel(y, a, b, eta, sd, "truncated_gaussian")
  ref <- stats::dnorm(y, eta, sd, log = TRUE) -
    log(stats::pnorm(b, eta, sd) - stats::pnorm(a, eta, sd))
  expect_equal(ll, ref, tolerance = 1e-10)
})

skip_on_cran()

test_that("a two-sided truncated kernel matches central differences", {
  skip_on_cran()
  load_cpp_fixture("truncated_gaussian")
  ref <- function(y, lo, hi, eta, sg) {
    stats::dnorm(y, eta, sg, log = TRUE) -
      log(stats::pnorm(hi, eta, sg) - stats::pnorm(lo, eta, sg))
  }
  h <- 1e-5
  for (eta in c(-0.8, 0.2, 1.4)) for (sg in c(0.4, 1.1)) {
    y <- 0.1; lo <- -0.5; hi <- 0.9
    out <- cpp_truncated_gaussian_terms(y, hi, eta, sg, lower = lo)
    expect_equal(out[["ll"]], ref(y, lo, hi, eta, sg), tolerance = 1e-10)
    g_fd <- (ref(y, lo, hi, eta + h, sg) - ref(y, lo, hi, eta - h, sg)) / (2 * h)
    expect_equal(out[["grad"]], g_fd, tolerance = 1e-5)
    nh_fd <- -(ref(y, lo, hi, eta + h, sg) - 2 * ref(y, lo, hi, eta, sg) +
               ref(y, lo, hi, eta - h, sg)) / h^2
    expect_equal(out[["neg_hess"]], nh_fd, tolerance = 1e-3)
    expect_gt(out[["neg_hess"]], 0)
  }
})

.rb_censored_data <- function(n = 400L, seed = 1L, re = FALSE) {
  set.seed(seed)
  g <- rep(seq_len(n / 10L), each = 10L)
  x <- rnorm(n)
  ystar <- 1 + 0.5 * x + (if (re) rnorm(n / 10L, 0, 0.5)[g] else 0) +
    rnorm(n, 0, 0.8)
  L <- stats::quantile(ystar, 0.3); U <- stats::quantile(ystar, 0.9)
  lo <- ifelse(ystar < L, NA, ifelse(ystar > U, U, ystar))
  hi <- ifelse(ystar < L, L, ifelse(ystar > U, NA, ystar))
  mid <- ystar > 1 & ystar < 1.3
  lo[mid] <- 1; hi[mid] <- 1.3
  data.frame(x = x, g = factor(g), lo = lo, hi = hi, y = ystar)
}

test_that("a censored fit reproduces survreg's gaussian likelihood and MLE", {
  skip_on_cran()
  skip_if_not_installed("survival")
  d <- .rb_censored_data()
  sr <- survival::survreg(survival::Surv(lo, hi, type = "interval2") ~ x,
                          data = d, dist = "gaussian")
  flat <- list(mean = 0, sd = 1000)
  # Conditioned at survreg's scale, the posterior mode under a flat prior is
  # its MLE, and the kernel's log-likelihood there is survreg's.
  fp <- tulpa(y | cens(lo, hi) ~ x, d, phi = sr$scale^2, beta_prior = flat)
  expect_identical(fp$backend, "nested_laplace_joint")
  expect_equal(unname(coef(fp)), unname(coef(sr)), tolerance = 1e-4)
  b <- list(lower = ifelse(is.na(d$lo), -Inf, d$lo),
            upper = ifelse(is.na(d$hi), Inf, d$hi))
  ll <- tulpa:::.bound_loglik(as.numeric(cbind(1, d$x) %*% coef(sr)), d$y,
                              "interval_gaussian", b, sr$scale^2)
  expect_equal(sum(ll), as.numeric(logLik(sr)), tolerance = 1e-8)

  # Integrated, the residual SD's interval holds survreg's estimate and the
  # coefficients sit within a tenth of a standard error of its MLE.
  f <- tulpa(y | cens(lo, hi) ~ x, d, beta_prior = flat)
  expect_true(f$phi_estimated)
  expect_lt(sqrt(f$theta_ci_lo[["phi_y"]]), sr$scale)
  expect_gt(sqrt(f$theta_ci_hi[["phi_y"]]), sr$scale)
  se <- sqrt(diag(vcov(sr)))[1:2]
  expect_true(all(abs(coef(f) - coef(sr)) < 0.1 * se))
})

test_that("the accessors read the censored likelihood and its bounds", {
  skip_on_cran()
  d <- .rb_censored_data(n = 300L, seed = 4L, re = TRUE)
  f <- tulpa(y | cens(lo, hi) ~ x + (1 | g), d)
  expect_identical(f$family, "interval_gaussian")
  expect_equal(f$y, d$y)

  # The pointwise log-likelihood is the kernel's at each draw's predictor.
  ll <- pointwise_loglik(f)
  expect_identical(dim(ll), c(400L, nrow(d)))
  exact <- which(!is.na(d$lo) & !is.na(d$hi) & d$lo == d$hi)
  eta <- fitted(f)
  phi <- f$phi
  ref <- stats::dnorm(d$y[exact], eta[exact], sqrt(phi), log = TRUE)
  # The draw average of a Gaussian log density sits below its value at the
  # mean predictor by the predictor's posterior variance over 2 phi.
  E <- tulpa:::.tulpa_eta_draws(f, ndraws = 400L, synth_seed = 1L)
  jensen <- -mean(apply(E[, exact, drop = FALSE], 2L, stats::var)) / (2 * phi)
  expect_lt(abs(mean(colMeans(ll)[exact] - ref) - jensen), 0.01)
  w <- suppressWarnings(loo::waic(f))
  expect_true(all(is.finite(w$pointwise[, "elpd_waic"])))

  # Group effects are read off the integrated iid block, on the effect scale.
  re <- ranef(f)
  df <- if (is.data.frame(re)) re else re[[1L]]
  expect_identical(nrow(df), nlevels(d$g))
  vc <- VarCorr(f)
  expect_identical(vc$source[vc$term == "Residual"], "estimated")
})

test_that("a truncated fit's replicates stay inside their bounds", {
  skip_on_cran()
  set.seed(6)
  n <- 400L
  x <- rnorm(n)
  ystar <- 1 + 0.5 * x + rnorm(n, 0, 0.6)
  d <- data.frame(x = x, y = ystar)[ystar > 0.7, ]
  f <- tulpa(y | trunc(0.7) ~ x, d)
  expect_identical(f$family, "truncated_gaussian")
  pp <- posterior_predict(f, ndraws = 50L)
  expect_true(all(pp >= 0.7))
  # Ignoring the truncation attenuates the slope; the truncated fit's
  # interval holds the truth.
  naive <- tulpa(y ~ x, d)
  expect_lt(coef(naive)[["x"]], 0.4)
  ci <- confint(f)
  expect_true(ci["x", 1] < 0.5 && 0.5 < ci["x", 2])
  expect_lt(abs(coef(f)[["x"]] - 0.5), abs(coef(naive)[["x"]] - 0.5))
})

test_that("a bounded response refuses what it cannot carry", {
  skip_on_cran()
  d <- .rb_censored_data(n = 100L)
  expect_error(tulpa(y | cens(lo, hi) ~ x, d, family = "poisson"),
               "family = 'gaussian'")
  expect_error(tulpa(y | cens(lo, hi) ~ x, d, mode = "hmc"),
               "does not carry them")
  expect_error(tulpa(y | cens(lo, hi) ~ x, d, weights = rep(1, 100)),
               "observation weights")
})

test_that("the joint arm reads a censored / truncated phi as a variance", {
  skip_on_cran()
  # One convention with the gaussian family: an arm fitted at phi = s^2
  # evaluates its kernels at residual SD s.
  set.seed(2)
  n <- 60L
  y <- rnorm(n, 0.3, 0.5)
  X <- matrix(1, n, 1)
  fit_at <- function(phi) {
    f <- tulpa_nested_laplace_joint(
      responses = list(y = list(y = y, X = X, family = "interval_gaussian",
                                phi = phi, lower = y, upper = y)),
      prior = list(), control = list(progress = FALSE))
    f$log_marginal
  }
  g <- tulpa_nested_laplace_joint(
    responses = list(y = list(y = y, X = X, family = "gaussian", phi = 0.25)),
    prior = list(), control = list(progress = FALSE))
  expect_equal(fit_at(0.25), g$log_marginal, tolerance = 1e-8)
})

test_that("coverage of beta and sigma at 30% censoring holds over seeds", {
  skip_if_not_slow()
  beta <- c(1, 0.5); s_eps <- 0.6; sg <- 0.5
  cov <- t(vapply(seq_len(20L), function(s) {
    set.seed(1000L + s)
    ng <- 30L; g <- rep(seq_len(ng), each = 10L); x <- rnorm(ng * 10L)
    ystar <- beta[1] + beta[2] * x + rnorm(ng, 0, sg)[g] +
      rnorm(ng * 10L, 0, s_eps)
    L <- beta[1] - 0.52 * sqrt(sg^2 + s_eps^2 + 0.25)
    d <- data.frame(x = x, g = factor(g), y = pmax(ystar, L),
                    lo = ifelse(ystar < L, -Inf, ystar),
                    hi = ifelse(ystar < L, L, ystar))
    f <- tulpa(y | cens(lo, hi) ~ x + (1 | g), d)
    ci <- confint(f)
    c(b1 = ci["x", 1] <= beta[2] && beta[2] <= ci["x", 2],
      s = sqrt(f$theta_ci_lo[["phi_y"]]) <= s_eps &&
        s_eps <= sqrt(f$theta_ci_hi[["phi_y"]]))
  }, logical(2)))
  expect_gte(mean(cov[, "b1"]), 0.85)
  expect_gte(mean(cov[, "s"]), 0.85)
})
