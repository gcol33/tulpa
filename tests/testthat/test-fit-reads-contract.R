# What the estimation-level accessors read, and in which layout they report it:
# interval column labels (gcol33/tulpa#756), one posterior read per fit
# (gcol33/tulpa#752), a declined logLik() never an unexplained NA
# (gcol33/tulpa#750), and a named `means` on every producer (gcol33/tulpa#758).

test_that("interval columns are labelled as stats::confint.default labels them", {
  for (level in c(0.95, 0.90, 0.80, 0.5, 0.99, 0.999)) {
    ref <- colnames(stats::confint(stats::lm(dist ~ speed, data = cars),
                                   level = level))
    expect_identical(tulpa:::.interval_colnames(level), ref, info = level)
  }
})

test_that("confint() and summary() carry those labels on a fit", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(120))
  d$y <- rpois(120, exp(0.3 + 0.4 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "poisson", mode = "laplace")
  expect_identical(colnames(confint(fit)), c("2.5 %", "97.5 %"))
  expect_identical(names(summary(fit))[3:4], c("2.5 %", "97.5 %"))
  expect_identical(colnames(confint(fit, level = 0.9)), c("5 %", "95 %"))
  expect_identical(names(summary(fit, level = 0.9))[3:4], c("5 %", "95 %"))
  expect_equal(unname(as.matrix(summary(fit)[, 3:4])), unname(confint(fit)),
               tolerance = 0)
})

# A fit reporting a closed-form Gaussian: every coefficient accessor reads that
# Gaussian, and none of them reads the draws sampled from it.
expect_gaussian_read <- function(fit, level = 0.9) {
  m <- fit$means
  V <- fit$cov
  se <- sqrt(diag(V))
  z <- stats::qnorm(1 - (1 - level) / 2)
  expect_identical(fit$reported_posterior, "gaussian")
  expect_equal(coef(fit), m, tolerance = 0)
  expect_equal(vcov(fit), V, tolerance = 0)
  s <- summary(fit, level = level)
  expect_equal(s$estimate, unname(m), tolerance = 0)
  expect_equal(s$std.error, unname(se), tolerance = 1e-12)
  ci <- confint(fit, level = level)
  expect_equal(unname(ci[, 1]), unname(m - z * se), tolerance = 1e-12)
  expect_equal(unname(ci[, 2]), unname(m + z * se), tolerance = 1e-12)
  td <- tidy(fit, conf.level = level)
  expect_equal(td$estimate, unname(m), tolerance = 0)
  expect_identical(td$term, names(m))
  # The draws are samples from that Gaussian, so their moments differ from it
  # by Monte-Carlo error; the accessors above do not carry that error.
  expect_false(isTRUE(all.equal(unname(colMeans(fit$draws)), unname(m),
                                tolerance = 0)))
}

test_that("an EP fit reports its EP Gaussian through every accessor", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(200))
  d$y <- rbinom(200, 1, plogis(0.2 + 0.5 * d$x))
  fit <- tulpa(y ~ x, data = d, family = "binomial", mode = "ep")
  expect_gaussian_read(fit)
  expect_identical(names(coef(fit)), c("(Intercept)", "x"))
})

test_that("a multinomial fit reports its Laplace Gaussian through every accessor", {
  skip_on_cran()
  set.seed(2)
  n <- 240L; x <- rnorm(n)
  eta <- cbind(0.4 + 0.8 * x, -0.2 - 0.6 * x)
  P <- cbind(exp(eta), 1); P <- P / rowSums(P)
  y <- factor(vapply(seq_len(n), function(i) sample.int(3L, 1L, prob = P[i, ]),
                     integer(1)))
  fit <- tulpa(y ~ x, data = data.frame(y = y, x = x), family = "multinomial")
  expect_gaussian_read(fit)
  expect_identical(names(coef(fit))[1:2], c("1:(Intercept)", "1:x"))
})

test_that("an ordinal fit reads its draws once: coef() is summary() less the cutpoints", {
  skip_on_cran()
  set.seed(3)
  n <- 300L; x <- rnorm(n)
  Fm <- plogis(outer(-0.8 * x, c(-1, 0.5, 2), "+"))
  P <- cbind(Fm, 1) - cbind(0, Fm)
  y <- ordered(vapply(seq_len(n), function(i) sample.int(4L, 1L, prob = P[i, ]),
                      integer(1)))
  fit <- tulpa_ordinal(y ~ x, data = data.frame(y = y, x = x),
                       control = list(seed = 1L))
  s <- summary(fit)
  expect_identical(names(coef(fit)), "x")
  expect_equal(unname(coef(fit)), s["x", "estimate"], tolerance = 0)
  expect_identical(rownames(vcov(fit)), rownames(s))
  expect_equal(s$std.error, unname(sqrt(diag(vcov(fit)))), tolerance = 1e-12)
  expect_equal(unname(coef(fit)), unname(colMeans(fit$draws)[1]), tolerance = 0)
})

test_that("a draw-based fit with no per-draw log posterior declines by name", {
  fit <- structure(list(draws = matrix(rnorm(40), 20, 2), n_fixed = 2L,
                        N = 50L, backend = "ess"),
                   class = "tulpa_fit")
  ll <- logLik(fit)
  expect_true(is.na(ll))
  expect_identical(attr(ll, "quantity"), "log_posterior_mean")
  expect_identical(attr(ll, "declined"), "no_log_posterior_recorded")
  expect_error(stats::AIC(fit), "is a log posterior mean")

  bare <- structure(list(N = 50L), class = "tulpa_fit")
  lb <- logLik(bare)
  expect_true(is.na(lb))
  expect_identical(attr(lb, "declined"), "no_goodness_quantity_recorded")
  expect_error(stats::AIC(bare), "reports no quantity \\(declined: no_goodness_quantity_recorded\\)")
  expect_no_match(tryCatch(stats::AIC(bare), error = conditionMessage), "\\bNA\\b")

  expect_error(tulpa:::.loglik_decline("not_a_reason"))
})

test_that("the default poisson random-intercept route and the VI kernel report a log posterior mean", {
  skip_on_cran()
  set.seed(1)
  d <- data.frame(x = rnorm(200), g = factor(rep(1:20, each = 10)))
  d$y <- rpois(200, exp(0.2 + 0.5 * d$x + rnorm(20, 0, 0.5)[d$g]))
  fits <- list(
    auto = tulpa(y ~ x + (1 | g), data = d, family = "poisson",
                 control = list(n_iter = 200L, warmup = 100L, seed = 1L)),
    vi = tulpa(y ~ x, data = d, family = "poisson", mode = "vi",
               control = list(seed = 1L))
  )
  expect_identical(fits$auto$backend, "re_cov_gibbs")
  for (nm in names(fits)) {
    ll <- logLik(fits[[nm]])
    expect_length(fits[[nm]]$log_prob, nrow(fits[[nm]]$draws))
    expect_true(is.finite(ll), info = nm)
    expect_null(attr(ll, "declined"), info = nm)
    expect_identical(attr(ll, "quantity"), "log_posterior_mean", info = nm)
    expect_equal(as.numeric(ll), mean(fits[[nm]]$log_prob), info = nm)
    expect_equal(glance(fits[[nm]])$logLik, as.numeric(ll), info = nm)
  }
})

test_that("means is named by parameter on the log-posterior samplers", {
  skip_on_cran()
  set.seed(4)
  d <- data.frame(x = rnorm(120), g = factor(rep(1:12, each = 10)))
  d$y <- rbinom(120, 1, plogis(0.2 + 0.5 * d$x + rnorm(12, 0, 0.6)[d$g]))
  for (m in c("mala", "imh_laplace", "pathfinder")) {
    ctl <- if (m == "pathfinder") list(n_draws = 200L) else
      list(n_iter = 200L, warmup = 100L)
    fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = m,
                 sigma_re = 0.6, control = ctl)
    expect_identical(names(fit$means), fit$param_names, info = m)
    expect_identical(fit$param_names[1:3], c("(Intercept)", "x", "g[1]"), info = m)
  }
})

test_that("a sampler fit with random effects and zero inflation reads beta_zi where the engine lays it", {
  skip_on_cran()
  set.seed(1)
  n <- 400L; G <- 20L
  d <- data.frame(x = rnorm(n), g = factor(rep(seq_len(G), each = 20L)))
  eta <- 0.4 + 0.5 * d$x + rnorm(G, 0, 0.6)[d$g]
  d$y <- ifelse(runif(n) < plogis(-1), 0L, rpois(n, exp(eta)))
  fit <- suppressWarnings(tulpa(
    y ~ x + (1 | g), data = d, family = "poisson", ziformula = ~ 1,
    mode = "hmc", control = list(n_iter = 300L, warmup = 150L, n_chains = 1L,
                                 seed = 1L)))

  layout  <- tulpa:::.tulpa_sampler_layout(fit)
  zi_cols <- tulpa:::.layout_span_cols(layout$beta_zi)
  ct_cols <- tulpa:::.layout_span_cols(layout$beta[[1L]])
  # The zero-inflation block follows the random effects and their variance
  # component, so it is not the column after the count coefficients.
  expect_gt(min(zi_cols), max(ct_cols) + 1L)

  zi_nm <- grep("^zi_", names(coef(fit)), value = TRUE)
  expect_identical(zi_nm, "zi_(Intercept)")
  expect_equal(unname(coef(fit)[zi_nm]),
               unname(colMeans(fit$draws[, zi_cols, drop = FALSE])), tolerance = 0)
  expect_equal(unname(coef(fit)[c("(Intercept)", "x")]),
               unname(colMeans(fit$draws[, ct_cols, drop = FALSE])), tolerance = 0)
  fd <- fit$draws[, c(ct_cols, zi_cols), drop = FALSE]
  expect_equal(unname(vcov(fit)), unname(stats::cov(fd)), tolerance = 1e-12)
  expect_equal(summary(fit)$std.error, unname(apply(fd, 2, stats::sd)),
               tolerance = 1e-12)
  expect_equal(tidy(fit)$estimate, unname(colMeans(fd)), tolerance = 0)

  # The random-effect draws exclude every fixed-effect column, so the variance
  # component and the group effects are both still there.
  rd <- tulpa:::.re_draws_mat(fit)
  expect_false(any(c("beta_zi[1]", "(Intercept)", "x") %in% colnames(rd)))
  expect_true("log_sigma_re" %in% colnames(rd))
  expect_equal(nrow(ranef(fit)), G)

  # The engine-eta path locates beta_zi through the same layout, so its
  # structural-zero logit averages to the one fitted() plugs coef() into.
  es <- tulpa:::.tulpa_eta_draws(fit)
  expect_equal(unname(colMeans(attr(es, "logit_zi"))),
               tulpa:::.tulpa_point_linpred(fit, NULL, "fitted")$logit_zi,
               tolerance = 1e-12)
  # fitted() is the population-level mean at the posterior-mean coefficients;
  # the per-draw population-level mean averages to it up to the Jensen gap of
  # a coefficient SE near 0.2 and Monte-Carlo error.
  ed <- tulpa:::.tulpa_eta_draws(fit, newdata = d)
  mu <- colMeans((1 - plogis(attr(ed, "logit_zi"))) * exp(ed))
  expect_lt(max(abs(fitted(fit) / mu - 1)), 0.1)
})

test_that("naming leaves a named or differently sized means as it is", {
  named <- list(param_names = c("a", "b"), means = c(u = 1, v = 2))
  expect_identical(names(tulpa:::.name_means_by_parameter(named)$means),
                   c("u", "v"))
  short <- list(param_names = c("a", "b", "c"), means = c(1, 2))
  expect_null(names(tulpa:::.name_means_by_parameter(short)$means))
  bare <- list(param_names = c("a", "b"), means = c(1, 2))
  expect_identical(names(tulpa:::.name_means_by_parameter(bare)$means),
                   c("a", "b"))
})
