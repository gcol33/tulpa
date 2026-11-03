
test_that("the bare temporal(col) naming path still parses (accessor unaffected)", {
  p <- tulpa_parse_formula(y ~ x + temporal(year))
  expect_identical(p$temporal_var, "year")
  expect_identical(p$n_temporal_field_blocks, 0L)
  expect_identical(deparse(p$fixed_formula), "y ~ x")
})

skip_on_cran_budget()
# Inline temporal varying-coefficient fields: temporal(formula = ~ 1 + x ||
# time, structure = "rw1"/"rw2"/"ar1") (gcol33/tulpa#91). The temporal mirror of
# the spatial() bar API: the intercept column is a smooth temporal level, a
# covariate column is a temporally varying slope (eta_i += x_i * f(time_i)).

# --------------------------------------------------------------------------- #
# (1) Construction                                                            #
# --------------------------------------------------------------------------- #

test_that("temporal(formula=, structure=) builds temporal blocks with svc_weight", {
  skip_on_cran_budget()
  f <- temporal(formula = ~ 1 + x || time, structure = "rw1")
  expect_s3_class(f, "tulpa_temporal_field")
  expect_identical(f$structure, "rw1")
  expect_true(f$has_intercept)

  d <- data.frame(time = rep(1:5, each = 3L), x = rnorm(15L))
  blocks <- tulpa:::.temporal_field_blocks(f, d)
  expect_length(blocks, 2L)                                # intercept + x slope
  expect_true(all(vapply(blocks, function(b) b$type, "") == "rw1"))
  expect_identical(blocks[[1]]$n_times, 5L)
  expect_null(blocks[[1]]$svc_weight)                      # intercept: no weight
  expect_false(is.null(blocks[[2]]$svc_weight))            # slope: per-row weight
  expect_equal(blocks[[2]]$svc_weight[[1]], d$x)

  # ar1 and rw2 select their structure.
  expect_identical(temporal(~ 1 || time, structure = "ar1")$structure, "ar1")
  expect_identical(temporal(~ 1 || time, structure = "rw2")$structure, "rw2")
})

test_that("temporal() field grammar is strict", {
  skip_on_cran_budget()
  # Single bar | (correlated) is reserved.
  expect_error(temporal(formula = ~ 1 + x | time), "Correlated")
  # Nested grouping is rejected.
  expect_error(temporal(formula = ~ 1 || time / g), "Nested")
  # A bar is required.
  expect_error(temporal(formula = ~ 1 + x), "grouping bar")
  # Unknown structure.
  expect_error(temporal(formula = ~ 1 || time, structure = "rw9"))
  # by= is reserved.
  expect_error(temporal(formula = ~ 1 || time, by = quote(g)), "not implemented")
})

# --------------------------------------------------------------------------- #
# (2) Formula integration                                                     #
# --------------------------------------------------------------------------- #

test_that("inline temporal() does not leak its bar into random effects", {
  skip_on_cran_budget()
  p <- tulpa_parse_formula(
    y ~ temporal(formula = ~ 1 + x || time, structure = "rw1") + (1 | site))
  expect_identical(p$n_temporal_field_blocks, 1L)
  # The (1 | site) bar is the only model random effect; the field's || time bar
  # is not counted as one.
  expect_identical(p$n_re_terms, 1L)
  expect_identical(p$random_effects[[1]]$group_var, "site")
  # The field's covariate x is stripped from the fixed part.
  expect_identical(deparse(p$fixed_formula), "y ~ 1")
  expect_null(p$temporal_var)
})

skip_on_cran()

# --------------------------------------------------------------------------- #
# (3) Parameter recovery: temporal level + temporally varying slope            #
# --------------------------------------------------------------------------- #

test_that("temporal() recovers the level and the time-varying slope (rw1)", {
  skip_on_cran()

  cors_lvl <- numeric(0)
  cors_slp <- numeric(0)
  for (seed in 1:3) {
    set.seed(seed)
    T <- 30L
    lvl <- cumsum(rnorm(T)); lvl <- lvl - mean(lvl)   # smooth level (rw1)
    slp <- cumsum(rnorm(T)); slp <- slp - mean(slp)   # time-varying slope
    n_per <- 40L
    time <- rep(seq_len(T), each = n_per)
    N <- length(time)
    x <- rnorm(N)
    eta <- 0.3 + lvl[time] + x * slp[time]
    y <- rnorm(N, eta, 0.4)
    d <- data.frame(y = y, x = x, time = time)

    fit <- suppressWarnings(tulpa(
      y ~ temporal(formula = ~ 1 + x || time, structure = "rw1"),
      data = d, family = "gaussian", mode = "laplace",
      control = list(n_draws = 300L)))

    expect_s3_class(fit, "tulpa_temporal_field_fit")
    expect_setequal(fit$temporal_field_names, c("time.Intercept", "time.x"))

    lvl_hat <- fit$temporal_fields[["time.Intercept"]]$mean
    slp_hat <- fit$temporal_fields[["time.x"]]$mean
    cors_lvl <- c(cors_lvl, abs(cor(lvl_hat, lvl)))
    cors_slp <- c(cors_slp, abs(cor(slp_hat, slp)))
  }
  # Both the level and the time-varying slope recover every seed -- the slope
  # recovery exercises the per-row svc_weight threaded onto the temporal block.
  expect_true(all(cors_lvl > 0.7))
  expect_true(all(cors_slp > 0.7))
})

test_that("ignoring the per-row weight degrades the slope field", {
  skip_on_cran()
  set.seed(7)
  T <- 30L
  slp <- cumsum(rnorm(T)); slp <- slp - mean(slp)
  n_per <- 60L
  time <- rep(seq_len(T), each = n_per)
  N <- length(time)
  x <- rnorm(N)
  y <- rnorm(N, 0.2 + x * slp[time], 0.4)
  d <- data.frame(y = y, x = x, time = time)

  fit <- suppressWarnings(tulpa(
    y ~ temporal(formula = ~ 0 + x || time, structure = "rw1"),
    data = d, family = "gaussian", mode = "laplace",
    control = list(n_draws = 200L)))
  slp_hat <- fit$temporal_fields[["time.x"]]$mean
  expect_gt(abs(cor(slp_hat, slp)), 0.7)
})

test_that("ar1 temporal field fits and reports rho", {
  skip_on_cran()
  set.seed(13)
  T <- 40L
  lvl <- as.numeric(arima.sim(list(ar = 0.7), n = T)); lvl <- lvl - mean(lvl)
  n_per <- 30L
  time <- rep(seq_len(T), each = n_per)
  y <- rnorm(length(time), 0.1 + lvl[time], 0.4)
  fit <- suppressWarnings(tulpa(
    y ~ temporal(formula = ~ 1 || time, structure = "ar1"),
    data = data.frame(y = y, time = time), family = "gaussian",
    mode = "laplace", control = list(n_draws = 200L)))
  expect_s3_class(fit, "tulpa_temporal_field_fit")
  h <- fit$temporal_field_hypers[["time.Intercept"]]
  expect_false(is.null(h$rho))               # ar1 exposes the correlation axis
  expect_length(h$rho, 3L)
  out <- paste(capture.output(print(fit)), collapse = "\n")
  expect_match(out, "rho")
  expect_match(out, "AR1")
})

test_that("inline temporal() fit's $y is the response vector, so diagnostics run (gcol33/tulpa#796)", {
  skip_on_cran()
  set.seed(1)
  n_t <- 30L
  d <- data.frame(tidx = rep(1:n_t, each = 5), x = rnorm(150))
  u <- cumsum(rnorm(n_t, 0, .3))
  d$y <- rpois(150, exp(0.2 + 0.4 * d$x + u[d$tidx]))
  ft <- suppressWarnings(tulpa(
    y ~ x + temporal(formula = ~ 1 || tidx, structure = "rw1"),
    data = d, family = "poisson"))

  expect_true(is.numeric(ft$y))
  expect_false(is.list(ft$y))
  expect_identical(ft$y, d$y)
  expect_false(is.null(ft$n_trials))
  expect_no_error(residuals(ft))
})

# A temporal field with a random intercept on site; poisson counts.
.tf_re_fixture <- function(seed, sigma_re = 0.4, n_groups = 12L) {
  set.seed(seed)
  n_t <- 20L; n <- 360L
  time <- rep(seq_len(n_t), length.out = n)
  g <- sample(n_groups, n, TRUE)
  lvl <- cumsum(rnorm(n_t, 0, 0.3)); lvl <- lvl - mean(lvl)
  u <- rnorm(n_groups, 0, sigma_re)
  d <- data.frame(time = time, g = factor(g), x = rnorm(n))
  d$y <- rpois(n, exp(0.3 + 0.4 * d$x + lvl[time] + u[d$g]))
  list(d = d, u = u)
}
.tf_re_formula <- y ~ x + temporal(formula = ~ 1 || time, structure = "rw1") +
  (1 | g)

test_that("an inline field integrates an unsupplied (1 | g) scale (#1119)", {
  skip_on_cran()
  fx <- .tf_re_fixture(1L)
  fit <- suppressWarnings(tulpa(.tf_re_formula, data = fx$d,
                                family = "poisson"))
  # The term is a latent block after the field, its SD an outer axis.
  expect_identical(fit$re_block_index, list(2L))
  expect_false(fit$re_block_conditioned)
  expect_true("b2.sigma" %in% colnames(fit$theta_grid))
  expect_gt(length(unique(fit$theta_grid[, "b2.sigma"])), 1L)
  expect_lt(fit$theta_ci_lo[["b2.sigma"]], 0.4)
  expect_gt(fit$theta_ci_hi[["b2.sigma"]], 0.4)
  vc <- VarCorr(fit)
  expect_identical(vc$source, "estimated")
  expect_equal(vc$sd, fit$theta_median[["b2.sigma"]])
  re <- ranef(fit)
  expect_identical(nrow(re), 12L)
  expect_gt(cor(re$estimate, fx$u), 0.7)

  # A supplied scale is conditioned on: a one-point axis.
  fit_c <- suppressWarnings(tulpa(.tf_re_formula, data = fx$d,
                                  family = "poisson", sigma_re = 0.4))
  expect_true(fit_c$re_block_conditioned)
  expect_identical(unique(fit_c$theta_grid[, "b2.sigma"]), 0.4)
  # VarCorr() lists the term at the supplied SD, marked as conditioned (#1126).
  vc_c <- VarCorr(fit_c)
  expect_identical(vc_c$term, "g")
  expect_equal(vc_c$sd, 0.4)
  expect_identical(vc_c$source, "conditioned")
})

test_that("an inline field's integrated (1 | g) scale covers the truth over seeds (#1119)", {
  skip_if_not_slow()
  hit <- vapply(seq_len(20L), function(s) {
    fit <- suppressWarnings(tulpa(.tf_re_formula, data = .tf_re_fixture(s)$d,
                                  family = "poisson"))
    fit$theta_ci_lo[["b2.sigma"]] <= 0.4 && 0.4 <= fit$theta_ci_hi[["b2.sigma"]]
  }, logical(1))
  expect_gte(mean(hit), 0.85)
})
