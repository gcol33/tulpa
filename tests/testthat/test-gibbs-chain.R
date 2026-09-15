# A Polya-Gamma Gibbs fit is one MCMC chain: `$draws` in the ModelData
# samplers' parameter naming, with the chain bookkeeping the chain accessors
# read, so diagnostics(), the posterior interop and the criteria see the sample.

test_that("the kernel blocks become sampler-named draw columns", {
  n <- 6L
  res <- list(
    beta = matrix(1:12 / 10, n, 2L),
    re = matrix(0.1, n, 3L), sigma_re = rep(2, n),
    trend = matrix(0.2, n, 4L), seasonal = matrix(0, n, 0L),
    short_term = matrix(0.3, n, 4L),
    sigma_trend = rep(0.5, n), sigma_seasonal = rep(1, n),
    sigma_short = rep(0.25, n), rho_short = rep(0.6, n)
  )
  X <- cbind("(Intercept)" = 1, x = 1:n)
  ch <- .pg_as_chain(res, c("temporal", "ar1_short"), X)

  expect_identical(colnames(ch$draws), c(
    "(Intercept)", "x", "log_sigma_re", "re[1]", "re[2]", "re[3]",
    "log_sigma2_trend", "trend[1]", "trend[2]", "trend[3]", "trend[4]",
    "log_sigma2_short", "short_term[1]", "short_term[2]", "short_term[3]",
    "short_term[4]", "logit_rho_short"))
  expect_identical(ch$param_names, colnames(ch$draws))
  expect_equal(unname(ch$draws[, "log_sigma_re"]), rep(log(2), n))
  expect_equal(unname(ch$draws[, "log_sigma2_trend"]), rep(log(0.25), n))
  expect_equal(unname(ch$draws[, "logit_rho_short"]), rep(stats::qlogis(0.8), n))
  expect_identical(ch$chain_id, rep(1L, n))
  expect_identical(ch$n_chains, 1L)
  expect_equal(ch$means, colMeans(ch$draws))

  # No random-intercept block: its scale is a placeholder, not a draw.
  res$re <- matrix(0, n, 0L)
  expect_false("log_sigma_re" %in% colnames(.pg_as_chain(res, "temporal", X)$draws))

  # An unnamed design gets the samplers' positional fixed-effect names.
  expect_identical(colnames(.pg_as_chain(res, "temporal", unname(X))$draws)[1:2],
                   c("beta[1]", "beta[2]"))

  expect_error(.pg_as_chain(res[setdiff(names(res), "trend")], "temporal", X),
               "no 'trend' block")
})

test_that("mode = 'gibbs' carries its chain to the chain accessors", {
  skip_on_cran()
  set.seed(1); N <- 300
  d <- data.frame(x = rnorm(N), g = factor(rep(1:30, each = 10)))
  d$yb <- rbinom(N, 1, plogis(0.2 + 0.5 * d$x + rnorm(30, 0, 0.6)[d$g]))
  fg <- tulpa(yb ~ x + (1 | g), d, family = "binomial", mode = "gibbs",
              control = list(n_iter = 300L, warmup = 150L, seed = 1L))

  expect_identical(fg$draws_kind, "chain")
  expect_equal(dim(fg$draws), c(150L, 33L))
  expect_identical(colnames(fg$draws)[1:4],
                   c("(Intercept)", "x", "log_sigma_re", "re[1]"))
  expect_identical(fg$param_names, colnames(fg$draws))
  expect_identical(fg$n_chains, 1L)
  for (nm in c("beta", "re", "sigma_re")) expect_null(fg[[nm]], info = nm)

  dg <- diagnostics(fg)
  expect_setequal(names(dg), c("parameter", "rhat", "ess_bulk", "ess_tail"))
  expect_identical(dg$parameter, colnames(fg$draws))
  expect_true(all(is.finite(dg$rhat)))

  expect_equal(unname(coef(fg)), unname(colMeans(fg$draws[, 1:2])))
  expect_equal(nrow(ranef(fg)), 30L)
  vc <- VarCorr(fg)
  expect_equal(vc$coef, "(Intercept)")
  expect_equal(vc$sd, mean(exp(fg$draws[, "log_sigma_re"])))

  skip_if_not_installed("posterior")
  df <- posterior::as_draws_df(fg)
  expect_equal(posterior::ndraws(df), 150L)
  expect_true("log_sigma_re" %in% posterior::variables(df))
})

test_that("the as_draws refusal describes what the fit carries", {
  fit <- structure(list(backend = "laplace", draws_kind = "iid", mode = c(0, 1),
                        H_beta = diag(2)),
                   class = "tulpa_fit")
  skip_if_not_installed("posterior")
  expect_error(as_draws_df(fit), "Laplace mode and its precision")
  chain <- structure(list(backend = "gibbs", draws_kind = "chain"),
                     class = "tulpa_fit")
  expect_error(as_draws_df(chain), "stamped as an MCMC chain")
})
