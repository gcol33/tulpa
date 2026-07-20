# lme4- / posterior-facing accessors: fixef(), as_draws() and the S3 registration
# that makes lme4::fixef() and posterior::as_draws() dispatch on a tulpa_fit.

sampler_fit <- function(seed = 1L) {
  set.seed(seed)
  df <- data.frame(x = rnorm(120))
  df$y <- rpois(120, exp(0.5 + 0.4 * df$x))
  tulpa(y ~ x, data = df, family = "poisson", mode = "exact",
        control = list(n_iter = 400, warmup = 200, seed = seed))
}

approx_fit <- function(seed = 2L) {
  set.seed(seed)
  n <- 200L
  g <- rep(seq_len(25L), each = 8L)
  df <- data.frame(x = rnorm(n), g = factor(g))
  df$y <- rpois(n, exp(0.4 + 0.6 * df$x + rnorm(25L, 0, 0.7)[g]))
  suppressMessages(
    tulpa(y ~ x + (1 | g), data = df, family = "poisson", mode = "laplace"))
}


# --- fixef -------------------------------------------------------------------

test_that("fixef() returns the fixed effects for both posterior shapes", {
  for (fit in list(sampler_fit(), approx_fit())) {
    fe <- fixef(fit)
    expect_true(is.numeric(fe))
    expect_identical(fe, coef(fit))
    expect_identical(names(fe), fit$fixed_names)
  }
})

test_that("fixef() excludes the random effects", {
  fit <- approx_fit()
  # coef()/fixef() are the fixed block only; the RE block is ranef()'s. A fit
  # whose fixef() length tracked the mode vector would silently return RE values.
  expect_length(fixef(fit), fit$n_fixed)
  expect_lt(length(fixef(fit)), length(fit$mode))
})

test_that("lme4::fixef() and nlme::ranef() dispatch on a tulpa_fit", {
  fit <- sampler_fit()
  if (requireNamespace("lme4", quietly = TRUE)) {
    expect_identical(lme4::fixef(fit), fixef(fit))
  } else {
    skip("lme4 not installed")
  }
})

test_that("nlme::fixef() dispatches on a tulpa_fit", {
  skip_if_not_installed("nlme")
  fit <- sampler_fit()
  expect_identical(nlme::fixef(fit), fixef(fit))
  expect_s3_class(nlme::ranef(fit), "data.frame")
})

test_that("registration is armed for packages that load after tulpa", {
  # Registering only what is loaded at .onLoad time would leave `lme4::fixef(fit)`
  # broken for the (normal) session that attaches lme4 afterwards, so the load
  # hook is the half that has to exist whether or not the package is installed.
  for (p in c("lme4", "nlme", "posterior")) {
    expect_gt(length(getHook(packageEvent(p, "onLoad"))), 0L)
  }
})

test_that(".s3_register tolerates a package without the generic", {
  # nlme and lme4 do not carry identical surfaces, and posterior's shape variants
  # have come and gone; a generic the other package lacks must be a non-event.
  expect_silent(tulpa:::.s3_register("stats::no_such_generic_xyz", "tulpa_fit"))
  expect_silent(tulpa:::.s3_register("stats::coef", "no_such_class_xyz"))
  expect_error(tulpa:::.s3_register("not_namespaced", "tulpa_fit"),
               "package::generic")
})


# --- as_draws ----------------------------------------------------------------

test_that("as_draws() converts a sampler fit to every posterior shape", {
  skip_if_not_installed("posterior")
  fit <- sampler_fit()

  expect_s3_class(as_draws(fit), "draws_array")
  expect_s3_class(as_draws_array(fit), "draws_array")
  expect_s3_class(as_draws_matrix(fit), "draws_matrix")
  expect_s3_class(as_draws_df(fit), "draws_df")
  expect_s3_class(as_draws_rvars(fit), "draws_rvars")

  arr <- as_draws_array(fit)
  expect_identical(dim(arr), dim(tulpa_draws_array(fit)))
  expect_identical(posterior::variables(arr), fit$param_names)
})

test_that("posterior::as_draws() dispatches on a tulpa_fit", {
  skip_if_not_installed("posterior")
  fit <- sampler_fit()
  expect_identical(posterior::as_draws(fit), as_draws(fit))
  expect_identical(posterior::as_draws_df(fit), as_draws_df(fit))
})

test_that("as_draws() refuses a Gaussian-approximation fit by default", {
  skip_if_not_installed("posterior")
  fit <- approx_fit()
  # Silently sampling the approximation would hand back an object every
  # posterior summary treats as a posterior sample, so it has to be opt-in.
  expect_error(as_draws(fit), "carries no posterior draws")
  expect_error(as_draws_matrix(fit), "carries no posterior draws")
  expect_error(as_draws(fit), "n_draws")
})

test_that("n_draws synthesizes from the Gaussian approximation", {
  skip_if_not_installed("posterior")
  fit <- approx_fit()
  d <- as_draws(fit, n_draws = 4000, seed = 42)
  expect_s3_class(d, "draws_array")
  expect_identical(dim(d), c(4000L, 1L, length(coef(fit))))
  expect_identical(posterior::variables(d), names(coef(fit)))

  # The synthesis must be N(coef, vcov): recover both moments from the draws.
  m <- as_draws_matrix(fit, n_draws = 20000, seed = 7)
  expect_equal(as.numeric(colMeans(m)), as.numeric(coef(fit)), tolerance = 0.05)
  expect_equal(as.numeric(stats::cov(m)), as.numeric(vcov(fit)),
               tolerance = 0.1)
})

test_that("the synthesis is seeded and leaves the RNG state untouched", {
  skip_if_not_installed("posterior")
  fit <- approx_fit()
  expect_identical(as_draws(fit, n_draws = 100, seed = 3),
                   as_draws(fit, n_draws = 100, seed = 3))

  set.seed(99)
  before <- .Random.seed
  invisible(as_draws(fit, n_draws = 100, seed = 3))
  expect_identical(.Random.seed, before)
})

test_that("n_draws is rejected or ignored where it does not apply", {
  skip_if_not_installed("posterior")
  expect_warning(as_draws(sampler_fit(), n_draws = 10), "ignored")
  expect_error(as_draws(approx_fit(), n_draws = 0), "positive integer")
  expect_error(as_draws(approx_fit(), n_draws = -5), "positive integer")
})


# --- `$` is exact on a fit object --------------------------------------------

test_that("`$` on a tulpa_fit does not partial-match", {
  # A fit is a list, so `$`'s default partial matching would let an ABSENT field
  # resolve to any longer field it prefixes. The shape tests in
  # .fit_fixed_table() / vcov() decide which posterior a fit carries by asking
  # whether `$draws` / `$mode` / `$modes` / `$cov` is NULL, so a partial match
  # there reads the wrong object entirely.
  fit <- sampler_fit()
  expect_true(is.matrix(fit$model_matrix))
  expect_null(fit$mode)            # would otherwise resolve to model_matrix
  expect_null(fit$model)
  expect_null(fit$mod)

  af <- approx_fit()
  expect_identical(af$draws_kind, "iid")
  expect_null(af$draws)            # would otherwise resolve to draws_kind
  expect_null(af$draw)
})

test_that("`$` still returns fields that are actually present", {
  fit <- sampler_fit()
  for (f in c("backend", "draws", "param_names", "fixed_names", "n_fixed",
              "model_matrix", "formula", "family", "N")) {
    expect_identical(fit[[f]], eval(call("$", fit, f)), info = f)
  }
  expect_false(is.null(fit$draws))
})

test_that("`$` exactness reaches subclasses and does not break assignment", {
  fit <- sampler_fit()
  expect_true(inherits(fit, "tulpa_fit"))
  expect_gt(length(class(fit)), 1L)   # a subclass, inheriting the method

  fit$brand_new_field <- 42
  expect_identical(fit$brand_new_field, 42)
  expect_identical(fit[["brand_new_field"]], 42)
  expect_null(fit$brand_new)          # assignment did not reintroduce partial matching
})

test_that("print() does not relabel a random-effect sd as the dispersion", {
  # print.tulpa_fit reports `x$sigma`. An AGQ fit carries `sigma_re` and no
  # `sigma`, so partial matching printed the RE standard deviation under the
  # "sigma:" label -- readable as the residual/dispersion scale.
  set.seed(5)
  n <- 200L
  g <- rep(seq_len(25L), each = 8L)
  df <- data.frame(g = factor(g))
  df$y <- rpois(n, exp(0.3 + rnorm(25L, 0, 0.6)[g]))
  fit <- tryCatch(suppressMessages(
    tulpa(y ~ 1 + (1 | g), data = df, family = "poisson", mode = "agq")),
    error = function(e) NULL)
  skip_if(is.null(fit), "agq fit unavailable")

  expect_null(fit$sigma)
  expect_false(is.null(fit$sigma_re))
  expect_false(any(grepl("^sigma:", utils::capture.output(print(fit)))))
})


# --- the draws accessors on a fit that carries none --------------------------

test_that("draws accessors return NULL, not the draws_kind tag", {
  # `$` partial-matches on lists, so `fit$draws` on a Gaussian-approximation fit
  # resolves to `draws_kind` -- the string "iid". Every draws accessor must read
  # the field exactly, or it hands callers a tag dressed as a draws matrix.
  fit <- approx_fit()
  expect_null(fit[["draws"]])
  expect_identical(fit$draws_kind, "iid")

  expect_null(posterior_sample(fit))
  expect_null(mcmc_draws(fit))
  expect_null(tulpa_draws_array(fit))
  expect_null(tulpa:::.tulpa_pooled_draws(fit))
  expect_null(tulpa:::.tulpa_chain_list(fit))
})

test_that("the draws accessors still find real draws", {
  fit <- sampler_fit()
  expect_true(is.matrix(posterior_sample(fit)) ||
                length(dim(posterior_sample(fit))) == 3L)
  expect_false(is.null(tulpa_draws_array(fit)))
  expect_equal(dim(tulpa_draws_array(fit))[3L], length(fit$param_names))
})

test_that("laplace_diagnostics() reports the absence of draws", {
  fit <- approx_fit()
  expect_message(res <- laplace_diagnostics(fit), "no posterior draws")
  expect_null(res)
})
