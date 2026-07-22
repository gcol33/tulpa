# Structural tests for the named front doors (tglmm / tgam). These exercise the
# contract narrowing -- signature, formula requirements, refusals, subclass and
# display -- not the statistical behaviour of any backend, which is the engine's
# own recovery suite.

test_that("each door's signature is tulpa()'s minus its withheld arguments", {
  # The doors carry an explicit signature so a user can read and tab-complete
  # it. This is the tripwire that keeps it from going stale: a new statistical
  # argument on tulpa() must be added to (or deliberately withheld from) every
  # door, and until then this fails.
  for (door in names(tulpa:::.TULPA_DOORS)) {
    spec     <- tulpa:::.TULPA_DOORS[[door]]
    expected <- setdiff(names(formals(tulpa::tulpa)), spec$withheld)
    got      <- names(formals(get(door, envir = asNamespace("tulpa"))))
    expect_identical(got, expected,
                     info = paste0(door, "() signature has drifted from tulpa()"))
  }
})

test_that("withheld arguments are absent from the door signatures", {
  expect_false("spatial"  %in% names(formals(tulpa::tglmm)))
  expect_false("temporal" %in% names(formals(tulpa::tglmm)))
  expect_false("spatial"  %in% names(formals(tulpa::tgam)))
  expect_false("temporal" %in% names(formals(tulpa::tgam)))
})

test_that("every door registry entry is well formed", {
  feats <- names(tulpa:::.DOOR_FEATURES)
  for (door in names(tulpa:::.TULPA_DOORS)) {
    spec <- tulpa:::.TULPA_DOORS[[door]]
    expect_true(all(spec$requires %in% feats))
    expect_true(all(spec$forbids %in% feats))
    # A door cannot both require and forbid the same structure.
    expect_length(intersect(spec$requires, spec$forbids), 0L)
    # Every feature named in a contract has message text.
    expect_true(all(c(spec$requires, spec$forbids) %in%
                      names(tulpa:::.DOOR_FEATURE_SYNTAX)))
    # Every field the door body and its messages read is present, so a new
    # registry entry cannot half-configure a door.
    for (f in c("subclass", "label", "long", "degenerate")) {
      expect_true(is.character(spec[[f]]) && nzchar(spec[[f]]),
                  info = paste0(door, ": registry field '", f, "' is missing"))
    }
  }
})

test_that("tglmm() requires a random-effect term", {
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  expect_error(tglmm(y ~ x, data = d, family = "gaussian"),
               "needs a random-effect term")
})

test_that("tgam() requires a smoother term", {
  d <- data.frame(y = rnorm(20), x = rnorm(20))
  expect_error(tgam(y ~ x, data = d, family = "gaussian"),
               "needs an s\\(\\.\\.\\.\\) smoother")
})

test_that("tglmm() refuses structures outside the model class", {
  d <- data.frame(y = rnorm(60), x = runif(60), g = factor(rep(1:6, 10)))
  # A smoother is a GAM, not a GLMM.
  expect_error(tglmm(y ~ s(x) + (1 | g), data = d, family = "gaussian"),
               "also carries an s\\(\\.\\.\\.\\) smoother")
})

test_that("doors redirect a withheld argument instead of reporting a typo", {
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  err <- tryCatch(
    tglmm(y ~ x + (1 | g), data = d, family = "gaussian",
          spatial = list(type = "icar")),
    error = conditionMessage)
  expect_match(err, "does not take `spatial`")
  expect_match(err, "tulpa\\(\\)")
})

test_that("a withheld argument cannot slip through by partial matching", {
  # `spat = ` would partial-match tulpa()'s `spatial` on the way through and fit
  # the very model the door exists to exclude.
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  expect_error(
    tglmm(y ~ x + (1 | g), data = d, family = "gaussian",
          spat = list(type = "icar")),
    "does not take `spatial`")
})

test_that("a door with no formula raises the standard missing-argument error", {
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  expect_error(tglmm(data = d, family = "gaussian"), "formula")
})

test_that("doors reject an unknown argument rather than ignoring it", {
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  expect_error(tglmm(y ~ x + (1 | g), data = d, family = "gaussian",
                     n_iter = 10),
               "unknown argument")
})

test_that("doors validate control names through the shared engine check", {
  d <- data.frame(y = rnorm(20), x = rnorm(20), g = factor(rep(1:4, 5)))
  expect_error(tglmm(y ~ x + (1 | g), data = d, family = "gaussian",
                     control = list(max_itr = 5)),
               "Unknown control knob")
})

test_that("the door label lookup reads the registry", {
  expect_null(tulpa:::.door_label_for(structure(list(), class = "tulpa_fit")))
  expect_identical(
    tulpa:::.door_label_for(structure(list(), class = c("tulpa_glmm", "tulpa_fit"))),
    "GLMM")
  expect_identical(
    tulpa:::.door_label_for(structure(list(), class = c("tulpa_gam", "tulpa_fit"))),
    "GAM")
})

test_that("tglmm() stamps its subclass, its own call, and prints VarCorr", {
  skip_on_cran()
  set.seed(4)
  n <- 200L
  g <- rep(seq_len(10), length.out = n)
  d <- data.frame(x = rnorm(n), g = factor(g))
  d$y <- rnorm(n, 0.3 + 0.5 * d$x + rnorm(10, 0, 0.6)[g], 1)

  fit <- tglmm(y ~ x + (1 | g), data = d, family = "gaussian",
               mode = "laplace", sigma_re = 0.6)

  expect_s3_class(fit, "tulpa_glmm")
  expect_s3_class(fit, "tulpa_fit")
  # The recorded call is the door's, not the tulpa() call it dispatched through.
  expect_identical(as.character(fit$call[[1L]]), "tglmm")

  # Through print(), not by name: the door's display is only reached if its
  # method is registered for S3 dispatch, which is part of what is being tested.
  out <- utils::capture.output(print(fit))
  expect_match(out[1], "tulpa GLMM fit")
  expect_true(any(grepl("Random effects", out, fixed = TRUE)))
  # sigma_re was supplied, so VarCorr must label it conditioned rather than
  # present the input as an estimate.
  expect_true(any(grepl("conditioned", out, fixed = TRUE)))

  # The door narrows the contract; it does not change the fit.
  direct <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian",
                  mode = "laplace", sigma_re = 0.6)
  expect_equal(unname(coef(fit)), unname(coef(direct)), tolerance = 1e-10)
})

test_that("tgam() stamps its subclass and lists its smoothers", {
  skip_on_cran()
  set.seed(5)
  d <- data.frame(x = runif(200, -2, 2))
  d$y <- rpois(200, exp(0.3 + sin(2 * d$x)))

  fit <- tgam(y ~ s(x, k = 12), data = d, family = "poisson")

  expect_s3_class(fit, "tulpa_gam")
  expect_identical(as.character(fit$call[[1L]]), "tgam")

  out <- utils::capture.output(print(fit))
  expect_match(out[1], "tulpa GAM fit")
  expect_true(any(grepl("s(x)", out, fixed = TRUE)))
  # The door subclass takes over dispatch from the nested-Laplace print method,
  # so the door composes that backend's own report rather than dropping it.
  expect_true(any(grepl("hyperparameters", out, fixed = TRUE)))

  sm <- smooth_effects(fit)
  expect_true(is.data.frame(sm) && nrow(sm) == 12L)
})
