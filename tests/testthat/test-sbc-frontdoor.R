# The exported SBC front door (gcol33/tulpa#380).
#
# The scorer itself -- the predictive shapes, the within-atom PIT, the CRPS
# closed forms and the exact simultaneous band -- is arbitrated in
# test-sbc-crps.R against closed forms, brute-force simulation, numerical
# integration and the published Kolmogorov critical value. Nothing here
# re-arbitrates it. What is pinned here is the DOOR: that it runs the same
# experiment the drivers do, that its two guards fire on the premises they
# describe and only on those, and that the result reads through its methods.
#
# Every model below is conjugate and solved in closed form, so the whole file
# runs without a fit. The engine fixture goes through the door in
# test-sbc-crps.R and test-posterior-sbc.R, where it already has the seeds and
# the sample sizes its verdicts were measured at.

# ---------------------------------------------------------------------------
# Conjugate fixtures: y_i ~ N(mu, 1) with mu ~ N(0, 1), whose posterior is
# N(v * sum(y), v), v = 1 / (1 + n).
# ---------------------------------------------------------------------------

fd_sim <- function(seed) {
  set.seed(seed)
  mu <- stats::rnorm(1)
  list(y = stats::rnorm(10L, mu, 1), theta = c(mu = mu))
}

fd_post <- function(y) {
  v <- 1 / (1 + length(y))
  c(mean = v * sum(y), var = v)
}

fd_fitter <- function(d) {
  p <- fd_post(d$y)
  list(exact  = list(mu = sbc_normal(p[["mean"]], sqrt(p[["var"]]))),
       narrow = list(mu = sbc_normal(p[["mean"]], sqrt(p[["var"]]) / 2)))
}

# The same model wired for posterior SBC.
fd_model <- function(n_obs = 20L, group_ids = NULL, pool = NULL) {
  set.seed(7L)
  d_obs <- list(y = stats::rnorm(n_obs, 0.4, 1), g = seq_len(n_obs))
  m <- list(
    data_obs = d_obs,
    fit = function(data) fd_post(data$y),
    draw_theta = function(fit, seed) {
      set.seed(seed)
      c(mu = stats::rnorm(1, fit[["mean"]], sqrt(fit[["var"]])))
    },
    simulate = function(theta, seed) {
      set.seed(seed)
      list(y = stats::rnorm(10L, theta[["mu"]], 1), g = seq_len(10L))
    },
    pool = function(obs, rep) list(y = c(obs$y, rep$y),
                                   g = c(obs$g, rep$g + length(obs$g))),
    arms = function(fit, data)
      list(exact = list(mu = sbc_normal(fit[["mean"]], sqrt(fit[["var"]])))))
  if (!is.null(group_ids)) m$group_ids <- group_ids
  if (!is.null(pool)) m$pool <- pool
  m
}

# ---------------------------------------------------------------------------
# 1. One implementation of the scorer
# ---------------------------------------------------------------------------

test_that("the scorer is the package's, not a private copy in the helper", {
  ns <- asNamespace("tulpa")
  for (f in c("sbc_mixture", "sbc_normal", "sbc_discrete", "sbc_rank",
              "sbc_draws", "sbc_pit", "sbc_fold", "sbc_crps",
              "sbc_crossing_prob", "sbc_ecdf_band", "sbc_ecdf_test",
              "recov_sbc", "recov_posterior_sbc", "sbc_report",
              "sbc_crps_compare")) {
    expect_true(exists(f, envir = ns, inherits = FALSE),
                label = sprintf("%s lives in the package", f))
    expect_identical(environment(get(f, envir = ns)), ns,
                     label = sprintf("%s is bound in the namespace", f))
  }
  # And the helper does not redefine any of them, which is the duplicate the
  # promotion exists to prevent.
  h <- readLines(test_path("helper-sbc.R"))
  defs <- grep("^\\s*(sbc_[a-z_0-9]+|recov_[a-z_0-9]+)\\s*<-\\s*function", h,
               value = TRUE)
  redefined <- sub("^\\s*([A-Za-z_.0-9]+)\\s*<-.*$", "\\1", defs)
  expect_length(intersect(redefined,
                          c("sbc_mixture", "sbc_normal", "sbc_discrete",
                            "sbc_rank", "sbc_draws", "sbc_pit", "sbc_fold",
                            "sbc_crps", "sbc_crossing_prob", "sbc_ecdf_band",
                            "sbc_ecdf_test", "recov_sbc",
                            "recov_posterior_sbc", "sbc_report",
                            "sbc_crps_compare")), 0L)
})

test_that("the five predictive shapes are exported and the verb is the only one", {
  ex <- getNamespaceExports("tulpa")
  expect_true(all(c("sbc", "sbc_mixture", "sbc_normal", "sbc_discrete",
                    "sbc_rank", "sbc_draws") %in% ex))
  # The drivers, the report and the band stay internal: `sbc()` is the verb.
  expect_false(any(c("recov_sbc", "recov_posterior_sbc", "sbc_report",
                     "sbc_crps_compare", "sbc_ecdf_band", "sbc_pit") %in% ex))
})

# ---------------------------------------------------------------------------
# 2. The door runs the driver's own experiment
# ---------------------------------------------------------------------------

test_that("the front door reproduces the driver bit for bit", {
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
             n_sim = 40L, seed = 11L)
  expect_s3_class(res, "sbc")
  expect_identical(res$pit,
                   recov_sbc(fd_sim, fd_fitter, n_seed = 40L, seed_off = 11L,
                             truth = "prior_draw"))
  expect_identical(res$report, sbc_report(res$pit, level = 0.95))
  expect_identical(attr(res$pit, "truth"), "prior_draw")
  expect_identical(res$crps_role, "proper posterior score")
  expect_equal(res$bands[["40"]]$coverage, 0.95, tolerance = 1e-5)
})

test_that("the exact arm calibrates and the known-bad control does not", {
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
             n_sim = 120L, seed = 5000L)
  r <- res$report
  expect_true(r$inside[r$arm == "exact"])
  expect_true(r$inside_folded[r$arm == "exact"])
  expect_gt(r$p_unif[r$arm == "exact"], 0.01)
  # A posterior half as wide as the truth is a symmetric dispersion error, so
  # the folded read is what has to catch it.
  expect_false(r$inside_folded[r$arm == "narrow"])
  # And the proper score ranks them the same way round.
  cmp <- summary(res, baseline = "exact")$compare
  expect_gt(cmp$delta[cmp$arm == "narrow"], 0)
})

test_that("n_ref reaches the fitter and quantities restrict the scoring", {
  fitter <- function(d, n_ref = 50L)
    list(a = list(mu = sbc_normal(mean(d$y), 0.3),
                  r = sbc_rank(sum(stats::runif(n_ref) < 0.5), n_ref)))
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fitter,
             n_sim = 20L, n_ref = 13L)
  expect_identical(unique(res$pit$kind[res$pit$quantity == "r"]), "rank")
  expect_true(all(res$pit$pit[res$pit$quantity == "r"] <= 1))
  expect_identical(res$n_ref, 13L)

  only <- sbc("prior_predictive", simulator = fd_sim, fitter = fitter,
              n_sim = 20L, quantities = "mu")
  expect_identical(unique(only$report$quantity), "mu")
})

# ---------------------------------------------------------------------------
# 3. The proper-prior guard
# ---------------------------------------------------------------------------

test_that("a truth held fixed across simulations is refused, naming the way out", {
  fixed <- function(seed) {
    set.seed(seed)
    list(y = stats::rnorm(10L, 0.3, 1), theta = c(mu = 0.3))
  }
  expect_error(
    sbc("prior_predictive", simulator = fixed, fitter = fd_fitter, n_sim = 30L),
    "PROPER prior")
  expect_error(
    sbc("prior_predictive", simulator = fixed, fitter = fd_fitter, n_sim = 30L),
    'experiment = "posterior"')

  # Admitted by the structural argument the caller asserts, which travels on
  # the result.
  ok <- sbc("prior_predictive", simulator = fixed, fitter = fd_fitter,
            n_sim = 30L, flat_prior = "mu")
  expect_identical(ok$premises$flat_prior, "mu")
  expect_identical(ok$premises$proper_prior, "verified")
  expect_identical(ok$premises$n_probed, 10L)
})

test_that("the declaration is checked in both directions", {
  # A quantity that DID move contradicts the declaration.
  expect_error(sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
                   n_sim = 20L, flat_prior = "mu"),
               "but the truth moved")
  # A declaration naming something the fitter does not score.
  expect_error(sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
                   n_sim = 20L, flat_prior = "sigma"),
               "does not score")
})

test_that("a rank quantity needs no truth and is exempt from the guard", {
  # Every arm reports it as a rank, so there is no prior to be proper.
  fitter <- function(d)
    list(a = list(r = sbc_rank(sum(stats::runif(40L) < 0.5), 40L)))
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fitter,
             n_sim = 20L)
  expect_identical(unique(res$report$quantity), "r")
  expect_true(all(is.na(res$pit$crps)))
})

# ---------------------------------------------------------------------------
# 4. The posterior experiment's two premises
# ---------------------------------------------------------------------------

test_that("the posterior door runs and records what it verified", {
  m <- fd_model(group_ids = function(data) data$g)
  res <- sbc("posterior", model = m, n_sim = 25L)
  expect_identical(res$experiment, "posterior")
  expect_identical(attr(res$pit, "truth"), "posterior_draw")
  expect_identical(res$premises$pooling, "verified")
  expect_identical(res$premises$fresh_groups, "verified (disjoint group labels)")
  # The exact posterior calibrates under the construction.
  expect_true(res$report$inside)

  # Without `group_ids` the premise is recorded as UNVERIFIED, not assumed.
  res2 <- sbc("posterior", model = fd_model(), n_sim = 25L)
  expect_identical(res2$premises$fresh_groups, "undeclared")
  expect_identical(res2$pit, res$pit)
})

test_that("a pool that drops the observed data is refused", {
  # The documented collapse: fitting the replicate ALONE is ordinary SBC under
  # a hand-made prior, and the sequential-updating identity does not hold.
  m <- fd_model(pool = function(obs, rep) rep)
  expect_error(sbc("posterior", model = m, n_sim = 5L),
               "AUGMENTED posterior")
  # And one that drops the replicate: nothing is being calibrated.
  m2 <- fd_model(pool = function(obs, rep) obs)
  expect_error(sbc("posterior", model = m2, n_sim = 5L),
               "not conditioned on")
  # A pool that merely WRAPS both is not refused: containment holds.
  m3 <- fd_model(pool = function(obs, rep)
    list(y = c(obs$y, rep$y), g = c(obs$g, rep$g + length(obs$g)),
         parts = list(obs = obs, rep = rep)))
  expect_s3_class(sbc("posterior", model = m3, n_sim = 5L), "sbc")
})

test_that("a replicate re-observing the same groups is refused", {
  m <- fd_model(group_ids = function(data) data$g,
                pool = function(obs, rep)
                  list(y = c(obs$y, rep$y), g = c(obs$g, rep$g)))
  expect_error(sbc("posterior", model = m, n_sim = 5L), "FRESH groups")
})

test_that("the two experiments do not accept each other's arguments", {
  m <- fd_model()
  expect_error(sbc("posterior", model = m, n_sim = 5L, flat_prior = "mu"),
               "prior-predictive experiment")
  expect_error(sbc("posterior", simulator = fd_sim, n_sim = 5L),
               "belong to the prior-predictive")
  expect_error(sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
                   model = m, n_sim = 5L),
               "posterior experiment's callback list")
  expect_error(sbc("prior_predictive", n_sim = 5L), "needs simulator")
  expect_error(sbc("posterior", n_sim = 5L), "needs model")
})

# ---------------------------------------------------------------------------
# 5. The CRPS scoping refusal survives the promotion
# ---------------------------------------------------------------------------

test_that("CRPS is refused as a ranking where it is not a proper posterior score", {
  # The door offers only the two experiments in which the score IS proper, so a
  # fixed-truth result cannot come out of it -- and the refusal underneath is
  # still enforced.
  expect_error(sbc("fixed", simulator = fd_sim, fitter = fd_fitter),
               "should be one of")
  fx <- recov_sbc(fd_sim, fd_fitter, n_seed = 4L, truth = "fixed")
  expect_error(sbc_crps_compare(fx, "exact"), "prior-predictive")

  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
             n_sim = 20L)
  expect_error(summary(res, baseline = "nosucharm"), "is not one of")
})

# ---------------------------------------------------------------------------
# 6. The methods
# ---------------------------------------------------------------------------

test_that("print, summary, plot and diagnostics read the result", {
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
             n_sim = 40L, seed = 3L)
  out <- utils::capture.output(print(res))
  expect_true(any(grepl("Simulation-based calibration", out)))
  expect_true(any(grepl("proper posterior score", out)))
  expect_true(any(grepl("inside the band", out)))

  s <- summary(res, baseline = "exact")
  expect_s3_class(s, "sbc_summary")
  sout <- utils::capture.output(print(s))
  expect_true(any(grepl("Paired CRPS against arm", sout)))

  # pdf() is the one device present on every platform. png() selects whatever
  # getOption("bitmapType") names, which resolves to the X11 driver on a
  # headless machine and cannot start there even where capabilities("png") is
  # TRUE. Nothing below reads raster output.
  pf <- tempfile(fileext = ".pdf")
  grDevices::pdf(pf, width = 6, height = 4)
  plot(res)
  plot(res, arm = "exact", folded = TRUE)
  grDevices::dev.off()
  expect_true(file.exists(pf))
  expect_error(plot(res, arm = "nosucharm"), "no \\(arm, quantity\\) selected")

  # A panel's annotation reads the sample the panel draws. A folded panel
  # carrying the raw p-value and the raw band verdict is the fold's own failure
  # mode: it reports "inside" over a curve outside the band, on exactly the
  # symmetric dispersion error the raw ECDF cancels. The columns are read off
  # `res$report`, whose folded p-value is `p_unif_folded`; `p_folded` is the
  # name the PRINT method renames it to, and reading that here would be
  # `NULL[i]`, a zero-length label rather than a wrong one.
  expect_true(all(c("p_unif", "inside", "p_unif_folded", "inside_folded") %in%
                    names(res$report)))
  r <- data.frame(p_unif = 0.90, inside = TRUE,
                  p_unif_folded = 0.0001, inside_folded = FALSE)
  expect_identical(.sbc_panel_note(r, 1L, folded = FALSE), "p = 0.9")
  expect_identical(.sbc_panel_note(r, 1L, folded = TRUE),
                   "p = 0.0001, outside band")

  d <- diagnostics(res)
  expect_true(all(c("arm", "quantity", "inside", "p_unif") %in% names(d)))
  expect_identical(attr(d, "experiment"), "prior_predictive")
})

test_that("a fit's reliability table carries the calibration verdict alongside", {
  skip_on_cran()
  set.seed(1L)
  n <- 120L
  x <- stats::rnorm(n)
  y <- stats::rbinom(n, 1L, stats::plogis(-0.2 + 0.6 * x))
  fit <- tulpa(y ~ x, data.frame(y = y, x = x), family = "binomial",
               mode = "smc")
  res <- sbc("prior_predictive", simulator = fd_sim, fitter = fd_fitter,
             n_sim = 30L)
  plain <- diagnostics(fit)
  both <- diagnostics(fit, sbc = res)
  expect_null(attr(plain, "sbc_verdict"))
  expect_identical(attr(both, "sbc_experiment"), "prior_predictive")
  # The verdict rides ALONGSIDE the reliability table; the table itself is
  # untouched.
  stripped <- both
  attr(stripped, "sbc_report") <- NULL
  attr(stripped, "sbc_experiment") <- NULL
  attr(stripped, "sbc_verdict") <- NULL
  expect_identical(plain, stripped)
  out <- utils::capture.output(print(both))
  expect_true(any(grepl("calibration \\(SBC", out)))
  expect_error(diagnostics(fit, sbc = "not an sbc result"), "takes an `sbc`")
})
