# What a fit's diagnostics and accessors SAY about themselves.
#
# Each of these was a report that read clean, or read as a different quantity
# than it was, on a fit that could not support the claim.

set.seed(1)
n  <- 200
g  <- rep(1:20, each = 10)
x  <- rnorm(n)
u  <- rnorm(20, 0, 0.6)
yp <- rpois(n, exp(0.3 + 0.4 * x + u[g]))
dd <- data.frame(x = x, yp = yp, g = factor(g), s = factor(g))

test_that("an uncomputed Pareto k-hat is a decline, not a clean pass", {
  # gcol33/tulpa#709: the VI kernel initialises psis_k to a -1 "not computed"
  # sentinel, which is FINITE, so the decline branch written for exactly this
  # case (`!is.finite(k)`) was never reached and .tulpa_khat_band(-1) read
  # "good" -- a diagnostic that never ran reported the cleanest verdict there is.
  skip_on_cran()
  fv <- tulpa(yp ~ x + (1 | g), data = dd, family = "poisson", mode = "vi",
              control = list(seed = 1L))
  expect_true(is.na(fv[["pareto_k"]]))
  expect_null(fv[["theta_grid"]])

  d <- diagnostics(fv)
  expect_true(is.na(attr(d, "pareto_k")))
  expect_true(is.na(attr(d, "pareto_k_band")))
  expect_false(is.null(attr(d, "pareto_k_declined")))

  # A genuine shape CAN be negative, so the band itself is unchanged: only the
  # sentinel is translated, at the one place it is known to be one.
  expect_identical(tulpa:::.tulpa_khat_band(-0.07), "good")
})

test_that("the approximation table carries no rhat / ESS columns", {
  # gcol33/tulpa#713: the provenance gate withholds Rhat and ESS on a non-chain
  # fit because they are vacuous there, then dispatched to a table that computed
  # them anyway under those exact names -- which is what check_diagnostics(),
  # plot_rhat() and any programmatic read consume.
  skip_on_cran()
  fv <- tulpa(yp ~ x + (1 | g), data = dd, family = "poisson", mode = "vi",
              control = list(seed = 1L))
  d <- diagnostics(fv)
  expect_false(any(c("rhat", "ess_bulk", "ess_tail") %in% names(d)))
  expect_true(all(c("mean", "sd", "n_draws", "mcse_mean") %in% names(d)))
  # mcse_mean is the quantity they were standing in for, and it is sd/sqrt(n).
  expect_equal(d$mcse_mean, d$sd / sqrt(d$n_draws), tolerance = 1e-12)
})

test_that("chain diagnostics name the parameters the fit names", {
  # gcol33/tulpa#714: mala / imh_laplace / pathfinder / vi store an UNNAMED
  # draws matrix beside a fully populated $param_names, so every reported row
  # read param1..paramN and "Rhat > 1.01: param3" could not be traced back to a
  # coefficient without counting columns by hand.
  skip_on_cran()
  fm <- suppressWarnings(tulpa(
    yp ~ x + (1 | g), data = dd, family = "poisson", mode = "mala",
    control = list(n_iter = 200L, warmup = 100L, seed = 1L)))
  expect_null(colnames(fm$draws))
  expect_identical(head(diagnostics(fm)$parameter, 2L),
                   head(fm$param_names, 2L))
  # and the 3-D accessor names the same axis
  expect_identical(dimnames(tulpa_draws_array(fm))[[3L]],
                   fm$param_names[seq_len(ncol(fm$draws))])
})

test_that("the resolver prefers real names and falls back positionally", {
  fit <- list(param_names = c("a", "b", "c"))
  expect_identical(tulpa:::.tulpa_draw_names(c("x", "y", "z"), fit, 3L),
                   c("x", "y", "z"))
  expect_identical(tulpa:::.tulpa_draw_names(NULL, fit, 3L), c("a", "b", "c"))
  expect_identical(tulpa:::.tulpa_draw_names(character(0), fit, 2L), c("a", "b"))
  expect_identical(tulpa:::.tulpa_draw_names(NULL, list(), 2L),
                   c("param1", "param2"))
})

test_that("glance() is one row whatever the tier", {
  # gcol33/tulpa#711: a nested fit's $converged is per outer grid cell, and a
  # vector there recycles every other column to its length, so
  # do.call(rbind, lapply(fits, glance)) produced a table whose row count
  # depended on each fit's grid size.
  skip_on_cran()
  W <- matrix(0, 20, 20)
  for (i in 1:19) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  fn <- tulpa(yp ~ x + spatial(s), data = dd, family = "poisson",
              spatial = list(type = "icar", adjacency = W),
              mode = "nested_laplace")
  expect_gt(length(fn[["converged"]]), 1L)
  expect_identical(nrow(glance(fn)), 1L)
  expect_true(is.logical(glance(fn)$converged) && length(glance(fn)$converged) == 1L)
})

test_that("logLik() says which quantity it is, and compare_models refuses a mix", {
  # gcol33/tulpa#712: the three tiers return a mean log POSTERIOR, a log
  # MARGINAL LIKELIHOOD and a log EVIDENCE under one name, and
  # compare_models(criterion = "loglik") ranked them in one table -- a model
  # choice made on an artefact of which tier fitted each.
  skip_on_cran()
  W <- matrix(0, 20, 20)
  for (i in 1:19) { W[i, i + 1L] <- 1; W[i + 1L, i] <- 1 }
  fn <- tulpa(yp ~ x + spatial(s), data = dd, family = "poisson",
              spatial = list(type = "icar", adjacency = W),
              mode = "nested_laplace")
  fm <- suppressWarnings(tulpa(
    yp ~ x + (1 | g), data = dd, family = "poisson", mode = "mala",
    control = list(n_iter = 200L, warmup = 100L, seed = 1L)))

  expect_identical(attr(logLik(fn), "quantity"), "log_evidence")
  expect_identical(attr(logLik(fm), "quantity"), "log_posterior_mean")

  expect_error(compare_models(nested = fn, mala = fm, criterion = "loglik"),
               "different quantities")
  # A set that agrees is ranked as before, with the quantity reported.
  out <- compare_models(a = fm, b = fm, criterion = "loglik")
  expect_identical(nrow(out), 2L)
  expect_identical(unique(out$quantity), "log_posterior_mean")
})

test_that("an AGQ fit reports its RE scale and says why it has no per-group effects", {
  # gcol33/tulpa#710: agq_fit ships a 0-ROW draws matrix, so every draws-based
  # VarCorr source returned NaN -> diag(NaN, 1) -> an error that
  # .print_re_section's tryCatch swallowed, printing the fit with no
  # Random-effects section while the estimated sigma_re sat on it; ranef()
  # returned an empty frame, indistinguishable from a model with no RE terms.
  skip_on_cran()
  fa <- tulpa(yp ~ x + (1 | g), data = dd, family = "poisson", mode = "agq")
  vc <- VarCorr(fa)
  expect_identical(nrow(vc), 1L)
  expect_identical(vc$source, "estimated")
  expect_equal(vc$sd, unname(fa$sigma_re), tolerance = 1e-10)

  expect_error(ranef(fa), "no per-group random effects")
  expect_silent(capture.output(print(fa)))

  # The 0-row tail is not RE draws.
  expect_null(tulpa:::.re_draws_mat(fa))
})

test_that("a non-NUTS backend refuses n_chains rather than ignoring it", {
  # gcol33/tulpa#704: only the NUTS branch reads n_chains; the same function
  # already hard-refuses mass_matrix and a warm start on the others, so silence
  # here was an omission -- a caller asking for four chains got one particle set.
  skip_on_cran()
  expect_error(
    tulpa:::tulpa_sample_glmm(
      y = yp, n_trials = NULL, X = cbind(1, x), family = "poisson",
      backend = "smc",
      control = list(n_chains = 4L, n_particles = 100L, seed = 1L)),
    "only read by the NUTS")
})

test_that("sample_glmm stamps its backend so the provenance gate can see it", {
  # gcol33/tulpa#693: tulpa_sample_glmm() closed without .finalize_fit(), so
  # the gate -- which treats an untagged fit as a chain -- computed Rhat and ESS
  # on SMC particles and VI draws.
  skip_on_cran()
  f <- tulpa:::tulpa_sample_glmm(
    y = yp, n_trials = NULL, X = cbind(1, x), family = "poisson",
    backend = "smc", control = list(n_particles = 200L, seed = 1L))
  expect_identical(f$backend, "smc")
  expect_identical(f$draws_kind, "iid")
  expect_false(tulpa:::.tulpa_is_chain(f))
})
