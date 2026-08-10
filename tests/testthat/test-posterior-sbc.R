# test-posterior-sbc.R
#
# Posterior SBC (helper-sbc.R sections 6, 8 and 9; gcol33/tulpa#339).
#
# Standard SBC draws the truth from the prior and reports calibration averaged
# over the whole generative distribution. Posterior SBC conditions the check on
# an observed data set: the truth comes from pi(theta | y_obs), the replicate
# from pi(y | theta'), and the augmented posterior pi(theta | y, y_obs) is what
# the PIT is taken against. That is the question a user actually asks, and the
# expensive ground truth the cheap per-fit reliability band should be scored on.
#
# THE ORDER OF THE SECTIONS IS THE ARGUMENT. A non-uniform posterior-SBC result
# is only a verdict on the ENGINE once the CONSTRUCTION is known to be right, so:
#   1. the pieces (seed split, quadrature marginal likelihood, prefit arms)
#   2. the construction, on the one model whose augmented posterior is closed
#      form -- it must be uniform, and it must FAIL when either premise of the
#      factorization is broken
#   3. only then, the engine on a fixture where the inner Laplace is an
#      approximation

# ---------------------------------------------------------------------------
# 1. The pieces
# ---------------------------------------------------------------------------

test_that("the driver hands the replicate its own seed", {
  # gcol33/tulpa#350. The split is the driver's, so a fixture that writes the
  # obvious `set.seed(seed)` in both callbacks is correct. Read the two seeds
  # off the callbacks themselves.
  got <- new.env(parent = emptyenv())
  m <- list(
    data_obs = list(), fit = function(data) list(),
    draw_theta = function(fit, seed) { got$draw <- c(got$draw, seed); c(a = 0) },
    simulate = function(theta, seed) { got$rep <- c(got$rep, seed); list() },
    pool = function(obs, rep) list(),
    arms = function(fit, data) list(only = list(a = sbc_normal(0, 1))))
  recov_posterior_sbc(m, n_seed = 4L, seed_off = 100L)
  expect_identical(got$draw, 101:104)
  expect_identical(got$rep, .sbc_rep_seed(101:104))
  # Distinct, and distinct from every other replicate's truth seed too, so no
  # pair of replicates shares a stream either.
  expect_length(intersect(got$draw, got$rep), 0L)
})

test_that("a fixture's replicate does not re-consume the truth draw's stream", {
  skip_on_cran()
  # The property the split exists for: with the seeds the driver derives, the
  # replicate's design, group effects and residuals do not come from the
  # uniforms that produced theta'. Drawing both from one stream would make the
  # truth determine the noise -- not p(y | theta'), and a non-uniform PIT with
  # nothing wrong in the inference.
  d_obs <- sbc_sim_gaussian(101L)
  cfg <- list(nr = 12L, spr = 4L, ntr = 1L, beta = c(-2.5, 0.8), su = 0.7,
              phi = 1.0, grid = SBC_RE_GRID)
  th <- c(beta1 = 0.4, beta2 = -1.1, sigma = 0.5)
  for (m in list(sbc_psbc_gaussian(d_obs),
                 sbc_psbc_re(sbc_sim_re(4242L, "binomial", cfg)))) {
    r <- m$simulate(th, .sbc_rep_seed(17L))
    # The head of the truth seed's stream is what `draw_theta` consumes. Nothing
    # in the replicate may equal it.
    set.seed(17L)
    head_of_stream <- stats::rnorm(1)
    expect_false(isTRUE(all.equal(r$X[1, 2], head_of_stream)))
  }

  # And the noise is a function of the seed alone, not of the parameter: at one
  # seed, two truths differing only in beta give replicates whose difference is
  # exactly the design term.
  m <- sbc_psbc_gaussian(d_obs)
  b1 <- c(beta1 = 0, beta2 = 0, sigma = 0.5)
  b2 <- c(beta1 = 3, beta2 = -2, sigma = 0.5)
  s <- .sbc_rep_seed(23L)
  r1 <- m$simulate(b1, s); r2 <- m$simulate(b2, s)
  expect_equal(r1$X, r2$X)
  expect_equal(r1$y - as.numeric(r1$X %*% c(0, 0)),
               r2$y - as.numeric(r2$X %*% c(3, -2)))
})

test_that("recov_posterior_sbc declares the experiment it ran", {
  m <- list(
    data_obs = list(), fit = function(data) list(),
    draw_theta = function(fit, seed) { set.seed(seed); c(a = stats::rnorm(1)) },
    simulate = function(theta, seed) list(),
    pool = function(obs, rep) list(),
    arms = function(fit, data) list(sharp = list(a = sbc_normal(0, 1)),
                                    blunt = list(a = sbc_normal(0, 3))))
  res <- recov_posterior_sbc(m, n_seed = 8L)
  expect_identical(attr(res, "truth"), "posterior_draw")
  # The CRPS is proper here, because the truth is drawn from the distribution
  # the forecast is a posterior update of -- the same reason it is proper under
  # "prior_draw" and improper under "fixed".
  expect_match(attr(res, "crps_role"), "^proper posterior score")
  cmp <- sbc_crps_compare(res, baseline = "sharp")
  expect_s3_class(cmp, "data.frame")
  # The truth really is N(0, 1) here, so the proper score prefers the arm that
  # says so and `blunt` must score worse.
  expect_gt(cmp$delta[cmp$arm == "blunt"], 0)

  bad <- recov_sbc(function(seed) list(theta = c(a = 0)),
                   function(d) list(only = list(a = sbc_normal(0, 1))),
                   n_seed = 3L, truth = "fixed")
  expect_error(sbc_crps_compare(bad, baseline = "only"), "prior-predictive")
})

test_that("recov_posterior_sbc requires every callback", {
  full <- list(data_obs = 1, fit = 1, draw_theta = 1, simulate = 1, pool = 1,
               arms = 1)
  for (nm in names(full)) {
    expect_error(recov_posterior_sbc(full[setdiff(names(full), nm)], n_seed = 1L),
                 nm, fixed = TRUE)
  }
})

test_that("the adaptive quadrature is EXACT on the gaussian closed form", {
  skip_on_cran()
  # The rank arm needs log p(y | beta, sigma) with the random effects integrated
  # out, and only the gaussian family has it in closed form -- so this is the one
  # place the quadrature can be held against an answer rather than against
  # itself. Recentred at the integrand's own mode the rule integrates a Gaussian
  # exactly, so TWO nodes already reproduce the closed form to machine
  # precision. A fixed rule laid on the prior does not: it stalls at 3.1e-03 by
  # 64 nodes once beta is a couple of units from its estimate, which is the
  # measurement the adaptive step exists for.
  d <- sbc_sim_gaussian(7L)
  d$family <- "gaussian"; d$ntr <- 1L
  for (par in list(c(-0.2, 0.7), c(1.5, -0.4), c(-3.0, 2.5))) {
    for (s in c(0.25, 0.6, 1.4, 3.0)) {
      exact <- sbc_loglik(d, par, s)
      lbl <- sprintf("beta = (%.1f, %.1f), sigma = %.2f", par[1], par[2], s)
      for (nq in c(2L, 4L, 32L)) {
        expect_lt(abs(sbc_loglik_re(d, par, s, n_quad = nq) - exact), 1e-11,
                  label = paste(lbl, "at", nq, "nodes"))
      }
    }
  }
})

test_that("the adaptive quadrature converges in the node count off the gaussian", {
  skip_on_cran()
  # No closed form here, so the check is convergence: the error against a
  # 128-node reference has to fall by orders of magnitude as nodes are added,
  # and the default node count has to be on the converged end of that.
  #
  # The tolerance is set by what the number is FOR. It feeds a rank of the
  # truth's log-likelihood among posterior draws, whose spread is of order one,
  # so a quadrature error four orders below that cannot move a rank. The
  # measured worst case at the default node count is 1.0e-05, on the poisson at
  # the top of the sigma grid.
  cfg <- list(nr = 12L, spr = 4L, ntr = 1L, beta = c(-2.5, 0.8), su = 0.7,
              phi = 1.0, grid = SBC_RE_GRID)
  for (fam in c("binomial", "poisson")) {
    d <- sbc_sim_re(4242L, fam, cfg)
    for (s in c(0.7, 2.0)) {
      ref <- sbc_loglik_re(d, cfg$beta, s, n_quad = 128L)
      err <- vapply(c(4L, 8L, 16L, 32L),
                    function(nq) abs(sbc_loglik_re(d, cfg$beta, s, n_quad = nq) - ref),
                    numeric(1))
      lbl <- sprintf("%s, sigma = %.2f", fam, s)
      expect_lt(err[4], 1e-4, label = paste(lbl, "at the default node count"))
      expect_lt(err[4], err[1] / 100, label = paste(lbl, "converging"))
    }
  }
})

test_that("the response law and its log density are the same parameterization", {
  skip_on_cran()
  # `sbc_draw_y()` simulates and `sbc_obs_loglik()` scores; a family whose phi
  # meant one thing in one and another in the other would pass every shape check
  # and silently mis-score. The mean of a large sample must sit at the density's
  # own mode-free first moment.
  eta <- rep(0.3, 20000L)
  for (fam in c("poisson", "binomial", "gaussian", "neg_binomial_2", "gamma")) {
    set.seed(11L)
    y <- sbc_draw_y(fam, eta, 10L, 3)
    mu <- switch(fam, poisson = exp(0.3), binomial = 10 * stats::plogis(0.3),
                 gaussian = 0.3, neg_binomial_2 = exp(0.3), gamma = exp(0.3))
    expect_lt(abs(mean(y) - mu) / mu, 0.05, label = fam)
    # The log density is finite on the support the draw produced.
    expect_true(all(is.finite(sbc_obs_loglik(fam, y, eta, 10L, 3))), label = fam)
  }
})

test_that("sbc_arms_gaussian reuses a supplied fit instead of re-solving", {
  skip_on_cran()
  d <- sbc_sim_gaussian(11L)
  f <- sbc_fit_nested(d)
  a1 <- sbc_arms_gaussian(d, phi_crossed = FALSE)
  a2 <- sbc_arms_gaussian(d, phi_crossed = FALSE, fit = f)
  expect_identical(names(a1), names(a2))
  expect_equal(a1$mixture$beta1$mu, a2$mixture$beta1$mu, tolerance = 0)
  expect_equal(a1$mixture$sigma$probs, a2$mixture$sigma$probs, tolerance = 0)
  expect_identical(a1$mixture$log_lik$rank, a2$mixture$log_lik$rank)
})

test_that("the reliability row reads the band off a fit with no draws", {
  skip_on_cran()
  # `diagnostics()` returns NULL on exactly these fits (gcol33/tulpa#348), so
  # the row composes the layers the way that reader does. When #347 / #348 land
  # this should keep reporting the same numbers.
  cfg <- list(nr = 12L, spr = 4L, ntr = 1L, beta = c(-2.5, 0.8), su = 0.7,
              phi = 1.0, grid = SBC_RE_GRID)
  d <- sbc_sim_re(4242L, "binomial", cfg)
  f <- sbc_fit_nested(d, control = list(diagnose_k = TRUE, diagnose_skew = TRUE))
  row <- sbc_reliability_row(f)
  expect_s3_class(row, "data.frame")
  expect_identical(nrow(row), 1L)
  expect_true(is.finite(row$pareto_k))
  expect_true(nzchar(row$reliability))
  # The composition is the one the shipped reader performs, not a parallel rule.
  expect_identical(row$pareto_k_band, .tulpa_khat_band(row$pareto_k))
})

# ---------------------------------------------------------------------------
# 2. The construction
#
# On the gaussian random-intercept fixture the augmented posterior is available
# in closed form, so the whole construction can be run with the EXACT posterior
# at both stages. Its PIT must then be uniform, and any departure is a defect in
# the construction -- the pooling, the seeding, the conditional independence of
# the replicate -- rather than in the engine.
#
# MEASURED, 1000 replicates around `sbc_sim_gaussian(101)`, `ks` the
# Kolmogorov-Smirnov departure from uniform and `p` the exact simultaneous
# p-value in the equal-local-levels family (raw read; the folded one agrees):
#
#   truth from the exact posterior      beta1 .0190 p .988   beta2 .0265 p .289
#                                       sigma .0212 p .773   loglik .0262 p .851
#   truth from the engine posterior     the same to four decimals -- a gaussian
#                                       log-likelihood is quadratic in eta, so
#                                       the inner Laplace IS the conditional
#                                       posterior and there is nothing to score
#   mis-scaled by 25% (wide / narrow)   ks .048 to .078, p <= 6.2e-08
#   replicate fit ALONE, not pooled     sigma .1037 p 2.7e-10, beta1 .0376
#                                       p 3.2e-04, loglik .0560 p .011
#   replicate re-observes y_obs's       beta1 .0946 p 0, sigma .1172 p 0,
#     regions, sharing their effects    loglik .1686 p 2.7e-14
#
# Both broken premises fail, and the SECOND one fails with the signature the
# theory predicts: sharing the group effects hits the intercept, the
# hyperparameter and the joint log-likelihood, while the slope -- which reads
# within-region contrasts that a shared per-region effect cancels out of --
# survives at p = .27. Dropping `y_obs` is the blunter failure of the two, worst
# on sigma (the quantity `y_obs` was most informative about) but not confined to
# it; the assertions below therefore pin sigma, which is the reliable signal,
# and do not claim the others are clean.
# ---------------------------------------------------------------------------

psbc_gauss_obs <- function() sbc_sim_gaussian(101L)

test_that("the exact posterior calibrates under the posterior-SBC construction", {
  skip_if_not_slow()
  n <- 400L
  # Through the EXPORTED front door (gcol33/tulpa#380), which is also where the
  # two premises below are checked: the fixture carries `group_ids`, so the
  # observable half of the fresh-groups premise is verified before the run.
  fit_sbc <- sbc("posterior",
                 model = sbc_psbc_gaussian(psbc_gauss_obs(), read = "exact"),
                 n_sim = n)
  expect_identical(fit_sbc$premises$pooling, "verified")
  expect_identical(fit_sbc$premises$fresh_groups,
                   "verified (disjoint group labels)")
  res <- fit_sbc$pit
  ref <- recov_posterior_sbc(sbc_psbc_gaussian(psbc_gauss_obs(),
                                               read = "exact"), n_seed = n)
  # `fit_obs` rides along on the driver's return and carries the observed fit's
  # wall-clock `$timing`, which is not part of the experiment.
  attr(res, "fit_obs") <- NULL
  attr(ref, "fit_obs") <- NULL
  expect_identical(res, ref)
  band <- sbc_ecdf_band(n, 0.999)
  u <- function(a, q) res$pit[res$arm == a & res$quantity == q]
  for (a in c("exact", "mixture")) {
    for (q in c("beta1", "beta2", "sigma", "log_lik")) {
      expect_true(sbc_ecdf_inside(u(a, q), band),
                  label = sprintf("%s / %s inside the 99.9%% band", a, q))
      expect_true(sbc_ecdf_inside(sbc_fold(u(a, q)), band),
                  label = sprintf("%s / %s folded inside the 99.9%% band", a, q))
    }
  }
  # On a gaussian response the inner Laplace is exact, so the engine's read and
  # the closed form are the same posterior and the construction cannot separate
  # them. That is why section 3 uses a different family.
  expect_lt(max(abs(u("exact", "beta1") - u("mixture", "beta1"))), 1e-4)

  # The harness can fail: a 25% dispersion error is symmetric, so the folded
  # read is the one that has to catch it.
  narrow_band <- sbc_ecdf_band(n, 0.95)
  for (a in c("wide", "narrow")) {
    for (q in c("beta1", "beta2")) {
      expect_false(sbc_ecdf_inside(sbc_fold(u(a, q)), narrow_band),
                   label = sprintf("%s / %s folded outside", a, q))
    }
  }
})

test_that("breaking either premise of the factorization breaks the calibration", {
  skip_if_not_slow()
  n <- 400L
  band <- sbc_ecdf_band(n, 0.95)
  d_obs <- psbc_gauss_obs()
  u <- function(res, q) res$pit[res$arm == "exact" & res$quantity == q]

  # (a) The documented pitfall: fit the replicate ALONE. Then the augmented
  # posterior is pi(theta | y_rep) rather than the sequential update
  # pi(theta | y_obs, y_rep), and the identity no longer holds.
  m <- sbc_psbc_gaussian(d_obs, read = "exact")
  m$pool <- function(obs, rep)
    list(y = rep$y, X = rep$X, region = as.integer(rep$region),
         N = length(rep$y), nr = rep$nr, spr = obs$spr, phi = obs$phi,
         grid = obs$grid)
  rA <- recov_posterior_sbc(m, n_seed = n)
  expect_false(sbc_ecdf_inside(u(rA, "sigma"), band))
  # The front door does not let this one through at all (gcol33/tulpa#380): the
  # pooled data set carries neither the observed values nor enough of them to
  # hold both, which is checkable before the first augmented fit.
  expect_error(sbc("posterior", model = m, n_sim = 4L), "AUGMENTED posterior")

  # (b) Conditional independence broken: the replicate re-observes the OBSERVED
  # regions, carrying their effects drawn from p(u | y_obs, theta) rather than
  # fresh ones. Then p(y_rep | theta, y_obs) != p(y_rep | theta).
  m2 <- sbc_psbc_gaussian(d_obs, read = "exact")
  m2$simulate <- function(theta, seed) {
    set.seed(seed)
    b <- c(theta[["beta1"]], theta[["beta2"]]); s <- theta[["sigma"]]
    mm <- d_obs$spr; ph <- d_obs$phi
    r <- d_obs$y - as.numeric(d_obs$X %*% b)
    sg <- as.numeric(rowsum(r, d_obs$region))
    pm <- (s^2 * sg) / (ph^2 + mm * s^2)
    pv <- s^2 * ph^2 / (ph^2 + mm * s^2)
    uu <- stats::rnorm(d_obs$nr, pm, sqrt(pv))
    X <- cbind(1, stats::rnorm(d_obs$N))
    list(y = as.numeric(X %*% b) + uu[d_obs$region] +
           stats::rnorm(d_obs$N, 0, ph),
         X = X, region = d_obs$region, nr = d_obs$nr)
  }
  rB <- recov_posterior_sbc(m2, n_seed = n)
  expect_false(sbc_ecdf_inside(u(rB, "sigma"), band))
  expect_false(sbc_ecdf_inside(u(rB, "log_lik"), band))
  # This one the front door CANNOT catch, and does not claim to. Its group
  # LABELS are fresh -- `pool()` still offsets them, so the pooled set carries
  # nr_obs + nr_rep of them -- and what is broken is where the effects at those
  # labels came from, which is inside the callback and not observable from
  # outside it. That is why `fresh_groups` reads "verified (disjoint group
  # labels)" and never "premise verified".
  expect_s3_class(sbc("posterior", model = m2, n_sim = 4L), "sbc")
})

# ---------------------------------------------------------------------------
# 3. The engine, where the inner Laplace is an approximation
# ---------------------------------------------------------------------------

test_that("the family-general fixture runs end to end and its arms are consistent", {
  skip_on_cran()
  cfg <- list(nr = 12L, spr = 4L, ntr = 1L, beta = c(-2.5, 0.8), su = 0.7,
              phi = 1.0, grid = SBC_RE_GRID)
  d_obs <- sbc_sim_re(4242L, "binomial", cfg)
  m <- sbc_psbc_re(d_obs, control = list(diagnose_k = FALSE,
                                         diagnose_skew = FALSE))
  res <- recov_posterior_sbc(m, n_seed = 6L)

  expect_setequal(unique(res$arm), c("mixture", "collapsed", "wide", "narrow"))
  expect_setequal(unique(res$quantity[res$arm == "mixture"]),
                  c("beta1", "beta2", "sigma", "log_lik"))
  expect_true(all(res$pit >= 0 & res$pit <= 1))
  # The truth is a draw from the observed fit's own posterior, so every sigma
  # truth sits on the grid the fits integrate over.
  th <- res$truth[res$arm == "mixture" & res$quantity == "sigma"]
  expect_true(all(vapply(th, function(x) any(abs(d_obs$grid - x) < 1e-10),
                         logical(1))))
  # The replicate doubles the design, so the augmented fit sees twice the data.
  expect_identical(attr(res, "n_seed"), 6L)
})
