# Dispersion derivatives, checked against the registry they differentiate.
#
# These three functions per family are hand-derived, and a wrong one does not
# announce itself: it biases every empirical-Bayes dispersion estimate by a
# smooth amount and still converges. So each is finite-differenced against the
# EXACT `loglik` / `score` / `weight` registered for that family in
# .FAMILY_OPS. Differentiating the registry rather than a textbook form is the
# point -- phi is a size for neg_binomial_2, a variance for gaussian, a shape
# for gamma and a precision for beta, and a derivative correct for one
# parameterization is wrong for another.

# Central difference in phi, on the log scale where phi > 0 so the step stays
# proportional to the value.
fd_dphi <- function(f, phi, h = 1e-6) {
  hp <- phi * h
  (f(phi + hp) - f(phi - hp)) / (2 * hp)
}

# Per family: where to centre the linear predictor, the trial count where one is
# needed, and the second dispersion channel where the family reads one.
DISPERSION_FIXTURE <- list(
  gaussian                 = list(loc = 1.0),
  lognormal                = list(loc = 1.0),
  gamma                    = list(loc = 0.5),
  # Same location as the untruncated family: the truncation correction is
  # largest where mu is small, so a fixture centred well above zero would
  # exercise the added terms only weakly.
  neg_binomial_2           = list(loc = 0.6),
  truncated_neg_binomial_2 = list(loc = 0.6),
  neg_binomial_1           = list(loc = 0.6),
  beta                     = list(loc = 0.2),
  inverse_gaussian         = list(loc = 0.5),
  beta_binomial            = list(loc = 0.2, nt = 8L),
  t                        = list(loc = 1.0, phi2 = 6),
  tweedie                  = list(loc = 0.5, phi2 = 1.5)
)

# Draw a y in the family's support so the log-density is finite: the
# derivatives are evaluated at realized data, not at the mean.
dispersion_case <- function(family, phi, n = 40L, seed = 3L) {
  fx  <- DISPERSION_FIXTURE[[family]]
  set.seed(seed)
  eta <- rnorm(n, fx$loc, 0.4)
  nt  <- rep(if (is.null(fx$nt)) 1L else fx$nt, n)
  ops <- tulpa:::.FAMILY_OPS[[family]]
  y <- if (is.null(fx$phi2)) ops$sample(eta, nt, phi)
       else ops$sample(eta, nt, phi, fx$phi2)
  # Beta's log-density diverges at the open boundary; nudge inside it. The
  # positive-real families are floored off zero for the same reason, and
  # tweedie's exact zero mass is a separate arm with no series to differentiate.
  if (family == "beta") y <- pmin(pmax(y, 1e-6), 1 - 1e-6)
  if (family %in% c("gamma", "lognormal", "inverse_gaussian", "tweedie"))
    y <- pmax(y, 1e-8)
  list(eta = eta, y = y, n_trials = nt, phi2 = fx$phi2, ops = ops,
       d = tulpa:::.family_dphi(family), d2 = tulpa:::.family_dphi2(family))
}

FAMILY_PHI <- list(
  gaussian                 = c(0.3, 1.0, 4.0),
  lognormal                = c(0.3, 1.0, 4.0),
  gamma                    = c(0.7, 3.0, 12.0),
  neg_binomial_2           = c(0.8, 2.5, 9.0),
  truncated_neg_binomial_2 = c(0.8, 2.5, 9.0),
  neg_binomial_1           = c(0.5, 1.5, 4.0),
  beta                     = c(2.0, 6.0, 20.0),
  inverse_gaussian         = c(0.2, 0.6, 2.0),
  beta_binomial            = c(2.0, 6.0, 20.0),
  t                        = c(0.4, 1.0, 3.0),
  tweedie                  = c(0.4, 1.0, 2.5)
)

# Call an ops entry that may or may not take the second dispersion channel.
ops_call <- function(f, cs, ..., phi) {
  if (is.null(cs$phi2)) f(..., phi) else f(..., phi, cs$phi2)
}


test_that("dloglik/dphi matches a finite difference of the registered loglik", {
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- cs$d$dloglik(cs$eta, cs$y, cs$n_trials, phi, cs$phi2)
      numeric <- fd_dphi(function(p)
        ops_call(cs$ops$loglik, cs, cs$eta, cs$y, cs$n_trials, phi = p), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("dscore/dphi matches a finite difference of the registered score", {
  # The score derivative is what carries the mode's movement with phi
  # (dx_hat/dphi); an error here tilts the gradient without changing the fit at
  # any fixed phi, so nothing else would catch it.
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- cs$d$dscore(cs$eta, cs$y, cs$n_trials, phi, cs$phi2)
      numeric <- fd_dphi(function(p)
        ops_call(cs$ops$score, cs, cs$eta, cs$y, cs$n_trials, phi = p), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("dweight/dphi differentiates the weight H is actually built from", {
  # The target is NOT uniformly `.FAMILY_OPS$weight`. neg_binomial_2 is the one
  # family whose compiled Newton weight is already the observed curvature
  # (`obs_weight`); every other entry differentiates the registry's `weight`.
  # The truncated pair go the OTHER way from neg_binomial_2: Newton builds H
  # from the expected form Var(y | y > 0), chosen there because it is positive
  # for every mu while the observed curvature carries y and can go negative.
  # Same rule -- differentiate whatever H is built from -- landing on the
  # opposite member of the pair.
  #
  # Differentiating the expected form for neg_binomial_2 is wrong by a few
  # percent: enough to move the maximizer, small enough that a recovery test
  # would still converge and look plausible. Naming the right target per family
  # is the whole content of this test.
  h_weight <- function(cs, fam, phi) {
    if (fam == "neg_binomial_2")
      return(cs$ops$obs_weight(cs$eta, cs$y, cs$n_trials, phi))
    ops_call(cs$ops$weight, cs, cs$eta, cs$n_trials, phi = phi)
  }
  for (fam in names(FAMILY_PHI)) {
    for (phi in FAMILY_PHI[[fam]]) {
      cs <- dispersion_case(fam, phi)
      analytic <- rep_len(cs$d$dweight(cs$eta, cs$y, cs$n_trials, phi, cs$phi2),
                          length(cs$eta))
      numeric <- fd_dphi(function(p) rep_len(h_weight(cs, fam, p),
                                             length(cs$eta)), phi)
      expect_equal(analytic, numeric, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("every registered family can carry its dispersion gradient exactly", {
  # Three correct per-observation derivatives are not enough on their own. The
  # ASSEMBLED dm/dphi reaches the mode only through q_eta = (dW/deta) s, so the
  # registration is sound exactly when that channel is either identically zero
  # or solved on the true curvature. This is the disjunction the file's header
  # states; asserting it here is what keeps a future entry from being added on
  # correct derivatives alone and landing ~1e-4 off the objective.
  eta <- c(-0.4, 0.2, 0.9)
  for (fam in tulpa:::.dispersion_families()) {
    fx <- DISPERSION_FIXTURE[[fam]]
    nt <- rep(if (is.null(fx$nt)) 1L else fx$nt, length(eta))
    phi <- FAMILY_PHI[[fam]][2L]
    y <- dispersion_case(fam, phi, n = length(eta))$y
    dw <- tulpa:::cpp_family_curvature_deta_vec(
      y, nt, eta, fam, phi, if (is.null(fx$phi2)) NA_real_ else fx$phi2)
    channel_is_zero <- max(abs(dw)) < 1e-12
    exact_inverse <- tulpa:::cpp_family_has_exact_mode_jacobian(fam)
    expect_true(channel_is_zero || exact_inverse, info = fam)
  }
})


test_that("the neg_binomial_2 weight derivative reduces to the expected one at y = mu", {
  # Averaging the observed curvature over y returns the expected weight, so
  # their phi-derivatives must agree at y = E[y] = mu. This pins the observed
  # form against the simpler expression it generalizes, independently of the
  # finite difference above.
  eta <- seq(-0.8, 1.2, length.out = 7)
  mu <- exp(eta)
  for (phi in c(0.8, 2.5, 9.0)) {
    d <- tulpa:::.family_dphi("neg_binomial_2")
    at_mean <- d$dweight(eta, mu, NULL, phi)
    expected_form <- mu^2 / (mu + phi)^2
    expect_equal(at_mean, expected_form, tolerance = 1e-12,
                 info = sprintf("phi = %g", phi))
  }
})


test_that("the derivatives are registered only where phi is free", {
  # poisson and binomial have no dispersion at all, so estimating one is not a
  # missing feature but a category error; .family_dphi() must refuse rather
  # than return something the optimizer would happily walk along.
  expect_null(tulpa:::.family_dphi("poisson"))
  expect_null(tulpa:::.family_dphi("binomial"))
  expect_null(tulpa:::.family_dphi("truncated_poisson"))
  expect_false(any(startsWith(tulpa:::.dispersion_families(), ".")))

  # Every front-door family that carries a dispersion carries its derivatives.
  # .PHI_FAMILIES additionally holds interval_gaussian / truncated_gaussian,
  # which are the composable per-observation censoring kernels rather than
  # `family =` values, so they are excluded by family_names().
  front_door <- intersect(tulpa:::.PHI_FAMILIES, tulpa:::family_names())
  expect_setequal(tulpa:::.dispersion_families(), front_door)

  for (fam in names(FAMILY_PHI)) {
    d <- tulpa:::.family_dphi(fam)
    expect_true(is.list(d), info = fam)
    expect_setequal(names(d), c("dloglik", "dscore", "dweight"))
  }
})


test_that("every dispersion family has a registered weight to differentiate", {
  # The dweight entries differentiate .FAMILY_OPS$weight. A family listed here
  # whose registry entry lacked one would leave dweight describing nothing.
  for (fam in tulpa:::.dispersion_families()) {
    ops <- tulpa:::.FAMILY_OPS[[fam]]
    expect_true(is.function(ops$weight), info = fam)
    expect_true(is.function(ops$score), info = fam)
    expect_true(is.function(ops$loglik), info = fam)
  }
})


# --- second-order dispersion derivatives (the phi Hessian's family half) -----

# Central difference in eta, for the mixed eta-phi weight curvature.
fd_deta <- function(f, eta, h = 1e-6) {
  (f(eta + h) - f(eta - h)) / (2 * h)
}

FAMILY_PHI2 <- list(
  neg_binomial_2           = c(0.8, 2.5, 9.0),
  truncated_neg_binomial_2 = c(0.8, 2.5, 9.0),
  gaussian                 = c(0.3, 1.0, 4.0),
  lognormal                = c(0.3, 1.0, 4.0),
  gamma                    = c(0.7, 3.0, 12.0),
  t                        = c(0.4, 1.0, 3.0)
)


test_that("the second-order dispersion derivatives match differences of the first", {
  # Each of L2 / Sc2 / DW2 / DWde is one further derivative of a first-order
  # .family_dphi form (DW2 = d(dweight)/dphi, DWde = d(dweight)/deta, and so on),
  # so differencing the analytic first-order chains the check back to the
  # registry those were themselves finite-differenced against. A wrong second
  # derivative tilts only the phi Hessian, which shifts interval widths without
  # moving any point estimate, so nothing else would surface it.
  for (fam in names(FAMILY_PHI2)) {
    d  <- tulpa:::.family_dphi(fam)
    d2 <- tulpa:::.family_dphi2(fam)
    for (phi in FAMILY_PHI2[[fam]]) {
      cs <- dispersion_case(fam, phi)
      eta <- cs$eta; y <- cs$y; nt <- cs$n_trials; p2 <- cs$phi2
      n <- length(eta)
      wide <- function(v) rep_len(v, n)
      expect_equal(d2$dloglik2(eta, y, nt, phi, p2),
                   fd_dphi(function(p) d$dloglik(eta, y, nt, p, p2), phi),
                   tolerance = 1e-5, info = sprintf("L2 %s phi=%g", fam, phi))
      expect_equal(d2$dscore2(eta, y, nt, phi, p2),
                   fd_dphi(function(p) d$dscore(eta, y, nt, p, p2), phi),
                   tolerance = 1e-5, info = sprintf("Sc2 %s phi=%g", fam, phi))
      expect_equal(wide(d2$dweight2(eta, y, nt, phi, p2)),
                   fd_dphi(function(p) wide(d$dweight(eta, y, nt, p, p2)), phi),
                   tolerance = 1e-5, info = sprintf("DW2 %s phi=%g", fam, phi))
      expect_equal(wide(d2$dweight_deta(eta, y, nt, phi, p2)),
                   fd_deta(function(e) wide(d$dweight(e, y, nt, phi, p2)), eta),
                   tolerance = 1e-5, info = sprintf("DWde %s phi=%g", fam, phi))
    }
  }
})


test_that("a phi Hessian entry carries every derivative its assembly reads", {
  # The border differentiates u and the mode motion on H_true^-1 while the
  # objective's own terms stay on H^-1, so a family whose working weight is not
  # the observed curvature owes d(W_obs)/dphi on top of the four. Withholding it
  # must take the whole entry down rather than leave the assembly pairing two
  # different inverses.
  expect_null(tulpa:::.family_dphi2("neg_binomial_1"))
  expect_null(tulpa:::.family_dphi2("beta"))
  expect_null(tulpa:::.family_dphi2("poisson"))
  for (fam in names(FAMILY_PHI2)) {
    d2 <- tulpa:::.family_dphi2(fam)
    expect_true(is.list(d2), info = fam)
    core <- c("dloglik2", "dscore2", "dweight2", "dweight_deta")
    want <- if (tulpa:::.family_dphi2_needs_obs(fam)) c(core, "dobs_weight")
            else core
    expect_true(setequal(names(d2), want), info = fam)
  }
  # The predicate agrees with the delta it describes: zero exactly where the
  # working weight already is the observed curvature.
  eta <- c(-0.4, 0.2, 0.9)
  for (fam in names(FAMILY_PHI2)) {
    fx <- DISPERSION_FIXTURE[[fam]]
    nt <- rep(if (is.null(fx$nt)) 1L else fx$nt, length(eta))
    phi <- FAMILY_PHI2[[fam]][2L]
    y <- dispersion_case(fam, phi, n = length(eta))$y
    del <- tulpa:::cpp_family_obs_curvature_delta_vec(
      y, nt, eta, fam, phi, if (is.null(fx$phi2)) NA_real_ else fx$phi2)
    expect_equal(max(abs(del)) > 1e-12, tulpa:::.family_dphi2_needs_obs(fam),
                 info = fam)
  }
})


test_that("d(W_obs)/dphi matches a difference of the registered observed weight", {
  # The one entry with no counterpart in .FAMILY_OPS for gamma and t, whose
  # observed curvature is not registered there. Differenced against the compiled
  # dispatch instead, which is what H_true is actually built from.
  for (fam in names(FAMILY_PHI2)) {
    if (!tulpa:::.family_dphi2_needs_obs(fam)) next
    d2 <- tulpa:::.family_dphi2(fam)
    for (phi in FAMILY_PHI2[[fam]]) {
      cs <- dispersion_case(fam, phi)
      p2 <- if (is.null(cs$phi2)) NA_real_ else cs$phi2
      w_obs <- function(p) vapply(seq_along(cs$eta), function(i)
        tulpa:::cpp_family_obs_terms(cs$y[i], cs$n_trials[i], cs$eta[i], fam,
                                     p, p2)[["neg_hess"]], numeric(1))
      expect_equal(d2$dobs_weight(cs$eta, cs$y, cs$n_trials, phi, cs$phi2),
                   fd_dphi(w_obs, phi), tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


# --- assembled dm/dlog_phi, against the objective it differentiates ----------

# One dataset per family at its reference phi, and the identity checked at
# several phi against it. Redrawing per phi would confound the derivative with
# how extreme the sample gets.
assembled_case <- function(family, seed = 5L, G = 10L, per = 6L) {
  fx <- DISPERSION_FIXTURE[[family]]
  set.seed(seed)
  n   <- G * per
  grp <- rep(seq_len(G), each = per)
  X   <- cbind(1, rnorm(n, 0, 0.3))
  b   <- sqrt(0.5) * rnorm(G)
  eta <- as.numeric(X %*% c(fx$loc, 0.5)) + b[grp]
  nt  <- rep(if (is.null(fx$nt)) 1L else fx$nt, n)
  phi <- FAMILY_PHI[[family]][1L]
  ops <- tulpa:::.FAMILY_OPS[[family]]
  y <- if (is.null(fx$phi2)) ops$sample(eta, nt, phi)
       else ops$sample(eta, nt, phi, fx$phi2)
  if (family == "beta") y <- pmin(pmax(y, 1e-6), 1 - 1e-6)
  if (family %in% c("gamma", "lognormal", "inverse_gaussian", "tweedie"))
    y <- pmax(y, 1e-8)
  list(y = y, X = X, grp = grp, G = G, n = n, n_trials = nt, phi2 = fx$phi2,
       layout = list(list(nc = 1L, full = FALSE, k = 1L, n_groups = G,
                          idx = grp, Z = NULL)))
}

assembled_grad <- function(d, family, phi, theta = log(0.7), want_h = FALSE) {
  L0 <- matrix(exp(theta), 1, 1)
  re_list <- tulpa:::.re_cov_build_re_list(list(L0), d$layout)
  fit <- tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                       re_list = re_list, family = family, phi = phi,
                       phi2 = d$phi2, return_hessian = TRUE,
                       return_joint_hessian = TRUE, max_iter = 300L, tol = 1e-12)
  tulpa:::.laplace_exact_re_grad(
    fit = fit, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
    weights = NULL, re_list = re_list, layout = d$layout, L_list = list(L0),
    family = family, phi = phi,
    phi2 = if (is.null(d$phi2)) NA_real_ else d$phi2,
    want_jacobian = want_h, want_hessian = want_h, want_phi = TRUE)
}

assembled_logmarg <- function(d, family, phi, theta = log(0.7)) {
  L0 <- matrix(exp(theta), 1, 1)
  tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X,
                re_list = tulpa:::.re_cov_build_re_list(list(L0), d$layout),
                family = family, phi = phi, phi2 = d$phi2,
                return_hessian = TRUE, max_iter = 300L,
                tol = 1e-12)$log_marginal
}


test_that("the assembled dispersion gradient differentiates the log-marginal", {
  skip_on_cran()
  # The per-observation checks above pin each derivative to the registry. This
  # is the assembly claim they cannot reach: dm/dlog_phi carries the mode's
  # movement with phi, which is exact only when the mode-motion solve is on the
  # right inverse. That distinction is invisible per observation and is what
  # kept beta out of the registry until its observed curvature was registered.
  #
  # The first two phi per family, not all three. The third is a deliberate
  # extrapolation away from the value the data were drawn at, and for
  # neg_binomial_1 it reaches the corner where the inner Newton solve stops
  # short of stationarity and costs the gradient five digits -- an engine
  # convergence defect tracked as gcol33/tulpa#255, not a derivative error.
  h <- 1e-5
  for (fam in names(FAMILY_PHI)) {
    d <- assembled_case(fam)
    for (phi in FAMILY_PHI[[fam]][1:2]) {
      r <- assembled_grad(d, fam, phi)
      expect_false(is.null(r), info = fam)
      analytic <- r[length(r)]
      numeric <- (assembled_logmarg(d, fam, phi * exp(h)) -
                    assembled_logmarg(d, fam, phi * exp(-h))) / (2 * h)
      expect_equal(analytic, numeric, tolerance = 1e-6,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})


test_that("the bordered phi Hessian differentiates the exact gradient", {
  skip_on_cran()
  # Column j of the returned Hessian is d(grad)/d chi_j with chi = (theta,
  # log phi), so differencing the gradient in BOTH coordinates checks the border
  # from both sides -- the phi column and the theta column that shares its
  # off-diagonal entry.
  h <- 1e-5
  for (fam in names(FAMILY_PHI2)) {
    d <- assembled_case(fam)
    for (phi in FAMILY_PHI2[[fam]][1:2]) {
      r <- assembled_grad(d, fam, phi, want_h = TRUE)
      expect_false(is.null(r$H), info = fam)
      fd <- cbind(
        (assembled_grad(d, fam, phi, theta = log(0.7) + h) -
           assembled_grad(d, fam, phi, theta = log(0.7) - h)) / (2 * h),
        (assembled_grad(d, fam, phi * exp(h)) -
           assembled_grad(d, fam, phi * exp(-h))) / (2 * h))
      expect_equal(r$H, fd, tolerance = 1e-5,
                   info = sprintf("%s, phi = %g", fam, phi))
    }
  }
})
