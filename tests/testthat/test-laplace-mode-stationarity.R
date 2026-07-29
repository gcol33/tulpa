# The inner Laplace solve's ACHIEVED residual (LaplaceResult::score_max), the
# stopping rule that has to leave it small, and the exact-gradient gate that
# refuses to differentiate through a mode that did not settle. gcol33/tulpa#255.
#
# The Laplace log-marginal feels a mode error only quadratically, so a fit can
# look healthy at a residual that costs its theta-gradient five digits -- log|H|
# is not stationary in x, and the exact gradient's whole derivation is the
# statement that the joint score at x_hat is zero. These tests pin the residual
# itself rather than the convergence flag, which only reports that the stopping
# rule was met.

# One random-intercept dataset plus the pieces needed to rebuild its joint
# penalized score by hand. The kernel carries a weak fixed-effect ridge
# (data.sigma_beta = 100 in laplace_core.cpp, tau = DEFAULT_TAU_BETA = 1e-4);
# omitting it puts a spurious ~5e-05 floor under every residual measured here.
.stn_sim <- function(fam, phi, seed = 5L, G = 12L, per = 8L, theta = log(0.7),
                     sd_true = NULL) {
  set.seed(seed)
  n <- G * per
  grp <- rep(seq_len(G), each = per)
  X <- cbind(1, rnorm(n, 0, 0.4))
  b <- (if (is.null(sd_true)) sqrt(0.5) else sd_true) * rnorm(G)
  eta <- as.numeric(X %*% c(0.6, 0.5)) + b[grp]
  n_trials <- rep(1L, n)
  y <- .FAMILY_OPS[[fam]]$sample(eta, n_trials, phi)
  layout <- list(list(nc = 1L, full = FALSE, k = 1L,
                      n_groups = G, idx = grp, Z = NULL))
  L0 <- matrix(exp(theta), 1, 1)
  list(y = y, X = X, grp = grp, G = G, n = n, n_trials = n_trials, fam = fam,
       layout = layout, L0 = L0, re_list = .re_cov_build_re_list(list(L0), layout),
       sig2 = exp(2 * theta), tau_beta = 1 / 100^2)
}

.stn_fit <- function(d, phi, max_iter = 600L, joint = TRUE) {
  tulpa_laplace(y = d$y, n_trials = d$n_trials, X = d$X, re_list = d$re_list,
                family = d$fam, phi = phi, return_hessian = TRUE,
                return_joint_hessian = joint,
                max_iter = as.integer(max_iter), tol = 1e-12)
}

# d(log p(y|x) + log p(x))/dx at a latent x, assembled independently of the
# kernel: the design's image of the per-observation score, less the fixed-effect
# ridge and the random-effect penalty.
.stn_joint_score <- function(d, x, phi) {
  eta <- as.numeric(d$X %*% x[1:2]) + x[2 + d$grp]
  sc <- .FAMILY_OPS[[d$fam]]$score(eta, d$y, d$n_trials, phi)
  c(as.numeric(crossprod(d$X, sc)) - d$tau_beta * x[1:2],
    as.numeric(tapply(sc, d$grp, sum)) - x[2 + seq_len(d$G)] / d$sig2)
}

# Newton continued on the OBSERVED curvature from a returned mode: the stationary
# point the solve was aiming at, found with a step that cannot be the one under
# test. Its own residual bounds how much of any gap is the solve's rather than the
# fixture's conditioning.
.stn_true_mode <- function(d, x0, phi, n_step = 60L) {
  Z <- cbind(d$X, model.matrix(~ factor(d$grp) - 1))
  ii <- 2 + seq_len(d$G)
  obs_w <- function(x) {
    h <- 1e-6
    e <- as.numeric(Z %*% x)
    -(.FAMILY_OPS[[d$fam]]$score(e + h, d$y, d$n_trials, phi) -
      .FAMILY_OPS[[d$fam]]$score(e - h, d$y, d$n_trials, phi)) / (2 * h)
  }
  xs <- x0
  for (k in seq_len(n_step)) {
    H <- crossprod(Z, obs_w(xs) * Z) + diag(c(rep(d$tau_beta, 2), rep(0, d$G)))
    H[ii, ii] <- H[ii, ii] + diag(1 / d$sig2, d$G)
    xs <- xs + tryCatch(solve(H, .stn_joint_score(d, xs, phi)),
                        error = function(e) rep(0, length(xs)))
  }
  xs
}

.stn_grad_vs_fd <- function(d, phi) {
  f <- .stn_fit(d, phi)
  r <- .laplace_exact_re_grad(
    fit = f, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
    weights = NULL, re_list = d$re_list, layout = d$layout,
    L_list = list(d$L0), family = d$fam, phi = phi, want_phi = TRUE)
  if (is.null(r)) return(NULL)
  h <- 1e-5
  fd <- (.stn_fit(d, phi * exp(h), joint = FALSE)$log_marginal -
         .stn_fit(d, phi * exp(-h), joint = FALSE)$log_marginal) / (2 * h)
  list(analytic = r[length(r)], fd = fd,
       rel = abs(r[length(r)] - fd) / max(1, abs(fd)))
}


test_that("score_max is the joint score at the mode the solve returned", {
  for (fam in c("poisson", "neg_binomial_2", "neg_binomial_1")) {
    d <- .stn_sim(fam, 1.5)
    f <- .stn_fit(d, 1.5)
    expect_false(is.null(f$score_max))
    hand <- max(abs(.stn_joint_score(d, f$mode, 1.5)))
    # Both are sums over the same terms in different orders, so they agree to
    # the accumulation floor rather than exactly. Compared on an absolute scale:
    # at a settled mode the quantity IS the floor.
    expect_lt(abs(f$score_max - hand), 1e-8)
  }
})


test_that("a slowly converging inner solve is run to stationarity, not stopped at it", {
  # neg_binomial_1's Newton weight is the quasi-likelihood mu / (1 + phi), which
  # sits far below the observed curvature at large phi. The solve then converges
  # linearly at a rate that crosses 0.707 between phi = 3 and phi = 4 -- exactly
  # where a stall test on the decrement (which shrinks by rate^2) starts reading
  # slow progress as a conditioning limit. The step-based test does not: these
  # solves take 80+ iterations and arrive.
  d <- .stn_sim("neg_binomial_1", 0.5)
  for (phi in c(4, 6)) {
    f <- .stn_fit(d, phi)
    expect_true(f$converged)
    expect_gt(f$n_iter, 60L)             # it really does iterate
    expect_lt(f$score_max, 1e-8)         # and it really does arrive
    expect_lt(max(abs(.stn_joint_score(d, f$mode, phi))), 1e-8)
  }
})


test_that("the inner solve reaches stationarity at a large random-effect scale", {
  # Where the Newton weight is NOT the observed curvature the working weight can
  # understate the true curvature by more than a factor of two, and the undamped
  # iteration is then locally divergent: the iterate walks away from the mode
  # geometrically while every step is accepted, because each one loses only a few
  # parts in 1e9 of objective. These solves returned a joint score of 1e-04 with
  # converged = TRUE. neg_binomial_1 carries the quasi-likelihood weight
  # mu / (1 + phi) and inverse_gaussian the Fisher weight 1 / (phi mu); both are
  # covered, so this is not one family's quirk.
  skip_on_cran()
  cfg <- list(
    list("neg_binomial_1",   6,    6L,  5),
    list("neg_binomial_1",   6,   25L,  5),
    list("inverse_gaussian", 2,    6L,  5),
    list("neg_binomial_1",   6,    6L,  NULL),
    list("neg_binomial_1",   6,   25L,  NULL),
    list("neg_binomial_1",   0.5, 25L,  5)     # damping must not disturb this one
  )
  for (cc in cfg) {
    lbl <- sprintf("%s phi=%g G=%d", cc[[1]], cc[[2]], cc[[3]])
    d <- .stn_sim(cc[[1]], cc[[2]], G = cc[[3]], theta = log(5),
                  sd_true = cc[[4]])
    f <- .stn_fit(d, cc[[2]], max_iter = 5000L)
    expect_true(f$converged, info = lbl)
    expect_lt(f$score_max, 1e-8)
    # The residual is the solve's, not the fixture's: the mode agrees with the
    # observed-curvature stationary point to near machine precision.
    xs <- .stn_true_mode(d, f$mode, cc[[2]])
    expect_lt(max(abs(xs - f$mode)) / max(1, max(abs(xs))), 1e-9)
  }
})


test_that("the near-mode trust factor damps only a decrement that grew", {
  # Away from the mode the scale is 1, so those solves take the trial sequence
  # they always did; the gate is a decrement of 1e-6.
  expect_equal(tulpa:::cpp_newton_trust_probe(c(1, 1e-2, 1e-5, 1e-6)),
               rep(1, 4))

  # Below the gate a FALLING decrement leaves the scale at 1: a solve that never
  # overshoots is untouched, which is what keeps every converging fit unchanged.
  expect_equal(tulpa:::cpp_newton_trust_probe(10^-c(7, 8, 9, 10, 11)),
               rep(1, 5))

  # A decrement that grows is the overshoot signature. The first reading below
  # the gate only primes the state, so damping starts on the second.
  expect_equal(tulpa:::cpp_newton_trust_probe(c(1e-7, 1e-7 * 2, 1e-7 * 4)),
               c(1, 0.5, 0.25))

  # Recovery is by 1.5 per iteration and capped at the full step.
  expect_equal(
    tulpa:::cpp_newton_trust_probe(c(1e-7, 2e-7, 4e-7, 1e-7, 1e-8, 1e-9, 1e-10)),
    c(1, 0.5, 0.25, 0.375, 0.5625, 0.84375, 1))

  # Crossing back above the gate resets both the scale and the memory, so a solve
  # driven away from the mode and back does not carry stale damping.
  expect_equal(tulpa:::cpp_newton_trust_probe(c(1e-7, 2e-7, 1e-3, 1e-7, 2e-7)),
               c(1, 0.5, 1, 1, 0.5))

  # The floor is 2^-20, reached only by a sustained run of growth. The sequence
  # has to grow slowly enough to stay under the gate throughout -- crossing back
  # above it resets, which is the case above.
  probe <- tulpa:::cpp_newton_trust_probe(1e-30 * 1.1^(0:24))
  expect_equal(probe[25], 2^-20)
  expect_true(all(diff(probe[-1]) <= 0))
})


test_that("the exact dispersion gradient holds at a large neg_binomial_1 phi", {
  # The regression the issue reports: at phi = 4 the assembled dm/dlog_phi was
  # 1.0e-04 away from a central difference of tulpa's own log_marginal, against
  # 1e-10 at phi <= 3. Nothing was wrong with the derivative; the mode it was
  # evaluated at had not settled.
  d <- .stn_sim("neg_binomial_1", 0.5)
  for (phi in c(0.5, 1.5, 4, 6)) {
    r <- .stn_grad_vs_fd(d, phi)
    expect_false(is.null(r))
    expect_lt(r$rel, 1e-6)
  }
})


test_that("the exact gradient refuses a mode that stopped short of stationarity", {
  d <- .stn_sim("neg_binomial_1", 0.5)
  phi <- 6

  starved <- .stn_fit(d, phi, max_iter = 20L)
  expect_false(starved$converged)
  expect_gt(starved$score_max, 1e-3)
  expect_warning(
    g <- .laplace_exact_re_grad(
      fit = starved, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
      weights = NULL, re_list = d$re_list, layout = d$layout,
      L_list = list(d$L0), family = d$fam, phi = phi, want_phi = TRUE),
    class = "tulpa_unsettled_mode")
  expect_null(g)

  # The same call on a settled fit returns the gradient with no warning, so the
  # gate is reading the residual and not the family or the layout.
  settled <- .stn_fit(d, phi, max_iter = 600L)
  expect_true(settled$converged)
  g_ok <- .laplace_exact_re_grad(
    fit = settled, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
    weights = NULL, re_list = d$re_list, layout = d$layout,
    L_list = list(d$L0), family = d$fam, phi = phi, want_phi = TRUE)
  expect_true(is.numeric(g_ok))
  expect_length(g_ok, 2L)
})


test_that("the settled-mode gate passes a fit that does not report a residual", {
  # Every kernel reports score_max, but the gate must not block a fit assembled
  # without one (a hand-built list in a downstream package, an older cached fit).
  d <- .stn_sim("poisson", 1)
  f <- .stn_fit(d, 1)
  f$score_max <- NULL
  g <- .laplace_exact_re_grad(
    fit = f, y = d$y, X = d$X, n_trials = d$n_trials, offset = NULL,
    weights = NULL, re_list = d$re_list, layout = d$layout,
    L_list = list(d$L0), family = d$fam, phi = 1)
  expect_true(is.numeric(g))
})


test_that("the refusals are reported once per fit, not once per inner solve", {
  # .with_unsettled_report collapses the per-solve signal the outer optimizers
  # would otherwise emit at every trial theta.
  expect_warning(
    out <- .with_unsettled_report({
      for (i in 1:3) {
        warning(structure(class = c("tulpa_unsettled_mode", "warning", "condition"),
                          list(call = NULL, message = paste("solve", i))))
      }
      "done"
    }, "caller_fn"),
    "declined at 3 inner solves")
  expect_identical(out, "done")

  # Nothing signalled, nothing reported.
  expect_silent(.with_unsettled_report("quiet", "caller_fn"))
})


test_that("the refusal count is readable while the wrapped expression runs", {
  # This is what lets the gradient-driven outer optimizer notice it was walking
  # blind and hand over to the derivative-free path. optim() receives a declined
  # gradient as zeros and cannot tell that from a stationary point, so without
  # the live count it can stop at theta0 and report the start as the estimate.
  st <- .new_unsettled_state()
  expect_identical(st$n, 0L)
  seen <- NULL
  expect_warning(
    .with_unsettled_report({
      warning(structure(class = c("tulpa_unsettled_mode", "warning", "condition"),
                        list(call = NULL, message = "solve")))
      seen <- st$n          # read DURING the expression
      NULL
    }, "caller_fn", st),
    "fell back to the derivative-free optimizer")
  expect_identical(seen, 1L)
})
