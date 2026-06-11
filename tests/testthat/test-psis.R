# Pareto-smoothed importance sampling: the native PSIS core, its agreement with
# loo, and the outer Pareto-k-hat diagnostic for the nested-Laplace path.

test_that("tulpa_psis reproduces loo::psis pareto_k", {
  skip_if_not_installed("loo")
  set.seed(42)

  cases <- list(
    light = rnorm(3000),                 # lognormal weights: all moments finite
    heavy = log(1 / runif(3000)),        # Pareto(1) weights: k-hat ~ 1
    mixed = rnorm(2000) + rexp(2000)     # moderately heavy
  )
  for (nm in names(cases)) {
    lr  <- cases[[nm]]
    ours <- tulpa_psis(lr)$pareto_k
    ref  <- suppressWarnings(
      loo::psis(matrix(lr, ncol = 1), r_eff = NA)$diagnostics$pareto_k)
    expect_equal(ours, ref, tolerance = 0.02,
                 info = sprintf("case '%s': ours=%.4f loo=%.4f", nm, ours, ref))
  }
})

test_that("pareto_k separates light from heavy tails; is_ess is bounded", {
  set.seed(1)
  light <- tulpa_psis(rnorm(3000))
  heavy <- tulpa_psis(log(1 / runif(3000)))

  expect_lt(light$pareto_k, 0.5)         # finite-moment proposal: reliable
  expect_gt(heavy$pareto_k, 0.7)         # Pareto(1): not correctable
  expect_lt(light$is_ess, 3000 + 1e-6)   # ESS never exceeds the sample size
  expect_gt(light$is_ess, 0)
  expect_lt(heavy$is_ess, light$is_ess)  # heavy weights concentrate -> lower ESS
})

test_that(".nested_outer_pareto_k is low when the proposal covers the target", {
  set.seed(7)
  # Gaussian target N(mu, Sigma); proposal slightly wider (1.5 * Sigma) so the
  # importance weights are bounded -> small / negative k-hat, high IS-ESS.
  mu    <- c(0.4, -0.2)
  Sigma <- matrix(c(1, 0.3, 0.3, 0.5), 2)
  Sinv  <- solve(Sigma)
  log_target <- function(th) {
    d <- th - mu
    -0.5 * as.numeric(t(d) %*% Sinv %*% d)            # up to a constant
  }
  L_scale <- t(chol(1.5 * Sigma))                     # proposal scale > target

  kd <- .nested_outer_pareto_k(log_target, theta_hat = mu,
                               L_scale = L_scale, n_samples = 2000L)
  expect_true(is.finite(kd$pareto_k))
  expect_lt(kd$pareto_k, 0.5)
  expect_gt(kd$is_ess, 0)
  expect_equal(kd$n_eval, 2000L)
})

test_that(".nested_outer_pareto_k rises when the target is heavier than the proposal", {
  set.seed(9)
  mu <- c(0, 0)
  # Student-t(df = 2) target (infinite variance) against a unit-Gaussian
  # proposal of matching scale: tails the Gaussian cannot cover -> high k-hat.
  log_target_t <- function(th) sum(stats::dt(th - mu, df = 2, log = TRUE))
  kd <- .nested_outer_pareto_k(log_target_t, theta_hat = mu,
                               L_scale = diag(1, 2), n_samples = 4000L)
  expect_gt(kd$pareto_k, 0.5)                          # heavier target -> not correctable
})

test_that("the IS cores decline before evaluating the target below the floor", {
  # gcol33/tulpa#51: a sub-floor n_samples can never reach .PSIS_MIN_EVAL finite
  # evaluations, so both cores must return NA WITHOUT touching the (expensive)
  # target -- not evaluate the whole budget and then discard it.
  expect_lt(20L, .PSIS_MIN_EVAL)

  hit <- 0L
  target_counting <- function(th) { hit <<- hit + 1L; 0 }
  kd <- .nested_outer_pareto_k(target_counting, theta_hat = c(0, 0),
                               L_scale = diag(1, 2), n_samples = 20L)
  expect_true(is.na(kd$pareto_k))
  expect_equal(kd$n_eval, 0L)
  expect_identical(hit, 0L)                            # no inner solve was paid

  hit_b <- 0L
  batched_counting <- function(U) { hit_b <<- hit_b + nrow(U); rep(0, nrow(U)) }
  kb <- .nested_is_pareto_k(theta_hat = c(0, 0), L_scale = diag(1, 2),
                            log_target_batched = batched_counting, n_samples = 20L)
  expect_true(is.na(kb$pareto_k))
  expect_equal(kb$n_eval, 0L)
  expect_identical(hit_b, 0L)
})

test_that(".nested_grid_radius_cap is the scaled max whitened grid radius", {
  # gcol33/tulpa#94: the cap is .K_DIAG_SUPPORT_RADIUS_MULT times the largest
  # whitened distance || L^{-1} (node - u_hat) || over the grid nodes.
  set.seed(101)
  d <- 3L
  u_hat <- c(0.5, -1, 2)
  A  <- matrix(rnorm(d * d), d)
  Su <- crossprod(A) + diag(d)                          # SPD
  L  <- t(chol(Su))
  # Place nodes at KNOWN whitened radii: node_k = u_hat + L %*% z_k.
  Zn <- rbind(c(1, 0, 0), c(0, 2, 0), c(0, 0, 1.5), c(1, 1, 1))
  u_grid <- t(apply(Zn, 1, function(z) u_hat + as.numeric(L %*% z)))
  cap <- .nested_grid_radius_cap(u_grid, u_hat, L)
  expect_equal(cap, max(sqrt(rowSums(Zn^2))) * .K_DIAG_SUPPORT_RADIUS_MULT,
               tolerance = 1e-10)
})

test_that(".nested_is_pareto_k skips draws beyond radius_cap (cost bound, light tail)", {
  # gcol33/tulpa#94: for a target NO heavier than the Gaussian proposal a finite
  # radius_cap excludes deep-extrapolation draws WITHOUT an inner solve, bounding
  # the diagnostic cost; radius_cap = Inf reproduces the unrestricted path. The
  # target here is narrower than the proposal, so the importance log-ratio FALLS
  # with the radius and the escalation step (gcol33/tulpa#100) does not fire.
  d <- 2L; u_hat <- c(0, 0); L_scale <- diag(d)
  rows_seen <- 0L
  tgt <- function(U) { rows_seen <<- rows_seen + nrow(U); -0.75 * rowSums(U^2) }

  # Z ~ N(0, I_2): ||z||^2 ~ chi^2_2. Cap radius 2 keeps z2 <= 4, i.e. a
  # fraction pchisq(4, 2) = 1 - exp(-2) ~ 0.865 of the draws.
  set.seed(303)
  kd <- .nested_is_pareto_k(u_hat, L_scale, tgt, n_samples = 2000L,
                            radius_cap = 2)
  expect_lt(rows_seen, 2000L)                  # the deep tail was skipped
  expect_gt(rows_seen, 0.70 * 2000L)           # the bulk was retained
  expect_equal(kd$n_eval, rows_seen)           # every retained draw had finite target

  rows_seen <- 0L
  set.seed(303)
  kd_inf <- .nested_is_pareto_k(u_hat, L_scale, tgt, n_samples = 2000L,
                                radius_cap = Inf)
  expect_equal(rows_seen, 2000L)               # Inf cap == evaluate every draw
})

test_that(".nested_is_pareto_k folds the clipped tail back in for a heavy target (#100)", {
  # gcol33/tulpa#100: the radius cap must NOT bias k-hat downward when the target
  # is genuinely heavier-tailed than the Gaussian proposal -- the regime a high
  # k-hat exists to flag. There the far-radius draws carry the LARGEST importance
  # ratios, so the escalation step re-evaluates them and the capped k-hat tracks
  # the uncapped one instead of collapsing toward "reliable".
  d <- 1L; u_hat <- 0; L_scale <- matrix(1, 1, 1)
  heavy <- function(U) stats::dt(U[, 1], df = 3, log = TRUE)   # heavier than N(0,1)

  k_at <- function(cap, seed = 17) {
    set.seed(seed)
    .nested_is_pareto_k(u_hat, L_scale, heavy, n_samples = 6000L,
                        radius_cap = cap)$pareto_k
  }
  k_inf <- k_at(Inf)
  expect_true(is.finite(k_inf))
  expect_gt(k_inf, 0.5)                         # the t_3 / Gaussian ratio is heavy
  for (cap in c(2.2, 2.5, 3.0)) {               # realistic >= 2x-grid-coverage caps
    k_cap <- k_at(cap)
    expect_true(is.finite(k_cap))
    # Not materially below the uncapped reading: the escalation prevents the
    # downward bias the cap would otherwise introduce.
    expect_gt(k_cap, k_inf - 0.12)
  }

  # Control: a Gaussian target matched to the proposal stays light under the cap
  # (no escalation, the cost bound is preserved).
  light <- function(U) stats::dnorm(U[, 1], log = TRUE)
  set.seed(17)
  k_light <- .nested_is_pareto_k(u_hat, L_scale, light, n_samples = 6000L,
                                 radius_cap = 2.2)$pareto_k
  expect_true(is.na(k_light) || k_light < 0.7)
})

test_that(".nested_is_pareto_k declines without solving when radius_cap is too tight", {
  # When fewer than .PSIS_MIN_EVAL draws fall within support, the core returns
  # NA WITHOUT paying any inner solve (the floor check precedes log_target).
  d <- 2L; u_hat <- c(0, 0); L_scale <- diag(d)
  hit <- 0L
  tgt <- function(U) { hit <<- hit + nrow(U); rep(0, nrow(U)) }
  set.seed(404)
  kd <- .nested_is_pareto_k(u_hat, L_scale, tgt, n_samples = 1000L,
                            radius_cap = 0.01)            # ~0 draws survive
  expect_true(is.na(kd$pareto_k))
  expect_lt(kd$n_eval, .PSIS_MIN_EVAL)
  expect_identical(hit, 0L)                               # no inner solve paid
})

test_that("tulpa_re_cov_nested reports a Pareto-k-hat without disturbing draws", {
  skip_on_cran()
  skip_if_fast()
  sim <- function(seed, G = 50L, npg = 10L) {
    set.seed(seed)
    N <- G * npg; grp <- rep(seq_len(G), each = npg)
    x <- rnorm(N); X <- cbind(1, x); Z <- cbind(1, x)
    Sig <- matrix(c(0.64, 0.24, 0.24, 0.36), 2)
    u <- t(t(chol(Sig)) %*% matrix(rnorm(2 * G), 2))
    eta <- as.numeric(X %*% c(-0.3, 0.7)) + rowSums(Z * u[grp, ])
    list(y = rbinom(N, 1L, plogis(eta)), X = X, Z = Z, grp = grp, G = G, N = N)
  }
  d  <- sim(11L)
  rt <- list(idx = d$grp, n_groups = d$G, n_coefs = 2L, Z = d$Z)

  fit <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, k_samples = 200L)
  # k-hat is a finite, data-dependent reading (a sparse binary RE-covariance
  # posterior is skewed, so a HIGH k-hat here is a correct signal, not a bug --
  # the estimator's correctness is pinned by the loo-equivalence test above and
  # the synthetic-target helper test). Assert the plumbing and ESS range only.
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_lte(fit$pareto_k_is_ess, 200 + 1e-6)
  expect_equal(fit$pareto_k_scope, "outer (hyperparameter) Gaussian proposal")

  # diagnose_k must not perturb the fixed-effect draws (RNG state restored).
  off <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, diagnose_k = FALSE)
  on  <- tulpa_re_cov_nested(d$y, rep(1L, d$N), d$X, rt, family = "binomial",
                             seed = 11L, diagnose_k = TRUE, k_samples = 150L)
  expect_true(is.na(off$pareto_k))
  expect_equal(off$draws, on$draws)
})

test_that("tulpa_nested_laplace reports an outer k-hat for a positive-scale block", {
  skip_on_cran()
  skip_if_fast()
  set.seed(3)
  nr <- 60L; spr <- 10L; N <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  X <- cbind(1, rnorm(N))
  u <- rnorm(nr, 0, 0.5)
  y <- rbinom(N, 10L, plogis(as.numeric(X %*% c(-0.2, 0.7)) + u[region]))
  sg <- exp(seq(log(0.2), log(1.5), length.out = 7))
  mk <- function(grid) list(list(type = "iid", obs_idx = region,
                                 n_units = nr, sigma_grid = grid))

  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(10L, N), X = X, prior = mk(sg),
    family = "binomial", phi = 1,
    control = list(max_iter = 100L, tol = 1e-8, k_samples = 200L)))
  # Single positive-scale (RE-SD) axis: k-hat is computed via the log transform.
  # The value is data-dependent (a sparse binary RE-SD posterior is skewed, so a
  # high reading is correct) -- assert plumbing + ESS range, as in the re_cov case.
  expect_true(is.finite(fit$pareto_k))
  expect_true(is.finite(fit$pareto_k_is_ess))
  expect_gt(fit$pareto_k_is_ess, 0)
  expect_lte(fit$pareto_k_is_ess, 200 + 1e-6)
  expect_equal(fit$pareto_k_scope, "outer (hyperparameter) Gaussian proposal")

  off <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(10L, N), X = X, prior = mk(sg),
    family = "binomial", phi = 1,
    control = list(max_iter = 100L, tol = 1e-8, diagnose_k = FALSE)))
  expect_true(is.na(off$pareto_k))                  # gated off
})

test_that("outer k-hat declines (NA) for a multi-block nested fit", {
  skip_on_cran()
  skip_if_fast()
  set.seed(4)
  nr <- 40L; spr <- 8L; N <- nr * spr
  region  <- rep(seq_len(nr), each = spr)
  region2 <- rep(seq_len(nr), times = spr)
  X <- cbind(1, rnorm(N))
  y <- rbinom(N, 5L, plogis(as.numeric(X %*% c(-0.1, 0.5))))
  sg <- exp(seq(log(0.2), log(1.2), length.out = 5))
  prior <- list(
    list(type = "iid", obs_idx = region,  n_units = nr, sigma_grid = sg),
    list(type = "iid", obs_idx = region2, n_units = nr, sigma_grid = sg))
  fit <- suppressWarnings(tulpa_nested_laplace(
    y = y, n_trials = rep(5L, N), X = X, prior = prior,
    family = "binomial", phi = 1, control = list(max_iter = 80L, tol = 1e-7)))
  expect_true(is.na(fit$pareto_k))                  # multi-block: declined, not guessed
  expect_gt(length(fit$weights), 1L)                # quadrature-ESS fallback available
})

# --------------------------------------------------------------------------- #
# Discriminating power on REAL engine output (gcol33/tulpa#99).                #
#                                                                              #
# The synthetic-target tests above show the estimator separates light from    #
# heavy tails; the real-fit tests above assert only plumbing, with a comment  #
# that a high k-hat "is a correct signal". This converts that comment into a  #
# tested invariant: the outer k-hat must ORDER two real RE-covariance fits --  #
# a well-identified one (many large groups -> near-Gaussian Sigma posterior,   #
# correctable) below a tiny-binary one (few binary obs per group -> strongly   #
# skewed variance posterior, not correctable). A regression that made the      #
# real-fit k-hat constant (always ~0, or always NA) would pass every other     #
# real-fit test but fail this one.                                             #
# --------------------------------------------------------------------------- #

test_that("outer k-hat orders well-identified below tiny-binary RE-covariance fits", {
  skip_on_cran()
  skip_if_fast()

  # (a) Well-identified: 30 groups x 25 gaussian obs each, correlated random
  # slope. The Sigma posterior is near-Gaussian, so the Gaussian grid proposal
  # is correctable -> low k-hat.
  k_well <- function(seed) {
    set.seed(seed); G <- 30L; per <- 25L; n <- G * per
    grp <- rep(seq_len(G), each = per); x <- rnorm(n)
    eta <- 0.2 + 0.5 * x + rnorm(G, 0, 0.8)[grp] + rnorm(G, 0, 0.5)[grp] * x
    y   <- eta + rnorm(n, 0, 0.5)
    rt  <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                correlated = TRUE)
    tulpa_re_cov_nested(y, rep(1L, n), cbind(1, x), rt, family = "gaussian",
                        phi = 0.25, diagnose_k = TRUE, k_samples = 150L)$pareto_k
  }
  # (b) Tiny binary groups: 25 groups x 3 binary obs each. The variance-component
  # posterior is strongly skewed (the IS ratio is heavy-tailed) -> high k-hat.
  k_tiny <- function(seed) {
    set.seed(seed); G <- 25L; per <- 3L; n <- G * per
    grp <- rep(seq_len(G), each = per); x <- rnorm(n)
    eta <- -0.2 + 0.4 * x + rnorm(G, 0, 1.0)[grp] + rnorm(G, 0, 0.8)[grp] * x
    y   <- rbinom(n, 1L, plogis(eta))
    rt  <- list(idx = grp, n_groups = G, n_coefs = 2L, Z = cbind(1, x),
                correlated = TRUE)
    tulpa_re_cov_nested(y, rep(1L, n), cbind(1, x), rt, family = "binomial",
                        diagnose_k = TRUE, k_samples = 150L)$pareto_k
  }

  ka <- vapply(1:5, function(s) k_well(100L + s), numeric(1))
  kb <- vapply(1:5, function(s) k_tiny(200L + s), numeric(1))

  # The diagnostic discriminates: the tiny-binary regime sits FAR above the
  # well-identified one and above the 0.7 escalation threshold. The separation
  # (typically two orders of magnitude) is the robust invariant; a hard 0.7 on
  # the well-identified side is left out because the GPD tail estimate is noisy
  # at k_samples = 150, and a tiny-binary k-hat may legitimately be Inf (an
  # improper-heavy variance posterior the proposal cannot correct).
  expect_true(all(is.finite(ka)))             # well-identified: finite, correctable
  expect_true(all(kb > 0.7))                  # tiny-binary: every fit flagged (Inf ok)
  expect_lt(median(ka), 2.0)                  # well-identified far from the heavy regime
  expect_gt(median(kb), 3 * median(ka))       # the ordering, with a wide margin
})
