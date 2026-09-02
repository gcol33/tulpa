# helper-sbc.R
#
# The SBC ENGINE FIXTURES. The scorer itself -- the predictive shapes, the PIT,
# the CRPS closed forms, the exact simultaneous band, and both drivers -- lives
# in `R/sbc.R` behind the exported `sbc()` front door (gcol33/tulpa#380), and
# these fixtures read THOSE functions. There is one implementation: a private
# copy here would be the duplicate scorer that promotion exists to prevent, so
# nothing in this file redefines any of them.
#
# What the package supplies, and what these fixtures call:
#   sbc_mixture / sbc_normal / sbc_discrete / sbc_rank / sbc_draws  (exported)
#   sbc_pit, sbc_fold, sbc_crps, sbc_crps_integral, sbc_sample, sbc_crps_mc
#   sbc_crossing_prob, sbc_ecdf_band, sbc_ecdf_inside, sbc_ecdf_dev,
#   sbc_ecdf_test
#   recov_sbc, recov_posterior_sbc, sbc_report, sbc_crps_compare
#   .sbc_with_seed, .sbc_rep_seed, .SBC_REP_SEED_OFFSET
# All of them are visible here because testthat runs in the package namespace;
# from a `dev_notes/` script, `devtools::load_all()` first and then source this
# file.
#
# LAYOUT. 7 the gaussian fixture whose posterior is closed form, 8 that fixture
# wired for posterior SBC, 9 the family-general fixture where the inner Laplace
# is an approximation.

# ---------------------------------------------------------------------------
# 7. The engine fixture
#
# A balanced gaussian random-intercept design fit through the nested-Laplace
# door, chosen because everything about it is available in closed form: the
# gaussian log-likelihood is exactly quadratic in eta, so the inner Laplace IS
# the conditional posterior, and integrating the latent field analytically gives
# the exact per-cell fixed-effect posterior and the exact cell weights. The
# engine's read is therefore scored against an independently written exact
# posterior rather than against another approximation.
#
# WHY THE PIT IS EXACTLY UNIFORM HERE DESPITE A FLAT PRIOR ON BETA. The nested
# door puts no prior on the fixed effects, and an improper prior cannot be drawn
# from -- so the truth is drawn from the prior on sigma alone and beta is held
# fixed. That is still an exact SBC experiment, by the structural argument for a
# location parameter under its Haar (flat) prior. Write bhat = bhat(y) for the
# generalized-least-squares estimate and r = y - X bhat for the residual. The
# posterior of t = beta - bhat given y is
#
#   p(t | y)  proportional to  integral p(sigma) f(-t | r, sigma) f_r(r; sigma) d sigma
#             =                integral p(sigma | r) f(-t | r, sigma) d sigma,
#
# where f( . | r, sigma) is the sampling density of bhat - beta_0 given r. Under
# sampling with sigma_0 ~ p(sigma), the conditional law of sigma_0 given r is
# that same p(sigma | r). So the posterior of t equals the sampling law of
# -(bhat - beta_0), and the PIT of beta_0 is exactly Uniform(0, 1) for every
# fixed beta_0 -- provided sigma_0 really is drawn from the engine's own prior
# on sigma, which for this grid is the discrete uniform over its cells. That is
# the one thing the simulator must not get wrong, and `auto_recenter = FALSE` is
# what keeps the fitted grid equal to the prior support.
# ---------------------------------------------------------------------------

SBC_GRID <- exp(seq(log(0.2), log(1.5), length.out = 7))
# `phi` in this harness is stated in the DOOR's own convention and handed over
# unconverted, which since `657f179` is the residual variance. 0.49 is the 0.7
# residual SD these fixtures have always simulated at, so the model is the one
# they always described (gcol33/tulpa#661).
#
# Nothing here restates that convention. Every place the harness needs the SD an
# `rnorm()` or a `dnorm()` wants, it asks the engine for it through
# `.phi_to_kernel()` -- the same call the doors make -- and the exact posterior
# derives its variance from the same place. So a further move of the convention
# moves the simulator with it, instead of silently rescaling every gaussian
# fixture the way it did twice (gcol33/tulpa#332 one way, #661 the other).
SBC_PHI  <- 0.49

# The residual SD and residual VARIANCE behind a `phi` written in the door's
# convention, whatever that convention currently is.
sbc_resid_sd  <- function(phi) .phi_to_kernel("gaussian", phi)
sbc_resid_var <- function(phi) sbc_resid_sd(phi)^2
SBC_BETA <- c(-0.2, 0.7)

# `nr` regions of `spr` observations each. The default is deliberately small:
# the sigma posterior then spreads over four to five of the seven cells, which
# is the regime where the fixed-effect mixture is farthest from the Gaussian
# matching its moments and the two reads of it are separable at all. Measured
# over 4000 prior-predictive replicates of the exact posterior, the intercept
# mixture's excess kurtosis is 1.36 here against 0.74 at nr = 10, spr = 5.
sbc_sim_gaussian <- function(seed, nr = 6L, spr = 4L, phi = SBC_PHI,
                             beta = SBC_BETA, grid = SBC_GRID) {
  set.seed(seed)
  sigma <- grid[sample.int(length(grid), 1L)]
  N <- nr * spr
  region <- rep(seq_len(nr), each = spr)
  x <- stats::rnorm(N)
  X <- cbind(1, x)
  u <- stats::rnorm(nr, 0, sigma)
  y <- as.numeric(X %*% beta) + u[region] +
       stats::rnorm(N, 0, sbc_resid_sd(phi))
  list(seed = seed, y = y, X = X, region = as.integer(region), N = N, nr = nr,
       spr = spr, phi = phi, grid = grid,
       theta = c(beta1 = beta[1], beta2 = beta[2], sigma = sigma))
}

# Exact per-cell quantities for the balanced design. V = sigma^2 Z Z' + phi I is
# block diagonal with blocks phi I_m + sigma^2 J_m, so V^-1 = a I - b J per block
# and every quadratic form is a group sum. `pv` is the residual VARIANCE behind
# whatever convention `phi` is written in -- the same quantity `sigma^2` is on
# the other term.
.sbc_exact_cell <- function(d, sigma, phi = d$phi) {
  m <- d$spr
  pv <- sbc_resid_var(phi)
  a <- 1 / pv
  b <- sigma^2 / (pv * (pv + m * sigma^2))
  X <- d$X; y <- d$y
  Xs <- rowsum(X, d$region)
  ys <- as.numeric(rowsum(y, d$region))
  XtVX <- a * crossprod(X) - b * crossprod(Xs)
  XtVy <- a * as.numeric(crossprod(X, y)) - b * as.numeric(crossprod(Xs, ys))
  ytVy <- a * sum(y^2) - b * sum(ys^2)
  Vb <- solve(XtVX)
  bh <- as.numeric(Vb %*% XtVy)
  G <- nrow(Xs); N <- length(y); p <- ncol(X)
  logdetV <- G * ((m - 1) * log(pv) + log(pv + m * sigma^2))
  # The marginal likelihood with the flat beta integrated out.
  lml <- -0.5 * logdetV - 0.5 * as.numeric(determinant(XtVX, TRUE)$modulus) -
    0.5 * (ytVy - sum(XtVy * bh)) - 0.5 * (N - p) * log(2 * pi)
  list(beta = bh, Vb = Vb, log_marg = lml, logdetV = logdetV, a = a, b = b)
}

# The exact posterior: a Gaussian mixture over the grid for beta, and the exact
# cell weights for sigma.
sbc_exact_post <- function(d, phi = d$phi) {
  cs <- lapply(d$grid, function(s) .sbc_exact_cell(d, s, phi))
  lm <- vapply(cs, function(z) z$log_marg, numeric(1))
  w <- exp(lm - max(lm)); w <- w / sum(w)
  list(mu = t(vapply(cs, function(z) z$beta, numeric(ncol(d$X)))),
       var = t(vapply(cs, function(z) diag(z$Vb), numeric(ncol(d$X)))),
       cov = lapply(cs, function(z) z$Vb),
       grid = d$grid, w = w, log_marg = lm)
}

# log p(y | beta, sigma) with the random effects integrated out, from the same
# block structure. This is what the joint log-likelihood rank ranks.
sbc_loglik <- function(d, beta, sigma, phi = d$phi) {
  z <- .sbc_exact_cell(d, sigma, phi)
  r <- d$y - as.numeric(d$X %*% beta)
  rs <- as.numeric(rowsum(r, d$region))
  q <- z$a * sum(r^2) - z$b * sum(rs^2)
  -0.5 * z$logdetV - 0.5 * q - 0.5 * length(d$y) * log(2 * pi)
}

# The door both fixtures fit through. `family` and `n_trials` default to the
# section-7 gaussian design and are read off `d` when it carries them, so the
# family-general fixture of section 9 needs no second door call.
#
# `auto_recenter = FALSE` is required, not a speed knob. Section 7's uniformity
# argument needs the fitted grid to equal the prior support; section 8's needs
# the sigma support at the truth-draw stage to be the one the augmented fit
# reports on. A recentred grid breaks both.
sbc_fit_nested <- function(d, phi = d$phi,
                           family = if (is.null(d$family)) "gaussian" else d$family,
                           ntr = if (is.null(d$ntr)) 1L else d$ntr,
                           control = NULL) {
  ctl <- list(max_iter = 200L, tol = 1e-10, n_threads = 1L,
              keep_grid_hessians = TRUE, diagnose_k = FALSE,
              diagnose_skew = FALSE, auto_recenter = FALSE)
  if (length(control)) ctl <- utils::modifyList(ctl, control)
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(ntr, d$N), X = d$X,
    prior = list(list(type = "iid", obs_idx = d$region,
                      n_units = d$nr, sigma_grid = d$grid)),
    family = family, phi = phi, control = ctl))
}

# The joint log-likelihood rank: draw `n_ref` (beta, sigma) pairs from the
# mixture an arm describes -- a cell by its weight, then that cell's Gaussian
# for beta -- and rank the truth's log-likelihood among them. It reads the whole
# data set at once, so it catches an approximation that gets each coefficient's
# marginal right while getting their joint dependence, or the hyperparameter,
# wrong, which per-coefficient PITs cannot see. The reference draws come from a
# stream pinned to the data seed, so the rank is reproducible.
#
# `ll_fn(d, beta, sigma, phi)` is the marginal log-likelihood the rank is formed
# on. It defaults to the closed form this fixture has; the family-general
# fixture of section 9 passes its quadrature counterpart, so the rank arm is
# available for every family rather than only the one with a closed form.
sbc_loglik_rank <- function(d, mu, cov, w, phi = d$phi, n_ref = 200L,
                            seed = d$seed, ll_fn = sbc_loglik,
                            grid = d$grid) {
  stopifnot(length(grid) == length(w), length(cov) == length(w),
            nrow(mu) == length(w))
  ll <- .sbc_with_seed(seed + 883L, {
    k <- sample.int(length(w), n_ref, replace = TRUE, prob = w)
    vapply(seq_len(n_ref), function(i) {
      L <- chol(cov[[k[i]]])
      b <- mu[k[i], ] + as.numeric(crossprod(L, stats::rnorm(ncol(mu))))
      ll_fn(d, b, grid[k[i]], phi)
    }, numeric(1))
  })
  ll0 <- ll_fn(d, unname(d$theta[c("beta1", "beta2")]),
               unname(d$theta[["sigma"]]), phi)
  sbc_rank(sum(ll < ll0), n_ref)
}

# Every arm the acceptance measurement scores. All but the last come off ONE
# solve per seed, plus the independently computed exact posterior:
#   exact        the analytic mixture -- the reference the PIT must be uniform on
#   mixture      the engine's retained per-cell components (gcol33/tulpa#336)
#   collapsed    the same two moments as one Gaussian (the pre-#336 read)
#   wide         collapsed with its SD multiplied by `bad_factor`
#   narrow       collapsed with its SD divided by `bad_factor`
#   phi_crossed  a SECOND solve at sqrt(phi) where the door reads the residual
#                VARIANCE -- the gcol33/tulpa#332 generator / inference
#                convention crossing, restated on the door's own axis after
#                657f179 moved it (gcol33/tulpa#661)
# `wide`, `narrow` and `phi_crossed` are the known-bad controls: a calibration
# harness that cannot fail is worthless, so a deliberately mis-scaled posterior
# has to land outside the band.
# `fit` accepts a solve the caller already has -- the posterior-SBC driver hands
# its augmented fit straight in, so the whole arm set still costs one solve per
# seed there rather than two.
sbc_arms_gaussian <- function(d, bad_factor = 1.25, n_ref = 200L,
                              phi_crossed = TRUE, fit = NULL) {
  f <- if (is.null(fit)) sbc_fit_nested(d) else fit
  P <- sbc_engine_post(f)
  E <- sbc_exact_post(d)
  mk <- function(p, phi = d$phi) {
    a <- list(beta1 = sbc_mixture(p$mu[, 1], p$var[, 1], p$w),
              beta2 = sbc_mixture(p$mu[, 2], p$var[, 2], p$w),
              sigma = sbc_discrete(p$grid, p$w))
    a$log_lik <- sbc_loglik_rank(d, p$mu, p$cov, p$w, phi = phi, n_ref = n_ref,
                                 grid = p$grid)
    a
  }
  se <- sqrt(pmax(diag(P$vcov), 0))
  coll <- function(mult) list(
    beta1 = sbc_normal(P$mean[1], mult * se[1]),
    beta2 = sbc_normal(P$mean[2], mult * se[2]))
  arms <- list(
    exact     = mk(E),
    mixture   = mk(P),
    collapsed = coll(1),
    wide      = coll(bad_factor),
    narrow    = coll(1 / bad_factor))
  if (phi_crossed) {
    # The broken control: hand the door the OTHER convention's number, which is
    # the kernel's, which is what `.phi_to_kernel()` returns.
    crossed <- .phi_to_kernel("gaussian", d$phi)
    arms$phi_crossed <- mk(sbc_engine_post(sbc_fit_nested(d, phi = crossed)),
                           phi = crossed)
  }
  arms
}

# The engine's posterior in the SAME shape `sbc_exact_post()` returns -- per-cell
# means, per-cell marginal variances, per-cell covariances and weights -- so a
# joint draw, a log-likelihood rank and an arm builder read either producer
# without knowing which one they got. `mean` / `vcov` carry the collapsed
# moments alongside, which is the one thing the exact side has no need of.
#
# THE PER-CELL COVARIANCES ARE TAKEN OVER THE SAME RETAINED CELLS `mu` / `var` /
# `w` are. `.nested_fixed_moments()` renormalizes over the cells that kept a
# block (gcol33/tulpa#342), so on a grid that dropped one, indexing
# `fit$grid_hessians` by the position of a weight would silently pair each
# component's mean with a different cell's covariance -- a wrong answer that
# every length and shape check passes. The predicate below is that function's.
sbc_engine_post <- function(fit) {
  mom <- .nested_fixed_moments(fit)
  if (is.null(mom)) stop("the fit retained no fixed-effect grid blocks")
  H <- fit$grid_hessians
  M <- fit$grid_modes
  w <- fit$weights / sum(fit$weights)
  keep <- which(is.finite(w) & w > 0 &
                  !vapply(H, is.null, logical(1)) &
                  !vapply(M, is.null, logical(1)))
  stopifnot(length(keep) == length(mom$w), length(keep) == nrow(mom$mu))
  list(mu = mom$mu, var = mom$var, w = mom$w,
       cov = lapply(keep, function(k) solve(H[[k]])),
       grid = as.numeric(fit$theta_grid)[keep],
       mean = mom$mean, vcov = mom$cov, mass = mom$mass)
}

# ---------------------------------------------------------------------------
# 8. The posterior-SBC fixture
#
# The same balanced gaussian random-intercept design as section 7, wired into
# `recov_posterior_sbc()`. Two things make it the right fixture to validate the
# construction on, and neither is available on a model whose posterior is only
# approximable:
#
#   * p(theta | y_obs, y_rep) is available in CLOSED FORM, so `read = "exact"`
#     runs the whole construction with the exact posterior at both stages. Its
#     `exact` arm must be uniform to within the simultaneous band, and any
#     departure is a defect in the CONSTRUCTION -- the seed split, the pooling,
#     the conditional independence of the replicate -- not in the engine. That
#     is the arbiter #339's cross-tabulation needs before any engine verdict it
#     produces can be believed.
#   * the exact augmented posterior really is the sequential update. With a flat
#     beta prior and the discrete sigma grid,
#     p(theta | y_obs, y_rep) proportional to p(theta | y_obs) p(y_rep | theta),
#     so refitting the POOLED data set is the paper's eq. (3) exactly, and the
#     improper prior is no obstacle because conditioning on y_obs has already
#     made the updating distribution proper.
#
# `read` selects which posterior the truth is drawn from, and therefore which
# arm the uniformity verdict belongs to: under `"exact"` it is the `exact` arm,
# under `"engine"` -- the paper's own choice for a real algorithm, and what #339
# measures -- it is `mixture`. The other arm is still reported and is still
# informative, but it mixes the two stages (approximate truth against exact
# update, or the reverse) and is not the calibration read of either.
#
# THE REPLICATE IS DRAWN ON FRESH REGIONS. sigma governs a per-region effect
# that theta does not carry, so a replicate observing the regions y_obs already
# saw would be dependent on y_obs given theta and would break the factorization.
# Fresh regions are what makes p(y_rep | theta', y_obs) = p(y_rep | theta').
# ---------------------------------------------------------------------------

sbc_psbc_gaussian <- function(d_obs, read = c("exact", "engine"),
                              nr_rep = d_obs$nr, bad_factor = 1.25,
                              n_ref = 200L, phi_crossed = FALSE) {
  read <- match.arg(read)
  P_exact <- sbc_exact_post(d_obs)

  list(
    data_obs = d_obs,
    fit = function(data) sbc_fit_nested(data),
    draw_theta = function(fit, seed) {
      P <- if (identical(read, "exact")) P_exact else sbc_engine_post(fit)
      set.seed(seed)
      k <- sample.int(length(P$w), 1L, prob = P$w)
      L <- chol(P$cov[[k]])
      b <- P$mu[k, ] + as.numeric(crossprod(L, stats::rnorm(ncol(P$mu))))
      c(beta1 = b[1], beta2 = b[2], sigma = P$grid[k])
    },
    # The driver hands this callback its own seed, decorrelated from the one
    # `draw_theta` consumed, so `set.seed(seed)` is all this has to do.
    simulate = function(theta, seed) {
      set.seed(seed)
      N <- nr_rep * d_obs$spr
      region <- rep(seq_len(nr_rep), each = d_obs$spr)
      X <- cbind(1, stats::rnorm(N))
      u <- stats::rnorm(nr_rep, 0, theta[["sigma"]])
      y <- as.numeric(X %*% c(theta[["beta1"]], theta[["beta2"]])) +
        u[region] + stats::rnorm(N, 0, sbc_resid_sd(d_obs$phi))
      list(y = y, X = X, region = region, nr = nr_rep)
    },
    pool = function(obs, rep) {
      list(y = c(obs$y, rep$y), X = rbind(obs$X, rep$X),
           region = as.integer(c(obs$region, rep$region + obs$nr)),
           N = obs$N + length(rep$y), nr = obs$nr + rep$nr, spr = obs$spr,
           phi = obs$phi, grid = obs$grid)
    },
    arms = function(fit, data) sbc_arms_gaussian(data, bad_factor = bad_factor,
                                                 n_ref = n_ref,
                                                 phi_crossed = phi_crossed,
                                                 fit = fit),
    # The grouping `sbc()` checks the observable half of the fresh-groups
    # premise on: `pool()` offsets the replicate's regions past the observed
    # ones, so the pooled set carries nr_obs + nr_rep distinct labels.
    group_ids = function(data) data$region)
}

# ---------------------------------------------------------------------------
# 9. A family-general posterior-SBC fixture
#
# Section 8 validates the CONSTRUCTION on the one model whose augmented
# posterior is available in closed form. It cannot produce a verdict on the
# ENGINE, because a gaussian log-likelihood is quadratic in eta: the inner
# Laplace IS the conditional posterior there, the engine's read equals the exact
# one to 7e-06 in the PIT, and there is no approximation left to score. This
# section supplies the fixture that has one -- the same IID random-intercept
# design under any built-in family, where the inner Gaussian is an
# approximation, `gamma_3` and the two Pareto k-hats are non-trivial, and the
# shipped reliability band therefore has something to say. That cross-tabulation
# is what gcol33/tulpa#339 asks for.
#
# THE MARGINAL LIKELIHOOD IS QUADRATURE HERE. The joint log-likelihood rank
# needs log p(y | beta, sigma) with the random effects integrated out, and only
# the gaussian family gives it in closed form. `sbc_loglik_re()` is a per-group
# Gauss-Hermite sum instead, whose nodes come from the Golub-Welsch
# eigendecomposition of the probabilists' Hermite recurrence -- twelve lines of
# base R rather than a dependency, and checked against section 7's closed form
# on the gaussian family, which is the one place both are available.
# ---------------------------------------------------------------------------

# Gauss-Hermite nodes and weights for integral f(u) N(u; 0, 1) du. The weights
# sum to 1, so the rule is a discrete probability measure and `sum(w * f(x))` is
# the expectation directly.
.sbc_gh <- local({
  cache <- new.env(parent = emptyenv())
  function(n) {
    key <- as.character(n)
    if (is.null(cache[[key]])) {
      i <- seq_len(n - 1L)
      J <- matrix(0, n, n)
      J[cbind(i, i + 1L)] <- sqrt(i)
      J[cbind(i + 1L, i)] <- sqrt(i)
      e <- eigen(J, symmetric = TRUE)
      o <- order(e$values)
      cache[[key]] <- list(x = e$values[o], w = (e$vectors[1L, o])^2)
    }
    cache[[key]]
  }
})

# The response law and its log density, as one pair so a family can never be
# simulated from one parameterization and scored under another. The recovery
# sweep's `recov_draw_y()` delegates to the draw side, so there is one
# definition of what each family's `phi` means across both harnesses.
sbc_draw_y <- function(family, eta, ntr, phi) {
  N <- length(eta)
  switch(family,
    poisson        = stats::rpois(N, exp(eta)),
    binomial       = stats::rbinom(N, ntr, stats::plogis(eta)),
    gaussian       = eta + stats::rnorm(N, 0, sbc_resid_sd(phi)),
    neg_binomial_2 = stats::rnbinom(N, mu = exp(eta), size = phi),
    gamma          = stats::rgamma(N, shape = phi, rate = phi / exp(eta)),
    beta           = {
      mu <- stats::plogis(eta)
      pmin(pmax(stats::rbeta(N, mu * phi, (1 - mu) * phi), 1e-4), 1 - 1e-4)
    },
    stop("unhandled family ", family))
}

sbc_obs_loglik <- function(family, y, eta, ntr, phi) {
  switch(family,
    poisson        = stats::dpois(y, exp(eta), log = TRUE),
    binomial       = stats::dbinom(y, ntr, stats::plogis(eta), log = TRUE),
    gaussian       = stats::dnorm(y, eta, sbc_resid_sd(phi), log = TRUE),
    neg_binomial_2 = stats::dnbinom(y, mu = exp(eta), size = phi, log = TRUE),
    gamma          = stats::dgamma(y, shape = phi, rate = phi / exp(eta),
                                   log = TRUE),
    beta           = {
      mu <- stats::plogis(eta)
      stats::dbeta(y, mu * phi, (1 - mu) * phi, log = TRUE)
    },
    stop("unhandled family ", family))
}

# log p(y | beta, sigma) with the per-region effect integrated out, by ADAPTIVE
# Gauss-Hermite: the rule is recentred and rescaled at each region's own
# integrand mode rather than laid on the prior.
#
# THE ADAPTIVE STEP IS NOT AN OPTIMIZATION. A fixed rule scaled by sigma places
# its nodes by the PRIOR, and the integrand is the likelihood times that prior --
# with `spr` observations per region the likelihood bump has width phi/sqrt(spr),
# which at the fixture's own settings is narrower than the node spacing and sits
# off centre whenever beta is away from its estimate. Measured against the
# gaussian closed form, the fixed rule stalls at 3.1e-03 by 64 nodes on a beta
# two units from the fit. The adaptive rule is EXACT there at any node count from
# 2 up -- recentred, the integrand is Gaussian and the rule integrates it
# exactly -- and off the gaussian it converges geometrically, to 1e-05 or better
# at the default 32 nodes. This is the same reason `tulpa_re_aghq()` adapts
# rather than laying a fixed grid.
#
# The mode is located by a coarse scan and refined by central-difference Newton,
# both evaluated across ALL regions at once, so the whole call costs one
# length-N density evaluation per scan point, per Newton step and per node --
# independent of the region count.
sbc_loglik_re <- function(d, beta, sigma, phi = d$phi, n_quad = 32L,
                          n_scan = 81L, n_newton = 3L) {
  gh <- .sbc_gh(n_quad)
  eta0 <- as.numeric(d$X %*% beta)
  gs <- sort(unique(d$region))
  gi <- match(d$region, gs)
  G  <- length(gs)

  # The log integrand per region, at one offset per region. The prior's
  # normalizer is left out and restored once at the end, where it cancels
  # against the rule's own 2 pi.
  hg <- function(u) {
    ll <- sbc_obs_loglik(d$family, d$y, eta0 + u[gi], d$ntr, phi)
    as.numeric(rowsum(ll, gi)) - 0.5 * u^2 / sigma^2
  }

  B <- 8 * sigma + 8
  grid_u <- seq(-B, B, length.out = n_scan)
  H <- vapply(grid_u, function(v) hg(rep(v, G)), numeric(G))
  uh <- grid_u[max.col(matrix(H, G), ties.method = "first")]

  step <- 1e-3 * (sigma + 1)
  curv <- function(u) {
    h0 <- hg(u); hp <- hg(u + step); hm <- hg(u - step)
    list(g1 = (hp - hm) / (2 * step),
         g2 = pmin((hp - 2 * h0 + hm) / step^2, -1e-8))
  }
  for (it in seq_len(n_newton)) {
    cv <- curv(uh)
    uh <- uh - cv$g1 / cv$g2
  }
  s <- 1 / sqrt(-curv(uh)$g2)

  M <- vapply(seq_len(n_quad),
              function(j) hg(uh + s * gh$x[j]) + log(gh$w[j]) + 0.5 * gh$x[j]^2,
              numeric(G))
  M <- matrix(M, G)
  m <- apply(M, 1L, max)
  sum(m + log(rowSums(exp(M - m))) + log(s) - log(sigma))
}

SBC_RE_GRID <- exp(seq(log(0.15), log(2.0), length.out = 9))

# One simulated data set of the design. `cfg` carries `nr` regions of `spr`
# observations, the true `beta` and RE standard deviation `su`, the family's
# `phi` in the door's own convention, and the sigma grid the fits integrate over.
sbc_sim_re <- function(seed, family, cfg) {
  set.seed(seed)
  N <- cfg$nr * cfg$spr
  region <- rep(seq_len(cfg$nr), each = cfg$spr)
  X <- cbind(1, stats::rnorm(N))
  u <- stats::rnorm(cfg$nr, 0, cfg$su)
  eta <- as.numeric(X %*% cfg$beta) + u[region]
  list(y = sbc_draw_y(family, eta, cfg$ntr, cfg$phi), X = X,
       region = as.integer(region), N = N, nr = cfg$nr, spr = cfg$spr,
       ntr = cfg$ntr, phi = cfg$phi, grid = cfg$grid, family = family,
       theta = c(beta1 = cfg$beta[1], beta2 = cfg$beta[2], sigma = cfg$su))
}

# The arms. No `exact` here -- the whole point of this fixture is that no closed
# form exists -- so `mixture` is the read under test and `wide` / `narrow` are
# the controls that make the harness falsifiable on it.
sbc_arms_re <- function(d, fit = NULL, bad_factor = 1.25, n_ref = 200L,
                        n_quad = 64L) {
  f <- if (is.null(fit)) sbc_fit_nested(d) else fit
  P <- sbc_engine_post(f)
  se <- sqrt(pmax(diag(P$vcov), 0))
  coll <- function(mult) list(
    beta1 = sbc_normal(P$mean[1], mult * se[1]),
    beta2 = sbc_normal(P$mean[2], mult * se[2]))
  arms <- list(
    mixture   = list(beta1 = sbc_mixture(P$mu[, 1], P$var[, 1], P$w),
                     beta2 = sbc_mixture(P$mu[, 2], P$var[, 2], P$w),
                     sigma = sbc_discrete(P$grid, P$w)),
    collapsed = coll(1),
    wide      = coll(bad_factor),
    narrow    = coll(1 / bad_factor))
  arms$mixture$log_lik <- sbc_loglik_rank(
    d, P$mu, P$cov, P$w, n_ref = n_ref, grid = P$grid,
    ll_fn = function(dd, b, s, phi) sbc_loglik_re(dd, b, s, phi, n_quad))
  arms
}

# The six callbacks, over the same design under any family. The replicate is
# drawn on FRESH regions for the reason section 8 gives: sigma governs a
# per-region effect theta does not carry, so re-observing the regions y_obs saw
# would couple the two data sets and break the factorization.
sbc_psbc_re <- function(d_obs, nr_rep = d_obs$nr, bad_factor = 1.25,
                        n_ref = 200L, n_quad = 64L, control = NULL) {
  list(
    data_obs = d_obs,
    fit = function(data) sbc_fit_nested(data, control = control),
    draw_theta = function(fit, seed) {
      P <- sbc_engine_post(fit)
      set.seed(seed)
      k <- sample.int(length(P$w), 1L, prob = P$w)
      b <- P$mu[k, ] + as.numeric(crossprod(chol(P$cov[[k]]),
                                            stats::rnorm(ncol(P$mu))))
      c(beta1 = b[1], beta2 = b[2], sigma = P$grid[k])
    },
    simulate = function(theta, seed) {
      set.seed(seed)
      N <- nr_rep * d_obs$spr
      region <- rep(seq_len(nr_rep), each = d_obs$spr)
      X <- cbind(1, stats::rnorm(N))
      u <- stats::rnorm(nr_rep, 0, theta[["sigma"]])
      eta <- as.numeric(X %*% c(theta[["beta1"]], theta[["beta2"]])) + u[region]
      list(y = sbc_draw_y(d_obs$family, eta, d_obs$ntr, d_obs$phi), X = X,
           region = region, nr = nr_rep)
    },
    pool = function(obs, rep) {
      list(y = c(obs$y, rep$y), X = rbind(obs$X, rep$X),
           region = as.integer(c(obs$region, rep$region + obs$nr)),
           N = obs$N + length(rep$y), nr = obs$nr + rep$nr, spr = obs$spr,
           ntr = obs$ntr, phi = obs$phi, grid = obs$grid, family = obs$family)
    },
    arms = function(fit, data) sbc_arms_re(data, fit = fit,
                                           bad_factor = bad_factor,
                                           n_ref = n_ref, n_quad = n_quad),
    group_ids = function(data) data$region)
}

# The shipped per-fit reliability read, as one row. This is the cheap local
# score gcol33/tulpa#339 cross-tabulates the expensive calibration check
# against, and `reliability` is the combined verdict `print.laplace_diagnostics()`
# prints.
#
# It composes the layers rather than calling `diagnostics()` because that reader
# returns NULL on exactly the fits measured here -- a single-block nested fit
# carries no `$draws`, and the whole table is gated on them even though every
# quantity below is read off the fit alone (gcol33/tulpa#348, and #347 for the
# absent accessor). The composition is the same one `.tulpa_approx_diag_table()`
# performs, so when either lands this becomes one call and the numbers do not
# move.
sbc_reliability_row <- function(fit) {
  psis    <- .tulpa_psis_reliability(fit)
  grid    <- .tulpa_grid_reliability(fit)
  inner   <- .tulpa_inner_skew_reliability(fit)
  inner_k <- .tulpa_inner_k_reliability(fit)
  or      <- function(x, nm, alt) if (is.null(x)) alt else x[[nm]]
  outer_band   <- .tulpa_khat_band(psis$pareto_k)
  inner_band   <- or(inner, "band", NA_character_)
  inner_k_band <- or(inner_k, "band", NA_character_)
  data.frame(
    reliability   = .tulpa_combined_reliability(
      outer_band, inner_band, or(inner, "declined", NA_character_),
      psis$pareto_k_declined, inner_k_band,
      or(inner_k, "declined", NA_character_)),
    pareto_k      = as.numeric(psis$pareto_k),
    pareto_k_band = as.character(outer_band),
    ess_grid      = as.numeric(or(grid, "ess_grid", NA_real_)),
    gamma3_max    = as.numeric(or(inner, "max_abs_gamma3", NA_real_)),
    gamma3_band   = as.character(inner_band),
    inner_k       = as.numeric(or(inner_k, "max_pareto_k", NA_real_)),
    inner_k_band  = as.character(inner_k_band),
    stringsAsFactors = FALSE)
}
