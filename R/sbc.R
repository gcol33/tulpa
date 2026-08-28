# sbc.R
#
# Simulation-based calibration (SBC): the posterior arbiter that reads the whole
# marginal CDF, alongside the fixed-truth recovery sweeps and the reliability
# band (`diagnostics()`).
#
# WHAT THIS MEASURES THAT COVERAGE DOES NOT. A coverage indicator at one or two
# nominal levels reads one or two points of the marginal CDF, so it cannot
# separate a biased posterior from an over-dispersed one from an asymmetric one,
# and it has almost no power when a candidate moves 0 or 1 trials of 200. Two
# instruments here read the whole distribution instead:
#
#   PIT / SBC   draw the truth from the prior, theta_s ~ p(theta), y_s ~
#               p(y | theta_s), fit, and take u_s = F_s(theta_s). Under correct
#               inference u_s ~ Uniform(0, 1) exactly, and the entire ECDF is
#               the measurement.
#   CRPS        integral [F(z) - 1(z >= theta)]^2 dz, paired seed by seed. It is
#               strictly proper, so it scores calibration and sharpness together
#               and is continuous where coverage is binary.
#
# THE SCOPING CONSTRAINT ON CRPS IS LOAD-BEARING. CRPS is a proper score for the
# POSTERIOR only in a prior-predictive experiment, where theta ~ p(theta) and
# y ~ p(y | theta). Hold theta = theta_0 fixed across seeds and the CRPS-optimal
# forecast becomes a point mass at theta_0, not the posterior -- a sharper,
# wrong posterior then wins. Every result carries the experiment it came from
# (`truth = "prior_draw"` / `"fixed"`), `sbc_report()` reports the role the
# score has under it, and `sbc_crps_compare()` -- the function that ranks arms
# as posterior approximations -- refuses to run on a fixed-truth experiment. On
# fixed truth the CRPS column is a descriptive loss and is labelled as one.
#
# PROVENANCE. Sections 1 to 6 were written and arbitrated as
# `tests/testthat/helper-sbc.R`, and were moved here unchanged so the
# scorer has ONE implementation: the tests now read these functions rather than
# a parallel private copy. The band, the within-atom PIT and both CRPS closed
# forms are held against closed forms, brute-force simulation, numerical
# integration and the published Kolmogorov critical value in
# `tests/testthat/test-sbc-crps.R`. Sections 7 to 9 of that file -- the engine
# fixtures, the only parts that call a fitter -- stayed there.
#
# LAYOUT. 1 predictive shapes, 2 PIT, 3 CRPS, 4 the simultaneous band, 5 the
# prior-SBC driver, 6 the posterior-SBC driver, 7 the
# exported front door `sbc()` and its methods.

# ---------------------------------------------------------------------------
# 1. Predictive distributions
#
# One small tagged representation per shape a backend can report. Everything
# downstream (PIT, CRPS, sampling) dispatches on `kind`, so a new backend shape
# is one entry in three switches rather than a parallel scorer.
# ---------------------------------------------------------------------------

# A Gaussian mixture, which is what an outer hyperparameter grid defines for a
# fixed effect: component k is N(mu_k, var_k) with weight w_k.

#' Predictive shapes an SBC fitter reports
#'
#' The tagged representations [sbc()] reads. A `fitter` (or a posterior-SBC
#' `arms`) callback returns a named list of ARMS, each a named list over
#' quantities, and each entry is one of these -- the shape the backend actually
#' reports for that quantity. Everything downstream (the PIT, the CRPS, drawing
#' from a predictive) dispatches on the `kind` tag, so a new backend shape is
#' one entry in three switches rather than a parallel scorer.
#'
#' These are the extension point, not alternative front doors: [sbc()] is the
#' verb, and these are the argument type it consumes.
#'
#' `sbc_mixture()` is what an outer hyperparameter grid defines for a fixed
#' effect -- component `k` is `N(mu_k, var_k)` with weight `w_k`, which is
#' exactly the mixture a nested-Laplace fit reports. `sbc_normal()` is the
#' one-component case, with its own constructor so a collapsed-moment read says
#' what it is. `sbc_discrete()` is a distribution on a finite support, which is
#' what a discrete hyperparameter grid defines for its own axis. `sbc_rank()` is
#' a rank of the truth among `n_ref` reference values, which is what a joint
#' log-likelihood comparison against posterior draws produces -- it needs no
#' entry in the simulator's `theta`, since the comparison against the truth
#' already happened when the rank was formed. `sbc_draws()` is for a backend
#' reporting no analytic marginal.
#'
#' The last three have ATOMS, so their PIT is randomized within the atom by
#' [sbc()]; reading a rank against a continuous uniform is the classic silent
#' SBC bug.
#'
#' @param mu,var,w Component means, variances and weights. `w` defaults to
#'   equal weights and is normalized.
#' @param mean,sd Mean and standard deviation of a single Gaussian.
#' @param support,probs Finite support and its probabilities, normalized.
#' @param rank,n_ref The rank in `0:n_ref` of the truth among `n_ref` reference
#'   values, and that reference count.
#' @param x Posterior draws.
#' @return A list carrying a `kind` tag and that shape's parameters.
#' @seealso [sbc()]
#' @examples
#' sbc_normal(0.3, 0.1)
#' sbc_mixture(mu = c(0, 1), var = c(1, 4), w = c(0.7, 0.3))
#' sbc_discrete(support = c(0.5, 1, 2), probs = c(0.2, 0.5, 0.3))
#' @name sbc_predictive
#' @export
sbc_mixture <- function(mu, var, w = NULL) {
  mu <- as.numeric(mu); var <- as.numeric(var)
  if (is.null(w)) w <- rep(1 / length(mu), length(mu))
  w <- as.numeric(w) / sum(w)
  stopifnot(length(mu) == length(var), length(mu) == length(w),
            all(is.finite(mu)), all(is.finite(var)), all(var >= 0))
  list(kind = "mixture", mu = mu, var = var, w = w)
}

# A single Gaussian: the one-component mixture, with its own constructor so a
# collapsed-moment read says what it is.

#' @rdname sbc_predictive
#' @export
sbc_normal <- function(mean, sd) sbc_mixture(mean, sd^2, 1)

# A distribution on a finite support, which is what a discrete hyperparameter
# grid defines for its own axis.

#' @rdname sbc_predictive
#' @export
sbc_discrete <- function(support, probs) {
  support <- as.numeric(support); probs <- as.numeric(probs) / sum(probs)
  o <- order(support)
  list(kind = "discrete", support = support[o], probs = probs[o])
}

# A rank r in {0, ..., n_ref} of the truth among n_ref reference values, which
# is what a joint log-likelihood comparison against posterior draws produces.

#' @rdname sbc_predictive
#' @export
sbc_rank <- function(rank, n_ref) {
  list(kind = "rank", rank = as.integer(rank), n_ref = as.integer(n_ref))
}

# Posterior draws, for a backend that reports no analytic marginal.

#' @rdname sbc_predictive
#' @export
sbc_draws <- function(x) list(kind = "draws", x = as.numeric(x))

# ---------------------------------------------------------------------------
# 2. PIT
#
# DISCRETENESS. A mixture PIT is continuous: F is analytic, so u = F(theta) is
# exactly Uniform(0, 1) under correct inference and the uniform reference needs
# no adjustment. A rank / draws / discrete PIT is not: the CDF has atoms, and
# reading u = rank / n_ref against a continuous uniform is the classic silent
# SBC bug, putting spurious mass at the ends at small n_ref. This file
# RANDOMIZES within the atom,
#
#     u = F(theta^-) + V * P(theta),   V ~ Uniform(0, 1),
#
# which for a rank r out of n_ref reference values is u = (r + V) / (n_ref + 1).
# That is exactly Uniform(0, 1) when the rank is discrete-uniform, so one
# uniform reference and one band serve every quantity. The randomizing draws
# come from a dedicated stream in `recov_sbc()`, so a result is reproducible and
# the fits are not perturbed by asking for the diagnostic.
# ---------------------------------------------------------------------------

sbc_pit <- function(dist, y, u = stats::runif(1)) {
  switch(dist$kind,
    mixture  = sum(dist$w * stats::pnorm(y, dist$mu, sqrt(pmax(dist$var, 0)))),
    discrete = {
      k <- which.min(abs(dist$support - y))
      if (abs(dist$support[k] - y) > 1e-8 * max(1, abs(y))) {
        # Off the support the atom is empty, so there is nothing to randomize
        # within and the CDF value is the whole answer.
        sum(dist$probs[dist$support < y])
      } else {
        sum(dist$probs[seq_len(k - 1L)]) + u * dist$probs[k]
      }
    },
    rank  = (dist$rank + u) / (dist$n_ref + 1),
    draws = (sum(dist$x < y) + u * (1 + sum(dist$x == y))) / (length(dist$x) + 1),
    stop("unhandled predictive kind ", dist$kind))
}

# The folded PIT. If u ~ Uniform(0, 1) then 2 |u - 1/2| ~ Uniform(0, 1) as well,
# so the same band applies -- and a symmetric over- or under-dispersion, which
# leaves the raw ECDF looking uniform because its two halves cancel, shows here
# as a one-sided departure.
sbc_fold <- function(u) 2 * abs(u - 0.5)

# ---------------------------------------------------------------------------
# 3. CRPS
#
# CRPS(F, y) = integral [F(z) - 1(z >= y)]^2 dz, equivalently the kernel score
# E|X - y| - (1/2) E|X - X'| for X, X' iid from F. The kernel form is what makes
# a Gaussian mixture closed: with A(m, s2) = E|N(m, s2)| the absolute first
# moment,
#
#   A(m, s2) = m (2 Phi(m/s) - 1) + 2 s phi(m/s),   s = sqrt(s2),
#
#   CRPS = sum_k w_k A(y - mu_k, s2_k)
#          - (1/2) sum_k sum_l w_k w_l A(mu_k - mu_l, s2_k + s2_l)
#
# (Grimit, Gneiting, Berrocal & Johnson 2006), so the nested tier's own mixture
# is scored with no Monte Carlo. The same kernel form gives the discrete case in
# one line. Both closed forms are held against numerical integration of the
# definition and against the Monte-Carlo kernel estimator in test-sbc-crps.R.
# ---------------------------------------------------------------------------

# Absolute first moment of N(m, s2), vectorized over m and s2. A degenerate
# component (s2 = 0) has E|N(m, 0)| = |m|, which the branch reproduces without
# a division by zero.
.sbc_abs_moment <- function(m, s2) {
  s <- sqrt(pmax(s2, 0))
  out <- abs(m)
  ok <- s > 0
  if (any(ok)) {
    z <- m[ok] / s[ok]
    out[ok] <- m[ok] * (2 * stats::pnorm(z) - 1) + 2 * s[ok] * stats::dnorm(z)
  }
  out
}

sbc_crps <- function(dist, y) {
  switch(dist$kind,
    mixture = {
      t1 <- sum(dist$w * .sbc_abs_moment(y - dist$mu, dist$var))
      t2 <- sum(outer(dist$w, dist$w) *
                  .sbc_abs_moment(outer(dist$mu, dist$mu, "-"),
                                  outer(dist$var, dist$var, "+")))
      t1 - 0.5 * t2
    },
    discrete = {
      s <- dist$support; p <- dist$probs
      sum(p * abs(s - y)) - 0.5 * sum(outer(p, p) * abs(outer(s, s, "-")))
    },
    draws = {
      x <- dist$x; n <- length(x)
      # The kernel estimator, with the sorted-value identity for the second term
      # so a length-n draw set costs O(n log n) rather than O(n^2).
      xs <- sort(x)
      mean(abs(x - y)) - sum((2 * seq_len(n) - n - 1) * xs) / n^2
    },
    rank = NA_real_,
    stop("unhandled predictive kind ", dist$kind))
}

# The definition, integrated numerically. An arbiter for the closed forms above;
# not used by the driver.
#
# The integrand jumps at y, and a trapezoid rule straddling a jump converges
# only as O(h). Splitting at y -- and, for a discrete predictive, at every atom
# -- leaves a smooth piece on each side, which buys O(h^2) and makes a few
# thousand points a tighter arbiter than a few hundred thousand straddling ones.
sbc_crps_integral <- function(dist, y, n_grid = 20001L) {
  stopifnot(dist$kind %in% c("mixture", "discrete"))
  if (identical(dist$kind, "mixture")) {
    sd <- sqrt(pmax(dist$var, 0))
    lo <- min(c(dist$mu - 12 * sd, y)) - 1
    hi <- max(c(dist$mu + 12 * sd, y)) + 1
    brk <- y
    cdf <- function(q) as.numeric(crossprod(dist$w, stats::pnorm(
      matrix(q, length(dist$mu), length(q), byrow = TRUE), dist$mu, sd)))
  } else {
    lo <- min(c(dist$support, y)) - 1
    hi <- max(c(dist$support, y)) + 1
    brk <- sort(unique(c(y, dist$support)))
    cdf <- function(q) vapply(q, function(z)
      sum(dist$probs[dist$support <= z]), numeric(1))
  }
  ends <- sort(unique(c(lo, brk, hi)))
  total <- 0
  for (s in seq_len(length(ends) - 1L)) {
    z <- seq(ends[s], ends[s + 1L], length.out = n_grid)
    # Evaluate strictly inside the panel so the indicator and the discrete CDF
    # are each constant across it.
    zi <- (z[-1] + z[-n_grid]) / 2
    g <- (cdf(zi) - as.numeric(zi >= y))^2
    total <- total + mean(g) * (ends[s + 1L] - ends[s])
  }
  total
}

# Draw from a predictive.
sbc_sample <- function(dist, n) {
  switch(dist$kind,
    mixture = {
      k <- sample.int(length(dist$w), n, replace = TRUE, prob = dist$w)
      stats::rnorm(n, dist$mu[k], sqrt(pmax(dist$var[k], 0)))
    },
    discrete = sample(dist$support, n, replace = TRUE, prob = dist$probs),
    draws = sample(dist$x, n, replace = TRUE),
    stop("cannot sample predictive kind ", dist$kind))
}

# The Monte-Carlo kernel estimator, the second independent arbiter.
sbc_crps_mc <- function(dist, y, n = 2e6L) {
  x <- sbc_sample(dist, n)
  x2 <- sbc_sample(dist, n)
  mean(abs(x - y)) - 0.5 * mean(abs(x - x2))
}

# ---------------------------------------------------------------------------
# 4. Simultaneous ECDF bands
#
# A POINTWISE BINOMIAL BAND IS NOT A SIMULTANEOUS BAND. At n = 200 the ECDF is
# read at 200 correlated points, and a band holding each of them at 95% holds
# all of them at far less. What follows calibrates a one-parameter band family
# so that the SIMULTANEOUS coverage is nominal, with the crossing probability
# computed exactly rather than simulated.
#
# THE EXACT CROSSING PROBABILITY. A band on the uniform ECDF is equivalent to a
# pair of bounds on the order statistics, g_i <= U_(i) <= h_i. Writing
# K(t) = #{i : U_i <= t}, the constraint U_(i) <= h_i is K(h_i) >= i and the
# constraint U_(i) >= g_i is K(g_i) <= i - 1, so at any point p
#
#     lo(p) = #{i : h_i <= p}  <=  K(p)  <=  #{i : g_i < p} = hi(p),
#
# and imposing that at the union of the g's and the h's is exactly the original
# event -- every other t is implied because K is non-decreasing. K is a Markov
# chain in p (K(p') | K(p) = k is k + Binomial(n - k, (p' - p) / (1 - p))), so
# the probability is a forward pass over those points with the state masked to
# [lo, hi]: exact, 2n transitions, and the mask keeps the live state window at
# the band's own width rather than all n + 1 counts.
#
# TWO BOUNDARY FAMILIES ride that one recursion:
#   "beta"  equal local levels -- g_i, h_i the gamma/2 and 1 - gamma/2 quantiles
#           of Beta(i, n - i + 1), the exact pointwise law of U_(i). Narrow in
#           the tails, where the ECDF has little room, and wide in the middle.
#           This is the band for an SBC ECDF-difference plot.
#   "ks"    a constant-width Kolmogorov-Smirnov band, |F_n(t) - t| <= d, i.e.
#           g_i = i/n - d and h_i = (i - 1)/n + d. The conservative cross-check:
#           its calibrated d reproduces the published Kolmogorov critical value
#           to four figures, which is an external arbiter on the recursion.
# Both are calibrated by bisection on their own parameter against the same exact
# crossing probability, and both are validated by measured simultaneous coverage
# over simulated uniform samples in test-sbc-crps.R.
# ---------------------------------------------------------------------------

# P(g_i <= U_(i) <= h_i for all i) for n = length(g) iid Uniform(0, 1) draws.
#
# Consecutive boundary points are about 1/(2n) apart, so the number of draws
# arriving in one step is 0, 1 or 2 with overwhelming probability and the
# transition is banded: the loop fills only the offsets d whose remaining
# binomial tail exceeds `tail_tol`, which turns an O(n w^2) pass into O(n w d).
# The discarded mass is bounded by 2n * tail_tol, machine precision at the
# default, so this is the same number the dense pass returns.
sbc_crossing_prob <- function(g, h, tail_tol = 1e-18) {
  n <- length(g)
  stopifnot(n >= 1L, length(h) == n, all(g <= h))
  pts <- sort(unique(c(g, h)))
  pts <- pts[pts > 0 & pts < 1]
  if (!length(pts)) return(1)
  lo <- vapply(pts, function(p) sum(h <= p), numeric(1))
  hi <- pmin(vapply(pts, function(p) sum(g < p), numeric(1)), n)
  prev <- 0
  from <- 0
  v <- 1
  for (j in seq_along(pts)) {
    if (hi[j] < lo[j]) return(0)
    q <- (pts[j] - prev) / (1 - prev)
    to <- lo[j]:hi[j]
    dmax <- min(n, stats::qbinom(tail_tol, n, q, lower.tail = FALSE) + 1)
    vn <- numeric(length(to))
    for (d in 0:dmax) {
      kt <- from + d
      ok <- kt >= lo[j] & kt <= hi[j] & d <= (n - from)
      if (!any(ok)) next
      idx <- kt[ok] - lo[j] + 1L
      vn[idx] <- vn[idx] + v[ok] * stats::dbinom(d, n - from[ok], q)
    }
    v <- vn
    from <- to
    prev <- pts[j]
  }
  sum(v)
}

.sbc_bounds <- function(n, type, par) {
  i <- seq_len(n)
  if (identical(type, "beta")) {
    list(g = stats::qbeta(par / 2, i, n - i + 1),
         h = stats::qbeta(1 - par / 2, i, n - i + 1))
  } else {
    list(g = pmax(0, i / n - par), h = pmin(1, (i - 1) / n + par))
  }
}

.sbc_band_cache <- new.env(parent = emptyenv())

# Simultaneous band for the ECDF of n iid Uniform(0, 1) values at the requested
# level. Returns the order-statistic bounds `g` / `h`, the calibrated family
# parameter, the realized simultaneous `coverage`, and the same band on the ECDF
# axis as step heights `ecdf_lo` / `ecdf_hi` at breakpoints `ecdf_at` (an ECDF
# inside the band has ecdf_lo(t) <= F_n(t) <= ecdf_hi(t) for every t).
sbc_ecdf_band <- function(n, level = 0.95, type = c("beta", "ks"), tol = 1e-7) {
  type <- match.arg(type)
  key <- sprintf("%s|%d|%.10f", type, n, level)
  if (!is.null(.sbc_band_cache[[key]])) return(.sbc_band_cache[[key]])
  alpha <- 1 - level
  f <- function(par) {
    bd <- .sbc_bounds(n, type, par)
    sbc_crossing_prob(bd$g, bd$h) - level
  }
  rng <- if (identical(type, "beta")) c(alpha / (4 * n), alpha) else
    c(0.2 / sqrt(n), min(0.999, 4 / sqrt(n)))
  par <- stats::uniroot(f, rng, tol = tol)$root
  bd <- .sbc_bounds(n, type, par)
  at <- sort(unique(c(0, bd$g, bd$h, 1)))
  out <- list(n = n, level = level, type = type, par = par,
              g = bd$g, h = bd$h, ecdf_at = at,
              ecdf_lo = vapply(at, function(p) sum(bd$h <= p), numeric(1)) / n,
              ecdf_hi = vapply(at, function(p) sum(bd$g <  p), numeric(1)) / n,
              coverage = sbc_crossing_prob(bd$g, bd$h))
  .sbc_band_cache[[key]] <- out
  out
}

# Does this sample's ECDF stay inside the band everywhere?
sbc_ecdf_inside <- function(u, band) {
  us <- sort(u)
  stopifnot(length(us) == band$n)
  all(us >= band$g) && all(us <= band$h)
}

# Kolmogorov-Smirnov distance of the sample from Uniform(0, 1), reported
# alongside the band as the scale-free size of the departure.
sbc_ecdf_dev <- function(u) {
  n <- length(u)
  us <- sort(u)
  i <- seq_len(n)
  max(max(i / n - us), max(us - (i - 1) / n))
}

# Exact simultaneous test of uniformity in the equal-local-levels family: shrink
# the band until it touches the observed sample, then report the probability
# that a uniform sample would have touched a band at least that tight. This is
# the continuous read the in-band indicator discretizes, and it is what a
# candidate arm is compared on.
sbc_ecdf_test <- function(u) {
  n <- length(u)
  us <- sort(u)
  i <- seq_len(n)
  p <- stats::pbeta(us, i, n - i + 1)
  gamma_obs <- 2 * min(pmin(p, 1 - p))
  if (gamma_obs <= 0) return(list(gamma = 0, p_value = 0, ks = sbc_ecdf_dev(u)))
  bd <- .sbc_bounds(n, "beta", gamma_obs)
  list(gamma = gamma_obs,
       p_value = 1 - sbc_crossing_prob(bd$g, bd$h),
       ks = sbc_ecdf_dev(u))
}

# ---------------------------------------------------------------------------
# 5. The driver
#
# `simulator(seed)` returns a list carrying `theta`, a named numeric vector of
# the true values of the scored quantities, plus whatever the fitter needs.
# `fitter(d)` returns a named list of ARMS, each a named list over quantities,
# each entry a predictive built by the constructors in section 1. Several arms
# read off one solve per seed the way `recov_sweep()` pairs its mixture /
# collapsed / skew reads, so an arm-to-arm difference carries no fit-to-fit
# noise; an arm that needs its own solve is one more entry in the same return.
#
# A quantity of kind "rank" needs no entry in `theta`: the comparison against
# the truth already happened when the rank was formed.
#
# `truth` declares the experiment and travels with the result -- "prior_draw"
# for the prior-predictive experiment that makes the PIT uniform and the CRPS a
# proper posterior score, "fixed" for a fixed-truth sweep where the CRPS column
# is a descriptive loss and the PIT is not uniform under correct inference, and
# "posterior_draw" for the posterior-SBC experiment of section 7, where the
# posterior given the observed data plays the role of the prior and both reads
# keep the meaning they have under "prior_draw".
# ---------------------------------------------------------------------------

recov_sbc <- function(simulator, fitter, n_seed, quantities = NULL,
                      seed_off = 0L,
                      truth = c("prior_draw", "fixed", "posterior_draw"),
                      rand_seed = 20240335L, progress = FALSE) {
  truth <- match.arg(truth)
  # The randomizing uniforms for the discrete PITs come from their own stream,
  # drawn up front with the ambient RNG state saved and restored, so requesting
  # the diagnostic cannot move a single simulated data set or fit.
  U <- .sbc_with_seed(rand_seed,
                      matrix(stats::runif(n_seed * 64L), n_seed, 64L))
  rows <- vector("list", n_seed)
  for (s in seq_len(n_seed)) {
    d <- simulator(seed_off + s)
    arms <- fitter(d)
    j <- 0L
    per <- list()
    for (a in names(arms)) {
      qs <- arms[[a]]
      nms <- if (is.null(quantities)) names(qs) else
        intersect(quantities, names(qs))
      for (q in nms) {
        dist <- qs[[q]]
        if (is.null(dist)) next
        j <- j + 1L
        th <- if (identical(dist$kind, "rank")) NA_real_ else {
          if (!q %in% names(d$theta))
            stop("simulator returned no truth for quantity '", q, "'")
          unname(d$theta[[q]])
        }
        per[[length(per) + 1L]] <- data.frame(
          seed = seed_off + s, arm = a, quantity = q, kind = dist$kind,
          truth = th,
          pit = sbc_pit(dist, th, u = U[s, ((j - 1L) %% 64L) + 1L]),
          crps = if (identical(dist$kind, "rank")) NA_real_ else
            sbc_crps(dist, th),
          stringsAsFactors = FALSE)
      }
    }
    rows[[s]] <- do.call(rbind, per)
    if (progress && s %% 25L == 0L) message("  sbc seed ", s, "/", n_seed)
  }
  res <- do.call(rbind, rows)
  res$pit_folded <- sbc_fold(res$pit)
  attr(res, "truth") <- truth
  attr(res, "n_seed") <- n_seed
  attr(res, "crps_role") <- switch(truth,
    prior_draw     = "proper posterior score",
    posterior_draw = "proper posterior score (updating prior = posterior at y_obs)",
    "descriptive loss (fixed truth)")
  res
}

# The two experiments in which the truth is drawn from the distribution the
# forecast is a posterior update of, so the PIT is uniform under exact
# inference and the CRPS is proper.
.SBC_PREDICTIVE_TRUTH <- c("prior_draw", "posterior_draw")

# Evaluate `expr` under a pinned RNG seed, leaving the ambient stream exactly
# where it was. Used for the randomizing uniforms and for the reference draws
# behind a log-likelihood rank, so both are reproducible without displacing the
# stream the simulators and fits run on.
.sbc_with_seed <- function(seed, expr) {
  had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
  old <- if (had) get(".Random.seed", envir = globalenv()) else NULL
  set.seed(seed)
  on.exit({
    if (had) assign(".Random.seed", old, envir = globalenv()) else
      suppressWarnings(rm(list = ".Random.seed", envir = globalenv()))
  }, add = TRUE)
  force(expr)
}

# Per (arm, quantity): the raw and folded uniformity reads against the
# simultaneous band, and the mean CRPS with its standard error.
sbc_report <- function(res, level = 0.95) {
  key <- unique(res[, c("arm", "quantity")])
  bands <- new.env(parent = emptyenv())
  out <- do.call(rbind, lapply(seq_len(nrow(key)), function(i) {
    sel <- res$arm == key$arm[i] & res$quantity == key$quantity[i]
    u <- res$pit[sel]
    n <- length(u)
    bk <- as.character(n)
    if (is.null(bands[[bk]])) bands[[bk]] <- sbc_ecdf_band(n, level)
    band <- bands[[bk]]
    tr <- sbc_ecdf_test(u)
    tf <- sbc_ecdf_test(sbc_fold(u))
    cr <- res$crps[sel]
    nc <- sum(!is.na(cr))
    data.frame(arm = key$arm[i], quantity = key$quantity[i], n = n,
               ks = tr$ks, inside = sbc_ecdf_inside(u, band),
               p_unif = tr$p_value,
               ks_folded = tf$ks,
               inside_folded = sbc_ecdf_inside(sbc_fold(u), band),
               p_unif_folded = tf$p_value,
               crps = if (nc) mean(cr, na.rm = TRUE) else NA_real_,
               crps_se = if (nc > 1L) stats::sd(cr, na.rm = TRUE) / sqrt(nc)
                         else NA_real_,
               stringsAsFactors = FALSE)
  }))
  attr(out, "crps_role") <- attr(res, "crps_role")
  attr(out, "band_level") <- level
  out
}

# Paired CRPS differences against a baseline arm, seed by seed and quantity by
# quantity. A negative `delta` is the arm scoring better than the baseline.
# Meaningful only in the prior-predictive experiment, so a fixed-truth result is
# refused here rather than quietly ranked.
#
# TWO READS OF THE SAME PAIRING, and only one of them is the verdict. `delta`
# with its `t` is the MEAN paired difference: that is the proper score, so the
# true posterior minimizes it and a ranking on it is the answer to "which
# approximation is better". `worse_frac` with its exact binomial `p_sign` is the
# MEDIAN read, which has far more power when the per-seed differences are
# heavy-tailed -- but it is not proper, and a too-sharp posterior wins most
# seeds while losing badly on the rest, so a sign test can rank it first. Read
# `p_sign` as a detector of a difference and `delta` as the direction that
# difference has.
sbc_crps_compare <- function(res, baseline) {
  if (!attr(res, "truth") %in% .SBC_PREDICTIVE_TRUTH) {
    stop("CRPS ranks posterior approximations only in a prior-predictive ",
         "experiment (truth = 'prior_draw'). This result was produced with ",
         "truth = '", attr(res, "truth"), "', where the CRPS-optimal forecast ",
         "is a point mass at the fixed truth and the column is a descriptive ",
         "loss.", call. = FALSE)
  }
  qs <- unique(res$quantity[!is.na(res$crps)])
  arms <- setdiff(unique(res$arm), baseline)
  do.call(rbind, lapply(qs, function(q) {
    b <- res[res$arm == baseline & res$quantity == q, ]
    if (!nrow(b)) return(NULL)
    do.call(rbind, lapply(arms, function(a) {
      x <- res[res$arm == a & res$quantity == q, ]
      if (!nrow(x)) return(NULL)
      m <- merge(b[, c("seed", "crps")], x[, c("seed", "crps")], by = "seed",
                 suffixes = c("_base", "_arm"))
      dlt <- m$crps_arm - m$crps_base
      se <- stats::sd(dlt) / sqrt(length(dlt))
      nw <- sum(dlt > 0)
      nt <- sum(dlt != 0)
      data.frame(quantity = q, arm = a, baseline = baseline,
                 n = length(dlt), delta = mean(dlt), se = se,
                 t = mean(dlt) / se, worse_frac = nw / length(dlt),
                 p_sign = if (nt) stats::binom.test(nw, nt)$p.value else NA_real_,
                 stringsAsFactors = FALSE)
    }))
  }))
}

# ---------------------------------------------------------------------------
# 6. Posterior SBC (Sailynoja, Schmitt, Buerkner & Vehtari, Statistics and
#    Computing 36:78, 2026, doi:10.1007/s11222-026-10825-9, Algorithm 2)
#
# WHAT IT ANSWERS THAT SECTION 5 DOES NOT. `recov_sbc(truth = "prior_draw")`
# draws the truth from the prior, so it reports self-consistency AVERAGED over
# the whole generative distribution. After observing a data set the question is
# narrower -- is the inference reliable in the posterior geometry THIS data set
# produces -- and averaging over the prior can both miss a defect confined to a
# small region and flag one the observed data rules out.
#
# THE CONSTRUCTION, and each part of it is load-bearing:
#
#   theta'_i ~ pi(theta | y_obs)              truth drawn from the posterior at
#                                             the observed data, not the prior
#   y_i      ~ pi(y | theta'_i)               a replicate at that truth
#   theta''  ~ pi(theta | y_i, y_obs)         the AUGMENTED posterior, which
#                                             conditions on BOTH data sets
#   u_i      = P(theta'' < theta'_i | y_i, y_obs)
#
# Sequential Bayesian updating is what makes this an SBC experiment: the paper's
# eq. (3) factorizes pi(y_obs, y, theta', theta'') = pi(theta' | y_obs)
# pi(y | theta') pi(theta'' | y, y_obs), which is the ordinary SBC identity with
# pi(theta | y_obs) in the role of the prior. Drop the y_obs from the augmented
# posterior -- fit the replicate ALONE -- and the identity no longer holds:
# that is ordinary SBC under a hand-made prior, and it is the mistake this
# construction is easiest to collapse into.
#
# CONDITIONAL INDEPENDENCE IS A CONSTRAINT ON `simulate()`. The factorization
# needs pi(y | theta', y_obs) = pi(y | theta'), so the replicate must be
# independent of the observed data given theta. For a hierarchical model whose
# random effects are integrated into the likelihood -- which is what the nested
# tier does -- theta carries no per-group value, so a replicate must be drawn on
# FRESH groups / cells rather than as a second observation of the ones y_obs
# already saw. Re-using the observed grouping couples the two data sets through
# the unmodelled group effects and silently breaks the identity.
#
# THE TRUTH COMES FROM THE ALGORITHM'S OWN APPROXIMATE POSTERIOR, deliberately.
# The identity holds for the exact posterior at both stages, so running it with
# the approximation at both stages is what turns non-uniformity into a verdict
# on the approximation -- which is the whole point here. The paper makes the
# same choice for amortized inference in its Section 4.3.
#
# AN IMPROPER PRIOR IS NO OBSTACLE, unlike in prior SBC. The nested door puts no
# prior on the fixed effects, so section 7's prior-SBC fixture has to hold beta
# fixed and lean on a structural argument to keep its PIT uniform. Posterior SBC
# draws every scored quantity from a proper distribution -- the posterior at
# y_obs -- so no such argument is needed and the fixed effects are scored the
# same way as everything else.
#
# `model` is a list of six callbacks, and nothing here knows what a data set or
# a fit is beyond passing them between the callbacks:
#   data_obs                the observed data set
#   fit(data)               the inference algorithm under test
#   draw_theta(fit, seed)   one named numeric draw from that fit's posterior
#   simulate(theta, seed)   a replicate, conditionally independent of data_obs
#   pool(data_obs, rep)     the augmented data set
#   arms(fit, data)         named arms of predictives, as in section 5
#
# THE DRIVER SPLITS THE TWO RNG STREAMS. `draw_theta` gets
# the replicate's seed `s` and `simulate` gets `.sbc_rep_seed(s)`, so the obvious
# `set.seed(seed)` at the top of each callback is the CORRECT fixture. Handing
# both the same seed instead makes the replicate's group effects and residuals
# re-consume the very uniforms that produced theta', so the truth determines the
# noise as well as the parameter -- not p(y | theta'), and it surfaces as a
# non-uniform PIT with nothing whatever wrong in the inference under test. A
# harness whose default failure mode is a false alarm against the engine is the
# wrong default, so the split lives here and no fixture carries an offset.
# ---------------------------------------------------------------------------

# The offset the driver derives the replicate's seed with. Its value is the one
# every fixture used to apply itself, so promoting the split into the driver
# leaves the seeds each fixture actually sees unchanged.
.SBC_REP_SEED_OFFSET <- 660000L

.sbc_rep_seed <- function(seed) seed + .SBC_REP_SEED_OFFSET

recov_posterior_sbc <- function(model, n_seed, quantities = NULL,
                                seed_off = 0L, rand_seed = 20240339L,
                                progress = FALSE) {
  need <- c("data_obs", "fit", "draw_theta", "simulate", "pool", "arms")
  miss <- setdiff(need, names(model))
  if (length(miss)) stop("model is missing: ", paste(miss, collapse = ", "))

  fit_obs <- model$fit(model$data_obs)

  simulator <- function(seed) {
    th <- model$draw_theta(fit_obs, seed)
    rp <- model$simulate(th, .sbc_rep_seed(seed))
    d <- model$pool(model$data_obs, rp)
    d$theta <- th
    d$seed <- seed
    d
  }
  fitter <- function(d) model$arms(model$fit(d), d)

  res <- recov_sbc(simulator, fitter, n_seed = n_seed, quantities = quantities,
                   seed_off = seed_off, truth = "posterior_draw",
                   rand_seed = rand_seed, progress = progress)
  attr(res, "fit_obs") <- fit_obs
  res
}

# ---------------------------------------------------------------------------
# 7. The front door
#
# ONE exported verb. `experiment` selects which of the two drivers above runs,
# and everything else is the callback contract those drivers read. A FIXED-TRUTH
# sweep is deliberately not offered: it is not an SBC experiment (the PIT is not
# uniform under correct inference there) and its CRPS column is a descriptive
# loss, so the refusal `sbc_crps_compare()` enforces is upstream of anything
# this door can produce. `recov_sweep()` in the recovery tests is where a
# fixed-truth sweep lives.
#
# TWO GUARDS, because each premise silently invalidates the result when it is
# broken and neither announces itself in the output.
#
#   The prior-predictive experiment draws the truth from the prior the fit
#   updates. The nested door puts NO prior on the fixed effects, so they cannot
#   be drawn from -- and a harness that scores them at a value held fixed across
#   seeds reports a non-uniform PIT that reads as an engine defect. What is
#   observable from outside is exactly that: a scored quantity whose truth does
#   not move across simulations was not drawn from a proper prior.
#   `.sbc_check_proper_prior()` probes the simulator (no fits) and errors,
#   naming `experiment = "posterior"`, which needs no proper prior. A location
#   parameter whose flat prior leaves the PIT uniform by a structural argument
#   is admitted through `flat_prior =`, which is itself checked -- a name that
#   DOES move is a contradicted declaration and errors too -- and travels on the
#   result.
#
#   The posterior experiment rests on two premises of the sequential-updating
#   factorization. `pool()` must return BOTH data sets, since fitting the
#   replicate alone is ordinary SBC under a hand-made prior; that is checked
#   from the sizes of what the callback returns. And the replicate must be
#   conditionally independent of the observed data given theta, which for a
#   hierarchical model means FRESH groups; that is checked whenever the caller
#   supplies `group_ids`, and reported as unverified when it does not. Both
#   checks ride the first `pool()` call, so neither costs a fit.
# ---------------------------------------------------------------------------

# Simulator probes for the proper-prior guard. Ten seeds because the check is
# "did this quantity move at all": on the coarsest prior support a fixture
# actually uses -- a seven-cell discrete sigma grid -- ten independent draws
# coincide with probability 7^-9, so a proper prior is not mistaken for a fixed
# one. The probe calls the simulator only, never the fitter.
.SBC_PROBE_SEEDS <- 10L

# The quantities one `fitter()` return scores, and the kinds each is reported
# under. A quantity every arm reports as a "rank" needs no entry in `theta`:
# the comparison against the truth already happened when the rank was formed.
.sbc_scored_kinds <- function(arms, quantities = NULL) {
  out <- list()
  for (a in names(arms)) {
    qs <- arms[[a]]
    nms <- if (is.null(quantities)) names(qs) else intersect(quantities, names(qs))
    for (q in nms) {
      d <- qs[[q]]
      if (is.null(d)) next
      out[[q]] <- unique(c(out[[q]], d$kind))
    }
  }
  out
}

.sbc_check_proper_prior <- function(simulator, fitter, seed_off, n_sim,
                                    quantities, flat_prior) {
  n_probe <- max(2L, min(as.integer(n_sim), .SBC_PROBE_SEEDS))
  d1 <- simulator(seed_off + 1L)
  kinds <- .sbc_scored_kinds(fitter(d1), quantities)
  if (!length(kinds)) {
    stop("fitter() returned no scored quantity", call. = FALSE)
  }
  scored <- names(kinds)[!vapply(kinds, function(k) all(k == "rank"), logical(1))]

  bad <- setdiff(flat_prior, scored)
  if (length(bad)) {
    stop("flat_prior names ", paste(sQuote(bad), collapse = ", "),
         ", which the fitter does not score as a quantity carrying a truth. ",
         "Scored: ", paste(sQuote(scored), collapse = ", "), ".", call. = FALSE)
  }
  if (!length(scored)) return(list(n_probe = 0L, scored = scored))

  th <- lapply(seq_len(n_probe), function(s) {
    if (s == 1L) d1$theta else simulator(seed_off + s)$theta
  })
  moves <- vapply(scored, function(q) {
    v <- vapply(th, function(t) {
      if (is.null(t) || !q %in% names(t)) NA_real_ else as.numeric(t[[q]])
    }, numeric(1))
    # A missing truth is the driver's own error to report, with its own message.
    if (anyNA(v)) return(NA)
    length(unique(v)) > 1L
  }, logical(1))

  fixed <- scored[!is.na(moves) & !moves]
  undeclared <- setdiff(fixed, flat_prior)
  if (length(undeclared)) {
    stop("prior-predictive SBC draws the truth from the prior the fit updates, ",
         "so every scored quantity must come from a PROPER prior. ",
         paste(sQuote(undeclared), collapse = ", "),
         if (length(undeclared) > 1L) " each took the same value in all "
         else " took the same value in all ", n_probe,
         " probed simulations, which is what an improper prior looks like from ",
         "here -- the nested-Laplace door puts no prior on the fixed effects, ",
         "so they cannot be drawn from. Either draw from a proper prior, or use ",
         "experiment = \"posterior\", which conditions on an observed data set ",
         "and needs no proper prior. If it is a location parameter whose flat ",
         "prior leaves the PIT uniform by a structural argument, name it in ",
         "flat_prior = and record that argument.", call. = FALSE)
  }
  contradicted <- intersect(flat_prior, scored[!is.na(moves) & moves])
  if (length(contradicted)) {
    stop("flat_prior declares ", paste(sQuote(contradicted), collapse = ", "),
         " held fixed under a flat prior, but the truth moved across the ",
         n_probe, " probed simulations. A quantity drawn from a proper prior ",
         "is scored without the declaration.", call. = FALSE)
  }
  list(n_probe = n_probe, scored = scored)
}

# The pooling premise, checked without knowing what a data set is.
#
# Two independent reads, and a pooling is refused only when BOTH fail, so a
# `pool()` that legitimately transforms or nests its inputs is not refused for
# it. CONTAINMENT is the direct statement of the premise -- the pooled set has
# to still carry the values each input carried -- and it passes a pooling that
# merely wraps them (`list(obs = obs, rep = rep)`) as well as one that
# concatenates. EXTENT is the fallback for a pooling that rewrites the values:
# whatever else it does, a data set holding both is at least as long as the two
# together.
.sbc_num_leaves <- function(x) {
  if (is.numeric(x)) return(list(as.numeric(x)))
  if (is.list(x) && length(x)) {
    return(unlist(lapply(x, .sbc_num_leaves), recursive = FALSE))
  }
  list()
}

# The longest numeric leaf, which for any data set is its observation-carrying
# one; a shorter leaf is metadata a pooled set may legitimately not extend.
.sbc_longest_leaf <- function(x) {
  L <- .sbc_num_leaves(x)
  if (!length(L)) return(numeric())
  L[[which.max(lengths(L))]]
}

.sbc_extent <- function(x) length(.sbc_longest_leaf(x))

.sbc_carries <- function(pooled, v) {
  if (!length(v)) return(TRUE)
  L <- .sbc_num_leaves(pooled)
  any(vapply(L, function(z) length(z) >= length(v) && all(v %in% z),
             logical(1)))
}

.sbc_check_pooling <- function(obs, rep, pooled) {
  v_o <- .sbc_longest_leaf(obs)
  v_r <- .sbc_longest_leaf(rep)
  big <- .sbc_extent(pooled) >= length(v_o) + length(v_r)
  if (!big && !.sbc_carries(pooled, v_o)) {
    stop("posterior SBC scores the AUGMENTED posterior pi(theta | y, y_obs), so ",
         "pool() must return the observed data set AND the replicate. What it ",
         "returned carries neither the observed data's values nor enough of ",
         "them to hold both data sets (extent ", .sbc_extent(pooled),
         " against ", length(v_o), " + ", length(v_r),
         "). Fitting the replicate by itself is ordinary SBC under a hand-made ",
         "prior, not posterior SBC, and the sequential-updating identity it ",
         "rests on does not hold.", call. = FALSE)
  }
  if (!big && !.sbc_carries(pooled, v_r)) {
    stop("pool() returned a data set carrying neither the replicate's values ",
         "nor enough of them to hold both data sets (extent ",
         .sbc_extent(pooled), " against ", length(v_o), " + ", length(v_r),
         "), so the replicate is not conditioned on and nothing is being ",
         "calibrated.", call. = FALSE)
  }
  invisible(TRUE)
}

# WHAT THIS VERIFIES, AND WHAT IT CANNOT. The premise is that the replicate is
# conditionally independent of the observed data given theta, which for a model
# whose group effects are integrated into the likelihood means the replicate is
# drawn on groups y_obs never saw. The observable half is the LABELS: a pooled
# data set with fewer groups than its two parts together is re-observing groups
# it already had. The other half -- that the simulator drew those groups' effects
# from the prior rather than from p(u | y_obs, theta) -- is not visible from
# outside the callback at all, so it is not claimed. A result records
# `"verified (disjoint group labels)"`, never "premise verified".
.sbc_check_fresh_groups <- function(group_ids, obs, rep, pooled) {
  if (is.null(group_ids)) return("undeclared")
  n_o <- length(unique(group_ids(obs)))
  n_r <- length(unique(group_ids(rep)))
  n_p <- length(unique(group_ids(pooled)))
  if (n_p != n_o + n_r) {
    stop("the replicate must be drawn on FRESH groups: the observed data has ",
         n_o, ", the replicate ", n_r, ", and the pooled data set ", n_p,
         " rather than ", n_o + n_r, ". A replicate observing groups y_obs ",
         "already saw is dependent on y_obs given theta -- the group effects ",
         "are not carried by theta -- which breaks the factorization posterior ",
         "SBC rests on.", call. = FALSE)
  }
  "verified (disjoint group labels)"
}

# Both posterior premises ride the first `pool()` call, so neither costs a fit.
.sbc_guard_model <- function(model, state) {
  base_pool <- model$pool
  if (!is.function(base_pool)) return(model)
  model$pool <- function(obs, rep) {
    pooled <- base_pool(obs, rep)
    if (!state$checked) {
      .sbc_check_pooling(obs, rep, pooled)
      state$fresh_groups <- .sbc_check_fresh_groups(model$group_ids, obs, rep,
                                                    pooled)
      state$checked <- TRUE
    }
    pooled
  }
  model
}

#' Simulation-based calibration
#'
#' Scores whether an inference algorithm's posterior is CALIBRATED, by reading
#' the whole marginal CDF rather than one or two nominal levels. Draw a truth
#' from the distribution the fit updates, simulate a data set at that truth,
#' fit, and take the probability integral transform (PIT) of the truth under the
#' reported posterior. Under exact inference those PIT values are exactly
#' Uniform(0, 1), so the entire ECDF is the measurement (Talts et al. 2018).
#'
#' @section The two experiments:
#' \describe{
#'   \item{`"prior_predictive"`}{ordinary SBC. `theta ~ p(theta)`,
#'     `y ~ p(y | theta)`, fit, PIT. It reports self-consistency AVERAGED over
#'     the whole generative distribution.}
#'   \item{`"posterior"`}{calibration CONDITIONAL on an observed data set
#'     (Sailynoja et al. 2026, Algorithm 2). `theta' ~ pi(theta | y_obs)`,
#'     `y ~ pi(y | theta')`, and the PIT is taken under the AUGMENTED posterior
#'     `pi(theta | y, y_obs)`. That is ordinary SBC with `pi(theta | y_obs)` in
#'     the role of the prior, so the same band, the same folded read and the
#'     same proper score all carry over -- and it needs no proper prior.}
#' }
#' A fixed-truth sweep is not an SBC experiment and is not offered here: its PIT
#' is not uniform under correct inference and its CRPS is a descriptive loss,
#' not a proper posterior score.
#'
#' @section What is reported:
#' Three reads of the same PIT sample, per (arm, quantity):
#' \describe{
#'   \item{raw}{the PIT ECDF against an exact SIMULTANEOUS band. A pointwise
#'     binomial band is not simultaneous -- at n = 100, holding each order
#'     statistic at 95 percent holds all of them together at 0.4471 -- so the
#'     band here is calibrated by bisection against the exact crossing
#'     probability of the uniform order statistics.}
#'   \item{folded}{`2 |u - 1/2|`, also uniform, and where a symmetric
#'     over- or under-dispersion shows after cancelling in the raw ECDF.}
#'   \item{CRPS}{the strictly proper score, closed form for the nested tier's
#'     own Gaussian mixture, paired seed by seed through
#'     `summary(x, baseline = )`.}
#' }
#' Every discrete PIT (a rank, a grid axis, a draw set) is randomized within its
#' atom, `u = F(theta^-) + V P(theta)`, so one uniform reference and one band
#' serve every quantity. Reading `rank / n_ref` against a continuous uniform is
#' the classic silent SBC bug.
#'
#' @section The callback contract:
#' For `experiment = "prior_predictive"`:
#' \describe{
#'   \item{`simulator(seed)`}{returns a list carrying `theta`, a named numeric
#'     vector of the true values of the scored quantities, plus whatever the
#'     fitter needs. It must be a pure function of its seed.}
#'   \item{`fitter(d)`}{returns a named list of ARMS, each a named list over
#'     quantities, each entry a predictive built by [sbc_mixture()] and its
#'     siblings. Several arms read off one solve per seed, so an arm-to-arm
#'     difference carries no fit-to-fit noise.}
#' }
#' For `experiment = "posterior"`, `model` is a list of six callbacks --
#' `data_obs`, `fit(data)`, `draw_theta(fit, seed)`, `simulate(theta, seed)`,
#' `pool(data_obs, replicate)`, `arms(fit, data)` -- plus an optional
#' `group_ids(data)` used to verify the fresh-groups premise below. The driver
#' hands `draw_theta` and `simulate` DIFFERENT seeds, so `set.seed(seed)` at the
#' top of each is the correct fixture; sharing one makes the replicate's noise a
#' function of the truth, which is not `p(y | theta')`.
#'
#' @section Two guards:
#' The prior-predictive experiment draws the truth from the prior, so an
#' IMPROPER prior cannot be used -- and the nested-Laplace door puts no prior on
#' the fixed effects. A scored quantity whose truth does not move across
#' simulations was not drawn from a proper prior, and `sbc()` errors on it
#' before spending the fits, pointing at `experiment = "posterior"`. A location
#' parameter whose flat prior leaves the PIT uniform by a structural argument is
#' admitted through `flat_prior`, which is itself checked and travels on the
#' result.
#'
#' The posterior experiment rests on two premises, each of which silently
#' invalidates the result when broken. The augmented posterior must condition on
#' BOTH data sets, so a `pool()` returning no more than the replicate (or no
#' more than the observed data) is refused. And the replicate must be
#' conditionally independent of the observed data given theta, which for a
#' hierarchical model whose group effects are integrated out means FRESH groups.
#' Supply `group_ids` and the observable half of that is verified -- the group
#' LABELS are disjoint -- and omit it and the result records the premise as
#' unverified rather than assumed. The other half, that the simulator drew those
#' groups' effects from the prior rather than conditionally on the observed
#' data, is not visible from outside the callback and is not claimed.
#'
#' @param object What to calibrate. `"prior_predictive"` (ordinary SBC) or
#'   `"posterior"` (calibration conditional on an observed data set) runs the
#'   driver against the callbacks given here; a fitted model object dispatches
#'   to that package's method, which builds the callbacks itself (see
#'   `tulpaObs::sbc.tobs_fit`).
#' @param ... Passed to methods.
#' @param simulator,fitter The prior-predictive callbacks; see the contract
#'   above. Required for `experiment = "prior_predictive"`.
#' @param model The list of posterior-SBC callbacks. Required for
#'   `experiment = "posterior"`.
#' @param n_sim Number of simulations.
#' @param quantities Optional character vector restricting which quantities are
#'   scored. Default scores every quantity every arm reports.
#' @param flat_prior Character vector of scored quantities held FIXED under a
#'   flat prior, admitted by a structural argument the caller is asserting.
#'   Prior-predictive only.
#' @param level Simultaneous band level.
#' @param seed Offset added to the simulation index, so simulation `s` runs the
#'   callbacks at `seed + s`.
#' @param n_ref Optional; when supplied it is passed to `fitter` (or to
#'   `model$arms`) as `n_ref`, the number of reference values a rank predictive
#'   is formed against. The callback must accept it.
#' @param control List of knobs: `progress` (default `FALSE`) and `rand_seed`,
#'   the pinned stream the within-atom randomizing uniforms come from (default
#'   the driver's own, so a result is reproducible and the fits are not
#'   perturbed by asking for the diagnostic).
#' @return An object of class `sbc`:
#' \describe{
#'   \item{`pit`}{one row per (simulation, arm, quantity): the truth, the raw
#'     and folded PIT, the CRPS and the predictive kind.}
#'   \item{`report`}{one row per (arm, quantity): the KS distance, whether the
#'     ECDF stayed inside the simultaneous band, the exact uniformity p-value,
#'     the same three folded, and the mean CRPS with its standard error.}
#'   \item{`bands`}{the calibrated simultaneous band, by sample size.}
#'   \item{`premises`}{what each guard concluded.}
#'   \item{`crps_role`}{the role the CRPS column has under this experiment.}
#' }
#' @references
#' Talts, Betancourt, Simpson, Vehtari & Gelman (2018). Validating Bayesian
#'   inference algorithms with simulation-based calibration. arXiv:1804.06788.
#'
#' Sailynoja, Schmitt, Buerkner & Vehtari (2026). Posterior SBC: simulation-based
#'   calibration checking conditional on data. \emph{Statistics and Computing}
#'   36:78. \doi{10.1007/s11222-026-10825-9}
#'
#' Grimit, Gneiting, Berrocal & Johnson (2006). The continuous ranked
#'   probability score for circular variables and its application to mesoscale
#'   forecast ensemble verification. \emph{QJRMS} 132(621C):2925-2942.
#' @seealso [sbc_mixture()] for the predictive shapes a fitter reports,
#'   [diagnostics()] for the single-fit reliability band that screens what this
#'   measures, [pit_residuals()] for single-fit posterior-predictive PIT
#'   residuals (a different quantity).
#' @examples
#' # A conjugate normal-normal model, whose posterior is exact, so its PIT must
#' # be uniform: the harness scoring itself.
#' sim <- function(seed) {
#'   set.seed(seed)
#'   mu <- rnorm(1)
#'   list(y = rnorm(10L, mu, 1), theta = c(mu = mu))
#' }
#' fitter <- function(d) {
#'   v <- 1 / (1 + length(d$y))
#'   list(exact  = list(mu = sbc_normal(v * sum(d$y), sqrt(v))),
#'        narrow = list(mu = sbc_normal(v * sum(d$y), sqrt(v) / 2)))
#' }
#' res <- sbc("prior_predictive", simulator = sim, fitter = fitter, n_sim = 60L)
#' res
#' summary(res, baseline = "exact")
#' @export
sbc <- function(object, ...) {
  UseMethod("sbc")
}

# Dispatch is what admits a fitted model here, so an unhandled class would
# otherwise land on R's stock "no applicable method" message, which names
# neither door.
#' @rdname sbc
#' @export
sbc.default <- function(object, ...) {
  stop("sbc() takes either the experiment name -- \"prior_predictive\" or ",
       "\"posterior\", with the callbacks that experiment needs -- or a fitted ",
       "model whose package registers a method. No sbc method is registered ",
       "for class \"", paste(class(object), collapse = "\", \""), "\".",
       call. = FALSE)
}

#' @rdname sbc
#' @export
sbc.character <- function(object = c("prior_predictive", "posterior"),
                          simulator = NULL, fitter = NULL, model = NULL,
                          n_sim = 100L, quantities = NULL,
                          flat_prior = character(), level = 0.95, seed = 0L,
                          n_ref = NULL, control = list(), ...) {
  experiment <- match.arg(object)
  ctl <- utils::modifyList(list(progress = FALSE, rand_seed = NULL),
                           as.list(control))
  n_sim <- as.integer(n_sim)
  stopifnot(length(n_sim) == 1L, n_sim >= 2L,
            length(level) == 1L, level > 0, level < 1)
  args <- list(n_seed = n_sim, quantities = quantities,
               seed_off = as.integer(seed), progress = isTRUE(ctl$progress))
  if (!is.null(ctl$rand_seed)) args$rand_seed <- ctl$rand_seed

  if (identical(experiment, "prior_predictive")) {
    if (!is.function(simulator) || !is.function(fitter)) {
      stop("experiment = \"prior_predictive\" needs simulator = and fitter =.",
           call. = FALSE)
    }
    if (!is.null(model)) {
      stop("model = is the posterior experiment's callback list; the ",
           "prior-predictive experiment takes simulator = and fitter =.",
           call. = FALSE)
    }
    fit_fn <- if (is.null(n_ref)) fitter else
      function(d) fitter(d, n_ref = n_ref)
    guard <- .sbc_check_proper_prior(simulator, fit_fn, as.integer(seed), n_sim,
                                     quantities, flat_prior)
    args$truth <- "prior_draw"
    res <- do.call(recov_sbc, c(list(simulator = simulator, fitter = fit_fn),
                                args))
    premises <- list(proper_prior = "verified",
                     flat_prior = flat_prior,
                     n_probed = guard$n_probe)
  } else {
    if (!is.null(simulator) || !is.null(fitter)) {
      stop("simulator = and fitter = belong to the prior-predictive ",
           "experiment; the posterior experiment takes model =.", call. = FALSE)
    }
    if (!is.list(model)) {
      stop("experiment = \"posterior\" needs model =, the list of callbacks.",
           call. = FALSE)
    }
    if (length(flat_prior)) {
      stop("flat_prior belongs to the prior-predictive experiment. Posterior ",
           "SBC draws every scored quantity from the posterior at the observed ",
           "data, which is proper whatever the prior was.", call. = FALSE)
    }
    if (!is.null(n_ref)) {
      base_arms <- model$arms
      model$arms <- function(fit, data) base_arms(fit, data, n_ref = n_ref)
    }
    state <- new.env(parent = emptyenv())
    state$checked <- FALSE
    state$fresh_groups <- "unreached"
    res <- do.call(recov_posterior_sbc,
                   c(list(model = .sbc_guard_model(model, state)), args))
    # Read the state rather than asserting it: a `pool()` the driver never
    # reached leaves both premises unchecked, and the result says so.
    premises <- list(pooling = if (state$checked) "verified" else "unreached",
                     fresh_groups = state$fresh_groups)
  }

  rep_tab <- sbc_report(res, level = level)
  ns <- sort(unique(rep_tab$n))
  bands <- lapply(ns, function(n) sbc_ecdf_band(n, level))
  names(bands) <- as.character(ns)

  structure(list(experiment = experiment, pit = res, report = rep_tab,
                 bands = bands, level = level, n_sim = n_sim,
                 seed = as.integer(seed), n_ref = n_ref,
                 quantities = quantities,
                 crps_role = attr(res, "crps_role"),
                 premises = premises, call = match.call()),
            class = "sbc")
}

# One verdict line over the whole result. A read outside the band is the strong
# statement; a read inside it is the weak one, so the wording does not promote
# "inside" to "calibrated".
.sbc_verdict <- function(report) {
  out <- report[!report$inside | !report$inside_folded, c("arm", "quantity")]
  if (!nrow(out)) return("no read left the band")
  sprintf("outside the band: %s",
          paste(sprintf("%s/%s", out$arm, out$quantity), collapse = ", "))
}

.sbc_report_frame <- function(report) {
  data.frame(arm = report$arm, quantity = report$quantity, n = report$n,
             ks = round(report$ks, 4), inside = report$inside,
             p_unif = signif(report$p_unif, 3),
             inside_folded = report$inside_folded,
             p_folded = signif(report$p_unif_folded, 3),
             crps = signif(report$crps, 5),
             crps_se = signif(report$crps_se, 3),
             stringsAsFactors = FALSE)
}

#' @export
print.sbc <- function(x, ...) {
  cat(sprintf("Simulation-based calibration -- %s, %d simulations\n",
              x$experiment, x$n_sim))
  cat(sprintf("  band: %.4g simultaneous (equal local levels, exact crossing probability)\n",
              x$level))
  cat("  CRPS: ", x$crps_role, "\n", sep = "")
  p <- x$premises
  if (!is.null(p$proper_prior)) {
    cat(sprintf("  proper prior: %s over %d probed simulations%s\n",
                p$proper_prior, p$n_probed,
                if (length(p$flat_prior))
                  sprintf("; flat prior asserted for %s",
                          paste(p$flat_prior, collapse = ", ")) else ""))
  }
  if (!is.null(p$pooling)) {
    cat(sprintf("  pooling: %s; fresh groups: %s\n", p$pooling, p$fresh_groups))
    if (identical(p$fresh_groups, "undeclared")) {
      cat("    supply model$group_ids to have the fresh-groups premise checked\n")
    }
  }
  cat("\n")
  print(.sbc_report_frame(x$report), row.names = FALSE)
  cat("\n", sprintf("%d of %d (arm, quantity) reads inside the band; %s.\n",
                    sum(x$report$inside), nrow(x$report),
                    .sbc_verdict(x$report)), sep = "")
  invisible(x)
}

#' @param object An `sbc` result.
#' @param baseline Optional arm name. When given, the summary also carries the
#'   PAIRED CRPS differences of every other arm against it, seed by seed. A
#'   negative `delta` is the arm scoring better. This is refused on an
#'   experiment where the CRPS is not a proper posterior score.
#' @param ... Ignored.
#' @rdname sbc
#' @export
summary.sbc <- function(object, baseline = NULL, ...) {
  out <- list(experiment = object$experiment, n_sim = object$n_sim,
              level = object$level, report = object$report,
              crps_role = object$crps_role, premises = object$premises)
  if (!is.null(baseline)) {
    if (!baseline %in% object$report$arm) {
      stop("baseline arm ", sQuote(baseline), " is not one of: ",
           paste(sQuote(unique(object$report$arm)), collapse = ", "),
           call. = FALSE)
    }
    out$compare <- sbc_crps_compare(object$pit, baseline)
    out$baseline <- baseline
  }
  structure(out, class = "sbc_summary")
}

#' @export
print.sbc_summary <- function(x, ...) {
  cat(sprintf("Simulation-based calibration -- %s, %d simulations\n",
              x$experiment, x$n_sim))
  cat("  CRPS: ", x$crps_role, "\n\n", sep = "")
  print(.sbc_report_frame(x$report), row.names = FALSE)
  if (!is.null(x$compare)) {
    cat(sprintf("\nPaired CRPS against arm %s (negative delta = better)\n",
                sQuote(x$baseline)))
    cp <- x$compare
    print(data.frame(quantity = cp$quantity, arm = cp$arm, n = cp$n,
                     delta = signif(cp$delta, 4), t = round(cp$t, 2),
                     worse_frac = round(cp$worse_frac, 3),
                     p_sign = signif(cp$p_sign, 3),
                     stringsAsFactors = FALSE), row.names = FALSE)
    cat("  delta with its t is the proper-score verdict; p_sign is a more\n",
        "  powerful detector of a difference but is not itself proper.\n",
        sep = "")
  }
  invisible(x)
}

#' @param x An `sbc` result.
#' @param arm,quantity Optional character vectors selecting which panels to
#'   draw. Default draws every (arm, quantity).
#' @param folded Plot the folded PIT instead of the raw one.
# The p-value and band verdict annotating one panel. It has to read the SAME
# sample the panel draws: the fold is the read that catches a symmetric
# dispersion error the raw ECDF cancels, so a folded panel carrying the raw
# verdict reports "inside" over a curve visibly outside the band, which is
# exactly the case the fold exists for.
.sbc_panel_note <- function(r, i, folded) {
  p  <- if (folded) r$p_unif_folded[i] else r$p_unif[i]
  ok <- if (folded) r$inside_folded[i] else r$inside[i]
  sprintf("p = %.3g%s", p, if (ok) "" else ", outside band")
}

#' @rdname sbc
#' @export
plot.sbc <- function(x, arm = NULL, quantity = NULL, folded = FALSE, ...) {
  r <- x$report
  if (!is.null(arm)) r <- r[r$arm %in% arm, , drop = FALSE]
  if (!is.null(quantity)) r <- r[r$quantity %in% quantity, , drop = FALSE]
  if (!nrow(r)) stop("no (arm, quantity) selected", call. = FALSE)
  op <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(op), add = TRUE)
  nr <- ceiling(sqrt(nrow(r)))
  nc <- ceiling(nrow(r) / nr)
  graphics::par(mfrow = c(nr, nc), mar = c(4, 4, 3.6, 1))
  for (i in seq_len(nrow(r))) {
    sel <- x$pit$arm == r$arm[i] & x$pit$quantity == r$quantity[i]
    u <- x$pit$pit[sel]
    if (folded) u <- sbc_fold(u)
    n <- length(u)
    band <- x$bands[[as.character(n)]]
    if (is.null(band)) band <- sbc_ecdf_band(n, x$level)
    at <- band$ecdf_at
    us <- sort(u)
    # The ECDF DIFFERENCE, which is the standard SBC read: a calibrated sample
    # wanders around zero inside the band.
    graphics::plot(range(at), range(c(band$ecdf_lo - at, band$ecdf_hi - at)),
                   type = "n", xlab = "PIT", ylab = "ECDF - uniform")
    graphics::title(main = sprintf("%s / %s%s", r$arm[i], r$quantity[i],
                                   if (folded) " (folded)" else ""),
                    line = 1.9, cex.main = 1)
    graphics::mtext(.sbc_panel_note(r, i, folded),
                    side = 3, line = 0.5, cex = 0.75)
    graphics::polygon(c(at, rev(at)),
                      c(band$ecdf_lo - at, rev(band$ecdf_hi - at)),
                      col = grDevices::grey(0.9), border = NA)
    graphics::abline(h = 0, col = "grey40")
    graphics::lines(c(0, us, 1), c(0, seq_len(n) / n, 1) - c(0, us, 1),
                    type = "s")
  }
  invisible(x)
}

#' @rdname diagnostics
#' @export
diagnostics.sbc <- function(fit, ...) {
  out <- fit$report
  attr(out, "experiment") <- fit$experiment
  attr(out, "crps_role") <- fit$crps_role
  attr(out, "band_level") <- fit$level
  attr(out, "verdict") <- .sbc_verdict(fit$report)
  out
}
