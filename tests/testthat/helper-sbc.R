# helper-sbc.R
#
# Simulation-based calibration (SBC) and a strictly proper score, as posterior
# arbiters alongside the fixed-truth recovery sweeps in
# test-nested-laplace-recovery.R (gcol33/tulpa#335).
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
# SOURCING. Sections 1 to 5 define functions over base + stats alone, so
# `source("tests/testthat/helper-sbc.R")` works from a dev_notes script as well
# as from testthat. Section 6 is the engine fixture and is the only part that
# calls tulpa, and only when invoked.

# ---------------------------------------------------------------------------
# 1. Predictive distributions
#
# One small tagged representation per shape a backend can report. Everything
# downstream (PIT, CRPS, sampling) dispatches on `kind`, so a new backend shape
# is one entry in three switches rather than a parallel scorer.
# ---------------------------------------------------------------------------

# A Gaussian mixture, which is what an outer hyperparameter grid defines for a
# fixed effect: component k is N(mu_k, var_k) with weight w_k.
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
sbc_normal <- function(mean, sd) sbc_mixture(mean, sd^2, 1)

# A distribution on a finite support, which is what a discrete hyperparameter
# grid defines for its own axis.
sbc_discrete <- function(support, probs) {
  support <- as.numeric(support); probs <- as.numeric(probs) / sum(probs)
  o <- order(support)
  list(kind = "discrete", support = support[o], probs = probs[o])
}

# A rank r in {0, ..., n_ref} of the truth among n_ref reference values, which
# is what a joint log-likelihood comparison against posterior draws produces.
sbc_rank <- function(rank, n_ref) {
  list(kind = "rank", rank = as.integer(rank), n_ref = as.integer(n_ref))
}

# Posterior draws, for a backend that reports no analytic marginal.
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
# is a descriptive loss and the PIT is not uniform under correct inference.
# ---------------------------------------------------------------------------

recov_sbc <- function(simulator, fitter, n_seed, quantities = NULL,
                      seed_off = 0L, truth = c("prior_draw", "fixed"),
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
  attr(res, "crps_role") <- if (identical(truth, "prior_draw"))
    "proper posterior score" else "descriptive loss (fixed truth)"
  res
}

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
  if (!identical(attr(res, "truth"), "prior_draw")) {
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
# 6. The engine fixture
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
SBC_PHI  <- 0.7
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
  y <- as.numeric(X %*% beta) + u[region] + stats::rnorm(N, 0, phi)
  list(seed = seed, y = y, X = X, region = as.integer(region), N = N, nr = nr,
       spr = spr, phi = phi, grid = grid,
       theta = c(beta1 = beta[1], beta2 = beta[2], sigma = sigma))
}

# Exact per-cell quantities for the balanced design. V = sigma^2 Z Z' + phi^2 I
# is block diagonal with blocks phi^2 I_m + sigma^2 J_m, so V^-1 = a I - b J per
# block and every quadratic form is a group sum.
.sbc_exact_cell <- function(d, sigma, phi = d$phi) {
  m <- d$spr
  a <- 1 / phi^2
  b <- sigma^2 / (phi^2 * (phi^2 + m * sigma^2))
  X <- d$X; y <- d$y
  Xs <- rowsum(X, d$region)
  ys <- as.numeric(rowsum(y, d$region))
  XtVX <- a * crossprod(X) - b * crossprod(Xs)
  XtVy <- a * as.numeric(crossprod(X, y)) - b * as.numeric(crossprod(Xs, ys))
  ytVy <- a * sum(y^2) - b * sum(ys^2)
  Vb <- solve(XtVX)
  bh <- as.numeric(Vb %*% XtVy)
  G <- nrow(Xs); N <- length(y); p <- ncol(X)
  logdetV <- G * ((m - 1) * log(phi^2) + log(phi^2 + m * sigma^2))
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
       w = w, log_marg = lm)
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

sbc_fit_nested <- function(d, phi = d$phi) {
  suppressWarnings(tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, d$N), X = d$X,
    prior = list(list(type = "iid", obs_idx = d$region,
                      n_units = d$nr, sigma_grid = d$grid)),
    family = "gaussian", phi = phi,
    control = list(max_iter = 200L, tol = 1e-10, n_threads = 1L,
                   keep_grid_hessians = TRUE, diagnose_k = FALSE,
                   diagnose_skew = FALSE, auto_recenter = FALSE)))
}

# The joint log-likelihood rank: draw `n_ref` (beta, sigma) pairs from the
# mixture an arm describes -- a cell by its weight, then that cell's Gaussian
# for beta -- and rank the truth's log-likelihood among them. It reads the whole
# data set at once, so it catches an approximation that gets each coefficient's
# marginal right while getting their joint dependence, or the hyperparameter,
# wrong, which per-coefficient PITs cannot see. The reference draws come from a
# stream pinned to the data seed, so the rank is reproducible.
sbc_loglik_rank <- function(d, mu, cov, w, phi = d$phi, n_ref = 200L,
                            seed = d$seed) {
  ll <- .sbc_with_seed(seed + 883L, {
    k <- sample.int(length(w), n_ref, replace = TRUE, prob = w)
    vapply(seq_len(n_ref), function(i) {
      L <- chol(cov[[k[i]]])
      b <- mu[k[i], ] + as.numeric(crossprod(L, stats::rnorm(ncol(mu))))
      sbc_loglik(d, b, d$grid[k[i]], phi)
    }, numeric(1))
  })
  ll0 <- sbc_loglik(d, unname(d$theta[c("beta1", "beta2")]),
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
#   phi_crossed  a SECOND solve at phi^2 where the door reads a residual SD --
#                the gcol33/tulpa#332 generator / inference convention crossing
# `wide`, `narrow` and `phi_crossed` are the known-bad controls: a calibration
# harness that cannot fail is worthless, so a deliberately mis-scaled posterior
# has to land outside the band.
sbc_arms_gaussian <- function(d, bad_factor = 1.25, n_ref = 200L,
                              phi_crossed = TRUE) {
  f <- sbc_fit_nested(d)
  mom <- .nested_fixed_moments(f)
  E <- sbc_exact_post(d)
  mk <- function(mu, var, w) list(beta1 = sbc_mixture(mu[, 1], var[, 1], w),
                                  beta2 = sbc_mixture(mu[, 2], var[, 2], w))
  se <- sqrt(pmax(diag(mom$cov), 0))
  coll <- function(mult) list(
    beta1 = sbc_normal(mom$mean[1], mult * se[1]),
    beta2 = sbc_normal(mom$mean[2], mult * se[2]))
  cellcov <- function(fit) lapply(seq_along(fit$weights),
                                  function(k) solve(fit$grid_hessians[[k]]))
  arms <- list(
    exact     = mk(E$mu, E$var, E$w),
    mixture   = mk(mom$mu, mom$var, mom$w),
    collapsed = coll(1),
    wide      = coll(bad_factor),
    narrow    = coll(1 / bad_factor))
  arms$exact$sigma   <- sbc_discrete(d$grid, E$w)
  arms$mixture$sigma <- sbc_discrete(d$grid, mom$w)
  arms$exact$log_lik   <- sbc_loglik_rank(d, E$mu, E$cov, E$w, n_ref = n_ref)
  arms$mixture$log_lik <- sbc_loglik_rank(d, mom$mu, cellcov(f), mom$w,
                                          n_ref = n_ref)
  if (phi_crossed) {
    fx <- sbc_fit_nested(d, phi = d$phi^2)
    mx <- .nested_fixed_moments(fx)
    arms$phi_crossed <- mk(mx$mu, mx$var, mx$w)
    arms$phi_crossed$sigma <- sbc_discrete(d$grid, mx$w)
    arms$phi_crossed$log_lik <- sbc_loglik_rank(d, mx$mu, cellcov(fx), mx$w,
                                                phi = d$phi^2, n_ref = n_ref)
  }
  arms
}
