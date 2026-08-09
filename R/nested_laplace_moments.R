# Posterior-moment machinery for the nested-Laplace fits: weighted mean / SD /
# median / CI over the outer hyperparameter grid, per-axis weighted quantiles,
# the axis-marginal Laplace-refined SD, and the multi-block joint / per-block
# marginal moments. Consumed by the nested-Laplace drivers in nested_laplace.R.

# Compute weighted theta_mean / theta_sd / theta_median / theta_ci_lo /
# theta_ci_hi from grid + weights. The mean and SD are produced via the
# usual `sum(w * x)` / `sum(w * x^2)` moments; the median and 2.5/97.5
# CI are produced via `.nl_axis_quantiles` so that asymmetric scale-like
# hyperparameters (sigma, alpha, range, phi) have a calibrated headline
# summary alongside mean +/- SD. Median is the recommended summary for
# right-skewed posteriors: `mean` of a weakly-identified positive ratio
# is pulled up by the right tail; `median` matches truth at small n.
.nl_posterior_moments <- function(res, type) {
  w <- res$weights
  tg <- res$theta_grid
  if (is.matrix(tg)) {
    res$theta_mean <- as.numeric(crossprod(w, tg))
    names(res$theta_mean) <- colnames(tg)
    res$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                                  res$theta_mean^2))
    names(res$theta_sd) <- colnames(tg)
  } else {
    ms <- .nl_wtd_mean_sd(tg, w)
    res$theta_mean <- ms$mean
    res$theta_sd   <- ms$sd
  }
  qs <- .nl_axis_quantiles(tg, res$log_marginal, res$refining_axis)
  res$theta_median <- qs$median
  res$theta_ci_lo  <- qs$ci_lo
  res$theta_ci_hi  <- qs$ci_hi
  res
}

# Marginal log-density along a single hyperparameter axis (logsumexp over
# the other-axis cells at each unique value). `vals` and `log_marg` are
# length n_cells; `keep` is an optional logical mask (cartesian + same-
# axis slice cells). Returns sorted unique axis values and the matching
# marginal log-density.
# Weighted quantile on a discrete (value, weight) distribution. Uses
# midpoint-of-mass cumulative probability (Type 7-like) plus linear
# interpolation, so quantiles vary smoothly with weights rather than
# snapping to grid cells.
#
#  * Aggregates duplicate values: weights at equal `values` are summed
#    before interpolation. Idempotent on already-unique axes; needed when
#    the joint grid carries repeated values (e.g. slice-cell refinement
#    re-uses the modal axis value across multiple Newton-Laplace cells).
#  * Filters non-finite values and non-positive weights.
#  * Returns `NA` per requested `probs` when the support is empty;
#    returns the unique support value when only one survives.
#
# Used by `.alpha_grid_moments` to surface the posterior median and 95%
# interval of `alpha` (a direct hyperparameter axis after the
# (sigma, alpha) reparameterization) from the joint nested-Laplace
# posterior. Weighted quantiles handle skewed/heavy-tailed marginals that
# `mean +/- 1.96 sd` summaries misrepresent.
#
# `outside` is the policy for a probability beyond the support's own cumulative
# range, `[p[1], p[n]]`, which the mid-mass convention leaves the outer half-cell
# of mass sitting in.
#
# `"clamp"` returns the extreme value, the convention a SAMPLE takes: beyond the
# extreme order statistic nothing is known about the tail, so the order statistic
# is the answer.
#
# `"extend"` is the convention a CELL PARTITION takes. A tensor grid's values are
# not order statistics, they are cell representatives: each carries the mass of
# an interval it sits at the centre of, so the extreme cell's mass reaches half a
# spacing past its coordinate and the support's edge is the design's own
# geometry rather than an unknown tail. Clamping there reports an interval the
# read cannot place its own outer half-cell of mass inside -- measured, on a
# prior-predictive experiment whose grid tiles the prior, as a PIT atom at 0 and
# 1 of exactly the outer half-cell mass (gcol33/tulpa#353). The half-spacing is
# mirrored in `log` when every value is positive, which is where a scale grid is
# equally spaced and where the edge cannot cross zero, and in the value itself
# otherwise -- the same `all(vals > 0)` test `.nl_laplace_at_mode_sd_axis()`
# picks its own coordinate with. Nothing inside `[values[1], values[n]]` moves:
# the interpolant, its knots and every probability the clamp did not bind at are
# unchanged.
#
# On weights that are a QUADRATURE DESIGN (`ccd_weights()`) the cumulative sum is
# not a CDF at all, and clamping reports the design's own extent as a posterior
# interval (gcol33/tulpa#308); `"na"` withholds the number instead. Such a
# support is summarized by `.nl_moment_quantile()`, which uses the moments the
# design does deliver.
.nl_wtd_quantile <- function(values, weights, probs,
                             outside = c("clamp", "extend", "na")) {
  outside <- match.arg(outside)
  ord <- order(values)
  v <- as.numeric(values)[ord]
  w <- as.numeric(weights)[ord]
  keep <- is.finite(v) & is.finite(w) & w > 0
  if (!any(keep)) return(rep(NA_real_, length(probs)))
  v <- v[keep]; w <- w[keep]
  # Aggregate runs of strictly-equal adjacent values (already sorted).
  # Cannot use factor(v) here: distinct doubles that share an
  # `as.character` print form (e.g. 0.4/0.7 and a near-equal ratio off
  # by ~1e-16) trigger "factor level [k] is duplicated". Group by
  # integer run-IDs derived from numeric equality on the sorted vector.
  if (length(v) > 1L) {
    is_first <- c(TRUE, v[-1L] != v[-length(v)])
    if (!all(is_first)) {
      grp <- cumsum(is_first)
      w   <- as.numeric(tapply(w, grp, sum))
      v   <- v[is_first]
    }
  }
  w_tot <- sum(w)
  if (!is.finite(w_tot) || w_tot <= 0) return(rep(NA_real_, length(probs)))
  w <- w / w_tot
  if (length(v) == 1L) return(rep(v[1L], length(probs)))
  p <- cumsum(w) - w / 2
  # The outer half-cells are two more knots: mass 0 at the lower edge and the
  # whole mass at the upper one, so the same interpolant carries them and the
  # interior knots keep their positions.
  if (identical(outside, "extend")) {
    e <- .nl_cell_edges(v)
    p <- c(0, p, 1)
    v <- c(e[1L], v, e[2L])
  }
  n <- length(v)
  # `approx` emits "collapsing to unique 'x' values" when tiny floor
  # weights against a dominant prefix sum produce numerically equal
  # midpoints. The collapse (tied p -> mean of v) is the right behavior
  # at that resolution -- the underlying mass between the tied cells is
  # zero up to floating-point error. Suppress the noise.
  na_outside <- identical(outside, "na")
  vapply(probs, function(q) {
    if (q <= p[1L]) return(if (na_outside) NA_real_ else v[1L])
    if (q >= p[n])  return(if (na_outside) NA_real_ else v[n])
    suppressWarnings(approx(p, v, xout = q, method = "linear")$y)
  }, numeric(1L))
}

# The outer edges of the cell partition a sorted vector of cell coordinates
# represents: the extreme cell mirrors the half-spacing it has. Mirrored in log
# when every coordinate is positive -- a scale grid is equally spaced there, and
# the edge stays positive -- and in the value itself otherwise.
.nl_cell_edges <- function(v) {
  n <- length(v)
  if (n < 2L) return(c(v[1L], v[n]))
  if (all(v > 0)) {
    u <- log(v)
    e <- c(exp(u[1L] - 0.5 * (u[2L] - u[1L])),
           exp(u[n] + 0.5 * (u[n] - u[n - 1L])))
    if (all(is.finite(e))) return(e)
  }
  c(v[1L] - 0.5 * (v[2L] - v[1L]), v[n] + 0.5 * (v[n] - v[n - 1L]))
}

# Monotone maps between a derived quantity's own domain and the unbounded
# coordinate a moment-matched interval is formed on. `positive` covers standard
# deviations, variances, precisions and ranges, `correlation` the (-1, 1)
# interval of a correlation, `unit` the (0, 1) interval of a mixing weight
# (BYM2's rho, a probability), and `unbounded` a covariance entry or a copy
# coefficient, whose sign is free. A registry rather than a branch, so a new
# derived quantity names its domain and inherits the interval.
#
# `in_domain` is the map's own support, checked before it is applied so a
# degenerate node (a collapsed scale, a correlation on the boundary) is reported
# as unsummarizable rather than reaching `log` / `atanh` as a NaN.
.NL_DOMAIN_TRANSFORM <- list(
  positive    = list(to = log,      from = exp,
                     in_domain = function(x) x > 0),
  correlation = list(to = atanh,    from = tanh,
                     in_domain = function(x) abs(x) < 1),
  unit        = list(to = stats::qlogis, from = stats::plogis,
                     in_domain = function(x) x > 0 & x < 1),
  unbounded   = list(to = identity, from = identity,
                     in_domain = function(x) rep(TRUE, length(x)))
)

# Median and interval of a quantity evaluated on a QUADRATURE DESIGN rather than
# sampled from the posterior.
#
# A central-composite design (`ccd_grid()` + `ccd_weights()`) is a moment rule:
# its nodes sit where they reproduce the first two moments of the integrand, and
# their positions carry no probability mass of their own. So the cumulative
# design weight across them is not a CDF, and a discrete weighted quantile over
# them returns an interval bounded by the design's own extent -- at k
# parameters, `theta_hat +/- 1.1 sqrt(k) sd`, whose Gaussian coverage is
# `2 Phi(1.1 sqrt(k)) - 1` no matter how much data there is (gcol33/tulpa#308).
#
# The moments ARE delivered, so the interval comes from them: the first two
# weighted moments on the domain's unbounded coordinate define a Gaussian there
# and its quantiles are mapped back. On a positive quantity that is a lognormal
# interval -- positive, asymmetric, and free to leave the node range -- and on a
# correlation it stays inside (-1, 1). At `probs = 0.5` the Gaussian quantile is
# its mean, so the median is the back-transformed first moment.
#
# Returns NA for every requested probability when no node carries usable weight,
# or when any weighted node falls outside the domain: dropping such a node would
# evaluate the moment rule on a design other than the one that was integrated.
.nl_moment_quantile <- function(values, weights, probs, domain = "unbounded") {
  tr <- .NL_DOMAIN_TRANSFORM[[domain]]
  if (is.null(tr)) {
    stop("unknown derived-quantity domain '", domain, "'.", call. = FALSE)
  }
  use <- is.finite(weights) & weights > 0 & is.finite(values)
  if (!any(use)) return(rep(NA_real_, length(probs)))
  v <- as.numeric(values)[use]
  if (!all(tr$in_domain(v))) return(rep(NA_real_, length(probs)))
  u <- tr$to(v)
  if (!all(is.finite(u))) return(rep(NA_real_, length(probs)))
  w <- as.numeric(weights)[use]
  w <- w / sum(w)
  m <- sum(w * u)
  s <- sqrt(max(0, sum(w * u^2) - m^2))
  tr$from(m + stats::qnorm(probs) * s)
}

# The node-set KINDS a summary can be taken off, and the outer-edge policy each
# one's geometry implies. One table rather than a chain of branches, so a kind
# is named in exactly one place and a caller reading the tag can tell which
# geometry it has (gcol33/tulpa#358).
#
# `density` -- a tensor grid. The weights are proportional to posterior mass and
# the values are CELL REPRESENTATIVES of a partition with known spacing, so the
# cumulative sum is a CDF and the extreme cell's mass reaches half a spacing past
# its coordinate: the read runs to the outer cells' own edges
# (`outside = "extend"`, gcol33/tulpa#353).
#
# `sample` -- equal-weight posterior DRAWS, what `tulpa_re_cov_gibbs()`'s sweep
# produces. The cumulative sum is a CDF here too, so the interior read is the
# same weighted quantile, but the values are ORDER STATISTICS rather than cell
# representatives: beyond the largest draw nothing is known about the tail, and
# half the gap between the two extreme draws is not a cell width. So the outer
# edge CLAMPS, the convention a sample takes. This is the distinction
# gcol33/tulpa#358 separates out; it binds only a probability outside
# `[1 / (2 n), 1 - 1 / (2 n)]`, which is why nothing measured moves at the
# 0.025 / 0.5 / 0.975 the backends report.
#
# `moment_rule` -- a central-composite design. Its weights reproduce the
# integrand's moments and the node positions carry no mass, so a cumulative sum
# is not a CDF at all and the interval comes from the moments on the quantity's
# own `domain` (gcol33/tulpa#308). It carries no `outside` policy because it
# never reaches the quantile read. A `moment_rule` quantity whose domain is NA
# has a support the engine will not guess -- a proper-CAR correlation on the
# adjacency eigenvalue interval -- and reports NA rather than the design's
# extent.
#
# `mixed` -- see below; it takes `density`'s policy on purpose.
.NL_SUPPORT <- list(
  density     = list(outside = "extend"),
  moment_rule = list(outside = NA_character_),
  mixed       = list(outside = "extend"),
  sample      = list(outside = "clamp")
)

.NL_SUPPORT_KINDS <- names(.NL_SUPPORT)

# Median and interval of one quantity, given what KIND of node set carries it.
#
# `support = "mixed"` is the locally CCD-refined grid, the one node set that is
# part cell masses and part quadrature design (gcol33/tulpa#311): the carried-over
# base cells hold their own mass, the refined cells' replacement clouds hold a
# partition-of-unity share of theirs placed at the design's radius. On the design
# part a cumulative sum is not a CDF, so the quantile there reads closer to the
# design's own per-axis extent than to a posterior property (gcol33/tulpa#317).
# It still takes the weighted quantile, because that is what measured best:
# scored against the converged m = 13 tensor reference on the four-axis
# multi-block fixture (noise floor 0.01716 on the endpoints, 0.03853 on the
# widths), summed absolute endpoint error over seven base grids is 0.63446 for
# the quantile against 0.73159 for collapsing each design block to its mean,
# 0.74464 for splitting the read into a mass CDF plus a per-cell moment-matched
# Gaussian, and 1.20402 for the #308 moment read; on analytic outer targets whose
# axis quantiles are known in closed form the same ordering holds in the
# design-dominated regime. So the value of naming the support is that the fit can
# SAY it is mixed and how much of the weight is design (`theta_interval_read` /
# `theta_interval_design_mass`), not that a different formula replaces it. That
# is why `"mixed"` takes the SAME `outside` policy as `"density"` and not the
# conservative one its design nodes would argue for: the tag records provenance,
# and a refined fit reading its interval off a different construction from the
# unrefined fit of the same model is the defect gcol33/tulpa#317 named.
#
# The single dispatcher for every consumer of the summaries, so a caller names
# its support and its domain and inherits the rest.
.nl_summary_quantile <- function(values, weights, probs,
                                 domain = NA_character_,
                                 support = .NL_SUPPORT_KINDS) {
  support <- match.arg(support)
  outside <- .NL_SUPPORT[[support]]$outside
  if (!is.na(outside)) {
    return(.nl_wtd_quantile(values, weights, probs, outside = outside))
  }
  if (length(domain) != 1L || is.na(domain)) {
    return(.nl_wtd_quantile(values, weights, probs, outside = "na"))
  }
  .nl_moment_quantile(values, weights, probs, domain)
}

# What kind of node set the producer left behind. `integration` names what RAN,
# which describes a HOMOGENEOUS support: the central-composite design is a moment
# rule, a Gibbs sweep leaves draws, and the tensor grid and its adaptive subset
# discretize the density. A locally CCD-refined grid carries both kinds at once
# and reports `"grid"`, so the per-cell `weight_kind` tag decides ahead of the
# producer name (gcol33/tulpa#317) -- otherwise a mixed support is read as a
# homogeneous density one and nothing downstream can tell.
#
# `"sample"` is the same distinction one layer up from `.NL_SUPPORT`: a sampler
# and a grid both leave a (value, weight) set whose cumulative sum is a CDF, and
# only the producer knows whether the values are order statistics or cell
# representatives, so the producer names itself and the read follows
# (gcol33/tulpa#358).
.nl_node_support <- function(integration, weight_kind = NULL) {
  if (length(weight_kind) > 1L) {
    kinds <- unique(weight_kind[!is.na(weight_kind)])
    if (length(kinds) > 1L) return("mixed")
  }
  switch(integration %||% "grid",
         ccd    = "moment_rule",
         sample = "sample",
         "density")
}

# What the reported per-axis intervals were read off, and the share of the
# integration weight sitting on nodes whose cumulative sum is not a CDF.
#
# `design_mass` is 0 on a pure density grid and on a sample, 1 on a global CCD,
# and the refined cells' share on a mixed one. It is the regime variable for the
# mixed read: the larger it is, the more of the reported interval comes from a
# moment rule being read as a CDF. Returned together so every consumer surfaces
# the same pair.
.nl_interval_provenance <- function(integration, weight_kind = NULL,
                                    weights = NULL) {
  read <- .nl_node_support(integration, weight_kind)
  dm <- if (identical(read, "moment_rule")) {
    1
  } else if (identical(read, "mixed") && !is.null(weights) &&
             length(weights) == length(weight_kind)) {
    sum(weights[weight_kind == "design"], na.rm = TRUE)
  } else if (identical(read, "mixed")) {
    NA_real_
  } else {
    0
  }
  list(read = read, design_mass = dm)
}

# Summary of a weighted mixture of Gaussians, one mixture per parameter.
#
# The continuous counterpart of `.nl_wtd_quantile`: where that summarizes a
# discrete (value, weight) set -- one number per grid cell -- this summarizes a
# posterior whose cells each contribute a whole Gaussian,
#   p(x_j) = sum_i w_i N(x_j; mu_ij, var_ij),
# which is what a nested-Laplace grid actually produces for a latent coefficient
# (per-node conditional mean and curvature, mixed by the node weights). Reporting
# only the spread BETWEEN nodes omits the within-node curvature and understates
# the posterior SD; reporting only one node's Gaussian omits the hyperparameter
# uncertainty the grid exists to carry. Both terms enter here.
#
# `mu` and `var` are n_node x n_par (a column is one parameter, a row one node);
# `w` is the node weight vector. Mean and SD are the exact mixture moments (law
# of total variance), and the quantiles invert the exact mixture CDF by bisection
# rather than assuming normality -- a mixture over a skewed hyperparameter
# posterior is itself skewed, so `mean +/- 1.96 sd` is not its interval. The
# bracket spans +/- 10 component SDs, which contains every quantile of the
# mixture for any `probs` in practice.
#
# Returns list(mean, sd, quantiles = n_par x length(probs)), or NULL when no node
# is usable. `var = NULL`, or variances that are missing / negative / non-finite,
# give the mixture mean with NA for SD and the quantiles: the between-node spread
# alone is not the posterior SD, so it is withheld rather than reported small.
.nl_gauss_mixture_summary <- function(mu, var, w, probs = c(0.025, 0.5, 0.975),
                                      n_bisect = 80L) {
  mu <- as.matrix(mu)
  w  <- as.numeric(w)
  np <- ncol(mu)
  ok <- is.finite(w) & w > 0 & is.finite(rowSums(mu))
  if (!any(ok) || np == 0L) return(NULL)
  mu <- mu[ok, , drop = FALSE]
  w  <- w[ok]; w <- w / sum(w)
  mean_p <- as.numeric(crossprod(w, mu))

  sdm <- NULL
  if (!is.null(var)) {
    v <- as.matrix(var)[ok, , drop = FALSE]
    if (all(is.finite(v)) && all(v >= 0)) sdm <- sqrt(v)
  }
  na_q <- matrix(NA_real_, np, length(probs))
  if (is.null(sdm)) {
    return(list(mean = mean_p, sd = rep(NA_real_, np), quantiles = na_q))
  }

  sd_p <- sqrt(pmax(as.numeric(crossprod(w, mu^2 + sdm^2)) - mean_p^2, 0))

  # A degenerate component (zero curvature-implied SD) is a step function in the
  # CDF; the floor keeps it one without branching the vectorized evaluation.
  s <- pmax(sdm, 1e-300)
  cdf <- function(q) {
    as.numeric(crossprod(w, stats::pnorm(
      matrix(q, nrow(mu), np, byrow = TRUE), mu, s)))
  }
  lo0 <- apply(mu - 10 * sdm, 2L, min)
  hi0 <- apply(mu + 10 * sdm, 2L, max)
  qs <- vapply(probs, function(p) {
    lo <- lo0; hi <- hi0
    for (it in seq_len(n_bisect)) {
      mid <- (lo + hi) / 2
      below <- cdf(mid) < p
      lo <- ifelse(below, mid, lo)
      hi <- ifelse(below, hi, mid)
    }
    (lo + hi) / 2
  }, numeric(np))
  list(mean = mean_p, sd = sd_p,
       quantiles = matrix(qs, np, length(probs)))
}

# Fixed-effect credible bounds on a nested-Laplace fit, and the provenance of
# whichever read produced them (gcol33/tulpa#336).
#
# The grid defines a Gaussian mixture per coefficient (see
# `.nested_fixed_moments()`). Its mean and variance are linear functionals and
# survive the collapse to one Gaussian; a quantile is nonlinear and does not, so
# the bounds invert the mixture CDF
#   F_j(b) = sum_k w_k Phi((b - mu_kj) / sqrt(V_kjj))
# through `.nl_gauss_mixture_summary()` -- the same construction `ranef()`
# already reports the per-group posterior with. `estimate`, `std.error` and
# `vcov()` are untouched: they are the moments either way.
#
# WHY THE #302 SKEW CORRECTION IS NOT COMPOSED WITH THIS. Its gamma_3 is
# computed by re-dispatching the kernel at the single fitted MAP cell, so the
# fit retains one gamma_3 per coefficient and not one per cell. The composed
# marginal sum_k w_k F^CF_kj is therefore not identified by retained state: it
# would need gamma_3(k, j). The three available substitutes are each an
# unbacked assertion -- a scalar Cornish-Fisher shift of the mixture quantile
# double-counts, since the mixture already carries the across-cell asymmetry;
# applying the MAP cell's gamma_3 to every component asserts the conditional
# skew is constant over the grid; applying it to the dominant component alone
# privileges one component with no approximation theorem behind it. So an
# enabled correction keeps the read it was measured on, and `declined` says why
# the mixture read did not run. The two corrections address different
# non-Gaussianities: this one across cells, #302 within the MAP cell.
#
# A GRID THAT DROPPED A POSITIVE-WEIGHT CELL still gets the mixture read. The
# moments renormalize over the cells that retained a block, so the components
# here carry the same weighting the reported `estimate` and `std.error` were
# formed under and the two reads describe one posterior -- the posterior
# CONDITIONAL ON THE RETAINED CELLS, not the full grid's, which the dropped mass
# is gone from either way. `mass` travels with the result and is the original
# retained share of the grid weight, so a caller reading the bounds can always
# tell a complete grid from a repaired one (gcol33/tulpa#342).
#
# `mom` is the `.nested_fixed_moments()` list, `idx` the reported coefficient
# columns, `est` / `se` their moments, `probs` the requested levels, `sc` the
# `.nl_skew_correction()` state. Returns list(q = length(est) x length(probs),
# applied, source, declined, mass).
.nl_fixed_interval <- function(mom, idx, est, se, probs, sc) {
  mass <- if (is.numeric(mom$mass) && length(mom$mass) == 1L) {
    as.numeric(mom$mass)
  } else NA_real_
  gaussian <- function(source, declined) {
    list(q = est + outer(se, stats::qnorm(probs)),
         applied = rep(FALSE, length(est)),
         source = source, declined = declined, mass = mass)
  }
  if (isTRUE(sc$enabled)) {
    mg <- .nl_skew_marginal(est, se, .nl_skew_gamma3_eligible(sc)[idx],
                            .nl_skew_gamma1_eligible(sc)[idx], probs,
                            enabled = TRUE)
    return(list(
      q = mg$q, applied = mg$applied, source = "skew_map_cell",
      declined = paste("skew_correct: gamma_3 is retained at the MAP cell",
                       "only, so the mixture components carry no per-cell",
                       "skew to compose"),
      mass = mass))
  }
  if (is.null(mom$mu) || is.null(mom$var) || is.null(mom$w)) {
    return(gaussian("gaussian_moment", "no retained mixture components"))
  }
  if (ncol(mom$mu) < max(idx)) {
    return(gaussian("gaussian_moment",
                    "retained component block is narrower than the reported coefficients"))
  }
  mx <- .nl_gauss_mixture_summary(mom$mu[, idx, drop = FALSE],
                                  mom$var[, idx, drop = FALSE],
                                  mom$w, probs = probs)
  if (is.null(mx) || anyNA(mx$quantiles)) {
    return(gaussian("gaussian_moment",
                    "component means or variances are not all usable"))
  }
  list(q = mx$quantiles, applied = rep(FALSE, length(est)),
       source = "mixture_cdf", declined = NA_character_, mass = mass)
}

# Weighted mean and SD of `values` under pre-normalized `weights` (which
# sum to 1). SD uses the E[x^2] - E[x]^2 form, floored at 0 to absorb the
# floating-point negatives that form can produce near a degenerate axis.
.nl_wtd_mean_sd <- function(values, weights) {
  mu <- sum(weights * values)
  list(mean = mu, sd = sqrt(max(0, sum(weights * values^2) - mu^2)))
}

# Weighted-quantile median + 2.5/97.5 empirical CI for every axis of a
# (scalar or matrix) theta_grid. Returns named numeric vectors so the
# joint nested-Laplace surface exposes `theta_median / theta_ci_lo /
# theta_ci_hi` alongside `theta_mean / theta_sd` for every hyperparameter.
#
# `tg` is a vector or matrix; `log_marginal` aligns with `tg` rows;
# `refining` is the per-cell refining-axis tag from mode-tracked
# refinement (NULL or all-"" outside the joint path). For each axis,
# slice cells from OTHER axes are dropped before computing the quantile
# -- those cells pin the current axis at a single non-varying value, so
# including them oversamples that value. Cartesian cells, same-axis
# slice cells, and same-axis consistency cells are kept. This is the
# same per-axis mask used by `.joint_recalibrate_axis_moments` for the
# mean/SD path.
#
# Returns list(median = named_vec, ci_lo = named_vec, ci_hi = named_vec).
# For scalar tg the returned vectors are length-1 with names = "value".
#
# `support` says what the supplied `weights` are and is passed through to
# `.nl_summary_quantile`; `domains` gives one `.NL_DOMAIN_TRANSFORM` name per
# axis, which a `"moment_rule"` support needs to form its interval (an axis whose
# support the engine will not guess carries NA and reports NA).
.nl_axis_quantiles <- function(tg, log_marginal, refining = NULL,
                                probs = c(0.025, 0.5, 0.975),
                                weights = NULL,
                                support = .NL_SUPPORT_KINDS,
                                domains = NULL) {
  support <- match.arg(support)
  if (is.null(dim(tg))) {
    tg <- matrix(as.numeric(tg), ncol = 1L,
                 dimnames = list(NULL, "value"))
  }
  # Empty grid (no outer-grid axes -- e.g. an lf-only fit): nothing to
  # quantilize; return empty named vectors. paste0 recycles the prefix
  # past zero-length integers, so we guard ncol(tg) == 0 explicitly.
  if (ncol(tg) == 0L) {
    empty <- setNames(numeric(0), character(0))
    return(list(median = empty, ci_lo = empty, ci_hi = empty))
  }
  nms <- colnames(tg) %||% paste0("V", seq_len(ncol(tg)))
  n_ax <- length(nms)
  lo  <- setNames(rep(NA_real_, n_ax), nms)
  med <- setNames(rep(NA_real_, n_ax), nms)
  hi  <- setNames(rep(NA_real_, n_ax), nms)
  if (is.null(refining)) refining <- rep("", nrow(tg))
  for (j in seq_len(n_ax)) {
    ax    <- nms[j]
    keep  <- refining == "" | refining == ax |
             refining == paste0("consistency_", ax)
    use   <- keep & is.finite(tg[, j])
    if (sum(use) == 0L) next
    # Precomputed integration weights (CCD design weights * exp(log-marginal),
    # passed for scattered node sets where the per-axis softmax of the raw
    # log-marginal is not the integration weight); otherwise the regular-grid
    # softmax of the log-marginal.
    if (!is.null(weights)) {
      ws  <- weights[use]
      if (!any(is.finite(ws)) || sum(ws, na.rm = TRUE) <= 0) next
      ws[!is.finite(ws)] <- 0
      ws  <- ws / sum(ws)
    } else {
      lm_u  <- log_marginal[use]
      m     <- max(lm_u)
      if (!is.finite(m)) next
      ws    <- exp(lm_u - m); ws <- ws / sum(ws)
    }
    dm <- if (length(domains) < j) NA_character_ else domains[[j]]
    qs <- .nl_summary_quantile(as.numeric(tg[use, j]), ws, probs, dm, support)
    lo[j]  <- qs[1L]
    med[j] <- qs[2L]
    hi[j]  <- qs[3L]
  }
  list(median = med, ci_lo = lo, ci_hi = hi)
}

.nl_axis_marginal_logdensity <- function(vals, log_marg, keep = NULL) {
  if (is.null(keep)) keep <- rep(TRUE, length(vals))
  v <- vals[keep]; l <- log_marg[keep]
  uv <- sort(unique(v))
  if (length(uv) == 0L) return(list(vals = uv, log_marg = numeric(0)))
  lm <- vapply(uv, function(u) {
    li <- l[v == u]
    if (length(li) == 0L) return(-Inf)
    m <- max(li)
    if (!is.finite(m)) return(-Inf)
    m + log(sum(exp(li - m)))
  }, numeric(1))
  list(vals = uv, log_marg = lm)
}

# Laplace-at-mode SD on a single axis. Fits a 3-point parabola to the
# marginal log-density at the modal cell and one neighbour on each side,
# returns sqrt(-1 / (2 a)) from the quadratic coefficient. When the
# axis values are positive (sigma / phi / tau) the parabola is fit on
# log(theta), and the SD is mapped back to the linear axis via the
# delta method (sigma_theta = theta_mode * sigma_log_theta). Returns NA
# when the mode sits at an axis edge or the parabola is concave up.
#
# `return_log_sd = TRUE` skips the delta back-map and returns sd(log theta)
# directly. Only meaningful with `log_axis = TRUE` -- returns NA otherwise.
.nl_laplace_at_mode_sd_axis <- function(vals, log_marg, log_axis = NULL,
                                        return_log_sd = FALSE) {
  if (length(vals) < 3L) return(NA_real_)
  ix <- which.max(log_marg)
  if (ix == 1L || ix == length(vals)) return(NA_real_)
  if (is.null(log_axis)) log_axis <- all(is.finite(vals)) && all(vals > 0)
  u <- if (log_axis) log(vals) else vals
  dm <- u[ix - 1L] - u[ix]
  dp <- u[ix + 1L] - u[ix]
  det <- dm * dp * (dm - dp)
  if (!is.finite(det) || abs(det) < .Machine$double.eps) return(NA_real_)
  lm_m <- log_marg[ix - 1L] - log_marg[ix]
  lm_p <- log_marg[ix + 1L] - log_marg[ix]
  a <- (lm_m * dp - lm_p * dm) / det
  if (!is.finite(a) || a >= 0) return(NA_real_)
  sd_u <- sqrt(-1 / (2 * a))
  if (return_log_sd) {
    if (log_axis) sd_u else NA_real_
  } else {
    if (log_axis) vals[ix] * sd_u else sd_u
  }
}

# Replace `theta_sd` (and `block_moments[[b]]$sd` when present) entries
# with the Laplace-at-mode SD wherever the 3-point fit succeeds. Axes
# with the mode at an edge or wrong-signed curvature keep their var-of-
# means SD. Grid-spacing-independent: fixes the symptom where a sharply
# peaked marginal log-likelihood on a coarse grid collapses var-of-
# means to ~0 and undercovers.
.nl_refit_axis_sd_laplace <- function(res, refining = NULL) {
  if (is.null(res$theta_grid) || is.null(res$log_marginal)) return(res)
  tg <- res$theta_grid
  if (!is.matrix(tg)) {
    marg <- .nl_axis_marginal_logdensity(as.numeric(tg), res$log_marginal)
    sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
    if (is.finite(sd_lam)) res$theta_sd <- sd_lam
    return(res)
  }
  if (is.null(refining)) refining <- res$refining_axis
  if (is.null(refining)) refining <- rep("", nrow(tg))
  col_names <- colnames(tg)
  if (!is.null(col_names) && !is.null(res$theta_sd)) {
    for (col in col_names) {
      if (!col %in% names(res$theta_sd)) next
      keep <- refining == "" | refining == col |
              refining == paste0("consistency_", col)
      marg <- .nl_axis_marginal_logdensity(tg[, col], res$log_marginal, keep)
      sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
      if (is.finite(sd_lam)) res$theta_sd[[col]] <- sd_lam
    }
  }
  if (!is.null(res$block_moments)) {
    for (b in seq_along(res$block_moments)) {
      bm <- res$block_moments[[b]]
      axis_cols <- bm$axis_cols
      if (is.null(axis_cols) || length(axis_cols) == 0L) next
      bare <- names(bm$sd)
      for (j in seq_along(axis_cols)) {
        col_ix <- axis_cols[j]
        col_name <- if (!is.null(col_names)) col_names[col_ix] else ""
        keep <- refining == "" | refining == col_name |
                refining == paste0("consistency_", col_name)
        marg <- .nl_axis_marginal_logdensity(tg[, col_ix], res$log_marginal,
                                              keep)
        sd_lam <- .nl_laplace_at_mode_sd_axis(marg$vals, marg$log_marg)
        if (is.finite(sd_lam)) res$block_moments[[b]]$sd[[j]] <- sd_lam
      }
    }
  }
  res
}

# Posterior moments for multi-block grids. Two flavours:
#  * joint moments: across all axes (same as single-block 2D scatter).
#  * per-block marginal moments: integrate out the other blocks' axes.
.nl_posterior_moments_multi <- function(out, prepared, axis_offsets, joint_grid) {
  w  <- out$weights
  tg <- joint_grid
  out$theta_mean <- as.numeric(crossprod(w, tg))
  names(out$theta_mean) <- colnames(tg)
  out$theta_sd <- sqrt(pmax(0, as.numeric(crossprod(w, tg^2)) -
                              out$theta_mean^2))
  names(out$theta_sd) <- colnames(tg)

  # Per-block marginals: for each block, sum weights over rows that share the
  # same per-block axis values, then take weighted mean/sd within those rows.
  # Because the joint grid is a Cartesian product, the "rows that share this
  # block's values" form n_block_rows groups indexed by idx[, b]. The marginal
  # weight for group g is sum of joint weights over its rows.
  B <- length(prepared)
  per_block_moments <- vector("list", B)
  for (b in seq_len(B)) {
    cols <- (axis_offsets[b] + 1L):axis_offsets[b + 1L]
    sub <- joint_grid[, cols, drop = FALSE]
    block_mean <- as.numeric(crossprod(w, sub))
    block_sd   <- sqrt(pmax(0, as.numeric(crossprod(w, sub^2)) - block_mean^2))
    # Use bare per-block axis names (e.g. "tau", "rho") instead of the
    # joint-grid-prefixed "b<N>.tau" -- the block index is already implicit
    # in the list position.
    bare_names <- .nl_block_axis_grid(prepared[[b]])$names
    names(block_mean) <- bare_names
    names(block_sd)   <- bare_names
    per_block_moments[[b]] <- list(
      type      = tolower(prepared[[b]]$type),
      mean      = block_mean,
      sd        = block_sd,
      axis_cols = cols
    )
  }
  out$block_moments <- per_block_moments

  # Weighted-quantile median + 2.5/97.5 CI per axis (calibrated summary
  # for right-skewed scale-like hyperparameters; see `.nl_axis_quantiles`).
  qs <- .nl_axis_quantiles(joint_grid, out$log_marginal, out$refining_axis)
  out$theta_median <- qs$median
  out$theta_ci_lo  <- qs$ci_lo
  out$theta_ci_hi  <- qs$ci_hi
  out
}
