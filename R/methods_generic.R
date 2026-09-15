# ============================================================================
# Generic S3 methods for tulpa_fit objects
#
# A tulpa_fit arrives in one of two posterior shapes:
#   * Sampler tier (mala/pathfinder/gibbs/nuts): a `$draws` matrix, columns in
#     [fixed, random] order. Summaries are empirical (column mean/sd/quantiles).
#   * Laplace tier: a `$mode` vector (fixed effects then random-effect values)
#     and `$H_beta`, the fixed-effect posterior precision. Fixed-effect
#     summaries are the Gaussian approximation -- mean = mode, sd =
#     sqrt(diag(H_beta^-1)), quantiles from qnorm. No Monte Carlo draws.
#
# tulpa() attaches `$n_fixed`, `$fixed_names`, `$param_names`, and `$re_layout`
# (see .tulpa_param_layout) so both shapes report meaningful names and a
# fixed/random split. The accessors below read that layout; model packages that
# define their own *.<class> methods are unaffected.
#
# Which shape a fit is in, is decided by testing whether a field is present --
# `$draws` vs `$mode` vs `$modes` vs `$cov`. That makes `$`'s default PARTIAL
# matching on lists an active hazard rather than a stylistic one: a fit is a
# list, so an absent field silently resolves to any longer field it prefixes,
# and the shape test then reads the wrong object. Real collisions on live fits:
#
#   $draws -> draws_kind    (every Laplace/EB fit: the string "iid")
#   $mode  -> model_matrix  (every sampler fit: the design matrix)
#   $sigma -> sigma_re      (AGQ fits: the RE sd printed as the dispersion)
#   $theta -> theta_hat     (EB fits)
#
# `$.tulpa_fit` below makes `$` exact for every fit and subclass, so the whole
# class of bug is unreachable rather than fixed one call site at a time -- and
# it covers model packages that set class = c("<model>_fit", "tulpa_fit") too.
# ============================================================================

# Exact-matching `$` for fit objects. `.subset2()` is the internal extractor and
# does no dispatch, so this cannot recurse; `match()` reproduces `$`'s contract
# of returning NULL for an absent field (where `[[` on a name would be free to
# error). Assignment is unaffected: `$<-` already matches exactly.
#' @export
`$.tulpa_fit` <- function(x, name) {
  i <- match(name, names(x))
  if (is.na(i)) NULL else .subset2(x, i)
}

# Canonical parameter layout from a model-data bundle. Returns the fixed-effect
# count and names, the full [fixed, random] name vector (random part in the
# group-major, coef-within-group order the Laplace mode and the sampler draws
# both use), and a per-term random-effect layout for ranef().
#' @keywords internal
.tulpa_param_layout <- function(bundle) {
  n_fixed     <- bundle$n_fixed %||% ncol(bundle$X) %||% 0L
  fixed_names <- bundle$fixed_names %||% colnames(bundle$X) %||%
    paste0("beta", seq_len(n_fixed))

  # Zero inflation appends a second fixed-effect block: the compiled mode is
  # [beta_count | beta_zi | RE], so the ZI coefficients extend the fixed prefix
  # that coef() / vcov() / confint() slice. Their `zi_` name prefix (set in
  # .zi_design) is what distinguishes them, and n_zi records the split point.
  n_zi     <- if (is.null(bundle$X_zi)) 0L else ncol(bundle$X_zi)
  zi_names <- if (n_zi > 0L) colnames(bundle$X_zi) else character(0)
  if (n_zi > 0L) {
    fixed_names <- c(fixed_names, zi_names)
    n_fixed     <- n_fixed + n_zi
  }

  re_terms  <- bundle$re_terms %||% list()
  re_layout <- vector("list", length(re_terms))

  for (k in seq_along(re_terms)) {
    rt <- re_terms[[k]]
    coef_labels <- c(if (isTRUE(rt$has_intercept)) "(Intercept)",
                     rt$slope_names %||% character(0))
    if (length(coef_labels) == 0L) coef_labels <- "(Intercept)"
    gv   <- rt$group_var %||% paste0("g", k)
    levs <- rt$levels %||% as.character(seq_len(rt$n_groups %||% 0L))
    re_layout[[k]] <- list(group_var = gv, levels = levs,
                           coef_labels = coef_labels,
                           n_groups = rt$n_groups %||% length(levs),
                           n_coefs = rt$n_coefs %||% length(coef_labels))
  }

  # Single source for the RE coefficient labels (group-major, coef-within-group);
  # ranef() reads the same helper off the layout it is handed.
  re_names <- .re_names_from_layout(re_layout)

  list(n_fixed     = n_fixed,
       fixed_names = fixed_names,
       n_zi        = n_zi,
       zi_names    = zi_names,
       param_names = c(fixed_names, re_names),
       re_layout   = re_layout)
}


# The engine's parameter layout for a ModelData sampler fit: the positions
# `cpp_tulpa_glmm_layout()` assigns when handed the inputs recorded in
# `$model_inputs`, which are the inputs the sampler received. NULL on a fit that
# records none.
#' @keywords internal
.tulpa_sampler_layout <- function(object) {
  mi <- object$model_inputs
  if (!is.list(mi)) return(NULL)
  cpp_tulpa_glmm_layout(
    y = mi$y, n_trials = mi$n_trials, X = mi$X, family = mi$family,
    phi = mi$phi, sigma_beta = mi$sigma_beta, offset_nullable = mi$offset,
    re_spec = mi$re_spec, spatial_spec = mi$spatial_spec,
    temporal_spec = mi$temporal_spec, sigma_re_scale = mi$sigma_re_scale,
    svc_spec = mi$svc_spec, tvc_spec = mi$tvc_spec, zi_spec = mi$zi_spec)
}

# Draw columns (1-based) of one `[start, end]` span of a sampler layout; none
# for an absent span.
#' @keywords internal
.layout_span_cols <- function(span) {
  if (length(span) != 2L || span[2L] < span[1L]) return(integer(0))
  seq.int(span[1L], span[2L])
}

# Draw columns holding the fixed effects, in `fixed_names` order: the count
# coefficients, then the zero-inflation coefficients. On a ModelData sampler fit
# they are where the engine layout puts them -- the zero-inflation block sits
# after the random effects and the variance components, not beside the count
# coefficients -- and on every other draw-carrying fit they lead the draws.
#' @keywords internal
.fixed_draw_cols <- function(object) {
  nc <- ncol(object$draws)
  layout <- if (.tulpa_linpred_source(object) == "sampler_model") {
    .tulpa_sampler_layout(object)
  }
  if (is.null(layout)) {
    return(seq_len(min(object$n_fixed %||% nc, nc)))
  }
  c(unlist(lapply(layout$beta, .layout_span_cols), use.names = FALSE),
    .layout_span_cols(layout$beta_zi))
}

# Fixed-effect posterior draws (n_samples x n_fixed), normalized across sampler
# shapes: the generic `$draws` matrix restricted to its fixed-effect columns
# (`.fixed_draw_cols()`), or the Gibbs `$beta` matrix (fixed effects only). NULL
# on the Laplace tier, which carries no draws.
#' @keywords internal
.fixed_draws_mat <- function(object) {
  if (is.matrix(object$draws) && nrow(object$draws) >= 1L &&
      ncol(object$draws) >= 1L) {
    return(object$draws[, .fixed_draw_cols(object), drop = FALSE])
  }
  if (is.matrix(object$beta) && ncol(object$beta) >= 1L) return(object$beta)
  NULL
}

# TRUE when `V` is a covariance this fit can report for `p` fixed effects: a
# square numeric matrix, free of NA, and at least `p` on a side. Guards the
# `cov_marginal` branches, which must not shadow the H_beta fallback with a
# matrix that would only fail later inside `diag()` or `solve()`.
#' @keywords internal
.is_usable_cov <- function(V, p) {
  is.matrix(V) && is.numeric(V) && nrow(V) == ncol(V) &&
    !anyNA(V) && length(p) == 1L && !is.na(p) && nrow(V) >= p && p >= 1L
}

# Random-effect posterior draws (n_samples x n_re): the `$draws` tail past the
# fixed block, or the Gibbs `$re` matrix. NULL when none.
#' @keywords internal
.re_draws_mat <- function(object) {
  # A 0-row draws matrix carries no posterior at all; handing its tail back as
  # RE draws gave colMeans() of nothing, i.e. NaN estimates (gcol33/tulpa#710).
  if (is.matrix(object$draws) && nrow(object$draws) > 0L) {
    rest <- setdiff(seq_len(ncol(object$draws)), .fixed_draw_cols(object))
    if (length(rest)) return(object$draws[, rest, drop = FALSE])
  }
  if (is.matrix(object$re) && ncol(object$re) >= 1L) return(object$re)
  NULL
}

# The random-effect COEFFICIENT draws of `.re_draws_mat()`. A named sampler
# tail is [latent (field + RE), hyperparameters], and its RE coefficients are
# exactly the `re[...]`-named columns, so the field / `log_sigma_re` / `L_re`
# columns are dropped. A tail naming no `re[...]` column is returned as it is
# (an unnamed `$re` matrix); callers check its width against the RE layout.
#' @keywords internal
.re_coef_draws <- function(object) {
  re <- .re_draws_mat(object)
  if (is.null(re)) return(NULL)
  re_cols <- .re_col_idx(colnames(re))
  if (length(re_cols)) re[, re_cols, drop = FALSE] else re
}

# Positions of the random-effect coefficient columns (`re[...]`) in a vector of
# sampler parameter names; empty for NULL names.
#' @keywords internal
.re_col_idx <- function(names) grep("^re\\[", names %||% character(0))

# TRUE when the fit carries posterior draws of the fixed effects (sampler tier).
#' @keywords internal
.has_draws <- function(object) !is.null(.fixed_draws_mat(object))

# TRUE when the posterior a fit reports is a Gaussian it states in closed form:
# `reported_posterior = "gaussian"`, with the mean in `mode` (or `means`) and
# the covariance in `cov`. Expectation propagation and a Laplace fit with no
# hyperparameter to integrate report such a Gaussian; the draws they also carry
# are samples FROM it, so moments and quantiles read off the draws would restate
# the stated Gaussian with Monte-Carlo error added.
#' @keywords internal
.reports_gaussian_posterior <- function(object) {
  identical(object$reported_posterior, "gaussian")
}

# The fixed-effect draws the coefficient summaries read: the fit's draws, except
# on a fit reporting a closed-form Gaussian, where the draws serve sampling only
# and the summaries read that Gaussian.
#' @keywords internal
.summary_draws_mat <- function(object) {
  if (.reports_gaussian_posterior(object)) return(NULL)
  .fixed_draws_mat(object)
}


# Grid-marginalized fixed-effect posterior for a nested-Laplace fit.
#
# What the grid actually defines for coefficient j is a Gaussian MIXTURE over
# the hyperparameter cells,
#   p(beta_j | y) = sum_g w_g N(mu_gj, V_gjj),
# so this returns both readings of it: the two moments, and the components they
# were formed from.
#
# Moments, via the law of total variance:
#   mean = sum_g w_g mu_g
#   cov  = sum_g w_g (V_g + mu_g mu_g') - mean mean'
# Components: `mu` / `var` are n_kept x p (a row is one cell), `w` the matching
# cell weights. A mean and a variance are linear functionals of the mixture and
# survive the collapse to one Gaussian; a quantile does not, which is why the
# components are kept rather than only the moments.
#
# mu_g / V_g are the per-grid fixed-effect mode / covariance retained under
# keep_grid_hessians (V_g = solve(grid_hessians[[g]])); w_g are the normalized
# grid weights. NULL when the per-grid pieces were not retained.
#
# One marginalizer for every nested tier: `tulpa_nested_laplace()` fills the
# pair from its own per-cell precision, and both `tulpa_nested_laplace_joint()`
# paths fill it through `.joint_attach_grid_fixed()`.
#
# A cell with zero integration weight contributes nothing to either moment, so
# it is skipped rather than multiplied in -- a pruned cell that carries no
# retained block would otherwise turn the whole covariance into NA.
#
# The retained weights are renormalized, so what is returned is the posterior
# CONDITIONAL ON THE RETAINED CELLS, not a reconstruction of the full grid: a
# dropped positive-weight cell gives exactly the moments of a grid that never
# held it, and the mass it carried is gone rather than redistributed back into
# the answer. Without the renormalization a dropped positive-weight cell shrinks
# the mean toward the origin by exactly the dropped mass.
#
# `mass` is the ORIGINAL retained share of the grid weight -- 1 on a complete
# grid, less on one that dropped a positive-weight cell -- so a reader can tell
# the two apart from the returned value alone. The returned `w` sums to one over
# the retained cells, which is the weighting BOTH the moments and the mixture
# components are formed under, so the two reads describe one posterior.
#
# `keep` is the grid-cell index each mixture component came from, so a caller
# that needs a component's FULL covariance (the mixture sampler,
# `tulpa_posterior_draws()`) reaches the same cell this read summarized rather
# than re-deriving which cells were retained. Only the component variances are
# returned in `var`, since that is all the mixture CDF needs.
#' @keywords internal
.nested_fixed_moments <- function(object) {
  H <- object$grid_hessians
  M <- object$grid_modes
  if (is.null(H) || is.null(M) || is.null(object$weights)) return(NULL)
  denom <- sum(object$weights)
  w <- object$weights / denom
  if (length(H) != length(w) || length(M) != length(w)) return(NULL)
  keep <- which(is.finite(w) & w > 0 &
                  !vapply(H, is.null, logical(1)) &
                  !vapply(M, is.null, logical(1)))
  if (!length(keep)) return(NULL)
  # Normalized off the raw weights rather than off `w`, so a grid that kept
  # every cell divides by the same sum it already divided by and its weights
  # come out bit-for-bit what they were.
  sub  <- sum(object$weights[keep])
  mass <- sub / denom
  if (!is.finite(mass) || mass <= 0) return(NULL)
  wk <- object$weights[keep] / sub
  p <- length(M[[keep[1]]])
  m <- numeric(p)
  S <- matrix(0, p, p)
  mu_k  <- matrix(NA_real_, length(keep), p)
  var_k <- matrix(NA_real_, length(keep), p)
  for (i in seq_along(keep)) {
    g  <- keep[i]
    Vg <- .nested_cell_fixed_cov(H[[g]], p)
    mu <- M[[g]]
    m <- m + wk[i] * mu
    S <- S + wk[i] * (Vg + tcrossprod(mu))
    mu_k[i, ]  <- mu
    var_k[i, ] <- diag(Vg)
  }
  list(mean = m, cov = S - tcrossprod(m),
       mu = mu_k, var = var_k, w = wk, mass = mass, keep = keep)
}


# Which outer-grid cells is a per-cell mode readable from? Those whose inner
# solve reached a mode. A cell that stalled reports its START vector as `$mode`
# -- the solver's own record of where it stopped, which the warm-start chain and
# the refinement passes need -- and that vector is not an estimate of anything.
# Every report that AVERAGES per-cell modes gates on this so the raw start
# cannot surface as a coefficient.
#
# A fit carrying no convergence flag answers "all readable": an absent flag is a
# backend that does not report one, not evidence of a stalled solve.
#' @keywords internal
.nested_converged_cells <- function(object, n_cell) {
  conv <- object$converged
  if (is.null(conv) || length(conv) != n_cell) return(rep(TRUE, n_cell))
  as.logical(conv) %in% TRUE
}

# Did any cell the weights put mass on reach a mode? The predicate the joint
# fixed-effect retention declines on, read here so both tiers share one
# definition of "this fit has a mode to report".
#' @keywords internal
.nested_any_weighted_converged <- function(object) {
  conv <- object$converged
  if (is.null(conv) || !length(conv)) return(TRUE)
  ok <- .nested_converged_cells(object, length(conv))
  w  <- object$weights
  keep <- if (is.null(w) || length(w) != length(ok)) {
    rep(TRUE, length(ok))
  } else {
    is.finite(w) & w > 0
  }
  # An all-NA or all-zero weight vector says nothing about WHICH cells matter,
  # so every cell is read rather than none.
  if (!any(keep)) keep <- rep(TRUE, length(ok))
  any(ok[keep])
}

# Fixed-effect covariance of ONE outer-grid cell: the inverse of the cell's
# retained marginal precision, or an all-NA p x p block when that precision is
# singular. Both readers of the retained mixture go through it -- the moment /
# component summary above, and the mixture sampler in R/posterior_draws.R, which
# needs the full matrix rather than its diagonal.
#' @keywords internal
.nested_cell_fixed_cov <- function(Hg, p) {
  tryCatch(solve(Hg), error = function(e) matrix(NA_real_, p, p))
}


# Fixed-effect coefficient table (estimate, std.error, conf.low, conf.high),
# normalized across posterior shapes. The single source the coefficient-facing
# methods (summary, coef, confint, vcov, tidy) read from.
#' @keywords internal
.fit_fixed_table <- function(object, level = 0.95) {
  a <- (1 - level) / 2

  fd <- .summary_draws_mat(object)
  if (!is.null(fd)) {
    idx <- seq_len(ncol(fd))
    nm  <- (object$fixed_names %||% object$param_names %||%
              colnames(fd) %||% paste0("param", idx))[idx]
    return(data.frame(
      term      = nm,
      estimate  = colMeans(fd),
      std.error = apply(fd, 2, stats::sd),
      conf.low  = apply(fd, 2, stats::quantile, a),
      conf.high = apply(fd, 2, stats::quantile, 1 - a),
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  # Nested-Laplace fit: the posterior is a mixture over the hyperparameter grid.
  # When the per-grid fixed-effect pieces are retained (keep_grid_hessians, the
  # tulpa() default for nested fits), the grid-marginalized mean and covariance
  # come from the law of total variance (.nested_fixed_moments). Otherwise only
  # the marginalized mean is available, so SE is reported as NA rather than a
  # misleadingly small between-grid-only value.
  if (is.matrix(object$modes) && !is.null(object$weights)) {
    p   <- object$n_fixed %||% ncol(object$modes)
    idx <- seq_len(p)
    nm  <- (object$fixed_names %||% object$param_names %||%
              paste0("beta", idx))[idx]
    mom <- .nested_fixed_moments(object)
    if (!is.null(mom)) {
      est <- mom$mean[idx]
      se  <- sqrt(pmax(diag(mom$cov)[idx], 0))
      # The bounds invert the Gaussian mixture the grid defines rather than
      # reading them off the collapsed Gaussian, except on a
      # fit carrying the skew correction, which keeps the MAP-cell read it
      # was measured on. `.nl_fixed_interval()` owns that choice and reports
      # which read ran; `applied` travels with the table so the reporting
      # methods can say which coefficients were skew-corrected.
      sc <- .nl_skew_correction(object, p)
      iv <- .nl_fixed_interval(mom, idx, est, se, c(a, 1 - a), sc)
      tab <- data.frame(
        term = nm, estimate = est, std.error = se,
        conf.low = iv$q[, 1L], conf.high = iv$q[, 2L],
        row.names = NULL, stringsAsFactors = FALSE
      )
      attr(tab, "skew_applied")     <- stats::setNames(iv$applied, nm)
      attr(tab, "interval_source")  <- iv$source
      attr(tab, "interval_declined") <- iv$declined
      attr(tab, "retained_mass")    <- iv$mass
      return(tab)
    }
    # The per-cell modes averaged over the grid, restricted to the cells whose
    # inner solve reached a mode. A stalled cell reports the vector its Newton
    # started from, so averaging it in reports a number that estimates nothing;
    # with no readable cell left the estimate is NA and `interval_declined` says
    # why, which is what makes the non-convergence impossible to read past.
    n_cell <- nrow(object$modes)
    w_all  <- if (length(object$weights) == n_cell) as.numeric(object$weights)
              else rep(1, n_cell)
    ok  <- .nested_converged_cells(object, n_cell) &
           is.finite(w_all) & w_all > 0
    tab <- data.frame(
      term = nm, estimate = NA_real_,
      std.error = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      row.names = NULL, stringsAsFactors = FALSE
    )
    if (!any(ok)) {
      attr(tab, "interval_declined") <- "not_converged"
      return(tab)
    }
    w <- w_all[ok] / sum(w_all[ok])
    tab$estimate <- as.numeric(crossprod(w, object$modes[ok, idx, drop = FALSE]))
    return(tab)
  }

  if (!is.null(object$mode) && !is.null(object$H_beta)) {
    p   <- object$n_fixed %||% nrow(object$H_beta)
    idx <- seq_len(p)
    est <- object$mode[idx]
    # A fit carrying `cov_marginal` has already marginalized the hyperparameter
    # uncertainty into its fixed-effect covariance (tulpa_eb(marginal = TRUE)),
    # so inverting H_beta here would report the narrower conditional intervals
    # the correction exists to replace.
    V   <- if (.is_usable_cov(object$cov_marginal, p)) {
      object$cov_marginal
    } else tryCatch(solve(object$H_beta), error = function(e) {
      warning("H_beta is singular; standard errors set to NA.", call. = FALSE)
      matrix(NA_real_, p, p)
    })
    se <- sqrt(pmax(diag(V)[idx], 0))
    z  <- stats::qnorm(1 - a)
    nm <- (object$fixed_names %||% object$param_names %||%
             paste0("beta", idx))[idx]
    return(data.frame(
      term      = nm,
      estimate  = est,
      std.error = se,
      conf.low  = est - z * se,
      conf.high = est + z * se,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  # Gaussian fit carrying the full-parameter covariance directly (AGQ, whose
  # inverse observed information is $cov rather than a precision $H_beta, and
  # every fit reporting a closed-form Gaussian): the fixed-effect block of $cov
  # gives the SEs, and the bounds are that Gaussian's quantiles.
  est_all <- object$mode %||% object$means
  if (!is.null(est_all) && is.matrix(object$cov) && !anyNA(object$cov)) {
    p   <- object$n_fixed %||% length(est_all)
    idx <- seq_len(p)
    est <- as.numeric(est_all)[idx]
    se  <- sqrt(pmax(diag(object$cov)[idx], 0))
    z   <- stats::qnorm(1 - a)
    nm  <- (object$fixed_names %||% object$param_names %||%
              names(est_all) %||% paste0("beta", idx))[idx]
    return(data.frame(
      term = nm, estimate = est, std.error = se,
      conf.low = est - z * se, conf.high = est + z * se,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  if (!is.null(object$means)) {
    est <- as.numeric(object$means)
    nm  <- object$param_names %||% names(object$means) %||%
      paste0("param", seq_along(est))
    return(data.frame(
      term = nm[seq_along(est)], estimate = est,
      std.error = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  stop("Cannot summarize this tulpa_fit: it carries no $draws, $mode/$H_beta, ",
       "$cov, or $means.", call. = FALSE)
}


#' Fixed-effect coefficients
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Named numeric vector of fixed-effect posterior means (the Laplace
#'   mode for the Laplace tier). Random effects come from [ranef()].
#' @export
coef.tulpa_fit <- function(object, ...) {
  tab <- .fit_fixed_table(object)
  stats::setNames(tab$estimate, tab$term)
}

# Compact fit display, robust across posterior shapes: reads only fields the
# accessors above normalize, so sampler-, Laplace-, and nested-tier fits all
# print without NA/NULL noise.
#' @export
print.tulpa_fit <- function(x, ...) {
  header <- "tulpa fit"
  if (!is.null(x$backend)) header <- paste0(header, "  (", x$backend, ")")
  cat(header, "\n")
  .print_fit_body(x)
  invisible(x)
}


# What every fit reports below its header. Split from the header so a subclass
# that prints its own (the nested-Laplace tier) composes this rather than
# restating it, and so each section is reached by every fit that has the
# structure it describes -- the random-effect covariance and the smoother terms
# are read off the fit, not off which function was called to produce it.
#' @keywords internal
.print_fit_body <- function(x) {
  n_obs <- x$N %||% x$n_obs
  if (!is.null(n_obs)) cat("  n_obs:", n_obs, "\n")
  fd <- .fixed_draws_mat(x)
  if (!is.null(fd)) cat("  posterior draws:", nrow(fd), "\n")
  tab <- tryCatch(.fit_fixed_table(x), error = function(e) NULL)
  if (!is.null(tab) && nrow(tab)) {
    cat("\nFixed effects:\n")
    print(data.frame(estimate  = round(tab$estimate, 4),
                     std.error = round(tab$std.error, 4),
                     row.names = tab$term))
  } else if (!is.null(x$means)) {
    cat("\nPosterior means:\n")
    print(round(x$means, 4))
  }
  if (is.numeric(x$sigma) && length(x$sigma) == 1L) {
    cat("\nsigma:", round(x$sigma, 4), "\n")
  }
  .print_re_section(x)
  .print_smooth_section(x)
  adl <- .tulpa_axis_dropped_line(.tulpa_axis_dropped(x))
  if (!is.null(adl)) cat("\n", adl, "\n", sep = "")
  invisible(x)
}

# Labels of the lower and upper interval columns at `level`, in the layout
# `stats::confint.default()` gives them ("2.5 %", "97.5 %", "5 %"), so an
# interval read by column name reads a tulpa fit as it reads an lm or glm fit.
#' @keywords internal
.interval_colnames <- function(level) {
  a <- (1 - level) / 2
  paste(format(100 * c(a, 1 - a), trim = TRUE, scientific = FALSE, digits = 3),
        "%")
}

#' Posterior summary of the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param level Credible-interval level (default 0.95).
#' @param ... Ignored.
#' @return Data frame: estimate, std.error, and lower/upper credible bounds, one
#'   row per fixed effect, the bound columns labelled as [confint.tulpa_fit()]
#'   labels them. Sampler tiers report empirical quantiles; the Laplace
#'   tier reports the Gaussian approximation, and so does any fit whose reported
#'   posterior is a closed-form Gaussian (expectation propagation, the
#'   multinomial Laplace fit), whose draws are then samples from that Gaussian
#'   and are not what the summary reads.
#'
#'   On a nested-Laplace fit the estimate and standard error are the
#'   hyperparameter-grid-marginalized moments, and the bounds invert the Gaussian
#'   mixture `sum_k w_k N(mu_kj, V_kjj)` that grid defines, rather than reading
#'   `mu +/- z sigma` off the single Gaussian matching those moments. An
#'   `interval_source` attribute records which read produced them
#'   (`"mixture_cdf"`, `"gaussian_moment"`, `"skew_map_cell"`, or
#'   `"skew_map_cell/mixture_cdf"` when the two are in play on different
#'   coefficients) and `interval_declined` says why, whenever the mixture read
#'   did not run. A
#'   `retained_mass` attribute gives the share of the grid weight whose cells
#'   retained a fixed-effect block: 1 on a complete grid, and below 1 on one
#'   that dropped a positive-weight cell, whose report is then the posterior
#'   conditional on the cells that remain.
#'
#'   Where no per-cell block was retained the estimate falls back to the
#'   grid-weighted average of the per-cell modes, restricted to the cells whose
#'   inner solve reached a mode. A fit where none did reports `NA` with
#'   `interval_declined = "not_converged"`, rather than the vector its Newton
#'   started from as an estimate.
#'
#'   With `control$skew_correct = TRUE` a coefficient whose inner-Laplace
#'   `gamma_3` is in the band it is valid on reports Cornish-Fisher quantiles
#'   instead, and a `skew_applied` attribute names which coefficients took the
#'   correction. That correction is measured at the MAP cell, so it is reported
#'   on its own and is not composed with the mixture read; a coefficient it
#'   declines keeps the mixture read rather than falling back further.
#'
#'   An `axis_fields_dropped` attribute carries the grid axes the fit's own
#'   resolved path could not read, one row per dropped field (block, type,
#'   field, path, integrates, reason). It is `NULL` whenever every supplied axis
#'   was used, which is the ordinary case; [diagnostic_summary()] reads the same
#'   record in sentences.
#'
#'   A `beta_prior` attribute carries the Gaussian fixed-effect prior the fit
#'   ran under, as `list(mean, sd)`. It is the engine default,
#'   `prior_normal(0, 2.5)`, whenever the caller supplied none, and `NULL` on
#'   the paths that express no Gaussian prior on the fixed effects.
#' @export
summary.tulpa_fit <- function(object, level = 0.95, ...) {
  tab <- .fit_fixed_table(object, level = level)
  out <- data.frame(
    estimate  = tab$estimate,
    std.error = tab$std.error,
    `conf.low`  = tab$conf.low,
    `conf.high` = tab$conf.high,
    row.names = tab$term, check.names = FALSE
  )
  names(out)[3:4] <- .interval_colnames(level)
  attr(out, "skew_applied")      <- attr(tab, "skew_applied")
  attr(out, "interval_source")   <- attr(tab, "interval_source")
  attr(out, "interval_declined") <- attr(tab, "interval_declined")
  attr(out, "retained_mass")     <- attr(tab, "retained_mass")
  attr(out, "axis_fields_dropped") <- .tulpa_axis_dropped(object)
  attr(out, "beta_prior")        <- object$beta_prior
  out
}

#' Credible intervals for the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param parm Parameter names or indices (default: all fixed effects).
#' @param level Interval level (default 0.95).
#' @param ... Ignored.
#' @return Matrix with lower and upper columns, labelled as
#'   [stats::confint.default()] labels them (`"2.5 %"` and `"97.5 %"` at the
#'   default level). A nested-Laplace fit carries
#'   `interval_source` / `interval_declined` (which read produced the bounds --
#'   by default the grid's Gaussian-mixture CDF), `retained_mass` (the share of
#'   the grid weight the bounds are conditional on), and `skew_applied`, one
#'   logical per reported coefficient saying whether its bounds are the
#'   inner-Laplace skew-corrected quantiles or not. See [summary.tulpa_fit()].
#' @export
confint.tulpa_fit <- function(object, parm = NULL, level = 0.95, ...) {
  tab <- .fit_fixed_table(object, level = level)
  ci <- as.matrix(tab[, c("conf.low", "conf.high")])
  rownames(ci) <- tab$term
  colnames(ci) <- .interval_colnames(level)
  sa <- attr(tab, "skew_applied")
  if (!is.null(parm)) {
    ci <- ci[parm, , drop = FALSE]
    if (!is.null(sa)) sa <- sa[parm]
  }
  attr(ci, "skew_applied")      <- sa
  attr(ci, "interval_source")   <- attr(tab, "interval_source")
  attr(ci, "interval_declined") <- attr(tab, "interval_declined")
  attr(ci, "retained_mass")     <- attr(tab, "retained_mass")
  ci
}

#' Variance-covariance matrix of the fixed effects
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Fixed-effect variance-covariance matrix (empirical for sampler tiers,
#'   `H_beta^-1` for the Laplace tier, the stated covariance on a fit whose
#'   reported posterior is a closed-form Gaussian).
#' @export
vcov.tulpa_fit <- function(object, ...) {
  fd  <- .summary_draws_mat(object)
  mom <- .nested_fixed_moments(object)
  if (!is.null(fd)) {
    V <- stats::cov(fd)
  } else if (!is.null(mom)) {
    p <- object$n_fixed %||% nrow(mom$cov)
    V <- mom$cov[seq_len(p), seq_len(p), drop = FALSE]
  } else if (.is_usable_cov(object$cov_marginal,
                            object$n_fixed %||% nrow(object$cov_marginal))) {
    p <- object$n_fixed %||% nrow(object$cov_marginal)
    V <- object$cov_marginal[seq_len(p), seq_len(p), drop = FALSE]
  } else if (!is.null(object$H_beta)) {
    p <- object$n_fixed %||% nrow(object$H_beta)
    V <- solve(object$H_beta)[seq_len(p), seq_len(p), drop = FALSE]
  } else if (is.matrix(object$cov) && !anyNA(object$cov)) {
    p <- object$n_fixed %||% nrow(object$cov)
    V <- object$cov[seq_len(p), seq_len(p), drop = FALSE]
  } else {
    stop("No posterior draws, grid moments, H_beta, or cov available for vcov().",
         call. = FALSE)
  }
  nm <- (object$fixed_names %||% object$param_names)[seq_len(ncol(V))]
  if (!is.null(nm)) dimnames(V) <- list(nm, nm)
  V
}

#' The fit's log-scale goodness quantity
#'
#' What this returns depends on what the fit computed, and the returned object
#' names it in a `quantity` attribute:
#'
#' \describe{
#'   \item{`"log_posterior_mean"`}{a sampler fit: the mean over the draws of
#'     the log joint posterior density, prior included, in that backend's own
#'     parameterization: the unconstrained coordinates with their Jacobians
#'     for the ModelData samplers (`hmc`, `ess`, `sghmc`, `sgld`, `mclmc`,
#'     `smc`, `vi`), and the natural scale each quantity is sampled on for the
#'     RE-covariance and Polya-Gamma Gibbs samplers. Values are therefore
#'     comparable across fits of the same backend only.}
#'   \item{`"log_evidence"`}{a deterministic fit that estimated no
#'     hyperparameter from the data: the log marginal probability of the data
#'     under the model as specified. A hyperparameter the fit integrated counts,
#'     and so does one the caller supplied (a `phi`, a `sigma_re`, an outer grid
#'     laid at one value), which is part of the model rather than an estimate. A
#'     Laplace fit of a model with nothing to integrate reports its log marginal
#'     likelihood here, which already is that quantity.}
#'   \item{`"log_marginal_likelihood"`}{a deterministic fit that estimated some
#'     hyperparameters by maximising over the same data (empirical Bayes, or
#'     `estimate_phi = TRUE`): the log marginal likelihood at those estimates.
#'     The `conditioned_on` attribute names them.}
#'   \item{`"log_likelihood"`}{a fit that maximised over every parameter it
#'     reports, with the random effects integrated out ([agq_fit()]): the
#'     maximised log-likelihood, with `df` the number of maximised parameters.}
#' }
#'
#' On a nested-Laplace fit the value is the log evidence of its outer grid,
#' `log sum_k exp(log_marginal_k + log_cell_k)`, where `log_marginal` already
#' carries each axis's hyperprior density on its integration coordinate and
#' `log_cell_k` is the cell's absolute volume there. Every default outer axis
#' carries a proper prior (a PC prior on each scale and range, a uniform on each
#' bounded axis), so the value is the evidence under that prior and does not move
#' with where the nodes were laid or how many there are, provided the grid
#' covers the posterior. A fit carrying an axis with no proper prior (a flat
#' random-effect covariance prior, a dispersion whose family has no PC prior yet,
#' an axis on a block that does not carry its coordinates) reports `NA` with
#' `declined = "improper_hyperprior"` and names the axes in a `declined_axes`
#' attribute. A central-composite (CCD) design reproduces moments and carries no
#' cell volume, so a fit integrated on one reports `NA` with a `declined`
#' attribute.
#'
#' A sampler fit whose producer kept no per-draw log posterior reports `NA`
#' with `quantity = "log_posterior_mean"` and
#' `declined = "no_log_posterior_recorded"`. The Polya-Gamma Gibbs routes that
#' recentre a proper field every sweep (the NNGP and multiscale-GP fields, and
#' the negative-binomial kernel's iid random-effect block) leave no written
#' density invariant and record none; a fit carrying neither draws
#' nor a log marginal declines with `"no_goodness_quantity_recorded"`. A value
#' of `NA` always carries a `declined` attribute.
#'
#' Values with a different `quantity` or `conditioned_on` are not on one scale,
#' and `compare_models(criterion = "loglik")` refuses such a set. Only a
#' `"log_likelihood"` is what AIC and BIC penalise, so [AIC.tulpa_fit()] and
#' [BIC.tulpa_fit()] refuse every other quantity; compare those fits by their
#' evidence, or by `compare_models(criterion = "waic")` / `"loo"`, which score
#' the pointwise predictive density.
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return A `logLik` object with `quantity` and `conditioned_on` attributes,
#'   and a `declined` attribute naming the reason when no value could be read.
#' @export
logLik.tulpa_fit <- function(object, ...) {
  declined <- NA_character_
  recorded <- !is.null(object$log_evidence) ||
              !is.null(object$log_evidence_declined)
  ll <- if (!is.null(object$log_prob)) {
    mean(object$log_prob, na.rm = TRUE)
  } else if (recorded) {
    # An outer integration its producer recorded, under the measure and the
    # hyperprior its weights took (`.nl_outer_log_evidence()`).
    v <- as.numeric(object$log_evidence %||% NA_real_)[1L]
    if (!is.finite(v)) {
      declined <- as.character(object$log_evidence_declined %||% NA_character_)[1L]
      if (is.na(declined)) declined <- .loglik_decline("no_finite_cell")
      v <- NA_real_
    }
    v
  } else if (length(object$log_marginal) == 1L) {
    as.numeric(object$log_marginal)
  } else if (length(object$log_marginal) > 1L) {
    # A per-cell vector with no record of the measure its weights took. The
    # cells' prior masses are part of the evidence, so a sum over the vector
    # alone is not one.
    declined <- .loglik_decline("outer_measure_not_recorded")
    NA_real_
  } else if (.has_draws(object)) {
    # A sampler whose producer kept no per-draw log posterior: the quantity a
    # draw-based fit reports cannot be formed from the draws alone.
    declined <- .loglik_decline("no_log_posterior_recorded")
    NA_real_
  } else {
    declined <- .loglik_decline("no_goodness_quantity_recorded")
    NA_real_
  }

  estimated <- .fit_estimated_hyperparameters(object)
  quantity <- if (!is.null(object$log_prob) ||
                  (!recorded && is.null(object$log_marginal) &&
                   .has_draws(object))) {
    "log_posterior_mean"
  } else if (!recorded && is.null(object$log_marginal)) {
    NA_character_
  } else if ((object$backend %||% "") %in% .ML_BACKENDS) {
    "log_likelihood"
  } else if (length(estimated)) {
    "log_marginal_likelihood"
  } else {
    "log_evidence"
  }

  # Free-parameter count, from the first source the fit actually carries. NOT a
  # `%||%` chain over `length()` calls: `length(NULL)` is 0, not NULL, so such a
  # chain stops at the first ABSENT candidate and every later fallback is
  # unreachable. Take the first candidate that resolves to a positive count. A
  # maximised log-likelihood counts every maximised parameter, which is its
  # `n_params`.
  df_candidates <- c(if (identical(quantity, "log_likelihood")) object$n_params,
                     object$n_fixed, length(object$mode), length(object$means))
  df_candidates <- df_candidates[is.finite(df_candidates) & df_candidates > 0]
  attr(ll, "df")   <- if (length(df_candidates)) as.integer(df_candidates[1L]) else 0L
  attr(ll, "nobs") <- object$N %||% NA_integer_
  attr(ll, "quantity") <- quantity
  attr(ll, "conditioned_on") <- if (identical(quantity, "log_marginal_likelihood"))
                                  estimated else character(0)
  if (!is.na(declined)) attr(ll, "declined") <- declined
  if (!is.na(declined) && length(object$log_evidence_declined_axes)) {
    attr(ll, "declined_axes") <- object$log_evidence_declined_axes
  }
  class(ll) <- "logLik"
  ll
}

# Backends whose log_marginal is maximised over every parameter they report,
# the random effects integrated out.
.ML_BACKENDS <- c("agq")

# Why logLik() reports no value, where the reason is logLik()'s own rather than
# one the producer recorded in `log_evidence_declined`.
.LOGLIK_DECLINE_REASONS <- c(
  # An outer integration was recorded but no cell produced a finite value.
  "no_finite_cell",
  # A per-cell log-marginal vector with no record of the cells' measure.
  "outer_measure_not_recorded",
  # A draw-based fit whose producer kept no per-draw log posterior.
  "no_log_posterior_recorded",
  # A fit carrying neither draws nor any log marginal.
  "no_goodness_quantity_recorded"
)

.loglik_decline <- function(reason) {
  stopifnot(reason %in% .LOGLIK_DECLINE_REASONS)
  reason
}

# The hyperparameters a deterministic fit estimated by maximising over the same
# data, which makes its log marginal likelihood conditional on those estimates
# rather than an evidence. A value the fit integrated, or one the caller
# supplied, is part of the model and is not listed.
#' @keywords internal
.fit_estimated_hyperparameters <- function(object) {
  out <- character(0)
  if (identical(object$backend, "eb")) out <- c(out, "re_covariance")
  if (isTRUE(object$phi_estimated)) out <- c(out, "phi")
  out
}

#' Information criteria on a tulpa fit
#'
#' AIC and BIC penalise a maximised log-likelihood. [logLik.tulpa_fit()] reports
#' one only for a fit that maximised over every parameter it reports
#' (`quantity = "log_likelihood"`, as [agq_fit()] does); a mean log posterior, a
#' log evidence and a conditional log marginal likelihood are not, and both
#' criteria refuse them rather than return a number. Compare those fits by
#' their evidence, or by `compare_models(criterion = "waic")` / `"loo"`.
#'
#' @param object A `tulpa_fit` object.
#' @param ... Further fits.
#' @param k Penalty per parameter.
#' @return As [stats::AIC()] / [stats::BIC()] for maximised log-likelihoods;
#'   errors otherwise.
#' @importFrom stats AIC BIC
#' @export
AIC.tulpa_fit <- function(object, ..., k = 2) {
  .check_information_criterion(list(object, ...), "AIC")
  NextMethod()
}

#' @rdname AIC.tulpa_fit
#' @export
BIC.tulpa_fit <- function(object, ...) {
  .check_information_criterion(list(object, ...), "BIC")
  NextMethod()
}

.check_information_criterion <- function(fits, what) {
  for (f in fits) {
    if (!inherits(f, "tulpa_fit")) next
    ll <- logLik(f)
    q  <- attr(ll, "quantity") %||% NA_character_
    if (!identical(q, "log_likelihood")) {
      reads <- if (is.na(q)) {
        sprintf("reports no quantity (declined: %s)",
                attr(ll, "declined") %||% "unrecorded")
      } else {
        paste("is a", gsub("_", " ", q))
      }
      stop(sprintf(paste0(
        "%s() needs a maximised log-likelihood; logLik() on this fit %s. ",
        "Compare fits by their evidence, or by compare_models(criterion = ",
        "\"waic\") / \"loo\"."), what, reads), call. = FALSE)
    }
  }
  invisible(NULL)
}

# The tidy / glance generics come from `generics` (the shared broom-ecosystem
# home), re-exported here -- defining our own would mask broom's when both are
# attached.

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy fixed-effect table (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param conf.level Interval level (default 0.95).
#' @param ... Ignored.
#' @return Data frame: term, estimate, std.error, conf.low, conf.high.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
#' tidy(fit)
#' }
#' @export
tidy.tulpa_fit <- function(x, conf.level = 0.95, ...) {
  .fit_fixed_table(x, level = conf.level)
}

#' Model-level summary statistics (broom-compatible)
#'
#' @param x A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Single-row data frame.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x, data = df, family = "poisson")
#' glance(fit)
#' }
#' @export
glance.tulpa_fit <- function(x, ...) {
  data.frame(
    n_fixed     = x$n_fixed %||% NA_integer_,
    n_samples   = x$n_samples %||% { fd <- .fixed_draws_mat(x); if (!is.null(fd)) nrow(fd) else NA_integer_ },
    logLik      = as.numeric(logLik(x)),
    n_divergent = if (!is.null(x$divergent)) sum(x$divergent) else NA_integer_,
    mean_accept = x$mean_accept %||% (if (!is.null(x$accept_prob)) mean(x$accept_prob) else NA_real_),
    # A nested fit's $converged is per outer grid cell, and a vector here
    # recycles every other column to its length -- so glance() returned one row
    # per cell against a documented single-row contract, and a
    # do.call(rbind, lapply(fits, glance)) silently produced a table whose row
    # count depended on each fit's grid size (gcol33/tulpa#711). Reduce it the
    # way .nested_any_weighted_converged() already reads it.
    converged   = all(x$converged %||% NA),
    stringsAsFactors = FALSE
  )
}

#' Random-effect summaries
#'
#' @details
#' What each backend reports for a group effect follows what it computes:
#' \itemize{
#'   \item Sampler tier and the RE-covariance Gibbs debias
#'     ([tulpa_re_cov_gibbs()], reached by `tulpa(..., control =
#'     list(re_cov = "gibbs"))`) draw the random effects jointly with everything
#'     else, so `estimate` / `sd` / bounds are the empirical posterior summaries.
#'   \item The RE-covariance integrator ([tulpa_re_cov_nested()]) carries a
#'     Gaussian per-group posterior at each `Sigma` node; the reported summaries
#'     are the exact moments and quantiles of the weighted mixture of those, so
#'     they carry both the within-node curvature and the `Sigma` uncertainty.
#'     A group effect the subspace debias selected
#'     (`control$subspace_debias`) is moved by the Metropolis sampler at every
#'     node instead, and is reported from those draws.
#'   \item The Laplace tier reports the conditional mode with no spread (`sd` and
#'     the bounds are `NA`), which is the only per-group quantity it forms.
#' }
#' The `source` column says per row which of these produced it: `"sampled"` for
#' a posterior draw summary, `"mixture"` for the node mixture, `"mode"` for a
#' conditional mode.
#' A fit whose backend never forms a per-group posterior at all (the adaptive
#' Gauss-Hermite inner marginal integrates each group out by quadrature) errors
#' with that reason rather than returning an empty table, which would be
#' indistinguishable from a model with no random effects. A model that genuinely
#' has none returns a zero-row data frame.
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Data frame with one row per random-effect coefficient: `term` (the
#'   group level, and the coefficient for a random slope), `estimate`, `sd`,
#'   the 2.5% / 97.5% bounds `conf.low` / `conf.high`, and `source` (which
#'   construction the row came from). `sd` and the bounds are `NA` on a backend
#'   that reports a point per group (see Details).
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(100), g = factor(rep(1:10, 10)))
#' df$y <- rpois(100, exp(0.3 * df$x))
#' fit <- tulpa(y ~ x + (1 | g), data = df, family = "poisson")
#' ranef(fit)
#' }
#' @export
ranef <- function(object, ...) UseMethod("ranef")

# Random-effect coefficient names implied by a fit's `re_layout` -- one per
# (term, level, coef), matching the order the RE block is stored in: `g[level]`
# for an intercept, `g.slope[level]` for a slope. The single source for the
# labels ranef() attaches, independent of the sampler's full param_names.
#' @keywords internal
.re_names_from_layout <- function(layout) {
  unlist(lapply(layout, function(rt) {
    gv <- rt$group_var; levs <- rt$levels
    cls <- rt$coef_labels %||% "(Intercept)"
    unlist(lapply(levs, function(lev)
      vapply(cls, function(cl)
        if (identical(cl, "(Intercept)")) sprintf("%s[%s]", gv, lev)
        else sprintf("%s.%s[%s]", gv, cl, lev),
        character(1))))
  }), use.names = FALSE) %||% character(0)
}

# Empirical per-coefficient summary of a draw matrix: the estimate / sd /
# bounds block of a ranef table, one row per column of `draws`. The single
# definition every branch that HAS draws reports through, so the sampler tier,
# the RE-covariance Gibbs debias and the per-coordinate overlay below cannot
# summarize the same kind of material three different ways.
#' @keywords internal
.ranef_empirical <- function(draws) {
  data.frame(
    estimate  = colMeans(draws),
    sd        = apply(draws, 2L, stats::sd),
    conf.low  = apply(draws, 2L, stats::quantile, 0.025),
    conf.high = apply(draws, 2L, stats::quantile, 0.975),
    row.names = NULL, stringsAsFactors = FALSE
  )
}

# Overlay the random effects the subspace debias sampled onto a
# Gaussian-mixture per-group table.
#
# A coordinate the selector pulled into S is moved by the Metropolis sampler at
# every integration node, and `tulpa_re_cov_nested()` recombines it on the node
# mixture the fixed-effect draws use (`re_debias_draws`, one column per
# selected coordinate, `re_debias_idx` its row in this table). Those rows are
# summarized empirically -- through `.ranef_empirical()`, the same routine the
# Gibbs branch reports `fit$re` with -- and every other row keeps the Gaussian
# mixture. `source` records which of the two produced each row, so the table
# never leaves the caller to infer that two constructions were interleaved.
#' @keywords internal
.ranef_overlay_sampled <- function(tab, object) {
  D <- object$re_debias_draws
  j <- object$re_debias_idx
  if (!is.matrix(D) || is.null(j) || length(j) != ncol(D) || !nrow(D)) {
    return(tab)
  }
  j <- as.integer(j)
  keep <- which(is.finite(j) & j >= 1L & j <= nrow(tab))
  if (!length(keep)) return(tab)
  emp <- .ranef_empirical(D[, keep, drop = FALSE])
  rows <- j[keep]
  for (cl in c("estimate", "sd", "conf.low", "conf.high")) {
    tab[[cl]][rows] <- emp[[cl]]
  }
  tab$source[rows] <- "sampled"
  tab
}

#' @rdname ranef
#' @export
ranef.tulpa_fit <- function(object, ...) {
  layout <- object$re_layout
  n_fixed <- object$n_fixed %||% 0L
  if (is.null(layout) || length(layout) == 0L) return(data.frame())

  # Nice RE labels from the layout (not the sampler's full param_names, which
  # also carries the latent field and variance-component hyperparameters).
  re_names <- .re_names_from_layout(layout)

  # A backend that fits random-effect terms without ever forming their per-group
  # posterior says so, with its reason. The model HAS random effects, so an empty
  # table would be indistinguishable from one that does not -- the reason is the
  # useful answer and it names the fits that do report them.
  if (is.character(object$ranef_unavailable)) {
    stop("ranef(): this fit carries no per-group random effects -- ",
         object$ranef_unavailable, call. = FALSE)
  }

  re <- .re_coef_draws(object)
  if (!is.null(re)) {
    if (!length(.re_col_idx(colnames(re))) &&
        ncol(re) != length(re_names)) {
      # No `re[`-named columns and the tail width does not match the RE layout:
      # this is the field / hyperparameter latent tail, not identifiable random
      # effects. Do not emit it mislabeled as ranef. (An unnamed matrix whose
      # width DOES match the layout -- e.g. a Gibbs fit's `$re` -- falls through
      # and is used as-is.)
      return(data.frame())
    }
    nm <- if (length(re_names) == ncol(re)) re_names else colnames(re) %||%
      seq_len(ncol(re))
    return(data.frame(
      term = nm,
      .ranef_empirical(re),
      source = "sampled",
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }

  # RE-covariance integrator (tulpa_re_cov_nested): each integration node carries
  # a Gaussian per-group posterior -- the conditional mean in `re_nodes` and its
  # marginal variance in `re_var_nodes` -- and the node weights summarize the
  # Sigma posterior. The marginal per-group posterior is that weighted mixture, so
  # mean / SD are its exact moments and the interval inverts its CDF (rather than
  # a normal approximation around the mean, which a mixture over a skewed Sigma
  # posterior is not). A node set without usable variances reports the mixture
  # mean with NA spread, never the between-node spread alone.
  if (is.matrix(object$re_nodes) && !is.null(object$weights) &&
      ncol(object$re_nodes) == length(re_names)) {
    mx <- .nl_gauss_mixture_summary(object$re_nodes, object$re_var_nodes,
                                    object$weights, probs = c(0.025, 0.975))
    if (!is.null(mx)) {
      return(.ranef_overlay_sampled(data.frame(
        term = re_names, estimate = mx$mean, sd = mx$sd,
        conf.low = mx$quantiles[, 1L], conf.high = mx$quantiles[, 2L],
        source = "mixture",
        row.names = NULL, stringsAsFactors = FALSE
      ), object))
    }
  }

  # Nested-Laplace fit: the BLUPs are the RE tail of each grid cell's latent mode,
  # grid-marginalized against the outer weights. Only emit when the tail width
  # after the fixed block matches the RE layout exactly (else the tail is a latent
  # field, not identifiable random effects -- same guard as the sampler branch).
  # SE is left NA: the between-grid spread alone understates the posterior SD (it
  # omits within-cell curvature), matching how the fixed table refuses a
  # misleadingly small between-grid-only SE.
  if (is.matrix(object$modes) && !is.null(object$weights)) {
    p      <- object$n_fixed %||% 0L
    M      <- object$modes
    n_tail <- ncol(M) - p
    # RE terms carried as `iid` latent blocks share the latent vector with
    # the field / smoother blocks, so the tail after the fixed block is wider than
    # the RE layout and the exact-width guard below cannot fire. The RE blocks are
    # appended LAST, which is what makes them addressable without knowing any
    # other block's width: their coefficients are the final `length(re_names)`
    # columns. SE stays NA for the same reason as the branch below -- the
    # between-grid spread omits the within-cell curvature.
    if (!is.null(object$re_block_index) && n_tail > length(re_names)) {
      w   <- object$weights / sum(object$weights)
      seg <- M[, (ncol(M) - length(re_names) + 1L):ncol(M), drop = FALSE]
      return(data.frame(
        term = re_names, estimate = as.numeric(crossprod(w, seg)),
        sd = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
        source = "mode",
        row.names = NULL, stringsAsFactors = FALSE
      ))
    }
    if (n_tail > 0L && n_tail == length(re_names)) {
      w   <- object$weights / sum(object$weights)
      est <- as.numeric(crossprod(w, M[, (p + 1L):ncol(M), drop = FALSE]))
      return(data.frame(
        term = re_names, estimate = est,
        sd = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
        source = "mode",
        row.names = NULL, stringsAsFactors = FALSE
      ))
    }
    return(data.frame())
  }

  if (!is.null(object$mode) && length(object$mode) > n_fixed) {
    blups <- object$mode[(n_fixed + 1L):length(object$mode)]
    nm <- if (length(re_names) == length(blups)) re_names
          else re_names[seq_len(length(blups))]
    return(data.frame(
      term = nm, estimate = blups,
      sd = NA_real_, conf.low = NA_real_, conf.high = NA_real_,
      source = "mode",
      row.names = NULL, stringsAsFactors = FALSE
    ))
  }
  data.frame()
}

#' Plot fixed-effect posteriors
#'
#' @param x A `tulpa_fit` object.
#' @param type One of `"density"`, `"trace"`, `"pairs"`, `"smooth"`. The Laplace
#'   tier has no draws, so it always shows the Gaussian densities of the fixed
#'   effects. `"smooth"` draws the fitted curve of each `s(...)` term and
#'   requires a fit carrying one.
#' @param term For `type = "smooth"`, which smoother to draw: index or covariate
#'   name. `NULL` (default) draws every one. Ignored by the other types.
#' @param ... Passed to plotting functions.
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   producing base-graphics plots of the fixed-effect posteriors.
#' @export
plot.tulpa_fit <- function(x, type = c("density", "trace", "pairs", "smooth"),
                           term = NULL, ...) {
  type <- match.arg(type)
  if (type == "smooth") return(.plot_smooths(x, term = term, ...))

  if (!.has_draws(x)) {
    tab <- .fit_fixed_table(x)
    np <- nrow(tab)
    old_par <- graphics::par(mfrow = c(min(np, 4), 1), mar = c(4, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(np, 4))) {
      m <- tab$estimate[j]; s <- tab$std.error[j]
      xs <- seq(m - 4 * s, m + 4 * s, length.out = 200)
      plot(xs, stats::dnorm(xs, m, s), type = "l", xlab = tab$term[j], ylab = "density")
      graphics::abline(v = m, col = "red", lty = 2)
    }
    return(invisible(x))
  }

  sub <- .fixed_draws_mat(x)
  p <- min(ncol(sub), 4L)
  sub <- sub[, seq_len(p), drop = FALSE]
  nms <- (x$fixed_names %||% x$param_names %||% colnames(sub))[seq_len(p)]

  if (type == "trace") {
    old_par <- graphics::par(mfrow = c(min(p, 4), 1), mar = c(2, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(p, 4))) {
      plot(sub[, j], type = "l", ylab = nms[j], xlab = "")
    }
  } else if (type == "density") {
    old_par <- graphics::par(mfrow = c(min(p, 4), 1), mar = c(4, 4, 1, 1))
    on.exit(graphics::par(old_par))
    for (j in seq_len(min(p, 4))) {
      d <- stats::density(sub[, j]); plot(d, main = "", xlab = nms[j])
      graphics::abline(v = mean(sub[, j]), col = "red", lty = 2)
    }
  } else if (type == "pairs" && p > 1) {
    graphics::pairs(sub, labels = nms, pch = ".", col = grDevices::rgb(0, 0, 0, 0.2))
  }
  invisible(x)
}


# Rebuild the fixed-effect design matrix for `newdata` from the fit's formula,
# matching how tulpa_build_model_data() built it (model.matrix on the parsed
# fixed-effects formula, response dropped).
#' @keywords internal
.tulpa_fixed_design <- function(object, newdata) {
  parsed <- tulpa_parse_formula(object$formula)
  tt <- stats::delete.response(stats::terms(parsed$fixed_formula))
  mf <- stats::model.frame(tt, data = newdata, na.action = stats::na.pass)
  X  <- stats::model.matrix(tt, mf)
  attr(X, "offset") <- stats::model.offset(mf)
  X
}

# Stop an observation-level accessor on a fit that does not carry what the
# accessor reads, naming the fit's class and the missing piece.
#' @keywords internal
.accessor_unavailable <- function(accessor, object, what) {
  stop(sprintf("%s() is not available for a %s fit: it does not carry %s.",
               accessor, class(object)[1L], what), call. = FALSE)
}

# The designs the observation-level accessors evaluate the linear predictors
# at: the training designs the fit stored (`newdata = NULL`), or the ones
# rebuilt from the fit's formulas at `newdata`. `X_zi` is the zero-inflation
# design (NULL on a fit without one) and `offset` the observation offset --
# the stored one, or the formula's offset() term evaluated on `newdata`.
#' @keywords internal
.tulpa_designs <- function(object, newdata, accessor) {
  if (is.null(newdata)) {
    if (is.null(object$model_matrix)) {
      .accessor_unavailable(accessor, object,
                            "the fixed-effect design ($model_matrix)")
    }
    return(list(X = object$model_matrix, X_zi = object$zi_model_matrix,
                offset = object$offset %||% 0))
  }
  if (is.null(object$formula)) {
    .accessor_unavailable(accessor, object,
                          "the model formula that rebuilds the design at `newdata`")
  }
  X <- .tulpa_fixed_design(object, newdata)
  X_zi <- if (!is.null(object$ziformula)) {
    .zi_design(object$ziformula, newdata, nrow(X))
  }
  list(X = X, X_zi = X_zi, offset = attr(X, "offset") %||% 0)
}

# Point linear predictors at the coefficient estimates: the count predictor
# `eta` (offset included) and, on a zero-inflated fit, the structural-zero logit
# `X_zi beta_zi`. The zero-inflation block of coef() is named by the columns of
# the zero-inflation design; every other coefficient is read against the count
# design. `X` / `X_zi` come back restricted to, and ordered by, the coefficients
# they multiply.
#' @keywords internal
.tulpa_point_linpred <- function(object, newdata, accessor) {
  D     <- .tulpa_designs(object, newdata, accessor)
  beta  <- coef(object)
  zi_nm <- colnames(D$X_zi)
  miss  <- setdiff(names(beta), c(colnames(D$X), zi_nm))
  if (length(miss)) {
    stop(if (is.null(newdata)) "the stored design" else "newdata",
         " cannot reproduce fixed-effect column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  count_nm <- setdiff(names(beta), zi_nm)
  X <- D$X[, count_nm, drop = FALSE]
  out <- list(eta = as.numeric(X %*% beta[count_nm]) + D$offset,
              logit_zi = NULL, X = X, X_zi = NULL, offset = D$offset,
              count_names = count_nm, zi_names = zi_nm)
  if (!is.null(zi_nm)) {
    out$X_zi     <- D$X_zi[, zi_nm, drop = FALSE]
    out$logit_zi <- as.numeric(out$X_zi %*% beta[zi_nm])
  }
  out
}

# FEM projector from the fitted SPDE mesh to arbitrary coordinates. Requires a
# mesh-backed spec built with a coordinate formula; a custom C/G/A spec has no
# mesh to re-project through.
#' @keywords internal
.spde_A_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp$mesh) || is.null(sp$coord_formula)) {
    stop("Spatial-field prediction at new coordinates needs an SPDE spec built ",
         "with a mesh and a coordinate formula, e.g. spatial_spde(~ x + y, ",
         "data); a custom C/G/A spec cannot project to new points. Pass ",
         "include_field = FALSE for the fixed-effect (population) prediction.",
         call. = FALSE)
  }
  vars <- all.vars(sp$coord_formula)
  miss <- setdiff(vars, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  coords <- cbind(newdata[[vars[1]]], newdata[[vars[2]]])
  tulpaMesh::fem_matrices(sp$mesh, obs_coords = coords)$A
}

# Project a fitted SPDE mesh-node field to arbitrary coordinates (kriging):
# A_new %*% w_hat with w_hat the posterior-mean node field.
#' @keywords internal
.spde_field_at <- function(object, newdata) {
  as.numeric(.spde_A_at(object, newdata) %*% object$spatial_effects)
}

# Krige a fitted HSGP field to coordinates (predict()). The field is
# grid-marginalised in C++ (cpp_hsgp_field_predict): for each hyperparameter
# grid cell the per-cell latent (modes tail) is spectral-scaled and projected
# through the Laplacian basis evaluated at the new coordinates, then weighted by
# the cell's posterior weight -- never a plug-in at the posterior mean. Returns
# NULL (decline, leaving eta at the fixed-effect prediction) when the fit is not
# a plain single-block HSGP nested fit (e.g. it carries an extra latent block),
# so the caller can fall through cleanly.
#' @keywords internal
.hsgp_field_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp) || !identical(sp$type, "hsgp") || is.null(object$modes) ||
      is.null(object$weights) || is.null(object$sigma2_grid) ||
      is.null(object$lengthscale_grid) || is.null(sp$coords_matrix)) {
    return(NULL)
  }
  m       <- as.integer(sp[["m"]])
  Mtot    <- m * m
  p_fixed <- object$n_fixed %||% ncol(object$model_matrix)
  modes   <- object$modes
  # The HSGP basis coefficients are the latent block immediately after the fixed
  # effects. Any additional latent tail (an iid RE) means this simple slice is
  # not the field, so decline rather than mis-index.
  if (ncol(modes) != p_fixed + Mtot) return(NULL)
  beta_grid <- modes[, p_fixed + seq_len(Mtot), drop = FALSE]

  coords_train <- as.matrix(sp$coords_matrix)
  if (is.null(newdata)) {
    coords_new <- coords_train
  } else {
    cv <- sp$coord_vars
    miss <- setdiff(cv, names(newdata))
    if (length(miss)) {
      stop("newdata is missing the coordinate column(s): ",
           paste(miss, collapse = ", "), call. = FALSE)
    }
    coords_new <- as.matrix(newdata[, cv, drop = FALSE])
    # Reapply the training coordinate standardisation (scale() recorded the
    # training centre / scale as attributes on coords_matrix) so the prediction
    # basis matches the fitted one.
    if (isTRUE(sp$scale_coords)) {
      ctr <- attr(coords_train, "scaled:center")
      scl <- attr(coords_train, "scaled:scale")
      if (!is.null(ctr) && !is.null(scl)) {
        coords_new <- sweep(sweep(coords_new, 2, ctr, "-"), 2, scl, "/")
      }
    }
  }
  # Strip attributes so the coords reach C++ as plain numeric matrices.
  ct <- .coords_2col(coords_train, "hsgp() prediction")
  cn <- .coords_2col(coords_new, "hsgp() prediction")
  as.numeric(cpp_hsgp_field_predict(
    ct, cn, m, as.numeric(sp[["c"]]),
    beta_grid, as.numeric(object$sigma2_grid),
    as.numeric(object$lengthscale_grid), as.numeric(object$weights)))
}

# Krige a fitted GP / NNGP field to coordinates (predict()). At the training
# locations (newdata = NULL) the field is the grid-marginalised posterior mode
# at each observation's location (exact, no re-kriging). At new coordinates the
# field is the NNGP conditional mean given the fitted field at the location's
# nearest training locations, computed per hyperparameter-grid cell in C++
# (cpp_gp_field_predict, reusing the fit's covariance + conditional kernels) and
# weighted over the grid. Returns NULL (decline) when the fit is not a plain
# single-block GP / NNGP nested fit.
#' @keywords internal
.gp_field_at <- function(object, newdata) {
  sp <- object$spatial
  if (is.null(sp) || !(sp$type %in% c("gp", "nngp")) || is.null(object$modes) ||
      is.null(object$weights) || is.null(object$theta_grid) ||
      is.null(sp$unique_coords)) {
    return(NULL)
  }
  tn <- object$theta_names
  si <- match("sigma2", tn); pj <- match("phi_gp", tn)
  if (is.na(si) || is.na(pj)) return(NULL)
  sigma2_grid <- object$theta_grid[, si]
  phi_grid    <- object$theta_grid[, pj]

  nloc    <- sp$n_unique %||% nrow(sp$unique_coords)
  p_fixed <- object$n_fixed %||% ncol(object$model_matrix)
  modes   <- object$modes
  # The field-at-locations latent is the block after the fixed effects; an extra
  # latent tail (an iid RE) means this slice is not the field, so decline.
  if (ncol(modes) != p_fixed + nloc) return(NULL)
  field_grid <- modes[, p_fixed + seq_len(nloc), drop = FALSE]
  uc <- .coords_plain(sp$unique_coords)

  if (is.null(newdata)) {
    # Training locations: the grid-marginalised field at each unique location,
    # mapped back to observations by obs_to_loc (exact -- no re-kriging).
    field_loc <- as.numeric(as.numeric(object$weights) %*% field_grid)
    otl <- sp$obs_to_loc
    if (is.null(otl)) return(NULL)
    return(field_loc[otl])
  }

  cv   <- sp$coord_vars
  miss <- setdiff(cv, names(newdata))
  if (length(miss)) {
    stop("newdata is missing the coordinate column(s): ",
         paste(miss, collapse = ", "), call. = FALSE)
  }
  ncoord <- as.matrix(newdata[, cv, drop = FALSE])
  if (isTRUE(sp$scale_coords)) {
    ctr <- attr(sp$coords_matrix, "scaled:center")
    scl <- attr(sp$coords_matrix, "scaled:scale")
    if (!is.null(ctr) && !is.null(scl)) {
      ncoord <- sweep(sweep(ncoord, 2, ctr, "-"), 2, scl, "/")
    }
  }
  nc  <- .coords_plain(ncoord)
  nn  <- as.integer(object$prior$nn %||% sp$nn %||% 10L)
  cty <- as.integer(object$prior$cov_type %||% 0L)
  as.numeric(cpp_gp_field_predict(nc, uc, field_grid, sigma2_grid, phi_grid,
                                  as.numeric(object$weights), nn, cty))
}

# The dispersion of an SPDE fit, for the R-side working weights. Every door
# stores `$phi` in the one engine convention (gaussian / lognormal: the
# residual variance), so a front-door tulpa() fit and a direct fit_spde() fit
# read the same way.
#' @keywords internal
.spde_phi_variance <- function(object) {
  object$phi %||% 1.0
}

# Linear-predictor SE at query points for an SPDE fit with the field included:
# Var(x*' beta + a*' w) = c' H^{-1} c with c = [x*, a*] and H the joint
# (beta, field) posterior precision at the fitted hyperparameters -- the
# working-weight cross term X'WA, the field precision Q(range, sigma), and
# the kernel's weak fixed-effect ridge (sigma_beta = 100). Conditional on the
# hyperparameters: a nested fit's grid spread in (range, sigma) is not
# propagated, so the SE is mildly optimistic when the hyperparameter posterior
# is wide. Integer-nu, no-RE fits only; everything else declines loudly.
#' @keywords internal
.spde_linpred_se <- function(object, X_new, A_new) {
  sp <- object$spatial
  if (.spde_nu_is_fractional(sp$nu)) {
    stop("field SE is not available for fractional-nu SPDE fits yet.",
         call. = FALSE)
  }
  X <- object$model_matrix
  if (is.null(X)) {
    stop("field SE needs the training design ($model_matrix); fit through ",
         "tulpa().", call. = FALSE)
  }
  p      <- ncol(X)
  n_mesh <- sp$n_mesh
  mode   <- object$mode
  if (is.null(mode) || length(mode) != p + n_mesh) {
    stop("field SE needs the joint (beta, field) mode with no extra ",
         "random-effect block; this fit's latent layout does not match.",
         call. = FALSE)
  }
  range_val <- object$range %||% object$nested$range_mean
  sigma_val <- object$sigma %||% object$nested$sigma_mean
  if (is.null(range_val) || is.null(sigma_val)) {
    stop("field SE needs the fitted (range, sigma); none stored.",
         call. = FALSE)
  }

  A    <- as(sp$A, "CsparseMatrix")
  beta <- mode[seq_len(p)]
  w    <- mode[p + seq_len(n_mesh)]
  eta  <- as.numeric(X %*% beta) + as.numeric(A %*% w) +
    (object$offset %||% 0)
  W <- glmm_weights(eta, object$family, object$n_trials,
                    .spde_phi_variance(object))

  .kt      <- .spde_kappa_tau(range_val, sigma_val, sp$nu)
  kappa    <- .kt$kappa
  tau_spde <- .kt$tau_spde
  Q <- .spde_precision_Q(sp, kappa, tau_spde)

  XtWX <- crossprod(X, W * X) + diag(1e-4, p)   # kernel ridge sigma_beta = 100
  WA   <- W * A
  AtWA <- Matrix::crossprod(A, WA)
  XtWA <- Matrix::crossprod(Matrix::Matrix(X, sparse = TRUE), WA)
  H <- rbind(
    cbind(Matrix::Matrix(XtWX, sparse = TRUE), XtWA),
    cbind(Matrix::t(XtWA), AtWA + Q)
  )
  Cq <- rbind(
    Matrix::t(Matrix::Matrix(X_new, sparse = TRUE)),
    Matrix::t(as(A_new, "CsparseMatrix"))
  )
  # Per-cell field SE sqrt(colSums(Cq (.) H^{-1} Cq)) is streamed column-by-
  # column in C++ (cpp_spde_field_se): the joint precision is factorized once,
  # then each query column is solved on its own, so the dense working set stays
  # O(p + n_mesh) rather than the (p + n_mesh) x n_cells dense H^{-1} Cq a large
  # prediction grid would otherwise form.
  if (ncol(Cq) == 0L) return(numeric(0))
  cpp_spde_field_se(
    as(as(H,  "generalMatrix"), "CsparseMatrix"),
    as(as(Cq, "generalMatrix"), "CsparseMatrix")
  )
}

#' Fitted values (population level)
#'
#' @description
#' In-sample mean response from the fixed effects and the observation offset
#' (`E[y] = g^{-1}(X beta + offset)`, trial-scaled for binomial). Random
#' effects are held at their prior mean of zero; group-level effects are in
#' [ranef()]. `y - fitted(object)` equals
#' `residuals(object, type = "response")`. On a zero-inflated fit
#' (`ziformula`) the mean is the mixture's, `(1 - pi) E[y | eta]` with
#' `pi = plogis(X_zi beta_zi)`; over a zero-truncated family that is the hurdle
#' mean.
#'
#' @param object A `tulpa_fit` object (must carry `$model_matrix`).
#' @param ... Ignored.
#' @return Numeric vector of fitted mean responses, length `nobs`.
#' @export
fitted.tulpa_fit <- function(object, ...) {
  lp <- .tulpa_point_linpred(object, NULL, "fitted")
  .response_mean(lp$eta, lp$logit_zi, object$family,
                 n_trials = object$n_trials, phi = object$phi %||% 1.0)
}

#' Residuals from a tulpa fit
#'
#' @description
#' Population-level residuals from the fixed-effect fitted mean: `"response"`
#' is `y - E[y | eta]` on the response scale (trial-scaled for binomial,
#' offset included); `"pearson"` additionally scales by the family standard
#' deviation `sqrt(Var(y | eta))` at the fitted linear predictor. Random
#' effects are held at zero, matching [fitted()]. On a zero-inflated fit both
#' the mean and the variance are the mixture's.
#'
#' @param object A `tulpa_fit` object carrying `$y` and `$model_matrix`.
#' @param type `"pearson"` (default) or `"response"`.
#' @param ... Ignored.
#' @return Numeric vector of length `nobs(object)`.
#' @export
residuals.tulpa_fit <- function(object, type = c("pearson", "response"), ...) {
  type <- match.arg(type)
  y <- object$y
  if (is.null(y)) .accessor_unavailable("residuals", object, "the response ($y)")
  lp  <- .tulpa_point_linpred(object, NULL, "residuals")
  phi <- object$phi %||% 1.0
  mu  <- .response_mean(lp$eta, lp$logit_zi, object$family,
                        n_trials = object$n_trials, phi = phi)
  r <- as.numeric(y) - mu
  if (type == "pearson") {
    v <- .response_variance(lp$eta, lp$logit_zi, object$family,
                            n_trials = object$n_trials, phi = phi,
                            phi2 = object$phi2)
    r <- r / sqrt(pmax(v, .Machine$double.eps))
  }
  r
}

#' Number of observations in a tulpa fit
#'
#' @param object A `tulpa_fit` object.
#' @param ... Ignored.
#' @return Integer observation count.
#' @export
nobs.tulpa_fit <- function(object, ...) {
  n <- object$N %||% (if (!is.null(object$y)) length(object$y) else NULL)
  if (is.null(n)) {
    stop("nobs() needs an observation count ($N or $y) on the fit.",
         call. = FALSE)
  }
  as.integer(n)
}

#' Predict at new covariate values (population level)
#'
#' @description
#' Prediction of the linear predictor at `newdata`, on the link or response
#' scale. The fixed-effect part is `X beta` with credible bounds from the
#' fixed-effect covariance ([vcov()]). For a fit carrying a continuous spatial
#' field, the posterior-mean field is interpolated (kriged) to the `newdata`
#' coordinates and added to the linear predictor by default, so `predict()`
#' gives the conditional (location-specific) prediction. Three continuous field
#' families are supported: an SPDE Matern field (`spatial_spde()`), projected
#' through the mesh; a Hilbert-space GP field (`spatial_gp(approx = "hsgp")`),
#' where the Laplacian basis is re-evaluated at the new coordinates (with the
#' training centring / boundary); and a GP / NNGP field (`spatial_gp()`),
#' interpolated by the NNGP conditional mean at each new location's nearest
#' training locations. The HSGP and GP/NNGP fields are marginalised over the
#' hyperparameter grid (not plugged in at the posterior mean). Ordinary random
#' effects are held at zero (population level); add group effects from [ranef()]
#' when needed. An areal (ICAR / BYM2 / CAR) or temporal (RW1 / RW2 / AR1) field
#' is held at zero in the same way, at the training design too; the in-sample
#' linear predictor with every fitted component is what [posterior_predict()]
#' draws and what `compare_models(criterion = "waic")` / `"loo"` score.
#'
#' @param object A `tulpa_fit` object.
#' @param newdata Data frame of covariates (and, for an SPDE fit, the coordinate
#'   columns named in the spec's coordinate formula). If `NULL`, predicts at the
#'   training design (requires `$model_matrix`).
#' @param type `"link"` (linear predictor) or `"response"` (mean scale,
#'   `E[y]`). For a binomial fit the `"response"` scale here is the per-trial
#'   success probability `g^{-1}(eta)` (there is no `n_trials` at `newdata`);
#'   this differs from [fitted()], which returns the trial-scaled expected
#'   count at the training design. On a zero-inflated fit (`ziformula`) the
#'   `"link"` scale is the count predictor and the `"response"` scale is the
#'   mixture mean `(1 - pi) E[y | eta]`, `pi = plogis(X_zi beta_zi)`, with the
#'   zero-inflation design rebuilt from the fit's `ziformula` at `newdata`.
#' @param se.fit If `TRUE`, also return the link-scale standard error and
#'   credible bounds. With an included SPDE field the SE propagates the joint
#'   (fixed-effect, field) posterior precision at the fitted hyperparameters
#'   -- including the cross term -- conditional on `(range, sigma)` (a nested
#'   fit's hyperparameter-grid spread is not propagated, so the bound is
#'   mildly optimistic when that posterior is wide). Integer-nu, no-RE SPDE
#'   fits only; other layouts decline with an explanation. On a zero-inflated
#'   fit `se.fit` is the count predictor's, and the response-scale bounds are
#'   quantiles of the mixture mean over draws of the count and zero-inflation
#'   coefficients jointly (pinned, so repeated calls agree).
#' @param level Credible-interval level (default 0.95).
#' @param include_field For a continuous-spatial fit (SPDE, HSGP, or GP/NNGP),
#'   add the kriged field to the prediction (default `TRUE`). `FALSE` gives the
#'   fixed-effect (population) prediction. Ignored for fits with no continuous
#'   field. For an HSGP or GP/NNGP fit the field is added to the point
#'   prediction but its uncertainty is not yet propagated into `se.fit` (the
#'   interval reflects the fixed-effect covariance only).
#' @param ... Ignored.
#' @return If `se.fit = FALSE`, a numeric vector. If `se.fit = TRUE`, a data
#'   frame with `fit`, `se.fit` (link scale), `lower`, `upper` on the requested
#'   scale.
#' @export
predict.tulpa_fit <- function(object, newdata = NULL,
                              type = c("link", "response"),
                              se.fit = FALSE, level = 0.95,
                              include_field = TRUE, ...) {
  type <- match.arg(type)
  # Count design and linear predictor (offset included, matching
  # fitted()/residuals()), plus the structural-zero logit on a zero-inflated fit.
  lp  <- .tulpa_point_linpred(object, newdata, "predict")
  X   <- lp$X
  eta <- lp$eta

  # Kriged SPDE field. At training data (newdata = NULL) reuse the fitted
  # projector; at new coordinates re-project the mesh-node field through the
  # spec's mesh. Only mesh-backed SPDE fits carry a field here; every other fit
  # leaves `eta` at the fixed-effect (population) prediction. With se.fit the
  # joint (beta, field) posterior precision propagates the field uncertainty
  # (see .spde_linpred_se).
  is_spde_field <- isTRUE(include_field) &&
    identical(object$spatial$type, "spde") &&
    !is.null(object$spatial_effects)
  spde_se <- NULL
  if (is_spde_field) {
    A_new <- if (is.null(newdata)) object$spatial$A
             else .spde_A_at(object, newdata)
    eta <- eta + as.numeric(A_new %*% object$spatial_effects)
    if (se.fit) {
      spde_se <- .spde_linpred_se(object, X, A_new)
    }
  }

  # Kriged HSGP field: add the grid-marginalised Laplacian-basis field at the
  # prediction coordinates. Field-uncertainty SE propagation is not yet wired
  # here (unlike SPDE), so with se.fit the interval reflects the fixed-effect
  # covariance only -- the point prediction still includes the field.
  if (isTRUE(include_field) && identical(object$spatial$type, "hsgp") &&
      !is.null(object$modes)) {
    hsgp_fld <- .hsgp_field_at(object, newdata)
    if (!is.null(hsgp_fld)) eta <- eta + hsgp_fld
  }

  # Kriged GP / NNGP field: the NNGP conditional mean at the prediction
  # coordinates (grid-marginalised). Same se.fit caveat as HSGP.
  if (isTRUE(include_field) && !is.null(object$spatial) &&
      object$spatial$type %in% c("gp", "nngp") && !is.null(object$modes)) {
    gp_fld <- .gp_field_at(object, newdata)
    if (!is.null(gp_fld)) eta <- eta + gp_fld
  }

  ph <- object$phi %||% 1.0
  response_mean <- function(e, z) .response_mean(e, z, object$family, phi = ph)
  if (!se.fit) {
    return(if (type == "response") response_mean(eta, lp$logit_zi) else eta)
  }

  se <- if (!is.null(spde_se)) {
    spde_se
  } else {
    V <- vcov(object)[lp$count_names, lp$count_names, drop = FALSE]
    sqrt(pmax(rowSums((X %*% V) * X), 0))
  }
  z  <- stats::qnorm(1 - (1 - level) / 2)
  lo <- eta - z * se
  hi <- eta + z * se
  if (type == "response") {
    if (is.null(lp$logit_zi)) {
      # One predictor through a monotone mean: the endpoints map through.
      m_lo <- response_mean(lo, NULL)
      m_hi <- response_mean(hi, NULL)
      lo <- pmin(m_lo, m_hi)
      hi <- pmax(m_lo, m_hi)
    } else {
      # The mixture mean moves with two predictors at once, so no endpoint map
      # exists; its quantiles are read over pinned, RNG-neutral draws of the
      # fixed block, the count and zero-inflation coefficients jointly.
      q <- .predict_zi_mean_quantiles(object, lp, eta - lp$eta + lp$offset,
                                      level)
      lo <- q$lower
      hi <- q$upper
    }
    eta <- response_mean(eta, lp$logit_zi)
  }
  data.frame(fit = eta, se.fit = se, lower = lo, upper = hi)
}

# Credible bounds for the zero-inflated mixture mean at the prediction design.
# `shift` is everything predict() adds to the fixed-effect count predictor (the
# offset, a kriged field), held at its point value across draws. The fixed block
# is drawn RNG-neutrally from a pinned seed, so predict() returns the same
# interval on every call and leaves the session stream untouched.
#' @keywords internal
.predict_zi_mean_quantiles <- function(object, lp, shift, level) {
  .preserve_seed_in_frame()
  set.seed(.PREDICT_ZI$seed)
  beta <- .fixed_coef_draws(object, .PREDICT_ZI$ndraws)$beta
  eta  <- sweep(beta[, lp$count_names, drop = FALSE] %*% t(lp$X), 2,
                rep_len(shift, nrow(lp$X)), "+")
  zlog <- beta[, lp$zi_names, drop = FALSE] %*% t(lp$X_zi)
  m <- matrix(.response_mean(eta, zlog, object$family, phi = object$phi %||% 1.0),
              nrow(eta))
  a <- (1 - level) / 2
  list(lower = apply(m, 2, stats::quantile, probs = a, names = FALSE),
       upper = apply(m, 2, stats::quantile, probs = 1 - a, names = FALSE))
}
