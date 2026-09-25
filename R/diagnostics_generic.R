# ============================================================================
# Generic spatial/temporal diagnostics and model comparison
# These operate on residuals or fit objects -- not model-specific.
# ============================================================================

# Assign default model<N> names to an unnamed / partially-named `list(...)`.
.name_models <- function(models) {
  nm <- names(models)
  auto <- paste0("model", seq_along(models))
  names(models) <- if (is.null(nm)) auto else ifelse(nzchar(nm), nm, auto)
  # A non-fit in `...` is almost always a mistyped option (e.g. passing
  # `criterion =` to model_average(), whose option is `weights =`); reject it
  # loudly rather than weight a character string as a candidate model.
  for (k in seq_along(models)) {
    if (is.atomic(models[[k]])) {
      stop(sprintf(paste0("Argument `%s` is not a fitted model. Model-",
                          "comparison options are named parameters (e.g. ",
                          "`criterion` / `weights`), not `...` entries."),
                   names(models)[k]), call. = FALSE)
    }
  }
  models
}

# A fit's [n_draws x n_obs] pointwise log-likelihood, or NULL when it carries
# none (a deterministic / point fit). Dispatches through pointwise_loglik()
# so a model package's own method (e.g. tobs_fit) is reached, not just the
# engine's tulpa_fit layout.
.compare_fit_loglik <- function(fit) {
  tryCatch(pointwise_loglik(fit), error = function(e) NULL)
}

# Per-model criterion summary (elpd, its SE, the effective number of parameters,
# and the pointwise elpd vector for SE-of-difference / stacking). Computed
# natively from the pointwise log-likelihood via tulpa_criteria() + tulpa_psis();
# falls back to a model-supplied WAIC (fit$waic_fn) when there are no pointwise
# draws, and to all-NA when neither is available -- so the comparison degrades
# gracefully to one row per model rather than erroring.
.model_criterion <- function(fit, criterion) {
  na <- list(elpd = NA_real_, se_elpd = NA_real_, p_eff = NA_real_,
             elpd_pw = NULL)
  ll <- .compare_fit_loglik(fit)
  if (!is.null(ll)) {
    cr <- tulpa_criteria(ll, criteria = criterion, pointwise = TRUE)
    if (criterion == "loo")
      return(list(elpd = cr$elpd_loo, se_elpd = cr$se_elpd_loo,
                  p_eff = cr$p_loo, elpd_pw = cr$pointwise$elpd_loo))
    return(list(elpd = cr$elpd_waic, se_elpd = cr$se_elpd_waic,
                p_eff = cr$p_waic, elpd_pw = cr$pointwise$elpd_waic))
  }
  if (criterion == "waic" && !is.null(fit$waic_fn)) {
    w <- tryCatch(fit$waic_fn(fit), error = function(e) NULL)
    if (!is.null(w))
      return(list(elpd = -0.5 * w$waic, se_elpd = NA_real_,
                  p_eff = w$p_waic %||% NA_real_, elpd_pw = NULL))
  }
  na
}

#' Compare models by information criteria
#'
#' @description
#' Rank fitted models best-first by an information criterion. `"waic"` and
#' `"loo"` use the native pointwise log-likelihood layer ([tulpa_criteria()] /
#' [tulpa_psis()]) -- WAIC and PSIS-LOO respectively, computed from each fit's
#' `[n_draws x n_obs]` pointwise log-likelihood (`fit$draws$log_lik`), with no
#' `loo` package dependency. `"loglik"` returns the (integrated) joint
#' log-likelihood with the parameter count. A fit carrying no pointwise
#' log-likelihood (a deterministic / point approximation) yields `NA` criterion
#' columns rather than an error, so the table always has one row per model.
#'
#' @param ... Named `tulpa_fit` objects.
#' @param criterion `"waic"` (default), `"loo"`, or `"loglik"`.
#' @return A data frame. For `"loglik"`: `model`, `n_params`, `logLik`,
#'   `quantity` and `conditioned_on` (which log-scale quantity [logLik()]
#'   reports, and the hyperparameters it holds fixed; fits that disagree on
#'   either are refused rather than ranked). For
#' `"waic"` / `"loo"` (ranked best-first): `model`, `elpd`, `se_elpd`,
#' `p_eff`, `ic` (`-2 * elpd`), `delta` (elpd gap to the best model),
#' `se_diff` (SE of that pointwise elpd difference), and `weight` (the
#' Akaike-style weight on the criterion).
#' @seealso [model_average()] for model-averaged predictions, [tulpa_criteria()]
#' and [tulpa_psis()] for the native criteria layer.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(120))
#' df$y <- rpois(120, exp(0.4 + 0.5 * df$x))
#' f1 <- tulpa(y ~ x, data = df, family = "poisson")
#' f2 <- tulpa(y ~ 1, data = df, family = "poisson")
#' compare_models(full = f1, null = f2, criterion = "waic")
#' }
#' @export
compare_models <- function(..., criterion = c("waic", "loo", "loglik")) {
  criterion <- match.arg(criterion)
  models <- .name_models(list(...))

  if (criterion == "loglik") {
    rows <- lapply(names(models), function(nm) {
      fit <- models[[nm]]
      l <- tryCatch(logLik(fit), error = function(e) NULL)
      n_par <- fit$n_params %||% fit$n_fixed %||%
        (if (is.matrix(fit$draws)) ncol(fit$draws) else NA_integer_)
      cond <- if (is.null(l)) character(0) else attr(l, "conditioned_on") %||% character(0)
      data.frame(model = nm, n_params = n_par,
                 logLik = if (is.null(l)) NA_real_ else as.numeric(l),
                 quantity = if (is.null(l)) NA_character_
                            else attr(l, "quantity") %||% NA_character_,
                 conditioned_on = paste(sort(cond), collapse = ", "),
                 stringsAsFactors = FALSE)
    })
    out <- do.call(rbind, rows)
    # A mean log posterior, a log evidence and a log marginal likelihood
    # conditional on some hyperparameters are different quantities on different
    # scales, and so are two conditional ones holding different hyperparameters
    # fixed. Ranking them in one table is a model choice made on an artefact of
    # how each was fitted (gcol33/tulpa#712, #723). Refuse rather than order.
    key <- ifelse(is.na(out$quantity), NA_character_,
                  paste0(out$quantity, " | ", out$conditioned_on))
    if (length(unique(key[!is.na(key)])) > 1L) {
      lab <- ifelse(nzchar(out$conditioned_on),
                    sprintf("%s = %s (conditional on %s)", out$model,
                            out$quantity, out$conditioned_on),
                    sprintf("%s = %s", out$model, out$quantity))
      stop("compare_models(criterion = \"loglik\") was given fits reporting ",
           "different quantities under logLik(): ", paste(lab, collapse = ", "),
           ". They are not on one scale. Use criterion = \"waic\" or ",
           "\"loo\", which score the pointwise predictive density on every ",
           "tier.", call. = FALSE)
    }
    return(out)
  }

  per     <- lapply(models, .model_criterion, criterion = criterion)
  elpd    <- vapply(per, `[[`, numeric(1), "elpd")
  se_elpd <- vapply(per, `[[`, numeric(1), "se_elpd")
  p_eff   <- vapply(per, `[[`, numeric(1), "p_eff")
  elpd_pw <- lapply(per, `[[`, "elpd_pw")

  ord  <- order(elpd, decreasing = TRUE, na.last = TRUE)
  best <- ord[1]
  delta   <- elpd - elpd[best]
  se_diff <- vapply(seq_along(models), function(k) {
    if (k == best || is.null(elpd_pw[[k]]) || is.null(elpd_pw[[best]]))
      return(NA_real_)
    dd <- elpd_pw[[k]] - elpd_pw[[best]]
    sqrt(length(dd) * stats::var(dd))
  }, numeric(1))
  rel <- exp(delta)                             # delta <= 0 -> Akaike weight on elpd
  weight <- rel / sum(rel, na.rm = TRUE)

  out <- data.frame(model = names(models), elpd = elpd, se_elpd = se_elpd,
                    p_eff = p_eff, ic = -2 * elpd, delta = delta,
                    se_diff = se_diff, weight = weight,
                    row.names = NULL, stringsAsFactors = FALSE)
  out[ord, , drop = FALSE]
}

# Native stacking weights (Yao, Vehtari, Simpson & Gelman 2018): the
# simplex-constrained model combination maximizing the leave-one-out stacked log
# predictive score, parameterized by a softmax of K-1 free coordinates so the
# optimization is unconstrained. `lpd` is the [n_obs x n_models] pointwise elpd.
# No `loo` dependency.
.tulpa_stacking_weights <- function(lpd) {
  K <- ncol(lpd)
  if (K == 1L) return(1)
  m <- apply(lpd, 1L, max)
  E <- exp(lpd - m)                             # exp(lpd_ik - max_k); + m_i drops out
  neg_score <- function(par) {
    a <- c(0, par); w <- exp(a - max(a)); w <- w / sum(w)
    -sum(log(pmax(as.numeric(E %*% w), 1e-300)))
  }
  opt <- stats::optim(rep(0, K - 1L), neg_score, method = "BFGS")
  a <- c(0, opt$par); w <- exp(a - max(a)); w / sum(w)
}

# Native pseudo-BMA / pseudo-BMA+ weights (Yao et al. 2018): a softmax of the
# per-model summed elpd; `bb = TRUE` (pseudo-BMA+) averages that softmax over
# Bayesian-bootstrap (Dirichlet(1)) reweightings of the observations to propagate
# the elpd's sampling uncertainty.
.tulpa_pbma_weights <- function(lpd, bb = FALSE, n_boot = 1000L) {
  K <- ncol(lpd); N <- nrow(lpd)
  softmax <- function(v) { e <- exp(v - max(v)); e / sum(e) }
  if (!bb) return(softmax(colSums(lpd)))
  W <- matrix(0, n_boot, K)
  for (b in seq_len(n_boot)) {
    g <- stats::rexp(N); g <- N * g / sum(g)    # Dirichlet(1,...,1) * N
    W[b, ] <- softmax(colSums(lpd * g))
  }
  colMeans(W)
}

# Latent block types (lower-case) that classify an UNTAGGED nested-Laplace
# block as a spatial or a temporal field (`.nl_block_roles()`); a block tulpa()
# built for `spatial =` / `temporal =` carries its role instead. `iid` is not
# temporal: it is how a `(1 | g)` term rides the nested path, and reading it as
# one reported a random-intercept SD as `sigma_temporal` (gcol33/tulpa#906).
.SPATIAL_NL_TYPES <- c("icar", "bym2", "car_proper", "gp", "nngp", "hsgp",
                       "spde", "rsr", "rsr_spde")
.TEMPORAL_NL_TYPES <- c("rw1", "rw2", "ar1", "seasonal")

# One row per outer-grid axis of a nested-Laplace fit: the bare hyperparameter
# name, the role of the field it belongs to ("spatial" / "temporal" /
# "other"), that block's type, and its index in the prior list. Multi-block
# grids prefix each axis with its block (`b<k>.<axis>`); a single block's axes
# are bare and belong to it; a fit_st_nested() grid is read through its own map.
.nested_axis_map <- function(object, axes) {
  if (inherits(object, "tulpa_st_nested")) return(.st_axis_map(object, axes))
  pr <- object$prior
  blocks <- if (is.null(pr)) list() else if (!is.null(pr$type)) list(pr) else pr
  roles <- .nl_block_roles(pr)
  types <- vapply(blocks, function(b) tolower(b$type %||% ""), character(1))
  bi <- suppressWarnings(as.integer(sub("^b([0-9]+)\\..*$", "\\1", axes)))
  bi[is.na(bi) & length(blocks) >= 1L] <- 1L
  ok <- !is.na(bi) & bi >= 1L & bi <= length(blocks)
  out <- data.frame(bare  = sub("^b[0-9]+\\.", "", axes),
                    role  = ifelse(ok, roles[pmax(bi, 1L)], "other"),
                    type  = ifelse(ok, types[pmax(bi, 1L)], NA_character_),
                    block = ifelse(ok, bi, NA_integer_),
                    lo = NA_real_, hi = NA_real_,
                    stringsAsFactors = FALSE)
  # A proper-CAR correlation lives on its adjacency's eigenvalue interval, which
  # the block carries.
  for (j in which(ok & out$type == "car_proper" & out$bare == "rho")) {
    rb <- blocks[[out$block[j]]]$rho_bounds
    if (length(rb) == 2L) { out$lo[j] <- min(rb); out$hi[j] <- max(rb) }
  }
  out
}

# Natural-scale -> interpretable-quantity maps for the nested-Laplace grid axes,
# so spatial_range() / temporal_corr() report the SAME quantity + row label at
# Tier 2 (nested grid) as at Tier 1 (sampler draws). Each entry composes the
# sampler's own log/logit -> interpretable transform with exp/expit, evaluated
# on the natural-scale grid axis. The derived quantity is computed PER GRID CELL
# and then weighted-summarized (marginalize the derived quantity, not the raw
# axis). Spatial: tau (precision) -> sigma = 1/sqrt(tau); sigma2 -> sigma =
# sqrt(sigma2); phi_gp / lengthscale (GP lengthscale) -> range = 3 * ell (the
# distance at which exp(-d/ell) ~ 0.05). Temporal mirrors the temporal sampler
# (tau reported as precision, ar1 rho, GP sigma / lengthscale).
# Natural-scale maps from a hyperparameter to its interpretable quantity.
# Single-sourced so the nested-grid path (the registries below) and the
# sampler-draw path (the transform_fn closures in spatial_range/temporal_corr)
# cannot drift -- both consume these, differing only in whether they first
# strip a log/logit link.
.hyper_nat <- list(
  range_from_lengthscale = function(v) 3 * v,   # exp(-d/ell) ~ 0.05 at d ~= 3*ell
  # The HSGP / HSGP-SVC basis approximates a SQUARED-EXPONENTIAL kernel,
  # k(d) = sigma^2 exp(-d^2 / (2 ell^2)) (src/hsgp_spectral.h), a different
  # decay convention from the exponential kernel range_from_lengthscale()
  # calibrates: exp(-d^2/(2 ell^2)) = 0.05 at d = ell * sqrt(-2 log(0.05)).
  range_from_se_lengthscale = function(v) sqrt(-2 * log(0.05)) * v,
  sigma_from_var         = function(v) sqrt(v),
  sigma_from_precision   = function(v) 1 / sqrt(v),
  identity               = function(v) v
)

# A continuous-time GP's lengthscale is sampled as a logit onto a declared
# interval, so the draw column is a position in that interval and not the
# lengthscale. These are the engine's own defaults
# (`temporal_gp_phi_prior_lower` / `_upper` and the TVC pair beside them in
# `inst/include/tulpa/model_data.h`), stated for STANDARDIZED times: both GP
# doors lay the interval out as `.gp_phi_bounds()` of the data's spread in the
# units the kernel measures lag in, and send it on the spec.
.GP_PHI_PRIOR_BOUNDS <- c(lower = 0.01, upper = 10)

# The lengthscale support in kernel units, for times whose spread (sd) in those
# units is `spread`. Scaled times (`scale_coords = TRUE`) have spread 1 and get
# the defaults; raw times get the same interval measured in their own spread,
# so the support is one statement about the data whichever units the kernel
# runs in. It used to stay (0.01, 10) in raw units, where a true lengthscale of
# 15 was unreachable (gcol33/tulpa#907).
.gp_phi_bounds <- function(spread = 1) {
  s <- if (is.numeric(spread) && length(spread) == 1L && is.finite(spread) &&
           spread > 0) spread else 1
  .GP_PHI_PRIOR_BOUNDS * s
}

# A lengthscale draw on the natural scale, in the USER's time units: the logit
# mapped onto the spec's own support (kernel units), times the scale the times
# were divided by before they reached the kernel (`time_scale`, 1 when they
# were not). `spec` is the validated temporal_gp() / temporal_tvc() spec.
#' @keywords internal
.gp_phi_from_logit <- function(raw, spec = NULL) {
  lo <- spec$phi_prior_lower %||% .GP_PHI_PRIOR_BOUNDS[["lower"]]
  hi <- spec$phi_prior_upper %||% .GP_PHI_PRIOR_BOUNDS[["upper"]]
  (lo + (hi - lo) / (1 + exp(-raw))) * (spec$time_scale %||% 1)
}
#
# A transformed entry also names the DOMAIN of the quantity it produces, which
# is what a moment-matched interval is formed on when the grid is a quadrature
# design rather than a discretized density (`.nl_summary_quantile`).
# A standard deviation and a range are `positive` whatever
# axis they came from. An identity entry carries no domain of its own and takes
# the AXIS's, from the same registry the outer Pareto-k unconstrains with, so a
# proper-CAR `rho` on the adjacency eigenvalue interval is declined rather than
# guessed.
.SPATIAL_HYPER_TRANSFORM <- list(
  tau         = list(name = "sigma", fn = .hyper_nat$sigma_from_precision,
                     domain = "positive"),
  sigma2      = list(name = "sigma", fn = .hyper_nat$sigma_from_var,
                     domain = "positive"),
  phi_gp      = list(name = "range", fn = .hyper_nat$range_from_lengthscale,
                     domain = "positive"),
  lengthscale = list(name = "range", fn = .hyper_nat$range_from_lengthscale,
                     domain = "positive"),
  sigma       = list(name = "sigma", fn = .hyper_nat$identity),
  range       = list(name = "range", fn = .hyper_nat$identity),
  rho         = list(name = "rho",   fn = .hyper_nat$identity)
)
.TEMPORAL_HYPER_TRANSFORM <- list(
  tau         = list(name = "precision",      fn = .hyper_nat$identity),
  rho         = list(name = "rho_ar1",        fn = .hyper_nat$identity),
  sigma       = list(name = "sigma_temporal", fn = .hyper_nat$identity),
  lengthscale = list(name = "lengthscale",    fn = .hyper_nat$identity)
)

# Weighted per-axis mean / sd / `probs`-quantile summary of a nested-Laplace
# fit's hyperparameter posterior, read from the outer grid + weights. With
# `transform` (an axis-name -> list(name, fn) map) each axis is mapped to its
# interpretable quantity PER CELL before the weighted summary, and the row is
# renamed; axes absent from the map are summarized raw. NULL when the grid is
# not retained.
.nested_hyper_summary <- function(object, probs, transform = NULL,
                                  keep_role = NULL) {
  tg <- object$theta_grid
  w  <- object$weights
  if (is.null(tg) || is.null(w)) return(NULL)
  w <- w / sum(w)
  if (!is.matrix(tg)) {
    tg <- matrix(tg, ncol = 1L,
                 dimnames = list(NULL, object$theta_names %||% "theta"))
  }
  axes <- colnames(tg) %||% object$theta_names %||%
    paste0("theta", seq_len(ncol(tg)))
  # Map every axis to the field it belongs to, so a mixed / spatiotemporal fit
  # can be restricted to just its spatial (or temporal) axes -- by the block's
  # ROLE, since its type does not say (an s(x) smoother is an rw2 block).
  amap <- .nested_axis_map(object, axes)
  bare_axes <- amap$bare
  keep <- if (is.null(keep_role)) seq_along(axes)
          else which(amap$role %in% keep_role)
  if (length(keep) == 0L) return(NULL)
  blocks <- object$prior
  if (!is.null(blocks$type)) blocks <- list(blocks)
  # The outer integrator decides how a quantile may be read off these weights: a
  # tensor grid's uniform cells discretize the density, a CCD is a moment rule
  # whose node positions carry no mass of their own.
  support <- .nl_node_support(object$integration, object$weight_kind)
  # Supplied whatever the support: a moment rule needs the domain to form its
  # interval at all, and a density read needs it to place its outer cell edges
  # inside the quantity's own support.
  geo <- .joint_axis_geometry(object)
  # And the WITHIN-CELL construction the fit's own reported intervals were read
  # with, so a derived-axis summary and `theta_ci_lo` /
  # `theta_ci_hi` on the same fit cannot be built two different ways.
  within <- .nl_within_cell_mode(object$within_cell_requested)
  rows <- lapply(keep, function(j) {
    v   <- tg[, j]
    tr  <- if (!is.null(transform)) transform[[bare_axes[j]]] else NULL
    nm  <- bare_axes[j]
    dm  <- if (length(geo$domain) < j) NA_character_ else geo$domain[[j]]
    at  <- if (length(geo$atom) < j) NA_real_ else geo$atom[[j]]
    if (!is.null(tr)) {                                 # marginalize derived
      v  <- tr$fn(v)
      nm <- tr$name
      if (!is.null(tr$domain)) dm <- tr$domain
      # A declared point mass is a LEVEL of the axis, so it maps with it -- the
      # same rule the cell values take.
      if (is.finite(at)) at <- tr$fn(at)
    }
    m  <- sum(w * v)
    s  <- sqrt(max(0, sum(w * v^2) - m^2))
    # A correlation on a bounded interval other than (0, 1) -- a proper CAR's
    # eigenvalue interval, a spatiotemporal AR1's (-1, 1) -- has no name in the
    # domain vocabulary, so the axis is undeclared and its outer cells were
    # mirrored past the support (q97.5 = 1.04 on an interval ending at 1;
    # gcol33/tulpa#906). Read it on that interval mapped to the unit one, whose
    # domain the partition does honour, and map back.
    lo <- amap$lo[j]; hi <- amap$hi[j]
    rows_j <- .nl_axis_cell_rows(tg, j, object$refining_axis)
    qs <- if (is.finite(lo) && is.finite(hi) && hi > lo &&
              all(v > lo & v < hi)) {
      lo + (hi - lo) * .nl_summary_quantile((v - lo) / (hi - lo), w, probs,
                                            "unit", support, within, NA_real_,
                                            rows_j)
    } else {
      .nl_summary_quantile(v, w, probs, dm, support, within, at, rows_j)
    }
    out <- data.frame(mean = m, sd = s, stringsAsFactors = FALSE)
    out[.quantile_colnames(probs)] <- as.list(qs)
    list(row = out, name = nm)
  })
  out <- do.call(rbind, lapply(rows, `[[`, "row"))
  # Two fields reporting the same quantity (the per-column blocks of an inline
  # spatial() / temporal() field, each with its own `sigma`) are told apart by
  # the block they came from, not by a positional suffix.
  nms <- vapply(rows, `[[`, character(1), "name")
  dup <- nms %in% nms[duplicated(nms)]
  if (any(dup)) {
    tag <- vapply(keep, function(j) {
      b <- amap$block[j]
      bn <- if (!is.na(b) && length(blocks) >= b) blocks[[b]]$name else NULL
      if (is.character(bn) && length(bn) == 1L) bn
      else if (!is.na(b)) paste0("b", b) else axes[j]
    }, character(1))
    nms[dup] <- paste0(nms[dup], "[", tag[dup], "]")
  }
  rownames(out) <- make.unique(nms)
  out
}

#' Extract spatial range and variance from a fitted spatial model
#'
#' Summarises the posterior of spatial hyperparameters. For sampler-tier fits
#' this reads the raw hyperparameter draws; for a nested-Laplace spatial fit it
#' summarises the outer hyperparameter grid. Works with ICAR, BYM2, GP (NNGP),
#' CAR, SPDE, and SVC spatial types.
#'
#' A range is reported in the units of the coordinates as supplied, also when
#' the field was fitted on standardized coordinates (`scale_coords = TRUE`,
#' the default of [spatial_gp()] and [spatial_svc()]). A `mode = "laplace"` fit
#' conditions on the field's hyperparameters rather than inferring them, so it
#' carries no posterior for this function to summarise.
#'
#' @param object A `tulpa_fit` object fitted with a spatial component.
#' @param probs Quantile probabilities for the summary (default 0.025, 0.975).
#' @return A data.frame with rows for each spatial hyperparameter and columns
#'   `mean`, `sd`, and one quantile column per entry of `probs` (named from the
#'   probability, e.g. `q2.5`, `q97.5` for the defaults).
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 12L
#' adj <- matrix(0, S, S)
#' for (i in 1:(S - 1)) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' d <- data.frame(region = factor(rep(1:S, each = 5)), x = rnorm(S * 5))
#' field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.5))))
#' d$y <- rpois(nrow(d), exp(0.3 + 0.4 * d$x + field[d$region]))
#' fit <- tulpa(y ~ x + spatial(region), data = d, family = "poisson",
#'              spatial = spatial_car(adj, level = "group", group_var = "region"))
#' spatial_range(fit)
#' }
#' @export
spatial_range <- function(object, probs = c(0.025, 0.975)) {
  # The kernel ran on coordinates divided by one common factor; a range is a
  # distance, so it converts back to the user's units by that factor
  # (gcol33/tulpa#907).
  .range_to_data_units(.spatial_range_kernel_units(object, probs),
                       .coord_scale(object$spatial))
}

# Multiply the `range*` rows of a hyperparameter summary by the coordinate
# scale `k`. Every column is a location or scale statistic of a positive
# quantity, so each maps by the same factor.
.range_to_data_units <- function(s, k) {
  if (!is.data.frame(s) || identical(k, 1)) return(s)
  r <- startsWith(rownames(s), "range")
  s[r, ] <- s[r, , drop = FALSE] * k
  s
}

# The empty-draws message of spatial_range() / temporal_corr(). A fit that
# carries the field (`what` names it) but no hyperparameter posterior is one
# that conditioned on the hyperparameters -- `mode = "laplace"` -- which is what
# to say rather than asking whether the model has the field at all
# (gcol33/tulpa#906).
.hyper_empty_msg <- function(object, field, what) {
  if (is.null(object[[field]])) {
    return(sprintf("No %s hyperparameters found. Is this a %s model?", what, what))
  }
  sprintf(paste0(
    "This fit carries a %s field but no posterior over its hyperparameters: ",
    "mode = \"laplace\" conditions on them rather than inferring them. Refit ",
    "with mode = \"nested_laplace\" (integrates them) or a sampler mode ",
    "(e.g. \"hmc\") to summarise them."), what)
}

.spatial_range_kernel_units <- function(object, probs) {
  # Nested-Laplace fits carry the hyperparameter posterior on the outer grid,
  # not as draw columns; summarize the grid for a pure-spatial nested fit.
  if (!is.null(object$theta_grid)) {
    s <- .nested_hyper_summary(object, probs,
                               transform = .SPATIAL_HYPER_TRANSFORM,
                               keep_role = "spatial")
    if (!is.null(s)) return(s)
  }
  # fit_spde()'s CCD / grid path (the "auto"/nested SPDE backend) carries its
  # own (range, sigma) outer-grid posterior in $nested rather than a generic
  # $theta_grid or draw columns: R/fit_spde_nested.R stores $range / $sigma on
  # the fit as the single anchor point the fixed effects were refit at, so
  # reading those instead of $nested would report a point, not a posterior.
  if (!is.null(object$nested) && !is.null(object$nested$range_grid) &&
      !is.null(object$nested$sigma_grid) && !is.null(object$nested$weights)) {
    return(.wtd_grid_summary(
      list(range = object$nested$range_grid, sigma = object$nested$sigma_grid),
      object$nested$weights, probs))
  }
  # The sampler's own column names (src/sampler_model_data.h): BYM2 samples
  # its total SD as `log_sigma_spatial` and its mixing weight as
  # `logit_rho_bym2`; a proper CAR its precision as `log_tau_spatial` and its
  # correlation as `logit_rho_car`. The BYM2 SD and the proper-CAR correlation
  # were missing from this table, so those fits reported one hyperparameter of
  # two (gcol33/tulpa#906).
  patterns <- c(
    range = "^(log_phi_gp|log_phi_gp_local|phi_gp)$",
    sigma = "^(log_sigma2_gp|log_sigma_bym2|log_sigma_spatial|log_tau_spatial)$",
    rho   = "^(logit_rho_bym2|logit_rho_car)$",
    sigma_local  = "^(log_sigma2_gp_local)$",
    sigma_regional = "^(log_sigma2_gp_regional)$",
    range_local  = "^(log_phi_gp_local)$",
    range_regional = "^(log_phi_gp_regional)$",
    # HSGP squared-exponential basis: src/tulpa_priors_hsgp.h.
    sigma_hsgp = "^log_sigma2_hsgp$",
    range_hsgp = "^log_lengthscale_hsgp$",
    # SVC: src/tulpa_priors_svc.h. The NNGP approximation's phi is a direct
    # exponential-kernel range like log_phi_gp; the HSGP approximation's is a
    # squared-exponential lengthscale like log_lengthscale_hsgp above -- same
    # draw-column name, different kernel, so the transform below reads
    # object$spatial$approx (spatial_svc()'s own spec, carried on every SVC
    # fit regardless of inference mode) to pick the right one.
    sigma_svc = "^log_sigma2_svc\\[[0-9]+\\]$",
    range_svc = "^log_phi_svc\\[[0-9]+\\]$"
  )
  transform_fn <- function(nm, raw, label) {
    if (grepl("^log_lengthscale_hsgp$", label)) {
      list(vals = .hyper_nat$range_from_se_lengthscale(exp(raw)), row = "range")
    } else if (grepl("^log_phi_svc", label)) {
      se <- identical(tolower(object$spatial$approx %||% "nngp"), "hsgp")
      fn <- if (se) .hyper_nat$range_from_se_lengthscale
            else .hyper_nat$range_from_lengthscale
      list(vals = fn(exp(raw)), row = sub("phi", "range", nm))
    } else if (grepl("^log_phi_gp", label)) {
      # phi is the range: every kernel is exp(-d / phi), so correlation decays
      # to ~0.05 at d = -phi * log(0.05) ~= 3 * phi.
      list(vals = .hyper_nat$range_from_lengthscale(exp(raw)),
           row = sub("phi", "range", nm))
    } else if (grepl("^log_sigma2", label)) {
      list(vals = .hyper_nat$sigma_from_var(exp(raw)), row = nm)
    } else if (grepl("^log_tau", label)) {
      list(vals = .hyper_nat$sigma_from_precision(exp(raw)), row = "sigma")
    } else if (grepl("^log_sigma", label)) {
      list(vals = exp(raw), row = nm)                  # log_sigma -> sigma
    } else if (identical(label, "logit_rho_car")) {
      # rho = lower + (upper - lower) * invlogit, on the adjacency's eigenvalue
      # interval (src/tulpa_priors_icar.h), not the unit one.
      rb <- object$spatial$rho_bounds %||%
        compute_car_rho_bounds(object$spatial$adjacency)
      list(vals = min(rb) + abs(diff(rb)) / (1 + exp(-raw)), row = nm)
    } else if (grepl("^logit_rho", label)) {
      list(vals = 1 / (1 + exp(-raw)), row = nm)       # logit_rho -> rho
    } else {
      list(vals = raw, row = nm)
    }
  }
  .hyperparam_summary(.fit_draws(object), patterns, transform_fn, probs,
                      .hyper_empty_msg(object, "spatial", "spatial"))
}

# Column names for a set of quantile probabilities, e.g. c(0.025, 0.975) ->
# c("q2.5", "q97.5"). Shared by spatial_range() and temporal_corr() so the
# reported columns always match the requested `probs`.
.quantile_colnames <- function(probs) {
  pct <- sprintf("%.6f", probs * 100)
  paste0("q", sub("\\.?0+$", "", pct))
}

# Shared scaffold for the spatial_range() / temporal_corr() hyperparameter
# summaries: find the first draw column matching each anchored `patterns` entry,
# map its draws to the natural scale via `transform_fn(nm, raw, label)` (returns
# `list(vals, row)`), and stack the per-parameter mean / sd / `probs`-quantile
# summaries. Errors with `empty_msg` when no hyperparameter column is present.
.hyperparam_summary <- function(draws, patterns, transform_fn, probs, empty_msg) {
  cn <- colnames(draws)
  found <- list()
  for (nm in names(patterns)) {
    idx <- grep(patterns[[nm]], cn)
    if (length(idx) > 0) found[[nm]] <- idx[1]
  }
  if (length(found) == 0) stop(empty_msg, call. = FALSE)
  rows <- lapply(names(found), function(nm) {
    tr  <- transform_fn(nm, draws[, found[[nm]]], cn[found[[nm]]])
    qs  <- stats::quantile(tr$vals, probs = probs)
    out <- data.frame(mean = mean(tr$vals), sd = stats::sd(tr$vals),
                      row.names = tr$row, stringsAsFactors = FALSE)
    out[.quantile_colnames(probs)] <- as.list(qs)
    out
  })
  do.call(rbind, rows)
}

# Weighted quantile by interpolated ECDF, for a weighted outer-grid posterior
# that carries no density/volume of its own (a CCD moment-rule design), so the
# grid-cell machinery `.nl_summary_quantile()` reads is not applicable. `v` and
# `w` are the grid's own points and (already-normalized) weights.
.ps_wtd_quantile <- function(v, w, probs) {
  ord <- order(v)
  v <- v[ord]; w <- w[ord]
  cw <- cumsum(w) / sum(w)
  stats::approx(cw, v, xout = probs, rule = 2, ties = "ordered")$y
}

# Weighted mean / sd / `probs`-quantile summary row for a named [range, sigma,
# ...] list of outer-grid quantities sharing one weight vector `w` -- the SPDE
# `$nested` CCD/grid representation, which carries neither `$theta_grid`
# (the generic nested-Laplace multi-block grid) nor draw columns.
.wtd_grid_summary <- function(named_vals, w, probs) {
  w <- w / sum(w)
  rows <- lapply(names(named_vals), function(nm) {
    v  <- named_vals[[nm]]
    m  <- sum(w * v)
    qs <- .ps_wtd_quantile(v, w, probs)
    out <- data.frame(mean = m, sd = sqrt(max(0, sum(w * v^2) - m^2)),
                      row.names = nm, stringsAsFactors = FALSE)
    out[.quantile_colnames(probs)] <- as.list(qs)
    out
  })
  do.call(rbind, rows)
}


#' Extract temporal correlation parameters from a fitted model
#'
#' Returns posterior summary for temporal hyperparameters (tau, rho for AR1,
#' sigma/lengthscale for temporal GP).
#'
#' @param object A `tulpa_fit` object fitted with a temporal component.
#' @param probs Quantile probabilities (default 0.025, 0.975).
#' @return A data.frame with rows for each temporal hyperparameter.
#' @examples
#' \donttest{
#' set.seed(127)
#' df <- data.frame(year = rep(1:20, each = 3), x = rnorm(60))
#' f <- as.numeric(arima.sim(list(ar = 0.7), 20, sd = 0.4))
#' df$count <- rpois(60, exp(1 + 0.3 * df$x + f[df$year]))
#' fit <- tulpa(count ~ x, data = df, family = "poisson",
#'              temporal = temporal_ar1("year"))
#' temporal_corr(fit)
#' }
#' @export
temporal_corr <- function(object, probs = c(0.025, 0.975)) {
  # Nested-Laplace fits carry the hyperparameter posterior on the outer grid,
  # not as draw columns; summarize the grid for a pure-temporal nested fit.
  if (!is.null(object$theta_grid)) {
    s <- .nested_hyper_summary(object, probs,
                               transform = .TEMPORAL_HYPER_TRANSFORM,
                               keep_role = "temporal")
    if (!is.null(s)) return(s)
    # latent(temporal_ar2()) / latent(temporal_ar()): a user-defined tgmrf
    # block tagged tulpa_temporal_latent_block (R/temporal_ar2.R) rather than
    # one of the built-in temporal block types above, so `.nl_block_roles()`
    # classifies its generic type = "tgmrf" as neither field.
    s <- .nl_ar_p_hyper_summary(object, probs)
    if (!is.null(s)) return(s)
  }
  patterns <- c(
    tau   = "^log_tau_temporal$",
    rho   = "^logit_rho_ar1$",
    sigma = "^log_sigma2_temporal_gp$",
    lengthscale = "^logit_phi_temporal_gp$",
    # temporal_multiscale(): src/tulpa_priors_mstemporal.h.
    sigma_trend    = "^log_sigma2_trend$",
    sigma_seasonal = "^log_sigma2_seasonal$",
    sigma_short    = "^log_sigma2_short$",
    rho_short      = "^logit_rho_short$",
    # temporal_tvc(): src/tulpa_priors_tvc.h. The discrete structures sample a
    # log-precision (+ AR1's correlation); a GP-evolving coefficient samples an
    # amplitude and a lengthscale per coefficient instead (gcol33/tulpa#847).
    tau_tvc = "^log_tau_tvc\\[[0-9]+\\]$",
    rho_tvc = "^logit_rho_tvc\\[[0-9]+\\]$",
    sigma_tvc_gp       = "^log_sigma2_tvc_gp\\[[0-9]+\\]$",
    lengthscale_tvc_gp = "^logit_phi_tvc_gp\\[[0-9]+\\]$"
  )
  transform_fn <- function(nm, raw, label) {
    if (nm == "tau") {
      list(vals = exp(raw), row = "precision")          # log_tau -> tau (precision)
    } else if (nm == "rho") {
      list(vals = 1 / (1 + exp(-raw)), row = "rho_ar1") # logit -> rho in (0,1)
    } else if (nm == "sigma") {
      list(vals = .hyper_nat$sigma_from_var(exp(raw)), row = "sigma_temporal")
    } else if (nm == "lengthscale") {
      list(vals = .gp_phi_from_logit(raw, object$temporal), row = "lengthscale")
    } else if (grepl("^log_sigma2_tvc_gp", label)) {
      list(vals = .hyper_nat$sigma_from_var(exp(raw)), row = nm)
    } else if (grepl("^logit_phi_tvc_gp", label)) {
      list(vals = .gp_phi_from_logit(raw, object$temporal), row = nm)
    } else if (grepl("^log_sigma2_(trend|seasonal|short)$", label)) {
      list(vals = .hyper_nat$sigma_from_var(exp(raw)), row = nm)
    } else if (nm == "rho_short") {
      # rho = 2 * invlogit(logit_rho_short) - 1, mapped to (-1, 1)
      # (src/tulpa_priors_mstemporal.h), not the (0, 1) of logit_rho_ar1 above.
      list(vals = 2 / (1 + exp(-raw)) - 1, row = "rho_short")
    } else if (grepl("^log_tau_tvc", label)) {
      list(vals = exp(raw), row = nm)                   # log_tau -> tau (precision)
    } else if (grepl("^logit_rho_tvc", label)) {
      list(vals = 2 / (1 + exp(-raw)) - 1, row = nm)     # (-1, 1), like rho_short
    } else {
      list(vals = raw, row = nm)
    }
  }
  .hyperparam_summary(.fit_draws(object), patterns, transform_fn, probs,
                      .hyper_empty_msg(object, "temporal", "temporal"))
}

# spatial_range()/temporal_corr()'s draw-column table and `.nl_block_roles()`
# both key off a block's string `type` or role tag; a latent(temporal_ar2()
# / temporal_ar()) block is a generic tgmrf() (type = "tgmrf") rather than one of
# the named built-in types, so it matches neither and is found instead via its
# class tag (R/temporal_ar2.R, the same tag temporal.tulpa_fit() uses through
# .nl_temporal_latent_block(), R/temporal_rtr_posteriors.R). Its grid axes are on
# the block's own unconstrained scale (log_tau, atanh_psi1..atanh_psi<p>) rather
# than the natural scale the built-in temporal blocks lay their grid on, so the
# transform composes exp() / tanh() with the natural-quantity map instead of
# reading .TEMPORAL_HYPER_TRANSFORM's identity entries.
.nl_ar_p_hyper_summary <- function(object, probs) {
  blocks <- object$blocks
  tg <- object$theta_grid
  w  <- object$weights
  if (!is.list(blocks) || !length(blocks) || is.null(tg) || is.null(w)) {
    return(NULL)
  }
  k <- which(vapply(blocks, inherits, logical(1),
                    what = "tulpa_temporal_latent_block"))
  if (!length(k)) return(NULL)
  k <- k[1L]
  if (!is.matrix(tg)) {
    tg <- matrix(tg, ncol = 1L,
                 dimnames = list(NULL, object$theta_names %||% "theta"))
  }
  axes <- colnames(tg) %||% paste0("theta", seq_len(ncol(tg)))
  sel  <- grepl(sprintf("^b%d\\.", k), axes)
  if (!any(sel)) return(NULL)
  bare <- sub(sprintf("^b%d\\.", k), "", axes[sel])
  cols <- which(sel)
  w    <- w / sum(w)
  rows <- lapply(seq_along(bare), function(j) {
    raw <- tg[, cols[j]]
    if (identical(bare[j], "log_tau")) {
      list(vals = exp(raw), row = "precision")
    } else if (grepl("^atanh_psi[0-9]+$", bare[j])) {
      list(vals = tanh(raw), row = sub("^atanh_", "", bare[j]))
    } else {
      list(vals = raw, row = bare[j])
    }
  })
  .wtd_grid_summary(stats::setNames(lapply(rows, `[[`, "vals"),
                                    vapply(rows, `[[`, character(1), "row")),
                    w, probs)
}


#' Fit a post-hoc linear model on estimated parameters
#'
#' Useful for exploring drivers of occupancy/detection/abundance variation
#' after model fitting. Fits a weighted linear model and optionally generates
#' bootstrap confidence intervals.
#'
#' @param formula Model formula (e.g., `psi_hat ~ trait1 + trait2`).
#' @param data A data.frame with response and predictors.
#' @param weights Optional weights (e.g., inverse of standard errors).
#' @param n_boot Number of bootstrap replicates for CI (default 1000, 0 to skip).
#' @param probs Quantile probabilities for bootstrap CI (default 0.025, 0.975).
#'
#' @importFrom stats lm quantile
#' @return A list of class `"post_hoc_lm"` with:
#'   \describe{
#'     \item{summary}{data.frame of coefficient estimates and CIs}
#'     \item{lm_fit}{the underlying `lm` object}
#'     \item{boot_coefs}{matrix of bootstrap coefficient samples (if `n_boot > 0`)}
#'     \item{R2}{R-squared from the fitted model}
#'   }
#' @examples
#' # Explore drivers of per-site estimates after fitting a model.
#' site <- data.frame(
#'   psi_hat = c(0.2, 0.5, 0.8, 0.4, 0.6, 0.3),
#'   se      = c(0.05, 0.04, 0.06, 0.05, 0.03, 0.05),
#'   trait   = c(1.0, 2.5, 3.8, 1.9, 3.1, 1.2)
#' )
#' fit <- post_hoc_lm(psi_hat ~ trait, data = site,
#'                    weights = 1 / site$se^2, n_boot = 200L)
#' fit
#' @export
post_hoc_lm <- function(formula, data, weights = NULL,
                        n_boot = 1000L, probs = c(0.025, 0.975)) {
  # lm() resolves a bare `weights` symbol in the formula's environment, not this
  # function's, so a local weights vector is passed as a data column referenced
  # by name in the model frame.
  lm_fit <- if (is.null(weights)) {
    lm(formula, data = data)
  } else {
    fit_data <- data
    fit_data[[".phl_w"]] <- weights
    lm(formula, data = fit_data, weights = .phl_w)
  }

  coefs <- summary(lm_fit)$coefficients
  result_summary <- data.frame(
    term = rownames(coefs),
    estimate = coefs[, "Estimate"],
    std.error = coefs[, "Std. Error"],
    statistic = coefs[, "t value"],
    p.value = coefs[, "Pr(>|t|)"],
    stringsAsFactors = FALSE, row.names = NULL
  )

  boot_coefs <- NULL
  if (n_boot > 0) {
    n <- nrow(data)
    boot_coefs <- matrix(NA_real_, n_boot, length(coef(lm_fit)))
    colnames(boot_coefs) <- names(coef(lm_fit))
    for (b in seq_len(n_boot)) {
      idx <- sample.int(n, n, replace = TRUE)
      boot_data <- data[idx, , drop = FALSE]
      boot_wts <- if (!is.null(weights)) weights[idx] else NULL
      boot_fit <- tryCatch(
        if (is.null(boot_wts)) {
          lm(formula, data = boot_data)
        } else {
          boot_data[[".phl_w"]] <- boot_wts
          lm(formula, data = boot_data, weights = .phl_w)
        },
        error = function(e) NULL
      )
      if (!is.null(boot_fit)) boot_coefs[b, ] <- coef(boot_fit)
    }
    # Add bootstrap CIs to summary
    for (j in seq_len(ncol(boot_coefs))) {
      qs <- quantile(boot_coefs[, j], probs = probs, na.rm = TRUE)
      result_summary$conf.low[j] <- qs[1]
      result_summary$conf.high[j] <- qs[2]
    }
  }

  out <- list(
    summary = result_summary,
    lm_fit = lm_fit,
    boot_coefs = boot_coefs,
    R2 = summary(lm_fit)$r.squared
  )
  class(out) <- "post_hoc_lm"
  out
}

#' @export
print.post_hoc_lm <- function(x, ...) {
  cat("Post-hoc linear model\n")
  cat(sprintf("R-squared: %.3f\n\n", x$R2))
  print(x$summary, row.names = FALSE, ...)
  invisible(x)
}


#' Model-averaged predictions
#'
#' @description
#' Combine fitted values from several models using native model weights computed
#' from the pointwise PSIS-LOO (or WAIC) elpd via [tulpa_psis()] -- no `loo`
#' package dependency. `"loo"` / `"waic"` give stacking weights (the
#' simplex-optimal predictive combination); `"pbma"` / `"pbma+"` give
#' pseudo-BMA(+) weights. Every model must carry an `[n_draws x n_obs]` pointwise
#' log-likelihood (`fit$draws$log_lik`) over the same observations.
#'
#' @param ... Named `tulpa_fit` objects fitted to the same observations.
#' @param weights `"loo"` (stacking, default), `"waic"`, `"pbma"`, or `"pbma+"`.
#' @param fitted_fn Function extracting a length-`n_obs` fitted vector from a fit
#' (default [fitted()]).
#' @return A list with `averaged` (the weighted fitted vector), `weights` (the
#' named model weights), and `comparison` (the [compare_models()] table).
#' @seealso [compare_models()].
#' @references Yao, Vehtari, Simpson & Gelman (2018). Using stacking to average
#' Bayesian predictive distributions. \emph{Bayesian Analysis} 13(3):917-1007.
#' @examples
#' \donttest{
#' set.seed(1)
#' df <- data.frame(x = rnorm(120))
#' df$y <- rpois(120, exp(0.4 + 0.5 * df$x))
#' f1 <- tulpa(y ~ x, data = df, family = "poisson", mode = "hmc",
#'             control = list(n_iter = 500L, warmup = 250L, seed = 1L))
#' f2 <- tulpa(y ~ 1, data = df, family = "poisson", mode = "hmc",
#'             control = list(n_iter = 500L, warmup = 250L, seed = 1L))
#' ma <- model_average(full = f1, null = f2, weights = "waic")
#' ma$weights
#' }
#' @export
model_average <- function(..., weights = c("loo", "waic", "pbma", "pbma+"),
                          fitted_fn = fitted) {
  weights <- match.arg(weights)
  models  <- .name_models(list(...))
  crit    <- if (weights == "waic") "waic" else "loo"

  pw <- lapply(models, function(fit) {
    ll <- .compare_fit_loglik(fit)
    if (is.null(ll))
      stop("Model averaging needs every fit's pointwise log-likelihood ",
           "(fit$draws$log_lik); a model without it cannot be weighted.",
           call. = FALSE)
    cr <- tulpa_criteria(ll, criteria = crit, pointwise = TRUE)
    if (crit == "loo") cr$pointwise$elpd_loo else cr$pointwise$elpd_waic
  })
  if (length(unique(vapply(pw, length, integer(1)))) != 1L)
    stop("All models must be fitted to the same observations ",
         "(pointwise log-likelihoods differ in length).", call. = FALSE)
  lpd <- do.call(cbind, pw)                     # [n_obs x n_models]

  w <- switch(weights,
              loo     = .tulpa_stacking_weights(lpd),
              waic    = .tulpa_stacking_weights(lpd),
              pbma    = .tulpa_pbma_weights(lpd, bb = FALSE),
              `pbma+` = .tulpa_pbma_weights(lpd, bb = TRUE))
  names(w) <- names(models)

  fits <- lapply(models, fitted_fn)
  first_vals <- lapply(fits, function(f) if (is.list(f)) f[[1]] else f)
  n <- length(first_vals[[1]])
  avg <- numeric(n)
  for (k in seq_along(models)) avg <- avg + w[k] * first_vals[[k]]
  list(averaged = avg, weights = w,
       comparison = compare_models(..., criterion = crit))
}
