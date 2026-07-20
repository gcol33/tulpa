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
# none (a deterministic / point fit). Wraps the erroring extractor.
.compare_fit_loglik <- function(fit) {
  tryCatch(.tulpa_fit_loglik(fit), error = function(e) NULL)
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
#' @return A data frame. For `"loglik"`: `model`, `n_params`, `logLik`. For
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
    ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
    n_par <- fit$n_params %||% fit$n_fixed %||%
      (if (is.matrix(fit$draws)) ncol(fit$draws) else NA_integer_)
      data.frame(model = nm, n_params = n_par, logLik = ll,
                      stringsAsFactors = FALSE)
  })
    return(do.call(rbind, rows))
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

# Latent block types (lower-case) carried on a nested-Laplace fit's `$prior`,
# whether a single block or a list of blocks. Used to route spatial_range() /
# temporal_corr() to the grid-based hyperparameter posterior.
.SPATIAL_NL_TYPES <- c("icar", "bym2", "car_proper", "gp", "nngp", "hsgp",
                       "spde", "rsr", "rsr_spde")
.TEMPORAL_NL_TYPES <- c("rw1", "rw2", "ar1", "iid", "seasonal")

.nested_block_types <- function(object) {
  pr <- object$prior
  if (is.null(pr)) return(character(0))
  if (!is.null(pr$type)) return(tolower(pr$type))
  vapply(pr, function(b) tolower(b$type %||% ""), character(1))
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
  sigma_from_var         = function(v) sqrt(v),
  sigma_from_precision   = function(v) 1 / sqrt(v),
  identity               = function(v) v
)
.SPATIAL_HYPER_TRANSFORM <- list(
  tau         = list(name = "sigma", fn = .hyper_nat$sigma_from_precision),
  sigma2      = list(name = "sigma", fn = .hyper_nat$sigma_from_var),
  phi_gp      = list(name = "range", fn = .hyper_nat$range_from_lengthscale),
  lengthscale = list(name = "range", fn = .hyper_nat$range_from_lengthscale),
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
                                  keep_types = NULL) {
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
  # Multi-block joint grids prefix each axis with its block index
  # (`b<idx>.<axis>`); map every axis to its block type so a mixed / spatiotemporal
  # fit can be restricted to just the spatial (or temporal) axes. Bare (single-
  # block) axis names have no prefix and belong to the sole block.
  block_types <- .nested_block_types(object)
  axis_block  <- suppressWarnings(as.integer(sub("^b([0-9]+)\\..*$", "\\1", axes)))
  axis_kind   <- vapply(seq_along(axes), function(j) {
    bi <- axis_block[j]
    if (!is.na(bi) && bi >= 1L && bi <= length(block_types)) block_types[bi]
    else if (length(block_types) >= 1L) block_types[1L] else NA_character_
  }, character(1))
  bare_axes <- sub("^b[0-9]+\\.", "", axes)
  keep <- if (is.null(keep_types)) seq_along(axes)
          else which(axis_kind %in% keep_types)
  if (length(keep) == 0L) return(NULL)
  rows <- lapply(keep, function(j) {
    v   <- tg[, j]
    tr  <- if (!is.null(transform)) transform[[bare_axes[j]]] else NULL
    nm  <- bare_axes[j]
    if (!is.null(tr)) { v <- tr$fn(v); nm <- tr$name }   # marginalize derived
    m  <- sum(w * v)
    s  <- sqrt(max(0, sum(w * v^2) - m^2))
    qs <- .nl_wtd_quantile(v, w, probs)
    out <- data.frame(mean = m, sd = s, row.names = nm,
                      stringsAsFactors = FALSE)
    out[.quantile_colnames(probs)] <- as.list(qs)
    out
  })
  do.call(rbind, rows)
}

#' Extract spatial range and variance from a fitted spatial model
#'
#' Summarises the posterior of spatial hyperparameters. For sampler-tier fits
#' this reads the raw hyperparameter draws; for a nested-Laplace spatial fit it
#' summarises the outer hyperparameter grid. Works with ICAR, BYM2, GP (NNGP),
#' CAR, SPDE, and SVC spatial types.
#'
#' @param object A `tulpa_fit` object fitted with a spatial component.
#' @param probs Quantile probabilities for the summary (default 0.025, 0.975).
#' @return A data.frame with rows for each spatial hyperparameter and columns
#'   `mean`, `sd`, and one quantile column per entry of `probs` (named from the
#'   probability, e.g. `q2.5`, `q97.5` for the defaults).
#' @export
spatial_range <- function(object, probs = c(0.025, 0.975)) {
  # Nested-Laplace fits carry the hyperparameter posterior on the outer grid,
  # not as draw columns; summarize the grid for a pure-spatial nested fit.
  if (!is.null(object$theta_grid)) {
    types <- .nested_block_types(object)
    if (length(types) && any(types %in% .SPATIAL_NL_TYPES)) {
      s <- .nested_hyper_summary(object, probs,
                                 transform = .SPATIAL_HYPER_TRANSFORM,
                                 keep_types = .SPATIAL_NL_TYPES)
      if (!is.null(s)) return(s)
    }
  }
  patterns <- c(
    range = "^(log_phi_gp|log_phi_gp_local|phi_gp)$",
    sigma = "^(log_sigma2_gp|log_sigma_bym2|log_tau_spatial)$",
    rho   = "^(logit_rho_bym2)$",
    sigma_local  = "^(log_sigma2_gp_local)$",
    sigma_regional = "^(log_sigma2_gp_regional)$",
    range_local  = "^(log_phi_gp_local)$",
    range_regional = "^(log_phi_gp_regional)$"
  )
  transform_fn <- function(nm, raw, label) {
    if (grepl("^log_phi_gp", label)) {
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
    } else if (grepl("^logit_rho", label)) {
      list(vals = 1 / (1 + exp(-raw)), row = nm)       # logit_rho -> rho
    } else {
      list(vals = raw, row = nm)
    }
  }
  .hyperparam_summary(.fit_draws(object), patterns, transform_fn, probs,
                      "No spatial hyperparameters found. Is this a spatial model?")
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


#' Extract temporal correlation parameters from a fitted model
#'
#' Returns posterior summary for temporal hyperparameters (tau, rho for AR1,
#' sigma/lengthscale for temporal GP).
#'
#' @param object A `tulpa_fit` object fitted with a temporal component.
#' @param probs Quantile probabilities (default 0.025, 0.975).
#' @return A data.frame with rows for each temporal hyperparameter.
#' @export
temporal_corr <- function(object, probs = c(0.025, 0.975)) {
  # Nested-Laplace fits carry the hyperparameter posterior on the outer grid,
  # not as draw columns; summarize the grid for a pure-temporal nested fit.
  if (!is.null(object$theta_grid)) {
    types <- .nested_block_types(object)
    if (length(types) && any(types %in% .TEMPORAL_NL_TYPES)) {
      s <- .nested_hyper_summary(object, probs,
                                 transform = .TEMPORAL_HYPER_TRANSFORM,
                                 keep_types = .TEMPORAL_NL_TYPES)
      if (!is.null(s)) return(s)
    }
  }
  patterns <- c(
    tau   = "^log_tau_temporal$",
    rho   = "^logit_rho_ar1$",
    sigma = "^log_sigma2_temporal_gp$",
    lengthscale = "^logit_phi_temporal_gp$"
  )
  transform_fn <- function(nm, raw, label) {
    if (nm == "tau") {
      list(vals = exp(raw), row = "precision")          # log_tau -> tau (precision)
    } else if (nm == "rho") {
      list(vals = 1 / (1 + exp(-raw)), row = "rho_ar1") # logit -> rho in (0,1)
    } else if (nm == "sigma") {
      list(vals = .hyper_nat$sigma_from_var(exp(raw)), row = "sigma_temporal")
    } else if (nm == "lengthscale") {
      list(vals = 1 / (1 + exp(-raw)), row = "lengthscale")
    } else {
      list(vals = raw, row = nm)
    }
  }
  .hyperparam_summary(.fit_draws(object), patterns, transform_fn, probs,
                      "No temporal hyperparameters found. Is this a temporal model?")
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
