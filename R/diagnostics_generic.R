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
  rel <- exp(0.5 * delta)                       # delta <= 0 -> Akaike weight on elpd
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
# No `loo` dependency (gcol33/tulpa#103).
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

#' Extract spatial range and variance from a fitted spatial model
#'
#' Summarises posterior draws of spatial hyperparameters.
#' Works with ICAR, BYM2, GP (NNGP), and SVC spatial types.
#'
#' @param object A `tulpa_fit` object fitted with a spatial component.
#' @param probs Quantile probabilities for the summary (default 0.025, 0.975).
#' @return A data.frame with rows for each spatial hyperparameter and columns
#'   `mean`, `sd`, `q025`, `q975`.
#' @export
spatial_range <- function(object, probs = c(0.025, 0.975)) {
  draws <- object$draws
  cn <- colnames(draws)

  # Map column name patterns to interpretable parameter names
  patterns <- c(
    range = "log_phi_gp|log_phi_gp_local|phi_gp",
    sigma = "log_sigma2_gp|log_sigma_bym2|log_tau_spatial",
    rho   = "logit_rho_bym2",
    sigma_local  = "log_sigma2_gp_local",
    sigma_regional = "log_sigma2_gp_regional",
    range_local  = "log_phi_gp_local",
    range_regional = "log_phi_gp_regional"
  )

  found <- list()
  for (nm in names(patterns)) {
    idx <- grep(paste0("^(", patterns[nm], ")$"), cn)
    if (length(idx) > 0) found[[nm]] <- idx[1]
  }

  if (length(found) == 0) {
    stop("No spatial hyperparameters found. Is this a spatial model?", call. = FALSE)
  }

  # Build summary rows
  rows <- lapply(names(found), function(nm) {
    raw <- draws[, found[[nm]]]
    # Transform to natural scale
    label <- cn[found[[nm]]]
    if (grepl("^log_phi_gp", label)) {
      # phi is the range: every kernel is exp(-d / phi), so correlation decays
      # to ~0.05 at d = -phi * log(0.05) ~= 3 * phi.
      vals <- 3 * exp(raw)
      row_name <- sub("phi", "range", nm)
    } else if (grepl("^log_sigma2", label)) {
      vals <- sqrt(exp(raw))  # log_sigma2 -> sigma
      row_name <- nm
    } else if (grepl("^log_tau", label)) {
      vals <- 1 / sqrt(exp(raw))  # log_tau -> sigma = 1/sqrt(tau)
      row_name <- "sigma"
    } else if (grepl("^log_sigma", label)) {
      vals <- exp(raw)  # log_sigma -> sigma
      row_name <- nm
    } else if (grepl("^logit_rho", label)) {
      vals <- 1 / (1 + exp(-raw))  # logit_rho -> rho
      row_name <- nm
    } else {
      vals <- raw
      row_name <- nm
    }
    qs <- quantile(vals, probs = probs)
    data.frame(mean = mean(vals), sd = sd(vals),
               q025 = qs[1], q975 = qs[2],
               row.names = row_name, stringsAsFactors = FALSE)
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
  draws <- object$draws
  cn <- colnames(draws)

  patterns <- c(
    tau   = "^log_tau_temporal$",
    rho   = "^logit_rho_ar1$",
    sigma = "^log_sigma2_temporal_gp$",
    lengthscale = "^logit_phi_temporal_gp$"
  )

  found <- list()
  for (nm in names(patterns)) {
    idx <- grep(patterns[nm], cn)
    if (length(idx) > 0) found[[nm]] <- idx[1]
  }

  if (length(found) == 0) {
    stop("No temporal hyperparameters found. Is this a temporal model?", call. = FALSE)
  }

  rows <- lapply(names(found), function(nm) {
    raw <- draws[, found[[nm]]]
    label <- cn[found[[nm]]]
    if (nm == "tau") {
      vals <- exp(raw)  # log_tau -> tau (precision)
      row_name <- "precision"
    } else if (nm == "rho") {
      vals <- 1 / (1 + exp(-raw))  # logit -> rho in (0,1)
      row_name <- "rho_ar1"
    } else if (nm == "sigma") {
      vals <- sqrt(exp(raw))  # log_sigma2 -> sigma
      row_name <- "sigma_temporal"
    } else if (nm == "lengthscale") {
      vals <- 1 / (1 + exp(-raw))  # logit -> (0,1) then scale
      row_name <- "lengthscale"
    } else {
      vals <- raw
      row_name <- nm
    }
    qs <- quantile(vals, probs = probs)
    data.frame(mean = mean(vals), sd = sd(vals),
               q025 = qs[1], q975 = qs[2],
               row.names = row_name, stringsAsFactors = FALSE)
  })
  do.call(rbind, rows)
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
