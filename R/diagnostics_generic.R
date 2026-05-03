# ============================================================================
# Generic spatial/temporal diagnostics and model comparison
# These operate on residuals or fit objects — not model-specific.
# ============================================================================

#' Compare models by information criteria
#' @param ... Named fit objects (must have `$log_prob` or support `logLik()`).
#' @param criterion `"waic"` or `"loglik"`.
#' @return Data frame with model comparison statistics.
#' @export
compare_models <- function(..., criterion = c("waic", "loglik")) {
  criterion <- match.arg(criterion)
  models <- list(...)
  if (is.null(names(models))) names(models) <- paste0("model", seq_along(models))

  rows <- lapply(names(models), function(nm) {
    fit <- models[[nm]]
    ll <- tryCatch(as.numeric(logLik(fit)), error = function(e) NA_real_)
    n_par <- fit$n_params %||% ncol(fit$draws)
    row <- data.frame(model = nm, n_params = n_par, logLik = ll,
                      stringsAsFactors = FALSE)
    if (criterion == "waic" && !is.null(fit$waic_fn)) {
      w <- tryCatch(fit$waic_fn(fit), error = function(e) NULL)
      row$WAIC <- if (!is.null(w)) w$waic else NA_real_
      row$p_waic <- if (!is.null(w)) w$p_waic else NA_real_
    }
    row
  })
  result <- do.call(rbind, rows)
  if ("WAIC" %in% names(result) && !all(is.na(result$WAIC))) {
    result$delta <- result$WAIC - min(result$WAIC, na.rm = TRUE)
    rel <- exp(-0.5 * result$delta)
    result$weight <- rel / sum(rel, na.rm = TRUE)
  }
  result
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
spatialRange <- function(object, probs = c(0.025, 0.975)) {
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
      # GP decay parameter -> effective range = 3/phi (Matern)
      vals <- 3 / exp(raw)
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
temporalCorr <- function(object, probs = c(0.025, 0.975)) {
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


#' Extract SVC posterior samples (convenience wrapper)
#'
#' Alias for `svc(object, summary = TRUE)`. Returns site-level
#' spatially varying coefficient summaries.
#'
#' @param object A `tulpa_fit` object fitted with SVCs.
#' @param terms Character vector of SVC term names to extract (default: all).
#' @param probs Quantile probabilities (default 0.025, 0.5, 0.975).
#' @return A `tulpa_svc_posterior` object.
#' @export
getSVCSamples <- function(object, terms = NULL, probs = c(0.025, 0.5, 0.975)) {
  svc(object, terms = terms, summary = TRUE, probs = probs)
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
#' @return A list of class `"postHocLM"` with:
#'   \describe{
#'     \item{summary}{data.frame of coefficient estimates and CIs}
#'     \item{lm_fit}{the underlying `lm` object}
#'     \item{boot_coefs}{matrix of bootstrap coefficient samples (if `n_boot > 0`)}
#'     \item{R2}{R-squared from the fitted model}
#'   }
#' @export
postHocLM <- function(formula, data, weights = NULL,
                      n_boot = 1000L, probs = c(0.025, 0.975)) {
  lm_fit <- if (is.null(weights)) {
    lm(formula, data = data)
  } else {
    lm(formula, data = data, weights = weights)
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
        if (is.null(boot_wts)) lm(formula, data = boot_data)
        else lm(formula, data = boot_data, weights = boot_wts),
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
  class(out) <- "postHocLM"
  out
}


#' Model-averaged predictions
#' @param ... Named fit objects.
#' @param criterion `"waic"` (default).
#' @param fitted_fn Function to extract fitted values from a fit (default: `fitted`).
#' @return List with averaged predictions and weights.
#' @export
modelAverage <- function(..., criterion = "waic", fitted_fn = fitted) {
  models <- list(...)
  if (is.null(names(models))) names(models) <- paste0("model", seq_along(models))
  comp <- compare_models(..., criterion = criterion)
  weights <- if ("weight" %in% names(comp)) comp$weight else rep(1/length(models), length(models))
  names(weights) <- comp$model

  fits <- lapply(models, fitted_fn)
  # Average first element of each fitted result
  first_vals <- lapply(fits, function(f) if (is.list(f)) f[[1]] else f)
  n <- length(first_vals[[1]])
  avg <- numeric(n)
  for (k in seq_along(models)) avg <- avg + weights[k] * first_vals[[k]]
  list(averaged = avg, weights = weights, comparison = comp)
}
