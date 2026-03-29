#' Posterior predictive checks for tulpa models
#'
#' @description
#' Visual and numerical checks comparing observed data to posterior
#' predictive distributions. Essential for assessing model fit.
#'
#' @name tulpa_validate
NULL

#' Posterior predictive check
#'
#' @description
#' Generate posterior predictive checks for a fitted tulpa model.
#' Compares observed data to replicated data from the posterior predictive
#' distribution.
#'
#' @param object A `tulpa_fit` object
#' @param type Type of check: "dens_overlay", "scatter", "intervals", "stat"
#' @param component Which component: "numerator", "denominator", or "both"
#' @param stat Function for "stat" type (default: mean)
#' @param ndraws Number of posterior draws to use (default: 50 for plots)
#' @param ... Additional arguments passed to bayesplot functions
#'
#' @return A ggplot object
#'
#' @examples
#' # pp_check requires a fitted model with posterior predictive draws
#' # See tulpa() examples for fitting models
#'
#' \donttest{
#' # Simulate data and fit model (slow, not run on CRAN)
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   count = rnbinom(n, size = 5, mu = 15),
#'   total = rnbinom(n, size = 8, mu = 100),
#'   x = rnorm(n),
#'   site = factor(rep(1:10, each = 5))
#' )
#' fit <- tulpa(
#'   count | total ~ x + (1 | site),
#'   data = df,
#'   family = tulpa_negbin_negbin(),
#'   iter = 200, warmup = 100, chains = 1
#' )
#' # Density overlay (requires bayesplot package)
#' # pp_check(fit, type = "dens_overlay")
#' }
#'
#' @export
pp_check <- function(object, ...) {
  UseMethod("pp_check")
}

#' @export
#' @rdname pp_check
pp_check.tulpa_fit <- function(object, type = c("dens_overlay", "scatter", "intervals", "stat"),
                                component = NULL,
                                stat = mean, ndraws = 50, ...) {

  type <- match.arg(type)

  if (!requireNamespace("bayesplot", quietly = TRUE)) {
    stop("Package 'bayesplot' is required for pp_check. Install with:\n",
         "  install.packages('bayesplot')", call. = FALSE)
  }

  # Check if posterior predictive draws are available
  # Model packages provide y_rep in draws
  if (is.null(object$draws$y_rep)) {
    stop("Posterior predictive draws not available.\n",
         "  pp_check requires models fitted with `predict = TRUE`.\n",
         "  Re-fit with: predict = TRUE", call. = FALSE)
  }

  # Extract posterior predictive draws
  yrep <- object$draws$y_rep

  # Observed data
  y <- object$.internal$hmc_data$y

  pp_check_single(y, yrep, type, stat, ndraws, "", ...)
}

#' Single component pp_check
#'
#' @keywords internal
pp_check_single <- function(y, yrep, type, stat, ndraws, title_suffix, ...) {

  # Subsample draws for visualization
  if (nrow(yrep) > ndraws) {
    idx <- sample(nrow(yrep), ndraws)
    yrep <- yrep[idx, , drop = FALSE]
  }

  # Create plot based on type
  if (type == "dens_overlay") {
    p <- bayesplot::ppc_dens_overlay(y, yrep, ...)
  } else if (type == "scatter") {
    # Use mean of yrep for scatter
    yrep_mean <- colMeans(yrep)
    p <- bayesplot::ppc_scatter_avg(y, yrep, ...)
  } else if (type == "intervals") {
    p <- bayesplot::ppc_intervals(y, yrep, ...)
  } else if (type == "stat") {
    p <- bayesplot::ppc_stat(y, yrep, stat = stat, ...)
  }

  p + ggplot2::ggtitle(paste0("Posterior predictive check", title_suffix))
}

#' Prior predictive simulation
#'
#' @description
#' Simulate data from the prior predictive distribution to check
#' whether priors are reasonable.
#'
#' @param formula A tulpa_formula object or formula components
#' @param family A tulpa_family object
#' @param data Data frame (structure used for dimensions)
#' @param priors Prior specification
#' @param n Number of prior predictive datasets to simulate
#'
#' @return A list with simulated datasets
#'
#' @examples
#' # prior_predict is not yet implemented
#' # See sim_tulpa() for data simulation
#'
#' @export
prior_predict <- function(formula, family, data, priors = NULL, n = 100) {
  stop("prior_predict not yet implemented", call. = FALSE)
}

#' LOO cross-validation
#'
#' @description
#' Compute leave-one-out cross-validation using Pareto-smoothed
#' importance sampling (PSIS-LOO).
#'
#' @param x A `tulpa_fit` object
#' @param ... Additional arguments passed to loo::loo
#'
#' @return A loo object
#'
#' @export
loo.tulpa_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  # Try combined log_lik first (all models should have this)
  # Handle both list and matrix draws formats
  if (is.list(x$draws) && !is.matrix(x$draws)) {
    ll <- x$draws$log_lik
  } else {
    ll <- NULL
  }

  if (!is.null(ll)) {
    return(loo::loo(ll, ...))
  }

  # Fall back to separate components for two-process models
  if (is.list(x$draws) && !is.matrix(x$draws)) {
    ll_num <- x$draws$log_lik_num
    ll_denom <- x$draws$log_lik_denom
  } else {
    ll_num <- NULL
    ll_denom <- NULL
  }

  if (is.null(ll_num) && is.null(ll_denom)) {
    stop("Log-likelihood not found in model output.", call. = FALSE)
  }

  # Sum components that exist
  if (!is.null(ll_num) && !is.null(ll_denom)) {
    ll <- ll_num + ll_denom
  } else if (!is.null(ll_num)) {
    ll <- ll_num
  } else {
    ll <- ll_denom
  }

  loo::loo(ll, ...)
}

#' WAIC computation
#'
#' @description
#' Compute Widely Applicable Information Criterion (WAIC).
#'
#' @param x A `tulpa_fit` object
#' @param ... Additional arguments passed to loo::waic
#'
#' @return A waic object
#'
#' @export
waic.tulpa_fit <- function(x, ...) {
  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  # Try combined log_lik first
  # Handle both list and matrix draws formats
  if (is.list(x$draws) && !is.matrix(x$draws)) {
    ll <- x$draws$log_lik
  } else {
    ll <- NULL
  }

  if (!is.null(ll)) {
    return(loo::waic(ll, ...))
  }

  # Fall back to separate components
  if (is.list(x$draws) && !is.matrix(x$draws)) {
    ll_num <- x$draws$log_lik_num
    ll_denom <- x$draws$log_lik_denom
  } else {
    ll_num <- NULL
    ll_denom <- NULL
  }

  if (is.null(ll_num) && is.null(ll_denom)) {
    stop("Log-likelihood not found in model output.", call. = FALSE)
  }

  if (!is.null(ll_num) && !is.null(ll_denom)) {
    ll <- ll_num + ll_denom
  } else if (!is.null(ll_num)) {
    ll <- ll_num
  } else {
    ll <- ll_denom
  }

  loo::waic(ll, ...)
}

#' Compare tulpa models
#'
#' @description
#' Compare multiple tulpa models using LOO-CV or WAIC.
#'
#' @param ... Multiple `tulpa_fit` objects to compare
#' @param criterion Comparison criterion: "loo" (default) or "waic"
#'
#' @return A loo_compare object
#'
#' @examples
#' \donttest{
#' # Fit models with different structures (slow, not run on CRAN)
#' set.seed(123)
#' n <- 50
#' df <- data.frame(
#'   y_num = rnbinom(n, size = 5, mu = 15),
#'   y_denom = rnbinom(n, size = 8, mu = 100),
#'   x = rnorm(n),
#'   z = rnorm(n)
#' )
#' fit1 <- tulpa(y_num | y_denom ~ x, data = df,
#'               family = tulpa_negbin_negbin(),
#'               iter = 200, warmup = 100, chains = 1)
#' fit2 <- tulpa(y_num | y_denom ~ x + z, data = df,
#'               family = tulpa_negbin_negbin(),
#'               iter = 200, warmup = 100, chains = 1)
#' # Compare (requires loo package)
#' # tulpa_compare(fit1, fit2)
#' }
#'
#' @export
tulpa_compare <- function(..., criterion = c("loo", "waic")) {
  criterion <- match.arg(criterion)

  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  models <- list(...)

  if (length(models) < 2) {
    stop("At least two models required for comparison", call. = FALSE)
  }

  # Check all are tulpa_fit objects
  if (!all(vapply(models, inherits, logical(1), "tulpa_fit"))) {
    stop("All objects must be tulpa_fit models", call. = FALSE)
  }

  # Compute criterion for each model
  if (criterion == "loo") {
    results <- lapply(models, loo.tulpa_fit)
  } else {
    results <- lapply(models, waic.tulpa_fit)
  }

  # Get model names from call
  model_names <- as.character(substitute(list(...)))[-1]
  names(results) <- model_names

  # Compare
  loo::loo_compare(results)
}


#' Model averaging for tulpa fits
#'
#' @description
#' Compute model-averaged predictions using stacking or pseudo-BMA weights.
#' Model weights are derived from LOO-CV or WAIC, and predictions are
#' combined accordingly.
#'
#' @param ... Multiple `tulpa_fit` objects to average
#' @param weights Method for computing weights: "loo" (stacking, default),
#'   "waic", "pbma" (pseudo-BMA), or "pbma+" (pseudo-BMA+ with Bayesian bootstrap).
#' @param newdata Optional new data for predictions. If NULL, uses fitted values.
#' @param type Type of prediction: "ratio" (default), "numerator", or "denominator".
#' @param summary Logical; if TRUE (default), return summary statistics. If FALSE,
#'   return full posterior draws.
#'
#' @return A `tulpa_average` object containing:
#' - `weights`: Model weights
#' - `predictions`: Model-averaged predictions
#' - `models`: List of model names
#'
#' @details
#' Model averaging accounts for model uncertainty by combining predictions
#' from multiple candidate models. The weighting methods are:
#'
#' - **stacking** (`weights = "loo"`): Optimal linear combination minimizing
#'   leave-one-out prediction error. Recommended default.
#' - **pseudo-BMA** (`weights = "pbma"`): Weights proportional to exp(-ELPD).
#'   Can be unstable with similar models.
#' - **pseudo-BMA+** (`weights = "pbma+"`): Bayesian bootstrap variant that
#'   accounts for estimation uncertainty.
#'
#' For ratio predictions, averaging is performed on the log scale to respect
#' the multiplicative nature of ratios.
#'
#' @examples
#' \donttest{
#' # Fit candidate models (slow, not run on CRAN)
#' set.seed(123)
#' n <- 60
#' df <- data.frame(
#'   count = rpois(n, lambda = 8),
#'   effort = rgamma(n, shape = 4, rate = 1),
#'   x = rnorm(n),
#'   z = rnorm(n),
#'   site = factor(rep(1:10, each = 6))
#' )
#' fit1 <- tulpa(count | effort ~ x + (1 | site), data = df,
#'               family = tulpa_poisson_gamma(),
#'               iter = 200, warmup = 100, chains = 1)
#' fit2 <- tulpa(count | effort ~ x + z + (1 | site), data = df,
#'               family = tulpa_poisson_gamma(),
#'               iter = 200, warmup = 100, chains = 1)
#' # Model averaging (requires loo package)
#' # avg <- tulpa_average(fit1, fit2)
#' # print(avg)
#' }
#'
#' @seealso [tulpa_compare()] for model comparison, [loo::stacking_weights()]
#'
#' @export
tulpa_average <- function(..., weights = c("loo", "waic", "pbma", "pbma+"),
                          newdata = NULL, type = c("ratio", "numerator", "denominator"),
                          summary = TRUE) {

  weights_method <- match.arg(weights)
  type <- match.arg(type)

  if (!requireNamespace("loo", quietly = TRUE)) {
    stop("Package 'loo' is required. Install with:\n",
         "  install.packages('loo')", call. = FALSE)
  }

  models <- list(...)

  if (length(models) < 2) {
    stop("At least two models required for averaging", call. = FALSE)
  }

  # Check all are tulpa_fit objects
  if (!all(vapply(models, inherits, logical(1), "tulpa_fit"))) {
    stop("All objects must be tulpa_fit models", call. = FALSE)
  }

  # Get model names from call
  model_names <- as.character(substitute(list(...)))[-1]
  if (length(model_names) != length(models)) {
    model_names <- paste0("model", seq_along(models))
  }
  names(models) <- model_names

  # Check data compatibility
  n_obs <- vapply(models, function(m) m$.internal$hmc_data$N, integer(1))
  if (length(unique(n_obs)) > 1) {
    stop("All models must be fitted to the same data", call. = FALSE)
  }

  # Compute weights
  model_weights <- compute_model_weights(models, weights_method)

  # Get predictions from each model
  predictions <- lapply(models, function(m) {
    get_predictions(m, newdata = newdata, type = type, summary = FALSE)
  })

  # Model-average the predictions
  avg_pred <- average_predictions(predictions, model_weights, type, summary)

  structure(
    list(
      weights = model_weights,
      predictions = avg_pred,
      models = model_names,
      type = type,
      weights_method = weights_method,
      n_models = length(models)
    ),
    class = "tulpa_average"
  )
}


#' Compute model weights
#'
#' @param models List of tulpa_fit objects
#' @param method Weighting method
#' @return Named numeric vector of weights
#' @keywords internal
compute_model_weights <- function(models, method) {

  # Compute LOO for each model
  loo_list <- lapply(models, function(m) {
    tryCatch(
      loo.tulpa_fit(m),
      error = function(e) {
        stop("Failed to compute LOO for model: ", e$message, call. = FALSE)
      }
    )
  })

  if (method == "loo") {
    # Stacking weights (optimal for prediction)
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::stacking_weights(lpd_matrix)

  } else if (method == "waic") {
    # WAIC-based weights
    waic_list <- lapply(models, waic.tulpa_fit)
    lpd_matrix <- sapply(waic_list, function(x) x$pointwise[, "elpd_waic"])
    weights <- loo::stacking_weights(lpd_matrix)

  } else if (method == "pbma") {
    # Pseudo-BMA weights
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::pseudobma_weights(lpd_matrix, BB = FALSE)

  } else if (method == "pbma+") {
    # Pseudo-BMA+ with Bayesian bootstrap
    lpd_matrix <- sapply(loo_list, function(x) x$pointwise[, "elpd_loo"])
    weights <- loo::pseudobma_weights(lpd_matrix, BB = TRUE)
  }

  weights <- as.numeric(weights)
  names(weights) <- names(models)
  weights
}


#' Get predictions from a model
#'
#' @param model A tulpa_fit object
#' @param newdata Optional new data
#' @param type Prediction type
#' @param summary Return summary or draws
#' @return Matrix of predictions (draws x observations)
#' @keywords internal
get_predictions <- function(model, newdata, type, summary) {

  if (!is.null(newdata)) {
    # Use predict method for new data
    pred <- predict(model, newdata = newdata, type = type, summary = FALSE)
    return(pred)
  }

  # Extract fitted values from model
  if (type == "ratio") {
    # Get ratio predictions
    if (!is.null(model$draws$ratio)) {
      pred <- model$draws$ratio
    } else {
      # Compute from eta
      eta_num <- model$draws$eta_num
      eta_denom <- model$draws$eta_denom
      if (!is.null(eta_num) && !is.null(eta_denom)) {
        pred <- exp(eta_num - eta_denom)
      } else {
        stop("Cannot extract ratio predictions from model", call. = FALSE)
      }
    }
  } else if (type == "numerator") {
    pred <- model$draws$mu_num
    if (is.null(pred) && !is.null(model$draws$eta_num)) {
      pred <- exp(model$draws$eta_num)
    }
  } else {
    pred <- model$draws$mu_denom
    if (is.null(pred) && !is.null(model$draws$eta_denom)) {
      pred <- exp(model$draws$eta_denom)
    }
  }

  if (is.null(pred)) {
    stop("Cannot extract predictions for type '", type, "'", call. = FALSE)
  }

  pred
}


#' Average predictions across models
#'
#' @param predictions List of prediction matrices
#' @param weights Model weights
#' @param type Prediction type
#' @param summary Return summary or draws
#' @return Averaged predictions
#' @keywords internal
average_predictions <- function(predictions, weights, type, summary) {

  n_models <- length(predictions)
  n_draws <- nrow(predictions[[1]])
  n_obs <- ncol(predictions[[1]])

  # For ratios, average on log scale
  if (type == "ratio") {
    # Convert to log scale
    log_preds <- lapply(predictions, log)

    # Weighted average on log scale
    avg_log <- matrix(0, nrow = n_draws, ncol = n_obs)
    for (i in seq_len(n_models)) {
      avg_log <- avg_log + weights[i] * log_preds[[i]]
    }

    # Back-transform
    avg_pred <- exp(avg_log)

  } else {
    # Simple weighted average
    avg_pred <- matrix(0, nrow = n_draws, ncol = n_obs)
    for (i in seq_len(n_models)) {
      avg_pred <- avg_pred + weights[i] * predictions[[i]]
    }
  }

  if (summary) {
    # Return summary statistics
    data.frame(
      mean = colMeans(avg_pred),
      sd = apply(avg_pred, 2, sd),
      q2.5 = apply(avg_pred, 2, quantile, probs = 0.025),
      q25 = apply(avg_pred, 2, quantile, probs = 0.25),
      q50 = apply(avg_pred, 2, quantile, probs = 0.50),
      q75 = apply(avg_pred, 2, quantile, probs = 0.75),
      q97.5 = apply(avg_pred, 2, quantile, probs = 0.975)
    )
  } else {
    avg_pred
  }
}


#' Print method for tulpa_average
#'
#' @param x A tulpa_average object
#' @param digits Number of digits for printing
#' @param ... Ignored
#'
#' @export
print.tulpa_average <- function(x, digits = 3, ...) {
  cat("tulpa model averaging\n")
  cat("=====================\n\n")
  cat("Method:", x$weights_method, "\n")
  cat("Models:", x$n_models, "\n")
  cat("Prediction type:", x$type, "\n\n")

  cat("Model weights:\n")
  for (i in seq_along(x$weights)) {
    cat(sprintf("  %s: %.3f\n", x$models[i], x$weights[i]))
  }

  cat("\n")
  if (is.data.frame(x$predictions)) {
    cat("Predictions summary (first 6 rows):\n")
    print(head(x$predictions), digits = digits)
    if (nrow(x$predictions) > 6) {
      cat(sprintf("  ... %d more rows\n", nrow(x$predictions) - 6))
    }
  } else {
    cat("Predictions:", nrow(x$predictions), "draws x",
        ncol(x$predictions), "observations\n")
  }

  invisible(x)
}


#' Extract predictions from tulpa_average
#'
#' @param object A tulpa_average object
#' @param ... Ignored
#'
#' @return Predictions data frame or matrix
#'
#' @export
fitted.tulpa_average <- function(object, ...) {
  object$predictions
}


#' Extract model weights from tulpa_average
#'
#' @param object A tulpa_average object
#' @param ... Ignored
#'
#' @return Named numeric vector of weights
#'
#' @importFrom stats weights
#' @export
weights.tulpa_average <- function(object, ...) {
  object$weights
}
