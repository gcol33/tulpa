#' Latent Factor Specification for Unmeasured Confounders
#'
#' @description
#' Specify latent factors to capture shared unmeasured confounders between
#' model processes. Latent factors are particularly useful
#' when you suspect that both processes are driven by common unmeasured variables.
#'
#' @return The constructor documented in this family ([latent_factor()])
#'   returns a `tulpa_latent` specification object for use in [tulpa()].
#'
#' @name tulpa_latent
NULL


#' Create a latent factor specification
#'
#' @description
#' Define latent factors for capturing unmeasured shared structure between
#' all processes. Factors are observation-level random effects
#' that enter both linear predictors when `shared = TRUE` (default).
#'
#' @param n_factors Integer; number of latent factors. Default is 1.
#'   More factors capture more complex unmeasured structure but increase
#'   computational cost and risk overfitting.
#' @param prior Prior for factor standard deviations. Default is a PC prior
#'   with P(sigma > 1) = 0.01, which shrinks toward simpler models.
#' @param shared Logical; if TRUE (default), latent factors enter both
#'   all process linear predictors identically. If FALSE,
#'   factors only affect the first process.
#' @param constraint Identifiability constraint for factors:
#'   - `"sum_to_zero"` (default): Factors sum to zero across observations
#'   - `"first_zero"`: First observation's factor is fixed to zero
#' @param scale Logical; if TRUE (default), factor loadings are standardized
#'   to have unit variance before applying sigma.
#'
#' @return A `tulpa_latent` object for use in [tulpa()].
#'
#' @details
#'
#' ## Why Use Latent Factors?
#'
#' When modeling multiple processes, they often share unmeasured
#' confounders. For example, in relative abundance data, both the focal species
#' count and total count might be affected by:
#' - Observer skill (unmeasured)
#' - Local microhabitat conditions (unmeasured)
#' - Weather on sampling day (unmeasured)
#'
#' Without accounting for these shared drivers, estimates can be biased.
#' Latent factors capture this shared structure without requiring the
#' confounders to be measured.
#'
#' ## Mathematical Model
#'
#' For observation i with K latent factors:
#'
#' \deqn{\eta^{num}_i = X^{num}_i \beta^{num} + \sum_{k=1}^{K} f_{ik} \sigma_k + \ldots}
#' \deqn{\eta^{denom}_i = X^{denom}_i \beta^{denom} + \sum_{k=1}^{K} f_{ik} \sigma_k + \ldots}
#'
#' where:
#' - \eqn{f_{ik} \sim N(0, 1)} are standardized factor scores
#' - \eqn{\sigma_k} are factor standard deviations with PC prior
#' - Identifiability: \eqn{\sum_i f_{ik} = 0} for each k
#'
#' Because factors enter both linear predictors identically (when shared),
#' they cancel in derived quantities (e.g., ratios, differences):
#'
#' \deqn{\eta^{(1)}_i - \eta^{(2)}_i}
#'
#' This means factors capture shared effects that would
#' otherwise bias derived quantities.
#'
#' ## Choosing n_factors
#'
#' - Start with `n_factors = 1` for simple unmeasured confounding
#' - Use `n_factors = 2-3` if you suspect multiple independent confounders
#' - More than 3 factors is rarely needed and risks overfitting
#' - The PC prior provides regularization, shrinking unneeded factors toward zero
#'
#' ## Relationship to Random Effects
#'
#' Latent factors differ from random effects in several ways:
#' - Random effects are grouped (e.g., site-level), factors are observation-level
#' - Random effects require grouping structure, factors don't
#' - Factors capture residual correlation not explained by observed predictors
#'
#' You can use both together: random effects for known grouping, factors for
#' residual unmeasured confounding.
#'
#' @examples
#' # Basic latent factor (single shared factor)
#' latent_factor()
#'
#' # Two latent factors
#' latent_factor(n_factors = 2)
#'
#' # Custom prior (more regularization)
#' latent_factor(n_factors = 1, prior = prior_pc(U = 0.5, alpha = 0.01))
#'
#' # Numerator-only factor (not shared)
#' latent_factor(n_factors = 1, shared = FALSE)
#'
#' \dontrun{
#' # Use in model fitting through a ratio model package (tulpaRatio owns the
#' # two-arm formula and the negbin/negbin ratio family):
#' fit <- tulpa(
#'   species_count | total_count ~ habitat + (1 | site),
#'   data = df,
#'   family = tulpaRatio::tulpa_negbin_negbin(),
#'   latent = latent_factor(n_factors = 2)
#' )
#' }
#'
#' @seealso [tulpa()] for model fitting, [prior_pc()] for prior specification
#'
#' @export
latent_factor <- function(
    n_factors = 1L,
    prior = NULL,
    shared = TRUE,
    constraint = c("sum_to_zero", "first_zero"),
    scale = TRUE
) {

  # Validate n_factors
  if (length(n_factors) != 1 || !is.numeric(n_factors) || n_factors < 1 ||
      n_factors != as.integer(n_factors)) {
    stop("`n_factors` must be a positive integer", call. = FALSE)
  }
  n_factors <- as.integer(n_factors)

  # Default prior
  if (is.null(prior)) {
    prior <- prior_pc(U = 1.0, alpha = 0.01)
  }


  # Validate prior
  if (!inherits(prior, "tulpa_prior")) {
    stop("`prior` must be a prior object (use prior_*() functions)", call. = FALSE)
  }

  # Validate shared
  if (!is.logical(shared) || length(shared) != 1) {
    stop("`shared` must be TRUE or FALSE", call. = FALSE)
  }

  # Validate constraint
  constraint <- match.arg(constraint)

  # Validate scale
  if (!is.logical(scale) || length(scale) != 1) {
    stop("`scale` must be TRUE or FALSE", call. = FALSE)
  }

  structure(
    list(
      n_factors = n_factors,
      prior = prior,
      shared = shared,
      constraint = constraint,
      scale = scale
    ),
    class = c("tulpa_latent", "list")
  )
}


#' Print method for tulpa_latent
#'
#' @param x A tulpa_latent object
#' @param ... Ignored
#'
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing the latent factor specification to the console.
#'
#' @export
print.tulpa_latent <- function(x, ...) {
  cat("Latent factor specification\n")
  cat("===========================\n\n")

  cat("Number of factors:", x$n_factors, "\n")
  cat("Shared:", ifelse(x$shared, "Yes (enters all processes)", "No (first process only)"), "\n")
  cat("Constraint:", x$constraint, "\n")
  cat("Scale:", ifelse(x$scale, "Yes", "No"), "\n\n")

  cat("Factor SD prior:\n")
  print_prior(x$prior, indent = "  ")

  invisible(x)
}


#' Validate latent factor specification against data
#'
#' @param latent A tulpa_latent object
#' @param N Number of observations
#'
#' @return Validated latent specification with additional computed fields
#' @keywords internal
validate_latent <- function(latent, N) {

  if (is.null(latent)) {
    return(NULL)
  }

  if (!inherits(latent, "tulpa_latent")) {
    stop("`latent` must be a tulpa_latent object from latent_factor()",
         call. = FALSE)
  }

  # Check n_factors is reasonable for data size
  if (latent$n_factors >= N) {
    stop("Number of latent factors (", latent$n_factors,
         ") must be less than number of observations (", N, ")",
         call. = FALSE)
  }

  # Warn if many factors relative to data
  if (latent$n_factors > sqrt(N)) {
    warning("Number of latent factors (", latent$n_factors,
            ") is large relative to data size. Consider using fewer factors.",
            call. = FALSE)
  }

  # Add computed fields
  latent$n_obs <- N
  latent$n_params <- N * latent$n_factors + latent$n_factors  # factors + sigmas

  latent
}


#' Prepare latent factor data for HMC
#'
#' @param latent A validated tulpa_latent object (or NULL)
#' @param N Number of observations
#'
#' @return List with latent factor data for C++
#' @keywords internal
prepare_latent_for_hmc <- function(latent, N) {

  if (is.null(latent)) {
    return(list(
      type = "none",
      n_factors = 0L,
      n_obs = as.integer(N),
      shared = FALSE,
      constraint = "sum_to_zero",
      scale = TRUE,
      sigma_prior_rate = 0.0  # Not used when type = "none"
    ))
  }

  # Validate
  latent <- validate_latent(latent, N)

  # Extract prior rate for PC prior (exponential prior on sigma)
  sigma_prior_rate <- if (latent$prior$distribution == "pc") {
    latent$prior$rate
  } else if (latent$prior$distribution == "exponential") {
    latent$prior$rate
  } else if (latent$prior$distribution == "half_normal") {
    # Convert half-normal to approximate exponential rate
    # E[X] = sd * sqrt(2/pi), so rate ~= sqrt(pi/2) / sd
    sqrt(pi / 2) / latent$prior$sd
  } else if (latent$prior$distribution == "half_cauchy") {
    # Use scale as rough guide
    1.0 / latent$prior$scale
  } else {
    # Default
    -log(0.01) / 1.0  # PC prior default
  }

  list(
    type = "latent",
    n_factors = as.integer(latent$n_factors),
    n_obs = as.integer(N),
    shared = latent$shared,
    constraint = latent$constraint,
    scale = latent$scale,
    sigma_prior_rate = sigma_prior_rate
  )
}


#' Initialize latent factor parameters
#'
#' @param latent_info Latent factor info from prepare_latent_for_hmc
#' @param seed Optional random seed
#'
#' @return Numeric vector of initial parameter values
#' @keywords internal
initialize_latent_params <- function(latent_info, seed = NULL) {

  if (latent_info$type == "none" || latent_info$n_factors == 0) {
    return(numeric(0))
  }

  .seed_scoped(seed)

  n_factors <- latent_info$n_factors
  n_obs <- latent_info$n_obs

  # Initialize log_sigma_factor (log scale, small positive values)
  log_sigma_init <- rep(-1.0, n_factors)  # sigma ~ 0.37

  # Initialize factor scores (standardized, sum-to-zero enforced)
  factor_init <- matrix(rnorm(n_obs * n_factors, 0, 0.1), nrow = n_obs)

  # Apply constraint
  if (latent_info$constraint == "sum_to_zero") {
    for (k in seq_len(n_factors)) {
      factor_init[, k] <- factor_init[, k] - mean(factor_init[, k])
    }
  } else if (latent_info$constraint == "first_zero") {
    for (k in seq_len(n_factors)) {
      factor_init[, k] <- factor_init[, k] - factor_init[1, k]
    }
  }

  # Flatten: [log_sigma_1, ..., log_sigma_K, f_1, ..., f_N*K]
  # But exclude one observation per factor for identifiability
  # Simpler: just return all and let C++ handle constraint
  c(log_sigma_init, as.vector(t(factor_init)))
}


#' Extract latent factor posteriors from fit
#'
#' @param fit A tulpa_fit object
#' @param summary Logical; if TRUE, return summary statistics. If FALSE,
#'   return full posterior draws.
#' @param probs Quantiles for summary. Default is c(0.025, 0.5, 0.975).
#'
#' @return If `summary = TRUE`, a data frame with columns for observation,
#'   factor, and summary statistics. If `summary = FALSE`, a matrix of
#'   posterior draws.
#'
#' @examples
#' \dontrun{
#' # After fitting a ratio model with latent factors (tulpaRatio owns the
#' # two-arm formula and the negbin/negbin ratio family):
#' fit <- tulpa(
#'   count | total ~ x,
#'   data = df,
#'   family = tulpaRatio::tulpa_negbin_negbin(),
#'   latent = latent_factor(n_factors = 2)
#' )
#'
#' # Get summary
#' factors <- latent_factors(fit)
#' head(factors)
#'
#' # Get full posterior draws
#' factor_draws <- latent_factors(fit, summary = FALSE)
#' }
#'
#' @export
latent_factors <- function(fit, summary = TRUE, probs = c(0.025, 0.5, 0.975)) {

  if (!inherits(fit, "tulpa_fit")) {
    stop("`fit` must be a tulpa_fit object", call. = FALSE)
  }

  # Check if model has latent factors
  if (is.null(fit$.internal$latent_info) ||
      fit$.internal$latent_info$type == "none") {
    stop("Model was not fit with latent factors", call. = FALSE)
  }

  latent_info <- fit$.internal$latent_info
  n_factors <- latent_info$n_factors
  n_obs <- latent_info$n_obs

  # Extract factor draws from samples
  draws <- fit$draws

  # Find factor columns
  factor_cols <- grep("^factor\\[", colnames(draws))

  if (length(factor_cols) == 0) {
    stop("No latent factor parameters found in posterior draws", call. = FALSE)
  }

  factor_draws <- draws[, factor_cols, drop = FALSE]

  if (!summary) {
    return(factor_draws)
  }

  # Compute summary
  n_samples <- nrow(factor_draws)

  # Parse column names to get obs and factor indices
  col_info <- do.call(rbind, lapply(colnames(factor_draws), function(nm) {
    # Pattern: factor[obs,factor]
    matches <- regmatches(nm, regexec("factor\\[(\\d+),(\\d+)\\]", nm))[[1]]
    if (length(matches) == 3) {
      c(obs = as.integer(matches[2]), factor = as.integer(matches[3]))
    } else {
      c(obs = NA, factor = NA)
    }
  }))
  col_info <- as.data.frame(col_info)

  # Compute statistics
  result <- data.frame(
    observation = col_info$obs,
    factor = col_info$factor,
    mean = colMeans(factor_draws),
    sd = apply(factor_draws, 2, sd)
  )

  # Add quantiles
  quants <- apply(factor_draws, 2, quantile, probs = probs)
  for (i in seq_along(probs)) {
    result[[paste0("q", probs[i] * 100)]] <- quants[i, ]
  }

  rownames(result) <- NULL
  result
}
