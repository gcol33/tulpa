#' Latent factor specification for unmeasured confounders
#'
#' @description
#' Specify latent factors to capture shared unmeasured confounders between
#' model processes. Latent factors are particularly useful
#' when you suspect that both processes are driven by common unmeasured variables.
#'
#' @return The constructor documented in this family ([latent_factor()])
#'   returns a `tulpa_latent` specification object consumed by ratio /
#'   multi-arm model packages built on tulpa (e.g. tulpaRatio), where the
#'   shared factor enters two or more linear predictors and cancels in the
#'   derived ratio. It is a multi-arm construct: the single-response [tulpa()]
#'   front door does not read it (for an in-formula latent Gaussian block on a
#'   single response, use `latent(tgmrf(...))`).
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
#' @return A `tulpa_latent` object consumed by ratio / multi-arm model
#'   packages built on tulpa (e.g. tulpaRatio); not read by the single-response
#'   [tulpa()] front door (use `latent(tgmrf(...))` for a single-response
#'   latent Gaussian block).
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
#' For observation i with K latent factors, on each of two model arms
#' (processes 1 and 2):
#'
#' \deqn{\eta^{(1)}_i = X^{(1)}_i \beta^{(1)} + \sum_{k=1}^{K} f_{ik} \sigma_k + \ldots}
#' \deqn{\eta^{(2)}_i = X^{(2)}_i \beta^{(2)} + \sum_{k=1}^{K} f_{ik} \sigma_k + \ldots}
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
#' # The spec is consumed by a ratio / multi-arm model package (tulpaRatio owns
#' # the two-arm `species | total ~ ...` formula and the negbin/negbin ratio
#' # family); it is that package's fitter, not the single-response tulpa() front
#' # door, that reads the `latent_factor()` object.
#'
#' @seealso [prior_pc()] for prior specification; `latent()` / [tgmrf()] for a
#'   single-response in-formula latent Gaussian block
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


#' @rdname latent_factor
#' @param x A `tulpa_latent` specification.
#' @param ... Ignored.
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


#' Extract latent factor posteriors from fit
#'
#' @param fit A fit sampled with latent factors: its draw matrix (`$draws` or
#'   `$samples`) holds columns named `factor[observation,factor]` and
#'   `$.internal$latent_info` records the layout. The default method reads any
#'   fit, of any class, that stores them this way.
#' @param summary Logical; if TRUE, return summary statistics. If FALSE,
#'   return full posterior draws.
#' @param probs Quantiles for summary. Default is c(0.025, 0.5, 0.975).
#' @param ... Passed to methods.
#'
#' @return If `summary = TRUE`, a data frame with columns for observation,
#'   factor, and summary statistics. If `summary = FALSE`, a matrix of
#'   posterior draws.
#'
#' @examples
#' # A fit sampled with latent factors comes from a model package (tulpaRatio
#' # reads the latent_factor() spec). The accessor reads two things off it:
#' # draw columns named factor[observation,factor] and the latent layout. Here
#' # both are simulated for three observations on two factors.
#' set.seed(1)
#' cols <- sprintf("factor[%d,%d]", rep(1:3, 2), rep(1:2, each = 3))
#' draws <- matrix(rnorm(200 * 6, mean = rep(c(-1, 0, 1), 2), sd = 0.3),
#'                 200, 6, byrow = TRUE, dimnames = list(NULL, cols))
#' fit <- structure(
#'   list(draws = draws,
#'        .internal = list(latent_info = list(type = "latent", n_factors = 2L,
#'                                            n_obs = 3L))),
#'   class = "tulpa_fit")
#'
#' latent_factors(fit)
#' dim(latent_factors(fit, summary = FALSE))
#'
#' @export
latent_factors <- function(fit, ...) {
  UseMethod("latent_factors")
}

#' @rdname latent_factors
#' @export
latent_factors.default <- function(fit, summary = TRUE,
                                   probs = c(0.025, 0.5, 0.975), ...) {

  # Check if model has latent factors
  if (!is.list(fit) || is.null(fit$.internal$latent_info) ||
      fit$.internal$latent_info$type == "none") {
    stop("Model was not fit with latent factors", call. = FALSE)
  }

  latent_info <- fit$.internal$latent_info
  n_factors <- latent_info$n_factors
  n_obs <- latent_info$n_obs

  # Extract factor draws from samples
  draws <- .fit_draws(fit)

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
