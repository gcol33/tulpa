#' Latent factor specification
#'
#' @description
#' Latent factors are unmeasured drivers shared by the responses of a model:
#' each unit carries a score on every factor and each response carries its own
#' loading on it. The specification built by [latent_factor()] is read by
#' [tulpa()] through its `factors =` argument (a long-format joint model, one
#' response level per row) and by model packages built on tulpa, such as
#' tulpaRatio, whose two-process models load the factors on the numerator and
#' the denominator.
#'
#' @return [latent_factor()] returns a `tulpa_latent` specification.
#'
#' @name tulpa_latent
NULL


#' Create a latent factor specification
#'
#' @description
#' Define latent factors for unmeasured structure shared by the responses of a
#' model. A unit `l` carries scores \eqn{u_{l f} \sim N(0, 1)} on factors
#' \eqn{f = 1, \ldots, K}, and response `s` loads on them with loadings
#' \eqn{\lambda_{s f}}, so its linear predictor gains
#' \eqn{\sum_f \lambda_{s f} u_{l f}}. The factors add the covariance
#' \eqn{\Lambda \Lambda^\top} between the linear predictors of the responses
#' of one unit.
#'
#' @param n_factors Integer; number of latent factors. Default is 1.
#' @param unit For [tulpa()]'s `factors =` argument: the name of the column of
#'   `data` holding the unit that carries the scores (e.g. `"site"`).
#' @param response For [tulpa()]'s `factors =` argument: the name of the
#'   column of `data` holding the response level each row belongs to (e.g.
#'   `"species"`). Each level has its own fixed effects and loadings.
#' @param prior Prior of the factor loadings, or `NULL` for the fitter's
#'   default. [tulpa()] carries a normal prior centred at zero on every
#'   loading, folded onto the positive half-line on the diagonal, and takes
#'   `prior_normal(0, sd)` (default `sd = 1`); a model package may read other
#'   priors (tulpaRatio defaults to a PC prior with P(loading > 1) = 0.01 on
#'   the diagonal loadings).
#' @param shared Logical; if TRUE (default), the factors load on every
#'   process of the model. If FALSE, they load on the first process only (a
#'   two-process model package's numerator); [tulpa()] requires TRUE.
#' @param constraint `NULL` (default) or the identifiability constraint a
#'   sampling fitter pins the scores with: `"sum_to_zero"` (each factor's
#'   scores sum to zero across units) or `"first_zero"` (the first unit's
#'   scores are zero). [tulpa()] requires `NULL`: its scores carry a proper
#'   N(0, 1) prior and each response's intercept absorbs their level.
#'
#' @return A `tulpa_latent` object.
#'
#' @details
#'
#' ## Identification
#'
#' The product \eqn{\lambda_{s f} u_{l f}} is unchanged by a rescaling, a sign
#' flip and, for several factors, a rotation of the scores against the
#' loadings. The unit variance of the scores fixes the scale; a loading matrix
#' that is lower-triangular with a positive diagonal fixes the sign and the
#' rotation. \eqn{\Lambda \Lambda^\top} is invariant to all three, so it is
#' the quantity to compare across fits.
#'
#' ## Choosing n_factors
#'
#' With `S` responses, \eqn{\Lambda} lower-triangular has
#' \eqn{S K - K (K - 1) / 2} free loadings against the \eqn{S (S + 1) / 2}
#' entries of a covariance between the responses, so at most `S` factors fit.
#' A two-process model is full rank at `n_factors = 2`.
#'
#' ## Relationship to Random Effects
#'
#' A random intercept is one response's deviation per group; a factor is a
#' deviation per unit that every response reads with its own loading, so it
#' carries the residual correlation between responses that the covariates do
#' not explain.
#'
#' @examples
#' # Two factors over sites, one response level per species
#' latent_factor(n_factors = 2, unit = "site", response = "species")
#'
#' # A tighter loading prior
#' latent_factor(unit = "site", response = "species",
#'               prior = prior_normal(0, 0.5))
#'
#' \donttest{
#' set.seed(1)
#' n_site <- 60; n_sp <- 4
#' u <- rnorm(n_site)
#' lambda <- c(0.8, 0.5, -0.6, 0.3)
#' d <- expand.grid(site = factor(seq_len(n_site)),
#'                  species = factor(paste0("sp", seq_len(n_sp))))
#' d$x <- rnorm(n_site)[d$site]
#' d$y <- rpois(nrow(d), exp(0.5 + 0.3 * d$x +
#'                           lambda[d$species] * u[d$site]))
#' fit <- tulpa(y ~ x, data = d, family = "poisson",
#'              factors = latent_factor(unit = "site", response = "species"))
#' fit$loadings
#' }
#'
#' @seealso [tulpa()] (`factors =`), [latent_factors()] for the scores,
#'   [prior_normal()]
#'
#' @export
latent_factor <- function(
    n_factors = 1L,
    unit = NULL,
    response = NULL,
    prior = NULL,
    shared = TRUE,
    constraint = NULL
) {

  if (length(n_factors) != 1 || !is.numeric(n_factors) || is.na(n_factors) ||
      n_factors < 1 || n_factors != as.integer(n_factors)) {
    stop("`n_factors` must be a positive integer", call. = FALSE)
  }
  n_factors <- as.integer(n_factors)

  for (nm in c("unit", "response")) {
    v <- get(nm)
    if (!is.null(v) && (!is.character(v) || length(v) != 1L || is.na(v) ||
                        !nzchar(v))) {
      stop("`", nm, "` must be NULL or the name of one column of `data`.",
           call. = FALSE)
    }
  }

  if (!is.null(prior) && !inherits(prior, "tulpa_prior")) {
    stop("`prior` must be NULL or a prior object (use prior_*() functions)",
         call. = FALSE)
  }

  if (!is.logical(shared) || length(shared) != 1 || is.na(shared)) {
    stop("`shared` must be TRUE or FALSE", call. = FALSE)
  }

  if (!is.null(constraint)) {
    constraint <- match.arg(constraint, c("sum_to_zero", "first_zero"))
  }

  structure(
    list(
      n_factors = n_factors,
      unit = unit,
      response = response,
      prior = prior,
      shared = shared,
      constraint = constraint
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
  if (!is.null(x$unit)) cat("Unit column:", x$unit, "\n")
  if (!is.null(x$response)) cat("Response column:", x$response, "\n")
  cat("Shared:", ifelse(x$shared, "Yes (loads on all processes)",
                        "No (first process only)"), "\n")
  cat("Constraint:", x$constraint %||% "none (proper score prior)", "\n\n")

  cat("Loading prior:\n")
  if (is.null(x$prior)) {
    cat("  the fitter's default\n")
  } else {
    print_prior(x$prior, indent = "  ")
  }

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
