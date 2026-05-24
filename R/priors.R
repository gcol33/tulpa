#' Prior specification for tulpa models
#'
#' @description
#' Specify priors for model parameters. Supports both PC (penalized complexity)
#' priors for variance components and standard distributions for other parameters.
#'
#' @param beta Prior for fixed effects. Default: `prior_normal(0, 2.5)`.
#' @param sigma Prior for random effect SDs. Default: PC prior with
#'   P(sigma > 1) = 0.01.
#' @param phi Prior for overdispersion parameter. Default: PC prior with
#'   P(phi > 10) = 0.01.
#' @param rho_temporal Prior for temporal autocorrelation. Default:
#'   `prior_beta(2, 2)` centered at 0.5.
#' @param rho_spatial Prior for spatial proportion (BYM2). Default:
#'   `prior_beta(1, 1)` (uniform).
#'
#' @return A `tulpa_priors` object
#'
#' @details
#' PC priors (Simpson et al., 2017) provide principled regularization that:
#' - Favors simpler models (smaller variance components)
#' - Has interpretable parameters (tail probabilities)
#' - Prevents overfitting with sparse data
#'
#' For other parameters, standard distributions are available via helper
#' functions: `prior_normal()`, `prior_half_normal()`, `prior_half_cauchy()`,
#' `prior_gamma()`, `prior_beta()`, `prior_exponential()`.
#'
#' @references
#' Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
#' Penalising model component complexity: A principled, practical approach to
#' constructing priors. Statistical Science, 32(1), 1-28.
#'
#' @examples
#' # Default priors
#' tulpa_priors()
#'
#' # Custom fixed effect prior
#' tulpa_priors(beta = prior_normal(0, 1))
#'
#' # Tighter random effect prior
#' tulpa_priors(sigma = prior_pc(U = 0.5, alpha = 0.01))
#'
#' # Half-Cauchy for random effect SD
#' tulpa_priors(sigma = prior_half_cauchy(2.5))
#'
#' # Informative prior for temporal correlation
#' tulpa_priors(rho_temporal = prior_beta(5, 2))  # Prior mode at ~0.8
#'
#' @export
tulpa_priors <- function(
    beta = NULL,
    sigma = NULL,
    phi = NULL,
    rho_temporal = NULL,
    rho_spatial = NULL
) {

  # Set defaults
  if (is.null(beta)) beta <- prior_normal(0, 2.5)
  if (is.null(sigma)) sigma <- prior_pc(U = 1.0, alpha = 0.01)
  if (is.null(phi)) phi <- prior_pc(U = 10.0, alpha = 0.01)
  if (is.null(rho_temporal)) rho_temporal <- prior_beta(2, 2)
  if (is.null(rho_spatial)) rho_spatial <- prior_beta(1, 1)

  # Validate each prior
  validate_prior(beta, "beta")
  validate_prior(sigma, "sigma")
  validate_prior(phi, "phi")
  validate_prior(rho_temporal, "rho_temporal")
  validate_prior(rho_spatial, "rho_spatial")

  structure(
    list(
      beta = beta,
      sigma = sigma,
      phi = phi,
      rho_temporal = rho_temporal,
      rho_spatial = rho_spatial
    ),
    class = "tulpa_priors"
  )
}


#' Validate a prior specification
#'
#' @param prior A prior object
#' @param name Parameter name for error messages
#' @keywords internal
validate_prior <- function(prior, name) {
  if (!inherits(prior, "tulpa_prior")) {
    stop(sprintf("'%s' must be a prior object (use prior_*() functions)", name),
         call. = FALSE)
  }
}

# Validate that a prior parameter is strictly positive. Centralised so every
# prior_*() constructor checks the same way and the error wording stays in sync.
.assert_positive <- function(value, name) {
  if (value <= 0) stop(sprintf("%s must be positive", name), call. = FALSE)
}


#' Normal prior
#'
#' @description
#' Specify a normal prior for a parameter.
#'
#' @param mean Prior mean. Default 0.
#' @param sd Prior standard deviation. Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @examples
#' prior_normal(0, 2.5)
#' prior_normal(0, 1)
#'
#' @export
prior_normal <- function(mean = 0, sd = 2.5) {
  .assert_positive(sd, "sd")
  structure(
    list(
      distribution = "normal",
      mean = mean,
      sd = sd
    ),
    class = c("tulpa_prior_normal", "tulpa_prior")
  )
}


#' Half-normal prior
#'
#' @description
#' Specify a half-normal (truncated at 0) prior for a positive parameter.
#'
#' @param sd Scale parameter. Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @examples
#' prior_half_normal(1)
#'
#' @export
prior_half_normal <- function(sd = 1) {
  .assert_positive(sd, "sd")
  structure(
    list(
      distribution = "half_normal",
      sd = sd
    ),
    class = c("tulpa_prior_half_normal", "tulpa_prior")
  )
}


#' Half-Cauchy prior
#'
#' @description
#' Specify a half-Cauchy prior for a positive parameter.
#' Has heavier tails than half-normal, often used for variance parameters.
#'
#' @param scale Scale parameter. Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @examples
#' prior_half_cauchy(2.5)
#'
#' @export
prior_half_cauchy <- function(scale = 2.5) {
  .assert_positive(scale, "scale")
  structure(
    list(
      distribution = "half_cauchy",
      scale = scale
    ),
    class = c("tulpa_prior_half_cauchy", "tulpa_prior")
  )
}


#' Gamma prior
#'
#' @description
#' Specify a gamma prior for a positive parameter.
#'
#' @param shape Shape parameter (alpha). Must be positive.
#' @param rate Rate parameter (beta). Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @details
#' Mean = shape/rate, Variance = shape/rate^2.
#'
#' @examples
#' prior_gamma(2, 0.1)  # Mean = 20, weakly informative
#' prior_gamma(1, 1)    # Exponential(1)
#'
#' @export
prior_gamma <- function(shape = 2, rate = 0.1) {
  .assert_positive(shape, "shape")
  .assert_positive(rate, "rate")
  structure(
    list(
      distribution = "gamma",
      shape = shape,
      rate = rate
    ),
    class = c("tulpa_prior_gamma", "tulpa_prior")
  )
}


#' Exponential prior
#'
#' @description
#' Specify an exponential prior for a positive parameter.
#'
#' @param rate Rate parameter (lambda). Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @examples
#' prior_exponential(1)
#'
#' @export
prior_exponential <- function(rate = 1) {
  .assert_positive(rate, "rate")
  structure(
    list(
      distribution = "exponential",
      rate = rate
    ),
    class = c("tulpa_prior_exponential", "tulpa_prior")
  )
}


#' Beta prior
#'
#' @description
#' Specify a beta prior for a parameter bounded in (0, 1).
#'
#' @param alpha First shape parameter. Must be positive.
#' @param beta Second shape parameter. Must be positive.
#'
#' @return A `tulpa_prior` object
#'
#' @details
#' - alpha = beta = 1: Uniform(0, 1)
#' - alpha = beta = 2: Symmetric, peaked at 0.5
#' - alpha > beta: Skewed toward 1
#' - alpha < beta: Skewed toward 0
#'
#' @examples
#' prior_beta(1, 1)   # Uniform
#' prior_beta(2, 2)   # Symmetric, centered at 0.5
#' prior_beta(5, 2)   # Skewed toward 1 (for high autocorrelation)
#'
#' @export
prior_beta <- function(alpha = 1, beta = 1) {
  .assert_positive(alpha, "alpha")
  .assert_positive(beta, "beta")
  structure(
    list(
      distribution = "beta",
      alpha = alpha,
      beta = beta
    ),
    class = c("tulpa_prior_beta", "tulpa_prior")
  )
}


#' Penalized complexity (PC) prior
#'
#' @description
#' Specify a PC prior for a positive parameter (typically a standard deviation).
#' PC priors shrink toward simpler models by penalizing deviation from a base model.
#'
#' @param U Upper bound. P(x > U) = alpha.
#' @param alpha Tail probability. Default 0.01.
#'
#' @return A `tulpa_prior` object
#'
#' @details
#' The PC prior is specified via: P(sigma > U) = alpha
#' - U is the upper bound you consider "large"
#' - alpha is the probability of exceeding it
#'
#' This implies an exponential prior with rate = -log(alpha) / U.
#'
#' @references
#' Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H. (2017).
#' Penalising model component complexity: A principled, practical approach to
#' constructing priors. Statistical Science, 32(1), 1-28.
#'
#' @examples
#' prior_pc(U = 1, alpha = 0.01)    # P(sigma > 1) = 0.01
#' prior_pc(U = 0.5, alpha = 0.05)  # Tighter, P(sigma > 0.5) = 0.05
#'
#' @export
prior_pc <- function(U = 1, alpha = 0.01) {
  .assert_positive(U, "U")
  if (alpha <= 0 || alpha >= 1) stop("alpha must be in (0, 1)", call. = FALSE)

  rate <- -log(alpha) / U

  structure(
    list(
      distribution = "pc",
      U = U,
      alpha = alpha,
      rate = rate
    ),
    class = c("tulpa_prior_pc", "tulpa_prior")
  )
}


#' Legacy prior specification (deprecated)
#'
#' @description
#' Old-style prior specification for backwards compatibility.
#' Use `tulpa_priors()` with `prior_*()` functions instead.
#'
#' @param sigma_U Upper bound for random effect SD.
#' @param sigma_alpha Tail probability.
#' @param phi_U Upper bound for overdispersion.
#' @param phi_alpha Tail probability for overdispersion.
#' @param beta_sd SD for normal prior on fixed effects.
#'
#' @return A `tulpa_priors` object
#'
#' @keywords internal
#' @export
tulpa_priors_legacy <- function(
    sigma_U = 1.0,
    sigma_alpha = 0.01,
    phi_U = 10.0,
    phi_alpha = 0.01,
    beta_sd = 2.5
) {

  # Validate
  .assert_positive(sigma_U, "sigma_U")
  if (sigma_alpha <= 0 || sigma_alpha >= 1) {
    stop("sigma_alpha must be in (0, 1)", call. = FALSE)
  }
  .assert_positive(phi_U, "phi_U")
  if (phi_alpha <= 0 || phi_alpha >= 1) {
    stop("phi_alpha must be in (0, 1)", call. = FALSE)
  }
  .assert_positive(beta_sd, "beta_sd")

  tulpa_priors(
    beta = prior_normal(0, beta_sd),
    sigma = prior_pc(U = sigma_U, alpha = sigma_alpha),
    phi = prior_pc(U = phi_U, alpha = phi_alpha)
  )
}

#' Print method for tulpa_priors
#'
#' @param x A tulpa_priors object
#' @param ... Ignored
#'
#' @export
print.tulpa_priors <- function(x, ...) {
  cat("tulpa prior specification\n")
  cat("=========================\n\n")

  cat("Fixed effects (beta):\n")
  print_prior(x$beta, indent = "  ")
  cat("\n")

  cat("Random effect SD (sigma):\n")
  print_prior(x$sigma, indent = "  ")
  cat("\n")

  cat("Overdispersion (phi):\n")
  print_prior(x$phi, indent = "  ")
  cat("\n")

  cat("Temporal autocorrelation (rho_temporal):\n")
  print_prior(x$rho_temporal, indent = "  ")
  cat("\n")

  cat("Spatial proportion (rho_spatial):\n")
  print_prior(x$rho_spatial, indent = "  ")

  invisible(x)
}


#' Print method for tulpa_prior
#'
#' @param x A tulpa_prior object
#' @param ... Ignored
#'
#' @export
print.tulpa_prior <- function(x, ...) {
  print_prior(x, indent = "")
  invisible(x)
}


#' Format a single prior for printing
#'
#' @description
#' Thin wrapper around `format()` for `tulpa_prior` objects. Each
#' `prior_*()` constructor attaches a subclass (e.g. `tulpa_prior_normal`)
#' so adding a new distribution is one new `format.tulpa_prior_<dist>()`
#' method — no central if/else to extend.
#'
#' @param prior A tulpa_prior object
#' @param indent Indentation string
#' @keywords internal
print_prior <- function(prior, indent = "") {
  cat(format(prior, indent = indent), "\n", sep = "")
}

#' @export
format.tulpa_prior_normal <- function(x, indent = "", ...) {
  sprintf("%sNormal(%.2f, %.2f)", indent, x$mean, x$sd)
}

#' @export
format.tulpa_prior_half_normal <- function(x, indent = "", ...) {
  sprintf("%sHalf-Normal(%.2f)", indent, x$sd)
}

#' @export
format.tulpa_prior_half_cauchy <- function(x, indent = "", ...) {
  sprintf("%sHalf-Cauchy(%.2f)", indent, x$scale)
}

#' @export
format.tulpa_prior_gamma <- function(x, indent = "", ...) {
  sprintf("%sGamma(%.2f, %.2f)  [mean = %.2f]",
          indent, x$shape, x$rate, x$shape / x$rate)
}

#' @export
format.tulpa_prior_exponential <- function(x, indent = "", ...) {
  sprintf("%sExponential(%.2f)  [mean = %.2f]",
          indent, x$rate, 1 / x$rate)
}

#' @export
format.tulpa_prior_beta <- function(x, indent = "", ...) {
  sprintf("%sBeta(%.2f, %.2f)  [mean = %.2f]",
          indent, x$alpha, x$beta, x$alpha / (x$alpha + x$beta))
}

#' @export
format.tulpa_prior_pc <- function(x, indent = "", ...) {
  paste0(
    sprintf("%sPC prior: P(x > %.2f) = %.3f", indent, x$U, x$alpha),
    "\n",
    sprintf("%s  => Exponential(%.3f)", indent, x$rate)
  )
}

#' @export
format.tulpa_prior <- function(x, indent = "", ...) {
  sprintf("%s%s", indent, x$distribution %||% "<unknown>")
}


#' Show default priors for a tulpa family
#'
#' @description
#' Display the default prior specifications used for each model family.
#' Useful for understanding what priors are applied before fitting
#' and as a starting point for customization.
#'
#' @param family A tulpa family object (e.g., `tulpa_negbin_negbin()`).
#'   If NULL (default), shows defaults for all families.
#' @param spatial Logical; if TRUE, include spatial priors. Default FALSE.
#' @param temporal Logical; if TRUE, include temporal priors. Default FALSE.
#'
#' @return Invisibly returns a `tulpa_priors` object with the defaults.
#'   Primarily called for its side effect of printing.
#'
#' @details
#' Default priors in tulpa follow these principles:
#'
#' - **Fixed effects (beta)**: Normal(0, 2.5) - weakly informative, allows
#'   coefficients roughly in \[-5, 5\] on the link scale.
#'
#' - **Random effect SD (sigma)**: PC prior with P(sigma > 1) = 0.01 - favors
#'   simpler models with smaller variance components.
#'
#' - **Overdispersion (phi)**: PC prior with P(phi > 10) = 0.01 - regularizes
#'   toward Poisson (phi -> Inf means less overdispersion in NB2).
#'
#' - **Temporal correlation (rho)**: Beta(2, 2) - symmetric prior centered
#'   at 0.5, appropriate for AR(1) correlation.
#'
#' - **Spatial mixing (rho_spatial)**: Beta(1, 1) = Uniform(0, 1) - no
#'   prior preference for structured vs. unstructured spatial variation.
#'
#' @examples
#' # Defaults for all families (no family argument)
#' priors_default()
#'
#' # Family-specific defaults take a family object from a model package
#' # (e.g. numdenom's tulpa_negbin_negbin()), so they are not run here.
#' \dontrun{
#' priors_default(tulpa_negbin_negbin())
#' priors_default(tulpa_binomial())
#'
#' # Including spatial parameters
#' priors_default(tulpa_negbin_negbin(), spatial = TRUE)
#'
#' # Use as a starting point for customization
#' my_priors <- priors_default(tulpa_poisson_gamma())
#' my_priors$beta <- prior_normal(0, 1)  # Tighter prior on fixed effects
#' }
#'
#' @seealso [tulpa_priors()] for creating custom priors
#'
#' @export
priors_default <- function(family = NULL, spatial = FALSE, temporal = FALSE) {

  # Get default priors
  defaults <- tulpa_priors()

  if (is.null(family)) {
    # Show defaults for all families
    cat("Default priors for tulpa models\n")
    cat("================================\n\n")

    cat("These defaults apply to all families unless overridden.\n\n")

    cat("Fixed effects (beta):\n")
    cat("  Normal(0, 2.5)\n")
    cat("  Interpretation: Coefficients roughly in [-5, 5] on link scale\n")
    cat("  Customization: prior_normal(mean, sd)\n\n")

    cat("Random effect SD (sigma):\n")
    cat("  PC prior: P(sigma > 1) = 0.01\n")
    cat("  Interpretation: Favors smaller variance components\n")
    cat("  Customization: prior_pc(U, alpha) or prior_half_normal(sd)\n\n")

    cat("Overdispersion (phi) [negbin/poisson_gamma only]:\n")
    cat("  PC prior: P(phi > 10) = 0.01\n")
    cat("  Interpretation: Regularizes toward Poisson\n")
    cat("  Customization: prior_pc(U, alpha) or prior_gamma(shape, rate)\n\n")

    if (temporal) {
      cat("Temporal autocorrelation (rho_temporal) [AR models only]:\n")
      cat("  Beta(2, 2)\n")
      cat("  Interpretation: Symmetric, centered at 0.5\n")
      cat("  Customization: prior_beta(alpha, beta)\n\n")
    }

    if (spatial) {
      cat("Spatial mixing (rho_spatial) [BYM2 only]:\n")
      cat("  Beta(1, 1) = Uniform(0, 1)\n")
      cat("  Interpretation: No preference for structured vs. unstructured\n")
      cat("  Customization: prior_beta(alpha, beta)\n\n")
    }

    cat("Family-specific notes:\n")
    cat("  negbin_negbin: Uses phi for both all processes\n")
    cat("  binomial: No overdispersion parameter (unless beta_binomial)\n")
    cat("  poisson_gamma: Uses phi for gamma shape parameter\n")

  } else {
    # Show defaults for specific family
    if (!inherits(family, "tulpa_family")) {
      stop("`family` must be a tulpa_family object", call. = FALSE)
    }

    family_name <- family$name

    cat(sprintf("Default priors for %s family\n", family_name))
    cat(paste(rep("=", nchar(family_name) + 24), collapse = ""), "\n\n")

    cat("Fixed effects (beta):\n")
    print_prior(defaults$beta, indent = "  ")
    cat("  Used for: All regression coefficients in all processes\n\n")

    cat("Random effect SD (sigma):\n")
    print_prior(defaults$sigma, indent = "  ")
    cat("  Used for: Standard deviation of group-level effects\n\n")

    # Family-specific parameters
    if (family_name %in% c("negbin_negbin", "negbin_gamma", "poisson_gamma", "gamma_gamma")) {
      cat("Overdispersion (phi):\n")
      print_prior(defaults$phi, indent = "  ")
      if (family_name %in% c("negbin_negbin", "negbin_gamma")) {
        cat("  Used for: NB2 size parameter (larger = less overdispersion)\n\n")
      } else if (family_name == "poisson_gamma") {
        cat("  Used for: Gamma shape parameter for effort/exposure\n\n")
      } else {
        cat("  Used for: Gamma shape parameters for both processes\n\n")
      }
    }

    if (family_name %in% c("beta_binomial_fixed", "beta_binomial")) {
      cat("Overdispersion (phi):\n")
      print_prior(defaults$phi, indent = "  ")
      cat("  Used for: Beta-binomial concentration parameter\n\n")
    }

    if (temporal) {
      cat("Temporal autocorrelation (rho_temporal):\n")
      print_prior(defaults$rho_temporal, indent = "  ")
      cat("  Used for: AR(1) correlation coefficient\n\n")
    }

    if (spatial) {
      cat("Spatial mixing (rho_spatial):\n")
      print_prior(defaults$rho_spatial, indent = "  ")
      cat("  Used for: BYM2 mixing proportion (structured vs. unstructured)\n\n")
    }

    cat("To customize, use tulpa_priors():\n")
    cat("  priors <- tulpa_priors(\n")
    cat("    beta = prior_normal(0, 1),\n")
    cat("    sigma = prior_pc(U = 0.5, alpha = 0.01)\n")
    cat("  )\n")
  }

  invisible(defaults)
}
