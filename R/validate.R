#' Posterior predictive checks for tulpa models
#'
#' @description
#' Visual and numerical checks comparing observed data to posterior
#' predictive distributions. Essential for assessing model fit.
#'
#' @return The functions documented in this family return a `ggplot` object
#'   ([pp_check()]) or a `tulpa_prior_predict` object ([prior_predict()]); see
#'   each function's own help page.
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
#' # pp_check is a generic; model packages (e.g. tulpaObs, tulpaRatio) provide
#' # the posterior-predictive method for their fits.
#' \donttest{
#' set.seed(123)
#' n <- 200L
#' df <- data.frame(y = rpois(n, 5), x = rnorm(n),
#'                  site = factor(rep(1:10, each = 20)))
#' fit <- tulpa(y ~ x + (1 | site), data = df, family = "poisson",
#'              mode = "hmc", control = list(n_iter = 500, warmup = 250))
#' # Density overlay (dispatches to the model package's pp_check method).
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

  # Stored replicates (model packages put y_rep in a list-shaped $draws);
  # otherwise generate them from the fit's own family via posterior_predict().
  yrep <- if (is.list(object$draws) && !is.matrix(object$draws)) {
    object$draws$y_rep
  } else NULL
  if (is.null(yrep)) {
    yrep <- tryCatch(
      posterior_predict(object, ndraws = max(ndraws, 100L)),
      error = function(e) {
        stop("Posterior predictive draws not available: ",
             conditionMessage(e), call. = FALSE)
      }
    )
  }

  # Observed data
  y <- object$.internal$hmc_data$y %||% object$y
  if (is.null(y)) {
    stop("pp_check() needs the observed response; this fit stores no $y.",
         call. = FALSE)
  }

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
#' Draw datasets from the prior predictive distribution: parameters are sampled
#' from their priors (no data conditioning) and pushed through the model's
#' linear predictor and the family's simulator.
#'
#' Useful for checking whether priors imply plausible data ranges before fitting.
#'
#' @param formula A model formula (e.g., `y ~ x + (1 | g)`). For multi-process
#'   families, a list of formulas keyed by process name.
#' @param family A `tulpa_family` object exposing a `simulate_fn` (see
#'   [tulpa_family()]). Model packages (tulpaRatio, tulpaObs) provide families;
#'   tests can build a minimal one with [tulpa_family()].
#' @param data Data frame containing covariates and grouping factors. Used for
#'   dimensions and design matrices; the response column may be absent or NA.
#' @param priors Prior specification ([tulpa_priors()]). If `NULL`, uses defaults.
#' @param n_draws Number of prior parameter draws. Default 100.
#' @param seed Optional integer seed for reproducibility.
#' @param ... Passed to `family$simulate_fn`.
#'
#' @return A `tulpa_prior_predict` object: a list with
#'   - `y`: list of length `n_draws`, each element the simulated response for
#'     that draw (matching whatever shape `family$simulate_fn` returns).
#'   - `theta`: list of length `n_draws` of parameter draws (`beta`, `sigma`,
#'     RE coefficients `u`, family-specific extras).
#'   - `linpred`: list of length `n_draws`, each a list of linear predictor
#'     vectors per process.
#'   - `family`: the family used.
#'   - `n_draws`, `n_obs`.
#'
#' @examples
#' # Toy Gaussian family for illustration
#' fam <- tulpa_family(
#'   name = "gaussian",
#'   simulate_fn = function(eta, params, n_obs, ...) {
#'     rnorm(n_obs, eta[[1]], params$sigma_y)
#'   },
#'   extra_params = list(sigma_y = prior_half_normal(1))
#' )
#' df <- data.frame(y = rep(0, 20), x = rnorm(20))
#' pp <- prior_predict(y ~ x, fam, df, n_draws = 50, seed = 1)
#' length(pp$y)  # 50
#'
#' @export
prior_predict <- function(formula, family, data,
                          priors = NULL, n_draws = 100,
                          seed = NULL, ...) {

  if (!is.null(seed)) {
    old_seed <- if (exists(".Random.seed", envir = .GlobalEnv)) {
      get(".Random.seed", envir = .GlobalEnv)
    } else NULL
    set.seed(seed)
    on.exit({
      if (is.null(old_seed)) {
        rm(".Random.seed", envir = .GlobalEnv)
      } else {
        assign(".Random.seed", old_seed, envir = .GlobalEnv)
      }
    }, add = TRUE)
  }

  validate_family(family)
  if (is.null(priors)) priors <- tulpa_priors()
  if (!inherits(priors, "tulpa_priors")) {
    stop("`priors` must be a tulpa_priors object", call. = FALSE)
  }
  n_draws <- as.integer(n_draws)
  if (length(n_draws) != 1L || is.na(n_draws) || n_draws < 1L) {
    stop("`n_draws` must be a positive integer", call. = FALSE)
  }

  process_names <- family$process_names
  n_processes <- length(process_names)
  formulas <- normalize_formulas(formula, process_names)

  parsed <- lapply(formulas, tulpa_parse_formula)
  built  <- lapply(parsed, tulpa_build_model_data, data = data)
  n_obs <- built[[1]]$n_obs

  y_list      <- vector("list", n_draws)
  theta_list  <- vector("list", n_draws)
  linpred_list <- vector("list", n_draws)

  for (d in seq_len(n_draws)) {
    theta <- draw_theta_from_priors(built, priors, family)
    eta <- build_eta(built, theta, family$link_inv, apply_link = FALSE)
    sim_args <- c(list(eta = eta, params = theta$extras, n_obs = n_obs), list(...))
    y_list[[d]] <- do.call(family$simulate_fn, sim_args)
    theta_list[[d]] <- theta
    linpred_list[[d]] <- eta
  }

  structure(
    list(
      y = y_list,
      theta = theta_list,
      linpred = linpred_list,
      family = family,
      n_draws = n_draws,
      n_obs = n_obs,
      process_names = process_names
    ),
    class = "tulpa_prior_predict"
  )
}


#' Print method for tulpa_prior_predict
#'
#' @param x A tulpa_prior_predict object
#' @param ... Ignored
#' @return The input `x`, returned invisibly. Called for the side effect of
#'   printing a summary of the prior predictive draws to the console.
#' @export
print.tulpa_prior_predict <- function(x, ...) {
  .print_draws_summary(x, "tulpa prior predictive draws", "Draws:     ", x$n_draws)
}


#' Plot method for tulpa_prior_predict
#'
#' @description
#' Density overlay of prior predictive draws. Requires bayesplot; falls back to
#' base graphics matplot of a subset of draws otherwise.
#'
#' @param x A tulpa_prior_predict object
#' @param process Process index or name (multi-process families)
#' @param max_draws Maximum draws to overlay. Default 50.
#' @param ... Passed through.
#' @return A `ggplot` object (via bayesplot) when bayesplot is installed;
#'   otherwise `NULL` invisibly, after drawing a base-graphics overlay. Called
#'   for the density overlay of the prior predictive draws.
#' @export
plot.tulpa_prior_predict <- function(x, process = 1L, max_draws = 50L, ...) {

  if (is.character(process)) {
    pname <- process
    process <- match(process, x$process_names)
    if (is.na(process)) stop("Unknown process: ", pname, call. = FALSE)
  }

  draws <- if (length(x$process_names) > 1L) {
    lapply(x$y, function(yi) yi[[process]])
  } else {
    x$y
  }

  # Coerce to draws x N matrix when possible
  lengths <- vapply(draws, length, integer(1))
  if (length(unique(lengths)) != 1L) {
    stop("Cannot plot: simulated draws have inconsistent lengths", call. = FALSE)
  }
  m <- do.call(rbind, lapply(draws, function(d) {
    if (is.numeric(d)) as.numeric(d) else as.numeric(unclass(d))
  }))

  n_keep <- min(max_draws, nrow(m))
  idx <- seq_len(n_keep)
  if (nrow(m) > n_keep) idx <- sample.int(nrow(m), n_keep)

  if (requireNamespace("bayesplot", quietly = TRUE)) {
    bayesplot::ppd_dens_overlay(m[idx, , drop = FALSE], ...)
  } else {
    graphics::matplot(t(m[idx, , drop = FALSE]), type = "l",
                      col = grDevices::adjustcolor("steelblue", alpha.f = 0.3),
                      lty = 1, xlab = "Observation", ylab = "y_rep",
                      main = sprintf("Prior predictive draws (%s)",
                                     x$process_names[process]))
  }
}

# Extract the [n_draws x n_obs] pointwise log-likelihood from a tulpa_fit:
# the combined `log_lik`, the sum of the two-process num/denom components, or
# -- for engine fits with a builtin character family and a stored response --
# computed from the linear-predictor posterior draws and the family registry.
.tulpa_fit_loglik <- function(x) {
  if (is.list(x$draws) && !is.matrix(x$draws)) {
    ll <- x$draws$log_lik
    if (!is.null(ll)) return(ll)
    ll_num <- x$draws$log_lik_num
    ll_denom <- x$draws$log_lik_denom
    if (!is.null(ll_num) && !is.null(ll_denom)) return(ll_num + ll_denom)
    if (!is.null(ll_num)) return(ll_num)
    if (!is.null(ll_denom)) return(ll_denom)
  }
  if (!is.null(x$y) && is.character(x$family) && length(x$family) == 1L) {
    eta <- tryCatch(.tulpa_eta_draws(x), error = function(e) NULL)
    if (!is.null(eta)) {
      n_obs <- ncol(eta)
      S <- nrow(eta)
      Y <- matrix(as.numeric(x$y), S, n_obs, byrow = TRUE)
      nt <- x$n_trials %||% 1
      NT <- matrix(as.numeric(nt), S, n_obs, byrow = TRUE)
      ll <- family_loglik(eta, Y, x$family, n_trials = NT,
                          phi = x$phi %||% 1.0, phi2 = x$phi2)
      dim(ll) <- dim(eta)
      return(ll)
    }
  }
  stop("Log-likelihood not found in model output.", call. = FALSE)
}
