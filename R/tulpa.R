# tulpa.R
# ------------------------------------------------------------------------------
# The unified model entry point. tulpa() parses a formula, builds the model-data
# bundle, lets the tier/mode system pick a backend, assembles the arguments that
# backend's input contract requires, and dispatches. This is the user-facing
# surface that ties the formula layer, the family math (R/family_loglik.R), the
# GLMM log-posterior builder (R/glmm_logpost.R), and the dispatch spine
# (R/inference_modes.R) together.
# ------------------------------------------------------------------------------

# Map a model-data bundle's RE terms to the `re_list` that tulpa_laplace()
# consumes. Intercept-only terms carry a marginal SD; slope terms are not yet
# wired through the design path (use the logpost path or tulpa_laplace directly).
.bundle_to_re_list <- function(bundle, sigma_re) {
  re <- bundle$re_terms %||% list()
  lapply(seq_along(re), function(k) {
    rt <- re[[k]]
    if ((rt$n_coefs %||% 1L) > 1L) {
      stop(sprintf(
        "RE term %d has %d coefficients (random slopes). Slope terms are not yet\n",
        k, rt$n_coefs),
        "wired through tulpa()'s design (Laplace) path. Use a logpost backend\n",
        "(mode = 'mala' / 'pathfinder'), or call tulpa_laplace() with an explicit\n",
        "re_list.", call. = FALSE)
    }
    list(idx = as.integer(rt$group_idx),
         n_groups = rt$n_groups,
         n_coefs = 1L,
         sigma = sigma_re[k])
  })
}


# Find the joint MAP and the positive-definite precision (-Hessian at the mode)
# of a GLMM log-posterior. This is the Laplace proposal imh_laplace consumes
# (it forms N(mode, scale^2 * precision^{-1})). Hessian is numerical via
# optimHess using the analytic gradient; suited to low-dimensional joints.
.glmm_mode_precision <- function(m, maxit = 500L) {
  opt <- stats::optim(m$init, fn = m$log_posterior, gr = m$grad_log_posterior,
                      method = "BFGS", control = list(fnscale = -1, maxit = maxit))
  H <- stats::optimHess(opt$par, fn = m$log_posterior, gr = m$grad_log_posterior)
  prec <- -H
  prec <- 0.5 * (prec + t(prec))   # symmetrise
  list(mode = opt$par, precision = prec, convergence = opt$convergence)
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control) {
  input <- BACKEND_REGISTRY[[backend]]$input

  if (input == "design") {
    if (backend == "laplace") {
      return(list(
        y = bundle$y, n_trials = n_trials, X = bundle$X,
        re_list = .bundle_to_re_list(bundle, sigma_re),
        family = family, phi = phi,
        offset = bundle$offset, beta_prior = beta_prior
      ))
    }
    if (backend == "gibbs") {
      re <- bundle$re_terms %||% list()
      if (length(re) != 1L || (re[[1]]$n_coefs %||% 1L) != 1L) {
        stop("Gibbs (tulpa_gibbs) supports exactly one random-intercept term ",
             "(a single `(1 | g)`). Use a logpost backend (mode = 'mala') for ",
             "richer RE structure, or call tulpa_gibbs() directly.",
             call. = FALSE)
      }
      if (!family %in% c("binomial", "neg_binomial_2")) {
        stop(sprintf(paste0(
          "Gibbs (tulpa_gibbs) supports family 'binomial' or 'neg_binomial_2'; ",
          "got '%s'. Use mode = 'laplace' or a logpost backend."), family),
          call. = FALSE)
      }
      if (!is.null(beta_prior$mean) && any(beta_prior$mean != 0)) {
        warning("Gibbs uses a mean-zero Gaussian prior on the fixed effects; ",
                "`beta_prior$mean` is ignored.", call. = FALSE)
      }
      # tulpa_gibbs samples the RE sd (prior_sigma_scale); `sigma_re` is unused.
      return(list(
        y = bundle$y,
        n_trials = n_trials %||% rep(1L, bundle$n_obs),
        X = bundle$X,
        group = as.integer(re[[1]]$group_idx),
        n_groups = re[[1]]$n_groups,
        family = family,
        iter = control$iter %||% control$n_iter %||% 2000L,
        warmup = control$warmup %||% 1000L,
        prior_beta_sd = beta_prior$sd %||% 10.0,
        prior_sigma_scale = control$prior_sigma_scale %||% 2.5,
        verbose = FALSE
      ))
    }
    stop(sprintf(
      "Backend '%s' is reachable but not yet wired through tulpa(). Call its\n",
      backend),
      "fitter directly (e.g. agq_fit()).", call. = FALSE)
  }

  if (input == "logpost") {
    m <- build_glmm_logpost(bundle, family, sigma_re = sigma_re,
                            n_trials = n_trials, phi = phi,
                            beta_prior = beta_prior %||% list(mean = 0, sd = 2.5))
    if (backend == "mala") {
      return(list(
        log_posterior = m$log_posterior,
        grad_log_posterior = m$grad_log_posterior,
        init = m$init,
        n_iter = control$n_iter %||% 2000L,
        warmup = control$warmup %||% (control$n_iter %||% 2000L) %/% 2L,
        epsilon = control$epsilon %||% 0.1
      ))
    }
    if (backend == "pathfinder") {
      return(list(
        log_posterior = m$log_posterior,
        init = m$init,
        grad_log_posterior = m$grad_log_posterior,
        n_draws = control$n_draws %||% 1000L
      ))
    }
    if (backend == "imh_laplace") {
      # Independence MH with a Laplace proposal: needs the MAP + precision.
      mp <- .glmm_mode_precision(m)
      n_iter <- control$n_iter %||% 2000L
      return(list(
        log_posterior = m$log_posterior,
        mode = mp$mode,
        hessian = mp$precision,
        n_iter = n_iter,
        warmup = control$warmup %||% (n_iter %/% 2L),
        scale = control$scale %||% 1.0
      ))
    }
    stop(sprintf(
      "Backend '%s' is reachable but not yet wired through tulpa(). Call its\n",
      backend),
      "fitter directly.", call. = FALSE)
  }

  stop(sprintf("Backend '%s' (input '%s') is not supported by tulpa() yet.",
               backend, input), call. = FALSE)
}


#' Fit a tulpa model
#'
#' @description
#' Single entry point for fitting a Bayesian hierarchical model. `tulpa()`
#' parses the formula, builds the model matrices, selects an inference backend
#' through the tier/mode system (see [inference_mode_info()]), assembles the
#' arguments that backend needs, and dispatches.
#'
#' The fit conditions on the random-effect standard deviations `sigma_re` (and,
#' for non-Gaussian dispersion, `phi`): both the Laplace (Tier 2) and the
#' sampler (Tier 1) paths target the posterior given these. Integrating over the
#' hyperparameters is the role of the nested-Laplace / EM layer.
#'
#' @section Coverage:
#' * **No random effects** and **random intercepts** (`(1 | g)`) are supported on
#'   the design path (`mode = "laplace"`) and the sampler path (`mode = "mala"`,
#'   `"pathfinder"`, `"imh_laplace"`).
#' * **Random slopes** (`(1 + x | g)`) are supported on the sampler path; the
#'   design path errors with guidance (use a logpost backend or `tulpa_laplace()`
#'   directly).
#' * `mode = "gibbs"` (Polya-Gamma) fits a single random-intercept model for
#'   `family = "binomial"` or `"neg_binomial_2"`, and **samples** the RE sd
#'   rather than conditioning on `sigma_re`; tune it via `control$prior_sigma_scale`
#'   and a mean-zero `beta_prior`.
#' * Selecting a backend whose kernel is C-ABI-only (e.g. `ess`, `sghmc`, `smc`)
#'   errors loudly -- those are reachable from model packages, not from R yet.
#'
#' @param formula A model formula. Fixed effects, `(1 | g)` / `(1 + x | g)`
#'   random effects, and `offset(...)` terms are recognised.
#' @param data A data frame.
#' @param family Character family name: one of [family_names()]
#'   (`"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`, `"beta"`).
#' @param mode Inference mode or backend. `"auto"` (default) picks the most
#'   reliable Tier 1/Tier 2 method expected to finish; a tier (`"exact"`,
#'   `"structured"`) or a backend name (`"laplace"`, `"mala"`, ...) forces it.
#' @param sigma_re Random-effect SDs to condition on: length 1 (recycled) or one
#'   per RE term. Defaults to 1 per term with a message.
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL`.
#' @param phi Dispersion/precision passed to the family (residual variance for
#'   gaussian, size for neg_binomial_2, precision for beta).
#' @param beta_prior Optional `list(mean, sd)` Gaussian prior on the fixed
#'   effects.
#' @param control Optional list of backend tuning arguments (e.g. `n_iter`,
#'   `warmup`, `epsilon` for `mala`; `n_draws` for `pathfinder`).
#' @param ... Reserved for future use.
#'
#' @return A `tulpa_fit` object carrying the backend's output plus
#'   `inference_mode`, `inference_tier`, `backend`, `selection_reason`,
#'   `formula`, and `family`.
#'
#' @seealso [inference_mode_info()], [tulpa_laplace()], [mala()], [pathfinder()]
#' @export
tulpa <- function(formula, data,
                  family = "gaussian",
                  mode = "auto",
                  sigma_re = NULL,
                  n_trials = NULL,
                  phi = 1.0,
                  beta_prior = NULL,
                  control = list(),
                  ...) {
  if (is.null(.FAMILY_OPS[[family]])) {
    stop(sprintf("Unknown family '%s'. Supported: %s.",
                 family, paste(family_names(), collapse = ", ")), call. = FALSE)
  }

  parsed <- tulpa_parse_formula(formula)
  bundle <- tulpa_build_model_data(parsed, data)
  K <- length(bundle$re_terms %||% list())

  has_latent <- (parsed$n_latent_blocks %||% 0L) > 0L

  fam_obj <- list(name = family, distribution = family)
  sel <- select_inference_mode(
    mode, family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_latent, has_temporal = FALSE, has_latent = has_latent
  )
  assert_backend_reachable(sel$backend)

  # Conditional backends (everything except the sigma-sampling Gibbs) need one
  # RE sd per term to condition on; resolve/recycle it after the backend is
  # known so Gibbs does not emit a misleading "conditioning" message.
  if (K > 0L && sel$backend != "gibbs") {
    if (is.null(sigma_re)) {
      sigma_re <- rep(1, K)
      message("tulpa(): `sigma_re` not supplied; conditioning on sigma_re = 1 for ",
              "each of the ", K, " RE term(s). Pass `sigma_re` to override.")
    } else if (length(sigma_re) == 1L) {
      sigma_re <- rep(sigma_re, K)
    } else if (length(sigma_re) != K) {
      stop(sprintf("`sigma_re` must have length 1 or %d (one per RE term).", K),
           call. = FALSE)
    }
  }

  args <- .tulpa_fitter_args(sel$backend, bundle, family, sigma_re,
                             n_trials, phi, beta_prior, control)

  # sel$backend is itself a valid mode, so dispatch resolves to the same backend.
  fit <- tulpa_dispatch(
    sel$backend, fitter_args = args,
    family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_latent, has_latent = has_latent
  )

  if (is.list(fit)) {
    fit$formula <- formula
    fit$family <- family
    fit$call <- match.call()
  }
  fit
}
