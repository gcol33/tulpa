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
# consumes. Intercept-only terms carry a marginal SD. A single CORRELATED
# slope term `(1 + x | g)` is auto-routed to the Sigma integrator before this
# point (see tulpa()), so it never reaches here; what remains is the
# uncorrelated `(1 + x || g)` case and slope terms mixed with other RE terms,
# which the conditional Laplace design path does not yet handle.
.bundle_to_re_list <- function(bundle, sigma_re) {
  re <- bundle$re_terms %||% list()
  lapply(seq_along(re), function(k) {
    rt <- re[[k]]
    if ((rt$n_coefs %||% 1L) > 1L) {
      stop(sprintf(
        "RE term %d has %d coefficients (random slopes). A *single* correlated\n",
        k, rt$n_coefs),
        "term `(1 + x | g)` is integrated over its covariance Sigma automatically\n",
        "(mode = 'laplace'); this term is uncorrelated `(... || g)` or mixed with\n",
        "other RE terms, which the design (Laplace) path does not yet handle. Use\n",
        "a logpost backend (mode = 'mala' / 'pathfinder'), or call tulpa_laplace()\n",
        "/ tulpa_re_cov_nested() directly.", call. = FALSE)
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


# Build the `prior` argument for tulpa_nested_laplace() from the formula's
# parsed latent blocks. Every `latent(...)` term resolves to a
# tulpa_latent_block (a tgmrf), which is itself a valid nested-Laplace prior
# block (it carries `type = "tgmrf"`). One block or many, the list is exactly
# the multi-block prior the driver consumes -- a length-1 list routes through
# the same multi-block path.
.latent_blocks_to_prior <- function(latent_blocks) {
  if (length(latent_blocks) == 0L) {
    stop("Internal error: .latent_blocks_to_prior() called with no blocks.",
         call. = FALSE)
  }
  latent_blocks
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control,
                               latent_blocks = list()) {
  input <- BACKEND_REGISTRY[[backend]]$input

  if (input == "nested") {
    if (backend != "nested_laplace") {
      stop(sprintf(paste0(
        "Backend '%s' is a nested engine driven by model packages, not the\n",
        "single-response tulpa() formula -- it needs multiple response arms,\n",
        "which a single formula cannot express. Call %s() directly."),
        backend, backend), call. = FALSE)
    }
    if (length(latent_blocks) == 0L) {
      stop("Backend 'nested_laplace' needs at least one `latent(...)` block ",
           "in the formula. For a plain GLMM use mode = 'laplace' / 'mala' / ",
           "'auto'.", call. = FALSE)
    }
    if (!is.null(beta_prior)) {
      warning("`beta_prior` is not threaded through tulpa()'s nested-Laplace ",
              "path; it is ignored. Call tulpa_nested_laplace() directly if you ",
              "need a fixed-effect prior here.", call. = FALSE)
    }
    # The nested driver carries a single iid random-intercept natively via
    # re_idx / n_re_groups / sigma_re (conditioned on, like the other tulpa()
    # backends). Richer RE structure should be modelled as an `iid` latent
    # block; surface that rather than silently dropping terms.
    re <- bundle$re_terms %||% list()
    re_idx      <- rep(0L, bundle$n_obs)
    n_re_groups <- 0L
    sigma_re_scalar <- 1.0
    if (length(re) > 0L) {
      if (length(re) > 1L || (re[[1]]$n_coefs %||% 1L) != 1L) {
        stop(paste0(
          "The nested-Laplace path supports at most one random-intercept term\n",
          "`(1 | g)` alongside latent blocks. For richer random-effect structure,\n",
          "model the extra grouping as an `iid` latent block, or call\n",
          "tulpa_nested_laplace() directly with an explicit RE layout."),
          call. = FALSE)
      }
      re_idx          <- as.integer(re[[1]]$group_idx)
      n_re_groups     <- re[[1]]$n_groups
      sigma_re_scalar <- sigma_re[1]
    }
    return(list(
      y           = bundle$y,
      n_trials    = n_trials %||% rep(1L, bundle$n_obs),
      X           = bundle$X,
      prior       = .latent_blocks_to_prior(latent_blocks),
      re_idx      = re_idx,
      n_re_groups = n_re_groups,
      sigma_re    = sigma_re_scalar,
      family      = family,
      phi         = phi,
      control     = control
    ))
  }

  if (input == "design") {
    if (backend %in% c("re_cov_nested", "re_cov_gibbs")) {
      # Correlated random-slope term `(1 + x | g)`: build the single `re_term`
      # the Sigma integrator / sampler consumes. The RE design Z is the
      # intercept column (if present) plus the slope columns, in coefficient
      # order (sigma_1 = intercept SD, sigma_2.. = slope SDs). The redirect in
      # tulpa() guarantees exactly one correlated term reaches here.
      rt0 <- (bundle$re_terms %||% list())[[1]]
      nc  <- rt0$n_coefs %||% 1L
      Z   <- cbind(
        if (isTRUE(rt0$has_intercept)) rep(1, bundle$n_obs) else NULL,
        rt0$slope_matrix
      )
      re_term <- list(idx = as.integer(rt0$group_idx),
                      n_groups = rt0$n_groups, n_coefs = nc,
                      Z = as.matrix(Z))
      common <- list(
        y = bundle$y, n_trials = n_trials %||% rep(1L, bundle$n_obs),
        X = bundle$X, re_term = re_term, family = family, phi = phi
      )
      if (backend == "re_cov_nested") {
        return(c(common, list(
          beta_prior  = beta_prior,
          integration = control$integration %||% "ccd",
          prior_sigma = control$prior_sigma %||% c(3, 0.05),
          eta         = control$eta %||% 2,
          n_per_axis  = control$n_per_axis %||% 5L,
          span        = control$span %||% 3,
          n_draws     = control$n_draws %||% 2000L,
          seed        = control$seed
        )))
      }
      # re_cov_gibbs: exact Metropolis-within-Gibbs debias.
      return(c(common, list(
        n_iter          = control$n_iter %||% 2000L,
        n_burnin        = control$n_burnin %||% control$warmup %||% 1000L,
        prior_df        = control$prior_df,
        prior_scale     = control$prior_scale,
        beta_prior_mean = beta_prior$mean %||% 0,
        beta_prior_sd   = beta_prior$sd %||% 100,
        seed            = control$seed
      )))
    }
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
#' * **Correlated random slopes** (`(1 + x | g)`, a single term) are supported on
#'   the Laplace (Tier 2) path: there is no scalar `sigma_re` to condition on, so
#'   the RE covariance `Sigma` is integrated rather than fixed. `mode = "laplace"`
#'   routes to the nested-Laplace `Sigma` integrator ([tulpa_re_cov_nested()],
#'   CCD design + PC/LKJ prior); `control$re_cov = "gibbs"` switches to the exact
#'   Metropolis-within-Gibbs debias ([tulpa_re_cov_gibbs()]). Both also run on the
#'   sampler path (`mode = "mala"` / `"pathfinder"`). Uncorrelated slopes
#'   (`(1 + x || g)`) and slope terms mixed with other RE terms still error on the
#'   design path with guidance (use a logpost backend or call the fitter directly).
#' * `mode = "gibbs"` (Polya-Gamma) fits a single random-intercept model for
#'   `family = "binomial"` or `"neg_binomial_2"`, and **samples** the RE sd
#'   rather than conditioning on `sigma_re`; tune it via `control$prior_sigma_scale`
#'   and a mean-zero `beta_prior`.
#' * **Latent prior blocks** (`latent(tgmrf(...))`) route to the nested-Laplace
#'   path (Tier 2), which integrates over the block hyperparameters. `mode =
#'   "auto"` and `"structured"` select it automatically when latent blocks are
#'   present; `mode = "nested_laplace"` forces it. At most one random-intercept
#'   `(1 | g)` term may accompany the blocks (model richer grouping as an `iid`
#'   block). Joint multi-arm nested models cannot be expressed by a
#'   single-response formula -- call [tulpa_nested_laplace_joint()] directly.
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
    has_spatial = FALSE, has_temporal = FALSE, has_latent = has_latent
  )

  # Latent prior blocks are consumed only by the nested-Laplace path. If the
  # user forced a non-nested backend (e.g. mode = "laplace" / "mala" / "exact"),
  # the blocks would otherwise be silently dropped -- fail loudly, and before
  # the generic reachability check so the latent-specific guidance wins (e.g.
  # mode = "exact" resolves to the unreachable `hmc`).
  if (has_latent && (BACKEND_REGISTRY[[sel$backend]]$input %||% "") != "nested") {
    stop(sprintf(paste0(
      "Formula has %d latent prior block(s) (`latent(...)`), which are integrated\n",
      "by the nested-Laplace backend. The selected backend '%s' (mode = '%s')\n",
      "does not consume latent blocks. Use mode = 'auto', 'structured', or\n",
      "'nested_laplace'."),
      parsed$n_latent_blocks, sel$backend, mode), call. = FALSE)
  }

  # A single correlated random-slope term `(1 + x | g)` has no scalar sigma_re
  # to condition on -- the inferred quantity is the RE covariance Sigma. Under
  # the Laplace (Tier 2) path, redirect to the nested-Laplace Sigma integrator
  # (default) or, with `control$re_cov = "gibbs"`, the exact Metropolis-within-
  # Gibbs debias. Multi-term / uncorrelated slope cases are left to
  # .bundle_to_re_list (which errors with guidance).
  re_terms <- bundle$re_terms %||% list()
  correlated_single <- length(re_terms) == 1L &&
    (re_terms[[1]]$n_coefs %||% 1L) > 1L && isTRUE(re_terms[[1]]$correlated)
  if (sel$backend == "laplace" && correlated_single) {
    re_cov_method <- match.arg(control$re_cov %||% "nested",
                               c("nested", "gibbs"))
    sel$backend <- if (re_cov_method == "gibbs") "re_cov_gibbs" else "re_cov_nested"
    ti <- get_backend_tier(sel$backend)
    sel$mode <- ti$mode; sel$tier <- ti$tier; sel$tier_name <- ti$name
    sel$reason <- sprintf(
      "single correlated random-slope term; RE covariance Sigma integrated via %s",
      sel$backend)
  }

  assert_backend_reachable(sel$backend)

  # Conditional backends (everything except the sigma-sampling Gibbs and the
  # Sigma-integrating re_cov backends) need one RE sd per term to condition on;
  # resolve/recycle it after the backend is known so the others do not emit a
  # misleading "conditioning" message.
  if (K > 0L && !sel$backend %in% c("gibbs", "re_cov_nested", "re_cov_gibbs")) {
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
                             n_trials, phi, beta_prior, control,
                             latent_blocks = parsed$latent_blocks)

  # sel$backend is itself a valid mode, so dispatch resolves to the same backend.
  fit <- tulpa_dispatch(
    sel$backend, fitter_args = args,
    family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = FALSE, has_latent = has_latent
  )

  if (is.list(fit)) {
    # Honour the front-door selection (including the correlated-RE redirect) over
    # tulpa_dispatch's re-resolution of the backend name, so `selection_reason`
    # reports the redirect rather than "user-specified backend".
    fit$inference_mode <- sel$mode
    fit$inference_tier <- sel$tier
    fit$backend <- sel$backend
    fit$selection_reason <- sel$reason
    fit$formula <- formula
    fit$family <- family
    fit$call <- match.call()
  }
  fit
}
