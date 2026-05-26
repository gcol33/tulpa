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
# consumes on the scalar-sigma_re design path. Multi-coefficient terms (any
# n_coefs > 1, correlated or uncorrelated) are auto-routed to the RE-covariance
# integrator before this point (see tulpa()), so only single-coefficient terms
# reach here: a random intercept `(1 | g)` (no Z) or a single random slope
# `(0 + x | g)` (which carries its slope column as Z). Each is conditioned on
# its marginal SD. The n_coefs > 1 guard is defensive (internal to this path).
.bundle_to_re_list <- function(bundle, sigma_re) {
  re <- bundle$re_terms %||% list()
  lapply(seq_along(re), function(k) {
    rt <- re[[k]]
    if ((rt$n_coefs %||% 1L) > 1L) {
      stop(sprintf(paste0(
        "Internal: RE term %d has %d coefficients (random slopes) on the scalar\n",
        "design path. Multi-coefficient terms should route to the RE-covariance\n",
        "integrator; use mode = 'laplace' (auto-redirects) or tulpa_re_cov_nested()."),
        k, rt$n_coefs), call. = FALSE)
    }
    list(idx = as.integer(rt$group_idx),
         n_groups = rt$n_groups,
         n_coefs = 1L,
         # `(1 | g)` has no Z (intercept indicator); `(0 + x | g)` supplies its
         # single slope column so the engine builds the right design.
         Z = if (isTRUE(rt$has_intercept)) NULL else rt$slope_matrix,
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


# Resolve a spatial(col) / temporal(col) column to 1-based contiguous unit
# indices for the per-observation field map. Integer/numeric columns are taken
# as already-1-based ids (matching adjacency row / coordinate order); factor or
# character columns are factored and the level order then defines the unit
# order (which must align with the adjacency / coordinate spec). `n_units`, when
# known (areal adjacency), bounds the index so an out-of-range id fails in R
# rather than indexing out of bounds in the C++ kernel.
.resolve_unit_index <- function(col, var, n_units = NULL) {
  if (is.factor(col)) {
    idx <- as.integer(col)
  } else if (is.numeric(col) && !anyNA(col) && all(col == as.integer(col))) {
    idx <- as.integer(col)
  } else {
    idx <- as.integer(as.factor(col))
  }
  if (anyNA(idx)) {
    stop("Spatial/temporal index column '", var, "' has missing values.",
         call. = FALSE)
  }
  if (min(idx) < 1L) {
    stop("Spatial/temporal index column '", var,
         "' must resolve to 1-based positive integers.", call. = FALSE)
  }
  if (!is.null(n_units) && max(idx) > n_units) {
    stop("Spatial index column '", var, "' references unit ", max(idx),
         " but the adjacency has only ", n_units, " unit(s).", call. = FALSE)
  }
  idx
}


# Convert the front-door spatial spec into the `prior` block that
# tulpa_nested_laplace() integrates over. Three families:
#  * Areal (icar/car/bym2/car_proper): built from type + adjacency + a 1-based
#    per-obs `spatial_idx` (as assembled in tulpa()), reusing
#    adjacency_to_csr_tulpa() -- the same 0-based CSR builder the conditional
#    spatial Laplace path uses. Intrinsic CAR ("car") shares the ICAR precision.
#    The nested driver consumes the 1-based per-obs spatial_idx.
#  * Continuous gp/nngp: built from a validated tulpa_gp spec (validate_gp() has
#    filled unique_coords / neighbor_info / obs_to_loc). The (coords, nn_order,
#    spatial_idx) convention matches the production conditional path
#    laplace_gp_at(): coords in ORIGINAL unique-location order, a 1-based
#    nn_order permutation (the nested registry subtracts 1), and spatial_idx
#    mapping each obs to its 1-based location. This is the indexing
#    batch_nngp_scatter() expects (it reads coords(nn_order[i])).
#  * Continuous hsgp: built from a validated tulpa_hsgp spec (validate_hsgp()
#    has filled coords_matrix at every observation). The Laplacian basis
#    (phi_basis N x M + matching lambda_eig) is built by cpp_hsgp_basis_2d --
#    a thin wrapper over setup_hsgp_2d, the single source of truth -- so the
#    basis math is never duplicated in R. The HSGP field is evaluated per
#    observation directly (no obs->location map).
# Field types outside .NL_FRONTDOOR_NESTED error rather than silently fall back.
.spatial_spec_to_nl_prior <- function(spatial) {
  type <- tolower(spatial$type)

  if (type %in% .NL_FRONTDOOR_AREAL) {
    adj <- as.matrix(spatial$adjacency)
    csr <- adjacency_to_csr_tulpa(adj)
    backend <- if (type == "car") "icar" else type
    prior <- list(
      type            = backend,
      spatial_idx     = as.integer(spatial$spatial_idx),
      n_spatial_units = as.integer(nrow(adj)),
      adj_row_ptr     = as.integer(csr$row_ptr),
      adj_col_idx     = as.integer(csr$col_idx),
      n_neighbors     = as.integer(csr$n_neighbors)
    )
    if (backend == "bym2") {
      prior$scale_factor <- as.numeric(spatial$scale_factor %||% 1.0)
    }
    if (backend == "car_proper" && !is.null(spatial$rho_bounds)) {
      prior$rho_bounds <- as.numeric(spatial$rho_bounds)
    }
    return(prior)
  }

  if (type == "hsgp") {
    # Hilbert-space GP: the field is a sum of Laplacian basis functions
    # evaluated at every observation, so the nested kernel takes a per-obs
    # phi_basis (N x M) + matching eigenvalues -- no obs->location map. The
    # basis is built by cpp_hsgp_basis_2d (over setup_hsgp_2d, the single
    # source of truth) from the validated coordinate matrix; sigma2/lengthscale
    # default in the registry.
    cm <- spatial$coords_matrix
    if (is.null(cm)) {
      stop("Internal: HSGP spatial spec is unvalidated (coords_matrix is NULL). ",
           "tulpa() validates it via validate_hsgp(); pass spatial_hsgp(~x+y).",
           call. = FALSE)
    }
    basis <- cpp_hsgp_basis_2d(as.matrix(cm), as.integer(spatial$m),
                               as.numeric(spatial$c))
    return(list(
      type       = "hsgp",
      phi_basis  = basis$phi_basis,
      lambda_eig = basis$lambda_eig
    ))
  }

  if (type %in% c("gp", "nngp")) {
    ni <- spatial$neighbor_info
    if (is.null(ni)) {
      stop("Internal: continuous spatial spec is unvalidated (neighbor_info is ",
           "NULL). tulpa() validates it via validate_gp(); pass spatial_gp(~x+y).",
           call. = FALSE)
    }
    n_spatial <- spatial$n_spatial %||% nrow(spatial$unique_coords)
    return(list(
      type        = "nngp",
      coords      = as.matrix(spatial$unique_coords),
      nn_idx      = as.matrix(ni$nn_idx),
      nn_dist     = as.matrix(ni$nn_dist),
      nn_order    = as.integer(ni$nn_order %||% seq_len(n_spatial)),
      n_spatial   = as.integer(n_spatial),
      nn          = as.integer(spatial$nn %||% ncol(ni$nn_idx)),
      cov_type    = gp_cov_type_for_laplace(spatial),
      spatial_idx = as.integer(spatial$obs_to_loc %||% seq_len(n_spatial))
    ))
  }

  stop(sprintf(paste0(
    "Nested-Laplace through tulpa() supports areal (%s) and continuous (%s)\n",
    "spatial fields; '%s' is not yet routed here. Call fit_spde() (spde)\n",
    "directly, or use mode = 'laplace' for a conditional fit at a fixed\n",
    "hyperparameter."),
    paste(.NL_FRONTDOOR_AREAL, collapse = ", "),
    paste(.NL_FRONTDOOR_CONTINUOUS, collapse = ", "), spatial$type), call. = FALSE)
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control,
                               latent_blocks = list(), spatial = NULL) {
  input <- BACKEND_REGISTRY[[backend]]$input

  if (input == "nested") {
    if (backend != "nested_laplace") {
      stop(sprintf(paste0(
        "Backend '%s' is a nested engine driven by model packages, not the\n",
        "single-response tulpa() formula -- it needs multiple response arms,\n",
        "which a single formula cannot express. Call %s() directly."),
        backend, backend), call. = FALSE)
    }
    # The nested driver integrates the hyperparameters of latent prior blocks
    # and/or an areal spatial field. A spatial(col) field becomes one areal
    # prior block (.spatial_spec_to_nl_prior); latent(...) terms are blocks
    # already. Either alone routes the single-block path; together they form a
    # multi-block prior (the driver integrates the joint hyperparameter grid).
    # At least one of the two must be present.
    spatial_block <- if (!is.null(spatial)) .spatial_spec_to_nl_prior(spatial) else NULL
    if (is.null(spatial_block) && length(latent_blocks) == 0L) {
      stop("Backend 'nested_laplace' needs at least one `latent(...)` block or ",
           "a spatial(col) field in the formula. For a plain GLMM use mode = ",
           "'laplace' / 'mala' / 'auto'.", call. = FALSE)
    }
    if (!is.null(beta_prior)) {
      warning("`beta_prior` is not threaded through tulpa()'s nested-Laplace ",
              "path; it is ignored. Call tulpa_nested_laplace() directly if you ",
              "need a fixed-effect prior here.", call. = FALSE)
    }
    if (is.null(spatial_block)) {
      prior <- .latent_blocks_to_prior(latent_blocks)   # latent-only (unchanged)
    } else if (length(latent_blocks) == 0L) {
      prior <- spatial_block                            # single areal block
    } else {
      prior <- c(list(spatial_block), latent_blocks)    # multi-block (spatial + latent)
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
      prior       = prior,
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
      # RE-covariance integrator / sampler: every RE term becomes a covariance
      # block. For each, the RE design Z is the intercept column (if present)
      # plus the slope columns, in coefficient order (sigma_1 = intercept SD,
      # sigma_2.. = slope SDs); `correlated` selects a full vs diagonal Sigma.
      # The redirect in tulpa() fires whenever a slope term is present.
      re <- bundle$re_terms %||% list()
      re_terms <- lapply(re, function(rt) {
        Z <- cbind(
          if (isTRUE(rt$has_intercept)) rep(1, bundle$n_obs) else NULL,
          rt$slope_matrix
        )
        list(idx        = as.integer(rt$group_idx),
             n_groups   = rt$n_groups,
             n_coefs    = rt$n_coefs %||% 1L,
             Z          = if (is.null(Z)) NULL else as.matrix(Z),
             correlated = isTRUE(rt$correlated),
             label      = rt$group_var)
      })
      common <- list(
        y = bundle$y, n_trials = n_trials %||% rep(1L, bundle$n_obs),
        X = bundle$X, re_terms = re_terms, family = family, phi = phi
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
      if (!is.null(spatial)) {
        # Spatial Laplace: route the field spec through tulpa_laplace(spatial=),
        # which dispatches on spatial$type (icar/car/bym2/spde/gp). At most one
        # random-intercept (1 | g) term may ride alongside the field -- the
        # spatial solvers consume a single RE block (re_list[[1]]); richer RE
        # structure is not supported here. beta_prior / offset are not threaded
        # through the spatial solvers, so reject them loudly rather than drop.
        re <- bundle$re_terms %||% list()
        if (length(re) > 1L || (length(re) == 1L && (re[[1]]$n_coefs %||% 1L) != 1L)) {
          stop("Spatial Laplace supports at most one random-intercept (1 | g) ",
               "term alongside the spatial field; drop the extra RE term(s).",
               call. = FALSE)
        }
        if (!is.null(beta_prior)) {
          stop("`beta_prior` is not supported on the spatial Laplace path; the ",
               "spatial solvers use a built-in weak fixed-effect prior. Drop ",
               "`beta_prior`, or use a sampler for a custom prior under a field.",
               call. = FALSE)
        }
        if (!is.null(bundle$offset)) {
          stop("offset() terms are not yet supported on the spatial Laplace ",
               "path through tulpa().", call. = FALSE)
        }
        return(list(
          y = bundle$y, n_trials = n_trials, X = bundle$X,
          re_list = .bundle_to_re_list(bundle, sigma_re),
          family = family, phi = phi, spatial = spatial
        ))
      }
      return(list(
        y = bundle$y, n_trials = n_trials, X = bundle$X,
        re_list = .bundle_to_re_list(bundle, sigma_re),
        family = family, phi = phi,
        offset = bundle$offset, beta_prior = beta_prior
      ))
    }
    if (backend == "gibbs") {
      re <- bundle$re_terms %||% list()
      # The spatial Polya-Gamma samplers carry the iid random-intercept block
      # alongside the field, so 0 or 1 `(1 | g)` term is allowed; the plain
      # Gibbs path requires exactly one. Either way the term must be a single
      # random intercept (no slopes).
      if (!is.null(spatial)) {
        if (length(re) > 1L || (length(re) == 1L && (re[[1]]$n_coefs %||% 1L) != 1L)) {
          stop("Spatial Gibbs supports at most one random-intercept (1 | g) ",
               "term alongside the spatial field.", call. = FALSE)
        }
      } else if (length(re) != 1L || (re[[1]]$n_coefs %||% 1L) != 1L) {
        stop("Gibbs (tulpa_gibbs) supports exactly one random-intercept term ",
             "(a single `(1 | g)`). Use a logpost backend (mode = 'mala') for ",
             "richer RE structure, or call tulpa_gibbs() directly.",
             call. = FALSE)
      }
      if (!is.null(spatial)) {
        if (family != "binomial") {
          stop(sprintf(paste0(
            "Spatial Gibbs supports family = 'binomial' only; got '%s'. Use ",
            "mode = 'laplace' for other families under a spatial field."), family),
            call. = FALSE)
        }
      } else if (!family %in% c("binomial", "neg_binomial_2")) {
        stop(sprintf(paste0(
          "Gibbs (tulpa_gibbs) supports family 'binomial' or 'neg_binomial_2'; ",
          "got '%s'. Use mode = 'laplace' or a logpost backend."), family),
          call. = FALSE)
      }
      if (!is.null(beta_prior$mean) && any(beta_prior$mean != 0)) {
        warning("Gibbs uses a mean-zero Gaussian prior on the fixed effects; ",
                "`beta_prior$mean` is ignored.", call. = FALSE)
      }
      # One `(1 | g)` -> that grouping; none -> a degenerate 0-group block that
      # the sampler treats as no iid RE (the spatial-only case).
      if (length(re) == 1L) {
        group <- as.integer(re[[1]]$group_idx); n_groups <- re[[1]]$n_groups
      } else {
        group <- rep(1L, bundle$n_obs); n_groups <- 0L
      }
      # tulpa_gibbs samples the RE sd (prior_sigma_scale); `sigma_re` is unused.
      return(list(
        y = bundle$y,
        n_trials = n_trials %||% rep(1L, bundle$n_obs),
        X = bundle$X,
        group = group,
        n_groups = n_groups,
        family = family,
        spatial = spatial,
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
#' * **Random slopes** are supported on the Laplace (Tier 2) path: there is no
#'   scalar `sigma_re` to condition on, so the RE covariance `Sigma` is integrated
#'   rather than fixed. This covers correlated terms (`(1 + x | g)`, a full
#'   `Sigma`), uncorrelated terms (`(1 + x || g)`, a diagonal `Sigma`), and
#'   several terms together (`(1 + x | g) + (1 | h)`) -- each term becomes a
#'   covariance block, and any accompanying `(1 | g)` term is integrated as a 1x1
#'   block (nothing is silently conditioned at `sigma_re = 1`). `mode = "laplace"`
#'   routes to the nested-Laplace `Sigma` integrator ([tulpa_re_cov_nested()],
#'   CCD design + PC/LKJ prior); `control$re_cov = "gibbs"` switches to the exact
#'   Metropolis-within-Gibbs debias ([tulpa_re_cov_gibbs()]). Both also run on the
#'   sampler path (`mode = "mala"` / `"pathfinder"`).
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
#' @param spatial Optional spatial-field spec. How it is addressed depends on the
#'   field family:
#'   * **Areal** (`"icar"`, `"car"`, `"bym2"`, `"car_proper"`): a list with `type`
#'     and `adjacency`, paired with a `spatial(col)` term in `formula` naming the
#'     per-observation unit column. Term and spec must be supplied together.
#'   * **Continuous** (`spatial_gp(~ lon + lat)` for an NNGP field,
#'     `spatial_hsgp(~ lon + lat)` for a Hilbert-space GP): the spec object
#'     carries the coordinate columns, so **no** `spatial(col)` term is used --
#'     observations are mapped to locations from their coordinates.
#'
#'   The mode selects how the spatial hyperparameter is handled:
#'   * `mode = "nested_laplace"`, `"structured"`, and `"auto"` (when not the
#'     binomial Gibbs case below) **integrate** the hyperparameter via
#'     [tulpa_nested_laplace()] -- the designed Tier 2 path, mirroring
#'     `latent(...)` blocks. Supported field types: areal `icar`/`car`/`bym2`/
#'     `car_proper` and continuous `gp`/`nngp`/`hsgp`. SPDE calls [fit_spde()]
#'     directly for now.
#'   * `mode = "laplace"` **conditions** on a fixed hyperparameter via
#'     [tulpa_laplace()] (the cheap explicit fit).
#'   * `mode = "gibbs"` routes the areal `icar`/`bym2` cases through the binomial
#'     Polya-Gamma samplers (Tier 1 exact); `mode = "auto"` picks this for a
#'     binomial `icar`/`bym2` field.
#' @param temporal Reserved: temporal terms are not yet routed through `tulpa()`.
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
                  spatial = NULL,
                  temporal = NULL,
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

  # Spatial field. The structure arrives via the `spatial=` argument; how it is
  # addressed depends on the field family:
  #  * Areal (icar/car/bym2/car_proper): a `spatial(col)` term names the
  #    per-observation unit column, resolved to a 1-based `spatial_idx` against
  #    the adjacency. Term and spec must appear together.
  #  * Continuous gp/nngp: a spatial_gp(~lon+lat) spec carries the coordinate
  #    columns; obs -> location is derived from the coordinates, so NO
  #    spatial(col) term is used. validate_gp() resolves the unique locations,
  #    obs_to_loc map, and NNGP neighbour structure onto the spec.
  spatial_spec <- spatial
  has_spatial  <- !is.null(parsed$spatial_var) || !is.null(spatial_spec)
  spatial_type <- NULL
  if (has_spatial) {
    if (is.null(spatial_spec)) {
      stop(sprintf(paste0(
        "Formula has a spatial(%s) term but the structure spec `spatial=` was ",
        "not supplied (e.g. list(type = 'icar', adjacency = W) or ",
        "spatial_gp(~ lon + lat))."),
        parsed$spatial_var), call. = FALSE)
    }
    if (is.null(spatial_spec$type)) {
      stop("`spatial$type` is required (e.g. 'icar', 'bym2', 'car', 'spde', 'gp').",
           call. = FALSE)
    }
    spatial_type <- spatial_spec$type
    sp_lc <- tolower(spatial_type)
    if (sp_lc %in% .NL_FRONTDOOR_CONTINUOUS) {
      # Coordinate-addressed field: coords come from the spec; no spatial(col)
      # term. validate_{gp,hsgp}() resolve the coordinate structure onto the
      # spec (gp/nngp: unique_coords / obs_to_loc / neighbor_info; hsgp:
      # coords_matrix at every observation for the basis builder).
      if (!is.null(parsed$spatial_var)) {
        stop("A continuous spatial field (", spatial_type, ") is addressed by ",
             "its coordinate columns in the spec; drop the spatial(",
             parsed$spatial_var, ") term from the formula.", call. = FALSE)
      }
      if (sp_lc == "hsgp") {
        if (!inherits(spatial_spec, "tulpa_hsgp")) {
          stop("An HSGP spatial field must be a spatial_hsgp(~ lon + lat) spec ",
               "object (it carries the coordinate columns); got a bare list.",
               call. = FALSE)
        }
        spatial_spec <- validate_hsgp(spatial_spec, data)
      } else {
        if (!inherits(spatial_spec, "tulpa_gp")) {
          stop("A continuous spatial field must be a spatial_gp(~ lon + lat) spec ",
               "object (it carries the coordinate columns); got a bare list.",
               call. = FALSE)
        }
        spatial_spec <- validate_gp(spatial_spec, data)
      }
    } else if (sp_lc %in% .NL_FRONTDOOR_AREAL) {
      # Areal field: spatial(col) names the per-observation unit.
      if (is.null(parsed$spatial_var)) {
        stop("`spatial=` was supplied but the formula has no spatial(col) term ",
             "naming the per-observation spatial unit. Add e.g. `+ spatial(region)`.",
             call. = FALSE)
      }
      if (!parsed$spatial_var %in% names(data)) {
        stop("spatial(", parsed$spatial_var, ") column not found in data.",
             call. = FALSE)
      }
      n_units <- if (!is.null(spatial_spec$adjacency)) {
        nrow(as.matrix(spatial_spec$adjacency))
      } else NULL
      spatial_spec$spatial_idx <-
        .resolve_unit_index(data[[parsed$spatial_var]], parsed$spatial_var, n_units)
    } else {
      stop("Spatial type '", spatial_type, "' is not yet routed through tulpa(). ",
           "Call its fitter directly (fit_spde() for SPDE), or use an areal ",
           "(icar/bym2/car/car_proper) or continuous (gp/nngp/hsgp) field.",
           call. = FALSE)
    }
  }

  # Temporal terms are not yet routed through tulpa() (planned -- see
  # dev_notes/plan_gibbs_spatial_frontdoor.md). Fail loudly rather than drop.
  if (!is.null(parsed$temporal_var) || !is.null(temporal)) {
    stop("Temporal terms are not yet routed through tulpa(). Call the temporal ",
         "fitter directly for now.", call. = FALSE)
  }

  fam_obj <- list(name = family, distribution = family)
  sel <- select_inference_mode(
    mode, family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_spatial, has_temporal = FALSE, has_latent = has_latent,
    spatial_type = spatial_type
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

  # A random-slope term (`(1 + x | g)` or `(1 + x || g)`) has no scalar sigma_re
  # to condition on -- the inferred quantity is the RE covariance Sigma. When any
  # term carries slopes, the Laplace (Tier 2) path redirects to the RE-covariance
  # integrator and treats EVERY term as a covariance block (correlated terms get
  # a full Sigma, uncorrelated `(... || g)` terms a diagonal one, and any
  # accompanying `(1 | g)` term a 1x1 block), so nothing is silently conditioned
  # at sigma_re = 1. `control$re_cov = "gibbs"` switches to the exact
  # Metropolis-within-Gibbs debias. Plain random-intercept-only models (no
  # slopes) keep the scalar-sigma_re design path via .bundle_to_re_list.
  re_terms <- bundle$re_terms %||% list()
  has_slope <- length(re_terms) > 0L &&
    any(vapply(re_terms, function(rt) (rt$n_coefs %||% 1L) > 1L, logical(1)))
  if (has_spatial && has_slope) {
    stop("Random-slope term(s) together with a spatial field are not supported ",
         "through tulpa() yet. Use a random intercept (1 | g) alongside the ",
         "spatial term, or drop the spatial field.", call. = FALSE)
  }
  if (sel$backend == "laplace" && has_slope) {
    re_cov_method <- match.arg(control$re_cov %||% "nested",
                               c("nested", "gibbs"))
    sel$backend <- if (re_cov_method == "gibbs") "re_cov_gibbs" else "re_cov_nested"
    ti <- get_backend_tier(sel$backend)
    sel$mode <- ti$mode; sel$tier <- ti$tier; sel$tier_name <- ti$name
    sel$reason <- sprintf(
      "random-slope term(s) present; RE covariance(s) integrated via %s (%d block(s))",
      sel$backend, length(re_terms))
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
                             latent_blocks = parsed$latent_blocks,
                             spatial = spatial_spec)

  # sel$backend is itself a valid mode, so dispatch resolves to the same backend.
  fit <- tulpa_dispatch(
    sel$backend, fitter_args = args,
    family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_spatial, has_latent = has_latent,
    spatial_type = spatial_type
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
