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

  # SPDE never reaches the generic converter -- tulpa() redirects an SPDE field
  # to the dedicated `spde` backend (fit_spde) before .tulpa_fitter_args builds a
  # nested prior. Any other type is genuinely unsupported on this path.
  stop(sprintf(paste0(
    "The generic nested-Laplace converter supports areal (%s) and continuous\n",
    "(%s) spatial fields; '%s' is not routed here. Use one of those, or\n",
    "mode = 'laplace' for a conditional fit at a fixed hyperparameter."),
    paste(.NL_FRONTDOOR_AREAL, collapse = ", "),
    paste(.NL_FRONTDOOR_CONTINUOUS, collapse = ", "), spatial$type), call. = FALSE)
}


# Convert a validated RW1 temporal spec into the areal ICAR prior block the
# nested driver integrates. An RW1 random walk on T ordered time points is an
# intrinsic CAR on a 1D chain: time t neighbours t-1 and t+1 (and t=T neighbours
# t=1 when cyclic), so the chain adjacency's ICAR precision Q = diag(n_nb) - W is
# exactly the RW1 precision. The per-observation time index becomes spatial_idx.
# This reuses adjacency_to_csr_tulpa() and the icar nested kernel -- the same
# tested path the areal spatial field takes.
.temporal_spec_to_nl_prior <- function(temporal) {
  T <- temporal$n_times
  if (is.null(T) || T < 2L) {
    stop("Internal: temporal spec is unvalidated (n_times missing). tulpa() ",
         "validates it via validate_temporal().", call. = FALSE)
  }
  W <- matrix(0, T, T)
  if (isTRUE(temporal$cyclic)) {
    for (t in seq_len(T)) {
      nx <- if (t == T) 1L else t + 1L
      W[t, nx] <- 1; W[nx, t] <- 1
    }
  } else {
    for (t in seq_len(T - 1L)) { W[t, t + 1L] <- 1; W[t + 1L, t] <- 1 }
  }
  csr <- adjacency_to_csr_tulpa(W)
  list(
    type            = "icar",
    spatial_idx     = as.integer(temporal$time_index),
    n_spatial_units = as.integer(T),
    adj_row_ptr     = as.integer(csr$row_ptr),
    adj_col_idx     = as.integer(csr$col_idx),
    n_neighbors     = as.integer(csr$n_neighbors)
  )
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control,
                               latent_blocks = list(), spatial = NULL,
                               temporal = NULL) {
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
    # and/or one field. A spatial(col) field becomes an areal prior block
    # (.spatial_spec_to_nl_prior); a temporal RW1 field becomes the same areal
    # block over its time chain (.temporal_spec_to_nl_prior); latent(...) terms
    # are blocks already. tulpa() forbids combining a spatial and a temporal
    # field, so at most one field block is present here. The field alone routes
    # the single-block path; with latent terms they form a multi-block prior.
    spatial_block  <- if (!is.null(spatial))  .spatial_spec_to_nl_prior(spatial)   else NULL
    temporal_block <- if (!is.null(temporal)) .temporal_spec_to_nl_prior(temporal) else NULL
    field_block <- spatial_block %||% temporal_block
    if (is.null(field_block) && length(latent_blocks) == 0L) {
      stop("Backend 'nested_laplace' needs at least one `latent(...)` block, a ",
           "spatial(col) field, or a temporal field. For a plain GLMM use mode = ",
           "'laplace' / 'mala' / 'auto'.", call. = FALSE)
    }
    if (!is.null(beta_prior)) {
      warning("`beta_prior` is not threaded through tulpa()'s nested-Laplace ",
              "path; it is ignored. Call tulpa_nested_laplace() directly if you ",
              "need a fixed-effect prior here.", call. = FALSE)
    }
    if (is.null(field_block)) {
      prior <- .latent_blocks_to_prior(latent_blocks)   # latent-only (unchanged)
    } else if (length(latent_blocks) == 0L) {
      prior <- field_block                              # single field block
    } else {
      prior <- c(list(field_block), latent_blocks)      # multi-block (field + latent)
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
    # Retain the per-grid fixed-effect Hessians by default so summary()/vcov()
    # can report the grid-marginalized fixed-effect SE (the within-grid Laplace
    # covariance is needed for it). Cheap for the small fixed-effect block;
    # users can switch it off via control$keep_grid_hessians = FALSE.
    control$keep_grid_hessians <- control$keep_grid_hessians %||% TRUE
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

  if (input == "spde") {
    # Continuous Matern SPDE field, nested-Laplace integrated over (range, sigma)
    # by fit_spde() (its own CCD / grid engine). fit_spde() takes the design
    # bundle (y, X) plus the self-contained SPDE spec; it has no RE / latent /
    # beta_prior / offset support, so reject those loudly rather than drop them.
    if (length(latent_blocks) > 0L) {
      stop("An SPDE spatial field cannot be combined with latent(...) blocks ",
           "through tulpa(); fit_spde() integrates a single Matern field. Drop ",
           "the latent block(s).", call. = FALSE)
    }
    re <- bundle$re_terms %||% list()
    if (length(re) > 0L) {
      stop("The SPDE nested-Laplace path does not support an additional ",
           "random-effect term yet; drop the (1 | g) term, or use mode = ",
           "'exact' for a sampler under the field.", call. = FALSE)
    }
    if (!is.null(beta_prior)) {
      stop("`beta_prior` is not supported on the SPDE path; fit_spde() uses a ",
           "built-in weak fixed-effect prior.", call. = FALSE)
    }
    if (!is.null(bundle$offset)) {
      stop("offset() terms are not yet supported on the SPDE path through tulpa().",
           call. = FALSE)
    }
    if (!family %in% c("binomial", "poisson", "neg_binomial_2")) {
      stop(sprintf(
        "SPDE supports family 'binomial', 'poisson', or 'neg_binomial_2'; got '%s'.",
        family), call. = FALSE)
    }
    method <- match.arg(control$method %||% "ccd", c("ccd", "grid"))
    return(list(
      y              = bundle$y,
      X              = bundle$X,
      spatial        = spatial,
      family         = family,
      n_trials       = n_trials %||% rep(1L, bundle$n_obs),
      range          = NULL,
      sigma          = NULL,
      nested_laplace = TRUE,
      method         = method,
      n_grid         = control$n_grid %||% 5L,
      phi            = phi,
      max_iter       = control$max_iter %||% 100L,
      tol            = control$tol %||% 1e-6,
      n_threads      = control$n_threads %||% 1L
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
#'     `spatial_hsgp(~ lon + lat)` for a Hilbert-space GP, `spatial_spde(~ lon +
#'     lat, data)` for a Matern SPDE field): the spec object carries the
#'     coordinate columns (the SPDE spec also carries the mesh + FEM matrices),
#'     so **no** `spatial(col)` term is used -- observations are mapped to
#'     locations from their coordinates.
#'
#'   The mode selects how the spatial hyperparameter is handled:
#'   * `mode = "nested_laplace"`, `"structured"`, and `"auto"` (when not the
#'     binomial Gibbs case below) **integrate** the hyperparameter -- the
#'     designed Tier 2 path, mirroring `latent(...)` blocks. Areal `icar`/`car`/
#'     `bym2`/`car_proper` and continuous `gp`/`nngp`/`hsgp` go through
#'     [tulpa_nested_laplace()]; SPDE is redirected to [fit_spde()], which
#'     integrates `(range, sigma)` with its own CCD / grid design.
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
  #  * Continuous gp/nngp/hsgp: a spatial_gp(~lon+lat) / spatial_hsgp(~lon+lat)
  #    spec carries the coordinate columns; obs -> location is derived from the
  #    coordinates, so NO spatial(col) term is used. validate_gp()/validate_hsgp()
  #    resolve the coordinate structure onto the spec.
  #  * Continuous spde: a spatial_spde(~lon+lat, data) spec is self-contained
  #    (mesh + FEM matrices built at construction); also coord-addressed (no
  #    spatial(col) term). Routed to the dedicated `spde` backend (fit_spde).
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
    # RSR is an areal field (icar/car) carrying a projection modifier: the spec
    # keeps the underlying $type but flags $rsr (spatial_rsr()). Route it as its
    # own gibbs-only areal type so it reaches the RSR Polya-Gamma sampler instead
    # of the plain areal / nested path, which would silently drop the projection.
    if (isTRUE(spatial_spec$rsr) || identical(sp_lc, "rsr")) {
      spatial_type <- "rsr"
      sp_lc <- "rsr"
    }
    if (sp_lc %in% c(.NL_FRONTDOOR_CONTINUOUS, .NL_FRONTDOOR_SPDE)) {
      # Coordinate-addressed field: coords come from the spec; no spatial(col)
      # term. gp/nngp/hsgp resolve their coordinate structure via validate_*()
      # (gp/nngp: unique_coords / obs_to_loc / neighbor_info; hsgp:
      # coords_matrix at every observation for the basis builder). SPDE is
      # self-contained -- spatial_spde() built the mesh + FEM matrices (A, C, G)
      # at construction -- so it only needs a dimension check.
      if (!is.null(parsed$spatial_var)) {
        stop("A continuous spatial field (", spatial_type, ") is addressed by ",
             "its coordinate columns in the spec; drop the spatial(",
             parsed$spatial_var, ") term from the formula.", call. = FALSE)
      }
      if (sp_lc == "spde") {
        # The SPDE projector A maps observations -> mesh nodes; it must have one
        # row per observation in `data`. spatial_spde() builds A from the same
        # data, so a mismatch means the spec was built from a different frame.
        n_a <- tryCatch(nrow(spatial_spec$A), error = function(e) NULL)
        if (is.null(n_a) || n_a != bundle$n_obs) {
          stop("SPDE projector matrix A has ", n_a %||% "?", " row(s) but `data` ",
               "has ", bundle$n_obs, " observation(s). Build the SPDE spec from ",
               "the same data (spatial_spde(~ lon + lat, data = <data>)).",
               call. = FALSE)
        }
      } else if (sp_lc == "hsgp") {
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
    } else if (sp_lc %in% c(.NL_FRONTDOOR_AREAL, "rsr")) {
      # Areal field: spatial(col) names the per-observation unit. RSR is areal
      # too (it carries an adjacency), and gibbs-only.
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
      if (sp_lc == "rsr") {
        # RSR is routed only through the binomial Polya-Gamma Gibbs sampler.
        if (family != "binomial") {
          stop("RSR spatial fields are fit by the binomial Polya-Gamma Gibbs ",
               "sampler; `family` must be 'binomial' (got '", family, "').",
               call. = FALSE)
        }
        # Build the unit-level projector orthogonal to the restrict_to design --
        # the whole point of the modifier. Honour the spec's restrict_to formula
        # rather than the full model design; dispatch_gibbs_spatial() consumes
        # the precomputed n_units x n_units projection.
        if (!is.null(spatial_spec$rsr_formula)) {
          X_rsr <- stats::model.matrix(spatial_spec$rsr_formula, data = data)
          spatial_spec$rsr_projection <-
            .rsr_unit_projection(X_rsr, spatial_spec$spatial_idx, n_units)
        }
      }
    } else {
      stop("Spatial type '", spatial_type, "' is not yet routed through tulpa(). ",
           "Use an areal (icar/bym2/car/car_proper) or continuous ",
           "(gp/nngp/hsgp/spde) field.", call. = FALSE)
    }
  }

  # Temporal field. A temporal_rw1() spec carries its own time_var (like a
  # continuous spatial spec carries its coordinates), so no temporal(col) term is
  # used. An RW1 temporal effect is an intrinsic CAR on a 1D chain (a ring when
  # cyclic): its precision is exactly the chain ICAR precision, so it integrates
  # through the same nested-Laplace areal path as an ICAR field. RW2/AR1, panel
  # (grouped) temporal effects, and temporal combined with a spatial field are
  # not yet front-door wired; surface that rather than drop terms.
  temporal_spec <- temporal
  has_temporal  <- !is.null(parsed$temporal_var) || !is.null(temporal_spec)
  if (has_temporal) {
    if (!is.null(parsed$temporal_var)) {
      stop("A temporal field is addressed by the time_var in its spec ",
           "(e.g. temporal_rw1(\"", parsed$temporal_var, "\")); drop the temporal(",
           parsed$temporal_var, ") term and pass `temporal=`.", call. = FALSE)
    }
    if (!inherits(temporal_spec, "tulpa_temporal")) {
      stop("`temporal=` must be a temporal_rw1() / temporal_rw2() / temporal_ar1() ",
           "spec object.", call. = FALSE)
    }
    if (!identical(temporal_spec$type, "rw1")) {
      stop("Only temporal_rw1() is routed through tulpa() so far; for '",
           temporal_spec$type, "' call the temporal fitter directly.", call. = FALSE)
    }
    if (!is.null(temporal_spec$group_var)) {
      stop("Grouped (panel) temporal effects are not yet routed through tulpa(); ",
           "drop `group_var`.", call. = FALSE)
    }
    if (has_spatial) {
      stop("A temporal field together with a spatial field is not yet supported ",
           "through tulpa(); fit one field at a time for now.", call. = FALSE)
    }
    temporal_spec <- validate_temporal(temporal_spec, data)
  }

  fam_obj <- list(name = family, distribution = family)
  sel <- select_inference_mode(
    mode, family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_spatial, has_temporal = has_temporal, has_latent = has_latent,
    spatial_type = spatial_type, temporal = temporal_spec
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

  # SPDE carries its own nested-Laplace integration engine: fit_spde() rebuilds
  # the Matern precision Q(range, sigma) per node via the FEM Q-builder and
  # integrates (range, sigma) with a CCD / grid design in R -- not the generic
  # registry grid that tulpa_nested_laplace drives. Every nested mode (auto,
  # structured, or the nested_laplace backend by name) selects the generic
  # nested_laplace backend for a spatial field; redirect that to the dedicated
  # `spde` backend so the SPDE field reaches its own integrator. The conditional
  # mode = "laplace" stays on the fixed-hyperparameter tulpa_laplace path.
  if (sel$backend == "nested_laplace" &&
      identical(tolower(spatial_type %||% ""), "spde")) {
    sel$backend <- "spde"
    ti <- get_backend_tier("spde")
    sel$mode <- ti$mode; sel$tier <- ti$tier; sel$tier_name <- ti$name
    sel$reason <- "SPDE spatial field; nested-Laplace over (range, sigma) via fit_spde()"
  }

  # A temporal RW1 field is an ICAR chain; the nested-Laplace areal path
  # integrates it. Every selection routes a temporal field there (the conditional
  # mode = "laplace" is not wired for temporal yet), mirroring the spatial-field
  # redirect.
  if (has_temporal) {
    sel$backend <- "nested_laplace"
    ti <- get_backend_tier("nested_laplace")
    sel$mode <- ti$mode; sel$tier <- ti$tier; sel$tier_name <- ti$name
    sel$reason <- "temporal RW1 field; nested-Laplace integration (ICAR chain)"
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
                             spatial = spatial_spec, temporal = temporal_spec)

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

    # Canonical parameter layout for the S3 accessors: the fixed-effect count and
    # names plus the [fixed, random] name vector both posterior shapes share, so
    # coef()/summary()/ranef() report real names and a fixed/random split.
    layout <- .tulpa_param_layout(bundle)
    fit$n_fixed     <- layout$n_fixed
    fit$fixed_names <- layout$fixed_names
    fit$param_names <- fit$param_names %||% layout$param_names
    fit$re_layout   <- layout$re_layout
    fit$N           <- fit$N %||% bundle$n_obs
    # Fixed-effect design for fitted()/predict(newdata = NULL).
    fit$model_matrix <- bundle$X
  }
  fit
}
