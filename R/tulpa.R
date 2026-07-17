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
           "tulpa() validates it via validate_hsgp(); pass spatial_gp(~x+y, approx = 'hsgp').",
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


# Pack a validated tulpa_gp (nngp) spec into the ModelData sampler's GP
# spatial_spec (mode = "exact" continuous-field NUTS). The sampler's GP prior
# requires a PC range anchor, which spatial_gp() does not expose, so a
# weakly-informative data-driven default is derived here (P(range < U) = alpha,
# U = median nearest-neighbour spacing). Index conventions match the hmc_gp
# kernels: field at unique-location order, nn_order 0-based (validated ordering
# is 1-based), nn_neighbor_dist row-major [i, j1, j2].
#' @keywords internal
.gp_sampler_spec <- function(spatial) {
  ni <- spatial$neighbor_info
  if (is.null(ni) || is.null(spatial$unique_coords)) {
    stop("continuous spatial spec is unvalidated (neighbor_info / unique_coords ",
         "NULL). tulpa() validates it via validate_gp().", call. = FALSE)
  }
  uc    <- as.matrix(spatial$unique_coords)
  n_loc <- nrow(uc)
  nn    <- as.integer(spatial$nn %||% ncol(ni$nn_idx))
  pos_d <- ni$nn_dist[is.finite(ni$nn_dist) & ni$nn_dist > 0]
  U     <- if (length(pos_d)) stats::median(pos_d) else 0.1
  if (!is.finite(U) || U <= 0) U <- 0.1
  list(
    type             = "nngp",
    coords           = matrix(as.numeric(uc), n_loc, 2),
    nn               = nn,
    nn_idx           = matrix(as.integer(ni$nn_idx), n_loc, nn),
    nn_dist          = matrix(as.numeric(ni$nn_dist), n_loc, nn),
    nn_neighbor_dist = as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1))),
    nn_order         = as.integer(ni$nn_order) - 1L,
    nn_order_inv     = as.integer(ni$nn_order_inv %||% seq_len(n_loc)) - 1L,
    obs_to_loc       = as.integer(spatial$obs_to_loc) - 1L,
    cov_type         = gp_cov_type_for_laplace(spatial),
    nu               = as.numeric(spatial$nu %||% 1.5),
    phi_prior_U      = as.numeric(U),
    phi_prior_alpha  = 0.05,
    sigma2_prior_U   = 2.0,
    sigma2_prior_alpha = 0.05
  )
}


# Pack a validated tulpa_hsgp spec into the ModelData sampler's HSGP
# spatial_spec (mode = "exact" continuous-field NUTS). The Laplacian basis is
# built in C++ by setup_hsgp_2d (the single source of truth) from the validated
# per-observation coordinate matrix, so only (coords, m, c) cross the boundary;
# the field is evaluated per observation, no obs->location map. The PC prior on
# sigma and LogNormal on lengthscale are hardcoded in compute_hsgp_spatial_prior
# (no data-driven anchor needed, unlike the NNGP range prior).
#' @keywords internal
.hsgp_sampler_spec <- function(spatial) {
  cm <- spatial$coords_matrix
  if (is.null(cm)) {
    stop("HSGP spatial spec is unvalidated (coords_matrix NULL). tulpa() ",
         "validates it via validate_hsgp().", call. = FALSE)
  }
  cm <- as.matrix(cm)
  list(
    type   = "hsgp",
    coords = matrix(as.numeric(cm), nrow(cm), 2),
    m      = as.integer(spatial$m),
    c      = as.numeric(spatial$c)
  )
}


# Pack a proper-CAR spec into the ModelData sampler's car_proper spatial_spec
# (mode = "exact" areal NUTS). Q(rho) = D - rho W is full-rank, so the sampler
# estimates rho jointly with tau and the field. The generic log-prior needs a
# differentiable log|Q(rho)|; its parameter-dependent part is sum_i log(1 - rho
# mu_i) with mu_i the eigenvalues of the symmetric normalized adjacency
# D^{-1/2} W D^{-1/2}. These are fixed data (independent of rho / tau), so they
# are computed once here -- reusing the same eigen-decomposition compute_car_-
# rho_bounds() uses -- and the C++ prior only evaluates the closed-form sum.
#' @keywords internal
.car_proper_sampler_spec <- function(spatial) {
  sp <- .spatial_spec_to_nl_prior(spatial)
  adjm <- as.matrix(spatial$adjacency)
  diag(adjm) <- 0
  nnb <- rowSums(adjm != 0)
  if (any(nnb == 0)) {
    stop("Proper-CAR exact NUTS requires a connected adjacency (every unit has ",
         ">= 1 neighbour); Q = D - rho W is singular at an isolated unit. Drop ",
         "isolated units or use an ICAR / BYM2 field.", call. = FALSE)
  }
  dm <- 1 / sqrt(nnb)
  Wn  <- adjm * outer(dm, dm)               # D^{-1/2} W D^{-1/2}, symmetric
  eig <- eigen(Wn, symmetric = TRUE, only.values = TRUE)$values
  rb  <- spatial$rho_bounds %||% compute_car_rho_bounds(spatial$adjacency)
  list(
    type            = "car_proper",
    spatial_idx     = sp$spatial_idx,
    n_spatial_units = sp$n_spatial_units,
    adj_row_ptr     = sp$adj_row_ptr,
    adj_col_idx     = sp$adj_col_idx,
    n_neighbors     = sp$n_neighbors,
    adj_eigenvalues = as.numeric(eig),
    rho_lower       = as.numeric(rb["lower"]),
    rho_upper       = as.numeric(rb["upper"])
  )
}


# Pack a validated tulpa_svc (NNGP spatially-varying coefficients) spec into the
# ModelData sampler's svc_spec (mode = "exact" only). Each SVC term j carries an
# NNGP field w_j(s); the generic log-post adds eta_i += sum_j X_svc[i,j] w_j(s_i).
# X_svc is the design subset (the varying coefficients' columns) in row-major
# [n_obs x n_svc]. NNGP conventions match the GP field (coords row-major; nn_idx
# 1-based; nn_order 0-based); the SVC kernel derives neighbour-pair distances
# from coords, so no nn_neighbor_dist. The PC range anchor spatial_svc() does not
# expose is derived from the median nearest-neighbour spacing.
#' @keywords internal
.svc_sampler_spec <- function(spatial, X) {
  ni <- spatial$neighbor_info
  if (is.null(ni) || is.null(spatial$coords_matrix) || is.null(spatial$svc_indices)) {
    stop("SVC spec is unvalidated (neighbor_info / coords_matrix / svc_indices ",
         "NULL). tulpa() validates it via validate_svc().", call. = FALSE)
  }
  cm    <- as.matrix(spatial$coords_matrix)
  n_obs <- nrow(cm)
  nn    <- as.integer(spatial$nn %||% ncol(ni$nn_idx))
  idx   <- as.integer(spatial$svc_indices)
  Xs    <- as.matrix(X)[, idx, drop = FALSE]        # [n_obs x n_svc]
  pos_d <- ni$nn_dist[is.finite(ni$nn_dist) & ni$nn_dist > 0]
  U     <- if (length(pos_d)) stats::median(pos_d) else 0.1
  if (!is.finite(U) || U <= 0) U <- 0.1
  list(
    coords          = matrix(as.numeric(cm), n_obs, 2),
    n_svc           = length(idx),
    nn              = nn,
    nn_idx          = matrix(as.integer(ni$nn_idx), n_obs, nn),
    nn_dist         = matrix(as.numeric(ni$nn_dist), n_obs, nn),
    nn_order        = as.integer(ni$nn_order) - 1L,
    nn_order_inv    = as.integer(ni$nn_order_inv %||% seq_len(n_obs)) - 1L,
    svc_indices     = idx,
    X_svc           = as.numeric(t(Xs)),            # row-major [n_obs x n_svc]
    cov_type        = gp_cov_type_for_laplace(spatial),
    phi_prior_U     = as.numeric(U),
    phi_prior_alpha = 0.05
  )
}


# Pack a validated tulpa_tvc (RW1 / RW2 / AR1 temporally-varying coefficients)
# spec into the ModelData sampler's tvc_spec (mode = "exact" only). Each TVC term
# j carries a temporal field w_j(g, t); the generic log-post adds
# eta_i += sum_j X_tvc[i,j] w_j(g_i, t_i). X_tvc is row-major [n_obs x n_tvc].
#' @keywords internal
.tvc_sampler_spec <- function(temporal, X) {
  st <- tolower(temporal$structure %||% "rw1")
  if (!st %in% c("rw1", "rw2", "ar1")) {
    stop("TVC exact NUTS supports structure 'rw1' / 'rw2' / 'ar1'; got '", st,
         "'. The 'gp' temporal structure is not front-door wired.", call. = FALSE)
  }
  idx <- as.integer(temporal$tvc_indices)
  Xt  <- as.matrix(X)[, idx, drop = FALSE]          # [n_obs x n_tvc]
  n_groups <- as.integer(temporal$n_groups %||% 1L)
  list(
    n_times     = as.integer(temporal$n_times),
    n_tvc       = length(idx),
    n_groups    = n_groups,
    time_index  = as.integer(temporal$time_index),
    group_index = as.integer(temporal$group_index %||% rep(1L, nrow(as.matrix(X)))),
    tvc_indices = idx,
    X_tvc       = as.numeric(t(Xt)),                # row-major [n_obs x n_tvc]
    structure   = st,
    cyclic      = isTRUE(temporal$cyclic)
  )
}


# Convert a validated temporal spec (rw1 / rw2 / ar1) into the nested-Laplace
# temporal prior block. The block format is the one the single-block registry
# (R/nested_laplace.R: `rw1` / `rw2` / `ar1` entries) and the multi-block
# converter (.nl_block_spec_for_cpp) both consume: `type` selects the temporal
# Q-builder, the per-observation time index becomes `temporal_idx`, and the
# registry fills the tau (and AR1 rho) grids. The same block drives both the
# lone-field temporal kernel (cpp_nested_laplace_temporal) and the temporal half
# of a spatio-temporal joint prior (a LatentBlock stacked on the spatial block).
# RW1 penalises first differences (the intrinsic CAR on a 1D chain), RW2 second
# differences, and AR1 carries a free correlation -- all three share the tested
# temporal kernel rather than a per-type path.
.temporal_spec_to_nl_prior <- function(temporal) {
  n_times <- temporal$n_times
  if (is.null(n_times) || n_times < 2L) {
    stop("Internal: temporal spec is unvalidated (n_times missing). tulpa() ",
         "validates it via validate_temporal().", call. = FALSE)
  }
  type <- tolower(temporal$type %||% "")
  if (!type %in% c("rw1", "rw2", "ar1")) {
    stop("tulpa() routes temporal types rw1, rw2, ar1 through nested Laplace; ",
         "got '", type, "'.", call. = FALSE)
  }
  # Panel (grouped) data: a separate walk per group, all sharing one tau (and one
  # AR1 rho). validate_temporal() resolved n_groups + the per-obs group_index;
  # flatten (group, time) into the block's 1-based node (group-1)*n_times + time
  # so the G chains occupy contiguous, disconnected blocks. A single walk is the
  # n_groups == 1 case (the flattened index reduces to time_index).
  n_groups <- as.integer(temporal$n_groups %||% 1L)
  if (n_groups > 1L) {
    g_idx <- as.integer(temporal$group_index)
    temporal_idx <- (g_idx - 1L) * as.integer(n_times) + as.integer(temporal$time_index)
  } else {
    temporal_idx <- as.integer(temporal$time_index)
  }
  out <- list(
    type         = type,
    temporal_idx = temporal_idx,
    n_times      = as.integer(n_times),
    n_groups     = n_groups
  )
  if (type == "rw1") out$cyclic <- isTRUE(temporal$cyclic)
  out
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control,
                               latent_blocks = list(), spatial = NULL,
                               temporal = NULL, weights = NULL,
                               phi2 = NULL, smoothers = list(),
                               re_prior = NULL) {
  # Statistical random-effect / variance-component hyperpriors ride in a single
  # `re_prior` list (a statistical argument), never in `control` (tuning only).
  rp <- re_prior %||% list()
  input <- BACKEND_REGISTRY[[backend]]$input

  # Observation weights scale each row's log-likelihood. Supported where the
  # likelihood carries a per-obs multiplier today: the non-spatial Laplace
  # kernel and the R log-posterior builder. Everything else refuses loudly
  # rather than silently fitting unweighted.
  if (!is.null(weights) &&
      !(input == "logpost" ||
        (backend == "laplace" && is.null(spatial)))) {
    stop(sprintf(paste0(
      "`weights` is not supported by backend '%s'. Weighted fits run through ",
      "mode = 'laplace' (non-spatial) or a log-posterior sampler ",
      "('mala', 'imh_laplace', 'pathfinder')."), backend), call. = FALSE)
  }

  # Second dispersion (Student-t df): threaded through the non-spatial Laplace
  # kernel, the log-posterior builder, and the ModelData samplers.
  if (!is.null(phi2) &&
      !(input %in% c("logpost", "modeldata") ||
        (backend == "laplace" && is.null(spatial)))) {
    stop(sprintf(paste0(
      "`phi2` is not supported by backend '%s'. Use mode = 'laplace' ",
      "(non-spatial), a log-posterior sampler, or a ModelData sampler."),
      backend), call. = FALSE)
  }

  # tulpa()'s `phi` for gaussian / lognormal is the residual VARIANCE (the
  # R-side registry convention); the compiled kernels behind the nested /
  # spde / modeldata backends parameterize by the residual SD. Convert once
  # at this boundary (tulpa_laplace converts internally; the re_cov / agq
  # paths already consume the variance).
  phi_sd <- if (family %in% c("gaussian", "lognormal")) sqrt(phi) else phi

  if (input == "nested") {
    if (backend != "nested_laplace") {
      stop(sprintf(paste0(
        "Backend '%s' is a nested engine driven by model packages, not the\n",
        "single-response tulpa() formula -- it needs multiple response arms,\n",
        "which a single formula cannot express. Call %s() directly."),
        backend, backend), call. = FALSE)
    }
    # The nested driver integrates the hyperparameters of latent prior blocks
    # and/or field(s). A spatial(col) field becomes an areal / continuous prior
    # block (.spatial_spec_to_nl_prior); a temporal field becomes a temporal
    # block (.temporal_spec_to_nl_prior); latent(...) terms are blocks already.
    # The blocks stack into one prior: a single field alone routes the
    # single-block path, while a spatial + temporal field (additive space-time)
    # or any field with latent terms forms a multi-block prior the joint driver
    # integrates -- every obs touches each block, so they are Laplace-marginalised
    # jointly (the spatio-temporal cross term is assembled from each block's idx).
    field_blocks <- c(
      if (!is.null(spatial))  list(.spatial_spec_to_nl_prior(spatial))   else list(),
      if (!is.null(temporal)) list(.temporal_spec_to_nl_prior(temporal)) else list(),
      smoothers
    )
    all_blocks <- c(field_blocks, latent_blocks)
    if (length(all_blocks) == 0L) {
      stop("Backend 'nested_laplace' needs at least one `latent(...)` block, ",
           "s(...) smoother, spatial(col) field, or temporal field. For a ",
           "plain GLMM use mode = 'laplace' / 'mala' / 'auto'.", call. = FALSE)
    }
    if (!is.null(beta_prior)) {
      stop("`beta_prior` is not threaded through tulpa()'s nested-Laplace path. ",
           "Rather than silently drop it, this errors: call ",
           "tulpa_nested_laplace() directly if you need a fixed-effect prior on ",
           "this path.", call. = FALSE)
    }
    # A length-1 list routes the single-block path; length > 1 the multi-block
    # joint path (both are handled by tulpa_nested_laplace()).
    prior <- if (length(all_blocks) == 1L) all_blocks[[1L]] else all_blocks
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
      phi         = phi_sd,
      # Forward only the keys the inner fitter reads: front-door-only knobs
      # (grid shape, backend selection) were consumed above and would trip
      # tulpa_nested_laplace()'s own whitelist.
      control     = .control_subset(control, .CONTROL_KEYS$nested_laplace)
    ))
  }

  if (input == "spde") {
    # Continuous Matern SPDE field, nested-Laplace integrated over (range, sigma)
    # by fit_spde() (its own CCD / grid engine). fit_spde() takes the design
    # bundle (y, X, offset) plus the self-contained SPDE spec; it has no RE /
    # latent / beta_prior support, so reject those loudly rather than drop them.
    if (length(latent_blocks) > 0L) {
      stop("An SPDE spatial field cannot be combined with latent(...) blocks ",
           "through tulpa(); fit_spde() integrates a single Matern field. Drop ",
           "the latent block(s).", call. = FALSE)
    }
    # A single iid random-intercept `(1 | g)` can ride alongside the Matern
    # field (conditioned on sigma_re, jointly Laplace-marginalised in the
    # kernel). Random slopes / multiple terms are not supported on this path.
    re <- bundle$re_terms %||% list()
    spde_re_idx <- NULL; spde_re_n <- 0L; spde_sigma_re <- 1.0
    if (length(re) > 0L) {
      if (length(re) > 1L || (re[[1]]$n_coefs %||% 1L) != 1L ||
          !isTRUE(re[[1]]$has_intercept)) {
        stop("The SPDE path supports at most one random-intercept `(1 | g)` ",
             "term alongside the field; drop the extra / random-slope term(s), ",
             "or use mode = 'exact' for a sampler under the field.",
             call. = FALSE)
      }
      spde_re_idx   <- as.integer(re[[1]]$group_idx)
      spde_re_n     <- as.integer(re[[1]]$n_groups)
      spde_sigma_re <- sigma_re[1]
    }
    if (!is.null(beta_prior)) {
      stop("`beta_prior` is not supported on the SPDE path; fit_spde() uses a ",
           "built-in weak fixed-effect prior.", call. = FALSE)
    }
    spde_fams <- BACKEND_REGISTRY$spde$families
    if (!family %in% spde_fams) {
      stop(sprintf(
        "SPDE supports family %s; got '%s'.",
        paste0("'", spde_fams, "'", collapse = ", "), family), call. = FALSE)
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
      phi            = phi_sd,
      offset         = bundle$offset,
      re_idx         = spde_re_idx,
      n_re_groups    = spde_re_n,
      sigma_re       = spde_sigma_re,
      control        = list(
        method    = method,
        n_grid    = control$n_grid %||% 5L,
        max_iter  = control$max_iter %||% 100L,
        tol       = control$tol %||% 1e-6,
        n_threads = control$n_threads %||% 1L
      )
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
        # `control$re_cov = "aghq"` is the nested integrator with an AGHQ inner
        # marginal: n_quad defaults to 9 there, to the plain joint Laplace (1)
        # otherwise. An explicit control$n_quad always wins.
        re_cov_method <- match.arg(control$re_cov %||% "nested",
                                   c("nested", "gibbs", "aghq"))
        n_quad <- as.integer(control$n_quad %||%
                               (if (re_cov_method == "aghq") 9L else 1L))
        return(c(common, list(
          beta_prior  = beta_prior,
          prior_sigma = rp$prior_sigma %||% c(3, 0.05),
          eta         = rp$eta %||% 2,
          n_quad      = n_quad,
          control     = list(
            integration = control$integration %||% "ccd",
            n_per_axis  = control$n_per_axis %||% 5L,
            span        = control$span %||% 3,
            n_draws     = control$n_draws %||% 2000L,
            seed        = control$seed
          )
        )))
      }
      # re_cov_gibbs: exact Metropolis-within-Gibbs debias.
      return(c(common, list(
        prior_df        = rp$prior_df,
        prior_scale     = rp$prior_scale,
        beta_prior      = beta_prior %||% list(mean = 0, sd = 100),
        control         = list(
          n_iter = control$n_iter %||% 2000L,
          warmup = control$warmup %||% 1000L,
          seed   = control$seed
        )
      )))
    }
    if (backend == "laplace") {
      if (!is.null(spatial)) {
        # Spatial Laplace: route the field spec through tulpa_laplace(spatial=),
        # which dispatches on spatial$type (icar/car/bym2/spde/gp). At most one
        # random-intercept (1 | g) term may ride alongside the field -- the
        # spatial solvers consume a single RE block (re_list[[1]]); richer RE
        # structure is not supported here. An offset() is threaded into the
        # field's linear predictor; beta_prior is not threaded through the
        # spatial solvers, so reject it loudly rather than drop.
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
        return(list(
          y = bundle$y, n_trials = n_trials, X = bundle$X,
          re_list = .bundle_to_re_list(bundle, sigma_re),
          family = family, phi = phi, spatial = spatial,
          offset = bundle$offset
        ))
      }
      return(list(
        y = bundle$y, n_trials = n_trials, X = bundle$X,
        re_list = .bundle_to_re_list(bundle, sigma_re),
        family = family, phi = phi, phi2 = phi2,
        offset = bundle$offset, beta_prior = beta_prior,
        weights = weights
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
      gibbs_fams <- BACKEND_REGISTRY$gibbs$families
      if (!is.null(spatial)) {
        if (!family %in% gibbs_fams) {
          stop(sprintf(paste0(
            "Spatial Gibbs supports family %s; got ",
            "'%s'. Use mode = 'laplace' for other families under a spatial field."),
            paste0("'", gibbs_fams, "'", collapse = ", "), family),
            call. = FALSE)
        }
      } else if (!family %in% gibbs_fams) {
        stop(sprintf(paste0(
          "Gibbs (tulpa_gibbs) supports family %s; ",
          "got '%s'. Use mode = 'laplace' or a logpost backend."),
          paste0("'", gibbs_fams, "'", collapse = ", "), family),
          call. = FALSE)
      }
      # tulpa_gibbs enforces a mean-zero fixed-effect prior (the Polya-Gamma
      # sampler is built for it); a non-zero beta_prior$mean errors there.
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
        beta_prior = beta_prior %||% list(mean = 0, sd = 10),
        prior_sigma_scale = rp$prior_sigma_scale %||% 2.5,
        spatial = spatial,
        control = list(
          n_iter    = control$n_iter %||% 2000L,
          warmup    = control$warmup %||% 1000L,
          thin      = control$thin %||% 1L,
          seed      = control$seed,
          n_threads = control$n_threads %||% 1L
        )
      ))
    }
    if (backend == "agq") {
      # Adaptive Gauss-Hermite quadrature: one intercept-only random-effect term,
      # families binomial / poisson / gaussian (the lme4::glmer(nAGQ=) scope).
      # n_quad = 1 is the joint Laplace; higher quadrature reduces the
      # small-cluster variance attenuation. agq_fit() optimizes the marginal
      # likelihood and estimates the RE sd, so no sigma_re is conditioned on.
      re <- bundle$re_terms %||% list()
      if (length(re) != 1L || (re[[1]]$n_coefs %||% 1L) != 1L ||
          !isTRUE(re[[1]]$has_intercept)) {
        stop("AGQ (mode = 'agq') supports exactly one random-intercept term ",
             "(a single `(1 | g)`). For random slopes or multiple terms use ",
             "mode = 'laplace' (RE-covariance integration), or call agq_fit() ",
             "directly.", call. = FALSE)
      }
      agq_fams <- BACKEND_REGISTRY$agq$families
      if (!family %in% agq_fams) {
        stop(sprintf(paste0(
          "AGQ supports family %s; got '%s'. ",
          "Use mode = 'laplace' or a sampler for other families."),
          paste0("'", agq_fams, "'", collapse = ", "), family),
          call. = FALSE)
      }
      if (!is.null(beta_prior)) {
        stop("`beta_prior` is not supported on the AGQ path; agq_fit() is a ",
             "marginal-likelihood fit with no fixed-effect prior. Drop ",
             "`beta_prior`, or use mode = 'laplace' for a Gaussian prior.",
             call. = FALSE)
      }
      # phi is the residual variance for gaussian; sigma_eps is its sd (binomial
      # / poisson ignore it). control$sigma_eps overrides.
      return(list(
        y          = bundle$y,
        X          = bundle$X,
        group      = as.integer(re[[1]]$group_idx),
        n_groups   = re[[1]]$n_groups,
        family     = family,
        n_trials   = n_trials,
        sigma_eps  = control$sigma_eps %||% (if (!is.null(phi)) sqrt(phi) else 1.0),
        n_quad     = control$n_quad %||% 7L,
        beta_init  = control$beta_init,
        sigma_init = control$sigma_init %||% 1.0,
        max_iter   = control$max_iter %||% 200L,
        tol        = control$tol %||% 1e-6,
        verbose    = control$verbose %||% FALSE
      ))
    }
    if (backend == "ep") {
      # Expectation Propagation: a fixed-effect GLM with a mean-zero Gaussian
      # coefficient prior. No random effects, spatial field, latent block, or
      # temporal structure -- EP places one Gaussian site per observation on the
      # scalar linear predictor and has no latent-block machinery.
      re <- bundle$re_terms %||% list()
      if (length(re) > 0L || !is.null(spatial) || length(latent_blocks) > 0L ||
          !is.null(temporal)) {
        stop("EP (mode = 'ep') fits a fixed-effect GLM only: drop the ",
             "random-effect / spatial / temporal / latent(...) term(s), or use ",
             "mode = 'laplace' / 'mala' / 'auto'.", call. = FALSE)
      }
      # EP consumes the residual VARIANCE directly for gaussian (its tilted
      # moments use phi = variance), matching tulpa()'s phi convention -- so pass
      # phi, not the sd-converted phi_sd.
      return(list(
        y          = bundle$y,
        X          = bundle$X,
        family     = family,
        phi        = phi %||% 1.0,
        n_trials   = n_trials,
        beta_prior = beta_prior %||% list(mean = 0, sd = 10),
        control    = .control_subset(control, .CONTROL_KEYS$ep)
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
                            beta_prior = beta_prior %||% list(mean = 0, sd = 2.5),
                            weights = weights, phi2 = phi2)
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

  if (input == "modeldata") {
    # The model-agnostic ModelData sampler kernels (hmc/ess/sghmc/sgld/mclmc/
    # smc/vi) sample the full latent vector compute_param_layout() lays out
    # (gcol33/tulpa#75). Random effects (intercept / slopes / correlated /
    # multi-term), an areal spatial field (ICAR / BYM2), and a temporal field
    # (RW1 / RW2 / AR1) are packed into per-block specs and threaded into the
    # ModelData builder; the kernels sample the variance-component
    # hyperparameters jointly with the latent + fixed effects (full Bayes), not
    # conditioning on them like the Laplace / logpost backends.
    n_obs <- bundle$n_obs

    # Random-effect spec: one entry per term, mirroring the re_cov packing.
    re <- bundle$re_terms %||% list()
    re_spec <- NULL
    if (length(re) > 0L) {
      re_spec <- list(
        idx        = lapply(re, function(rt) as.integer(rt$group_idx)),
        ngroups    = vapply(re, function(rt) as.integer(rt$n_groups), integer(1)),
        ncoefs     = vapply(re, function(rt) as.integer(rt$n_coefs %||% 1L), integer(1)),
        correlated = vapply(re, function(rt) isTRUE(rt$correlated), logical(1)),
        Z          = lapply(re, function(rt) {
          Z <- cbind(
            if (isTRUE(rt$has_intercept)) rep(1, n_obs) else NULL,
            rt$slope_matrix
          )
          if (is.null(Z)) NULL else as.matrix(Z)
        })
      )
    }

    # Areal spatial spec (ICAR / BYM2). Reuses the nested-Laplace areal converter
    # for the adjacency CSR + per-obs unit index; continuous (gp/nngp/hsgp),
    # CAR_proper, and SPDE fields are not threaded through this path (the generic
    # ESS Gaussian-prior block / the dedicated SPDE sampler own those).
    # Spatially-varying coefficients ride the spatial= slot as their own sampler
    # input (svc_spec), not the areal/field spatial_spec.
    svc_spec_arg <- NULL
    spatial_spec_arg <- NULL
    if (!is.null(spatial) && tolower(spatial$type %||% "") == "svc") {
      svc_spec_arg <- .svc_sampler_spec(spatial, bundle$X)
    } else if (!is.null(spatial) && tolower(spatial$type %||% "") %in% c("gp", "nngp")) {
      spatial_spec_arg <- .gp_sampler_spec(spatial)
    } else if (!is.null(spatial) && tolower(spatial$type %||% "") == "hsgp") {
      spatial_spec_arg <- .hsgp_sampler_spec(spatial)
    } else if (!is.null(spatial) && tolower(spatial$type %||% "") == "car_proper") {
      spatial_spec_arg <- .car_proper_sampler_spec(spatial)
    } else if (!is.null(spatial)) {
      sp <- .spatial_spec_to_nl_prior(spatial)
      if (!sp$type %in% c("icar", "bym2")) {
        stop(sprintf(paste0(
          "Backend '%s' samples areal (icar / bym2 / car_proper) or continuous\n",
          "GP / NNGP / HSGP spatial fields; the field type '%s' is not threaded\n",
          "through this path. Use a nested-Laplace mode ('auto' / 'structured' /\n",
          "'nested_laplace'), or fit_spde() for SPDE."),
          backend, sp$type), call. = FALSE)
      }
      spatial_spec_arg <- list(
        type            = sp$type,
        spatial_idx     = sp$spatial_idx,
        n_spatial_units = sp$n_spatial_units,
        adj_row_ptr     = sp$adj_row_ptr,
        adj_col_idx     = sp$adj_col_idx,
        n_neighbors     = sp$n_neighbors,
        scale_factor    = sp$scale_factor %||% 1.0
      )
    }

    # Temporal spec (RW1 / RW2 / AR1). The generic eta assembler recombines the
    # within-group time index and the group index itself, so pass them
    # unflattened (not the nested block's combined node index).
    temporal_spec_arg <- NULL
    tvc_spec_arg <- NULL
    if (!is.null(temporal) && tolower(temporal$type %||% "") == "tvc") {
      # Temporally-varying coefficients ride the temporal= slot as their own
      # sampler input (tvc_spec), not the shared-field temporal_spec.
      tvc_spec_arg <- .tvc_sampler_spec(temporal, bundle$X)
    } else if (!is.null(temporal)) {
      ttype <- tolower(temporal$type %||% "")
      if (!ttype %in% c("rw1", "rw2", "ar1")) {
        stop(sprintf(paste0(
          "Backend '%s' samples temporal fields rw1 / rw2 / ar1; got '%s'."),
          backend, ttype), call. = FALSE)
      }
      n_groups <- as.integer(temporal$n_groups %||% 1L)
      temporal_spec_arg <- list(
        type      = ttype,
        time_idx  = as.integer(temporal$time_index),
        n_times   = as.integer(temporal$n_times),
        n_groups  = n_groups,
        group_idx = if (n_groups > 1L) as.integer(temporal$group_index) else NULL,
        cyclic    = isTRUE(temporal$cyclic)
      )
    }

    # ESS samples Gaussian-prior latent blocks with an isotropic proposal, which
    # cannot carry the structured spatial / temporal precision; the gradient /
    # density kernels can. Redirect rather than fail deep in C++.
    if (backend == "ess" &&
        (!is.null(spatial_spec_arg) || !is.null(temporal_spec_arg) ||
         !is.null(svc_spec_arg) || !is.null(tvc_spec_arg))) {
      stop(paste0(
        "Backend 'ess' samples latent Gaussian blocks with an isotropic prior\n",
        "and cannot carry the structured spatial / temporal precision. Use\n",
        "mode = 'hmc' / 'mclmc' / 'smc' / 'vi' for a sampler under a field, or a\n",
        "nested-Laplace mode for an integrated field."), call. = FALSE)
    }

    return(list(
      y             = bundle$y,
      n_trials      = n_trials %||% rep(1L, n_obs),
      X             = bundle$X,
      family        = family,
      backend       = backend,
      phi           = phi_sd,
      phi2          = phi2,
      offset        = bundle$offset,
      fixed_names   = bundle$fixed_names,
      re_spec       = re_spec,
      spatial_spec  = spatial_spec_arg,
      temporal_spec = temporal_spec_arg,
      svc_spec      = svc_spec_arg,
      tvc_spec      = tvc_spec_arg,
      sigma_re_scale = rp$sigma_re_scale %||% 2.5,
      # The fixed-effect prior SD is the statistical `beta_prior` (mean-zero on
      # this sampler path), not a control knob; inject it into the sampler's
      # sigma_beta. Other perf knobs forward from control.
      sigma_beta    = if (!is.null(beta_prior))
                        .beta_prior_ridge_sd(beta_prior, 10) else 10.0,
      control       = .control_subset(control, .CONTROL_KEYS$sample_glmm)
    ))
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
#'   `"pathfinder"`, `"imh_laplace"`, and the ModelData kernels `"hmc"` / `"sghmc"`
#'   / `"sgld"` / `"mclmc"` / `"smc"` / `"vi"` / `"ess"`).
#' * **Random slopes** are supported on the Laplace (Tier 2) path: there is no
#'   scalar `sigma_re` to condition on, so the RE covariance `Sigma` is integrated
#'   rather than fixed. This covers correlated terms (`(1 + x | g)`, a full
#'   `Sigma`), uncorrelated terms (`(1 + x || g)`, a diagonal `Sigma`), and
#'   several terms together (`(1 + x | g) + (1 | h)`) -- each term becomes a
#'   covariance block, and any accompanying `(1 | g)` term is integrated as a 1x1
#'   block (nothing is silently conditioned at `sigma_re = 1`). `mode = "laplace"`
#'   routes to the nested-Laplace `Sigma` integrator ([tulpa_re_cov_nested()],
#'   CCD design + PC/LKJ prior); `control$re_cov = "gibbs"` switches to the exact
#'   Metropolis-within-Gibbs debias ([tulpa_re_cov_gibbs()]), and
#'   `control$re_cov = "aghq"` keeps the nested integrator but replaces the
#'   inner joint-Laplace marginal with adaptive Gauss-Hermite quadrature
#'   (`control$n_quad`, default 9 there; see `n_quad` in
#'   [tulpa_re_cov_nested()]). Both also run on the
#'   sampler path (`mode = "mala"` / `"pathfinder"`).
#' * `mode = "gibbs"` (Polya-Gamma) fits a single random-intercept model for
#'   `family = "binomial"` or `"neg_binomial_2"`, and **samples** the RE sd
#'   rather than conditioning on `sigma_re`; tune it via
#'   `re_prior$prior_sigma_scale` and a mean-zero `beta_prior`.
#' * **Latent prior blocks** (`latent(tgmrf(...))`) route to the nested-Laplace
#'   path (Tier 2), which integrates over the block hyperparameters. `mode =
#'   "auto"` and `"structured"` select it automatically when latent blocks are
#'   present; `mode = "nested_laplace"` forces it. At most one random-intercept
#'   `(1 | g)` term may accompany the blocks (model richer grouping as an `iid`
#'   block). Joint multi-arm nested models cannot be expressed by a
#'   single-response formula -- call [tulpa_nested_laplace_joint()] directly.
#' * The ModelData sampler kernels (`"hmc"`, `"ess"`, `"sghmc"`, `"sgld"`,
#'   `"mclmc"`, `"smc"`, `"vi"`) thread the full latent vector -- fixed effects,
#'   random effects (all forms), areal spatial (`icar` / `bym2`), and temporal
#'   (`rw1` / `rw2` / `ar1`) -- through one ModelData builder and sample the
#'   variance components jointly with the field. `ess` carries random effects but
#'   declines a structured spatial / temporal block (its isotropic Gaussian-prior
#'   block cannot represent the graph precision); continuous-coordinate fields
#'   (`gp` / `nngp` / `hsgp` / `spde`), `car_proper`, and exotic latent blocks stay
#'   on the dedicated nested-Laplace / SPDE / Polya-Gamma paths.
#'
#' @param formula A model formula. Fixed effects, `(1 | g)` / `(1 + x | g)`
#'   random effects, and `offset(...)` terms are recognised.
#' @param data A data frame.
#' @param family Character family name: one of [family_names()]
#'   (`"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`, `"beta"`,
#'   ...), or a categorical response family -- `"multinomial"`
#'   (baseline-category logit via [tulpa_multinomial()]), `"ordinal"`
#'   (cumulative logit via [tulpa_ordinal()]), or `"ordinal_probit"`
#'   (cumulative probit). Categorical families take fixed-effect models only.
#' @param mode Inference mode or backend. `"auto"` (default) picks the most
#'   reliable Tier 1/Tier 2 method expected to finish; a tier (`"exact"`,
#'   `"structured"`) or a backend name (`"laplace"`, `"mala"`, ...) forces it.
#' @param sigma_re Random-effect SDs to condition on: length 1 (recycled) or one
#'   per RE term. Defaults to 1 per term with a message.
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL`.
#' @param weights Optional observation weights (non-negative numeric vector,
#'   length `nrow(data)`): each observation's log-likelihood contribution is
#'   scaled by its weight (prior / frequency weights, e.g. survey weights or
#'   aggregated-data counts -- a weight of 2 is equivalent to duplicating the
#'   row). Supported on the non-spatial Laplace path (`mode = "laplace"`) and
#'   the log-posterior samplers (`mala`, `imh_laplace`, `pathfinder`); other
#'   backends reject weights loudly.
#' @param phi Dispersion/precision passed to the family (residual variance for
#'   gaussian and lognormal, size for neg_binomial_2, precision for beta,
#'   scale for t). The variance convention holds across every backend; the
#'   SD-parameterized compiled kernels receive `sqrt(phi)` at the boundary.
#' @param phi2 Optional second dispersion: the Student-t degrees of freedom
#'   (`family = "t"`; default 4 when `NULL`). Supported on the non-spatial
#'   Laplace path, the log-posterior samplers, and the ModelData samplers.
#' @param beta_prior Optional `list(mean, sd)` Gaussian prior on the fixed
#'   effects.
#' @param re_prior Optional `list()` of random-effect / variance-component
#'   hyperpriors (statistical, so they live in the signature rather than in
#'   `control`). Recognised entries, each consumed by the backend that needs it:
#'   `prior_sigma` (PC-prior anchor `c(U, alpha)` on a free RE covariance SD,
#'   `mode = "laplace"` random slopes), `eta` (LKJ concentration for a
#'   correlated RE covariance), `prior_df` / `prior_scale` (inverse-Wishart on
#'   the RE covariance, `control$re_cov = "gibbs"`), `prior_sigma_scale`
#'   (half-Cauchy scale on the RE SD for `mode = "gibbs"`), and `sigma_re_scale`
#'   (half-Cauchy scale on the RE / BYM2 SD for the ModelData samplers).
#' @param spatial Optional spatial-field spec. How it is addressed depends on the
#'   field family:
#'   * **Areal** (`"icar"`, `"car"`, `"bym2"`, `"car_proper"`): a list with `type`
#'     and `adjacency`, paired with a `spatial(col)` term in `formula` naming the
#'     per-observation unit column. Term and spec must be supplied together.
#'   * **Continuous** (`spatial_gp(~ lon + lat)` for an NNGP field,
#'     `spatial_gp(~ lon + lat, approx = 'hsgp')` for a Hilbert-space GP, `spatial_spde(~ lon +
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
#' @param temporal Optional temporal field spec ([temporal_rw1()],
#'   [temporal_rw2()], or [temporal_ar1()]), integrated by nested Laplace. A
#'   plain field routes the single-block temporal kernel; a `group_var` panel
#'   spec fits a separate walk per group sharing one hyperparameter; combined
#'   with an areal `spatial` field it forms an additive space-time joint prior.
#' @param control Optional list of backend tuning arguments (e.g. `n_iter`,
#'   `warmup`, `epsilon` for `mala`; `n_draws` for `pathfinder`).
#' @param ... Reserved for future statistical arguments. Nothing is read from
#'   it today, so any entry errors: a stray name here is a misspelled argument
#'   or a tuning knob that belongs in `control`.
#'
#' @return A `tulpa_fit` object carrying the backend's output plus
#'   `inference_mode`, `inference_tier`, `backend`, `selection_reason`,
#'   `formula`, and `family`. Two field-name conventions to know when
#'   reaching into the object directly (the generic accessors handle both):
#'   on nested-Laplace fits `$weights` is the hyperparameter GRID weights;
#'   user observation weights are stored as `$obs_weights`. `$draws` is a
#'   draws matrix on engine fits, while model-package fits may carry a list
#'   (`$y_rep`, `$log_lik`) under the same name.
#'
#' @seealso [inference_mode_info()], [tulpa_laplace()], [mala()], [pathfinder()]
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 200L
#' g <- sample(letters[1:12], n, replace = TRUE)
#' d <- data.frame(
#'   y = rbinom(n, 1, plogis(-0.3 + 0.6 * rnorm(n))),
#'   x = rnorm(n),
#'   g = g
#' )
#' # Random-intercept logistic GLMM, Laplace tier.
#' fit <- tulpa(y ~ x + (1 | g), data = d, family = "binomial", mode = "laplace")
#' coef(fit)
#' summary(fit)
#' }
#' @references
#' Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
#' Gaussian models by using integrated nested Laplace approximations.
#' \emph{JRSS-B} 71(2):319-392.
#' Hoffman & Gelman (2014). The No-U-Turn Sampler: adaptively setting path
#' lengths in Hamiltonian Monte Carlo. \emph{JMLR} 15(47):1593-1623.
#' @export
tulpa <- function(formula, data,
                  family = "gaussian",
                  mode = "auto",
                  sigma_re = NULL,
                  n_trials = NULL,
                  weights = NULL,
                  phi = 1.0,
                  phi2 = NULL,
                  beta_prior = NULL,
                  re_prior = NULL,
                  spatial = NULL,
                  temporal = NULL,
                  control = list(),
                  ...) {
  # `...` exists so future statistical arguments can be added without a
  # signature break; nothing is read from it today, so a stray entry is a
  # misspelled or misplaced argument (e.g. `familly =`, or a tuning knob that
  # belongs in `control = list()`), not a silent no-op.
  dots <- list(...)
  if (length(dots)) {
    nm <- names(dots) %||% rep("", length(dots))
    nm[!nzchar(nm)] <- "<unnamed>"
    stop(sprintf(
      "unknown argument(s) to tulpa(): %s. Tuning knobs go in `control = list()`.",
      paste(nm, collapse = ", ")), call. = FALSE)
  }
  .check_control(control, .CONTROL_KEYS$tulpa, "tulpa")
  .check_control(re_prior, .RE_PRIOR_KEYS, "tulpa (re_prior)")

  # Categorical responses are families, not separate verbs: the front door
  # routes them to the multinomial / cumulative-link Laplace drivers. The link
  # rides the family string ("ordinal_probit"), matching the engine's
  # family_<link> convention. Fixed-effect models only for now -- latent
  # structure under a categorical response is tracked engine work (C6).
  if (family %in% c("multinomial", "ordinal", "ordinal_probit")) {
    pf <- tulpa_parse_formula(formula)
    if (pf$n_re_terms > 0L || pf$n_latent_blocks > 0L ||
        (pf$n_smooth_terms %||% 0L) > 0L ||
        (pf$n_spatial_field_blocks %||% 0L) > 0L ||
        (pf$n_temporal_field_blocks %||% 0L) > 0L ||
        !is.null(pf$spatial_var) || !is.null(pf$temporal_var) ||
        !is.null(spatial) || !is.null(temporal) ||
        !is.null(sigma_re) || !is.null(n_trials) || !is.null(weights) ||
        !is.null(phi2)) {
      stop(sprintf(paste0(
        "family = '%s' supports fixed-effect models only through tulpa(); ",
        "random effects, smoothers, and spatial / temporal structure are not ",
        "wired for categorical responses yet."), family), call. = FALSE)
    }
    if (!mode %in% c("auto", "structured", "laplace")) {
      stop(sprintf(paste0(
        "family = '%s' is fit by its Laplace driver; mode = '%s' is not ",
        "available. Use mode = 'auto'."), family, mode), call. = FALSE)
    }
    fit <- if (family == "multinomial") {
      tulpa_multinomial(formula, data,
                        beta_prior = beta_prior %||% list(mean = 0, sd = 10),
                        control = .control_subset(control,
                                                  .CONTROL_KEYS$multinomial))
    } else {
      tulpa_ordinal(formula, data,
                    link = if (family == "ordinal_probit") "probit" else "logit",
                    beta_prior = beta_prior %||% list(mean = 0, sd = 10),
                    control = .control_subset(control, .CONTROL_KEYS$ordinal))
    }
    fit$call <- match.call()
    return(fit)
  }

  .family_or_stop(family)
  .validate_family_phi(family, phi)
  if (!is.null(phi2)) .phi2_or_stop(family, phi2)
  # Tweedie requires the power up front (and in (1, 2)); fail before fitting.
  if (identical(family, "tweedie")) .tweedie_power(phi2)

  parsed <- tulpa_parse_formula(formula)
  bundle <- tulpa_build_model_data(parsed, data)
  .validate_family_counts(family, bundle$y)
  # The model is built with na.action = na.pass (prior_predict() allows an NA
  # response), so tulpa() must reject non-finite fitting inputs itself: unlike
  # glm()/lm() it does not drop incomplete cases, and an NA/NaN/Inf would flow
  # silently into the C++ kernels as a NaN estimate.
  .assert_finite_model_inputs(bundle$X, bundle$y)
  if (!is.null(weights)) {
    weights <- as.numeric(weights)
    if (length(weights) != bundle$n_obs || anyNA(weights) ||
        any(!is.finite(weights)) || any(weights < 0)) {
      stop("`weights` must be a non-negative finite numeric vector of length ",
           "nrow(data) (", bundle$n_obs, ").", call. = FALSE)
    }
  }
  K <- length(bundle$re_terms %||% list())

  has_latent <- (parsed$n_latent_blocks %||% 0L) > 0L

  # Inline areal varying-coefficient field(s): spatial(graph = , formula =
  # ~ ... || cell). Each bar term expands to independent CAR blocks (one per
  # design column, slope columns carrying a per-row weight) and is fit through
  # the single-arm joint nested-Laplace path, which threads that weight.
  if ((parsed$n_spatial_field_blocks %||% 0L) > 0L) {
    if (!is.null(weights)) {
      stop("`weights` is not supported on the inline spatial-field path.",
           call. = FALSE)
    }
    return(.tulpa_fit_spatial_field(parsed, bundle, data, family, mode, phi,
                                    sigma_re, n_trials, control, formula,
                                    sys.call()))
  }

  # Inline temporal varying-coefficient field(s): temporal(formula = ~ ... ||
  # time, structure = ). The temporal mirror of the spatial-field path -- each
  # bar term expands to independent rw1 / rw2 / ar1 blocks (slope columns
  # carrying a per-row weight) and is fit through the same single-arm joint
  # nested-Laplace path.
  if ((parsed$n_temporal_field_blocks %||% 0L) > 0L) {
    if (!is.null(weights)) {
      stop("`weights` is not supported on the inline temporal-field path.",
           call. = FALSE)
    }
    return(.tulpa_fit_temporal_field(parsed, bundle, data, family, mode, phi,
                                     sigma_re, n_trials, control, formula,
                                     sys.call()))
  }

  # Spatial field. The structure arrives via the `spatial=` argument; how it is
  # addressed depends on the field family:
  #  * Areal (icar/car/bym2/car_proper): a `spatial(col)` term names the
  #    per-observation unit column, resolved to a 1-based `spatial_idx` against
  #    the adjacency. Term and spec must appear together.
  #  * Continuous gp/nngp/hsgp: a spatial_gp(~lon+lat) / spatial_gp(~lon+lat, approx="hsgp")
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
    if (sp_lc == "svc") {
      # Spatially-varying coefficients: coordinate-addressed (coords from the
      # spec, no spatial(col) term) and design-dependent (the varying columns
      # are resolved against the model matrix). Exact-NUTS only -- there is no
      # nested-Laplace SVC front door -- so validate_svc() needs the built X.
      if (!is.null(parsed$spatial_var)) {
        stop("A spatially-varying-coefficient field is addressed by its ",
             "coordinate columns in the spec; drop the spatial(",
             parsed$spatial_var, ") term.", call. = FALSE)
      }
      if (!inherits(spatial_spec, "tulpa_svc")) {
        stop("A spatially-varying-coefficient field must be a ",
             "spatial_svc(~ lon + lat, terms = ...) spec object.", call. = FALSE)
      }
      if (identical(tolower(spatial_spec$approx %||% "nngp"), "hsgp")) {
        stop("HSGP-approximated SVC is not front-door wired yet; use ",
             "spatial_svc(~ lon + lat, approx = 'nngp').", call. = FALSE)
      }
      spatial_spec <- validate_svc(spatial_spec, data, bundle$X)
    } else if (sp_lc %in% c(.NL_FRONTDOOR_CONTINUOUS, .NL_FRONTDOOR_SPDE)) {
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
          stop("An HSGP spatial field must be a spatial_gp(~ lon + lat, approx = 'hsgp') spec ",
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
      stop("Unknown spatial type '", spatial_type, "'. `spatial$type` must be one ",
           "of: areal icar/car/bym2/car_proper, continuous gp/nngp/hsgp/spde, or ",
           "rsr.", call. = FALSE)
    }
  }

  # Temporal field. A temporal_rw1() / temporal_rw2() / temporal_ar1() spec
  # carries its own time_var (like a continuous spatial spec carries its
  # coordinates), so no temporal(col) term is used. All three integrate through
  # the nested-Laplace temporal kernel: RW1 penalises first differences (the
  # intrinsic CAR on a 1D chain, a ring when cyclic), RW2 second differences, and
  # AR1 carries a free correlation. A temporal field alongside a nested-wired
  # spatial field (areal or gp/nngp/hsgp) forms an additive space-time prior the
  # joint driver integrates as a [spatial, temporal] block stack. SPDE / RSR
  # spatial fields run their own single-field integrators and cannot host a
  # temporal block through the front door yet; surface that rather than drop
  # terms.
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
           "or temporal_tvc() spec object.", call. = FALSE)
    }
    if (identical(tolower(temporal_spec$type %||% ""), "tvc")) {
      # Temporally-varying coefficients: design-dependent (the varying columns
      # resolve against the model matrix) and exact-NUTS only. Validate against
      # the built X; combine with fixed effects only for now.
      if (has_spatial || has_latent) {
        stop("A temporally-varying-coefficient field cannot be combined with a ",
             "spatial or latent(...) field through tulpa() yet. Fit the TVC field ",
             "on its own.", call. = FALSE)
      }
      temporal_spec <- validate_tvc(temporal_spec, data, bundle$X)
    } else {
    if (!tolower(temporal_spec$type %||% "") %in% c("rw1", "rw2", "ar1")) {
      stop("tulpa() routes temporal_rw1() / temporal_rw2() / temporal_ar1(); for '",
           temporal_spec$type, "' call the temporal fitter directly.", call. = FALSE)
    }
    # Panel (grouped) temporal: a separate walk per group sharing one tau, routed
    # as a single grouped temporal block through cpp_nested_laplace_temporal. The
    # multi-block joint path does not carry the per-group temporal layout yet, so
    # panel must be the only latent structure -- reject it alongside a spatial or
    # latent block rather than silently fit one chain over the flattened nodes.
    if (!is.null(temporal_spec$group_var) && (has_spatial || has_latent)) {
      stop("A grouped (panel) temporal field cannot be combined with a spatial ",
           "or latent(...) field through tulpa() yet (the joint path has no ",
           "per-group temporal layout). Fit the panel temporal field on its own.",
           call. = FALSE)
    }
    if (has_spatial && !tolower(spatial_type %||% "") %in% .NL_FRONTDOOR_AREAL) {
      stop("A temporal field can accompany an areal (icar/car/bym2/car_proper) ",
           "spatial field through the joint nested-Laplace path; the '",
           spatial_type, "' field is fit by its own integrator (continuous ",
           "gp/nngp/hsgp and SPDE space-time kernels exist but are not front-door ",
           "wired yet; RSR is sampler-only) and cannot host a temporal block ",
           "through tulpa() yet. Fit one field at a time, or use an areal field ",
           "for space-time.", call. = FALSE)
    }
    temporal_spec <- validate_temporal(temporal_spec, data)
    }
  }

  # Covariate smoothers s(x): RW1/RW2 GMRF blocks over the binned covariate --
  # the temporal-field construction with bins as nodes (R/smoother.R), riding
  # the same nested-Laplace kernels (single block alone, the joint stack
  # alongside an areal spatial or temporal field).
  smooth_specs <- list()
  if ((parsed$n_smooth_terms %||% 0L) > 0L) {
    if (!is.null(temporal_spec) && !is.null(temporal_spec$group_var)) {
      stop("A grouped (panel) temporal field cannot be combined with s(...) ",
           "smoothers through tulpa() yet.", call. = FALSE)
    }
    if (has_spatial && !tolower(spatial_type %||% "") %in% .NL_FRONTDOOR_AREAL) {
      stop("s(...) smoothers can accompany an areal (icar/car/bym2/car_proper) ",
           "spatial field through the joint nested-Laplace path; the '",
           spatial_type, "' field is fit by its own integrator and cannot ",
           "host smoother blocks through tulpa() yet.", call. = FALSE)
    }
    smooth_specs <- lapply(parsed$smooth_calls, .smooth_block_from_call,
                           data = data,
                           env = environment(formula) %||% parent.frame())
  }
  has_smooth <- length(smooth_specs) > 0L

  fam_obj <- list(name = family, distribution = family)
  sel <- select_inference_mode(
    mode, family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_spatial, has_temporal = has_temporal, has_latent = has_latent,
    spatial_type = spatial_type, temporal = temporal_spec
  )

  # Spatially- / temporally-varying coefficients are sampled only by the
  # generic ModelData sampler (there is no nested-Laplace SVC/TVC front door),
  # so they require a Tier-1 exact mode. A nested / Laplace mode would otherwise
  # silently drop the varying-coefficient field -- fail loudly with guidance.
  is_svc_fit <- identical(tolower(spatial_type %||% ""), "svc")
  is_tvc_fit <- !is.null(temporal_spec) &&
    identical(tolower(temporal_spec$type %||% ""), "tvc")
  if ((is_svc_fit || is_tvc_fit) &&
      (BACKEND_REGISTRY[[sel$backend]]$input %||% "") != "modeldata") {
    stop(sprintf(paste0(
      "%s coefficients are sampled by the exact ModelData NUTS backend; the\n",
      "selected backend '%s' (mode = '%s') does not carry the varying-coefficient\n",
      "field. Use mode = 'exact'."),
      if (is_svc_fit) "Spatially-varying" else "Temporally-varying",
      sel$backend, mode), call. = FALSE)
  }

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
  # Metropolis-within-Gibbs debias; `control$re_cov = "aghq"` keeps the nested
  # integrator with an AGHQ inner marginal (n_quad defaults to 9 there). Plain
  # random-intercept-only models (no slopes) keep the scalar-sigma_re design
  # path via .bundle_to_re_list.
  re_terms <- bundle$re_terms %||% list()
  has_slope <- length(re_terms) > 0L &&
    any(vapply(re_terms, function(rt) (rt$n_coefs %||% 1L) > 1L, logical(1)))
  if (has_spatial && has_slope) {
    stop("Random-slope term(s) together with a spatial field are not supported ",
         "through tulpa() yet. Use a random intercept (1 | g) alongside the ",
         "spatial term, or drop the spatial field.", call. = FALSE)
  }
  # Random-slope terms have no scalar `sigma_re` to condition on: the RE
  # covariance must be integrated. Every backend that would otherwise route
  # through the scalar-`sigma_re` GLMM log-posterior (`build_glmm_logpost`
  # applies one `sigma_re[k]` per term, dropping the intercept/slope
  # correlation) is redirected to a covariance-integrating fitter. The
  # deterministic Laplace mode integrates via nested Laplace; the sampler
  # modes integrate via the exact Metropolis-within-Gibbs debias.
  slope_scalar_backends <- c("laplace", "mala", "pathfinder", "imh_laplace")
  if (has_slope && sel$backend %in% slope_scalar_backends) {
    default_re_cov <- if (sel$backend == "laplace") "nested" else "gibbs"
    re_cov_method <- match.arg(control$re_cov %||% default_re_cov,
                               c("nested", "gibbs", "aghq"))
    backend <- if (re_cov_method == "gibbs") "re_cov_gibbs" else "re_cov_nested"
    sel <- .sel_redirect(sel, backend, sprintf(
      "random-slope term(s) present; RE covariance(s) integrated via %s (%d block(s))",
      backend, length(re_terms)))
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
    sel <- .sel_redirect(
      sel, "spde",
      "SPDE spatial field; nested-Laplace over (range, sigma) via fit_spde()")
  }

  # A temporal field (rw1/rw2/ar1) integrates through the nested-Laplace temporal
  # kernel; with a spatial field present it joins a [spatial, temporal] joint
  # prior. The auto / structured / conditional-Laplace selections route a
  # temporal field there (the conditional mode = "laplace" is not wired for
  # temporal yet), mirroring and superseding the spatial-field redirect above
  # when both fields are present. An explicitly chosen ModelData sampler backend
  # (hmc / ess / sghmc / sgld / mclmc / smc / vi) consumes the temporal field
  # directly (gcol33/tulpa#75), so it keeps its selection rather than being
  # redirected.
  if (has_temporal && BACKEND_REGISTRY[[sel$backend]]$input != "modeldata") {
    sel <- .sel_redirect(sel, "nested_laplace", if (has_spatial) {
      sprintf("%s spatial field + temporal %s field; joint nested-Laplace integration",
              spatial_type, temporal_spec$type)
    } else {
      sprintf("temporal %s field; nested-Laplace integration", temporal_spec$type)
    })
  }

  # Covariate smoothers are temporal-shaped blocks and integrate through the
  # same nested-Laplace kernels; redirect every selection there (mirroring the
  # temporal redirect above). The ModelData samplers do not thread smoother
  # blocks, so an explicitly chosen one errors rather than dropping the terms.
  if (has_smooth && sel$backend != "nested_laplace") {
    if (BACKEND_REGISTRY[[sel$backend]]$input == "modeldata") {
      stop("s(...) smoothers are not threaded through the ModelData samplers; ",
           "use mode = 'auto', 'structured', or 'nested_laplace'.",
           call. = FALSE)
    }
    sel <- .sel_redirect(sel, "nested_laplace", sprintf(
      "covariate smoother%s s(...); nested-Laplace integration",
      if (length(smooth_specs) > 1L) "s" else ""))
  }

  # Tier-1 exact reference for a continuous SPDE field: the generic ModelData /
  # logpost samplers do not carry the FEM Matern precision, so a Tier-1 mode
  # (`mode = "exact"` / a Tier-1 sampler) under an SPDE field routes to the SPDE
  # NUTS engine (tulpa_nuts_spde, via fit_spde(mode = "nuts")) -- the exact
  # counterpart to the nested-Laplace SPDE path. Joint over the Matern
  # hyperparameters (range, sigma are sampled).
  if (identical(tolower(spatial_type %||% ""), "spde") && isTRUE(sel$tier == 1L)) {
    if (length(bundle$re_terms %||% list()) > 0L) {
      stop("An SPDE field with a random-effect term under a Tier-1 exact mode is ",
           "not supported; use mode = 'laplace' / 'auto' (which support one ",
           "`(1 | g)` term), or drop the RE term.", call. = FALSE)
    }
    fit <- fit_spde(
      y = bundle$y, X = bundle$X, spatial = spatial_spec, family = family,
      n_trials = n_trials, mode = "nuts",
      control = .control_subset(control, .CONTROL_KEYS$nuts_spde))
    fit$formula <- formula
    fit$family <- family
    fit$call <- match.call()
    fit$inference_mode <- "exact"
    fit$inference_tier <- 1L
    fit$selection_reason <-
      "SPDE field, Tier-1 mode: exact NUTS over the Matern field + hyperparameters"
    fit$N <- fit$N %||% bundle$n_obs
    fit$model_matrix <- fit$model_matrix %||% bundle$X
    fit$y <- fit$y %||% bundle$y
    return(.finalize_fit(fit, backend = "spde", draws_kind = "chain",
                         n_fixed = ncol(bundle$X),
                         fixed_names = colnames(bundle$X)))
  }

  assert_backend_reachable(sel$backend)

  # Conditional backends (everything except the sigma-sampling Gibbs, the
  # Sigma-integrating re_cov backends, the marginal-likelihood AGQ fit that
  # estimates the RE sd, and the ModelData samplers that draw the RE sd jointly)
  # need one RE sd per term to condition on; resolve/recycle it after the backend
  # is known so the others do not emit a misleading "conditioning" message.
  if (K > 0L && !sel$backend %in% c("gibbs", "re_cov_nested", "re_cov_gibbs", "agq") &&
      BACKEND_REGISTRY[[sel$backend]]$input != "modeldata") {
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
                             spatial = spatial_spec, temporal = temporal_spec,
                             weights = weights, phi2 = phi2,
                             smoothers = lapply(smooth_specs, `[[`, "block"),
                             re_prior = re_prior)

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
    # Fixed-effect design for fitted()/predict(newdata = NULL), plus the pieces
    # posterior_predict() needs to rebuild the in-sample linear predictor and
    # push it through the family sampler: offset, response, trials, dispersion,
    # and the per-term RE row design (group index + slope columns).
    fit$model_matrix <- bundle$X
    fit$offset       <- fit$offset %||% bundle$offset
    fit$y            <- fit$y %||% bundle$y
    fit$n_trials     <- fit$n_trials %||% n_trials
    # Named obs_weights: nested fits already carry grid `$weights`.
    fit$obs_weights  <- weights
    fit$phi          <- fit$phi %||% phi
    fit$phi2         <- phi2
    fit$re_design    <- lapply(bundle$re_terms %||% list(), function(rt) {
      rt[c("group_idx", "has_intercept", "slope_matrix", "n_groups", "n_coefs")]
    })
    # Attach the validated spatial spec so predict() can krige the field to new
    # coordinates (HSGP basis / GP-NNGP conditional mean). fit_spde already sets
    # $spatial; the nested gp/nngp/hsgp path does not, so fill it here.
    if (!is.null(spatial_spec)) fit$spatial <- fit$spatial %||% spatial_spec
    # Smoother metadata for smooth_effects(): node locations plus the block
    # sizes needed to index the latent tail of the per-grid modes.
    if (length(smooth_specs) > 0L) {
      fit$smooth_terms    <- lapply(smooth_specs, `[[`, "meta")
      fit$n_latent_blocks <- parsed$n_latent_blocks %||% 0L
    }
  }
  fit
}
