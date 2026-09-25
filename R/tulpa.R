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


# A plain scalar random intercept `(1 | g)`: exactly one coefficient and that
# coefficient IS the intercept. Every backend that carries a random effect only
# through a group index (the nested-Laplace native RE, the Polya-Gamma Gibbs
# samplers, AGQ, SPDE) can represent this and nothing else -- a no-intercept
# single slope `(0 + x | g)` has n_coefs == 1 too but rides its slope column as
# a design, which those group-index-only paths would silently drop. The single
# predicate keeps that distinction in one place (design principle 5).
.is_scalar_re_intercept <- function(rt) {
  (rt$n_coefs %||% 1L) == 1L && isTRUE(rt$has_intercept)
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


# Starting point and metric for the front door's MALA: the posterior mode and
# the Laplace covariance there, the inverse of the precision above.
#
# MALA's own default is the identity metric from the builder's zero start, and
# on a GLMM that is a metric off by an order of magnitude in both directions --
# a fixed slope resolved to ~0.05, a group effect spanning its whole prior SD --
# with the fixed intercept and the group effects tied along a ridge the data
# never resolve. The step size dual-averages down to the narrowest direction
# and the chain crawls along the rest: bulk ESS 1 to 9 of 1000 on the intercept
# of a 15-group poisson (1 | g), point estimates 0.31 to 0.48 across seeds
# against imh_laplace's 0.41 (gcol33/tulpa#878). The dense Laplace covariance
# carries both the scales and that correlation; the MH step keeps the target
# exact whatever metric is used, so a poor Laplace fit costs efficiency, never
# correctness. Falls back to the precision's diagonal, then to the builder's
# own start, when the mode search or the factorization fails.
.glmm_mala_metric <- function(m) {
  fallback <- list(init = m$init)
  mp <- tryCatch(.glmm_mode_precision(m), error = function(e) NULL)
  if (is.null(mp) || !all(is.finite(mp$mode)) ||
      !is.finite(m$log_posterior(mp$mode))) {
    return(fallback)
  }
  init <- stats::setNames(mp$mode, names(m$init))
  S <- tryCatch(chol2inv(chol(mp$precision)), error = function(e) NULL)
  if (!is.null(S) && all(is.finite(S))) {
    return(list(init = init, mass_matrix = S))
  }
  dg <- diag(mp$precision)
  if (all(is.finite(dg) & dg > 0)) {
    return(list(init = init, mass_diag = 1 / dg))
  }
  list(init = init)
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
      # Per-component scaling beyond the reference scale (gcol33/tulpa#902).
      prior$node_prec <- spatial$node_prec
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
    # The basis does not carry the coordinates, so their extent travels with
    # the block: it anchors the lengthscale's default prior.
    return(c(list(
      type       = "hsgp",
      phi_basis  = basis$phi_basis,
      lambda_eig = basis$lambda_eig
    ), .hp_coord_fields(cm)))
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
      cov_type    = gp_cov_type(spatial),
      spatial_idx = as.integer(spatial$obs_to_loc %||% seq_len(n_spatial))
    ))
  }

  if (type == "spde") {
    # An SPDE field normally reaches its own integrator (fit_spde), whose outer
    # grid is (range, sigma) and whose `sigma_re` is a scalar it conditions on.
    # This converter is the route taken when the formula ALSO carries a random
    # intercept: as a block here, the field is integrated beside the RE term's
    # own `iid` block, so the RE SD becomes an outer axis instead of being
    # pinned at 1 (gcol33/tulpa#817).
    #
    # Integer nu only. Fractional nu in fit_spde() is the operator-based
    # rational construction (`.spde_rational_assemble()`), not the shifted-alpha
    # precision `make_spde_block` builds, and fit_spde() refuses a random
    # effect alongside it anyway -- so there is no fractional + RE fit for this
    # route to take over, and it does not claim one.
    if (.spde_nu_is_fractional(spatial$nu)) {
      stop("A random-effect term alongside a fractional-nu SPDE field is not ",
           "supported yet; use an integer nu, or drop the random-effect term.",
           call. = FALSE)
    }
    return(list(
      type    = "spde",
      n_mesh  = as.integer(spatial$n_mesh),
      # The single-arm converter reads a scalar `n_obs` (the joint one a
      # per-arm vector). A is n_obs x n_mesh, so the projector states it.
      n_obs   = as.integer(nrow(spatial$A)),
      A_x     = as.numeric(spatial$A_x),
      A_i     = as.integer(spatial$A_i),
      A_p     = as.integer(spatial$A_p),
      C0_diag = as.numeric(spatial$C0_diag),
      G1_x    = as.numeric(spatial$G1_x),
      G1_i    = as.integer(spatial$G1_i),
      G1_p    = as.integer(spatial$G1_p),
      nu      = as.numeric(spatial$nu)
    ))
  }

  # Any other type is genuinely unsupported on this path.
  stop(sprintf(paste0(
    "The generic nested-Laplace converter supports areal (%s) and continuous\n",
    "(%s) spatial fields; '%s' is not routed here. Use one of those, or\n",
    "mode = 'laplace' for a conditional fit at a fixed hyperparameter."),
    paste(.NL_FRONTDOOR_AREAL, collapse = ", "),
    paste(.NL_FRONTDOOR_CONTINUOUS, collapse = ", "), spatial$type), call. = FALSE)
}


# Pack a validated tulpa_gp (nngp) spec into the ModelData sampler's GP
# spatial_spec (mode = "exact" continuous-field NUTS). The sampler's GP prior
# requires a PC RANGE anchor, which spatial_gp() does not expose, so a
# weakly-informative data-driven default is derived here (P(range < U) = alpha,
# U = median nearest-neighbour spacing). The AMPLITUDE anchor it does expose,
# and this reads it: the pair was hardcoded to (2.0, 0.05) here against the
# engine's own (1.0, 0.01) -- one PC anchor with two defaults in two files, and
# a user setting it on an NNGP spec silently got neither
# (gcol33/tulpa#700). Index conventions match the hmc_gp
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
    coords           = .coords_2col(uc, "gp() / nngp() under a sampler mode"),
    nn               = nn,
    nn_idx           = matrix(as.integer(ni$nn_idx), n_loc, nn),
    nn_dist          = matrix(as.numeric(ni$nn_dist), n_loc, nn),
    nn_neighbor_dist = as.numeric(aperm(ni$nn_neighbor_dist, c(3, 2, 1))),
    nn_order         = as.integer(ni$nn_order) - 1L,
    nn_order_inv     = as.integer(ni$nn_order_inv %||% seq_len(n_loc)) - 1L,
    obs_to_loc       = as.integer(spatial$obs_to_loc) - 1L,
    cov_type         = gp_cov_type(spatial),
    nu               = as.numeric(spatial$nu %||% 1.5),
    phi_prior_U      = as.numeric(U),
    phi_prior_alpha  = 0.05,
    sigma2_prior_U     = as.numeric(spatial$sigma2_prior_U %||% 1),
    sigma2_prior_alpha = as.numeric(spatial$sigma2_prior_alpha %||% 0.01),
    # Non-centered (z ~ N(0, I), field reconstructed as w = f(z, sigma2, phi))
    # is the default: the centered parameterization funnels the field amplitude
    # against (sigma2, phi) under NUTS. "collapsed" is deprecated and falls back
    # to centered here.
    gp_parameterization =
      if (identical(spatial$parameterization, "noncentered")) 1L else 0L
  )
}


# Pack a validated tulpa_multiscale (multi-scale NNGP) spec into the
# ModelData sampler's multiscale spatial_spec (mode = "exact" continuous-field
# NUTS). Two independent NNGP scales (local / regional), each with its own
# neighbour structure computed by validate_gp() (same compute_nngp_neighbors()
# helper spatial_gp() uses, so the field/index conventions match
# .gp_sampler_spec() exactly: field at unique-location order, nn_order
# 0-based, nn_neighbor_dist row-major [i, j1, j2]). The sigma2 PC-prior
# anchors are not yet exposed by spatial_multiscale(), so this uses the same
# fixed weakly-informative default .gp_sampler_spec() does rather than
# inventing a data-driven one; the range bounds are the ones the user already
# set via range_local / range_regional.
#' @keywords internal
.msgp_sampler_spec <- function(spatial) {
  nil <- spatial$neighbor_info_local
  nir <- spatial$neighbor_info_regional
  if (is.null(nil) || is.null(nir) || is.null(spatial$unique_coords)) {
    stop("multiscale spatial spec is unvalidated (neighbor_info_local / ",
         "neighbor_info_regional / unique_coords NULL). tulpa() validates ",
         "it via validate_gp().", call. = FALSE)
  }
  uc <- as.matrix(spatial$unique_coords)
  n_loc <- nrow(uc)
  nn_local <- as.integer(spatial$nn_local %||% ncol(nil$nn_idx))
  nn_regional <- as.integer(spatial$nn_regional %||% ncol(nir$nn_idx))
  # "auto" resolves to non-centered, matching spatial_gp()'s default. The
  # remaining spatial_multiscale(sampler=) modes name strategies the exact-NUTS
  # path does not implement; surface that rather than silently sampling a
  # different parameterization than the one asked for.
  sampler <- spatial$sampler %||% "auto"
  if (!sampler %in% c("auto", "noncentered", "centered")) {
    stop("spatial_multiscale(sampler = \"", sampler, "\") is not implemented on ",
         "the exact-NUTS path; it samples the two scales as \"noncentered\" ",
         "(the default) or \"centered\".", call. = FALSE)
  }
  noncentered <- !identical(sampler, "centered")
  list(
    type                      = "multiscale",
    coords                    = .coords_2col(uc, "the multiscale GP sampler spec"),
    nn_local                  = nn_local,
    nn_idx_local              = matrix(as.integer(nil$nn_idx), n_loc, nn_local),
    nn_dist_local             = matrix(as.numeric(nil$nn_dist), n_loc, nn_local),
    nn_neighbor_dist_local    = as.numeric(aperm(nil$nn_neighbor_dist, c(3, 2, 1))),
    nn_order_local            = as.integer(nil$nn_order) - 1L,
    nn_order_inv_local        = as.integer(nil$nn_order_inv %||% seq_len(n_loc)) - 1L,
    nn_regional                = nn_regional,
    nn_idx_regional            = matrix(as.integer(nir$nn_idx), n_loc, nn_regional),
    nn_dist_regional           = matrix(as.numeric(nir$nn_dist), n_loc, nn_regional),
    nn_neighbor_dist_regional  = as.numeric(aperm(nir$nn_neighbor_dist, c(3, 2, 1))),
    nn_order_regional          = as.integer(nir$nn_order) - 1L,
    nn_order_inv_regional      = as.integer(nir$nn_order_inv %||% seq_len(n_loc)) - 1L,
    obs_to_loc                = as.integer(spatial$obs_to_loc) - 1L,
    cov_type                  = gp_cov_type(spatial),
    range_local_lower         = as.numeric(spatial$range_local[1]),
    range_local_upper         = as.numeric(spatial$range_local[2]),
    range_regional_lower      = as.numeric(spatial$range_regional[1]),
    range_regional_upper      = as.numeric(spatial$range_regional[2]),
    # Each scale's range carries the PC prior the GP path uses, anchored at
    # that scale's own declared lower bound: P(range < lower) = alpha. The
    # bounds are no longer a hard box; they state the
    # plausible interval, the lower end anchors the prior, and the pair places
    # the sampler's starting range.
    range_local_prior_alpha    = 0.05,
    range_regional_prior_alpha = 0.05,
    sigma2_local_prior_U        = 2.0,
    sigma2_local_prior_alpha    = 0.05,
    sigma2_regional_prior_U     = 2.0,
    sigma2_regional_prior_alpha = 0.05,
    # Non-centered (z ~ N(0, I) per scale, each field reconstructed as
    # w = f(z, sigma2, phi)) avoids the field/hyperparameter funnel
    # gp_parameterization / svc_parameterization document, independently per
    # scale.
    msgp_parameterization = if (noncentered) 1L else 0L
  )
}


# Pack a validated tulpa_hsgp spec into the ModelData sampler's HSGP
# spatial_spec (mode = "exact" continuous-field NUTS). The Laplacian basis is
# built in C++ by setup_hsgp_2d (the single source of truth) from the validated
# per-observation coordinate matrix, so only (coords, m, c) cross the boundary;
# the field is evaluated per observation, no obs->location map. The PC prior on
# sigma carries the spec's own anchors, defaulting to P(sigma > 1) = 0.01; the
# LogNormal on the lengthscale is fixed in compute_hsgp_spatial_prior.
#' @keywords internal
.hsgp_sampler_spec <- function(spatial) {
  cm <- spatial$coords_matrix
  if (is.null(cm)) {
    stop("HSGP spatial spec is unvalidated (coords_matrix NULL). tulpa() ",
         "validates it via validate_hsgp().", call. = FALSE)
  }
  cm <- as.matrix(cm)
  out <- list(
    type   = "hsgp",
    coords = .coords_2col(cm, "hsgp() under a sampler mode"),
    m      = as.integer(spatial$m),
    c      = as.numeric(spatial$c)
  )
  if (!is.null(spatial$sigma2_prior_U)) {
    out$sigma2_prior_U     <- as.numeric(spatial$sigma2_prior_U)
    out$sigma2_prior_alpha <- as.numeric(spatial$sigma2_prior_alpha)
  }
  out
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


# Pack a validated tulpa_svc spatially-varying coefficients spec into the
# ModelData sampler's svc_spec (mode = "exact" only). Each SVC term j carries a
# spatial field w_j(s); the generic log-post adds eta_i += sum_j X_svc[i,j] w_j(s_i).
# X_svc is the design subset (the varying coefficients' columns) in row-major
# [n_obs x n_svc], built identically on either branch. Two field
# representations (spatial_svc(approx=)):
#  * NNGP: coords row-major; nn_idx 1-based; nn_order 0-based; the SVC kernel
#    derives neighbour-pair distances from coords, so no nn_neighbor_dist. The
#    PC range anchor spatial_svc() does not expose is derived from the median
#    nearest-neighbour spacing.
#  * HSGP: the field is a sum of Laplacian basis functions evaluated at every
#    observation (same construction as .hsgp_sampler_spec() for a plain
#    spatial field), so only per-observation coords + (m, c_boundary) cross
#    the boundary -- build_sampler_model_inputs() builds the shared basis via
#    setup_hsgp_2d() and keys the block on `data.svc_is_hsgp`
#    (gcol33/tulpa#813). There is no obs->location map and no NNGP range
#    anchor (the HSGP path puts LogNormal(0, 1) on the log-lengthscale
#    directly; see compute_svc_prior()).
#' @keywords internal
.svc_sampler_spec <- function(spatial, X) {
  idx <- as.integer(spatial$svc_indices)
  if (is.null(spatial$coords_matrix) || is.null(idx)) {
    stop("SVC spec is unvalidated (coords_matrix / svc_indices NULL). ",
         "tulpa() validates it via validate_svc().", call. = FALSE)
  }
  cm <- as.matrix(spatial$coords_matrix)
  Xs <- as.matrix(X)[, idx, drop = FALSE]           # [n_obs x n_svc]

  if (identical(tolower(spatial$approx %||% "nngp"), "hsgp")) {
    return(list(
      approx      = "hsgp",
      coords      = .coords_2col(cm, "svc(approx = 'hsgp') under a sampler mode"),
      n_svc       = length(idx),
      svc_indices = idx,
      X_svc       = as.numeric(t(Xs)),               # row-major [n_obs x n_svc]
      m           = as.integer(spatial$m),
      c           = as.numeric(spatial$c_boundary)
    ))
  }

  ni <- spatial$neighbor_info
  if (is.null(ni)) {
    stop("SVC spec is unvalidated (neighbor_info NULL). tulpa() validates ",
         "it via validate_svc().", call. = FALSE)
  }
  n_obs <- nrow(cm)
  nn    <- as.integer(spatial$nn %||% ncol(ni$nn_idx))
  pos_d <- ni$nn_dist[is.finite(ni$nn_dist) & ni$nn_dist > 0]
  U     <- if (length(pos_d)) stats::median(pos_d) else 0.1
  if (!is.finite(U) || U <= 0) U <- 0.1
  list(
    approx          = "nngp",
    coords          = .coords_2col(cm, "svc() under a sampler mode"),
    n_svc           = length(idx),
    nn              = nn,
    nn_idx          = matrix(as.integer(ni$nn_idx), n_obs, nn),
    nn_dist         = matrix(as.numeric(ni$nn_dist), n_obs, nn),
    nn_order        = as.integer(ni$nn_order) - 1L,
    nn_order_inv    = as.integer(ni$nn_order_inv %||% seq_len(n_obs)) - 1L,
    svc_indices     = idx,
    X_svc           = as.numeric(t(Xs)),            # row-major [n_obs x n_svc]
    cov_type        = gp_cov_type(spatial),
    phi_prior_U     = as.numeric(U),
    phi_prior_alpha = 0.05,
    # Non-centered by default, matching .gp_sampler_spec()'s
    # gp_parameterization: each term's field is reconstructed as
    # w_j = f(z_j, sigma2_j, phi_j), which removes the funnel that attenuates a
    # weakly identified field's amplitude (sd ratio 0.98 against 0.33 centered,
    # on the one-trial binomial fixture, at a tenth of the cost). Only usable
    # once the soft sum-to-zero pin stopped fighting the reparameterization.
    svc_parameterization =
      if (identical(spatial$parameterization, "centered")) 0L else 1L
  )
}


# The structures the TVC block's density actually has a branch for
# (`tvc_log_prior()` / `tvc_log_prior_gp()` in src/hmc_tvc.h dispatch on
# TemporalType). One predicate, asked by the sampler spec below and by the
# front door's wrong-mode message, so the two cannot disagree about what is
# fittable.
#' @keywords internal
.TVC_STRUCTURES <- c("rw1", "rw2", "ar1", "gp")

#' @keywords internal
.tvc_structure_or_stop <- function(structure) {
  st <- tolower(structure %||% "rw1")
  if (!st %in% .TVC_STRUCTURES) {
    stop(sprintf(
      "A temporally-varying coefficient evolves as %s; got '%s'.",
      paste(shQuote(.TVC_STRUCTURES), collapse = " / "), st), call. = FALSE)
  }
  st
}

# Pack a validated tulpa_tvc spec into the ModelData sampler's tvc_spec
# (mode = "exact" only). Each TVC term j carries a temporal field w_j(g, t); the
# generic log-post adds eta_i += sum_j X_tvc[i,j] w_j(g_i, t_i). X_tvc is
# row-major [n_obs x n_tvc].
#
# `rw1` / `rw2` / `ar1` read the time index as a position on a grid and send
# nothing further. `gp` is the continuous-time structure: it additionally sends
# where the distinct instants SIT and the kernel its covariance is built from,
# the same three fields temporal_gp() sends (gcol33/tulpa#847).
#' @keywords internal
.tvc_sampler_spec <- function(temporal, X) {
  st <- .tvc_structure_or_stop(temporal$structure)
  idx <- as.integer(temporal$tvc_indices)
  Xt  <- as.matrix(X)[, idx, drop = FALSE]          # [n_obs x n_tvc]
  n_groups <- as.integer(temporal$n_groups %||% 1L)
  spec <- list(
    n_times     = as.integer(temporal$n_times),
    n_tvc       = length(idx),
    n_groups    = n_groups,
    time_index  = as.integer(temporal$time_index),
    group_index = as.integer(temporal$group_index %||% rep(1L, nrow(as.matrix(X)))),
    tvc_indices = idx,
    X_tvc       = as.numeric(t(Xt)),                # row-major [n_obs x n_tvc]
    structure   = st,
    cyclic      = isTRUE(temporal$cyclic),
    sigma_prior_U     = as.numeric(temporal$sigma_prior_U %||% 1),
    sigma_prior_alpha = as.numeric(temporal$sigma_prior_alpha %||% 0.01)
  )
  if (identical(st, "gp")) {
    if (!is.numeric(temporal$time_values) ||
        length(temporal$time_values) != spec$n_times) {
      stop("Internal: a GP TVC spec is unvalidated (time_values missing). ",
           "tulpa() validates it via validate_tvc().", call. = FALSE)
    }
    spec$time_values <- as.numeric(temporal$time_values)
    spec$cov         <- as.character(temporal$cov %||% "exponential")
    spec$nu          <- temporal$nu
    # The period in the units `time_values` now carries, which is what the
    # kernel measures its lag in.
    spec$period      <- temporal$period_scaled %||% temporal$period
    spec$phi_prior_lower <- temporal$phi_prior_lower %||% .GP_PHI_PRIOR_BOUNDS[["lower"]]
    spec$phi_prior_upper <- temporal$phi_prior_upper %||% .GP_PHI_PRIOR_BOUNDS[["upper"]]
  }
  spec
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
  if (identical(type, "multiscale")) {
    stop("The multiscale temporal block (trend + seasonal + short-term) is fit ",
         "by the sampler; there is no multiscale nested-Laplace kernel. Use ",
         "mode = 'hmc'.", call. = FALSE)
  }
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
  if (type %in% c("rw1", "rw2")) out$cyclic <- isTRUE(temporal$cyclic)
  # AR1 rho prior travels on the block; the outer nested-Laplace grid reweights
  # by it (.nl_apply_ar1_rho_prior). NULL stays the default Uniform(-1, 1).
  if (type == "ar1" && !is.null(temporal$rho_prior)) {
    out$rho_prior <- temporal$rho_prior
  }
  out
}


# The fixed-effect prior a backend actually received, read back off the
# assembled fitter arguments: `beta_prior` where the fitter takes the list,
# `sigma_beta` where it takes the scalar mean-zero SD. A backend that carries
# neither expresses no Gaussian prior on the fixed effects (the nested-Laplace,
# SPDE and spatial-Laplace paths hold their own field-conditional prior), and
# reports NULL rather than a default it did not apply.
.beta_prior_applied <- function(args, resolved) {
  if (!is.null(args$beta_prior)) return(args$beta_prior)
  if (!is.null(args$sigma_beta)) return(list(mean = 0, sd = args$sigma_beta))
  # A logpost backend receives closures that already carry the prior, so the
  # resolved object is what went into them.
  if (!is.null(args$log_posterior)) return(resolved)
  NULL
}


# Assemble the fitter argument list for a backend from the model pieces. Routes
# on the backend's input contract (BACKEND_REGISTRY$<backend>$input). Backends
# that are reachable but not yet wired through tulpa() error with guidance.
# `control$n_threads` under mode = "auto" (gcol33/tulpa#911). The sampler
# branch of .tulpa_fitter_args() refuses the knob, because a sampler reads no
# thread count and a knob dropped in silence is what the control check exists
# to prevent. That refusal is right for a mode the caller NAMED; under "auto"
# the caller cannot know which backend the router will pick, and the knob is
# one every Laplace candidate reads, so refusing it made the same call fail or
# succeed depending on the model's terms. Under auto the knob is dropped with a
# message naming the backend that ignored it; an explicit mode keeps the error.
#' @keywords internal
.auto_drop_unread_threads <- function(control, sel) {
  if (is.null(control$n_threads) || isTRUE(sel$explicit) ||
      !identical(BACKEND_REGISTRY[[sel$backend]]$input, "modeldata")) {
    return(control)
  }
  message("tulpa(): mode = 'auto' resolved to the sampler backend '",
          sel$backend, "', which does not read `control$n_threads`; it is ",
          "ignored. A sampler run's OpenMP teams are sized from ",
          "`control$n_chains` and the environment (OMP_NUM_THREADS).")
  control$n_threads <- NULL
  control
}

.tulpa_fitter_args <- function(backend, bundle, family, sigma_re,
                               n_trials, phi, beta_prior, control,
                               latent_blocks = list(), spatial = NULL,
                               temporal = NULL, weights = NULL,
                               phi2 = NULL, smoothers = list(),
                               re_prior = NULL, zi_prior = NULL,
                               hyperprior = "proper",
                               warm_start = NULL, estimate_phi = FALSE,
                               beta_prior_default = NULL) {
  if (identical(hyperprior, "flat") && !backend %in% .hyperprior_backends()) {
    stop(sprintf(paste0(
      "`hyperprior = \"flat\"` is not read by backend '%s', which carries ",
      "priors of its own. It applies to: %s."),
      backend, paste(.hyperprior_backends(), collapse = ", ")), call. = FALSE)
  }
  # Scalar prior SD on the beta_zi block; the engine default when unset. Read
  # once here so every ZI-carrying backend receives the same number.
  zi_prior_sd <- .normalize_zi_prior(zi_prior)
  # The fixed-effect prior is a modelling statement and a backend is a
  # computational choice, so the default is resolved ONCE at the front door and
  # passed in: `y ~ x` and `y ~ x + (1 | g)` route to different backends and
  # must not be fitted under different priors because of it. The branches that
  # cannot express a Gaussian fixed-effect prior still test the SUPPLIED
  # `beta_prior`, so the resolved default does not turn their guard into an
  # error on every fit.
  beta_prior_default <- beta_prior_default %||% .tulpa_default_beta_prior()
  # Statistical random-effect / variance-component hyperpriors ride in a single
  # `re_prior` list (a statistical argument), never in `control` (tuning only).
  rp <- re_prior %||% list()
  .check_re_prior_backend(rp, backend)
  input <- BACKEND_REGISTRY[[backend]]$input

  # Observation weights scale each row's log-likelihood. Supported where the
  # likelihood carries a per-obs multiplier today: the non-spatial Laplace
  # kernel (directly, or as re_cov_nested's inner solve) and the R
  # log-posterior builder. Everything else refuses loudly rather than silently
  # fitting unweighted.
  if (!is.null(weights) &&
      !.backend_carries_weights(backend, spatial = !is.null(spatial))) {
    stop(sprintf(paste0(
      "`weights` is not supported by backend '%s'. Weighted fits run through ",
      "mode = 'laplace' (non-spatial), 're_cov_nested', or a log-posterior ",
      "sampler ('mala', 'imh_laplace', 'pathfinder')."), backend),
      call. = FALSE)
  }

  # Second dispersion (Student-t df, Tweedie power). The backend list is derived
  # from the registry (.phi2_backends()) rather than restated here. The one
  # qualification the list cannot express: `laplace` carries phi2 through the
  # non-spatial compiled kernel only, the spatial solvers having no phi2 channel
  # -- which is what tulpa_laplace() refuses for the same pair.
  phi2_carried <- backend %in% .phi2_backends() &&
    !(backend == "laplace" && !is.null(spatial))
  if (!is.null(phi2) && !phi2_carried) {
    stop(sprintf(paste0(
      "`phi2` is not supported by backend '%s', which would fit at the ",
      "family's default second dispersion instead. Available for: %s ",
      "(`laplace` non-spatially)."),
      backend, paste(.phi2_backends(), collapse = ", ")), call. = FALSE)
  }

  # Zero inflation. `.zi_backends()` (checked upstream, before `spatial` is in
  # scope there) lists `laplace` unconditionally -- it reaches the mixture
  # through the non-spatial two-process spec (X_zi + a beta_zi block); the
  # spatial solvers (icar/car/bym2/car_proper/gp/nngp/hsgp/spde) have no
  # zero-inflation channel, and tulpa_laplace() itself refuses `X_zi` together
  # with `spatial`. Un-qualified, the spatial branch below built its argument
  # list with no `X_zi` at all, so the mixture was silently never fit while
  # `coef()` still reported a `zi_(Intercept)` from the parsed formula
  # (gcol33/tulpa#793). Same "non-spatially" qualification as `weights` and
  # `phi2` above.
  zi_carried <- !(backend == "laplace" && !is.null(spatial))
  if (!is.null(bundle$X_zi) && backend %in% .zi_backends() && !zi_carried) {
    stop(sprintf(paste0(
      "`ziformula` is not threaded through backend '%s' with a spatial ",
      "field: the spatial Laplace solvers have no zero-inflation channel. ",
      "Drop `ziformula` or `spatial`, or use a non-spatial mode."), backend),
      call. = FALSE)
  }

  if (input == "nested") {
    if (!isTRUE(BACKEND_REGISTRY[[backend]]$dispatchable %||% TRUE)) {
      stop(sprintf(paste0(
        "Backend '%s' is a nested engine driven by model packages, not the\n",
        "single-response tulpa() formula -- it needs multiple response arms,\n",
        "which a single formula cannot express. Call %s() directly."),
        backend, BACKEND_REGISTRY[[backend]]$fitter %||% backend), call. = FALSE)
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
    # The `spatial =` / `temporal =` fields carry a `role` tag. A block's type
    # does not say what it is for -- an s(x) smoother is an rw1 / rw2 block and
    # a `(1 | g)` term an iid one -- so the accessors that read one field back
    # off the fit (temporal(), spatial_range(), temporal_corr()) find it by
    # role (`.nl_block_roles()`), never by type or position (gcol33/tulpa#903,
    # #906).
    with_role <- function(blk, role) { blk$role <- role; blk }
    field_blocks <- c(
      if (!is.null(spatial))
        list(with_role(.spatial_spec_to_nl_prior(spatial), "spatial")) else list(),
      if (!is.null(temporal))
        list(with_role(.temporal_spec_to_nl_prior(temporal), "temporal")) else list(),
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
    # Each random-effect term becomes an `iid` latent block, so its SD is
    # integrated on the outer grid alongside the other blocks' hyperparameters
    # rather than conditioned at a scalar. The driver's native
    # re_idx / n_re_groups / sigma_re channel conditions, which on a formula whose
    # other structure IS integrated made the RE the one variance component the fit
    # never estimated -- and at the default sigma_re = 1, a number the data never
    # produced. It is left unused here.
    #
    # `sigma_re` supplied explicitly still conditions, as a one-point sigma_grid:
    # the iid registry entry documents that a length-1 grid fixes the field at
    # that SD, so conditioning is the degenerate case of the same path rather
    # than a second one.
    #
    # The RE blocks go LAST, which is what lets ranef() find them: their
    # coefficients are then the tail of the latent vector, so the accessor slices
    # the last sum(n_groups) columns without needing any other block's width. Any
    # other position would mean re-deriving every preceding block's latent size
    # here, a second source of truth for something the driver already knows.
    re <- bundle$re_terms %||% list()
    re_blocks <- list()
    re_conditioned <- !is.null(sigma_re)
    if (length(re) > 0L) {
      bad <- !vapply(re, .is_scalar_re_intercept, logical(1))
      if (any(bad)) {
        stop(paste0(
          "The nested-Laplace path carries random-INTERCEPT terms `(1 | g)` as\n",
          "iid latent blocks; a random-slope term has no iid form (it needs a `Z`\n",
          "design). Drop the slope, or fit the covariance directly with\n",
          "mode = 're_cov_nested' / 're_cov_gibbs' (which cannot carry the\n",
          "smoother / field blocks in this formula)."), call. = FALSE)
      }
      # One value per term is the front door's contract; recycle it here because
      # the nested backend is exempt from tulpa()'s own recycling (it does not
      # condition by default).
      s_re <- if (re_conditioned) {
        s <- as.numeric(sigma_re)
        if (length(s) == 1L) rep(s, length(re)) else s
      } else NULL
      if (!is.null(s_re) && length(s_re) != length(re)) {
        stop(sprintf("`sigma_re` must have length 1 or %d (one per RE term).",
                     length(re)), call. = FALSE)
      }
      re_blocks <- lapply(seq_along(re), function(m) {
        blk <- list(type = "iid",
                    obs_idx = as.integer(re[[m]]$group_idx),
                    n_units = as.integer(re[[m]]$n_groups))
        if (!is.null(s_re)) blk$sigma_grid <- s_re[m]
        blk
      })
    }
    all_blocks <- c(all_blocks, re_blocks)
    # A length-1 list routes the single-block path; length > 1 the multi-block
    # joint path (both are handled by tulpa_nested_laplace()). An `iid` block is
    # multi-block-only, which holds here: the length-0 check above already
    # required a field / smoother / latent block, so RE blocks are never alone.
    prior <- if (length(all_blocks) == 1L) all_blocks[[1L]] else all_blocks
    out <- list(
      y           = bundle$y,
      n_trials    = n_trials %||% rep(1L, bundle$n_obs),
      X           = bundle$X,
      prior       = prior,
      # The native RE channel is unused: the terms are blocks now.
      re_idx      = rep(0L, bundle$n_obs),
      n_re_groups = 0L,
      sigma_re    = 1.0,
      family      = family,
      phi         = phi,
      offset      = bundle$offset,
      hyperprior  = hyperprior,
      # Forward only the keys the inner fitter reads: front-door-only knobs
      # (grid shape, backend selection) were consumed above and would trip
      # tulpa_nested_laplace()'s own whitelist.
      control     = .control_subset(control, .CONTROL_KEYS$nested_laplace)
    )
    # Where the RE blocks landed in the prior list, and whether their SD was
    # integrated or conditioned. Read by VarCorr() / ranef() so they report the
    # integrated posterior instead of falling through to the conditioning
    # fallback. Carried as ATTRIBUTES, not list elements: this list is the
    # fitter's argument list, and an extra element would reach
    # tulpa_nested_laplace() as an unknown argument.
    if (length(re_blocks)) {
      attr(out, "re_block_index") <-
        length(all_blocks) - length(re_blocks) + seq_along(re_blocks)
      attr(out, "re_block_conditioned") <- re_conditioned
    }
    return(out)
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
      if (length(re) > 1L || !.is_scalar_re_intercept(re[[1]])) {
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
    return(list(
      y              = bundle$y,
      X              = bundle$X,
      spatial        = spatial,
      family         = family,
      n_trials       = n_trials %||% rep(1L, bundle$n_obs),
      range          = NULL,
      sigma          = NULL,
      nested_laplace = TRUE,
      phi            = phi,
      offset         = bundle$offset,
      re_idx         = spde_re_idx,
      n_re_groups    = spde_re_n,
      sigma_re       = spde_sigma_re,
      hyperprior     = hyperprior,
      control        = .control_subset(control, .CONTROL_KEYS$spde)
    ))
  }

  if (input == "design") {
    if (backend %in% c("re_cov_nested", "re_cov_gibbs", "eb")) {
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
      # Zero inflation rides into the Laplace inner solve `eb` and
      # `re_cov_nested` share, so the covariance is estimated / integrated under
      # the mixture rather than under a model missing it. re_cov_gibbs builds
      # its own conditional and carries no second process; it is absent from
      # .zi_backends(), so the refusal upstream has already fired by here.
      if (!is.null(bundle$X_zi)) {
        common$X_zi <- bundle$X_zi
        common$zi_prior_sd <- zi_prior_sd
      }
      # The second dispersion rides into the same Laplace inner solve `eb` and
      # `re_cov_nested` share, so the covariance is estimated / integrated at the
      # degrees of freedom (or variance power) asked for rather than at the
      # kernel's default. re_cov_gibbs has no phi2 channel and is absent from
      # .phi2_backends(), so the refusal upstream has already fired by here.
      if (!is.null(phi2)) common$phi2 <- phi2
      # An offset changes the model, so it is threaded where the inner solve
      # carries it and refused where it does not -- never dropped. The
      # Metropolis-within-Gibbs sampler has no offset term at all.
      has_offset <- !is.null(bundle$offset) && any(bundle$offset != 0)
      if (backend == "re_cov_gibbs") {
        if (has_offset) {
          stop("`offset()` is not supported by the re_cov_gibbs backend. Use ",
               "mode = 're_cov_nested' or mode = 'eb', which thread the offset ",
               "through their Laplace inner solve.", call. = FALSE)
        }
      } else {
        common$offset <- bundle$offset
      }
      if (backend == "eb") {
        # Same blocks, same hyperprior, same inner solve as re_cov_nested --
        # tulpa_eb() stops at the maximizer instead of integrating around it.
        # `marginal` is a formal argument of tulpa_eb() rather than one of its
        # control knobs, so it is lifted out of `control` here; the front door
        # has no other way to request intervals that carry the hyperparameter
        # uncertainty rather than condition on theta_hat.
        return(c(common, list(
          beta_prior   = beta_prior_default,
          prior_sigma  = rp$prior_sigma,
          eta          = rp$eta,
          hyperprior   = hyperprior,
          n_quad       = as.integer(control$n_quad %||% 1L),
          estimate_phi = estimate_phi,
          # `[[` not `$`: the latter partial-matches, so `marginal_step` alone
          # would switch the correction on.
          marginal     = isTRUE(control[["marginal"]]),
          control      = .control_subset(control,
                                         setdiff(.CONTROL_KEYS$eb, "marginal"))
        )))
      }
      if (backend == "re_cov_nested") {
        # `control$re_cov = "aghq"` is the nested integrator with an AGHQ inner
        # marginal: n_quad defaults to 9 there, to the plain joint Laplace (1)
        # otherwise. An explicit control$n_quad always wins.
        re_cov_method <- .re_cov_method(control, "nested")
        n_quad <- as.integer(control$n_quad %||%
                               (if (re_cov_method == "aghq") 9L else 1L))
        return(c(common, list(
          weights     = weights,
          beta_prior  = beta_prior_default,
          prior_sigma = rp$prior_sigma,
          eta         = rp$eta,
          hyperprior  = hyperprior,
          n_quad      = n_quad,
          control     = .control_subset(control, .CONTROL_KEYS$re_cov_nested)
        )))
      }
      # re_cov_gibbs: exact Metropolis-within-Gibbs debias.
      return(c(common, list(
        prior_df        = rp$prior_df,
        prior_scale     = rp$prior_scale,
        beta_prior      = beta_prior_default,
        control         = .control_subset(control, .CONTROL_KEYS$re_cov_gibbs)
      )))
    }
    if (backend == "laplace") {
      # tulpa_laplace() takes its numerical knobs as plain formals, and this
      # arg list used to carry none of them: `max_iter` / `tol` / `n_threads`
      # passed tulpa()'s union check and were dropped, as was every knob only
      # another backend reads (`n_iter`, `adapt_delta`, `adaptive_grid`, ...).
      # Validate against what the fitter reads, then forward it
      # (gcol33/tulpa#870, the #770 fix for this branch). An unset knob is
      # omitted so tulpa_laplace()'s own formal default applies.
      tulpa_check_control(control, .CONTROL_KEYS$laplace,
                          "tulpa[mode = 'laplace']")
      numerics <- .drop_null(list(max_iter  = control$max_iter,
                                  tol       = control$tol,
                                  n_threads = control$n_threads))
      if (!is.null(spatial)) {
        # Spatial Laplace: route the field spec through tulpa_laplace(spatial=),
        # which dispatches on spatial$type (icar/car/bym2/spde/gp). At most one
        # random-intercept (1 | g) term may ride alongside the field -- the
        # spatial solvers consume a single RE block (re_list[[1]]); richer RE
        # structure is not supported here. An offset() is threaded into the
        # field's linear predictor; beta_prior is not threaded through the
        # spatial solvers, so reject it loudly rather than drop.
        re <- bundle$re_terms %||% list()
        if (length(re) > 1L || (length(re) == 1L && !.is_scalar_re_intercept(re[[1]]))) {
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
        return(c(list(
          y = bundle$y, n_trials = n_trials, X = bundle$X,
          re_list = .bundle_to_re_list(bundle, sigma_re),
          family = family, phi = phi, spatial = spatial,
          offset = bundle$offset
        ), numerics))
      }
      return(c(list(
        y = bundle$y, n_trials = n_trials, X = bundle$X,
        re_list = .bundle_to_re_list(bundle, sigma_re),
        family = family, phi = phi, phi2 = phi2,
        offset = bundle$offset, beta_prior = beta_prior_default,
        weights = weights, X_zi = bundle$X_zi,
        zi_prior_sd = zi_prior_sd,
        # Keeps the joint latent precision (`H_latent`) on the fit, so the
        # linear-predictor draws behind posterior_predict() and WAIC / LOO take
        # the random effects jointly with the fixed effects (gcol33/tulpa#871).
        return_joint_hessian = TRUE
      ), numerics))
    }
    if (backend == "gibbs") {
      re <- bundle$re_terms %||% list()
      # The spatial Polya-Gamma samplers carry the iid random-intercept block
      # alongside the field, so 0 or 1 `(1 | g)` term is allowed; the plain
      # Gibbs path requires exactly one. Either way the term must be a single
      # random intercept (no slopes).
      if (!is.null(spatial)) {
        if (length(re) > 1L || (length(re) == 1L && !.is_scalar_re_intercept(re[[1]]))) {
          stop("Spatial Gibbs supports at most one random-intercept (1 | g) ",
               "term alongside the spatial field.", call. = FALSE)
        }
      } else if (length(re) != 1L || !.is_scalar_re_intercept(re[[1]])) {
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
      # An offset changes the model, so it is refused here rather than dropped:
      # the Polya-Gamma kernels (plain, spatial and temporal alike) carry no
      # offset term in their linear predictor at all.
      if (!is.null(bundle$offset) && any(bundle$offset != 0)) {
        stop("`offset()` is not supported by the gibbs (Polya-Gamma) backend. ",
             "Use mode = 'laplace', 're_cov_nested' or a logpost backend ",
             "(mode = 'mala'), which thread the offset through.", call. = FALSE)
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
        beta_prior = beta_prior_default,
        prior_sigma_scale = rp$prior_sigma_scale %||% 2.5,
        spatial = spatial,
        control = .control_subset(control, .CONTROL_KEYS$gibbs)
      ))
    }
    if (backend == "agq") {
      # Adaptive Gauss-Hermite quadrature: one intercept-only random-effect term,
      # families binomial / poisson / gaussian (the lme4::glmer(nAGQ=) scope).
      # n_quad = 1 is the joint Laplace; higher quadrature reduces the
      # small-cluster variance attenuation. agq_fit() optimizes the marginal
      # likelihood and estimates the RE sd, so no sigma_re is conditioned on.
      # agq_fit() is a marginal-likelihood maximizer with no `control` list of
      # its own (plain formals), so sampler knobs like `n_iter` / `seed` /
      # `n_chains` / `thin` passed tulpa()'s union check and did nothing
      # (gcol33/tulpa#770); reject them here instead.
      tulpa_check_control(control, .CONTROL_KEYS$agq, "tulpa[mode = 'agq']")
      re <- bundle$re_terms %||% list()
      if (length(re) != 1L || !.is_scalar_re_intercept(re[[1]])) {
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
      # `phi` is the residual VARIANCE at every door (gcol33/tulpa#560);
      # agq_fit() takes the SD. One conversion, no second spelling:
      # `control$sigma_eps` used to override here, which put the same quantity
      # in `control` under a second name AND in the other convention
      # (gcol33/tulpa#675). It is refused by name below.
      return(.drop_null(list(
        y          = bundle$y,
        X          = bundle$X,
        group      = as.integer(re[[1]]$group_idx),
        n_groups   = re[[1]]$n_groups,
        family     = family,
        n_trials   = n_trials,
        sigma_eps  = if (!is.null(phi)) sqrt(phi) else 1.0,
        n_quad     = control$n_quad,
        offset     = bundle$offset,
        beta_init  = control$beta_init,
        sigma_init = control$sigma_init,
        max_iter   = control$max_iter,
        tol        = control$tol,
        verbose    = control$verbose
      )))
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
      return(list(
        y          = bundle$y,
        X          = bundle$X,
        family     = family,
        phi        = phi %||% 1.0,
        phi2       = phi2,
        n_trials   = n_trials,
        beta_prior = beta_prior_default,
        offset     = bundle$offset,
        control    = .control_subset(control, .CONTROL_KEYS$ep)
      ))
    }
    stop(sprintf(paste0(
      "Backend '%s' is reachable but not yet wired through tulpa(). Call its ",
      "fitter directly (e.g. agq_fit())."), backend), call. = FALSE)
  }

  if (input == "logpost") {
    # `mala()` / `pathfinder()` / `imh_laplace()` take their tuning knobs as
    # plain formals rather than a `control` list, so they cannot self-validate
    # the way `tulpa_gibbs()` / `tulpa_ep()` / `tulpa_sample_glmm()` do; a knob
    # only some OTHER backend reads (e.g. `n_chains`, or `agq`'s `n_quad`)
    # passed tulpa()'s union check upstream and was then silently dropped
    # rather than forwarded (gcol33/tulpa#770). Re-validate here, against the
    # selected backend's own key set, now that the backend is fixed.
    tulpa_check_control(control, .CONTROL_KEYS[[backend]],
                        sprintf("tulpa[mode = '%s']", backend))
    m <- build_glmm_logpost(bundle, family, sigma_re = sigma_re,
                            n_trials = n_trials, phi = phi,
                            beta_prior = beta_prior_default,
                            weights = weights, phi2 = phi2)
    # A knob the caller did not set is OMITTED, so the backend's own formal
    # supplies it: restating `n_iter = control$n_iter %||% 2000L` here put the
    # same default in two files, and a bump on one side would have been
    # invisible from the other -- the drift gcol33/tulpa#632 measured on
    # `k_samples` (gcol33/tulpa#676).
    if (backend == "mala") {
      pre <- .glmm_mala_metric(m)
      return(.drop_null(list(
        log_posterior = m$log_posterior,
        grad_log_posterior = m$grad_log_posterior,
        init = pre$init,
        mass_matrix = pre$mass_matrix,
        mass_diag = pre$mass_diag,
        n_iter = control$n_iter,
        warmup = control$warmup,
        epsilon = control$epsilon,
        thin = control$thin,
        seed = control$seed,
        verbose = control$verbose
      )))
    }
    if (backend == "pathfinder") {
      return(.drop_null(list(
        log_posterior = m$log_posterior,
        init = m$init,
        grad_log_posterior = m$grad_log_posterior,
        n_draws = control$n_draws,
        max_iter = control$max_iter,
        tol = control$tol,
        seed = control$seed,
        verbose = control$verbose
      )))
    }
    if (backend == "imh_laplace") {
      # Independence MH with a Laplace proposal: needs the MAP + precision.
      mp <- .glmm_mode_precision(m)
      return(.drop_null(list(
        log_posterior = m$log_posterior,
        mode = mp$mode,
        hessian = mp$precision,
        n_iter = control$n_iter,
        warmup = control$warmup,
        scale = control$scale,
        thin = control$thin,
        seed = control$seed,
        verbose = control$verbose
      )))
    }
    stop(sprintf(paste0(
      "Backend '%s' is reachable but not yet wired through tulpa(). Call its ",
      "fitter directly."), backend), call. = FALSE)
  }

  if (input == "modeldata") {
    # The model-agnostic ModelData sampler kernels (hmc/ess/sghmc/sgld/mclmc/
    # smc/vi) sample the full latent vector compute_param_layout() lays out
    #. Random effects (intercept / slopes / correlated /
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
    } else if (!is.null(spatial) && tolower(spatial$type %||% "") == "multiscale") {
      if (isTRUE((spatial$approx %||% "nngp") == "hsgp")) {
        stop(sprintf(paste0(
          "Backend '%s' samples the multi-scale field via NNGP; ",
          "spatial_multiscale(approx = \"hsgp\") is not threaded through this ",
          "path, or through nested-Laplace (the multi-scale field has no ",
          "nested-Laplace kernel at all -- see .FRONTDOOR_MULTISCALE). Use ",
          "approx = \"nngp\"."), backend),
          call. = FALSE)
      }
      spatial_spec_arg <- .msgp_sampler_spec(spatial)
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
      # The ModelData sampler's BYM2 carries ONE scale for the whole graph, in
      # an exported struct; per-component scaling (a disconnected graph, an
      # island) lives on the nested-Laplace, Laplace and Gibbs kernels, so the
      # sampler refuses it rather than fit a differently scaled model
      # (gcol33/tulpa#902).
      if (identical(sp$type, "bym2") && !is.null(sp$node_prec)) {
        stop(sprintf(paste0(
          "Backend '%s' scales a BYM2 field with one factor for the whole ",
          "graph, and this graph has several connected components (or an ",
          "isolated node), which are scaled separately. Fit it with ",
          "mode = 'auto' / 'nested_laplace' / 'laplace', or mode = 'gibbs' for ",
          "a binomial response."), backend), call. = FALSE)
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
    } else if (!is.null(temporal) &&
               tolower(temporal$type %||% "") == "multiscale") {
      # Multi-scale temporal (trend + seasonal + short-term) is one block with
      # its own layout, so it rides temporal_spec under its own type rather than
      # being flattened into a single field. build_sampler_model_inputs() reads
      # these names directly onto MultiscaleTemporalData.
      n_groups <- as.integer(temporal$n_groups %||% 1L)
      temporal_spec_arg <- list(
        type        = "multiscale",
        time_index  = as.integer(temporal$time_index),
        n_times     = as.integer(temporal$n_times),
        n_groups    = n_groups,
        group_index = if (n_groups > 1L) as.integer(temporal$group_index) else NULL,
        trend       = tolower(temporal$trend %||% "none"),
        seasonal    = if (!is.null(temporal$seasonal))
                        as.integer(temporal$seasonal) else NULL,
        short_term  = tolower(temporal$short_term %||% "none"),
        shared      = isTRUE(temporal$shared %||% TRUE)
      )
    } else if (!is.null(temporal) &&
               tolower(temporal$type %||% "") == "gp") {
      # Continuous-time GP over irregularly-spaced times. The field carries one
      # effect per UNIQUE instant, so it needs where those instants sit
      # (time_values) on top of the per-observation index the discrete kernels
      # use. Everything else the kernel needs -- which covariance, its
      # smoothness / period, the parameterization -- rides along, since a
      # covariance choice that reaches no further than the spec object is
      # silently run as something else.
      n_groups <- as.integer(temporal$n_groups %||% 1L)
      temporal_spec_arg <- list(
        type             = "gp",
        time_idx         = as.integer(temporal$time_index),
        time_values      = as.numeric(temporal$time_values),
        n_times          = as.integer(temporal$n_times),
        n_groups         = n_groups,
        group_idx        = if (n_groups > 1L) as.integer(temporal$group_index) else NULL,
        cyclic           = FALSE,
        cov              = temporal$cov %||% "exponential",
        nu               = temporal$nu,
        # The period in the kernel's own time units (validate_temporal_gp);
        # equal to the declared one when scale_coords = FALSE.
        period           = temporal$period_scaled %||% temporal$period,
        parameterization = temporal$parameterization %||% "noncentered",
        # The lengthscale support in the same kernel units (validate_temporal_gp).
        phi_prior_lower  = temporal$phi_prior_lower %||% .GP_PHI_PRIOR_BOUNDS[["lower"]],
        phi_prior_upper  = temporal$phi_prior_upper %||% .GP_PHI_PRIOR_BOUNDS[["upper"]]
      )
    } else if (!is.null(temporal)) {
      ttype <- tolower(temporal$type %||% "")
      if (!ttype %in% c("rw1", "rw2", "ar1")) {
        stop(sprintf(paste0(
          "Backend '%s' samples temporal fields rw1 / rw2 / ar1 / gp; got '%s'."),
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
      if (ttype == "ar1") {
        ab <- .ar1_rho_beta_ab(temporal$rho_prior)
        temporal_spec_arg$rho_prior_a <- ab[1L]
        temporal_spec_arg$rho_prior_b <- ab[2L]
      }
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

    # `n_threads` reaches this point because tulpa()'s control surface is the
    # union over the backends it dispatches, and no sampler kernel reads it. The
    # outermost region in a sampler run is the across-chain loop, sized from
    # n_chains and the environment (OMP_NUM_THREADS, OMP_THREAD_LIMIT, the
    # two-core cap R CMD check sets), and the intra-chain field-gradient regions
    # size themselves from that same environment. Dropping the knob on the way in
    # is the silent no-op the control check exists to prevent, so it is refused
    # where it is written.
    if (!is.null(control$n_threads)) {
      stop(paste0(
        "Backend '", backend, "' does not read `control$n_threads`. A sampler\n",
        "run's OpenMP teams are sized from `control$n_chains` and the\n",
        "environment; set OMP_NUM_THREADS to bound them. The knob applies to\n",
        "the nested-Laplace, SPDE and RE-covariance modes. A GP / SVC fit\n",
        "reproduces at a fixed `control$seed` at any team size."),
        call. = FALSE)
    }

    sampler_args <- list(
      y             = bundle$y,
      n_trials      = n_trials %||% rep(1L, n_obs),
      X             = bundle$X,
      family        = family,
      backend       = backend,
      phi           = phi,
      phi2          = phi2,
      offset        = bundle$offset,
      fixed_names   = bundle$fixed_names,
      re_spec       = re_spec,
      spatial_spec  = spatial_spec_arg,
      temporal_spec = temporal_spec_arg,
      svc_spec      = svc_spec_arg,
      tvc_spec      = tvc_spec_arg,
      zi_spec       = if (is.null(bundle$X_zi)) NULL
                      else list(X = bundle$X_zi, prior_sd = zi_prior_sd),
      sigma_re_scale = rp$sigma_re_scale %||% 2.5,
      # The fixed-effect prior SD is the statistical `beta_prior` (mean-zero on
      # this sampler path), not a control knob; inject it into the sampler's
      # sigma_beta. Other perf knobs forward from control.
      sigma_beta    = .beta_prior_ridge_sd(beta_prior_default),
      control       = .control_subset(control, .CONTROL_KEYS$sample_glmm)
    )

    # Resolved here rather than at the front door because the source fit and the
    # layout probe both need the assembled sampler arguments -- the same design
    # matrices, specs and priors the kernel is about to receive.
    if (!is.null(warm_start)) {
      sampler_args$warm_start <- .resolve_warm_start(
        warm_start  = warm_start,
        args        = sampler_args,
        re_terms    = .bundle_to_re_list(bundle, sigma_re),
        sigma_re    = sigma_re,
        beta_prior  = beta_prior_default,
        n_chains    = as.integer(sampler_args$control$n_chains %||% 4L))
    }

    return(sampler_args)
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
#'   CCD design, PC + LKJ hyperprior by default -- see `hyperprior`);
#'   `control$re_cov = "gibbs"` switches to the exact
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
#'   present; `mode = "nested_laplace"` forces it. Several random-intercept
#'   `(1 | g)` terms may accompany the blocks; the one-term restriction belongs
#'   to the Polya-Gamma spatial Gibbs sampler (`mode = "gibbs"`), which updates
#'   one RE block alongside the field. Joint multi-arm nested models cannot be
#'   expressed by a single-response formula -- call
#'   [tulpa_nested_laplace_joint()] directly.
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
#'   random effects, and `offset(...)` terms are recognised. As in [stats::lm()] and
#'   lme4, aliased fixed-effect columns of a rank-deficient design are dropped
#'   with a warning, and unused levels of a grouping factor are dropped. A
#'   binomial response may be logical (`TRUE` a success).
#' @param data A data frame. Incomplete rows (an `NA` in the response, a
#'   predictor or a grouping variable) are refused, not dropped.
#' @param family Character family name: one of [family_names()]
#'   (`"binomial"`, `"poisson"`, `"neg_binomial_2"`, `"gaussian"`, `"beta"`,
#'   ...), or a categorical response family -- `"multinomial"`
#'   (baseline-category logit via [tulpa_multinomial()]), `"ordinal"`
#'   (cumulative logit via [tulpa_ordinal()]), or `"ordinal_probit"`
#'   (cumulative probit). Categorical families take fixed-effect models only.
#' @param mode Inference mode or backend. `"auto"` (default) picks the most
#'   reliable Tier 1/Tier 2 method expected to finish; a tier (`"exact"`,
#'   `"structured"`) or a backend name (`"laplace"`, `"mala"`, ...) forces it.
#'   `"eb"` estimates the random-effect covariance(s) by empirical Bayes instead
#'   of conditioning on `sigma_re` (see [tulpa_eb()]); it is opt-in by name,
#'   because its intervals are conditional on that estimate rather than marginal
#'   over it. The maximized likelihood integrates the fixed effects out rather
#'   than profiling them, so with `hyperprior = "flat"` the estimate is
#'   restricted (REML-type) -- it matches `glmmTMB(..., REML = TRUE)`, not the
#'   ML fit of `lme4::glmer()`.
#' @param sigma_re Random-effect SDs to condition on: length 1 (recycled) or one
#'   per RE term. Defaults to 1 per term with a warning. Ignored by every
#'   backend that DETERMINES the RE scale itself -- by integrating it
#'   (`re_cov_nested`, `re_cov_gibbs`), sampling it (`gibbs` and the ModelData
#'   samplers, which carry `log_sigma_re` in the latent vector) or maximizing
#'   over it (`eb`, `agq`) -- and supplying it there warns.
#' @param n_trials Binomial denominators (length `nrow(data)`), or `NULL`.
#' @param weights Optional observation weights (non-negative numeric vector,
#'   length `nrow(data)`): each observation's log-likelihood contribution is
#'   scaled by its weight (prior / frequency weights, e.g. survey weights or
#'   aggregated-data counts -- a weight of 2 is equivalent to duplicating the
#'   row). Supported on the non-spatial Laplace path (`mode = "laplace"`), the
#'   nested-Laplace random-effect covariance integrator (`re_cov_nested`, which
#'   `mode = "auto"` / `"structured"` pick for a weighted random-effect model,
#'   so its scale is still integrated) and the log-posterior samplers (`mala`,
#'   `imh_laplace`, `pathfinder`); other backends reject weights loudly.
#' @template phi
#' @param estimate_phi Estimate the dispersion from the data instead of
#'   conditioning on `phi`, which then supplies the starting value. `log(phi)`
#'   joins the empirical-Bayes maximization as one further coordinate carrying
#'   the exact derivative of the Laplace log-marginal, so the estimate is
#'   ML-II: the hyperprior covers the random-effect covariances only and the
#'   dispersion enters unpenalized. That marginal integrates the fixed effects
#'   out, so the estimate is the restricted (REML-type) one -- for a gaussian
#'   response, the REML residual variance. `fit$phi` is the estimate and
#'   `fit$phi_estimated` distinguishes it from a conditioned value.
#'
#'   Available under `mode = "eb"`, and for the families whose dispersion
#'   derivative is registered (see [tulpa_eb()]). Any other mode errors rather
#'   than fitting at the starting value under a name that says otherwise.
#' @param phi2 Optional second dispersion: the Student-t degrees of freedom
#'   (`family = "t"`; default 4 when `NULL`) or the Tweedie variance power
#'   (`family = "tweedie"`, required -- a defaulted power would be a statistical
#'   decision the caller never made). Supported on the non-spatial Laplace path,
#'   the random-effect covariance paths (`mode = "eb"` and the nested `Sigma`
#'   integrator, which thread it into their inner Laplace solve), the
#'   log-posterior samplers, and the ModelData samplers. Backends without a
#'   `phi2` channel refuse it rather than fit at the family's default.
#'   `estimate_phi` covers `phi` alone; `phi2` is always conditioned on.
#' @param beta_prior Optional `list(mean, sd)` Gaussian prior on the fixed
#'   effects. `NULL` takes the engine default, `prior_normal(0, 2.5)`, on every
#'   backend that carries a fixed-effect prior -- the prior is a modelling
#'   statement, so the backend `mode = "auto"` selects does not change it. The
#'   nested-Laplace and SPDE paths hold their own field-conditional prior and
#'   reject a supplied `beta_prior`. The resolved prior is reported on the fit
#'   as `$beta_prior`.
#' @param re_prior Optional `list()` of random-effect / variance-component
#'   hyperpriors (statistical, so they live in the signature rather than in
#'   `control`). Recognised entries, each consumed by the backend that needs it:
#'   `prior_sigma` (PC-prior anchor `c(U, alpha)` on a free RE covariance SD,
#'   used when `hyperprior = "proper"`), `eta` (LKJ concentration for a
#'   correlated RE covariance, same condition), `prior_df` / `prior_scale`
#'   (inverse-Wishart on the RE covariance, `control$re_cov = "gibbs"`),
#'   `prior_sigma_scale` (half-Cauchy scale on the RE SD for `mode = "gibbs"`),
#'   and `sigma_re_scale` (half-Cauchy scale on the RE / BYM2 SD for the
#'   ModelData samplers). An entry the resolved backend does not read is an
#'   error naming the backends that do.
#' @param hyperprior The outer hyperparameter prior, `"proper"` (default) or
#'   `"flat"`, forwarded to every route that integrates or maximizes over
#'   hyperparameters: the nested-Laplace path (see [tulpa_nested_laplace()]),
#'   the SPDE path ([fit_spde()]), and the random-effect covariance routes
#'   (`mode = "laplace"` with random slopes, [tulpa_re_cov_nested()];
#'   `mode = "eb"`, [tulpa_eb()]). `"proper"` is each route's normalised default
#'   (the PC prior on a scale, the PC range prior, PC + LKJ over a free
#'   covariance); `"flat"` folds none of the engine's own, so the fit reports no
#'   evidence. A density the call states applies under either. Every other
#'   backend carries priors of its own, and `"flat"` there is an error rather
#'   than a choice that silently does nothing.
#' @param ziformula Optional one-sided formula for the zero-inflation
#'   probability, e.g. `~ 1` for a constant structural-zero rate or `~ x` to
#'   model it. The response becomes a mixture: with probability
#'   `plogis(X_zi beta_zi)` the observation is a structural zero, otherwise it
#'   is drawn from `family`. Available for the count families with a compiled
#'   zero-inflated kernel; paired with `truncated_poisson` or
#'   `truncated_neg_binomial_2` it is the hurdle model, since the base
#'   `P(Y = 0)` is then 0 and the mixture degenerates to the two-part
#'   likelihood. Backends that do not carry the mixture refuse it rather than
#'   fit the model without it.
#' @param zi_prior Optional `list(sd)` Gaussian prior on the zero-inflation
#'   coefficients, `beta_zi ~ N(0, sd^2)`; `NULL` (default) uses 2.5. One scalar
#'   SD applies to the whole block, and the mean is fixed at 0, because that is
#'   what the compiled kernels carry. The prior is what identifies the logit
#'   where a level contributes no zeros -- there the likelihood is monotone in
#'   that coefficient and alone would send it to `-Inf`. `sd = Inf` removes the
#'   penalty. Ignored without `ziformula`.
#' @param warm_start Optional starting point for the NUTS sampler, from a
#'   cheaper fit of the same model: `"eb"` or `"laplace"` fits one first, or
#'   pass an existing fit from either mode. The sampler then starts at that
#'   mode with an inverse mass read off its curvature, instead of at the origin
#'   with a structural one. Chains after the first are dispersed around the
#'   mode at the fit's own scale, so between-chain spread -- which `rhat()`
#'   compares against -- is not collapsed by the shared starting point. Only
#'   the NUTS/HMC backends take one; the rest error rather than ignore it. Not
#'   available under a spatial, temporal or GP field, whose hyperparameters
#'   neither source fit estimates.
#'
#'   The variance-component slots take an adapting mass by default, because a
#'   plug-in fit estimates no curvature for them. Passing a fit from
#'   [tulpa_eb()] with `marginal = TRUE` supplies one: its `theta_cov` gives
#'   each `log_sigma_re` slot a posterior variance to start from. This applies
#'   to uncorrelated terms, whose hyperparameter coordinates are the log
#'   standard deviations the sampler holds; a correlated term stays adapting,
#'   since its log-Cholesky coordinates are not the sampler's.
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
#' @param temporal Optional temporal field spec, integrated by nested Laplace
#'   for the discrete walks and sampled for the continuous ones:
#'   [temporal_rw1()], [temporal_rw2()], [temporal_ar1()] (nested Laplace);
#'   [temporal_gp()] and [temporal_multiscale()], which are sampler-path only --
#'   their hyperparameters are sampled jointly with the field, so `mode = "auto"`
#'   routes them to the exact ModelData sampler. A plain field routes the
#'   single-block temporal kernel; a `group_var` panel spec fits a separate walk
#'   per group sharing one hyperparameter; combined with an areal `spatial`
#'   field it forms an additive space-time joint prior.
#' @param control Optional list of backend tuning arguments. Each backend
#'   accepts its own set, checked at the door: `n_iter` / `warmup` / `epsilon`
#'   (`mala`), `n_draws` (`pathfinder`), `n_chains` / `max_treedepth` /
#'   `adapt_delta` / `mass_matrix` (`hmc`), the outer-grid knobs
#'   `?tulpa_nested_laplace` documents (`n_per_axis`, `prune`, `screen_iters`,
#'   `diagnose_k`, `within_cell`, `checkpoint`, the `progress*` family, ...),
#'   and `re_cov` (`"nested"` / `"gibbs"` / `"aghq"`), which selects the
#'   RE-covariance integrator on any random-effect model.
#'
#'   `n_iter` does not count the same thing on every sampler, and `warmup`
#'   (alias `n_warmup`) is validated against it before any sampler runs:
#'   * On `hmc`, `ess`, `sghmc`, `sgld`, `gibbs`, `mala` and `imh_laplace`,
#'     `n_iter` is the TOTAL number of iterations, warmup included, so the fit
#'     keeps `n_iter - warmup` draws per chain and `warmup < n_iter` is
#'     required (`n_iter = 2000, warmup = 1000` keeps 1000).
#'   * On `re_cov_gibbs` (`control$re_cov = "gibbs"`) and `mclmc`, `n_iter` is
#'     the number of KEPT iterations and warmup runs on top of it, so any
#'     `warmup >= 0` is valid (`n_iter = 1000, warmup = 1000` keeps 1000).
#'   * `smc` and `vi` read neither (`n_particles`, `vi_max_iter` instead).
#'
#'   The default `warmup` is `n_iter %/% 2` on the ModelData samplers and 1000
#'   on `gibbs` / `re_cov_gibbs`.
#'
#'   One statistical knob lives here rather than in the signature: `marginal`
#'   (`mode = "eb"` only) turns on the marginal-Laplace covariance correction,
#'   which widens the reported intervals to account for the hyperparameter
#'   uncertainty EB conditions on. It is a formal argument of [tulpa_eb()] and
#'   is forwarded from `control` by `tulpa()` alone, so it has no meaning on any
#'   other backend and is refused there. See [tulpa_eb()].
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
                  estimate_phi = FALSE,
                  phi2 = NULL,
                  beta_prior = NULL,
                  re_prior = NULL,
                  hyperprior = c("proper", "flat"),
                  ziformula = NULL,
                  zi_prior = NULL,
                  warm_start = NULL,
                  spatial = NULL,
                  temporal = NULL,
                  control = list(),
                  ...) {
  # Whether the DISPERSION was the caller's or the signature's, recorded before
  # anything can overwrite `phi` (gcol33/tulpa#849).
  phi_supplied <- !missing(phi)
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
  # Named before the whitelist so a moved knob says where it went rather than
  # arriving as "unknown control knob" (gcol33/tulpa#675).
  if ("sigma_eps" %in% names(control)) {
    stop("`control$sigma_eps` is the gaussian residual SD, a second spelling ",
         "of the `phi` argument in the other convention. Pass `phi` (the ",
         "residual VARIANCE) instead.", call. = FALSE)
  }
  tulpa_check_control(control, .CONTROL_KEYS$tulpa, "tulpa")
  if ("hyperprior" %in% names(re_prior)) {
    stop("`re_prior$hyperprior` is the `hyperprior` argument of tulpa(). Pass ",
         "`hyperprior = \"proper\"` or `\"flat\"` there.", call. = FALSE)
  }
  tulpa_check_control(re_prior, .RE_PRIOR_KEYS, "tulpa (re_prior)")
  # The RE-covariance hyperprior anchors, named as the user set them rather
  # than as whichever backend builds the prior (gcol33/tulpa#894).
  if (!is.null(re_prior$eta)) {
    .check_lkj_eta(re_prior$eta, "re_prior$eta", "tulpa()")
  }
  if (!is.null(re_prior$prior_sigma)) {
    .check_pc_anchor_pair(re_prior$prior_sigma, "re_prior$prior_sigma",
                          "tulpa()")
  }
  hyperprior <- .hp_choice(match.arg(hyperprior))
  if (!is.logical(estimate_phi) || length(estimate_phi) != 1L ||
      is.na(estimate_phi)) {
    stop("`estimate_phi` must be TRUE or FALSE.", call. = FALSE)
  }

  # Argument shapes, before anything indexes them. Each of these is a plausible
  # user mistake that used to surface an R internal from deep inside the call --
  # "the condition has length > 1", "argument is of length zero",
  # "is.numeric(y) || is.integer(y) is not TRUE" -- naming no argument
  # (gcol33/tulpa#679).
  if (inherits(family, "family")) family <- .family_object_to_name(family)
  if (!is.character(family) || length(family) != 1L || is.na(family)) {
    stop("`family` must be a single family name (a string), or a stats::family ",
         "object. See ?tulpa for the accepted names.", call. = FALSE)
  }
  if (is.null(mode) || !is.character(mode) || length(mode) != 1L || is.na(mode)) {
    stop("`mode` must be a single string: 'auto', a tier name, or a backend ",
         "name. See ?tulpa.", call. = FALSE)
  }
  if (!inherits(formula, "formula") || length(formula) != 3L) {
    stop("`formula` must be two-sided, e.g. y ~ x. A one-sided formula names ",
         "no response to fit.", call. = FALSE)
  }
  if (is.data.frame(data) && nrow(data) == 0L) {
    stop("`data` has no rows.", call. = FALSE)
  }

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
                        beta_prior = beta_prior %||% .tulpa_default_beta_prior(),
                        control = .control_subset(control,
                                                  .CONTROL_KEYS$multinomial))
    } else {
      tulpa_ordinal(formula, data,
                    link = if (family == "ordinal_probit") "probit" else "logit",
                    beta_prior = beta_prior %||% .tulpa_default_beta_prior(),
                    control = .control_subset(control, .CONTROL_KEYS$ordinal))
    }
    fit$call <- match.call()
    return(fit)
  }

  family <- .canonical_family(family)
  .family_or_stop(family)
  .validate_family_compiled(family)
  .validate_family_phi(family, phi)
  if (!is.null(phi2)) .phi2_or_stop(family, phi2)
  # Tweedie requires the power up front (and in (1, 2)); fail before fitting.
  if (identical(family, "tweedie")) .tweedie_power(phi2)

  parsed <- tulpa_parse_formula(formula)
  bundle <- tulpa_build_model_data(parsed, data)
  bundle$y <- .numeric_response(family, bundle$y)
  .assert_complete_groups(bundle$re_terms)

  # A cbind(successes, failures) response carries its own denominators.
  n_trials <- .resolve_pair_trials(family, bundle$n_trials, n_trials)

  # One R-side reading of `n_trials`, so every door answers the same
  # (gcol33/tulpa#677). It used to be checked only at the C++ boundary, which
  # `laplace` reaches and `mala` / `imh_laplace` do not: a scalar was an error
  # on one door and a recycled vector on the others, and a `n_trials` handed to
  # a non-binomial family was read by nothing at all -- no signal for a user who
  # meant a binomial and typed poisson.
  n_trials <- .normalize_n_trials(family, n_trials, bundle$n_obs)

  # Zero inflation: a second linear predictor for the structural-zero logit.
  # The compiled Laplace kernel carries it as process 1 (eta[1]); see
  # src/builtin_family_zi.h. Validated here so an unsupported family fails at
  # the front door rather than reaching a kernel that would ignore it.
  #
  # Over a zero-truncated family this is the hurdle model: the mixture
  # degenerates to log(pi) at y = 0, so the zeros are carried by the zero
  # component and the family's own `y >= 1` requirement no longer applies to
  # them -- which is why the count check runs after the design is known.
  X_zi <- .zi_design(ziformula, data, bundle$n_obs)
  if (!is.null(X_zi)) {
    .validate_family_zi(family)
    .validate_family_zi_compiled(family)
    bundle$X_zi <- X_zi
  }
  .validate_family_support(family, bundle$y, n_trials = n_trials,
                           zi = !is.null(X_zi))
  # The model is built with na.action = na.pass (prior_predict() allows an NA
  # response), so tulpa() must reject non-finite fitting inputs itself: unlike
  # glm()/lm() it does not drop incomplete cases, and an NA/NaN/Inf would flow
  # silently into the C++ kernels as a NaN estimate.
  .assert_finite_model_inputs(bundle$X, bundle$y, n_trials = n_trials,
                              offset = bundle$offset)
  bundle$X <- .drop_aliased_fixed(bundle$X)
  bundle$n_fixed <- ncol(bundle$X)
  bundle$fixed_names <- colnames(bundle$X)
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

  # The inline field paths below return before `.tulpa_fitter_args()`, where
  # `re_prior` is checked against the backend; their joint nested-Laplace
  # driver reads none of its keys (gcol33/tulpa#893).
  if ((parsed$n_spatial_field_blocks %||% 0L) > 0L ||
      (parsed$n_temporal_field_blocks %||% 0L) > 0L) {
    .check_re_prior_backend(re_prior %||% list(), "nested_laplace_joint")
  }

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
    # One normalisation of the type string, before any consumer reads it. The
    # branches here lowercase it and select_inference_mode() compares it
    # case-sensitively, so an un-normalised `type = "ICAR"` reached the selector
    # as a type it does not know and was reported as unsupported
    # (gcol33/tulpa#671). Writing it back means the spec carries the canonical
    # spelling wherever it travels.
    spatial_type <- tolower(spatial_spec$type)
    spatial_spec$type <- spatial_type
    # An adjacency reaching the door as a bare list never passed a constructor,
    # so the graph check runs on whatever `spatial$adjacency` holds, whichever
    # way it got here (gcol33/tulpa#670). The check is idempotent, so a spec
    # built by spatial_icar() / spatial_car() is not re-reported.
    if (!is.null(spatial_spec$adjacency)) {
      spatial_spec$adjacency <-
        .validate_adjacency_arg(spatial_spec$adjacency, "spatial$adjacency")
    }
    sp_lc <- spatial_type
    # RSR is a MODIFIER on a field, not a field type: the spec keeps its own
    # `$type` and flags `$rsr` (spatial_rsr()). The backend selector sees "rsr"
    # so the fit reaches a Polya-Gamma sampler that applies the projection
    # rather than a plain path that would silently drop it, while `sp_lc` keeps
    # the underlying type so the spec still validates as the field it is -- an
    # areal one against its adjacency, a continuous one against its coordinates
    # (gcol33/tulpa#848). A bare `type = "rsr"` predates the modifier and has
    # always meant an areal field.
    if (isTRUE(spatial_spec$rsr) || identical(sp_lc, "rsr")) {
      if (identical(sp_lc, "rsr")) {
        sp_lc <- "icar"
        spatial_spec$type <- "icar"
        spatial_spec$rsr  <- TRUE
      }
      spatial_type <- "rsr"
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
      spatial_spec <- validate_svc(spatial_spec, data, bundle$X)
    } else if (sp_lc %in% c(.NL_FRONTDOOR_CONTINUOUS, .NL_FRONTDOOR_SPDE,
                            .FRONTDOOR_MULTISCALE)) {
      # Coordinate-addressed field: coords come from the spec; no spatial(col)
      # term. gp/nngp/hsgp resolve their coordinate structure via validate_*()
      # (gp/nngp: unique_coords / obs_to_loc / neighbor_info; hsgp:
      # coords_matrix at every observation for the basis builder). Multiscale
      # shares validate_gp() -- the same unique-location resolution, run once
      # per scale to derive neighbor_info_local / neighbor_info_regional. SPDE
      # is self-contained -- spatial_spde() built the mesh + FEM matrices
      # (A, C, G) at construction -- so it only needs a dimension check.
      if (!is.null(parsed$spatial_var)) {
        stop("A continuous spatial field (", spatial_type, ") is addressed by ",
             "its coordinate columns in the spec; drop the spatial(",
             parsed$spatial_var, ") term from the formula.", call. = FALSE)
      }
      if (sp_lc == "spde") {
        # The SPDE projector A maps observations -> mesh nodes; it must have one
        # row per observation in `data`. spatial_spde() builds A from the same
        # data, so a mismatch means the spec was built from a different frame.
        .check_spde_rows(spatial_spec, bundle$n_obs, "tulpa()")
      } else if (sp_lc == "hsgp") {
        if (!inherits(spatial_spec, "tulpa_hsgp")) {
          stop("An HSGP spatial field must be a spatial_gp(~ lon + lat, approx = 'hsgp') spec ",
               "object (it carries the coordinate columns); got a bare list.",
               call. = FALSE)
        }
        spatial_spec <- validate_hsgp(spatial_spec, data)
      } else {
        if (!inherits(spatial_spec, c("tulpa_gp", "tulpa_multiscale"))) {
          stop("A continuous spatial field must be a spatial_gp(~ lon + lat) or ",
               "spatial_multiscale(~ lon + lat) spec object (it carries the ",
               "coordinate columns); got a bare list.", call. = FALSE)
        }
        spatial_spec <- validate_gp(spatial_spec, data)
      }
      if (isTRUE(spatial_spec$rsr)) {
        # A restricted continuous field: the projector is built at the unique
        # locations the field is indexed by, which validate_gp() has just
        # resolved (gcol33/tulpa#848).
        spatial_spec <- .attach_rsr_projection(
          spatial_spec, data, family,
          obs_to_field = as.integer(spatial_spec$obs_to_loc),
          n_field = as.integer(spatial_spec$n_spatial %||%
                                 nrow(spatial_spec$unique_coords)))
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
      if (is.null(spatial_spec$adjacency)) {
        stop("An areal spatial field needs its adjacency matrix ",
             "(spatial_car(adjacency, ...)).", call. = FALSE)
      }
      adj_mat <- as.matrix(spatial_spec$adjacency)
      n_units <- nrow(adj_mat)
      # The unit column is matched to graph nodes by the one resolver every
      # areal door shares: by rownames(adjacency) when the ids are labels,
      # never by their sort order (gcol33/tulpa#900).
      spatial_spec$spatial_idx <- .resolve_spatial_idx(
        data[[parsed$spatial_var]], n_units, adj_mat, parsed$spatial_var)
      if (isTRUE(spatial_spec$rsr)) {
        # The unit-level projector orthogonal to the restrict_to design -- the
        # whole point of the modifier. dispatch_gibbs_spatial() consumes the
        # precomputed n_units x n_units projection.
        spatial_spec <- .attach_rsr_projection(
          spatial_spec, data, family,
          obs_to_field = spatial_spec$spatial_idx,
          n_field = n_units)
      }
    } else {
      stop("Unknown spatial type '", spatial_type, "'. `spatial$type` must be one ",
           "of: areal icar/car/bym2/car_proper, continuous ",
           "gp/nngp/hsgp/spde/multiscale, or rsr.", call. = FALSE)
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
           "/ temporal_gp() or temporal_tvc() spec object.", call. = FALSE)
    }
    if (identical(tolower(temporal_spec$type %||% ""), "gp")) {
      # Continuous-time GP over irregular times. Sampler-path only: the field is
      # a dense T x T Gaussian whose hyperparameters are sampled jointly, and
      # there is no nested-Laplace kernel laying a grid over them.
      if (has_spatial || has_latent) {
        stop("A temporal GP field cannot be combined with a spatial or ",
             "latent(...) field through tulpa() yet. Fit the temporal GP on ",
             "its own.", call. = FALSE)
      }
      temporal_spec <- validate_temporal_gp(temporal_spec, data)
    } else if (identical(tolower(temporal_spec$type %||% ""), "tvc")) {
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
    # multiscale rides the sampler path only; the nested-Laplace entry
    # (.temporal_spec_to_nl_prior) rejects it with its own message, since there
    # is no multiscale nested-Laplace kernel.
    if (!tolower(temporal_spec$type %||% "") %in%
        c("rw1", "rw2", "ar1", "multiscale")) {
      stop("tulpa() routes temporal_rw1() / temporal_rw2() / temporal_ar1() / ",
           "temporal_multiscale(); for '", temporal_spec$type,
           "' call the temporal fitter directly.", call. = FALSE)
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
      # gcol33/tulpa#812: the front-door gap named below.
      stop("A temporal field can accompany an areal (icar/car/bym2/car_proper) ",
           "spatial field through tulpa()'s joint nested-Laplace path; the '",
           spatial_type, "' field is fit by its own integrator through this ",
           "front door (continuous gp/nngp/hsgp and SPDE fields are each fit ",
           "one at a time here; RSR is sampler-only) and cannot host a ",
           "temporal block through tulpa() yet. A continuous (hsgp/nngp) ",
           "spatial field plus a temporal field IS fitted, directly, by ",
           "fit_st_nested(spatial_type = 'hsgp' or 'nngp', ...) -- it is not ",
           "yet routed through this formula front door. ",
           "Fit one field at a time here, use an areal field for space-time ",
           "through tulpa(), or call fit_st_nested() directly.", call. = FALSE)
    }
    # The multiscale validator is the superset: it resolves a
    # temporal_multiscale() spec and delegates every other spec to
    # validate_temporal(), so single-component fields are unaffected.
    temporal_spec <- validate_temporal_multiscale(temporal_spec, data)
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

  # Computed here (ahead of mode selection) so a bare `(1 | g)` term can steer
  # auto's default choice the same way a random slope already does -- see the
  # has_slope-based redirect below, which reuses this same `re_terms`.
  re_terms <- bundle$re_terms %||% list()
  has_re <- length(re_terms) > 0L

  fam_obj <- list(name = family, distribution = family)
  # The per-call features a backend can refuse at dispatch. Without them the
  # auto selector picked backends that then errored on the very call that
  # selected them (gcol33/tulpa#666, #681, #769). Kept in a variable (not just
  # inlined into the call below) so the slope redirect further down can
  # re-check the SAME features against its own redirect target.
  call_feat <- list(
    offset     = !is.null(bundle$offset) && any(bundle$offset != 0),
    weights    = !is.null(weights),
    ziformula  = !is.null(bundle$X_zi),
    phi2       = !is.null(phi2),
    n_re_terms = length(re_terms)
  )
  sel <- select_inference_mode(
    mode, family = fam_obj, n_obs = bundle$n_obs,
    has_spatial = has_spatial, has_temporal = has_temporal, has_latent = has_latent,
    spatial_type = spatial_type, temporal = temporal_spec, has_re = has_re,
    feat = call_feat
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
    # Only recommend `mode = "exact"` when it would in fact take this spec.
    # The structure check the sampler entry applies is the same one
    # `.tvc_sampler_spec()` / `.svc_sampler_spec()` make, so ask it here rather
    # than pointing the user at a mode that refuses for a second reason
    # (gcol33/tulpa#814).
    hint <- tryCatch({
      if (is_tvc_fit) .tvc_structure_or_stop(temporal_spec$structure)
      "Use mode = 'exact'."
    }, error = function(e) {
      paste0("mode = 'exact' does not take it either: ", conditionMessage(e))
    })
    stop(sprintf(paste0(
      "%s coefficients are sampled by the exact ModelData NUTS backend; the\n",
      "selected backend '%s' (mode = '%s') does not carry the varying-coefficient\n",
      "field. %s"),
      if (is_svc_fit) "Spatially-varying" else "Temporally-varying",
      sel$backend, mode, hint), call. = FALSE)
  }

  # spatial_rsr()'s projection is applied only inside the two binomial
  # Polya-Gamma Gibbs kernels that read $rsr_projection
  # (cpp_pg_binomial_gibbs_rsr() on an adjacency, cpp_pg_binomial_gibbs_gp_rsr()
  # on an NNGP field): the spec keeps its underlying $type for every other
  # consumer, so nested_laplace / laplace / hmc / the other backends would read
  # that type and fit the PLAIN (unprojected) field while still reporting
  # $spatial$rsr = TRUE -- silently dropping the projection rather than fitting
  # it (gcol33/tulpa#792). Fail loudly instead; only an explicit or
  # auto-selected gibbs backend carries the projection.
  is_rsr_fit <- identical(tolower(spatial_type %||% ""), "rsr")
  if (is_rsr_fit && !identical(sel$backend, "gibbs")) {
    stop(sprintf(paste0(
      "spatial_rsr() is fit only by the binomial Polya-Gamma Gibbs sampler: ",
      "every other backend reads the underlying field's $type ('%s') and ",
      "would fit the plain, unprojected field. The selected backend '%s' ",
      "(mode = '%s') does not carry the RSR projection. Use mode = 'gibbs' ",
      "or 'auto'."),
      spatial_spec$type, sel$backend, mode), call. = FALSE)
  }

  # A continuous spatial field (gp / nngp / hsgp) plus a formula RE term turns
  # the nested fit into a multi-block prior, and the multi-block converter
  # behind nested_laplace (.nl_block_spec_for_cpp(), R/nested_laplace.R) has no
  # gp / nngp / hsgp arm -- only icar / bym2 / car_proper / rw1 / rw2 / ar1 /
  # iid / spde / tgmrf. `auto` already routes around this (feat$continuous_spatial_re
  # in auto_select_mode()); an EXPLICIT mode = "nested_laplace" bypasses that
  # selector entirely (it is itself a backend name), so it needs its own
  # front-door refusal here rather than the deep, post-125-cell-grid C++ error
  # this used to reach (gcol33/tulpa#794).
  if (identical(sel$backend, "nested_laplace") && has_re &&
      tolower(spatial_type %||% "") %in% .NL_FRONTDOOR_CONTINUOUS) {
    stop(sprintf(paste0(
      "spatial_gp() (%s) with a random-intercept term is not supported by ",
      "mode = 'nested_laplace': its multi-block converter carries no %s ",
      "arm. Use mode = 'laplace' (conditions the RE at `sigma_re`) or ",
      "mode = 'exact' (samples the RE jointly via the ModelData NUTS ",
      "sampler)."), spatial_type, spatial_type), call. = FALSE)
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
  # path via .bundle_to_re_list when an explicit conditional mode names it
  # (`mode = "laplace"` / `"mala"` / ...); auto routes them to the
  # covariance-integrating backend too (see auto_select_mode()),
  # so `re_terms` and `has_re` are computed once, above, ahead of mode
  # selection, and reused here.
  has_slope <- has_re &&
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
  # Zero inflation reaches the kernels through two different channels: the
  # spec-driven Laplace path carries it as a second process, the sampler paths
  # as the `logit_zi` callback argument. Any other backend would fit the model
  # WITHOUT the mixture and return a plausible non-zero-inflated answer, so
  # refuse rather than redirect silently -- a redirect would change the
  # requested inference method, and ignoring `ziformula` would change the model.
  if (!is.null(bundle$X_zi) && !sel$backend %in% .zi_backends()) {
    stop(sprintf(paste0(
      "`ziformula` is not carried by backend '%s', which would fit the model ",
      "without the zero-inflation component. Zero inflation is available for: ",
      "%s."), sel$backend, paste(.zi_backends(), collapse = ", ")),
      call. = FALSE)
  }

  # auto's RE arm picks a covariance integrator itself, so the redirect below --
  # which reads `control$re_cov` only off a conditional backend -- never saw the
  # knob there, and `control$re_cov = "nested"` under auto ran re_cov_gibbs
  # anyway. A caller who NAMES the integrator gets it on this path too, refused
  # rather than swapped when that integrator cannot carry the call.
  if (has_re && identical(sel$requested, "auto") && !is.null(control$re_cov) &&
      sel$backend %in% c("re_cov_gibbs", "re_cov_nested")) {
    re_cov_method <- .re_cov_method(control, "nested")
    want <- if (re_cov_method == "gibbs") "re_cov_gibbs" else "re_cov_nested"
    if (!identical(want, sel$backend)) {
      if (!.auto_backend_ok(want, fam_obj, call_feat)) {
        conflicting <- names(call_feat)[vapply(call_feat, isTRUE, logical(1))]
        stop(sprintf(paste0(
          "control$re_cov = '%s' names %s, which does not carry this call's ",
          "feature(s) (%s). Drop control$re_cov to let auto pick the ",
          "integrator that does."), re_cov_method, want,
          paste(conflicting, collapse = ", ")), call. = FALSE)
      }
      sel <- .sel_redirect(sel, want, sprintf(
        "control$re_cov = '%s' requested; RE covariance(s) integrated via %s",
        re_cov_method, want), notify = FALSE)
    }
  }

  slope_scalar_backends <- c("laplace", "mala", "pathfinder", "imh_laplace")
  # A slope term MUST have its covariance integrated -- there is no scalar
  # sigma_re to condition on -- and a caller who NAMES an integrator gets one
  # whatever the term shape. `control$re_cov` used to be read only in the slope
  # case, so on a `(1 | g)` model any value, including a typo, was accepted with
  # no effect and the fit silently conditioned at sigma_re = 1
  # (gcol33/tulpa#668). The default path is unchanged: an unset knob still
  # redirects only for a slope.
  if (has_re && (has_slope || !is.null(control$re_cov)) &&
      sel$backend %in% slope_scalar_backends) {
    default_re_cov <- if (sel$backend == "laplace") "nested" else "gibbs"
    re_cov_method <- .re_cov_method(control, default_re_cov)
    backend <- if (re_cov_method == "gibbs") "re_cov_gibbs" else "re_cov_nested"
    # An auto-driven redirect (not a caller NAMING control$re_cov) picked
    # `backend` from sel$backend's shape alone, ignoring the call's features --
    # auto's own RE arm (auto_select_mode()) already tried BOTH covariance
    # integrators against these features and fell through to a scalar backend
    # only because neither carries the call (e.g. weights, which neither
    # re_cov_gibbs nor re_cov_nested threads). Redirecting there anyway just
    # traded the scalar backend's silent sigma_re = 1 conditioning for that
    # backend's own confusing refusal (gcol33/tulpa#769). Try the other
    # integrator, and only if it also refuses does this refuse -- naming both,
    # since a slope term categorically cannot fit through mala / laplace /
    # pathfinder / imh_laplace either.
    if (has_slope && !isTRUE(sel$explicit) &&
        !.auto_backend_ok(backend, fam_obj, call_feat)) {
      other <- if (backend == "re_cov_gibbs") "re_cov_nested" else "re_cov_gibbs"
      if (.auto_backend_ok(other, fam_obj, call_feat)) {
        backend <- other
        re_cov_method <- if (other == "re_cov_gibbs") "gibbs" else "nested"
      } else {
        conflicting <- names(call_feat)[vapply(call_feat, isTRUE, logical(1))]
        stop(sprintf(paste0(
          "auto: this random-slope term has no scalar `sigma_re` to condition ",
          "on, so its covariance must be integrated via re_cov_gibbs or ",
          "re_cov_nested -- but neither carries this call's feature(s) (%s). ",
          "Use an explicit mode naming a backend that carries it, or drop the ",
          "conflicting feature."), paste(conflicting, collapse = ", ")),
          call. = FALSE)
      }
    }
    # notify = FALSE: a slope term has no scalar `sigma_re` for the requested
    # conditional mode to condition on, so this is the documented route for the
    # structure rather than a capability taken away. Recorded on the fit, not
    # warned about on every such fit.
    why <- if (has_slope) "random-slope term(s) present"
           else sprintf("control$re_cov = '%s' requested", re_cov_method)
    sel <- .sel_redirect(sel, backend, sprintf(
      "%s; RE covariance(s) integrated via %s (%d block(s))",
      why, backend, length(re_terms)), notify = FALSE)
  }
  # Warn once whenever the fit DETERMINES the RE scale itself -- by integrating
  # it (re_cov_nested / re_cov_gibbs, reached via the redirect above or by
  # name), by sampling it (gibbs, and every ModelData sampler, which put
  # log_sigma_re in the latent vector) or by maximizing over it (eb, agq) -- and
  # a scalar `sigma_re` was also supplied, since it is silently unused there.
  # `?tulpa` promised this warning for all of them and only two routes gave it
  # (gcol33/tulpa#669); the list is the registry-derived one so the doc and the
  # code have a single referent.
  if (!is.null(sigma_re) && has_re &&
      sel$backend %in% .re_scale_estimating_backends()) {
    verb <- switch(sel$backend,
                   eb = , agq = "estimated",
                   re_cov_nested = , re_cov_gibbs = "integrated",
                   "sampled")
    warning(sprintf(paste0(
      "`sigma_re` is ignored for mode = '%s': the RE scale is %s, not ",
      "conditioned on a scalar SD. Drop `sigma_re`, or use mode = 'laplace' to ",
      "condition on it."), sel$backend, verb), call. = FALSE)
  }

  # The exact-logpost backends (mala / pathfinder / imh_laplace) build their
  # target from build_glmm_logpost(), a fixed-effect + scalar-RE log-posterior
  # with no spatial term: a spatial field reaching one of them is silently
  # absent from eta rather than refused (gcol33/tulpa#791). Redirect to
  # nested-Laplace the same way a temporal field is redirected below, so the
  # field is fit rather than dropped; the SVC guard above has already refused
  # an svc field on these backends, and the SPDE / multi-block checks that
  # follow still apply to the redirected selection.
  if (has_spatial &&
      identical(BACKEND_REGISTRY[[sel$backend]]$input %||% "", "logpost")) {
    sel <- .sel_redirect(sel, "nested_laplace", sprintf(
      "%s spatial field; nested-Laplace integration (mode = '%s' carries no spatial term)",
      spatial_type, mode), notify = TRUE)
  }

  # SPDE carries its own nested-Laplace integration engine: fit_spde() rebuilds
  # the Matern precision Q(range, sigma) per node via the FEM Q-builder and
  # integrates (range, sigma) with a CCD / grid design in R -- not the generic
  # registry grid that tulpa_nested_laplace drives. Every nested mode (auto,
  # structured, or the nested_laplace backend by name) selects the generic
  # nested_laplace backend for a spatial field; redirect that to the dedicated
  # `spde` backend so the SPDE field reaches its own integrator. The conditional
  # mode = "laplace" stays on the fixed-hyperparameter tulpa_laplace path.
  #
  # ... EXCEPT with a random-effect term. fit_spde()'s grid has no RE-SD axis;
  # it takes a scalar `sigma_re` and conditions on it, which at the default of
  # 1 makes the RE the one variance component a nested fit never estimates --
  # a number the data never produced, on the path whose whole point is
  # integrating the hyperparameters (gcol33/tulpa#817). The generic nested
  # driver already carries an `spde` block type, so the field goes there as a
  # block beside the RE's own `iid` block and both SDs are integrated on one
  # outer grid. Integer nu only: fractional nu is the operator-based rational
  # construction fit_spde() owns, and it refuses an RE term regardless.
  spde_re_integrated <- isTRUE(has_re) &&
    !.spde_nu_is_fractional(spatial_spec$nu %||% 1)
  if (sel$backend == "nested_laplace" &&
      identical(tolower(spatial_type %||% ""), "spde") &&
      !spde_re_integrated) {
    # notify = FALSE: same mode and tier, reaching the engine that carries the FEM
    # precision. Nothing the caller asked for is lost.
    sel <- .sel_redirect(
      sel, "spde",
      "SPDE spatial field; nested-Laplace over (range, sigma) via fit_spde()",
      notify = FALSE)
  }

  # A temporal field (rw1/rw2/ar1) integrates through the nested-Laplace temporal
  # kernel; with a spatial field present it joins a [spatial, temporal] joint
  # prior. The auto / structured / conditional-Laplace selections route a
  # temporal field there (the conditional mode = "laplace" is not wired for
  # temporal yet), mirroring and superseding the spatial-field redirect above
  # when both fields are present. An explicitly chosen ModelData sampler backend
  # (hmc / ess / sghmc / sgld / mclmc / smc / vi) consumes the temporal field
  # directly, so it keeps its selection rather than being redirected.
  #
  # `notify` follows whether a TIER was lost, not just a backend name: a
  # Tier-2 request (mode = "laplace") loses nothing here -- the conditional
  # Laplace path carries no temporal kernel, so nested_laplace is the
  # documented route for the structure rather than a capability taken away,
  # and stays silent. An explicit Tier-1 request (gibbs, mala, imh_laplace,
  # pathfinder, ...) DOES lose its tier -- it is redirected to a Tier-2
  # approximation instead of the exact sampler it asked for -- and that is
  # exactly the case the front door's override contract warns about
  # (gcol33/tulpa#768).
  if (has_temporal && BACKEND_REGISTRY[[sel$backend]]$input != "modeldata") {
    sel <- .sel_redirect(sel, "nested_laplace", if (has_spatial) {
      sprintf("%s spatial field + temporal %s field; joint nested-Laplace integration",
              spatial_type, temporal_spec$type)
    } else {
      sprintf("temporal %s field; nested-Laplace integration", temporal_spec$type)
    }, notify = isTRUE(sel$tier == 1L))
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
    if (identical(hyperprior, "flat")) {
      stop("`hyperprior = \"flat\"` is not read by the SPDE NUTS engine, which ",
           "samples (range, sigma) under the field's PC priors. Use ",
           "mode = 'laplace' / 'auto' for the nested SPDE path.", call. = FALSE)
    }
    # The SPDE NUTS engine runs one chain; a caller-supplied n_chains is
    # refused rather than silently ignored (`.control_subset()` below would
    # otherwise drop it, since it is not in the nuts_spde key set at all).
    if (!is.null(control$n_chains) && as.integer(control$n_chains) != 1L) {
      stop("`control$n_chains` is not supported on the SPDE Tier-1 (exact ",
           "NUTS) route -- it runs a single chain. Drop it, or run several ",
           "fits at different seeds and combine the draws.", call. = FALSE)
    }
    # This branch reaches ANY explicit Tier-1 backend under an SPDE field
    # (gibbs, mala, ess, ...), not only the natural mode = 'exact' / 'hmc'
    # route that maps to it. A request NAMING one of the others asked for an
    # algorithm this field has no implementation of, so it is refused rather
    # than run as NUTS under a warning (gcol33/tulpa#912; the warning was
    # gcol33/tulpa#768, and a script that suppresses warnings got a different
    # sampler in silence). A TIER request ("exact") promised a tier, not an
    # algorithm, and that promise holds here, so it keeps the recorded override.
    # The natural route is not an override at all, since nothing was lost: it
    # is the same exact-NUTS tier reaching its own field-specific engine.
    if (isTRUE(sel$explicit) && identical(sel$requested, sel$backend) &&
        !identical(sel$backend, "hmc")) {
      stop(sprintf(paste0(
        "mode = '%s' has no implementation for an SPDE field. Its exact ",
        "(Tier-1) sampler is NUTS over the Matern field and hyperparameters: ",
        "pass mode = 'exact' or 'hmc' for it, or mode = 'auto' / ",
        "'nested_laplace' for the nested-Laplace SPDE path."), sel$backend),
        call. = FALSE)
    }
    sel <- .sel_redirect(sel, "spde",
      "SPDE field, Tier-1 mode: exact NUTS over the Matern field + hyperparameters",
      notify = !identical(sel$backend, "hmc"))
    if (!is.null(sel$overridden) && isTRUE(sel$overridden$notify)) {
      warning(sprintf(paste0(
        "mode = '%s' was overridden: fitted with backend 'spde' instead of ",
        "'%s' -- %s. Pass mode = 'exact' to request the SPDE NUTS engine ",
        "directly, or mode = 'auto' to fit without asking for a mode."),
        sel$overridden$requested, sel$overridden$backend, sel$reason),
        call. = FALSE)
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
    fit$mode_overridden <- sel$overridden
    fit$N <- fit$N %||% bundle$n_obs
    fit$model_matrix <- fit$model_matrix %||% bundle$X
    fit$y <- fit$y %||% bundle$y
    return(.finalize_fit(fit, backend = "spde", draws_kind = "chain",
                         n_fixed = ncol(bundle$X),
                         fixed_names = colnames(bundle$X)))
  }

  # An explicit `mode` that the redirects above moved off is reported, never
  # silently downgraded: the caller asked for a named inference method and got a
  # different one because the model structure requires it. A `warning()` rather
  # than a `message()` so a script that promotes warnings, or a chunk that traps
  # them, actually sees it; `sel$reason` (and so `fit$selection_reason`) carries
  # the same statement for anyone reading the fit afterwards.
  # The reason states WHY the structure forced the move (and differs per
  # redirect: a smoother is only threaded through the nested kernels, while a
  # random slope has no scalar SD to condition on), so the warning states the
  # fact and defers the cause to it rather than asserting one of its own.
  if (!is.null(sel$overridden) && isTRUE(sel$overridden$notify)) {
    warning(sprintf(paste0(
      "mode = '%s' was overridden: fitted with backend '%s' instead of '%s' -- ",
      "%s. Pass mode = '%s' to request that directly, or mode = 'auto' to fit ",
      "without asking for a mode."),
      sel$overridden$requested, sel$backend, sel$overridden$backend,
      sel$reason, sel$backend), call. = FALSE)
  }

  assert_backend_reachable(sel$backend)

  # The dispersion is estimated by the empirical-Bayes outer maximization, the
  # one path that carries log(phi) as a coordinate of the outer objective.
  # Every other backend takes `phi` as a value to condition on, so honouring
  # the argument there would mean fitting at the starting value while the fit
  # reported an estimate.
  if (isTRUE(estimate_phi) && !identical(sel$backend, "eb")) {
    stop("`estimate_phi = TRUE` is available under mode = 'eb'; ",
         "mode resolved to '", sel$backend, "', which conditions on `phi`.",
         call. = FALSE)
  }

  # Conditional backends (everything except the sigma-sampling Gibbs, the
  # Sigma-integrating re_cov backends, the Sigma-maximizing EB fit, the
  # marginal-likelihood AGQ fit that estimates the RE sd, and the ModelData
  # samplers that draw the RE sd jointly) need one RE sd per term to condition
  # on; resolve/recycle it after the backend is known so the others do not emit
  # a misleading "conditioning" message.
  # `nested_laplace` is in the exempt list because it no longer conditions: each
  # RE term becomes an `iid` latent block whose SD is integrated on the outer grid
  # alongside the other blocks' hyperparameters. A `sigma_re` supplied
  # explicitly still conditions there, via the one-point grid the iid registry
  # entry documents, so it is passed through rather than defaulted here.
  # A `warning()`, not a `message()`, so a script that promotes warnings (or a
  # chunk that traps them) actually sees that a variance component was fixed
  # rather than estimated.
  # The resolved SDs are kept for the fit (`$sigma_re_conditioned`), so
  # VarCorr() / print() report the value the fit conditioned on rather than
  # re-evaluating the call in a frame that is not the caller's
  # (gcol33/tulpa#868).
  sigma_re_conditioned <- NULL
  if (K > 0L &&
      !sel$backend %in% c("gibbs", "re_cov_nested", "re_cov_gibbs", "eb", "agq",
                          "nested_laplace") &&
      BACKEND_REGISTRY[[sel$backend]]$input != "modeldata") {
    if (is.null(sigma_re)) {
      sigma_re <- rep(1, K)
      # A tier mode (auto / structured) integrates an unsupplied scale wherever
      # a backend carries the call (gcol33/tulpa#787), so landing on a
      # conditional backend from one is a change of estimand the caller did not
      # ask for; name what caused it rather than leave the generic line
      # (gcol33/tulpa#874).
      req <- sel$requested %||% ""
      why <- ""
      if (nzchar(req) && !req %in% ALL_BACKENDS) {
        on_feat <- names(call_feat)[vapply(call_feat, isTRUE, logical(1))]
        why <- sprintf(paste0(
          " mode = '%s' integrates an unsupplied RE scale where a backend ",
          "carries the call, but it resolved to '%s'%s, which conditions."),
          req, sel$backend,
          if (length(on_feat)) sprintf(
            " (no scale-integrating backend carries this call's %s)",
            paste(on_feat, collapse = ", ")) else "")
      }
      warning("tulpa(): `sigma_re` not supplied; conditioning on sigma_re = 1 for ",
              "each of the ", K, " RE term(s).", why, " Pass `sigma_re` to ",
              "override.", call. = FALSE)
    } else if (length(sigma_re) == 1L) {
      sigma_re <- rep(sigma_re, K)
    } else if (length(sigma_re) != K) {
      stop(sprintf("`sigma_re` must have length 1 or %d (one per RE term).", K),
           call. = FALSE)
    }
    sigma_re_conditioned <- sigma_re
  } else if (K > 0L && identical(sel$backend, "nested_laplace") &&
             !is.null(sigma_re)) {
    # An explicit sigma_re conditions the nested path too (a one-point grid).
    sigma_re_conditioned <- if (length(sigma_re) == 1L) rep(sigma_re, K)
                            else sigma_re
  }

  # The same sentence for the DISPERSION, which had none (gcol33/tulpa#849).
  # An unsupplied `phi` conditions at the signature's 1.0, which for a gaussian
  # is a residual variance the data usually contradicts -- measured 1 against a
  # truth of 0.2025 -- and `posterior_predict()` / `bayes_R2()` / WAIC all read
  # it. A caller who did not pass `phi` did not choose 1; nothing said there was
  # a choice. Raised only for the families that READ a dispersion, so binomial
  # and poisson stay quiet, and not when `estimate_phi` is on, where the value
  # is a starting point rather than a conditioning one.
  if (!phi_supplied && !isTRUE(estimate_phi) &&
      .family_base(family) %in% .PHI_FAMILIES) {
    warning("tulpa(): `phi` not supplied; conditioning on phi = 1 for ",
            "family = '", family, "'. Pass `phi` to override, or ",
            "`estimate_phi = TRUE` with mode = 'eb' to estimate it.",
            call. = FALSE)
  }

  # Resolved before backend dispatch; `beta_prior` itself stays as supplied, so
  # the branches that reject a fixed-effect prior still see NULL when none was.
  beta_prior_resolved <- beta_prior %||% .tulpa_default_beta_prior()

  control <- .auto_drop_unread_threads(control, sel)

  args <- .tulpa_fitter_args(sel$backend, bundle, family, sigma_re,
                             n_trials, phi, beta_prior, control,
                             latent_blocks = parsed$latent_blocks,
                             spatial = spatial_spec, temporal = temporal_spec,
                             weights = weights, phi2 = phi2,
                             smoothers = lapply(smooth_specs, `[[`, "block"),
                             re_prior = re_prior, zi_prior = zi_prior,
                             hyperprior = hyperprior,
                             warm_start = warm_start,
                             estimate_phi = estimate_phi,
                             beta_prior_default = beta_prior_resolved)

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
    # Machine-readable counterpart of the reason's override clause: the mode the
    # caller asked for and the backend it would have used, or NULL when the fit
    # ran the requested method (or was never asked for one).
    fit$mode_overridden <- sel$overridden
    fit$formula <- formula
    fit$family <- family
    fit$beta_prior <- .beta_prior_applied(args, beta_prior_resolved)
    fit$sigma_re_conditioned <- sigma_re_conditioned
    fit$call <- match.call()
    # The laplace branch asks for the joint precision to keep `H_latent`
    # (gcol33/tulpa#871); the kernel-curvature copy that comes with it is read
    # by nothing on a front-door fit.
    if (identical(sel$backend, "laplace")) fit$H_joint <- NULL

    # Canonical parameter layout for the S3 accessors: the fixed-effect count and
    # names plus the [fixed, random] name vector both posterior shapes share, so
    # coef()/summary()/ranef() report real names and a fixed/random split.
    layout <- .tulpa_param_layout(bundle)
    fit$n_fixed     <- layout$n_fixed
    fit$fixed_names <- layout$fixed_names
    fit <- .finalize_fit(fit, param_names = layout$param_names)
    fit$re_layout   <- layout$re_layout
    # Nested path only: the RE terms were carried as `iid` latent blocks whose SD
    # the outer grid integrated, so VarCorr() reads those blocks' posterior and
    # ranef() slices their latent segment (the LAST sum(n_groups) columns, the RE
    # blocks having been appended last). Absent on every other backend.
    fit$re_block_index       <- attr(args, "re_block_index")
    fit$re_block_conditioned <- attr(args, "re_block_conditioned")
    fit$N           <- fit$N %||% bundle$n_obs
    # Fixed-effect design for fitted()/predict(newdata = NULL), plus the pieces
    # posterior_predict() needs to rebuild the in-sample linear predictor and
    # push it through the family sampler: offset, response, trials, dispersion,
    # and the per-term RE row design (group index + slope columns).
    fit$model_matrix <- bundle$X
    # The zero-inflation predictor's design and the formula that rebuilds it
    # at `newdata`; both absent on a fit without `ziformula`.
    fit$ziformula        <- if (!is.null(bundle$X_zi)) ziformula
    fit$zi_model_matrix  <- bundle$X_zi
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
    # The temporal spec rides along for the same reason, and for the
    # accessors: temporal() / tvc() read the field layout (time levels, term
    # names, components) off the fit, and the draw columns the sampler emits
    # -- trend[k], tvc_w[u] -- carry no layout of their own.
    if (!is.null(temporal_spec)) fit$temporal <- fit$temporal %||% temporal_spec
    # Smoother metadata for smooth_effects(): node locations plus the block
    # sizes needed to index the latent tail of the per-grid modes.
    if (length(smooth_specs) > 0L) {
      fit$smooth_terms    <- lapply(smooth_specs, `[[`, "meta")
      fit$n_latent_blocks <- parsed$n_latent_blocks %||% 0L
    }
  }
  # A chain that has not mixed is flagged here, where every sampler door
  # returns, rather than only by a diagnostic the caller has to think to run
  # (gcol33/tulpa#875, #878).
  .tulpa_check_fit_convergence(fit)
}
