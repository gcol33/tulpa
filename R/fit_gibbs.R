# Polya-Gamma Gibbs sampler dispatch: drives the cpp_pg_binomial_gibbs_* /
# cpp_pg_negbin_gibbs spatial and temporal samplers -- shared GLMM argument
# assembly, per-structure input marshalling, the spatial / temporal dispatchers,
# and the tulpa_gibbs() front door.

# Neighbor-list form of an adjacency matrix for the Polya-Gamma Gibbs spatial
# samplers. They take `adj_list` as an R list whose j-th element is the 1-based
# neighbor indices of unit j (the C++ subtracts 1 internally), plus the
# `n_neighbors` count -- in contrast to the CSR row_ptr/col_idx form
# adjacency_to_csr_tulpa() builds for the Laplace kernels.
adjacency_to_list_tulpa <- function(adj) {
  if (inherits(adj, "sparseMatrix")) adj <- as.matrix(adj)
  n <- nrow(adj)
  adj_list <- lapply(seq_len(n), function(i) as.integer(which(adj[i, ] != 0)))
  list(adj_list = adj_list,
       n_neighbors = vapply(adj_list, length, integer(1)))
}

# Shared `cpp_pg_binomial_gibbs_*` argument block (the universal GLMM inputs
# every spatial / temporal Polya-Gamma sampler takes: response, design, the iid
# RE block, iteration controls and the fixed-effect / RE-scale priors). Factored
# out so dispatch_gibbs_spatial() and dispatch_gibbs_temporal() assemble it once
# (principle #5) and the per-sampler do.call only adds the structure inputs.
.pg_gibbs_common_args <- function(y, n_trials, X, re_group, n_re_groups,
                                  iter, warmup, thin,
                                  prior_beta_sd, prior_sigma_re_scale,
                                  verbose, n_threads) {
  list(
    y = as.numeric(y), n = as.integer(n_trials), X = X,
    re_group = as.integer(re_group), n_re_groups = as.integer(n_re_groups),
    n_iter = as.integer(iter), n_warmup = as.integer(warmup),
    thin = as.integer(thin),
    prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_re_scale,
    store_eta = FALSE, verbose = verbose, n_threads = as.integer(n_threads)
  )
}

# Pull the (nn_idx, nn_dist, nn_order, nn) tuple a Polya-Gamma NNGP Gibbs
# sampler consumes out of a validated GP spec's `neighbor_info`. nn_order is
# stored 1-based (ordered-position -> original location index); the C++ kernels
# index 0-based, so subtract 1 -- the same convention as laplace_gp_at(). nn_idx
# / nn_dist are passed through (1-based ordered-position neighbour ids).
.gp_gibbs_nn_inputs <- function(neighbor_info, n_spatial) {
  list(
    nn_idx   = as.matrix(neighbor_info$nn_idx),
    nn_dist  = as.matrix(neighbor_info$nn_dist),
    nn_order = as.integer((neighbor_info$nn_order %||% seq_len(n_spatial)) - 1L),
    nn       = as.integer(ncol(neighbor_info$nn_idx))
  )
}

# The continuous-field Polya-Gamma Gibbs samplers (gp / multiscale_gp) carry NO
# observation->location index: they map observation row i straight to spatial
# location i (`gp_contrib[i] = w[i]` for `i < n_spatial`). So they require one
# observation per unique location, in coordinate order. Surface that constraint
# loudly rather than silently fitting the trailing observations with no field.
# (Flagged for the post-recovery-net refactor that would add an obs->loc map;
# see dev_notes/plan_gibbs_spatial_frontdoor.md.)
.gp_gibbs_require_one_obs_per_loc <- function(spatial, n_obs, label) {
  n_spatial  <- spatial$n_spatial %||% nrow(spatial$unique_coords)
  obs_to_loc <- as.integer(spatial$obs_to_loc %||% seq_len(n_obs))
  if (n_spatial != n_obs || !identical(obs_to_loc, seq_len(n_obs))) {
    stop("The binomial ", label, " Gibbs sampler maps observation i directly to ",
         "location i (it carries no observation->location index), so it needs ",
         "one observation per unique location in coordinate order. Got ", n_obs,
         " observation(s) for ", n_spatial, " location(s). Use mode = 'nested' ",
         "or 'laplace' for repeated-location designs.", call. = FALSE)
  }
  as.integer(n_spatial)
}

# Unit-level RSR projector P_perp = I - Q Q', orthogonalising the spatial field
# against the (unit-aggregated) fixed-effect design so the field cannot absorb
# covariate signal (Reich et al. 2006). X is per observation; collapse it to one
# row per spatial unit (mean over each unit's observations -- the identity for
# the canonical one-observation-per-unit areal design) before projecting.
.rsr_unit_projection <- function(X, spatial_group, n_units) {
  X <- as.matrix(X)
  X_unit <- matrix(0.0, n_units, ncol(X))
  cnt    <- integer(n_units)
  for (i in seq_len(nrow(X))) {
    u <- spatial_group[i]
    X_unit[u, ] <- X_unit[u, ] + X[i, ]
    cnt[u]      <- cnt[u] + 1L
  }
  pos <- cnt > 0
  X_unit[pos, ] <- X_unit[pos, ] / cnt[pos]
  compute_rsr_projection(X_unit)
}

#' Dispatch a spatial Polya-Gamma Gibbs fit to the correct sampler
#'
#' The Gibbs analogue of [dispatch_laplace_spatial()]: routes on `spatial$type`
#' to the matching `cpp_pg_<family>_gibbs_<structure>` sampler, building the
#' neighbour-list / coordinate inputs each one needs. The binomial Polya-Gamma
#' augmentation backs the full areal (icar/bym2/rsr) + continuous (gp/nngp/
#' multiscale_gp) family; `neg_binomial_2` is backed by the single areal ICAR
#' negbin sampler (`cpp_pg_negbin_gibbs_spatial`), the only negbin spatial kernel.
#' @keywords internal
dispatch_gibbs_spatial <- function(y, n_trials, X, re_group, n_re_groups,
                                   spatial, family,
                                   iter, warmup, thin = 1L,
                                   prior_beta_sd = 10.0,
                                   prior_sigma_re_scale = 2.5,
                                   verbose = FALSE, n_threads = 1L) {
  if (!family %in% c("binomial", "neg_binomial_2")) {
    stop("Spatial Gibbs supports family 'binomial' or 'neg_binomial_2'; got '",
         family, "'. Use mode = 'laplace' for other families under a spatial field.",
         call. = FALSE)
  }
  # RSR keeps the underlying areal $type (icar/car) and flags $rsr; normalise it
  # to the "rsr" route so the projection is applied rather than a plain areal fit.
  spatial_type <- tolower(spatial$type %||% "")
  if (isTRUE(spatial$rsr)) spatial_type <- "rsr"

  # Areal samplers (icar / bym2 / rsr) share the neighbour-list block; the negbin
  # areal kernel below reuses the same block.
  if (spatial_type %in% c("icar", "bym2", "rsr")) {
    adj <- spatial$adjacency
    if (is.null(adj)) {
      stop("Spatial Gibbs (", spatial_type, ") needs `spatial$adjacency`.",
           call. = FALSE)
    }
    al <- adjacency_to_list_tulpa(adj)
    areal <- list(
      spatial_group   = as.integer(spatial$spatial_idx %||% seq_len(nrow(adj))),
      n_spatial_units = as.integer(nrow(adj)),
      adj_list        = al$adj_list,
      n_neighbors     = al$n_neighbors
    )
  }

  # Negative-binomial spatial Gibbs: a single areal ICAR kernel. No bym2/rsr/gp
  # negbin samplers exist, so a non-ICAR field under negbin is rejected rather
  # than silently downgraded. The kernel carries no trial count `n`, adds the
  # dispersion-r prior, and shares the iid RE block + ICAR neighbour list.
  if (identical(family, "neg_binomial_2")) {
    if (spatial_type != "icar") {
      stop("Negative-binomial spatial Gibbs is wired for the areal ICAR field ",
           "only; got '", spatial_type, "'. Use family = 'binomial' for ",
           "bym2 / rsr / gp fields, or mode = 'laplace'.", call. = FALSE)
    }
    return(cpp_pg_negbin_gibbs_spatial(
      y = as.integer(y), X = X,
      re_group = as.integer(re_group), n_re_groups = as.integer(n_re_groups),
      spatial_group = areal$spatial_group, n_spatial_units = areal$n_spatial_units,
      adj_list = areal$adj_list, n_neighbors = areal$n_neighbors,
      n_iter = as.integer(n_iter), n_warmup = as.integer(warmup),
      thin = as.integer(thin),
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_re_scale,
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01,
      prior_r_shape   = spatial$prior_r_shape %||% 1.0,
      prior_r_rate    = spatial$prior_r_rate  %||% 0.1,
      r_init          = spatial$r_init %||% 5.0,
      store_eta = FALSE, verbose = verbose, n_threads = as.integer(n_threads)
    ))
  }

  # Binomial spatial Gibbs: the full areal + continuous sampler family. The
  # shared GLMM input block (with the binomial trial count `n`) is assembled once.
  common <- .pg_gibbs_common_args(
    y, n_trials, X, re_group, n_re_groups, iter, warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, verbose, n_threads
  )

  if (spatial_type == "icar") {
    do.call(cpp_pg_binomial_gibbs_spatial, c(common, areal, list(
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    )))
  } else if (spatial_type == "bym2") {
    do.call(cpp_pg_binomial_gibbs_bym2, c(common, areal, list(
      scale_factor              = spatial$scale_factor %||% 1.0,
      prior_sigma_spatial_scale = spatial$prior_sigma_spatial_scale %||% 2.5,
      prior_rho_alpha           = spatial$prior_rho_alpha %||% 0.5,
      prior_rho_beta            = spatial$prior_rho_beta  %||% 0.5
    )))
  } else if (spatial_type == "rsr") {
    # Restricted spatial regression: an ICAR field projected orthogonal to the
    # covariates each sweep. Reuse a caller-supplied projector if validate_rsr()
    # built one, else assemble the unit-level P_perp from the model design.
    rsr_n  <- areal$n_spatial_units
    P_perp <- spatial$rsr_projection %||%
      .rsr_unit_projection(X, areal$spatial_group, rsr_n)
    if (nrow(P_perp) != rsr_n || ncol(P_perp) != rsr_n) {
      stop("RSR projection matrix is ", nrow(P_perp), "x", ncol(P_perp),
           " but must be ", rsr_n, "x", rsr_n, " (one row/col per spatial unit).",
           call. = FALSE)
    }
    do.call(cpp_pg_binomial_gibbs_rsr, c(common, areal, list(
      # Row-major flatten for the C++ `rsr_projection[s * rsr_n + k]` indexing;
      # P_perp is symmetric so t() is a no-op but keeps the convention explicit.
      rsr_projection  = as.numeric(t(P_perp)),
      rsr_n           = as.integer(rsr_n),
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    )))
  } else if (spatial_type %in% c("gp", "nngp")) {
    if (is.null(spatial$neighbor_info)) {
      stop("Spatial Gibbs (", spatial_type, ") needs a validated spatial_gp() ",
           "spec (neighbor_info is NULL). Call validate_gp(spatial, data) first, ",
           "or fit through tulpa() which validates automatically.", call. = FALSE)
    }
    n_spatial <- .gp_gibbs_require_one_obs_per_loc(spatial, length(y), spatial_type)
    nn_in     <- .gp_gibbs_nn_inputs(spatial$neighbor_info, n_spatial)
    do.call(cpp_pg_binomial_gibbs_gp, c(common, list(
      coords         = as.matrix(spatial$unique_coords),
      nn_idx         = nn_in$nn_idx,
      nn_dist        = nn_in$nn_dist,
      nn_order       = nn_in$nn_order,
      n_spatial      = n_spatial,
      nn             = nn_in$nn,
      sigma2_gp_init = spatial$sigma2_gp %||% 1.0,
      phi_gp_init    = spatial$phi_gp %||% 1.0,
      cov_type       = gp_cov_type_for_laplace(spatial)
    )))
  } else if (spatial_type %in% c("multiscale", "multiscale_gp")) {
    # Both scales reuse the shared NNGP kriging conditional
    # (tulpa::pg_nngp_conditional), so cov_type (exponential / Matern 3/2 / 5/2)
    # is honoured exactly as in the single-scale sampler.
    if (is.null(spatial$neighbor_info_local) ||
        is.null(spatial$neighbor_info_regional)) {
      stop("Spatial Gibbs (multiscale GP) needs a validated spatial_multiscale() ",
           "spec (neighbor_info_local / _regional are NULL). Call ",
           "validate_gp(spatial, data) first, or fit through tulpa().",
           call. = FALSE)
    }
    n_spatial <- .gp_gibbs_require_one_obs_per_loc(spatial, length(y), "multiscale GP")
    loc <- .gp_gibbs_nn_inputs(spatial$neighbor_info_local, n_spatial)
    reg <- .gp_gibbs_nn_inputs(spatial$neighbor_info_regional, n_spatial)
    rl  <- spatial$range_local    %||% c(0.01, 1)
    rr  <- spatial$range_regional %||% c(1, 10)
    do.call(cpp_pg_binomial_gibbs_multiscale_gp, c(common, list(
      coords               = as.matrix(spatial$unique_coords),
      nn_idx_local         = loc$nn_idx,
      nn_dist_local        = loc$nn_dist,
      nn_order_local       = loc$nn_order,
      nn_local             = loc$nn,
      nn_idx_regional      = reg$nn_idx,
      nn_dist_regional     = reg$nn_dist,
      nn_order_regional    = reg$nn_order,
      nn_regional          = reg$nn,
      n_spatial            = n_spatial,
      sigma2_local_init    = 1.0,
      phi_local_init       = mean(rl),
      sigma2_regional_init = 1.0,
      phi_regional_init    = mean(rr),
      cov_type             = gp_cov_type_for_laplace(spatial),
      prior_phi_local_lower    = rl[1], prior_phi_local_upper    = rl[2],
      prior_phi_regional_lower = rr[1], prior_phi_regional_upper = rr[2]
    )))
  } else {
    stop("Spatial Gibbs not wired for type '", spatial_type, "'. Supported: ",
         "icar, bym2, rsr, gp/nngp, multiscale_gp. ",
         "See dev_notes/plan_gibbs_spatial_frontdoor.md.", call. = FALSE)
  }
}


#' Dispatch a temporal Polya-Gamma Gibbs fit
#'
#' The temporal analogue of [dispatch_gibbs_spatial()]: maps a validated
#' [temporal_multiscale()] spec onto the multiscale temporal Polya-Gamma
#' sampler (`cpp_pg_binomial_gibbs_temporal`), which composes an additive
#' RW1 trend + cyclic-RW1 seasonal + AR1/IID short-term decomposition. Binomial
#' only. The C++ kernel implements an RW1 trend (rw2 is rejected here rather than
#' silently downgraded).
#' @keywords internal
dispatch_gibbs_temporal <- function(y, n_trials, X, re_group, n_re_groups,
                                    temporal, family,
                                    iter, warmup, thin = 1L,
                                    prior_beta_sd = 10.0,
                                    prior_sigma_re_scale = 2.5,
                                    verbose = FALSE, n_threads = 1L) {
  if (!identical(family, "binomial")) {
    stop("Temporal Gibbs supports family = 'binomial' only; got '", family,
         "'. Use mode = 'laplace' for other families under a temporal field.",
         call. = FALSE)
  }
  if (is.null(temporal$time_index) || is.null(temporal$n_times)) {
    stop("Temporal Gibbs needs a validated temporal_multiscale() spec ",
         "(time_index / n_times are NULL). Call ",
         "validate_temporal_multiscale(temporal, data) first, or fit through ",
         "tulpa().", call. = FALSE)
  }

  trend_type <- switch(temporal$trend %||% "none",
    none = 0L, rw1 = 1L,
    rw2  = stop("The temporal Gibbs sampler implements an RW1 trend; rw2 is not ",
                "wired. Use trend = 'rw1', or mode = 'laplace' for an RW2 trend.",
                call. = FALSE),
    stop("Unknown temporal trend '", temporal$trend, "'.", call. = FALSE))
  short_type <- switch(temporal$short_term %||% "none",
    none = 0L, ar1 = 1L, iid = 2L,
    stop("Unknown temporal short_term '", temporal$short_term, "'.", call. = FALSE))

  common <- .pg_gibbs_common_args(
    y, n_trials, X, re_group, n_re_groups, iter, warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, verbose, n_threads
  )
  do.call(cpp_pg_binomial_gibbs_temporal, c(common, list(
    time_idx        = as.integer(temporal$time_index),
    n_times         = as.integer(temporal$n_times),
    seasonal_period = as.integer(temporal$seasonal %||% 0L),
    trend_type      = trend_type,
    short_type      = short_type
  )))
}


#' Compute GLM working weights for Laplace Hessian
#'
#' Thin wrapper over the family-ops registry ([family_weight()]) so the weight
#' formulas live in exactly one place (`R/family_loglik.R`). Unknown families
#' fall back to unit weights, preserving the historical permissive behaviour.
#' @keywords internal
glmm_weights <- function(eta, family, n_trials = NULL, phi = 1.0, phi2 = NULL) {
  if (is.null(.FAMILY_OPS[[family]])) return(rep(1, length(eta)))
  as.numeric(family_weight(eta, family, n_trials, phi, phi2))
}


#' Fit via Polya-Gamma Gibbs sampler
#'
#' @description
#' Public API for PG Gibbs sampling. Used by model packages for
#' binomial and negative binomial GLMMs.
#'
#' @param y Response vector
#' @param n_trials Trial sizes (binomial)
#' @param X Design matrix
#' @param group Integer vector of group indices (1-based)
#' @param n_groups Number of groups
#' @param family Character: "binomial" or "neg_binomial_2"
#' @param n_iter Total iterations
#' @param warmup Warmup iterations
#' @param prior_beta_sd Prior SD for betas
#' @param prior_sigma_scale Prior scale for RE sigma
#' @param spatial Optional spatial spec. When supplied the fit routes to the
#'   matching spatial Polya-Gamma Gibbs sampler via [dispatch_gibbs_spatial()];
#'   `group`/`n_groups` are the iid random-effect block carried alongside the
#'   field. The full areal + continuous family is available for
#'   `family = "binomial"`; `family = "neg_binomial_2"` is backed by the areal
#'   ICAR negbin sampler only. Supported `type`s:
#'   * areal -- `"icar"`, `"bym2"`, `"rsr"`: a list with `type`, `adjacency` and
#'     a 1-based `spatial_idx` per observation (e.g.
#'     `list(type = "icar", adjacency = W, spatial_idx = unit)`). `"rsr"` reuses
#'     `spatial$rsr_projection` if present, else builds the unit-level projector
#'     from the design.
#'   * continuous -- `"gp"`/`"nngp"` (a validated [spatial_gp()] spec) and
#'     `"multiscale_gp"` (a validated [spatial_multiscale()] spec). These
#'     samplers carry no observation->location map, so they require one
#'     observation per unique location in coordinate order.
#' @param temporal Optional temporal spec: a validated [temporal_multiscale()]
#'   object. Routes to the multiscale temporal Polya-Gamma sampler via
#'   [dispatch_gibbs_temporal()] (binomial only; RW1 trend + cyclic seasonal +
#'   AR1/IID short-term). Cannot be combined with `spatial`.
#' @param verbose Print progress
#' @param n_threads Number of threads
#'
#' @return List with beta draws, RE draws, sigma_re draws (plus the spatial
#'   field draws when `spatial` is supplied)
#'
#' @export
tulpa_gibbs <- function(y, n_trials, X, group, n_groups,
                        family = "binomial",
                        n_iter = 2000L, warmup = 1000L,
                        prior_beta_sd = 10.0, prior_sigma_scale = 2.5,
                        spatial = NULL, temporal = NULL,
                        verbose = FALSE, n_threads = 1L) {

  # Spatial / temporal field present: route to the matching Polya-Gamma Gibbs
  # sampler (the Gibbs analogue of tulpa_laplace(spatial = ...)). `group` /
  # `n_groups` become the iid random-effect block carried alongside the field.
  # There is no joint spatial+temporal Gibbs sampler -- reject rather than
  # silently dropping one field.
  if (!is.null(spatial) && !is.null(temporal)) {
    stop("Combined spatial + temporal Gibbs is not available; supply only one ",
         "of `spatial` / `temporal`.", call. = FALSE)
  }
  if (!is.null(spatial)) {
    return(dispatch_gibbs_spatial(
      y = y, n_trials = if (is.null(n_trials)) rep(1L, length(y)) else n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      spatial = spatial, family = family,
      iter = n_iter, warmup = warmup,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    ))
  }
  if (!is.null(temporal)) {
    return(dispatch_gibbs_temporal(
      y = y, n_trials = if (is.null(n_trials)) rep(1L, length(y)) else n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      temporal = temporal, family = family,
      iter = n_iter, warmup = warmup,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    ))
  }

  if (family == "binomial") {
    cpp_pg_binomial_gibbs(
      y = as.numeric(y), n = as.integer(n_trials), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(n_iter), n_warmup = as.integer(warmup),
      thin = 1L,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    )
  } else if (family %in% c("neg_binomial_2", "negbin")) {
    cpp_pg_negbin_gibbs(
      y = as.numeric(y), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(n_iter), n_warmup = as.integer(warmup),
      thin = 1L,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_r_shape = 1.0, prior_r_rate = 0.1, r_init = 5.0,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    )
  } else {
    stop(sprintf("Gibbs not available for family '%s'", family), call. = FALSE)
  }
}
