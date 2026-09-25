# Polya-Gamma Gibbs sampler dispatch: drives the cpp_pg_binomial_gibbs_* /
# cpp_pg_negbin_gibbs spatial and temporal samplers -- shared GLMM argument
# assembly, per-structure input marshalling, the spatial / temporal dispatchers,
# and the tulpa_gibbs() front door.

#' Draw Polya-Gamma random variates
#'
#' Vectorized `PG(b, z)` draws via the Polson, Scott & Windle (2013) sampler
#' (`tulpa::rpg_vec()`), the kernel every Polya-Gamma Gibbs fitter in this
#' package draws its auxiliary weights from. Exposed as a door for consumer
#' packages fitting their own Polya-Gamma Gibbs models, which cannot reach an
#' RNG through `LinkingTo` the way a likelihood kernel would.
#'
#' @param b Integer vector of PG shape parameters (trial counts); one draw
#'   per element.
#' @param z Numeric vector of tilting parameters, the same length as `b`.
#' @return A numeric vector of `PG(b, z)` draws, the same length as `b`.
#' @export
#' @examples
#' tulpa_rpg(rep(1L, 5), rnorm(5))
tulpa_rpg <- function(b, z) {
  cpp_rpg(as.integer(b), as.numeric(z))
}

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

# Draw columns for the blocks a Polya-Gamma kernel returns, in the parameter
# naming the ModelData samplers give the same quantity (sampler_param_names() in
# src/sampler_model_data.h), so a Gibbs chain and an HMC chain on one model are
# read by the same accessors. Each layout names the kernel blocks one route adds,
# with the stem of their columns and the map from the kernel's scale to the
# sampler coordinate: a standard deviation, precision, variance, range or size on
# the log scale, a standard deviation reported as a variance through log(sd^2),
# a mixing weight on its logit, an AR1 correlation on (-1, 1) through
# logit((rho + 1) / 2). A matrix block becomes `stem[k]` columns, a vector block
# one `stem` column. `with` names the field a scalar scales: a kernel stores a
# scale for every component it supports, and one whose field has no columns was
# never sampled, so it contributes no column.
#
# A kernel block absent from its route's layouts (the BYM2 combined field, the
# RSR field before projection, the linear predictor) is a deterministic function
# of the columns kept and is not part of the sampled state.
.PG_DRAW_LAYOUT <- list(
  re = list(
    sigma_re = list(stem = "log_sigma_re", map = log, with = "re"),
    re       = list(stem = "re")
  ),
  icar = list(
    tau     = list(stem = "log_tau_spatial", map = log),
    spatial = list(stem = "phi_spatial")
  ),
  bym2 = list(
    sigma_spatial = list(stem = "log_sigma_spatial", map = log),
    rho           = list(stem = "logit_rho_bym2", map = stats::qlogis),
    phi_scaled    = list(stem = "phi_spatial"),
    theta         = list(stem = "theta_spatial")
  ),
  gp = list(
    sigma2_gp = list(stem = "log_sigma2_gp", map = log),
    phi_gp    = list(stem = "log_phi_gp", map = log),
    gp        = list(stem = "gp_w")
  ),
  multiscale_gp = list(
    sigma2_local    = list(stem = "log_sigma2_gp_local", map = log),
    phi_local       = list(stem = "log_phi_gp_local", map = log),
    w_local         = list(stem = "gp_local"),
    sigma2_regional = list(stem = "log_sigma2_gp_regional", map = log),
    phi_regional    = list(stem = "log_phi_gp_regional", map = log),
    w_regional      = list(stem = "gp_regional")
  ),
  temporal = list(
    sigma_trend    = list(stem = "log_sigma2_trend",
                          map = function(s) 2 * log(s), with = "trend"),
    trend          = list(stem = "trend"),
    sigma_seasonal = list(stem = "log_sigma2_seasonal",
                          map = function(s) 2 * log(s), with = "seasonal"),
    seasonal       = list(stem = "seasonal"),
    sigma_short    = list(stem = "log_sigma2_short",
                          map = function(s) 2 * log(s), with = "short_term"),
    short_term     = list(stem = "short_term")
  ),
  ar1_short = list(
    rho_short = list(stem = "logit_rho_short",
                     map = function(r) stats::qlogis((r + 1) / 2))
  ),
  negbin = list(
    r = list(stem = "log_phi", map = log)
  )
)

# One MCMC chain from a Polya-Gamma kernel's per-block draw storage: the
# `[n_saved x n_param]` draws matrix with the fixed effects first, then the
# random-intercept block and every block the route's `layout` keys name, plus the
# chain bookkeeping the chain accessors read (`chain_id`, `n_chains`) and the
# column means.
.pg_as_chain <- function(res, layout, X) {
  X <- as.matrix(X)
  fixed_names <- colnames(X) %||% paste0("beta[", seq_len(ncol(X)), "]")
  spec <- do.call(c, unname(.PG_DRAW_LAYOUT[c("re", layout)]))

  cols   <- list(as.matrix(res[["beta"]]))
  labels <- list(fixed_names)
  for (b in names(spec)) {
    s <- spec[[b]]
    v <- res[[b]]
    if (is.null(v)) {
      stop("The Polya-Gamma kernel returned no '", b, "' block for layout ",
           paste(layout, collapse = " + "), ".", call. = FALSE)
    }
    if (!is.null(s$with) && NCOL(res[[s$with]]) == 0L) next
    m <- as.matrix(v)
    if (ncol(m) == 0L) next
    if (!is.null(s$map)) m[] <- s$map(as.numeric(m))
    cols   <- c(cols, list(m))
    labels <- c(labels, list(if (is.matrix(v))
      sprintf("%s[%d]", s$stem, seq_len(ncol(m))) else s$stem))
  }
  draws <- do.call(cbind, cols)
  dimnames(draws) <- list(NULL, unlist(labels))
  out <- list(draws = draws, means = colMeans(draws),
              chain_id = rep(1L, nrow(draws)), n_chains = 1L,
              n_samples = nrow(draws), n_params = ncol(draws),
              param_names = colnames(draws))
  # The kernel's log joint density at each retained state, on the scale it
  # samples each quantity on. A kernel whose chain leaves no written density
  # invariant returns none, and the fit then carries none.
  if (!is.null(res[["log_prob"]])) out$log_prob <- as.numeric(res[["log_prob"]])
  out
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
.gp_gibbs_require_one_obs_per_loc <- function(spatial, n_obs, label) {
  n_spatial  <- spatial$n_spatial %||% nrow(spatial$unique_coords)
  obs_to_loc <- as.integer(spatial$obs_to_loc %||% seq_len(n_obs))
  if (n_spatial != n_obs || !identical(obs_to_loc, seq_len(n_obs))) {
    stop("The binomial ", label, " Gibbs sampler maps observation i directly to ",
         "location i (it carries no observation->location index), so it needs ",
         "one observation per unique location in coordinate order. Got ", n_obs,
         " observation(s) for ", n_spatial, " location(s). Use mode = ",
         "'nested_laplace' or 'laplace' for repeated-location designs.",
         call. = FALSE)
  }
  as.integer(n_spatial)
}

# RSR projector P_perp = I - Q Q' at the FIELD's own resolution, orthogonalising
# the spatial field against the aggregated fixed-effect design so the field
# cannot absorb covariate signal (Reich et al. 2006). X is per observation;
# collapse it to one row per field coordinate (mean over that coordinate's
# observations -- the identity for the canonical one-observation-per-coordinate
# design) before projecting.
#
# `obs_to_field` is the 1-based observation -> field-coordinate map, which is
# `spatial_idx` for an areal field and `obs_to_loc` for a continuous one, and
# `n_field` the number of coordinates. The projector is the same construction
# either way; only the map differs (gcol33/tulpa#848).
.rsr_unit_projection <- function(X, obs_to_field, n_field) {
  X <- as.matrix(X)
  X_unit <- matrix(0.0, n_field, ncol(X))
  cnt    <- integer(n_field)
  for (i in seq_len(nrow(X))) {
    u <- obs_to_field[i]
    X_unit[u, ] <- X_unit[u, ] + X[i, ]
    cnt[u]      <- cnt[u] + 1L
  }
  pos <- cnt > 0
  X_unit[pos, ] <- X_unit[pos, ] / cnt[pos]
  compute_rsr_projection(X_unit)
}

# The projection `spatial_rsr()` declares, attached to the spec at the field's
# own resolution. One builder for both field shapes: the RSR modifier is
# binomial-Gibbs-only whichever field it restricts, and `restrict_to` is the
# design it orthogonalises against -- the spec's own formula, not the full model
# design.
.attach_rsr_projection <- function(spatial_spec, data, family,
                                   obs_to_field, n_field) {
  if (!identical(family, "binomial")) {
    stop("An RSR spatial field is fit by the binomial Polya-Gamma Gibbs ",
         "sampler; `family` must be 'binomial' (got '", family, "').",
         call. = FALSE)
  }
  if (is.null(spatial_spec$rsr_formula)) return(spatial_spec)
  X_rsr <- stats::model.matrix(spatial_spec$rsr_formula, data = data)
  spatial_spec$rsr_projection <-
    .rsr_unit_projection(X_rsr, obs_to_field, n_field)
  spatial_spec
}

#' Dispatch a spatial Polya-Gamma Gibbs fit to the correct sampler
#'
#' The Gibbs analogue of [dispatch_laplace_spatial()]: routes on `spatial$type`
#' to the matching `cpp_pg_<family>_gibbs_<structure>` sampler, building the
#' neighbour-list / coordinate inputs each one needs. The binomial Polya-Gamma
#' augmentation backs the full areal (icar/bym2/rsr) + continuous (gp/nngp/
#' multiscale_gp) family; `neg_binomial_2` is backed by the single areal ICAR
#' negbin sampler (`cpp_pg_negbin_gibbs_spatial`), the only negbin spatial kernel.
#' @return The one-chain list [tulpa_gibbs()] finalizes (see its Value).
#' @keywords internal
dispatch_gibbs_spatial <- function(y, n_trials, X, re_group, n_re_groups,
                                   spatial, family,
                                   iter, warmup, thin = 1L,
                                   prior_beta_sd = .tulpa_prior_sd("gibbs"),
                                   prior_sigma_re_scale = 2.5,
                                   verbose = FALSE, n_threads = 1L) {
  if (!family %in% c("binomial", "neg_binomial_2")) {
    stop("Spatial Gibbs supports family 'binomial' or 'neg_binomial_2'; got '",
         family, "'. Use mode = 'laplace' for other families under a spatial field.",
         call. = FALSE)
  }
  # RSR keeps the underlying $type and flags $rsr; normalise it to the route
  # that applies the projection rather than to a plain fit that would drop it.
  # Which route depends on the field it restricts: an areal one goes to the
  # adjacency kernel, a continuous one to the NNGP kernel (gcol33/tulpa#848).
  # spatial_car()'s exported "car" is the same intrinsic ICAR field the nested
  # path already treats it as (gcol33/tulpa#819).
  spatial_type <- .areal_gibbs_type(tolower(spatial$type %||% ""))
  if (isTRUE(spatial$rsr)) {
    spatial_type <- if (spatial_type %in% .RSR_CONTINUOUS) "gp_rsr" else "rsr"
  }

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
    return(.pg_as_chain(cpp_pg_negbin_gibbs_spatial(
      y = as.integer(y), X = X,
      re_group = as.integer(re_group), n_re_groups = as.integer(n_re_groups),
      spatial_group = areal$spatial_group, n_spatial_units = areal$n_spatial_units,
      adj_list = areal$adj_list, n_neighbors = areal$n_neighbors,
      n_iter = as.integer(iter), n_warmup = as.integer(warmup),
      thin = as.integer(thin),
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_re_scale,
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01,
      prior_r_shape   = spatial$prior_r_shape %||% 1.0,
      prior_r_rate    = spatial$prior_r_rate  %||% 0.1,
      r_init          = spatial$r_init %||% 5.0,
      store_eta = FALSE, verbose = verbose, n_threads = as.integer(n_threads)
    ), c("icar", "negbin"), X))
  }

  # Binomial spatial Gibbs: the full areal + continuous sampler family. The
  # shared GLMM input block (with the binomial trial count `n`) is assembled once.
  common <- .pg_gibbs_common_args(
    y, n_trials, X, re_group, n_re_groups, iter, warmup, thin,
    prior_beta_sd, prior_sigma_re_scale, verbose, n_threads
  )

  if (spatial_type == "icar") {
    .pg_as_chain(do.call(cpp_pg_binomial_gibbs_spatial, c(common, areal, list(
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    ))), "icar", X)
  } else if (spatial_type == "bym2") {
    .pg_as_chain(do.call(cpp_pg_binomial_gibbs_bym2, c(common, areal, list(
      scale_factor              = spatial$scale_factor %||% 1.0,
      prior_sigma_spatial_scale = spatial$prior_sigma_spatial_scale %||% 2.5,
      prior_rho_alpha           = spatial$prior_rho_alpha %||% 0.5,
      prior_rho_beta            = spatial$prior_rho_beta  %||% 0.5
    ))), "bym2", X)
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
    .pg_as_chain(do.call(cpp_pg_binomial_gibbs_rsr, c(common, areal, list(
      # Row-major flatten for the C++ `rsr_projection[s * rsr_n + k]` indexing;
      # P_perp is symmetric so t() is a no-op but keeps the convention explicit.
      rsr_projection  = as.numeric(t(P_perp)),
      rsr_n           = as.integer(rsr_n),
      prior_tau_shape = spatial$prior_tau_shape %||% 1.0,
      prior_tau_rate  = spatial$prior_tau_rate  %||% 0.01
    ))), "icar", X)
  } else if (spatial_type %in% c("gp", "nngp", "gp_rsr")) {
    if (is.null(spatial$neighbor_info)) {
      stop("Spatial Gibbs (", spatial_type, ") needs a validated spatial_gp() ",
           "spec (neighbor_info is NULL). Call validate_gp(spatial, data) first, ",
           "or fit through tulpa() which validates automatically.", call. = FALSE)
    }
    n_spatial <- .gp_gibbs_require_one_obs_per_loc(spatial, length(y), spatial_type)
    nn_in     <- .gp_gibbs_nn_inputs(spatial$neighbor_info, n_spatial)
    gp_args <- c(common, list(
      coords         = as.matrix(spatial$unique_coords),
      nn_idx         = nn_in$nn_idx,
      nn_dist        = nn_in$nn_dist,
      nn_order       = nn_in$nn_order,
      n_spatial      = n_spatial,
      nn             = nn_in$nn,
      sigma2_gp_init = spatial$sigma2_gp %||% 1.0,
      phi_gp_init    = spatial$phi_gp %||% 1.0,
      cov_type       = gp_cov_type(spatial)
    ))
    if (spatial_type == "gp_rsr") {
      # Restricted continuous field. The projector is at the field's own
      # resolution -- one row per unique location -- which is the same
      # `.rsr_unit_projection()` the areal route builds, over `obs_to_loc`
      # instead of `spatial_idx`.
      P_perp <- spatial$rsr_projection %||%
        .rsr_unit_projection(X, as.integer(spatial$obs_to_loc %||%
                                             seq_len(nrow(as.matrix(X)))),
                             n_spatial)
      if (nrow(P_perp) != n_spatial || ncol(P_perp) != n_spatial) {
        stop("RSR projection matrix is ", nrow(P_perp), "x", ncol(P_perp),
             " but must be ", n_spatial, "x", n_spatial,
             " (one row/col per spatial location).", call. = FALSE)
      }
      .pg_as_chain(do.call(cpp_pg_binomial_gibbs_gp_rsr, c(gp_args, list(
        # Row-major flatten for the C++ `rsr_projection[s * rsr_n + k]`
        # indexing; P_perp is symmetric so t() is a no-op but keeps the
        # convention explicit.
        rsr_projection = as.numeric(t(P_perp)),
        rsr_n          = as.integer(n_spatial)
      ))), "gp", X)
    } else {
      .pg_as_chain(do.call(cpp_pg_binomial_gibbs_gp, gp_args), "gp", X)
    }
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
    .pg_as_chain(do.call(cpp_pg_binomial_gibbs_multiscale_gp, c(common, list(
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
      cov_type             = gp_cov_type(spatial),
      prior_phi_local_lower    = rl[1], prior_phi_local_upper    = rl[2],
      prior_phi_regional_lower = rr[1], prior_phi_regional_upper = rr[2]
    ))), "multiscale_gp", X)
  } else {
    stop("Spatial Gibbs not wired for type '", spatial_type, "'. Supported: ",
         "icar, bym2, rsr, gp/nngp (restricted or not), multiscale_gp.",
         call. = FALSE)
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
#' @return The one-chain list [tulpa_gibbs()] finalizes (see its Value).
#' @keywords internal
dispatch_gibbs_temporal <- function(y, n_trials, X, re_group, n_re_groups,
                                    temporal, family,
                                    iter, warmup, thin = 1L,
                                    prior_beta_sd = .tulpa_prior_sd("gibbs"),
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
  .pg_as_chain(do.call(cpp_pg_binomial_gibbs_temporal, c(common, list(
    time_idx        = as.integer(temporal$time_index),
    n_times         = as.integer(temporal$n_times),
    seasonal_period = as.integer(temporal$seasonal %||% 0L),
    trend_type      = trend_type,
    short_type      = short_type
  ))), c("temporal", if (short_type == 1L) "ar1_short"), X)
}


#' Compute GLM working weights for Laplace Hessian
#'
#' Thin wrapper over the family-ops registry so the weight formulas live in
#' exactly one place (`R/family_loglik.R`). This is the weight the ENGINE's
#' Laplace Hessian carries, which is what every caller here rebuilds `H` from,
#' so which of the two curvatures it returns is decided by the compiled
#' dispatch rather than chosen independently: `cpp_family_working_weight_is_observed()`
#' names the families whose compiled working weight IS the observed curvature,
#' and for those the y-free expected form is a different function
#' (gcol33/tulpa#824).
#'
#' `neg_binomial_2` is the one family where that bites: its compiled branch
#' returns `(y + phi) phi mu / (mu + phi)^2` while the registry's y-free
#' `weight` is the expected `mu phi / (mu + phi)`, which at a response away
#' from the mean differ by tens of percent. Every other family either has no
#' separate observed form or is one the compiled side answers with the expected
#' weight, so `y` changes nothing and may be omitted.
#' @keywords internal
glmm_weights <- function(eta, family, n_trials = NULL, phi = 1.0, phi2 = NULL,
                         y = NULL) {
  # Resolution (including the `<family>_<link>` forms) belongs to .family_ops();
  # duplicating the lookup here is what made a suffixed family fit in the engine
  # and then fail on the R-side Hessian.
  .family_or_stop(family)
  if (!.glmm_weight_needs_y(family)) {
    return(as.numeric(family_weight(eta, family, n_trials, phi, phi2)))
  }
  if (is.null(y)) {
    stop(sprintf(paste0("Family '%s' has an observed working weight, so the ",
                        "Laplace Hessian depends on the response; pass `y`."),
                 family), call. = FALSE)
  }
  as.numeric(.family_obs_weight(eta, y, family, n_trials, phi, phi2))
}

# A family whose compiled working weight IS the observed curvature AND whose
# observed curvature is a different FUNCTION from the expected one (it carries
# y). Only then does the registry's y-free weight fail to reproduce the Hessian
# the engine's Laplace actually used -- for poisson, binomial or a
# canonical-link family the two coincide identically, so nothing is needed.
#' @keywords internal
.glmm_weight_needs_y <- function(family) {
  ops <- .family_ops(family)
  !is.null(ops$obs_weight) &&
    isTRUE(tryCatch(cpp_family_working_weight_is_observed(family),
                    error = function(e) FALSE))
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
#' @param beta_prior Fixed-effect prior as `list(mean, sd)`: a mean-zero
#'   (`mean = 0`) Gaussian on every coefficient with SD `sd` (default
#'   the engine default, `prior_normal(0, 2.5)`). The Polya-Gamma sampler uses a mean-zero prior,
#'   so a non-zero `mean` errors.
#' @param prior_sigma_scale Prior scale for RE sigma (statistical; default 2.5).
#' @details For `family = "neg_binomial_2"` the Polya-Gamma weights are drawn
#'   at the exact real shape `PG(y + r, eta)` and the dispersion `r` is
#'   updated by a random-walk Metropolis-Hastings step on `log(r)` whose
#'   stationary support is bounded to `r` in `[0.1, 500]`; data favouring a
#'   dispersion outside that range pile up at the boundary.
#'
#'   Every sampler that moves a latent effect's level through the first
#'   coefficient -- the negative-binomial kernels, and the binomial kernels
#'   carrying a `spatial` or `temporal` field -- leaves `eta` unchanged only
#'   when the first column of `X` is an all-ones intercept, and those routes
#'   error on a design without one. An intrinsic field (ICAR, the structured
#'   BYM2 part, RW1) is reported centred with its level in the intercept, and
#'   its sweep carries the intercept's prior through the field mean; a proper
#'   one (the negative-binomial iid block, an NNGP field) has the level it
#'   shares with the intercept drawn from its full conditional, which both
#'   priors define.
#'
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
#' @param control A named list of numerical / tuning knobs (statistical
#'   arguments stay in the signature above): `n_iter` (default 2000), `warmup`
#'   (default 1000), `thin` (default 1, applied on every route including the
#'   spatial and temporal ones; the run keeps `ceiling((n_iter - warmup) / thin)`
#'   draws), `seed` (`NULL` draws from the session
#'   RNG; the Polya-Gamma kernels use R's RNG, so a seed makes the fit
#'   reproducible), `verbose` (default FALSE), `n_threads` (default 1).
#'
#' @return A `tulpa_fit` holding one MCMC chain: `draws`, the
#'   `[n_saved x n_param]` matrix of retained draws, with `chain_id`,
#'   `n_chains`, `means` (column means) and `param_names`. Columns follow the
#'   parameter naming of the ModelData samplers: the fixed effects (the column
#'   names of `X`, else `beta[j]`), `log_sigma_re` and `re[g]` for the
#'   random-intercept block, then the route's own blocks -- `log_phi` (the
#'   negative-binomial size); `log_tau_spatial` and `phi_spatial[u]` (ICAR, and
#'   RSR's projected field); `log_sigma_spatial`, `logit_rho_bym2`,
#'   `phi_spatial[u]` and `theta_spatial[u]` (BYM2); `log_sigma2_gp`,
#'   `log_phi_gp` and `gp_w[u]` (GP); the `_local` / `_regional` counterparts
#'   with `gp_local[u]` / `gp_regional[u]` (multiscale GP); `log_sigma2_trend`,
#'   `trend[t]`, `log_sigma2_seasonal`, `seasonal[s]`, `log_sigma2_short`,
#'   `short_term[t]` and, for an AR1 short-term component, `logit_rho_short`
#'   (temporal). Scales are stored on the log scale and correlations on the
#'   logit scale, as the samplers store them; a component the model does not
#'   carry has no column.
#'
#' @examples
#' set.seed(1)
#' G <- 20L; npg <- 15L; n <- G * npg
#' grp <- rep(seq_len(G), each = npg)
#' X <- cbind(1, rnorm(n))
#' b <- rnorm(G, 0, 0.6)
#' y <- rbinom(n, 1, plogis(X %*% c(-0.2, 0.5) + b[grp]))
#' \donttest{
#' fit <- tulpa_gibbs(y, rep(1L, n), X, grp, G, family = "binomial",
#'                    control = list(n_iter = 500L, warmup = 250L))
#' coef(fit)
#' diagnostics(fit)
#' }
#' @export
tulpa_gibbs <- function(y, n_trials, X, group, n_groups,
                        family = "binomial",
                        beta_prior = .tulpa_default_beta_prior("gibbs"),
                        prior_sigma_scale = 2.5,
                        spatial = NULL, temporal = NULL,
                        control = list()) {

  tulpa_check_control(control, .CONTROL_KEYS$gibbs, "tulpa_gibbs")
  family        <- .canonical_family(family)
  n_iter        <- as.integer(control$n_iter %||% 2000L)
  warmup        <- as.integer(control$warmup %||% 1000L)
  .check_run_length(n_iter, warmup, "tulpa_gibbs")
  thin          <- as.integer(control$thin %||% 1L)
  verbose       <- isTRUE(control$verbose)
  n_threads     <- as.integer(control$n_threads %||% 1L)
  prior_beta_sd <- .beta_prior_ridge_sd(beta_prior, .tulpa_prior_sd("gibbs"))
  .seed_scoped(control$seed)

  # Spatial / temporal field present: route to the matching Polya-Gamma Gibbs
  # sampler (the Gibbs analogue of tulpa_laplace(spatial = ...)). `group` /
  # `n_groups` become the iid random-effect block carried alongside the field.
  # There is no joint spatial+temporal Gibbs sampler -- reject rather than
  # silently dropping one field.
  if (!is.null(spatial) && !is.null(temporal)) {
    stop("Combined spatial + temporal Gibbs is not available; supply only one ",
         "of `spatial` / `temporal`.", call. = FALSE)
  }

  # Design validation (nrow(X) == length(y), n_trials length, group range) up
  # front, so a mismatch fails in R with a clear message rather than indexing
  # out of bounds in the Polya-Gamma kernel. neg_binomial_2 ignores n_trials.
  ntr_in <- if (family == "binomial") n_trials else n_trials %||% rep(1L, length(y))
  vd <- .validate_glm_design(y, X, ntr_in, "tulpa_gibbs")
  if (length(group) != vd$N) {
    stop(sprintf("tulpa_gibbs: length(group) (%d) must equal length(y) (%d).",
                 length(group), vd$N), call. = FALSE)
  }
  # n_groups == 0 marks "no iid random-effect block" (the spatial-only front
  # door passes a placeholder group of all 1s); only range-check a real block.
  grp_i <- as.integer(group)
  if (n_groups > 0L &&
      (anyNA(grp_i) || min(grp_i) < 1L || max(grp_i) > n_groups)) {
    stop(sprintf(paste0("tulpa_gibbs: `group` must be 1-based integers in ",
                        "[1, n_groups = %d]; got range [%d, %d]."),
                 n_groups, min(grp_i), max(grp_i)), call. = FALSE)
  }

  res <- if (!is.null(spatial)) {
    dispatch_gibbs_spatial(
      y = y, n_trials = vd$n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      spatial = spatial, family = family,
      iter = n_iter, warmup = warmup, thin = thin,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    )
  } else if (!is.null(temporal)) {
    dispatch_gibbs_temporal(
      y = y, n_trials = vd$n_trials,
      X = X, re_group = group, n_re_groups = n_groups,
      temporal = temporal, family = family,
      iter = n_iter, warmup = warmup, thin = thin,
      prior_beta_sd = prior_beta_sd, prior_sigma_re_scale = prior_sigma_scale,
      verbose = verbose, n_threads = n_threads
    )
  } else if (family == "binomial") {
    .pg_as_chain(cpp_pg_binomial_gibbs(
      y = as.numeric(y), n = as.integer(vd$n_trials), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(n_iter), n_warmup = as.integer(warmup),
      thin = thin,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    ), character(0), X)
  } else if (identical(family, "neg_binomial_2")) {
    .pg_as_chain(cpp_pg_negbin_gibbs(
      y = as.numeric(y), X = X,
      group = as.integer(group), n_groups = as.integer(n_groups),
      n_iter = as.integer(n_iter), n_warmup = as.integer(warmup),
      thin = thin,
      prior_beta_sd = prior_beta_sd,
      prior_sigma_scale = prior_sigma_scale,
      prior_r_shape = 1.0, prior_r_rate = 0.1, r_init = 5.0,
      store_eta = FALSE, verbose = verbose,
      n_threads = as.integer(n_threads)
    ), "negbin", X)
  } else {
    stop(sprintf(
      "Gibbs not available for family '%s'. Supported: %s.",
      family,
      paste(c("binomial", "neg_binomial_2"), collapse = ", ")
    ), call. = FALSE)
  }

  # Finalize so direct callers get a real tulpa_fit (print/coef/summary
  # dispatch) instead of the raw draw list; tulpa_dispatch re-finalizing the
  # routed path is a no-op (every field fills with %||%).
  .finalize_fit(res, backend = "gibbs", n_fixed = ncol(X),
                fixed_names = res$param_names[seq_len(ncol(X))],
                data = list(y = y, n_trials = vd$n_trials,
                           model_matrix = X, family = family))
}
