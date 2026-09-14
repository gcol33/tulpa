# Batched multi-response joint nested-Laplace wrapper.
#
# Marshals `responses` + a list-or-single block `prior` to the batched C++ entry
# `cpp_nested_laplace_joint_multi_batch`, mirroring `.joint_dispatch_multi`'s
# arm / block-spec / grid construction. Dense tensor grid only (no CCD), all-
# coupled cell-coupling families (occu_cover). Per-arm per-species dispersion is
# `phi_batch`; an arm that integrates its dispersion instead carries it as an
# outer axis through `phi_grid_batch`, crossed onto the latent grid exactly as
# the multi-block driver crosses `phi_grid`, with each species at its own nodes.
# Returns `per_species` (length n_batch; each list(log_marginal, modes, n_iter,
# score_max, converged, -- when store_Q -- Q_csc_{p,i,x}_per_grid + Q_csc_n,
# `theta_grid` (that species' outer grid, dispersion columns included), and the
# outer integration `.joint_multi_attach_integration()` attaches:
# log_hyperprior*, log_quad, axis_support, weights)), `theta_grid` (the latent
# axis columns over the cell layout every species shares), `axis_offsets`. Each
# species' `log_marginal` carries the folded hyperprior and its weights take the
# cell measure, exactly as the multi-block driver builds them on the same grid.
#
# `y_batch`        : length n_arms list; element k is a [N_k x n_batch] response
#                    matrix (species columns) for a data arm, or NULL for a
#                    no-data arm (psi).
# `phi_batch`      : [n_arms x n_batch] per-arm per-species dispersion.
# `phi_grid_batch` : NULL, or a length-n_batch list whose element s is species
#                    s's `phi_grid` (the shape `tulpa_nested_laplace_joint()`
#                    takes). Every species must carry axes on the same arms with
#                    the same node counts, so the batch shares one cell layout;
#                    the node values are the species' own.
#
# Shared marshaling: responses + prior -> arms / copy spec / blocks_spec /
# axis_offsets / per-species outer grids / C++ grid. Single source of truth for
# the batched call and the single-species comparison so both hit a
# byte-identical outer grid.
# @keywords internal
.tulpa_nl_joint_marshal <- function(responses, prior, copy = NULL,
                                    phi_grid_batch = NULL, n_batch = 1L) {
  prior_list <- if (.is_multi_block_prior(prior)) prior else list(prior)
  n_arms <- length(responses)
  arms <- lapply(seq_along(responses),
                 function(k) .normalise_joint_arm_multi(responses[[k]], k))
  arm_names <- names(responses) %||% paste0("arm", seq_along(responses))
  cp <- .resolve_copy_multi(copy, responses, prior_list)
  B <- length(prior_list)

  per_block <- lapply(seq_len(B), function(b) {
    copy_pos <- if (cp$has_copy) match((b - 1L), cp$copy_blocks_zero) else NA_integer_
    is_copy  <- !is.na(copy_pos)
    alpha_grid_b <- if (is_copy) cp$alpha_grids[[copy_pos]] else numeric(0)
    .joint_block_axis_grid(prior_list[[b]], is_copy, alpha_grid_b, b)
  })
  block_grids <- lapply(per_block, function(x) x$grid)
  prepared    <- lapply(per_block, function(x) x$prepared)
  axis_counts  <- vapply(block_grids, ncol, integer(1))
  axis_offsets <- as.integer(c(0L, cumsum(axis_counts)))
  axis_names   <- unlist(lapply(seq_along(block_grids), function(b) {
    cn <- colnames(block_grids[[b]])
    if (is.null(cn) || length(cn) == 0L) character(0) else paste0("b", b, ".", cn)
  }))
  blocks_spec <- lapply(seq_along(prepared), function(b) {
    .joint_block_spec_for_cpp(prepared[[b]], n_arms, b, arms = arms)
  })

  row_counts <- vapply(block_grids, nrow, integer(1))
  idx <- do.call(expand.grid, lapply(row_counts, seq_len))
  latent_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
    block_grids[[b]][idx[[b]], , drop = FALSE]
  }))
  if (ncol(latent_grid) > 0L) colnames(latent_grid) <- axis_names

  sg <- .tulpa_nl_joint_species_grids(latent_grid, phi_grid_batch, n_batch,
                                      axis_names, arm_names)
  cpp_grid <- .joint_multi_cpp_grid(sg$grids[[1L]], axis_offsets, B, cp)

  list(arms = arms, cp = cp, blocks_spec = blocks_spec,
       axis_offsets = axis_offsets, cpp_grid = cpp_grid,
       prepared = prepared, B = B,
       latent_grid = latent_grid[sg$latent_row, , drop = FALSE],
       joint_grids = sg$grids, phi_grid_per_arm = sg$phi_grid_per_arm,
       multi_block = .is_multi_block_prior(prior),
       arm_names = arm_names)
}

# Each species' outer grid: the latent grid crossed with that species' own
# dispersion axes (`.joint_multi_cross_phi()`, the multi-block driver's cross).
# Returns `grids` (length n_batch), `latent_row` (the latent row each cell
# repeats, shared by every species) and `phi_grid_per_arm` (NULL, or a length
# n_arms list whose entry k is NULL or the [n_grid x n_batch] matrix of arm k's
# dispersion per cell and species, in the R-side convention).
.tulpa_nl_joint_species_grids <- function(latent_grid, phi_grid_batch, n_batch,
                                          axis_names, arm_names) {
  n_batch <- as.integer(n_batch)
  shared <- list(grids = rep(list(latent_grid), n_batch),
                 latent_row = seq_len(nrow(latent_grid)),
                 phi_grid_per_arm = NULL)
  if (is.null(phi_grid_batch)) return(shared)
  if (!is.list(phi_grid_batch) || length(phi_grid_batch) != n_batch) {
    stop("`phi_grid_batch` must be NULL or a list of length n_batch (",
         n_batch, "), one `phi_grid` per species.", call. = FALSE)
  }
  axes <- lapply(phi_grid_batch, function(pg)
    .normalise_phi_grid(pg, arm_names))
  n_nodes <- vapply(axes, function(a) {
    if (is.null(a)) integer(length(arm_names))
    else vapply(a, length, integer(1))
  }, integer(length(arm_names)))
  n_nodes <- matrix(n_nodes, nrow = length(arm_names))
  if (any(n_nodes != n_nodes[, 1L])) {
    stop("`phi_grid_batch`: every species must carry a dispersion axis on the ",
         "same arms with the same node count, so the batch shares one outer ",
         "cell layout; the node values may differ by species.", call. = FALSE)
  }
  if (all(n_nodes[, 1L] == 0L)) return(shared)

  crossed <- lapply(axes, function(a)
    .joint_multi_cross_phi(latent_grid, a, axis_names))
  grids <- lapply(crossed, `[[`, "grid")
  n_grid <- nrow(grids[[1L]])
  per_species <- lapply(grids, .joint_multi_phi_per_arm, arm_names = arm_names)
  phi_grid_per_arm <- lapply(seq_along(arm_names), function(k) {
    if (n_nodes[k, 1L] == 0L) return(NULL)
    matrix(unlist(lapply(per_species, `[[`, k), use.names = FALSE),
           nrow = n_grid, ncol = n_batch)
  })
  list(grids = grids, latent_row = crossed[[1L]]$latent_row,
       phi_grid_per_arm = phi_grid_per_arm)
}

# The outer integration of one species' kernel result on its outer grid
# `joint_grid`: the regularizing hyperprior folded into `log_marginal`, then the
# cell measure and weights, through the same two steps the multi-block driver
# takes.
.tulpa_nl_joint_integrate <- function(r, joint_grid, m, prior_sigma = NULL,
                                      prior_alpha = NULL, prior_phi = NULL,
                                      copy_atom_mass = .TULPA_COPY_ATOM_MASS,
                                      copy_slab = "exponential") {
  fn_sigma <- .joint_parse_hyperprior(prior_sigma, "prior_sigma", m$multi_block)
  fn_alpha <- .joint_parse_hyperprior(prior_alpha, "prior_alpha", m$multi_block)
  fn_phi   <- .joint_parse_sigma_prior(prior_phi, "prior_phi")
  copy_slab <- .hyper_check_copy_slab(copy_slab)
  families  <- .joint_multi_hp_families(m$arms, m$arm_names)
  hp <- .joint_multi_hyperprior(joint_grid, fn_sigma, fn_alpha, fn_phi,
                                blocks = m$prepared, families = families,
                                copy_atom_mass = copy_atom_mass)
  r <- .nl_fold_hyperprior(r, list(hp))
  r$theta_grid <- joint_grid
  .joint_multi_attach_integration(
    r, joint_grid, m$axis_offsets, m$B, m$prepared, families,
    fn_sigma, fn_alpha, fn_phi, copy_slab, copy_atom_mass)$res
}

# @keywords internal
tulpa_nl_joint_batch <- function(responses, prior, copy = NULL,
                                 n_batch, y_batch, phi_batch,
                                 max_iter = 200L, tol = 1e-6,
                                 cell_coupling = "separable",
                                 store_Q = TRUE,
                                 prior_sigma = NULL, prior_alpha = NULL,
                                 prior_phi = NULL,
                                 copy_atom_mass = .TULPA_COPY_ATOM_MASS,
                                 copy_slab = "exponential",
                                 phi_grid_batch = NULL) {
  m <- .tulpa_nl_joint_marshal(responses, prior, copy,
                               phi_grid_batch = phi_grid_batch,
                               n_batch = n_batch)
  ka <- .joint_phi_args_to_kernel(list(arms_list = m$arms,
                                       phi_batch = phi_batch,
                                       phi_grid_per_arm = m$phi_grid_per_arm))
  res <- cpp_nested_laplace_joint_multi_batch(
    arms_list          = ka$arms_list,
    copy_arms          = as.integer(m$cp$copy_arms_zero),
    copy_blocks        = as.integer(m$cp$copy_blocks_zero),
    blocks_spec        = m$blocks_spec,
    theta_grid         = m$cpp_grid,
    axis_offsets       = m$axis_offsets,
    n_batch            = as.integer(n_batch),
    y_batch            = y_batch,
    phi_batch          = ka$phi_batch,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    cell_coupling_name = as.character(cell_coupling),
    store_Q            = isTRUE(store_Q),
    phi_grid_per_arm   = ka$phi_grid_per_arm
  )
  # Structural metadata shared by every species (the design is species-
  # invariant): the latent-vector layout and the latent (sigma, alpha, ...)
  # columns over the shared cell layout. Each species carries its own full
  # outer grid, so a consumer reshapes each per_species slice + the layout into
  # a single-species engine-fit for post-processing.
  res$per_species <- Map(.tulpa_nl_joint_integrate, res$per_species,
                         m$joint_grids,
                         MoreArgs = list(m = m, prior_sigma = prior_sigma,
                                         prior_alpha = prior_alpha,
                                         prior_phi = prior_phi,
                                         copy_atom_mass = copy_atom_mass,
                                         copy_slab = copy_slab))
  res$arm_layout <- .joint_multi_layout(m$arms, m$prepared)
  res$theta_grid <- m$latent_grid
  res
}

# Single-species fit at the SAME dense grid as tulpa_nl_joint_batch (validation
# oracle). Calls the existing single-species entry .cpp_joint_multi.
# @keywords internal
tulpa_nl_joint_single <- function(responses, prior, copy = NULL,
                                  max_iter = 200L, tol = 1e-6,
                                  cell_coupling = "separable", store_Q = FALSE,
                                  phi_grid = NULL) {
  m <- .tulpa_nl_joint_marshal(responses, prior, copy,
                               phi_grid_batch = if (!is.null(phi_grid))
                                 list(phi_grid))
  .cpp_joint_multi(
    arms_list          = m$arms,
    copy_arms          = as.integer(m$cp$copy_arms_zero),
    copy_blocks        = as.integer(m$cp$copy_blocks_zero),
    blocks_spec        = m$blocks_spec,
    theta_grid         = m$cpp_grid,
    axis_offsets       = m$axis_offsets,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    n_threads          = 1L,
    x_init_nullable    = NULL,
    store_Q            = isTRUE(store_Q),
    phi_grid_per_arm   = m$phi_grid_per_arm,
    n_threads_outer    = 1L,
    tile_ids           = NULL,
    tile_pilot_cells   = NULL,
    prune_tol          = 0.0,
    force_sparse       = FALSE,
    cell_coupling_name = as.character(cell_coupling),
    inner_refresh      = 1L
  )
}
