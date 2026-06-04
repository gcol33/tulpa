# Batched multi-response joint nested-Laplace wrapper (gcol33/tulpa#66).
#
# Marshals `responses` + a list-or-single block `prior` to the batched C++ entry
# `cpp_nested_laplace_joint_multi_batch`, mirroring `.joint_dispatch_multi`'s
# arm / block-spec / grid construction. First cut: dense tensor grid only (no
# CCD, no phi-grid axis -- per-arm per-species dispersion is supplied fixed via
# `phi_batch`), all-coupled cell-coupling families (occu_cover). Returns the C++
# result: `per_species` (length n_batch; each list(log_marginal, weights, modes,
# n_iter)), `theta_grid`, `axis_offsets`.
#
# `y_batch`  : length n_arms list; element k is a [N_k x n_batch] response matrix
#              (species columns) for a data arm, or NULL for a no-data arm (psi).
# `phi_batch`: [n_arms x n_batch] per-arm per-species dispersion.
#
# Shared marshaling: responses + prior -> arms / copy spec / blocks_spec /
# axis_offsets / dense C++ grid. Single source of truth for the batched call and
# the single-species comparison so both hit a byte-identical outer grid.
# @keywords internal
.tulpa_nl_joint_marshal <- function(responses, prior, copy = NULL) {
  prior_list <- if (.is_multi_block_prior_joint(prior)) prior else list(prior)
  n_arms <- length(responses)
  arms <- lapply(seq_along(responses),
                 function(k) .normalise_joint_arm_multi(responses[[k]], k))
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
  joint_grid <- do.call(cbind, lapply(seq_along(block_grids), function(b) {
    block_grids[[b]][idx[[b]], , drop = FALSE]
  }))
  if (ncol(joint_grid) > 0L) colnames(joint_grid) <- axis_names
  cpp_grid <- .joint_multi_cpp_grid(joint_grid, axis_offsets, B, cp)

  list(arms = arms, cp = cp, blocks_spec = blocks_spec,
       axis_offsets = axis_offsets, cpp_grid = cpp_grid)
}

# @keywords internal
tulpa_nl_joint_batch <- function(responses, prior, copy = NULL,
                                 n_batch, y_batch, phi_batch,
                                 max_iter = 200L, tol = 1e-6,
                                 cell_coupling = "separable") {
  m <- .tulpa_nl_joint_marshal(responses, prior, copy)
  cpp_nested_laplace_joint_multi_batch(
    arms_list          = m$arms,
    copy_arms          = as.integer(m$cp$copy_arms_zero),
    copy_blocks        = as.integer(m$cp$copy_blocks_zero),
    blocks_spec        = m$blocks_spec,
    theta_grid         = m$cpp_grid,
    axis_offsets       = m$axis_offsets,
    n_batch            = as.integer(n_batch),
    y_batch            = y_batch,
    phi_batch          = phi_batch,
    max_iter           = as.integer(max_iter),
    tol                = as.numeric(tol),
    cell_coupling_name = as.character(cell_coupling)
  )
}

# Single-species fit at the SAME dense grid as tulpa_nl_joint_batch (validation
# oracle). Calls the existing single-species entry .cpp_joint_multi.
# @keywords internal
tulpa_nl_joint_single <- function(responses, prior, copy = NULL,
                                  max_iter = 200L, tol = 1e-6,
                                  cell_coupling = "separable", store_Q = FALSE) {
  m <- .tulpa_nl_joint_marshal(responses, prior, copy)
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
    phi_grid_per_arm   = NULL,
    n_threads_outer    = 1L,
    tile_ids           = NULL,
    tile_pilot_cells   = NULL,
    prune_tol          = 0.0,
    force_sparse       = FALSE,
    cell_coupling_name = as.character(cell_coupling),
    inner_refresh      = 1L
  )
}
