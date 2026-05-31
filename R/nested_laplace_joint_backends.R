# Single-block backend dispatch table + kernel call.
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R.

# --- backend dispatch table --------------------------------------------------
#
# Adding a new joint backend means appending one entry here. Each entry
# bundles the four backend-specific concerns: grid construction, kernel call,
# theta-grid materialisation for the result, and latent-layout metadata.

# Each entry now owns three concerns: grid construction, theta-grid
# materialisation, and latent-layout metadata. The kernel call is shared
# across all single-block backends -- they all route through
# `.joint_call_kernel_via_multi()` which packs the single-block prior into
# a length-1 multi-block spec and dispatches via
# `cpp_nested_laplace_joint_multi`. This is the J-E unification: one
# inner C++ entry, one R-side post-processing path. The legacy
# per-backend `cpp_nested_laplace_joint_{bym2,icar,car_proper}` shims and
# their bespoke LatentBlock construction are gone.
.joint_backends <- list(
    bym2 = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_axis <- prior$rho_grid %||% c(0.2, 0.5, 0.8, 0.95)
            .joint_cartesian(list(sigma = sigma_axis, rho = rho_axis),
                              has_copy, alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0,
                                force_sparse = FALSE,
                                cell_coupling = "separable",
                                hessian_pd_mode = 0L,
                                step_curvature_mode = 0L) {
            .joint_call_kernel_via_multi("bym2", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol,
                                          force_sparse = force_sparse,
                                          cell_coupling = cell_coupling,
                                          hessian_pd_mode = hessian_pd_mode,
                                          step_curvature_mode = step_curvature_mode)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, rho = grids$rho,
                      alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma, rho = grids$rho)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 2L,
                          spatial_block_names = c("phi_start", "theta_start"))
        }
    ),

    icar = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            .joint_cartesian(list(sigma = sigma_axis), has_copy,
                              alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0,
                                force_sparse = FALSE,
                                cell_coupling = "separable",
                                hessian_pd_mode = 0L,
                                step_curvature_mode = 0L) {
            .joint_call_kernel_via_multi("icar", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol,
                                          force_sparse = force_sparse,
                                          cell_coupling = cell_coupling,
                                          hessian_pd_mode = hessian_pd_mode,
                                          step_curvature_mode = step_curvature_mode)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    ),

    car_proper = list(
        build_grids = function(prior, has_copy, alpha_axis, phi_axes = NULL) {
            sigma_axis   <- prior$sigma_grid %||%
                exp(seq(log(0.1), log(3), length.out = 5))
            rho_car_axis <- prior$rho_car_grid %||% c(0.5, 0.8, 0.95, 0.99)
            .joint_cartesian(list(sigma = sigma_axis, rho_car = rho_car_axis),
                              has_copy, alpha_axis, phi_axes)
        },
        call_kernel = function(arms, prior, cp, grids, max_iter, tol,
                                n_threads, x_init, store_Q = FALSE,
                                arm_names = NULL, n_threads_outer = 1L,
                                tile_warm = TRUE, prune_tol = 0.0,
                                force_sparse = FALSE,
                                cell_coupling = "separable",
                                hessian_pd_mode = 0L,
                                step_curvature_mode = 0L) {
            .joint_call_kernel_via_multi("car_proper", arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = n_threads_outer,
                                          tile_warm = tile_warm,
                                          prune_tol = prune_tol,
                                          force_sparse = force_sparse,
                                          cell_coupling = cell_coupling,
                                          hessian_pd_mode = hessian_pd_mode,
                                          step_curvature_mode = step_curvature_mode)
        },
        theta_grid = function(grids, has_copy) {
            base <- if (has_copy) {
                cbind(sigma = grids$sigma, rho_car = grids$rho_car,
                      alpha = grids$alpha)
            } else {
                cbind(sigma = grids$sigma, rho_car = grids$rho_car)
            }
            .append_phi_columns(base, grids)
        },
        layout = function(arms, prior) {
            .joint_layout(arms, prior$n_spatial_units, n_spatial_blocks = 1L,
                          spatial_block_names = "phi_start")
        }
    )
)

# Single-block call_kernel: route through cpp_nested_laplace_joint_multi by
# packing the legacy prior + per-arm spatial_idx into a length-1
# blocks_spec list and a theta_grid matrix matching the C++ side's axis
# conventions:
#
#   bym2  +copy: axes = (sigma_occ, sigma_pos, rho)   -- unit-precision
#                latent. R side feeds sigma_occ = sigma, sigma_pos =
#                alpha * sigma from the (sigma, alpha) outer grid; the
#                C++ reads two named columns and uses them as per-arm
#                scaling factors. Mathematically identical to the legacy
#                (sigma_occ, sigma_pos) layout cell-by-cell, but the SET
#                of cells covers alpha evenly instead of forming a
#                Cartesian product in sigma_pos.
#   bym2  -copy: axes = (sigma, rho)                  -- sigma rolled into
#                d_fac directly (no copy block).
#   icar  +copy: axes = (sigma_occ, sigma_pos)        -- as above.
#   icar  -copy: axes = (tau,)                        -- tau on prior;
#                grid$sigma is translated to tau = 1 / sigma^2.
#   car_proper +copy: (sigma_occ, sigma_pos, rho_car) -- as above.
#   car_proper -copy: (sigma, rho_car)                -- sigma in d_fac.
#
# The legacy backends used `grid$sigma` for both the donor amplitude
# (copy case) and the prior scale (no-copy case). The C++ kernel is
# unchanged after the reparameterization; only the physical cells it
# visits change.
.joint_call_kernel_via_multi <- function(type, arms, prior, cp, grids,
                                          max_iter, tol, n_threads,
                                          x_init, store_Q, arm_names,
                                          n_threads_outer = 1L,
                                          tile_warm = TRUE,
                                          prune_tol = 0.0,
                                          force_sparse = FALSE,
                                          cell_coupling = "separable",
                                          hessian_pd_mode = 0L,
                                          step_curvature_mode = 0L) {
    n_arms <- length(arms)
    spi <- lapply(arms, function(a) as.integer(a$spatial_idx))

    block_spec <- list(
        type            = type,
        n_spatial_units = as.integer(prior$n_spatial_units),
        adj_row_ptr     = as.integer(prior$adj_row_ptr),
        adj_col_idx     = as.integer(prior$adj_col_idx),
        n_neighbors     = as.integer(prior$n_neighbors),
        spatial_idx     = spi
    )
    if (type == "bym2") {
        block_spec$scale_factor <- as.numeric(prior$scale_factor %||% 1.0)
    }

    # Construct theta_grid columns in the order the C++ kernel expects.
    # Names get a `b1.` prefix to match cpp_nested_laplace_joint_multi's
    # naming convention. The C++ axis names `sigma_occ` / `sigma_pos` are
    # preserved as a contract with the kernel; the R outer grid lives in
    # (sigma, alpha) space and materializes sigma_pos = alpha * sigma at
    # the kernel-call boundary.
    if (cp$has_copy) {
        sigma_occ_col <- as.numeric(grids$sigma)
        sigma_pos_col <- as.numeric(grids$sigma) * as.numeric(grids$alpha)
        cols <- switch(
            type,
            bym2       = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col,
                              b1.rho       = grids$rho),
            icar       = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col),
            car_proper = list(b1.sigma_occ = sigma_occ_col,
                              b1.sigma_pos = sigma_pos_col,
                              b1.rho_car   = grids$rho_car)
        )
        copy_block <- 0L
    } else {
        cols <- switch(
            type,
            bym2       = list(b1.sigma = grids$sigma, b1.rho = grids$rho),
            icar       = list(b1.tau   = 1.0 / (as.numeric(grids$sigma)^2)),
            car_proper = list(b1.sigma = grids$sigma, b1.rho_car = grids$rho_car)
        )
        copy_block <- -1L
    }
    theta_grid <- do.call(cbind, lapply(cols, as.numeric))
    colnames(theta_grid) <- names(cols)
    axis_offsets <- as.integer(c(0L, length(cols)))

    phi_grid_per_arm <- .joint_phi_grid_per_arm(grids, arm_names)

    # Tile partition for the three-tier warm-start (dev_notes/speedup.md
    # Opt 2). Tile = unique (sigma, [rho/rho_car,] phi_<arm>...) per cell,
    # i.e. every axis except the copy coefficient alpha. The joint mode is
    # tile-constant on the donor arm (Q + design depend on (sigma, rho)),
    # so any non-alpha cell in the same tile gives a much better warm-
    # start than the global pilot for the inner Newton at boundary alpha
    # cells. Only active when n_threads_outer > 1 and the user opts in
    # (tile_warm = TRUE, the default).
    tile_partition <- NULL
    if (isTRUE(tile_warm) && cp$has_copy &&
        as.integer(n_threads_outer) > 1L &&
        !is.null(grids$alpha) && length(grids$alpha) > 0L) {
        sigma_vec <- as.numeric(grids$sigma)
        n_grid <- length(sigma_vec)
        non_alpha_cols <- list(sigma = sigma_vec)
        if (!is.null(grids$rho)) {
            non_alpha_cols$rho <- as.numeric(grids$rho)
        }
        if (!is.null(grids$rho_car)) {
            non_alpha_cols$rho_car <- as.numeric(grids$rho_car)
        }
        phi_col_names <- grep("^phi_", names(grids), value = TRUE)
        for (nm in phi_col_names) {
            non_alpha_cols[[nm]] <- as.numeric(grids[[nm]])
        }
        non_alpha_mat <- do.call(cbind, non_alpha_cols)
        tile_partition <- .joint_compute_tile_partition(
            non_alpha_mat, as.numeric(grids$alpha), n_grid)
    }

    res <- cpp_nested_laplace_joint_multi(
        arms_list    = arms,
        copy_arm     = as.integer(cp$copy_arm_zero),
        copy_block   = copy_block,
        blocks_spec  = list(block_spec),
        theta_grid   = theta_grid,
        axis_offsets = axis_offsets,
        max_iter     = as.integer(max_iter),
        tol          = as.numeric(tol),
        n_threads    = as.integer(n_threads),
        x_init_nullable = x_init,
        store_Q      = isTRUE(store_Q),
        phi_grid_per_arm = phi_grid_per_arm,
        n_threads_outer = as.integer(n_threads_outer),
        tile_ids        = tile_partition$tile_ids,
        tile_pilot_cells = tile_partition$tile_pilot_cells,
        prune_tol       = as.numeric(prune_tol),
        force_sparse    = isTRUE(force_sparse),
        cell_coupling_name = as.character(cell_coupling),
        hessian_pd_mode = as.integer(hessian_pd_mode),
        step_curvature_mode = as.integer(step_curvature_mode)
    )
    # Strip the C++-side theta_grid / axis_offsets — the backend's
    # `theta_grid()` callback rebuilds them with the user-facing bare
    # names in tulpa_nested_laplace_joint().
    res$theta_grid   <- NULL
    res$axis_offsets <- NULL
    res
}

# --- cheap-pass prune safety gate --------------------------------------------
#
# After a pruned kernel run, decide whether the kept set can be trusted. The
# cheap screen ranks cells; the full inner Newton only ran on survivors. If
# the screen's ranking disagrees with the full-solve ranking the prune may
# have dropped the true posterior mode (gcol33/tulpa#43), so we WARN and fall
# back to the full grid rather than silently returning the pruned answer.
#
# Two triggers, either of which forces the fallback:
#   1) argmax disagreement: the cheap-screen argmax cell is not the cell the
#      full solve favours among survivors (or the cheap argmax was itself
#      pruned). The kernel flags this in `prune_argmax_disagree`.
#   2) cheap-vs-full gap collapse: the kept set's posterior collapses onto one
#      cell (low quadrature ESS) AND the cheap screen badly mis-estimated that
#      cell's log-marginal (`prune_cheap_full_gap` large relative to the spread
#      of full log-marginals across kept cells). A large gap on the cell the
#      whole posterior sits on means the screen could just as easily have
#      ranked it below the threshold and pruned it.
#
# `res` is the pruned kernel result (already carries the prune_* fields).
# `resolve_full` is a zero-argument thunk re-running the SAME kernel call with
# `prune_tol = 0` (the full grid). Returns either `res` (gate passed) or the
# full-grid result (gate tripped). When the gate trips, the returned result
# carries `prune_fallback_triggered = TRUE` and `prune_fallback_reason`.
.joint_prune_safety_gate <- function(res, resolve_full,
                                      gap_abs_floor = 5.0,
                                      gap_frac = 0.5,
                                      ess_collapse_frac = 0.05) {
    # No prune ran (prune off, prune_tol = 0, or single-cell grid): nothing
    # to gate. The kernel only emits prune_mask when it actually screened.
    if (is.null(res$prune_mask)) return(res)

    disagree <- isTRUE(res$prune_argmax_disagree)

    # Gap-collapse trigger. Quadrature ESS over the kept (finite-weight)
    # cells; "collapse" = ESS below a small fraction of the kept count.
    lm   <- res$log_marginal
    kept <- is.finite(lm)
    n_kept <- sum(kept)
    gap_collapse <- FALSE
    if (n_kept >= 1L && !is.null(res$prune_cheap_full_gap) &&
        is.finite(res$prune_cheap_full_gap)) {
        w <- .nl_normalise_weights(lm)
        ess <- if (sum(w^2) > 0) (sum(w)^2) / sum(w^2) else 1
        lm_kept <- lm[kept]
        lm_spread <- if (n_kept > 1L) diff(range(lm_kept)) else 0
        gap_thresh <- max(gap_abs_floor, gap_frac * lm_spread)
        gap_collapse <- (ess <= max(1.0, ess_collapse_frac * n_kept)) &&
                        (res$prune_cheap_full_gap > gap_thresh)
    }

    if (!disagree && !gap_collapse) return(res)

    reason <- if (disagree && gap_collapse) {
        "cheap-screen argmax disagrees with full-solve argmax and the kept posterior collapses onto a cell the screen badly mis-estimated"
    } else if (disagree) {
        "cheap-screen argmax disagrees with the full-solve argmax"
    } else {
        "kept posterior collapses onto a cell whose cheap-vs-full log-marginal gap is large"
    }
    warning(sprintf(
        "tulpa_nested_laplace_joint(): cheap-pass prune is unreliable for this fit (%s); falling back to the full grid. Set control$prune = FALSE to silence this, or leave it -- the full grid is correct.",
        reason), call. = FALSE)

    full <- resolve_full()
    full$prune_fallback_triggered <- TRUE
    full$prune_fallback_reason    <- reason
    full
}
