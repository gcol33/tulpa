# Adaptive tensor-lattice outer integration for the joint multi-block
# nested-Laplace path (gcol33/tulpa, low-dimensional companion to the CCD).
#
# Part of the joint nested-Laplace driver; the public entry point
# tulpa_nested_laplace_joint() lives in nested_laplace_joint.R, the outer-grid
# dispatch in nested_laplace_joint_multi.R, the central-composite design in
# nested_laplace_joint_ccd.R.
#
# Why a third integrator. The dense tensor evaluates every cell of the Cartesian
# product of the per-axis hyperparameter grids; each cell is one inner Newton
# solve. When the outer hyperparameter posterior concentrates -- a strongly
# identified field SD, a sharp Beta precision, the ess-approximately-1 regime of
# a large areal field -- almost every tensor cell sits many log-units below the
# mode and contributes negligible integration weight, yet still pays a full inner
# solve. The global CCD (nested_laplace_joint_ccd.R) fixes this at HIGH latent
# dimension (d >= 4) by a mode-find plus a 1 + 2d + 2^d design, but it declines
# below that: its Newton mode-find (several batched stencil solves per round) is
# not worth it when the tensor is small, and its Gaussian design needs a
# well-conditioned outer curvature. This integrator fills the low-d gap (the
# occupancy + cover copy model is d = 2: field sigma and copy alpha) without a
# mode-find: it evaluates a coarse subsample of the SAME lattice to locate the
# mass, then floods outward on the fine lattice keeping only cells within a
# log-density cutoff of the peak. The far tail is never evaluated.
#
# Exactness relative to the dense tensor. Every kept cell is a genuine node of
# the user's fine tensor lattice at the same uniform spacing, so the integration
# weight is the plain log-marginal softmax (dnode = NULL), identical to the dense
# path. The adaptive grid IS the dense tensor with the far-tail cells (those the
# flood-fill proved lie below `max_lm - cutoff` on every approach) omitted. Two
# uniform-weight softmaxes over nested cell sets differ only by the omitted
# cells' mass: each omitted cell carries normalised weight < exp(-cutoff), so the
# posterior (hyperparameter moments, per-cell fixed-effect mixture, derived-
# quantity draws) agrees with the dense tensor to O(n_cells * exp(-cutoff)). At
# the default cutoff (10 log-units, weight ratio ~4.5e-5) that is ~1e-3 relative
# even summed over a full 100-cell grid, and shrinks as the posterior sharpens
# (the regime where the speedup is largest). A diffuse posterior keeps more cells
# and approaches the dense grid; if the kept set would rival the dense grid the
# builder declines and the caller runs the dense tensor, so the adaptive path can
# only match or beat the dense wall-clock, never lose accuracy.

# Does the adaptive lattice engage for this integration mode and axis count? The
# dense tensor is the safe default, so "grid" never engages and "auto" keeps the
# tensor at the trivial d = 0 (no outer axes) case; "grid_adaptive" is the
# explicit opt-in. The CCD owns d >= 4 ("auto") / d >= 3 ("ccd"); this integrator
# targets the 1 <= d <= 3 band the CCD declines, but engages on request at any d
# (it is a strict subset of the tensor, so it is never wrong to try -- it simply
# declines back to the tensor when it cannot beat it).
.joint_adaptive_engage <- function(integration, d_axes) {
    if (!identical(integration, "grid_adaptive")) return(FALSE)
    is.finite(d_axes) && d_axes >= 1L
}

# Run a kernel probe with the fit's checkpoint / progress side effects stripped,
# so a grid-selection solve neither pollutes the checkpoint file nor ticks the
# progress bar (the CCD and local-CCD paths strip the same options inline).
.joint_with_quiet_opts <- function(expr) {
    op <- options(
        tulpa.nl_checkpoint = list(path = "", resume = TRUE),
        tulpa.nl_progress   = .nl_progress_args(list(progress = FALSE)))
    on.exit(options(op), add = TRUE)
    force(expr)
}

# Coarse seed indices for one axis: endpoints plus every `stride`-th interior
# level, and at least `min_pts` levels overall, so the seed brackets the mass on
# every axis while staying a strict subsample of the fine lattice (each seed
# index is a fine-lattice index, so its solve is reused by the flood, never
# wasted). Returns a sorted unique integer vector in 1..n.
.joint_adaptive_seed_axis <- function(n, stride = 2L, min_pts = 3L) {
    if (n <= 1L) return(seq_len(max(n, 0L)))
    idx <- unique(as.integer(c(1L, seq(1L, n, by = max(stride, 1L)), n)))
    if (length(idx) < min_pts) {
        idx <- unique(as.integer(round(seq(1L, n, length.out = min(min_pts, n)))))
    }
    sort(idx)
}

# Axis-wise +/- 1 lattice neighbours of a set of integer index rows, clamped to
# the [1, dims] box. Returns the unique neighbour rows (as an integer matrix)
# NOT already present in `have_keys` (the caller's seen-set), so the flood only
# ever proposes un-evaluated cells. Deterministic row order (axis-major, +before-)
# for run-to-run reproducibility.
.joint_adaptive_neighbors <- function(idx_mat, dims, have_keys, key_fn) {
    D <- length(dims)
    props <- vector("list", 0L)
    for (j in seq_len(D)) {
        for (delta in c(1L, -1L)) {
            cand <- idx_mat
            cand[, j] <- cand[, j] + delta
            ok <- cand[, j] >= 1L & cand[, j] <= dims[j]
            if (any(ok)) props[[length(props) + 1L]] <- cand[ok, , drop = FALSE]
        }
    }
    if (length(props) == 0L) return(matrix(integer(0), 0L, D))
    cand <- do.call(rbind, props)
    k <- key_fn(cand)
    keep <- !duplicated(k) & !(k %in% have_keys)
    cand[keep, , drop = FALSE]
}

# Flood-fill the fine lattice from a coarse seed, keeping cells within `cutoff`
# log-units of the running peak.
#
# Args:
#   axis_values  list length D of sorted numeric fine grids (ALL outer axes,
#                latent block axes and phi axes alike). The lattice index i on
#                axis j maps to the physical value axis_values[[j]][i].
#   eval_fn      function(idx_mat) -> list(log_marginal = numeric[nrow],
#                modes = [nrow x n_x] matrix or NULL): evaluate the inner joint
#                marginal (hyperprior already baked in) at the physical theta of
#                each integer index row. The caller batches these into one
#                parallel C++ kernel call, so the flood evaluates a whole layer
#                at once. Non-finite marginals (a non-converged inner Newton at a
#                degenerate hyperpoint) are treated as -Inf: never kept, never
#                expanded.
#   cutoff       keep / expand a cell when log_marginal >= max_lm - cutoff.
#   stride       coarse-seed subsample stride per axis.
#   max_frac     decline (return NULL) if the evaluated set would exceed this
#                fraction of the dense grid -- a diffuse posterior the tensor
#                serves as well, so fall back rather than pay flood overhead.
#   max_layers   hard cap on flood layers (a backstop; the cutoff normally stops
#                the flood well before this).
#
# Returns list(idx = kept integer index matrix [m x D], log_marginal = numeric[m],
# modes = [m x n_x] or NULL, n_eval, n_dense) with the cells kept for the
# quadrature (those within `cutoff` of the peak, plus the one-cell boundary ring
# the flood evaluated to prove the cutoff -- included at their true low weight,
# exactly as the dense tensor would carry them). NULL when the builder declines
# (degenerate lattice, all-failed seed, or the max_frac cap).
.joint_adaptive_flood <- function(axis_values, eval_fn, cutoff = 10,
                                  stride = 2L, max_frac = 0.75,
                                  max_layers = 64L) {
    D    <- length(axis_values)
    dims <- vapply(axis_values, length, integer(1))
    if (D == 0L || any(dims < 1L)) return(NULL)
    n_dense <- prod(as.numeric(dims))
    if (!is.finite(n_dense) || n_dense < 1) return(NULL)
    cap <- max(1L, floor(max_frac * n_dense))

    key_fn <- function(M) {
        if (nrow(M) == 0L) return(character(0))
        do.call(paste, c(lapply(seq_len(ncol(M)), function(j) M[, j]), sep = ","))
    }

    # Accumulators for the evaluated set (index rows, marginals, modes) and a
    # seen-set of keys so a cell is solved at most once.
    acc_idx <- matrix(integer(0), 0L, D)
    acc_lm  <- numeric(0)
    acc_md  <- NULL
    seen    <- character(0)

    absorb <- function(idx_mat) {
        if (nrow(idx_mat) == 0L) return(invisible(NULL))
        ev <- eval_fn(idx_mat)
        lm <- as.numeric(ev$log_marginal)
        lm[!is.finite(lm)] <- -Inf
        acc_idx <<- rbind(acc_idx, idx_mat)
        acc_lm  <<- c(acc_lm, lm)
        if (!is.null(ev$modes)) {
            md <- ev$modes
            if (is.null(acc_md)) {
                acc_md <<- md
            } else if (ncol(md) == ncol(acc_md)) {
                acc_md <<- rbind(acc_md, md)
            }
        }
        seen <<- c(seen, key_fn(idx_mat))
        invisible(NULL)
    }

    # Coarse seed: subsample every axis, Cartesian product, evaluate in one call.
    seed_axes <- lapply(dims, .joint_adaptive_seed_axis, stride = stride)
    seed_idx  <- as.matrix(expand.grid(seed_axes, KEEP.OUT.ATTRS = FALSE))
    dimnames(seed_idx) <- NULL
    storage.mode(seed_idx) <- "integer"
    absorb(seed_idx)
    if (!any(is.finite(acc_lm))) return(NULL)

    # Early diffuse-posterior decline. The seed is a coarse read on how the mass
    # spreads: if a large fraction of seed cells already sit within `cutoff` of
    # the peak, the posterior is diffuse and the kept region would rival the dense
    # grid, so the flood cannot beat the tensor. Bail now -- having paid only the
    # seed (~2^-d of the grid) -- and let the caller run the dense tensor, rather
    # than flooding to the cap first. The threshold is `max_frac`, the same
    # kept-fraction ceiling the post-flood cap uses, so both declines are
    # consistent. (A sharp posterior keeps few seed cells and passes here.)
    seed_max <- max(acc_lm)
    if (is.finite(seed_max)) {
        seed_keep_frac <- mean(acc_lm >= seed_max - cutoff)
        if (seed_keep_frac >= max_frac) return(NULL)
    }

    # Flood: from every currently-kept cell (within cutoff of the running max),
    # propose its un-evaluated lattice neighbours and evaluate the whole layer.
    # Stop when a layer proposes nothing new (the kept region is closed under the
    # cutoff) or a backstop trips.
    for (layer in seq_len(max_layers)) {
        max_lm <- max(acc_lm)
        if (!is.finite(max_lm)) return(NULL)
        keep_mask <- acc_lm >= max_lm - cutoff
        if (!any(keep_mask)) return(NULL)
        frontier <- acc_idx[keep_mask, , drop = FALSE]
        prop <- .joint_adaptive_neighbors(frontier, dims, seen, key_fn)
        if (nrow(prop) == 0L) break
        if (nrow(acc_idx) + nrow(prop) > cap) return(NULL)
        absorb(prop)
    }

    # Final kept set: every cell within the cutoff of the peak. The flood also
    # evaluated a one-cell ring just past the cutoff (to prove the boundary);
    # those below-cutoff cells are dropped from the quadrature (their dense-grid
    # weight is < exp(-cutoff), the truncation the exactness bound accounts for).
    max_lm <- max(acc_lm)
    keep <- which(acc_lm >= max_lm - cutoff)
    if (length(keep) == 0L) return(NULL)
    # A diffuse posterior whose kept region rivals the dense grid is served just
    # as well by the tensor; decline so the caller pays the simpler dense path.
    if (length(keep) >= cap) return(NULL)

    list(idx          = acc_idx[keep, , drop = FALSE],
         log_marginal = acc_lm[keep],
         modes        = if (!is.null(acc_md)) acc_md[keep, , drop = FALSE] else NULL,
         n_eval       = nrow(acc_idx),
         n_dense      = as.integer(min(n_dense, .Machine$integer.max)))
}

# Build the adaptive outer grid for the joint multi-block path.
#
# `axis_values` is the length-D list of per-axis fine grids over ALL outer axes
# in joint_grid column order (latent block axes first, then phi axes); `col_names`
# names those columns. `eval_theta(theta_mat)` evaluates the inner joint marginal
# (hyperprior baked in) at a physical [S x D] theta matrix (colnames = col_names)
# and returns list(log_marginal, modes) -- it owns the latent / phi column split
# and the parallel kernel call. Returns list(grid, log_marginal, modes, info) with
# `grid` the [m x D] physical node matrix (colnames = col_names) at uniform lattice
# weight (the caller uses the plain softmax, dnode = NULL), or NULL to fall back to
# the dense tensor.
.joint_adaptive_grid <- function(axis_values, col_names, eval_theta,
                                 cutoff = 10, stride = 2L, max_frac = 0.75,
                                 verbose = FALSE) {
    D <- length(axis_values)
    if (D == 0L) return(NULL)
    # Map integer lattice indices -> physical theta, then hand to eval_theta.
    eval_idx <- function(idx_mat) {
        theta <- matrix(0, nrow(idx_mat), D, dimnames = list(NULL, col_names))
        for (j in seq_len(D)) theta[, j] <- axis_values[[j]][idx_mat[, j]]
        eval_theta(theta)
    }
    fl <- tryCatch(
        .joint_adaptive_flood(axis_values, eval_idx, cutoff = cutoff,
                              stride = stride, max_frac = max_frac),
        error = function(e) NULL)
    if (is.null(fl)) return(NULL)

    grid <- matrix(0, nrow(fl$idx), D, dimnames = list(NULL, col_names))
    for (j in seq_len(D)) grid[, j] <- axis_values[[j]][fl$idx[, j]]
    if (isTRUE(verbose)) {
        message(sprintf(
            "tulpa joint: outer integration: adaptive lattice (%d axes, %d cells kept of %d dense; %d solves)",
            D, nrow(grid), fl$n_dense, fl$n_eval))
    }
    list(grid         = grid,
         log_marginal = fl$log_marginal,
         modes        = fl$modes,
         info         = list(n_cells = nrow(grid), n_eval = fl$n_eval,
                             n_dense = fl$n_dense, cutoff = cutoff))
}
