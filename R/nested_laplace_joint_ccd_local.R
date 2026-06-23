# Local central-composite-design refinement of the joint multi-block outer grid
# (gcol33/tulpa#64).
#
# The global CCD (nested_laplace_joint_ccd.R) replaces the whole tensor grid with
# one design oriented by a mode-find over the joint hyperparameter posterior.
# Local CCD is the complementary tool: it KEEPS a (coarse) tensor base grid and
# refines only a few high-weight, mutually non-adjacent cells with a small
# curvature-aware node cloud each. It earns its place only when it lets the base
# grid be coarser than a single-resolution tensor product would need -- i.e. at
# moderate-to-high latent dimension (d >= 4), where a uniformly fine tensor is
# k^d-expensive but the posterior mass sits in a handful of cells. In the d <= 3
# regime the tensor grid is already cheap and dense and boundary/interior grid
# refinement (hyper_grid_refine.R) covers a too-narrow grid; local CCD there is
# overhead, so the engage gate declines it.
#
# Two properties make it safe to layer on top of the existing integration:
#
#   * Weight conservation (no double-count). A refined cell `c` carries outer
#     design weight `Delta_c` (1 on a tensor base, the cell's CCD weight on a
#     CCD base). It is REPLACED by its local CCD sub-nodes, each given design
#     weight `Delta_c * delta_j` where the `delta_j` are the corrected R-INLA CCD
#     weights (ccd_weights(), partition of unity: sum_j delta_j = 1). On a flat
#     integrand across the cell the sub-nodes' mass sums back to the cell's:
#     Delta_c * sum_j delta_j * exp(ell_c) = Delta_c * exp(ell_c). Refining a
#     cell can only RE-ESTIMATE its mass, never inflate the total. The single
#     channel for this is `dnode`, consumed by .joint_integration_weights(): the
#     weighting code is untouched.
#
#   * No mode-find, no line search. The local Gaussian scale comes from a
#     diagonal finite-difference of the OUTER log-marginal over the cell's own
#     grid neighbours -- values already computed for the base grid -- so the only
#     new inner solves are the off-centre CCD nodes themselves (the centre node
#     coincides with the cell and reuses its solve). Curvature for free, nodes
#     warm-started from the cell's inner mode.
#
# The node cloud of each refined cell is clamped to the cell's Voronoi half-box
# (half the distance to each neighbour on each axis), so clouds of distinct cells
# never overlap and never spill into an unrefined neighbour's mass. Combined with
# the mutually-non-adjacent cell selection, no region of hyperparameter space is
# counted twice.
#
# This refines over the LATENT block axes only; an active phi dispersion tensor
# is held fixed per cell (the global CCD treats phi the same way), and the
# refinement declines when a phi grid is active. It also requires a tensor base
# grid (the finite-difference stencil needs axis neighbours), declining on a
# scattered global-CCD base.

# Does local CCD refinement engage at this latent-axis count? Mirrors the global
# CCD's d >= 4 "auto" threshold: below it the tensor grid is cheap/dense and grid
# refinement suffices, so local CCD is pure overhead.
.joint_local_ccd_engage <- function(d_axes) {
    is.finite(d_axes) && d_axes >= 4L
}

# Per-axis up / down grid-neighbour row index for every cell. Two rows are
# neighbours along axis j when they are identical in every OTHER column (other
# latent axes AND phi columns) and adjacent in the sorted unique levels of axis
# j. Returns integer matrices `up` / `dn` [n x d] (NA where the cell sits at an
# edge of that axis), in u-space coordinates `U` keyed by the physical grid
# `key_cols` (all columns except the axis being differenced).
.joint_local_ccd_neighbors <- function(U, grid_phys, latent_cols) {
    n <- nrow(U)
    d <- ncol(U)
    up <- matrix(NA_integer_, n, d)
    dn <- matrix(NA_integer_, n, d)
    fmt <- function(v) sprintf("%.10g", v)
    all_cols <- seq_len(ncol(grid_phys))
    for (j in seq_len(d)) {
        col_j <- latent_cols[j]
        other <- setdiff(all_cols, col_j)
        if (length(other)) {
            parts <- lapply(other, function(cc) fmt(grid_phys[, cc]))
            key <- do.call(paste, c(parts, sep = "\r"))
        } else {
            key <- rep("", n)
        }
        for (g in unique(key)) {
            rows <- which(key == g)
            if (length(rows) < 2L) next
            ord  <- rows[order(U[rows, j])]
            for (p in seq_along(ord)) {
                if (p > 1L)            dn[ord[p], j] <- ord[p - 1L]
                if (p < length(ord))   up[ord[p], j] <- ord[p + 1L]
            }
        }
    }
    list(up = up, dn = dn)
}

# Diagonal local outer curvature at cell `c` from its grid neighbours, in
# u-space. Per axis the 3-point divided-difference second derivative of the log-
# marginal (unequal spacing safe) gives d2_j; the cell is a refinement candidate
# only if it has both neighbours on EVERY axis and is concave (d2_j < 0) on every
# axis -- a genuine interior local peak, the only place a local Gaussian is
# defined. Returns the per-axis sd (1 / sqrt(-d2)) and the per-axis Voronoi
# half-widths (half the u-distance to the lower / upper neighbour), or NULL when
# the cell is not a candidate.
.joint_local_ccd_cell_curv <- function(c, U, lm, up, dn, eps = 1e-8) {
    d <- ncol(U)
    sd      <- numeric(d)
    half_lo <- numeric(d)
    half_hi <- numeric(d)
    for (j in seq_len(d)) {
        iu <- up[c, j]; id <- dn[c, j]
        if (is.na(iu) || is.na(id)) return(NULL)
        u_c <- U[c, j]; u_hi <- U[iu, j]; u_lo <- U[id, j]
        l_c <- lm[c];   l_hi <- lm[iu];   l_lo <- lm[id]
        if (!all(is.finite(c(u_c, u_hi, u_lo, l_c, l_hi, l_lo)))) return(NULL)
        du_hi <- u_hi - u_c; du_lo <- u_c - u_lo
        if (du_hi <= 0 || du_lo <= 0) return(NULL)
        slope_hi <- (l_hi - l_c) / du_hi
        slope_lo <- (l_c - l_lo) / du_lo
        d2 <- 2 * (slope_hi - slope_lo) / (u_hi - u_lo)
        if (!is.finite(d2) || d2 >= -eps) return(NULL)
        sd[j]      <- 1 / sqrt(-d2)
        half_lo[j] <- 0.5 * du_lo
        half_hi[j] <- 0.5 * du_hi
    }
    list(sd = sd, half_lo = half_lo, half_hi = half_hi)
}

# Greedy mutually-non-adjacent selection: take the highest-weight candidate, drop
# it and all its grid neighbours from contention, repeat up to `max_cells`. Keeps
# refined node clouds from overlapping (adjacent peaked cells on one contiguous
# blob are re-sampling the same mass; one recentred design captures it once).
.joint_local_ccd_select <- function(cands, w, up, dn, max_cells) {
    ord <- cands[order(w[cands], decreasing = TRUE)]
    chosen  <- integer(0)
    blocked <- logical(max(c(up, dn, cands), na.rm = TRUE))
    for (c in ord) {
        if (length(chosen) >= max_cells) break
        if (isTRUE(blocked[c])) next
        chosen <- c(chosen, c)
        nb <- c(up[c, ], dn[c, ])
        nb <- nb[!is.na(nb)]
        blocked[c] <- TRUE
        if (length(nb)) blocked[nb] <- TRUE
    }
    chosen
}

# Refine a joint multi-block outer grid with local CCD node clouds.
#
# Args:
#   joint_grid   [n x p] physical outer-grid matrix (latent axes + any phi cols).
#   log_marginal [n] per-cell inner log-marginal (hyperprior already baked in).
#   modes        [n x n_x] inner modes (warm starts), or NULL.
#   dnode        [n] current outer design weights, or NULL (uniform tensor base).
#   latent_axes  character: the latent axis column names to refine over.
#   tags         named character per latent axis ("log"/"logit01"/"identity"),
#                or NULL to DECLINE (an unguessable-support axis, e.g. rho_car).
#   eval_nodes   function(theta_mat, warm) -> list(log_marginal, modes): evaluate
#                the inner marginal (hyperprior baked in) at new node rows
#                (full-coordinate matrix, colnames = colnames(joint_grid)), warm-
#                started from `warm` (the refined cell's inner mode, or NULL).
#   max_cells    cap on refined cells (hard cap on extra solve fan-out).
#   f0           CCD factorial-corner radius per whitened axis (INLA default 1.1).
#   verbose      announce the refinement summary.
#
# Returns list(joint_grid, log_marginal, modes, dnode, info) with the refined
# cells replaced by their node clouds, or NULL when refinement declines (NULL
# tags, no candidate peaked cells, or a degenerate layout).
.joint_local_ccd_refine <- function(joint_grid, log_marginal, modes = NULL,
                                     dnode = NULL, latent_axes, tags,
                                     eval_nodes, max_cells = 8L, f0 = 1.1,
                                     verbose = FALSE) {
    if (is.null(tags)) return(NULL)
    latent_cols <- match(latent_axes, colnames(joint_grid))
    if (anyNA(latent_cols)) return(NULL)
    d <- length(latent_cols)
    n <- nrow(joint_grid)
    if (d < 2L || n < 3L) return(NULL)

    dn_w <- dnode %||% rep(1, n)
    w    <- .joint_integration_weights(log_marginal, dnode)
    if (all(is.na(w))) return(NULL)

    # u-space latent coordinates (the curvature / CCD scale live on the
    # unconstrained axis where the posterior is closest to Gaussian).
    U <- matrix(0, n, d)
    for (j in seq_len(d)) U[, j] <- .joint_pareto_fwd(tags[j], joint_grid[, latent_cols[j]])

    nb <- .joint_local_ccd_neighbors(U, joint_grid, latent_cols)

    ccd   <- ccd_grid(d, f_0 = sqrt(d) * f0)
    delta <- ccd_weights(ccd)
    # Outermost design reach per (whitened) axis: the axial node sits this far in
    # z. The per-cell design scale is shrunk so the whole cloud fits the cell's
    # Voronoi box (node_reach * sd <= half), see below.
    node_reach <- max(abs(ccd$z))

    # Candidate interior peaked cells, with per-cell diagonal curvature.
    curv  <- vector("list", n)
    cands <- integer(0)
    for (c in seq_len(n)) {
        cc <- .joint_local_ccd_cell_curv(c, U, log_marginal, nb$up, nb$dn)
        if (is.null(cc)) next
        curv[[c]] <- cc
        cands <- c(cands, c)
    }
    if (length(cands) == 0L) return(NULL)

    chosen <- .joint_local_ccd_select(cands, w, nb$up, nb$dn, max_cells)
    if (length(chosen) == 0L) return(NULL)

    is_centre <- ccd$kind == "center"
    z_off <- ccd$z[!is_centre, , drop = FALSE]
    delta_off    <- delta[!is_centre]
    delta_centre <- delta[is_centre][1L]

    n_x <- if (is.matrix(modes)) ncol(modes) else 0L
    new_grid_blocks <- list()
    new_lm_blocks   <- list()
    new_mode_blocks <- list()
    new_dn_blocks   <- list()
    n_nodes_added   <- 0L

    for (c in chosen) {
        cc  <- curv[[c]]
        u_c <- U[c, ]
        # Shrink the design scale so the whole node cloud fits the cell's Voronoi
        # box (node_reach * sd <= the symmetric half-width on every axis). The
        # part of the local Gaussian beyond the box belongs to the neighbouring
        # cells -- which carry their own mass -- so truncating to the box keeps the
        # CCD an unbiased in-cell quadrature with valid design weights, instead of
        # clamping nodes (which would break the weight / position correspondence
        # and over-disperse). A symmetric shrink keeps the +/- pairs balanced, so a
        # flat integrand leaves the cell's first moment unchanged.
        half <- pmin(cc$half_lo, cc$half_hi)
        sd_design <- pmin(cc$sd, half / node_reach)
        u_nodes <- sweep(z_off %*% diag(sd_design, d, d), 2L, u_c, FUN = "+")
        for (j in seq_len(d)) {
            u_nodes[, j] <- pmin(pmax(u_nodes[, j], u_c[j] - cc$half_lo[j]),
                                 u_c[j] + cc$half_hi[j])
        }
        # Map back to physical theta; phi / non-latent columns are held at the
        # cell's values.
        theta_nodes <- matrix(rep(joint_grid[c, ], each = nrow(u_nodes)),
                              nrow = nrow(u_nodes))
        colnames(theta_nodes) <- colnames(joint_grid)
        for (j in seq_len(d)) {
            theta_nodes[, latent_cols[j]] <- .joint_pareto_inv(tags[j], u_nodes[, j])$theta
        }
        warm <- if (n_x > 0L) as.numeric(modes[c, ]) else NULL
        ev <- eval_nodes(theta_nodes, warm)
        lm_off <- as.numeric(ev$log_marginal)
        if (length(lm_off) != nrow(theta_nodes)) return(NULL)

        # Replacement block: centre node reuses the cell's own solve; off-centre
        # nodes carry the freshly evaluated marginal. Design weights Delta_c *
        # delta_j conserve the cell's mass on a flat integrand (sum_j delta_j = 1).
        grid_blk <- rbind(joint_grid[c, , drop = FALSE], theta_nodes)
        lm_blk   <- c(log_marginal[c], lm_off)
        dn_blk   <- dn_w[c] * c(delta_centre, delta_off)
        if (n_x > 0L) {
            mode_off <- if (is.matrix(ev$modes) && nrow(ev$modes) == nrow(theta_nodes))
                ev$modes else matrix(NA_real_, nrow(theta_nodes), n_x)
            mode_blk <- rbind(modes[c, , drop = FALSE], mode_off)
            new_mode_blocks[[length(new_mode_blocks) + 1L]] <- mode_blk
        }
        new_grid_blocks[[length(new_grid_blocks) + 1L]] <- grid_blk
        new_lm_blocks[[length(new_lm_blocks) + 1L]]     <- lm_blk
        new_dn_blocks[[length(new_dn_blocks) + 1L]]     <- dn_blk
        n_nodes_added <- n_nodes_added + nrow(theta_nodes)
    }

    keep <- setdiff(seq_len(n), chosen)
    out_grid <- rbind(joint_grid[keep, , drop = FALSE],
                      do.call(rbind, new_grid_blocks))
    colnames(out_grid) <- colnames(joint_grid)
    out_lm   <- c(log_marginal[keep], unlist(new_lm_blocks, use.names = FALSE))
    out_dn   <- c(dn_w[keep], unlist(new_dn_blocks, use.names = FALSE))
    out_modes <- if (n_x > 0L)
        rbind(modes[keep, , drop = FALSE], do.call(rbind, new_mode_blocks)) else NULL

    if (isTRUE(verbose)) {
        message(sprintf(
            "tulpa joint: local CCD refinement: %d cell(s), +%d nodes (%d -> %d cells)",
            length(chosen), n_nodes_added, n, nrow(out_grid)))
    }

    list(joint_grid   = out_grid,
         log_marginal = out_lm,
         modes        = out_modes,
         dnode        = out_dn,
         info         = list(n_cells_refined = length(chosen),
                             n_nodes_added   = n_nodes_added,
                             cells           = chosen,
                             n_cells_before  = n,
                             n_cells_after   = nrow(out_grid)))
}
