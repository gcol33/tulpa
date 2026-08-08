# Local central-composite-design refinement of the joint multi-block outer grid

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
#     Conservation is the statement on a FLAT integrand, and the integrand
#     refinement is selected for is peaked, so the two estimates a grid then
#     holds are not on the same footing: a refined cell's mass is re-estimated by
#     a 25-node rule (at d = 4) while every unrefined sibling keeps the base
#     grid's single atom, and the cloud's nodes sit nearer the peak than the
#     cell's own coordinate did, so the refined cell's share rises. That
#     asymmetry is real and is what `log_mass_ratio` reports per cell
#     (gcol33/tulpa#323). What it costs was MEASURED by coverage rather than
#     argued from the grid (gcol33/tulpa#320): over 150 seeds on a four-axis
#     coarse-grid fixture fit twice on the same data, pooled fixed-effect
#     coverage is 0.8800 refined against 0.8800 unrefined at nominal 0.95 and
#     0.7300 against 0.7267 at 0.80, on a standard error of 0.0126. The width
#     does move the way the asymmetry predicts -- the intercept's mean 95%
#     interval is 2.9% narrower with refinement on -- and that buys 0 of 300
#     trials at 0.95 and 1 at 0.80. The measurement bounds the calibration cost
#     under a percentage point; it does not resolve it to zero, 150 seeds having
#     no power below about one seed. So the estimator asymmetry stands as a known
#     and instrumented property of this refinement rather than a defect it is
#     worth spending an inner-solve budget to remove (gcol33/tulpa#319).
#
#     On the hyperparameter axis the same re-estimation is the point: the
#     `sigma_1` interval is more than fourfold sharper (0.2330 against 1.0590 on
#     that fixture) with half the posterior-median bias, while still covering 149
#     of 150 against a nominal 0.95, so what the refined cell's rising share
#     removes there is conservative-side slack a four-level grid left behind.
#
#   * No mode-find, no line search. The local Gaussian scale comes from a finite
#     difference of the OUTER log-marginal over the cell's own grid neighbours --
#     values already computed for the base grid -- so the only new inner solves
#     are the off-centre CCD nodes themselves (the centre node coincides with
#     the cell and reuses its solve). Curvature for free, nodes warm-started
#     from the cell's inner mode. The stencil reaches the corner neighbours as
#     well as the axis ones, so the scale is the MARGINAL spread per axis; the
#     diagonal alone gives the conditional spread, which on a correlated
#     posterior is narrower (gcol33/tulpa#316).
#
# The cloud earns the cell only where the cell's own outer log-marginal is close
# to the quadratic the cloud was placed from. A central composite design
# identifies a full quadratic exactly, so the least-squares residual of the
# nodes' measured log-marginals against that quadratic is the part of the cell
# the design cannot represent, and it is available from the nodes already
# evaluated. Above `skew_max` on that score the cell is put back as its own mass
# atom (.joint_local_ccd_misfit(), gcol33/tulpa#318): on an outer target that is
# quadratic in the transformed coordinate the score is identically zero and the
# refinement is untouched, while on a skewed one the design reports its own
# geometry rather than the cell's mass and the atom is measurably closer.
#
# What passing that score certifies is bounded, and the bound is the whole of
# it: the design can REPRESENT the cell. It says nothing about the cell's own
# coordinate being a REPRESENTATIVE POINT of the cell. The linear term sits in
# its own columns of the design matrix, so a cell whose outer log-marginal is a
# perfectly good quadratic that simply is not centred on the cell fits exactly
# and scores near zero however steep the gradient across it; a design centred
# there reproduces the local quadratic exactly and can still return node masses
# dominated by one corner. That displacement is the OTHER quantity the same
# least-squares fit estimates, and it is reported rather than discarded: the
# `offset` in `info` is the L2 norm of the whitened gradient, the cell's own peak
# measured from the cell's coordinate in units of the marginal spread the
# whitening used, carried for refined and declined cells alike. Nothing is gated
# on it -- a gradient across the cell is a cross-cell estimator question, an axis
# orthogonal to the local shape `skew_max` reads (gcol33/tulpa#321).
#
# That norm is a displacement and not yet a comparable one: it says nothing about
# how sharply the cell's log-marginal falls away in the direction the gradient
# points, so two cells with the same norm and curvature an order of magnitude
# apart are displaced by very different amounts. The same fit estimates the
# curvature too -- a central composite design identifies a full quadratic
# exactly, so the whitened Hessian sits in the same coefficient vector
# (H_jj = 2 c_jj, H_jk = c_jk for j < k) -- and `mode_gain = 0.5 g' (-H)^-1 g` is
# the displacement in local curvature units, i.e. the nats the quadratic model
# predicts the log-density gains by moving the expansion centre to the cell's own
# fitted peak. It declines to NA where -H is not positive definite: a cell whose
# fitted quadratic is not concave has no interior peak to be displaced from, and
# the ratio would report a number carrying no such reading (gcol33/tulpa#324).
#
# The third axis is mass, and neither a shape score nor a centring score reaches
# it. A refined cell carries two estimates of its own mass -- the coarse atom
# `Delta_c exp(ell_c)` the base grid gave it, and the refined cloud
# `Delta_c sum_j delta_j exp(ell_j)` its nodes give it -- so their ratio,
#   log(M1 / M0) = logSumExp_j(log delta_j + ell_j - ell_c),
# over the full node set INCLUDING the centre, is post-processing of numbers
# already in hand and costs no inner solve. That comparison is the embedded-rule
# local error indicator classical adaptive cubature uses to decide whether a
# subdivided region's estimate is comparable to its unrefined siblings': two
# rules of different degree on the SAME region, their difference the indicator.
# This grid computes both rules, so the indicator is free. `log_mass_ratio`, the
# two masses it is formed from, and `max_node_weight` -- the share the single
# largest node takes of its own cell's refined mass -- are recorded per cell
# (gcol33/tulpa#323).
#
# The three scores are orthogonal, and a cell can fail any one of them with the
# other two clean: `misfit` is non-quadraticity (can the design represent the
# cell's shape?), `offset` / `mode_gain` are off-centring (is the cell's
# coordinate a representative point of it?), and `log_mass_ratio` is mass
# correction (did refining change how much mass this cell competes for?). None
# of the three gates; `skew_max` reads `misfit` and nothing else.
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

# Local outer curvature at cell `c` from its grid neighbours, in u-space.
#
# Per axis the 3-point divided-difference second derivative of the log-marginal
# (unequal spacing safe) gives d2_j; the cell is a refinement candidate only if
# it has both neighbours on EVERY axis and is concave (d2_j < 0) on every axis --
# a genuine interior local peak, the only place a local Gaussian is defined.
#
# `1 / sqrt(-d2_j)` is the CONDITIONAL scale along axis j: the spread with every
# other axis held at the cell. The summary reports MARGINAL spreads, and on a
# correlated outer posterior -- a sigma-alpha copy ridge is one -- the two differ
# by sqrt(H_jj (H^-1)_jj), so a design scaled by the conditional spread stands in
# for a posterior narrower than the marginal it is read as (gcol33/tulpa#316).
# The mixed second derivatives that close the gap come from the cell's own
# CORNER grid neighbours -- the axis neighbour OF an axis neighbour, which a cell
# interior on every axis always has, and which the tensor base already evaluated
# -- so `sd` costs no extra inner solve.
#
# Only the SCALE changes. The design stays axis-aligned, because the summary
# reads a weighted quantile over the refined grid and on design weights that is
# close to the design's own per-axis EXTENT: an axis-aligned design puts an axial
# node at f_0 * sd_j on coordinate j, while a design rotated by the Cholesky
# factor of the same covariance puts it at f_0 * L[j, k] and reports an interval
# that depends on the arbitrary order of the axes.
#
# Returns the per-axis sd, the per-axis Voronoi half-widths (half the u-distance
# to the lower / upper neighbour) and `sd_marginal` (`NULL` when a corner is
# missing -- a grid a previous pass already spliced nodes into -- or the local
# Hessian is not negative definite, which a wide finite-difference window can
# easily produce), or NULL when the cell is not a candidate.
.joint_local_ccd_cell_curv <- function(c, U, lm, up, dn, eps = 1e-8) {
    d <- ncol(U)
    sd      <- numeric(d)
    half_lo <- numeric(d)
    half_hi <- numeric(d)
    H       <- matrix(0, d, d)
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
        H[j, j]    <- d2
    }
    list(sd = sd, half_lo = half_lo, half_hi = half_hi,
         sd_marginal = .joint_local_ccd_marginal_sd(c, U, lm, up, dn, H))
}

# Per-axis MARGINAL sd at cell `c`: `sqrt(diag((-H)^-1))` where `H` carries the
# diagonal second derivatives already computed and the mixed ones differenced
# over the cell's corner neighbours,
#   H_jk = (l(+j+k) - l(+j-k) - l(-j+k) + l(-j-k)) / ((u_hi_j - u_lo_j)(u_hi_k - u_lo_k)),
# the same central-difference stencil `.joint_ccd_fd_stencil()` uses for the
# global CCD's mode Hessian. NULL when any corner is missing or `-H` is not
# positive definite, which is the caller's signal to keep the conditional scale.
.joint_local_ccd_marginal_sd <- function(c, U, lm, up, dn, H) {
    d <- ncol(U)
    if (d < 2L) return(NULL)
    for (j in seq_len(d - 1L)) for (k in (j + 1L):d) {
        pj <- up[c, j]; mj <- dn[c, j]
        idx <- c(up[pj, k], dn[pj, k], up[mj, k], dn[mj, k])
        if (anyNA(idx)) return(NULL)
        den <- (U[up[c, j], j] - U[dn[c, j], j]) * (U[up[c, k], k] - U[dn[c, k], k])
        v <- (lm[idx[1L]] - lm[idx[2L]] - lm[idx[3L]] + lm[idx[4L]]) / den
        if (!is.finite(v)) return(NULL)
        H[j, k] <- v; H[k, j] <- v
    }
    negH <- -0.5 * (H + t(H))
    # Positive definiteness is what makes the inverse a covariance; a wide
    # finite-difference window can leave the estimate indefinite even when every
    # diagonal entry is concave.
    if (is.null(tryCatch(chol(negH), error = function(e) NULL))) return(NULL)
    S <- tryCatch(solve(negH), error = function(e) NULL)
    if (is.null(S)) return(NULL)
    s <- sqrt(diag(S))
    if (!all(is.finite(s)) || any(s <= 0)) return(NULL)
    s
}

# How far the cell's own outer log-marginal is from the quadratic the design was
# placed from, read off the design's OWN nodes.
#
# A central composite design is built to identify a full quadratic response
# exactly, so the least-squares residual of the measured node log-marginals
# against intercept + gradient + symmetric Hessian in the whitened offset
# `z = (u - u_c) / sd` is the part of the cell the design cannot see. It costs no
# inner solve: the nodes are evaluated whatever the score says.
#
# Reported as a standardized cubic magnitude, on the same convention the
# inner-Laplace `gamma_3` uses: a cubic term of standardized skewness `g`
# contributes `g |z|^3 / 6` at whitened radius `|z|`, so
# `gamma = 6 * rms(residual) / rms(|z|^3)`. The whitening is the cell's own
# MARGINAL spread (the design's scale after the Voronoi shrink is a smaller
# number, and the quantity being standardized is a property of the posterior, not
# of the box it was truncated to).
#
# `offset` is the second reading of the same fit, and a different axis: the L2
# norm of the whitened gradient `fit$coefficients[2:(d + 1)]`, i.e. how far the
# cell's own peak sits from the cell's coordinate, in the same marginal-spread
# units the whitening put the design in. `misfit` cannot see it by construction,
# the linear term having its own columns, so an off-centre but exactly quadratic
# cell scores zero misfit at any `offset` (gcol33/tulpa#321).
#
# `mode_gain` is that same displacement in the cell's own curvature units. The
# design identifies the full quadratic, so the whitened Hessian is in the same
# coefficient vector -- the quadratic columns are laid out `(j, k)` for `j <= k`,
# so `H[j, j] = 2 c_jj` and `H[j, k] = H[k, j] = c_jk` -- and
# `0.5 g' (-H)^-1 g` is the log-density gain in nats the quadratic model predicts
# from moving the expansion centre to the cell's fitted peak. Comparable across
# cells whose curvature differs, which `offset` is not; `offset` is kept because
# it is the unscaled reading and survives cases the scaled one cannot be formed
# in (gcol33/tulpa#324).
#
# All three NA when the design cannot identify the quadratic at all, which the
# caller treats the same as a failing score: an unverified local Gaussian.
# `offset` alone is NA on a rank-deficient design, where the aliased linear
# coefficients come back NA and a norm over them would report a displacement that
# was never estimated. `mode_gain` is additionally NA wherever `-H` is not
# positive definite -- a saddle-shaped cell has no interior peak to be displaced
# from -- the same guard `.joint_local_ccd_marginal_sd()` applies to the
# finite-difference Hessian.
.joint_local_ccd_misfit <- function(u_nodes, u_c, lm_nodes, sd) {
    d <- length(u_c)
    none <- list(misfit = NA_real_, offset = NA_real_, mode_gain = NA_real_)
    if (!all(is.finite(lm_nodes)) || !all(is.finite(sd)) || any(sd <= 0))
        return(none)
    Z <- sweep(sweep(u_nodes, 2L, u_c, FUN = "-"), 2L, sd, FUN = "/")
    if (!all(is.finite(Z))) return(none)
    cols <- list(rep(1, nrow(Z)))
    for (j in seq_len(d)) cols[[length(cols) + 1L]] <- Z[, j]
    for (j in seq_len(d)) for (k in j:d)
        cols[[length(cols) + 1L]] <- Z[, j] * Z[, k]
    X <- do.call(cbind, cols)
    if (nrow(X) <= ncol(X)) return(none)
    fit <- tryCatch(stats::lm.fit(X, lm_nodes), error = function(e) NULL)
    if (is.null(fit)) return(none)
    r3 <- sqrt(mean((rowSums(Z^2)^1.5)^2))
    if (!is.finite(r3) || r3 <= 0) return(none)
    g <- fit$coefficients[seq_len(d) + 1L]
    ok_g <- all(is.finite(g))
    list(misfit    = 6 * sqrt(mean(fit$residuals^2)) / r3,
         offset    = if (ok_g) sqrt(sum(g^2)) else NA_real_,
         mode_gain = if (ok_g)
             .joint_local_ccd_mode_gain(g, fit$coefficients, d) else NA_real_)
}

# `0.5 * g' (-H)^-1 g` in nats, with `H` read back out of the quadratic columns
# of the design matrix `.joint_local_ccd_misfit()` built. The column walk here
# mirrors that construction exactly -- intercept, then the `d` linear columns,
# then `(j, k)` for `k >= j` in ascending `j` -- so the coefficient a term is
# read from is the coefficient it was fitted for. The cross terms `z_j z_k`
# appear once each, hence the factor 2 on the diagonal and none off it.
#
# NA on any aliased quadratic coefficient (an unestimated Hessian entry) and on a
# non-concave fit, where `(-H)^-1` is not a covariance and the quadratic form is
# not a displacement.
.joint_local_ccd_mode_gain <- function(g, coefs, d) {
    H <- matrix(0, d, d)
    p <- d + 1L
    for (j in seq_len(d)) for (k in j:d) {
        p <- p + 1L
        cjk <- coefs[p]
        if (!is.finite(cjk)) return(NA_real_)
        if (j == k) H[j, j] <- 2 * cjk else { H[j, k] <- cjk; H[k, j] <- cjk }
    }
    negH <- -H
    R <- tryCatch(chol(negH), error = function(e) NULL)
    if (is.null(R)) return(NA_real_)
    v <- backsolve(R, backsolve(R, g, transpose = TRUE))
    val <- 0.5 * sum(g * v)
    if (!is.finite(val)) NA_real_ else val
}

# The cell's two estimates of its own mass, and their ratio.
#
# `delta` are the design weights of the whole node set INCLUDING the centre (a
# partition of unity) and `lm_all` the matching log-marginals, the centre first.
# The coarse atom is `Delta_c exp(ell_c)` and the refined cloud is
# `Delta_c sum_j delta_j exp(ell_j)`, so
#   log(M1 / M0) = logSumExp_j(log delta_j + ell_j) - ell_c,
# and `Delta_c` cancels out of it. Both masses are returned on the log scale with
# `Delta_c` carried, so they are the cell's actual contributions to the integral
# rather than a normalized pair, and their difference is the ratio exactly.
#
# `max_node_weight` is the share the single largest node takes of its own cell's
# refined mass. It is the one weight-conditioning reading worth keeping here:
# `ccd_weights()` is strictly positive and sums to 1, so `sum|w| / |sum w|` is
# identically 1 and the weight entropy is a constant of the design.
.joint_local_ccd_mass <- function(delta, lm_all, dn_c) {
    none <- list(log_mass_ratio = NA_real_, log_mass_coarse = NA_real_,
                 log_mass_refined = NA_real_, max_node_weight = NA_real_)
    if (length(delta) != length(lm_all) || length(delta) == 0L) return(none)
    if (!all(is.finite(delta)) || any(delta <= 0)) return(none)
    if (!all(is.finite(lm_all)) || !is.finite(dn_c) || dn_c <= 0) return(none)
    lv  <- log(delta) + lm_all
    lse <- .tulpa_logsumexp(lv)
    if (!is.finite(lse)) return(none)
    list(log_mass_ratio   = lse - lm_all[1L],
         log_mass_coarse  = log(dn_c) + lm_all[1L],
         log_mass_refined = log(dn_c) + lse,
         max_node_weight  = exp(max(lv) - lse))
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
#   eval_nodes   function(theta_mat, warm) -> list(log_marginal, modes,
#                cov_blocks): evaluate the inner marginal at new node rows
#                (full-coordinate matrix, colnames = colnames(joint_grid)), warm-
#                started from `warm` (the refined cell's inner mode, or NULL).
#   max_cells    cap on refined cells (hard cap on extra solve fan-out).
#   f0           CCD factorial-corner radius per whitened axis (INLA default 1.1).
#   skew_max     a cell keeps its cloud only while `.joint_local_ccd_misfit()`'s
#                `misfit` stays below this; above it the cell is put back as its
#                own mass atom (gcol33/tulpa#318). The same call's `offset` /
#                `mode_gain`, and the cell's coarse-vs-refined mass comparison,
#                are recorded for every cell either way and gate nothing
#                (gcol33/tulpa#321, #323, #324).
#   verbose      announce the refinement summary.
#   cov_blocks   length-n list of per-cell fixed-effect covariance blocks, or
#                NULL. Carried cell-for-cell exactly like `modes`, so a refined
#                grid keeps the retention aligned with the weights it is
#                integrated against (gcol33/tulpa#307) instead of declining.
#
# Returns list(joint_grid, log_marginal, modes, cov_blocks, dnode, info) with the
# refined cells replaced by their node clouds, or NULL when refinement declines
# (NULL tags, no candidate peaked cells, or a degenerate layout).
.joint_local_ccd_refine <- function(joint_grid, log_marginal, modes = NULL,
                                     dnode = NULL, latent_axes, tags,
                                     eval_nodes, max_cells = 8L, f0 = 1.1,
                                     verbose = FALSE, cov_blocks = NULL,
                                     skew_max = .nl_diag("gamma3_ok")) {
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
    have_cov <- is.list(cov_blocks) && length(cov_blocks) == n
    new_grid_blocks <- list()
    new_lm_blocks   <- list()
    new_mode_blocks <- list()
    new_cov_blocks  <- list()
    new_dn_blocks   <- list()
    n_nodes_added   <- 0L
    # Each refined cell's share of the BASE grid's integration weight, before
    # any node was placed. `design_mass` on the fit is the post-refinement
    # share, which is a different number: the cloud's nodes sit nearer the peak
    # than the cell's own coordinate did, so refining a cell raises the share
    # the refined region holds. Reporting both says how concentrated the base
    # grid already was, separately from how much the refinement concentrated it
    # (gcol33/tulpa#316). The gap between the two is the cross-cell estimator
    # asymmetry the header describes; `log_mass_ratio` is its per-cell reading
    # and coverage bounds what it costs (gcol33/tulpa#319, gcol33/tulpa#320).
    cell_share      <- numeric(0)
    refined         <- integer(0)
    misfit          <- numeric(0)
    offset          <- numeric(0)
    mode_gain       <- numeric(0)
    declined_cells  <- integer(0)
    declined_misfit <- numeric(0)
    declined_offset <- numeric(0)
    declined_gain   <- numeric(0)
    # The coarse-vs-refined mass comparison, per cell, on both sides of the gate
    # (gcol33/tulpa#323). Accumulated as a list of the four readings so the two
    # sides stay one code path.
    mass_kept     <- list()
    mass_declined <- list()

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
        #
        # The scale is the MARGINAL spread where the corner stencil gives one,
        # since that is what the per-axis summary reads; the diagonal stencil's
        # conditional spread is the fallback (gcol33/tulpa#316).
        half <- pmin(cc$half_lo, cc$half_hi)
        sd_design <- pmin(cc$sd_marginal %||% cc$sd, half / node_reach)
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

        # The cloud stands in for the cell's mass only while the cell's own outer
        # log-marginal is close to the quadratic the cloud was placed from. Where
        # it is not, the design reports its own geometry instead of the cell's
        # mass and the cell is measurably better off as the atom it was
        # (gcol33/tulpa#318). The score is read off the nodes just evaluated, so
        # the decision costs no further solve -- the ones already spent are the
        # price of finding out. Passing it says the design can represent the
        # cell's shape, and only that: the same fit's `offset` / `mode_gain` are
        # how far the cell's own peak sits from the coordinate the cloud was
        # centred on -- unscaled, and in the cell's own curvature units -- which a
        # third-order score cannot see and which nothing here gates on
        # (gcol33/tulpa#321, #324).
        fit_c <- .joint_local_ccd_misfit(rbind(u_c, u_nodes), u_c,
                                         c(log_marginal[c], lm_off),
                                         cc$sd_marginal %||% cc$sd)
        # Both mass estimates of this cell are now in hand -- the coarse atom the
        # base grid gave it and the cloud its own nodes give it -- so their
        # comparison is arithmetic, and it is recorded whichever way the gate
        # goes: the nodes were evaluated before the score was read
        # (gcol33/tulpa#323).
        mass_c <- .joint_local_ccd_mass(c(delta_centre, delta_off),
                                        c(log_marginal[c], lm_off), dn_w[c])

        mis <- fit_c$misfit
        if (!isTRUE(mis < skew_max)) {
            declined_cells  <- c(declined_cells, c)
            declined_misfit <- c(declined_misfit, mis)
            declined_offset <- c(declined_offset, fit_c$offset)
            declined_gain   <- c(declined_gain, fit_c$mode_gain)
            mass_declined[[length(mass_declined) + 1L]] <- mass_c
            next
        }

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
        if (have_cov) {
            cov_off <- if (is.list(ev$cov_blocks) &&
                           length(ev$cov_blocks) == nrow(theta_nodes))
                ev$cov_blocks else vector("list", nrow(theta_nodes))
            new_cov_blocks[[length(new_cov_blocks) + 1L]] <-
                c(cov_blocks[c], cov_off)
        }
        new_grid_blocks[[length(new_grid_blocks) + 1L]] <- grid_blk
        new_lm_blocks[[length(new_lm_blocks) + 1L]]     <- lm_blk
        new_dn_blocks[[length(new_dn_blocks) + 1L]]     <- dn_blk
        cell_share <- c(cell_share, w[c])
        refined    <- c(refined, c)
        misfit     <- c(misfit, mis)
        offset     <- c(offset, fit_c$offset)
        mode_gain  <- c(mode_gain, fit_c$mode_gain)
        mass_kept[[length(mass_kept) + 1L]] <- mass_c
        n_nodes_added <- n_nodes_added + nrow(theta_nodes)
    }

    # One numeric vector per mass reading, in cell order, on each side of the
    # gate. An empty side gives a zero-length numeric, matching the empty
    # `misfit` / `offset` vectors beside it.
    mass_col <- function(acc, nm)
        vapply(acc, function(m) m[[nm]], numeric(1))

    keep <- setdiff(seq_len(n), refined)
    out_grid <- rbind(joint_grid[keep, , drop = FALSE],
                      do.call(rbind, new_grid_blocks))
    colnames(out_grid) <- colnames(joint_grid)
    out_lm   <- c(log_marginal[keep], unlist(new_lm_blocks, use.names = FALSE))
    out_dn   <- c(dn_w[keep], unlist(new_dn_blocks, use.names = FALSE))
    out_modes <- if (n_x > 0L)
        rbind(modes[keep, , drop = FALSE], do.call(rbind, new_mode_blocks)) else NULL
    out_cov <- if (have_cov)
        c(cov_blocks[keep], unlist(new_cov_blocks, recursive = FALSE)) else NULL

    if (isTRUE(verbose)) {
        message(sprintf(
            "tulpa joint: local CCD refinement: %d cell(s), +%d nodes (%d -> %d cells)",
            length(refined), n_nodes_added, n, nrow(out_grid)))
        if (length(declined_cells)) {
            message(sprintf(
                paste0("tulpa joint: local CCD refinement: %d cell(s) put back as ",
                       "mass atoms, local quadratic misfit %.3f (max) >= %.3f"),
                length(declined_cells), max(declined_misfit, na.rm = TRUE),
                skew_max))
        }
    }

    # Which kind of weight each output row carries. A carried-over base cell
    # holds the mass of its own cell; a node of a replacement cloud holds a
    # partition-of-unity share of its cell's mass, placed at the design's radius
    # rather than where that share sits. The two are on the same footing for the
    # moments (which is what the design reproduces) and not for a cumulative sum
    # (gcol33/tulpa#311), so the grid says which it carries instead of leaving a
    # consumer to read one kind off `integration`.
    out_kind <- c(rep("mass", length(keep)),
                  rep("design", nrow(out_grid) - length(keep)))

    list(joint_grid   = out_grid,
         log_marginal = out_lm,
         modes        = out_modes,
         cov_blocks   = out_cov,
         dnode        = out_dn,
         weight_kind  = out_kind,
         info         = list(n_cells_refined = length(refined),
                             n_nodes_added   = n_nodes_added,
                             cells           = refined,
                             cell_share      = cell_share,
                             misfit          = misfit,
                             offset          = offset,
                             mode_gain       = mode_gain,
                             log_mass_ratio   = mass_col(mass_kept, "log_mass_ratio"),
                             log_mass_coarse  = mass_col(mass_kept, "log_mass_coarse"),
                             log_mass_refined = mass_col(mass_kept, "log_mass_refined"),
                             max_node_weight  = mass_col(mass_kept, "max_node_weight"),
                             skew_max        = skew_max,
                             cells_declined  = declined_cells,
                             misfit_declined = declined_misfit,
                             offset_declined = declined_offset,
                             mode_gain_declined = declined_gain,
                             log_mass_ratio_declined =
                                 mass_col(mass_declined, "log_mass_ratio"),
                             log_mass_coarse_declined =
                                 mass_col(mass_declined, "log_mass_coarse"),
                             log_mass_refined_declined =
                                 mass_col(mass_declined, "log_mass_refined"),
                             max_node_weight_declined =
                                 mass_col(mass_declined, "max_node_weight"),
                             n_cells_declined = length(declined_cells),
                             n_cells_before  = n,
                             n_cells_after   = nrow(out_grid),
                             n_design_nodes  = sum(out_kind == "design")))
}
