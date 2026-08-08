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
# The same stencil that scales the design also says what the base grid's own mass
# estimate throws away, and that reading needs no node cloud at all. A cell's
# shipped mass is the midpoint atom `M_c^(0) = Delta exp(ell_c)` --
# `.joint_integration_weights()` softmaxes `log_marginal`, and the default grid
# being geometric is uniform in `u = log theta` and carries no volume element --
# so the GRADIENT of the log-marginal across the cell is invisible to it by
# construction. That is the `offset` blind spot one layer down: a cell read as if
# its own coordinate were a representative point of it. Integrating the local
# quadratic `ell(u_c + u) ~ ell_c + g'u - 0.5 u'A u` over the cell's own box
# rather than evaluating it at the centre gives
#   M_c^(Q) = exp(ell_c + 0.5 g'A^-1 g) (2 pi)^(d/2) |A|^(-1/2)
#             P{N(A^-1 g, A^-1) in box_c},
# whose `0.5 g'A^-1 g` is precisely `mode_gain`, the nats the atom drops. It
# costs no inner solve: `g` and the diagonal of `A` are the first and second
# divided differences of the SAME three points the curvature already came from
# (`.joint_local_ccd_diff3()`), all of them base-grid values (gcol33/tulpa#326).
#
# Two things about the form. First, it is the exact box integral and not the
# second-order expansion of the integrand,
# `Delta exp(ell_c) [1 + (1/6) sum_i h_i^2 (ell_ii + g_i^2)]`, which is a
# `|g_i h_i| << 1` statement and so fails on exactly the cells the correction
# exists for: at `g h` of 0.5 / 1 / 2 / 3 / 4.44 nats it recovers 98.8% / 95.1% /
# 82.0% / 64.1% / 38.5% of the exact correction, degrades monotonically into the
# regime of interest, and can go negative. 4.44 nats is the displacement the
# fixture that motivated this carries. Second, `A` is taken DIAGONAL, which turns
# the rectangle probability into a product of `pnorm` differences over the axes
# -- closed form, no new dependency -- and, more than cheapness, keeps the
# decline PER AXIS: an MVN rectangle probability needs `A` positive definite,
# while the integral over a BOUNDED box is finite at either sign of `a_i`, so a
# cell concave on three axes and convex on the fourth contributes an exact factor
# on the three instead of declining whole. The correlated form is worth its
# rectangle routine only if the diagonal one moves the read.
#
# Mass is half of what the midpoint atom drops, and the other half is PLACE. The
# per-axis read summarises axis j as a weighted quantile over the grid
# COORDINATES (`.nl_axis_quantiles()` hands `tg[, j]` to
# `.nl_summary_quantile()`), so a cell whose log-marginal carries a gradient
# contributes its whole mass at a point that mass is not centred on. Correcting
# the mass alone gives such a cell the right weight in the wrong place, and the
# two fail independently -- a cell can have either one right and the other wrong
# (gcol33/tulpa#327). The place is the same local quadratic's first moment over
# the same box, so per axis it is the doubly-truncated normal mean
#   mu = g/a, s = a^(-1/2), alpha = (-h_lo - mu)/s, beta = (h_hi - mu)/s,
#   ubar = mu + s (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha)),
# reducing to the cell's own coordinate exactly at `g = 0`, which is the whole
# content of the current placement. It reads the same `(g, A)` off the same
# stencil, so like the mass it costs no inner solve, and it declines per axis for
# the same reason: the first moment over a BOUNDED box is finite at either sign
# of `a_i`, so a convex axis is integrated rather than throwing the cell away.
#
# Two properties of it are load-bearing. The barycentre lies INSIDE the cell's
# box -- it is the first moment of a positive density over that box -- so an atom
# moved to it never crosses into a neighbour's territory and the grid stays a
# partition of the same space; the runtime check that it does is a check on the
# arithmetic, not on the integral. And it applies to the HYPERPARAMETER axes
# only: `v` IS the outer coordinate, so its barycentre is a property of outer
# cell geometry alone. A fixed effect is not summarised this way at all --
# `.nested_fixed_moments()` marginalizes each cell's OWN `(beta_k, Vb_k)` from
# its own inner solve (gcol33/tulpa#305) -- so moving that cell's location asks
# what `beta` would have been at the moved point, which is another inner solve or
# an interpolation across cells. Anything reported on a fixed effect stays where
# its own solve put it.
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

# Three-point divided differences of the outer log-marginal at the MIDDLE point
# of an unequally spaced stencil.
#
# Three points identify exactly one quadratic, so its derivatives at `u_c` are
# the only first and second derivatives the stencil carries. Written on the two
# one-sided slopes
#   slope_hi = (l_hi - l_c) / du_hi,  slope_lo = (l_c - l_lo) / du_lo,
# they are
#   d1 = (slope_hi du_lo + slope_lo du_hi) / (du_lo + du_hi),
#   d2 = 2 (slope_hi - slope_lo) / (u_hi - u_lo),
# the spacing-weighted MEAN of the two slopes and their DIFFERENCE: the same pair
# of numbers read two ways, collapsing to the central differences
# (l_hi - l_lo) / (2h) and (l_hi - 2 l_c + l_lo) / h^2 on equal spacing. Both are
# returned from the one call because a caller wanting the curvature has already
# paid for the gradient and vice versa; differencing the stencil twice would be
# the same arithmetic written down twice.
#
# NULL on a non-finite value or a non-increasing stencil, which is the caller's
# signal that this axis carries no local quadratic at all.
.joint_local_ccd_diff3 <- function(u_lo, u_c, u_hi, l_lo, l_c, l_hi) {
    if (!all(is.finite(c(u_lo, u_c, u_hi, l_lo, l_c, l_hi)))) return(NULL)
    du_hi <- u_hi - u_c
    du_lo <- u_c - u_lo
    if (du_hi <= 0 || du_lo <= 0) return(NULL)
    slope_hi <- (l_hi - l_c) / du_hi
    slope_lo <- (l_c - l_lo) / du_lo
    d1 <- (slope_hi * du_lo + slope_lo * du_hi) / (du_lo + du_hi)
    d2 <- 2 * (slope_hi - slope_lo) / (u_hi - u_lo)
    if (!is.finite(d1) || !is.finite(d2)) return(NULL)
    list(d1 = d1, d2 = d2, half_lo = 0.5 * du_lo, half_hi = 0.5 * du_hi)
}

# The whole local quadratic at cell `c`, one axis at a time: the gradient `g`,
# the diagonal of the curvature `d2`, and the cell's own Voronoi half-widths
# (half the u-distance to each neighbour). NULL when the cell is missing a
# neighbour on any axis -- a boundary cell has no centred stencil there, and a
# one-sided one would report a gradient and a curvature the grid never measured.
#
# No sign condition is imposed here. Concavity is what a local GAUSSIAN needs and
# `.joint_local_ccd_cell_curv()` gates on it; the box integral below is finite at
# either sign, so it reads the same stencil without the gate.
.joint_local_ccd_cell_stencil <- function(c, U, lm, up, dn) {
    d <- ncol(U)
    g <- numeric(d); d2 <- numeric(d)
    half_lo <- numeric(d); half_hi <- numeric(d)
    for (j in seq_len(d)) {
        iu <- up[c, j]; id <- dn[c, j]
        if (is.na(iu) || is.na(id)) return(NULL)
        fd <- .joint_local_ccd_diff3(U[id, j], U[c, j], U[iu, j],
                                     lm[id], lm[c], lm[iu])
        if (is.null(fd)) return(NULL)
        g[j]       <- fd$d1
        d2[j]      <- fd$d2
        half_lo[j] <- fd$half_lo
        half_hi[j] <- fd$half_hi
    }
    list(g = g, d2 = d2, half_lo = half_lo, half_hi = half_hi)
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
# Returns the stencil's own `g` / `d2`, the per-axis sd, the per-axis Voronoi
# half-widths (half the u-distance to the lower / upper neighbour) and
# `sd_marginal` (`NULL` when a corner is missing -- a grid a previous pass
# already spliced nodes into -- or the local Hessian is not negative definite,
# which a wide finite-difference window can easily produce), or NULL when the
# cell is not a candidate.
.joint_local_ccd_cell_curv <- function(c, U, lm, up, dn, eps = 1e-8) {
    st <- .joint_local_ccd_cell_stencil(c, U, lm, up, dn)
    if (is.null(st)) return(NULL)
    if (any(st$d2 >= -eps)) return(NULL)
    d <- ncol(U)
    H <- matrix(0, d, d)
    diag(H) <- st$d2
    list(g           = st$g,
         d2          = st$d2,
         sd          = 1 / sqrt(-st$d2),
         half_lo     = st$half_lo,
         half_hi     = st$half_hi,
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

# The log gap between the two tail probabilities has to be resolved against the
# absolute size of the logs it is a difference of: `pnorm(log.p = TRUE)` carries
# each to about `eps` of its own magnitude, so a gap below this fraction of that
# magnitude is mostly rounding. Below it the box is a sliver of the Gaussian it
# sits in, the exact factor is 1 to within `a w^2 / 24`, and the numeric route
# reads it exactly.
.LCCD_BOX_GAP <- 1e-4

# log(Phi(z_hi) - Phi(z_lo)), formed on whichever tail keeps the two arguments
# from cancelling: in the upper half-line the two `pnorm` values are both within
# rounding of 1 and their difference is carried by the upper tails instead. Both
# branches take the difference through `expm1` of a log gap, so a pair of nodes
# separated by a hair of probability is as accurate as a pair separated by all
# of it -- until the gap itself falls under the rounding of the logs, where the
# reading is refused (NA) rather than returned wrong.
.joint_local_ccd_log_pnorm_diff <- function(z_lo, z_hi) {
    if (!is.finite(z_lo) || !is.finite(z_hi) || z_hi <= z_lo) return(NA_real_)
    if (z_lo > 0) {
        a <- stats::pnorm(z_lo, lower.tail = FALSE, log.p = TRUE)
        b <- stats::pnorm(z_hi, lower.tail = FALSE, log.p = TRUE)
    } else {
        a <- stats::pnorm(z_hi, log.p = TRUE)
        b <- stats::pnorm(z_lo, log.p = TRUE)
    }
    if (!is.finite(a) || !is.finite(b)) return(NA_real_)
    gap <- b - a
    if (gap >= 0 || -gap < .LCCD_BOX_GAP * max(1, abs(a))) return(NA_real_)
    a + log(-expm1(gap))
}

# Above this the closed form's two divergent pieces are no longer being
# subtracted in double precision. `0.5 g^2 / a` is the height of the local peak
# above the cell, and it and the log of the `pnorm` difference grow against each
# other -- a steep gradient or a flat curvature sends both out -- while their sum
# stays finite, so the absolute error in the log factor is about
# `eps * 0.5 g^2 / a`; at 1e10 that is 2e-6 nats, and the numeric route takes
# over before it grows past it.
.LCCD_BOX_CANCEL <- 1e10

# One axis's factor of the local-quadratic box integral, on the log scale.
#
# The cell's mass under the local quadratic separates across axes when `A` is
# diagonal, so axis i contributes
#   I_i = int_{-h_lo}^{h_hi} exp(g_i u - 0.5 a_i u^2) du,
# and the factor reported here is `I_i / (h_lo + h_hi)`: the axis's integral
# divided by the axis's own extent, so the product over axes is the cell's mass
# divided by the cell's own box volume and the whole thing is a dimensionless
# MULTIPLIER on the midpoint atom `Delta_c exp(ell_c)`, equal to 1 on a cell the
# log-marginal is flat across.
#
# `a_i > 0` (the concave axis) is the Gaussian case. Completing the square with
# `mu = g/a` and `s = a^(-1/2)`,
#   I_i = exp(0.5 g^2/a) sqrt(2 pi / a) (Phi((h_hi - mu)/s) - Phi((-h_lo - mu)/s)),
# closed form and needing no rectangle probability. It is evaluated in logs
# throughout: `0.5 g^2/a` is unbounded as the local peak leaves the box while the
# `pnorm` difference vanishes at the matching rate, and only their logs stay in
# range.
#
# `a_i <= 0` is the convex axis, and declining the CELL there is what the
# separability buys us out of. The integral over a BOUNDED box is finite at
# either sign -- it is `int exp(g u + 0.5 |a| u^2) du`, an imaginary-error /
# Dawson function rather than a `pnorm` difference -- so a cell concave on three
# axes and convex on the fourth still has an exact factor on the three. Rather
# than carry an `erfi` (whose `exp(t^2)` overflows well inside the range of `t`
# this stencil produces, and which base R does not have), the offending axis is
# integrated numerically: `stats::integrate()` on a smooth integrand over a
# bounded interval is machine-precision, and the exponent is shifted by its own
# maximum over the box first so the integrand is bounded by 1 whatever the
# curvature. It costs no inner solve, which is the property that matters -- the
# grid's log-marginals are already in hand.
#
# The same numeric route takes the near-flat concave axis, where the closed
# form's own pieces cancel (see `.LCCD_BOX_CANCEL`).
#
# NULL when the axis carries no usable factor at all, which the caller reads as
# a decline to the midpoint factor of 1 on THAT axis.
.joint_local_ccd_axis_box <- function(g, a, h_lo, h_hi) {
    w <- h_lo + h_hi
    if (!is.finite(g) || !is.finite(a) || !is.finite(w) || w <= 0) return(NULL)
    if (a > 0 && 0.5 * g^2 / a < .LCCD_BOX_CANCEL) {
        s  <- 1 / sqrt(a)
        mu <- g / a
        ld <- .joint_local_ccd_log_pnorm_diff((-h_lo - mu) / s, (h_hi - mu) / s)
        val <- 0.5 * g^2 / a + 0.5 * log(2 * pi / a) + ld - log(w)
        if (is.finite(val)) return(list(log_factor = val, route = "closed"))
    }
    val <- .joint_local_ccd_axis_box_numeric(g, a, h_lo, h_hi)
    if (is.null(val)) return(NULL)
    list(log_factor = val, route = "numeric")
}

# The same axis factor by bounded one-dimensional quadrature. The exponent
# `g u - 0.5 a u^2` is shifted by its maximum over the box before exponentiating:
# on a concave axis that maximum sits at the local peak `g / a` clamped into the
# box, on a convex one at whichever endpoint is higher, so the integrand is in
# (0, 1] and neither overflows nor underflows to zero everywhere.
.joint_local_ccd_axis_box_numeric <- function(g, a, h_lo, h_hi) {
    expo <- function(u) g * u - 0.5 * a * u^2
    cand <- c(-h_lo, h_hi)
    if (a > 0) cand <- c(cand, min(max(g / a, -h_lo), h_hi))
    m <- max(expo(cand))
    if (!is.finite(m)) return(NULL)
    val <- tryCatch(
        stats::integrate(function(u) exp(expo(u) - m), -h_lo, h_hi,
                         rel.tol = .Machine$double.eps^0.75)$value,
        error = function(e) NA_real_)
    if (!is.finite(val) || val <= 0) return(NULL)
    out <- m + log(val) - log(h_lo + h_hi)
    if (!is.finite(out)) NULL else out
}

# The cell's local-quadratic box mass, as the log of its multiplier on the
# midpoint atom: `log(M_c^(Q) / (Delta_c exp(ell_c)))`, zero where the cell's
# log-marginal is flat across it.
#
# `A` is taken DIAGONAL, `a_j = -d2_j` from the stencil, which is what makes the
# box integral a product of `pnorm` differences instead of a multivariate normal
# rectangle probability. The correlated form would need one, and would need `A`
# positive definite to have one at all -- the guard
# `.joint_local_ccd_marginal_sd()` already declines under -- so the diagonal rule
# is both the cheaper measurement and the one that survives an indefinite
# finite-difference quadratic.
#
# `n_axes_declined` counts the axes that fell back to the midpoint factor of 1,
# so a multiplier is never read as a full-cell correction when part of the cell
# went unread.
.joint_local_ccd_cell_box_mass <- function(st) {
    d <- length(st$g)
    lf <- 0
    n_closed <- 0L; n_numeric <- 0L; n_declined <- 0L
    for (j in seq_len(d)) {
        f <- .joint_local_ccd_axis_box(st$g[j], -st$d2[j],
                                       st$half_lo[j], st$half_hi[j])
        if (is.null(f)) { n_declined <- n_declined + 1L; next }
        lf <- lf + f$log_factor
        if (identical(f$route, "closed")) n_closed <- n_closed + 1L
        else n_numeric <- n_numeric + 1L
    }
    list(log_box_ratio = lf, n_axes_closed = n_closed,
         n_axes_numeric = n_numeric, n_axes_declined = n_declined)
}

# Every cell of a tensor outer grid, scored by the same rule.
#
# The per-cell multiplier is the whole of what the estimator contributes to the
# integration: the softmax in `.joint_integration_weights()` normalises the
# common cell volume `Delta` away, so `exp(log_box_ratio)` enters as a `dnode`
# and nothing else has to change. That cancellation is a property of the GRID,
# not of the rule -- it holds because a geometric grid is uniform in
# `u = log theta` and its interior cells are congruent -- so a caller reusing the
# multiplier form on a grid whose cells differ in volume has to carry `Delta_c`
# itself.
#
# A cell missing a neighbour on any axis has no centred stencil and takes the
# midpoint atom it already had: `log_box_ratio` 0, `computed` FALSE. That is the
# same decline `.joint_local_ccd_cell_curv()` returns NULL for, one gate earlier.
.joint_local_ccd_box_mass <- function(joint_grid, log_marginal, latent_axes,
                                      tags) {
    n <- nrow(joint_grid)
    out <- list(log_box_ratio = rep(0, n), computed = rep(FALSE, n),
                n_axes_closed = 0L, n_axes_numeric = 0L, n_axes_declined = 0L)
    if (is.null(tags)) return(out)
    latent_cols <- match(latent_axes, colnames(joint_grid))
    if (anyNA(latent_cols)) return(out)
    d <- length(latent_cols)
    if (d < 1L || n < 3L) return(out)

    U <- matrix(0, n, d)
    for (j in seq_len(d))
        U[, j] <- .joint_pareto_fwd(tags[j], joint_grid[, latent_cols[j]])
    nb <- .joint_local_ccd_neighbors(U, joint_grid, latent_cols)

    for (c in seq_len(n)) {
        st <- .joint_local_ccd_cell_stencil(c, U, log_marginal, nb$up, nb$dn)
        if (is.null(st)) next
        bm <- .joint_local_ccd_cell_box_mass(st)
        out$log_box_ratio[c] <- bm$log_box_ratio
        out$computed[c]      <- TRUE
        out$n_axes_closed    <- out$n_axes_closed + bm$n_axes_closed
        out$n_axes_numeric   <- out$n_axes_numeric + bm$n_axes_numeric
        out$n_axes_declined  <- out$n_axes_declined + bm$n_axes_declined
    }
    out
}

# log|phi(z_lo) - phi(z_hi)| and its sign, the numerator of the truncated mean.
#
# Both standard normal densities are written as `exp(-0.5 z^2)` times the common
# constant, so the difference is `exp(hi) - exp(lo)` with `hi = max`, `lo = min`
# of the two exponents. Factoring the larger out leaves `1 - exp(lo - hi)`, taken
# through `expm1` so a pair of endpoints equidistant from the local peak (where
# the two densities agree to many figures and their difference is all that
# survives) is as accurate as a pair on opposite tails. `sign = 0` is the exactly
# balanced box, whose barycentre is the local peak itself.
#
# NULL on a non-finite argument, which the caller reads as a decline to the
# bounded quadrature.
.joint_local_ccd_log_dnorm_diff <- function(z_lo, z_hi) {
    if (!is.finite(z_lo) || !is.finite(z_hi)) return(NULL)
    la <- -0.5 * z_lo^2
    lb <- -0.5 * z_hi^2
    if (la == lb) return(list(sign = 0, log_abs = -Inf))
    hi <- max(la, lb)
    lg <- log(-expm1(min(la, lb) - hi))
    if (!is.finite(hi) || !is.finite(lg)) return(NULL)
    list(sign = if (la > lb) 1 else -1,
         log_abs = hi + lg - 0.5 * log(2 * pi))
}

# Above this the closed form's `mu + s (...)` is no longer being formed in double
# precision. The barycentre lies in the box while `mu = g/a` is the unbounded
# location of the local peak, so once the peak leaves the box the two terms are
# equal and opposite to within the box width: the result is a difference of
# numbers of size `|mu|` and the rounding of each is carried into it. The
# `pnorm` / `dnorm` logs the correction is built from are accurate to about `eps`
# of their OWN magnitude, which for a peak `P = 0.5 g^2 / a` nats above the
# origin is `eps P`, so the absolute error in the barycentre is about
# `eps P |mu|` and the error as a fraction of the cell's own width is
# `eps P |mu| / w`. Capping that product at 1e6 holds the fraction under 2.2e-10,
# and the bounded quadrature -- which forms the same ratio from two integrals
# that are both O(w) and never sees `mu` at all -- takes over beyond it. Measured
# at a product of 2.0e5 the error is 4.7e-11 of the cell width, which is
# `eps` times that product to two figures, so the budget is the observed one.
.LCCD_BAR_CANCEL <- 1e6

# The slack on the in-box check, as a fraction of the cell's own width. Set at
# the closed form's own error budget above, so a barycentre resting on an edge is
# clamped to it rather than refused, while a genuine escape is refused.
.LCCD_BAR_INBOX <- 1e-9

# One axis's mass barycentre under the local quadratic: the first moment of
# `exp(g u - 0.5 a u^2)` over `[-h_lo, h_hi]` divided by its integral, in the
# cell's own u-space offset (so 0 IS the cell's coordinate).
#
# `a > 0` (the concave axis) is the Gaussian case, and the ratio is the mean of
# `N(mu, s^2)` truncated to the box,
#   ubar = mu + s (phi(alpha) - phi(beta)) / (Phi(beta) - Phi(alpha)),
# with `alpha` / `beta` the standardized box edges. Numerator and denominator are
# both formed in logs -- the denominator by the same
# `.joint_local_ccd_log_pnorm_diff()` the box mass uses, so a box that is a
# sliver of the Gaussian it sits in is read on the tail that does not cancel --
# and combined as one exponential, which is what keeps a box far out in the tail
# (where both differences underflow on their own) from returning 0/0.
#
# `a <= 0` is the convex axis, and the near-flat concave axis is the same case
# one gate later (the closed form's `mu` runs away, see `.LCCD_BAR_CANCEL`).
# Both take the bounded quadrature: the first moment over a BOUNDED box is finite
# at either sign of `a`, so the axis has an answer either way, and the exponent is
# shifted by its own maximum over the box first so the integrand is bounded by 1
# whatever the curvature. Declining the AXIS rather than the CELL is the same
# separability the box mass buys.
#
# The result is checked to lie in the cell's own box before it is returned.
# Analytically it must -- a first moment of a positive density over the box is a
# convex combination of points in the box -- so a violation is an arithmetic
# failure, and the axis is declined (its atom stays at the cell's coordinate)
# rather than shipped into a neighbouring cell's territory.
#
# NULL when the axis carries no usable barycentre at all, which the caller reads
# as a decline to an offset of 0 on THAT axis.
.joint_local_ccd_axis_bary <- function(g, a, h_lo, h_hi) {
    w <- h_lo + h_hi
    if (!is.finite(g) || !is.finite(a) || !is.finite(w) || w <= 0) return(NULL)
    if (h_lo < 0 || h_hi < 0) return(NULL)
    val <- NULL; route <- NULL
    if (a > 0) {
        s  <- 1 / sqrt(a)
        mu <- g / a
        if (0.5 * g^2 / a * max(1, abs(mu) / w) < .LCCD_BAR_CANCEL) {
            alpha <- (-h_lo - mu) / s
            beta  <- ( h_hi - mu) / s
            ld <- .joint_local_ccd_log_pnorm_diff(alpha, beta)
            ln <- .joint_local_ccd_log_dnorm_diff(alpha, beta)
            if (is.finite(ld) && !is.null(ln)) {
                corr <- if (ln$sign == 0) 0 else ln$sign * s * exp(ln$log_abs - ld)
                v <- mu + corr
                if (is.finite(v)) { val <- v; route <- "closed" }
            }
        }
    }
    if (is.null(val)) {
        v <- .joint_local_ccd_axis_bary_numeric(g, a, h_lo, h_hi)
        if (is.null(v)) return(NULL)
        val <- v; route <- "numeric"
    }
    tol <- .LCCD_BAR_INBOX * w
    if (val < -h_lo - tol || val > h_hi + tol) return(NULL)
    list(u_bar = min(max(val, -h_lo), h_hi), route = route)
}

# The same axis barycentre by bounded one-dimensional quadrature: the ratio of
# `int u exp(g u - 0.5 a u^2) du` to `int exp(g u - 0.5 a u^2) du` over the box,
# both with the exponent shifted by its maximum over the box before
# exponentiating (on a concave axis that maximum sits at the local peak `g / a`
# clamped into the box, on a convex one at whichever endpoint is higher). The
# shift cancels out of the ratio and keeps both integrands in (0, 1], so neither
# overflows nor underflows to zero everywhere; and because neither integral is
# ever larger than the box, this route forms the barycentre without the `mu` the
# closed form subtracts.
.joint_local_ccd_axis_bary_numeric <- function(g, a, h_lo, h_hi) {
    expo <- function(u) g * u - 0.5 * a * u^2
    cand <- c(-h_lo, h_hi)
    if (a > 0) cand <- c(cand, min(max(g / a, -h_lo), h_hi))
    m <- max(expo(cand))
    if (!is.finite(m)) return(NULL)
    quad <- function(f) tryCatch(
        stats::integrate(f, -h_lo, h_hi,
                         rel.tol = .Machine$double.eps^0.75)$value,
        error = function(e) NA_real_)
    i0 <- quad(function(u) exp(expo(u) - m))
    i1 <- quad(function(u) u * exp(expo(u) - m))
    if (!is.finite(i0) || i0 <= 0 || !is.finite(i1)) return(NULL)
    val <- i1 / i0
    if (!is.finite(val)) NULL else val
}

# The cell's mass barycentre, as the per-axis offset from the cell's own
# coordinate in u-space (all zeros on a cell the log-marginal is flat across).
#
# `A` is taken DIAGONAL, `a_j = -d2_j`, the same reading the box mass makes of
# the same stencil: with `A` diagonal the local quadratic factorizes across axes,
# so the joint barycentre is the vector of per-axis barycentres and no
# multivariate first moment over a rectangle is needed.
#
# `bary_shift` is the largest share of its own half-cell any axis's atom moves,
# measured against the half-width on the side it moves toward, so the in-box
# property bounds it in [0, 1] and it is comparable across cells of different
# extent. `n_axes_declined` counts the axes that kept the cell's own coordinate.
.joint_local_ccd_cell_bary <- function(st) {
    d <- length(st$g)
    u_bar <- numeric(d)
    n_closed <- 0L; n_numeric <- 0L; n_declined <- 0L
    for (j in seq_len(d)) {
        f <- .joint_local_ccd_axis_bary(st$g[j], -st$d2[j],
                                        st$half_lo[j], st$half_hi[j])
        if (is.null(f)) { n_declined <- n_declined + 1L; next }
        u_bar[j] <- f$u_bar
        if (identical(f$route, "closed")) n_closed <- n_closed + 1L
        else n_numeric <- n_numeric + 1L
    }
    share <- ifelse(u_bar >= 0, u_bar / st$half_hi, -u_bar / st$half_lo)
    share[!is.finite(share)] <- 0
    list(u_bar = u_bar,
         bary_shift = if (d > 0L) max(share) else NA_real_,
         n_axes_closed = n_closed, n_axes_numeric = n_numeric,
         n_axes_declined = n_declined)
}

# Every cell of a tensor outer grid moved to its own mass barycentre.
#
# Returns the perturbed PHYSICAL grid alongside the u-space offsets, because the
# per-axis summary reads physical theta. The map back is the axis's own
# `.joint_pareto_inv()`, and that it is a mean in `u` pushed through `exp` rather
# than a mean in theta is a DESIGN CHOICE of the quantile read, not an
# approximation slipped past:
#
#   the grid stores physical theta on a geometric grid while the quadratic the
#   barycentre comes from is fitted in `u = log theta` (which is why
#   `log_marginal` needs no Jacobian on that axis, gcol33/tulpa#179), so `ubar`
#   is a mean in `u` and `exp(E[u]) != E[exp(u)]`;
#
#   the quantile read does not want a mean. Its atom is a REPRESENTATIVE POINT
#   of the cell, and a weighted quantile is equivariant under a monotone map, so
#   mapping `ubar` through `exp` gives the atom the representative point of the
#   cell measured in the coordinate the quadratic was fitted in -- and, since
#   `exp` is monotone and the barycentre is in the box, keeps it inside its own
#   cell, which the moment-free reading depends on.
#
# The MOMENT read is a different question with a different answer: `theta_mean` /
# `theta_sd` are moments in theta, so they need the transformed truncated moment
# rather than the transformed barycentre, and nothing here touches them.
#
# A cell missing a neighbour on any axis has no centred stencil and stays where
# it is: offset 0, `computed` FALSE, the same decline the box mass makes one gate
# earlier. Non-latent columns (an active phi dispersion tensor) are never moved.
.joint_local_ccd_barycentre <- function(joint_grid, log_marginal, latent_axes,
                                        tags) {
    n <- nrow(joint_grid)
    out <- list(joint_grid = joint_grid,
                u_offset = matrix(0, n, length(latent_axes)),
                bary_shift = rep(NA_real_, n), computed = rep(FALSE, n),
                n_axes_closed = 0L, n_axes_numeric = 0L, n_axes_declined = 0L)
    if (is.null(tags)) return(out)
    latent_cols <- match(latent_axes, colnames(joint_grid))
    if (anyNA(latent_cols)) return(out)
    d <- length(latent_cols)
    if (d < 1L || n < 3L) return(out)

    U <- matrix(0, n, d)
    for (j in seq_len(d))
        U[, j] <- .joint_pareto_fwd(tags[j], joint_grid[, latent_cols[j]])
    nb <- .joint_local_ccd_neighbors(U, joint_grid, latent_cols)

    for (c in seq_len(n)) {
        st <- .joint_local_ccd_cell_stencil(c, U, log_marginal, nb$up, nb$dn)
        if (is.null(st)) next
        bc <- .joint_local_ccd_cell_bary(st)
        out$u_offset[c, ]   <- bc$u_bar
        out$bary_shift[c]   <- bc$bary_shift
        out$computed[c]     <- TRUE
        out$n_axes_closed   <- out$n_axes_closed + bc$n_axes_closed
        out$n_axes_numeric  <- out$n_axes_numeric + bc$n_axes_numeric
        out$n_axes_declined <- out$n_axes_declined + bc$n_axes_declined
    }

    # Only a cell that moves is rewritten. A round trip through
    # `.joint_pareto_fwd()` / `.joint_pareto_inv()` is the identity in exact
    # arithmetic and not in floating point, so writing every cell back would
    # perturb the coordinates of the cells the rule declined to move -- and a
    # rule has to leave what it declines exactly as it found it, or a read taken
    # under it is not attributable to the cells it did move.
    tg <- joint_grid
    for (j in seq_len(d)) {
        off <- out$u_offset[, j]
        mv  <- off != 0
        if (any(mv)) {
            tg[mv, latent_cols[j]] <-
                .joint_pareto_inv(tags[j], U[mv, j] + off[mv])$theta
        }
    }
    out$joint_grid <- tg
    out
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
    # The per-cell recordings that gate nothing, on both sides of the gate: the
    # coarse-vs-refined mass comparison (gcol33/tulpa#323), the box-integral mass
    # (gcol33/tulpa#326) and the barycentre shift (gcol33/tulpa#327).
    # Accumulated as a list of named readings so the two sides stay one code
    # path and a reading added to it is reported on both.
    rec_kept     <- list()
    rec_declined <- list()

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
        rec_c <- .joint_local_ccd_mass(c(delta_centre, delta_off),
                                       c(log_marginal[c], lm_off), dn_w[c])
        # A third reading of the same cell's mass, and the only one of the three
        # that needs no node at all: the base grid's own neighbours already give
        # the local quadratic, and integrating it over the cell's box instead of
        # evaluating it at the centre is the correction the midpoint atom drops
        # (gcol33/tulpa#326). Recorded beside the cloud's ratio so the two
        # re-estimates of the same atom are read together. `bary_shift` is the
        # place half of the same quadratic -- the largest share of its own
        # half-cell any axis's atom moves under the barycentre
        # (gcol33/tulpa#327) -- and the two are read together for the same
        # reason: a cell can be given the right mass at the wrong place.
        rec_c <- c(rec_c,
                   list(log_box_ratio =
                            .joint_local_ccd_cell_box_mass(cc)$log_box_ratio,
                        bary_shift =
                            .joint_local_ccd_cell_bary(cc)$bary_shift))

        mis <- fit_c$misfit
        if (!isTRUE(mis < skew_max)) {
            declined_cells  <- c(declined_cells, c)
            declined_misfit <- c(declined_misfit, mis)
            declined_offset <- c(declined_offset, fit_c$offset)
            declined_gain   <- c(declined_gain, fit_c$mode_gain)
            rec_declined[[length(rec_declined) + 1L]] <- rec_c
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
        rec_kept[[length(rec_kept) + 1L]] <- rec_c
        n_nodes_added <- n_nodes_added + nrow(theta_nodes)
    }

    # One numeric vector per recording, in cell order, on each side of the gate.
    # An empty side gives a zero-length numeric, matching the empty `misfit` /
    # `offset` vectors beside it.
    rec_col <- function(acc, nm)
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
                             log_mass_ratio   = rec_col(rec_kept, "log_mass_ratio"),
                             log_mass_coarse  = rec_col(rec_kept, "log_mass_coarse"),
                             log_mass_refined = rec_col(rec_kept, "log_mass_refined"),
                             max_node_weight  = rec_col(rec_kept, "max_node_weight"),
                             log_box_ratio    = rec_col(rec_kept, "log_box_ratio"),
                             bary_shift       = rec_col(rec_kept, "bary_shift"),
                             skew_max        = skew_max,
                             cells_declined  = declined_cells,
                             misfit_declined = declined_misfit,
                             offset_declined = declined_offset,
                             mode_gain_declined = declined_gain,
                             log_mass_ratio_declined =
                                 rec_col(rec_declined, "log_mass_ratio"),
                             log_mass_coarse_declined =
                                 rec_col(rec_declined, "log_mass_coarse"),
                             log_mass_refined_declined =
                                 rec_col(rec_declined, "log_mass_refined"),
                             max_node_weight_declined =
                                 rec_col(rec_declined, "max_node_weight"),
                             log_box_ratio_declined =
                                 rec_col(rec_declined, "log_box_ratio"),
                             bary_shift_declined =
                                 rec_col(rec_declined, "bary_shift"),
                             n_cells_declined = length(declined_cells),
                             n_cells_before  = n,
                             n_cells_after   = nrow(out_grid),
                             n_design_nodes  = sum(out_kind == "design")))
}
