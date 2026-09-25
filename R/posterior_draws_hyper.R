# The hyperparameter half of a nested-Laplace posterior draw.
#
# A draw from the outer-grid mixture is two steps -- pick a cell by weight, then
# draw that cell's inner Gaussian -- and `tulpa_posterior_draws()` gives the
# LATENT half of the second step a genuine within-cell Gaussian. The
# hyperparameter half had none: a draw's `sigma` / `alpha` / `phi` was read
# straight off the chosen cell's grid coordinate, so a fit integrating a 5-node
# axis produced draws taking 5 distinct values on it whatever the sample size.
# That is an ATOM, not a marginal, and it is the same object gcol33/tulpa#337
# named ("the read collapses each box to its midpoint") -- which #337 and #353
# fixed for the reported interval (`.nl_summary_quantile()`) and nowhere else,
# so every consumer of raw draws still had the atom (gcol33/tulpa#823).
#
# THE FIX IS NOT A NEW CONSTRUCTION. Each of the two within-cell reads the
# engine ships already DEFINES a continuous density on an axis, and each is a
# per-cell mixture whose component is available in closed form. Sampling that
# component conditional on the cell the draw came from is what this file does,
# so the hyperparameter draws reproduce the fit's own `theta_ci_lo` /
# `theta_median` / `theta_ci_hi` to Monte Carlo error rather than approximating
# them:
#
#   * `box_uniform` -- cell c's whole mass `w_c` sits uniformly on its own box
#     `[e_c, e_{c+1}]` (`.nl_box_quantile()` interpolates linearly between those
#     edges), so the conditional is `Uniform(e_c, e_{c+1})`. Summed over cells
#     the density on box c is `w_c / (e_{c+1} - e_c)`: the box read exactly.
#   * `chord` -- the CDF's knots are `(cumsum(w) - w / 2, v)`, so between two
#     adjacent coordinates the density is uniform with mass
#     `(w_c + w_{c+1}) / 2` and cell c contributes HALF its mass to the segment
#     on either side. The conditional is therefore an equal mixture of
#     `Uniform(v_{c-1}, v_c)` and `Uniform(v_c, v_{c+1})`. Summed over cells the
#     density on `[v_c, v_{c+1}]` is `(w_c + w_{c+1}) / (2 (v_{c+1} - v_c))`:
#     the chord read exactly.
#
# Neither conditional depends on the weights, only on the partition geometry --
# which is what makes this safe: the marginal comes out right for ANY cell
# weighting, so the jitter cannot disagree with the sampler that chose the cell.
# The weights are read only where the READ itself filters on them (the chord
# knots are the positive-weight coordinates, the box partition is every
# coordinate; see `.nl_box_quantile()`'s own note on why the two differ).
#
# The outer half-cell is part of it. A `density` support's outermost cell
# reaches half a spacing past its coordinate (`outside = "extend"`), which is
# the region a truth drawn beyond the outermost node lands in and the region no
# node-valued draw can reach. A `sample` support clamps instead, and there the
# extreme cell's outward half IS an atom at its coordinate -- handled by the
# same code, with the neighbour set to the coordinate itself.

# The within-cell geometry of ONE axis: the coordinates a draw's cell value is
# matched against, and the interval each of them spreads its mass over.
#
# `kind` is the construction that produced it -- `"box_uniform"`, `"chord"`, or
# `"none"` when the axis carries no continuization and a draw keeps its node
# value. `lo` / `hi` are per-coordinate: the box's two edges under
# `box_uniform`, the left and right NEIGHBOUR coordinates (outer edge where
# there is none) under `chord`. `declined` is why the requested construction did
# not run, from the vocabulary `.nl_summary_quantile_read()` already uses, plus
# the two this path adds for a support that reaches no CDF at all.
#
# A DECLARED POINT MASS is a coordinate whose `lo` and `hi` are the coordinate
# itself, which both reads already turn into the level exactly -- the same
# degenerate interval a `clamp` support's extreme cell takes above, reached
# from the other direction.
#
# On an axis a refinement pass re-tiled (`rows`, `.nl_axis_cell_rows()`) a
# cell's box depends on its row as well as its value, so the box-uniform
# geometry is PER CELL: `values` is the axis coordinate of every cell and
# `lo` / `hi` the box that cell owns (`.nl_cell_boxes()`), flagged `per_cell`.
.nl_hyper_axis_geometry <- function(v, w, domain, within, outside,
                                    atom = NA_real_, rows = NULL) {
  none <- function(declined) {
    list(kind = "none", values = numeric(0), lo = numeric(0),
         hi = numeric(0), declined = declined)
  }
  if (is.na(outside)) return(none("support_moment_rule"))

  if (!is.null(rows) && !identical(within, "chord")) {
    ia <- length(atom) == 1L && is.finite(atom) && any(v == atom) &&
          !any(v < atom, na.rm = TRUE)
    cont <- if (ia) is.na(v) | v != atom else rep(TRUE, length(v))
    rc <- list(row = rows$row[cont], base = rows$base[cont])
    bx <- .nl_cell_boxes(v[cont], domain, rc)
    if (is.null(bx)) {
      g <- .nl_hyper_axis_geometry(v, w, domain, "chord", outside, atom)
      g$declined <- "boxes_do_not_tile"
      return(g)
    }
    lo <- hi <- v
    lo[cont] <- bx$lo
    hi[cont] <- bx$hi
    return(list(kind = "box_uniform", per_cell = TRUE, values = v,
                lo = lo, hi = hi, declined = NA_character_))
  }

  # A declared point mass is not a cell: its box is its own coordinate, so a
  # draw landing in it IS the level and the continuum's partition is laid over
  # what is left. Prepended rather than special-cased in the draw, because
  # `lo == hi == value` already makes both within-cell reads return the level
  # exactly -- and because the continuum's own partition is then built on a node
  # set the axis's declared support contains, which is what stops it reaching
  # half a spacing below zero (gcol33/tulpa#854).
  ia <- length(atom) == 1L && is.finite(atom) && any(v == atom) &&
        !any(v < atom, na.rm = TRUE)
  if (ia) {
    keep <- v != atom
    g <- .nl_hyper_axis_geometry(v[keep], w[keep], domain, within, outside)
    if (identical(g$kind, "none")) return(g)
    g$values <- c(atom, g$values)
    g$lo     <- c(atom, g$lo)
    g$hi     <- c(atom, g$hi)
    return(g)
  }

  chord <- function(declined = NA_character_) {
    # The chord read's knots ARE the positive-weight coordinates, so its
    # partition is taken off the same atoms `.nl_wtd_quantile()` builds.
    a <- .nl_axis_atoms(v, w)
    if (is.null(a)) return(none("no_usable_node"))
    uv <- a$v
    n <- length(uv)
    if (n < 2L) return(none("single_node"))
    e <- if (identical(outside, "extend")) {
      .nl_cell_edges(uv, domain)
    } else {
      # `clamp`: nothing is known past the extreme order statistic, so the
      # outward half of each extreme cell's mass stays on its coordinate.
      c(uv[1L], uv[n])
    }
    list(kind = "chord", values = uv,
         lo = c(e[1L], uv[-n]), hi = c(uv[-1L], e[2L]),
         declined = declined)
  }

  if (identical(within, "chord")) return(chord())
  fin <- is.finite(v)
  if (!any(fin)) return(none("no_usable_node"))
  uv <- sort(unique(as.numeric(v[fin])))
  if (length(uv) < 2L) return(none("single_node"))
  e <- .nl_box_edges_from(.nl_cell_partition(uv, domain), uv)
  if (is.null(e)) return(chord("boxes_do_not_tile"))
  list(kind = "box_uniform", values = uv,
       lo = e[-length(e)], hi = e[-1L], declined = NA_character_)
}

# The geometry coordinate each cell value is matched to: the cell itself on a
# per-cell geometry, otherwise the position of its value among the geometry's
# coordinates (NA for a value the axis read filtered out).
.nl_hyper_axis_index <- function(geom, v_cell, cells) {
  if (isTRUE(geom$per_cell)) cells else match(v_cell, geom$values)
}

# One axis's per-draw coordinate: the cell value continuized across that cell's
# own geometry, by the inverse CDF of the cell-conditional at `u`. A draw whose
# cell value is not one of the geometry's coordinates -- reachable only where
# the draws were allocated over a cell set the axis read filtered out -- keeps
# its node value rather than being matched to a neighbouring box.
#
# `u` is the draw's uniform on this axis. Each cell-conditional is reached by
# its inverse CDF, so ANY uniform input -- independent across axes or tied
# through a copula -- reproduces the per-axis marginal exactly.
.nl_hyper_axis_draw <- function(geom, v_cell, cells = NULL,
                                u = stats::runif(length(v_cell))) {
  if (identical(geom$kind, "none")) return(v_cell)
  k <- .nl_hyper_axis_index(geom, v_cell, cells)
  out <- if (identical(geom$kind, "box_uniform")) {
    geom$lo[k] + u * (geom$hi[k] - geom$lo[k])
  } else {
    # Half the cell's mass on either side, each uniform on its own segment:
    # the lower half of `u` spans the left segment, the upper half the right.
    left <- u < 0.5
    ifelse(left,
           geom$lo[k] + 2 * u * (geom$values[k] - geom$lo[k]),
           geom$values[k] + (2 * u - 1) * (geom$hi[k] - geom$values[k]))
  }
  miss <- is.na(k) | !is.finite(out)
  out[miss] <- v_cell[miss]
  out
}

# Every cell's within-cell conditional on one axis, read on the axis's
# unconstrained coordinate `to` (`.NL_DOMAIN_TRANSFORM`): its mean `m`, its
# standard deviation `s`, and the coordinate span `width` the cell's mass is
# spread over. Taken by Gauss-Legendre quadrature over the uniform that
# `.nl_hyper_axis_draw()` itself inverts, on each half of (0, 1) separately so
# the chord's kink at 1/2 falls on a panel edge -- the moments are those of the
# draws, not of a second description of the geometry. A cell the axis does not
# continuize carries zero spread.
.nl_hyper_axis_cell_moments <- function(geom, v, to) {
  n <- length(v)
  none <- list(m = to(v), s = numeric(n), width = rep(NA_real_, n))
  if (identical(geom$kind, "none")) return(none)
  gl <- .nl_gauss_legendre_unit(16L)
  uq <- c(gl$x / 2, 0.5 + gl$x / 2)
  wq <- c(gl$w, gl$w) / 2
  cells <- seq_len(n)
  tq <- vapply(uq, function(u) {
    to(.nl_hyper_axis_draw(geom, v, cells, rep(u, n)))
  }, numeric(n))
  tq <- matrix(tq, n)
  m <- drop(tq %*% wq)
  s <- sqrt(pmax(drop(tq^2 %*% wq) - m^2, 0))
  k <- .nl_hyper_axis_index(geom, v, cells)
  width <- rep(NA_real_, n)
  ok <- !is.na(k)
  width[ok] <- to(geom$hi[k[ok]]) - to(geom$lo[k[ok]])
  bad <- is.na(k) | !is.finite(m) | !is.finite(s)
  m[bad] <- none$m[bad]
  s[bad] <- 0
  list(m = m, s = s, width = width)
}

# Nodes and weights of the n-point Gauss-Legendre rule on (0, 1)
# (Golub-Welsch: eigen-decomposition of the Jacobi matrix).
.nl_gauss_legendre_unit <- function(n) {
  k <- seq_len(n - 1L)
  b <- k / sqrt(4 * k^2 - 1)
  J <- matrix(0, n, n)
  J[cbind(k, k + 1L)] <- b
  J[cbind(k + 1L, k)] <- b
  e <- eigen(J, symmetric = TRUE)
  o <- order(e$values)
  list(x = (e$values[o] + 1) / 2, w = e$vectors[1L, o]^2)
}

# The Gaussian-copula correlation that ties the axes' within-cell uniforms
# together (gcol33/tulpa#859).
#
# Jittering each axis independently keeps every marginal but not the joint: in
# the cells where two axes both spread the within-cell covariance is zero, so
# a derived quantity along the direction the posterior pins down -- a product
# of anticorrelated scales -- picks up the full box variance of both axes. Any
# coupling of the uniforms leaves every marginal exactly as it is
# (`.nl_hyper_axis_draw()` inverts each cell-conditional at its own uniform),
# so the joint is free to set, and it is set to the posterior's own
# orientation.
#
# That orientation is read off the grid's LOG DENSITY, not off its node
# weights. On the unconstrained coordinate t of every axis
# (`.NL_DOMAIN_TRANSFORM`), each cell's density is its mass over the span its
# mass is spread across, and a weighted quadratic fit of the log density over
# the cells where every continuized axis spreads gives the local Gaussian's
# precision; its inverse gives the target correlation `rho*` of each pair. The
# node weights' own correlation is NOT a usable target: where the grid does
# not resolve the ridge-orthogonal direction (a tensor grid under a strong
# correlation), the weights concentrate on a few cells along the ridge and
# understate the posterior's spread across it, which the curvature does not.
#
# For a pair (i, j), with the cell-conditional means `m` and sds `s` on t
# (`.nl_hyper_axis_cell_moments()`), the draws carry
#
#   Var(t_i)      = Var_w(m_i) + E_w[s_i^2]
#   Cov(t_i, t_j) = Cov_w(m_i, m_j) + rho_u E_w[s_i s_j]
#
# with `rho_u` the correlation the copula induces between the two within-cell
# coordinates (to first order in the cell's width on t), and `rho_u` solves
# Cov / sqrt(Var Var) = rho*. A Gaussian copula at `r` gives uniforms with
# Pearson correlation (6 / pi) asin(r / 2), inverted here, and the matrix is
# projected to the nearest correlation matrix by clipping its eigenvalues,
# since pairwise solves need not be jointly positive definite.
#
# No coupling (the identity) is returned where the orientation is not
# identified: fewer than two continuized axes, fewer cells than the quadratic
# has coefficients, or a fitted curvature that is not negative definite.
#
# The quadratic is exact on a Gaussian and an approximation elsewhere. Two
# alternatives were measured against it on skewed and curved posteriors with
# exact reference moments (gcol33/tulpa#861): the mass-weighted mean of the
# cells' own finite-difference curvatures, and the law of total covariance over
# the box partition with each box shaped by its local quadratic. Both are
# exact on a Gaussian too, and both read the log product of two anticorrelated
# skewed scales 13-60% (respectively up to 30%) wide at K = 4 .. 9, where the
# global quadratic stays within -5% / +3%. The moment form is also unstable
# on a grid that does not resolve the posterior: a box's local quadratic
# extrapolated to its corners can outweigh the grid's own mass by orders of
# magnitude.
.nl_hyper_copula <- function(tg, w, geoms, domains) {
  p <- ncol(tg)
  R <- diag(p)
  dimnames(R) <- list(colnames(tg), colnames(tg))
  act <- which(vapply(geoms, function(g) !identical(g$kind, "none"), TRUE))
  q <- length(act)
  if (q < 2L) return(R)
  to <- lapply(act, function(j) {
    d <- if (length(domains) < j) NA_character_ else domains[[j]]
    tr <- if (is.na(d)) NULL else .NL_DOMAIN_TRANSFORM[[d]]
    if (is.null(tr)) identity else tr$to
  })
  mom <- lapply(seq_len(q), function(a) {
    .nl_hyper_axis_cell_moments(geoms[[act[a]]], as.numeric(tg[, act[a]]),
                                to[[a]])
  })
  t_node <- vapply(seq_len(q), function(a) to[[a]](as.numeric(tg[, act[a]])),
                   numeric(nrow(tg)))
  t_node <- matrix(t_node, nrow(tg))
  S <- vapply(mom, `[[`, numeric(nrow(tg)), "s")
  M <- vapply(mom, `[[`, numeric(nrow(tg)), "m")
  W <- vapply(mom, `[[`, numeric(nrow(tg)), "width")
  S <- matrix(S, nrow(tg)); M <- matrix(M, nrow(tg)); W <- matrix(W, nrow(tg))
  ld <- log(w) - rowSums(log(W))
  use <- is.finite(w) & w > 0 & rowSums(S > 0) == q & is.finite(ld) &
         rowSums(!is.finite(t_node)) == 0L
  n_coef <- 1L + q + q * (q + 1L) / 2L
  if (sum(use) <= n_coef) return(R)

  ww <- w[use] / sum(w[use])
  tt <- t_node[use, , drop = FALSE]
  ctr <- colSums(ww * tt)
  sc <- sqrt(colSums(ww * sweep(tt, 2L, ctr)^2))
  if (any(!is.finite(sc) | sc <= 0)) return(R)
  z <- sweep(sweep(tt, 2L, ctr), 2L, sc, "/")
  pr <- which(upper.tri(diag(q), diag = TRUE), arr.ind = TRUE)
  X <- cbind(1, z, z[, pr[, 1L], drop = FALSE] * z[, pr[, 2L], drop = FALSE])
  # The fit's weights are floored where `lm.wfit()`'s rank test (tolerance
  # `tol` on the sqrt-weighted columns) can still see a cell, a factor 100
  # above it. Under a strong correlation on a coarse grid the off-ridge cells
  # carry masses far below that (about 1e-130 at K = 3, rho = -0.97), and at
  # their raw masses the design reads as rank 3 of 6 although every cell's log
  # density is finite and, on a Gaussian, exactly quadratic. A floored cell
  # still weighs at most 1e-10 of the modal one, so a grid whose cells are all
  # above the floor is fitted exactly as before (gcol33/tulpa#860).
  tol <- 1e-7
  wfit <- pmax(ww, (100 * tol)^2 * max(ww))
  fitq <- stats::lm.wfit(X, ld[use], wfit, tol = tol)
  if (fitq$rank < ncol(X)) return(R)
  b <- fitq$coefficients[-(seq_len(1L + q))]
  H <- matrix(0, q, q)
  H[pr] <- b
  H <- H + t(H)                # d2/dz^2: 2 b_jj on the diagonal, b_jk off it
  P <- -H / outer(sc, sc)      # precision on t
  eP <- eigen(P, symmetric = TRUE, only.values = TRUE)$values
  if (min(eP) <= 0) return(R)
  Sig <- solve(P)
  rho_star <- stats::cov2cor(Sig)

  mt <- M[use, , drop = FALSE]
  st <- S[use, , drop = FALSE]
  wcov <- function(a, b) sum(ww * (a - sum(ww * a)) * (b - sum(ww * b)))
  Ru <- diag(q)
  for (a in seq_len(q - 1L)) for (c in (a + 1L):q) {
    va <- wcov(mt[, a], mt[, a]) + sum(ww * st[, a]^2)
    vc <- wcov(mt[, c], mt[, c]) + sum(ww * st[, c]^2)
    ess <- sum(ww * st[, a] * st[, c])
    ru <- (rho_star[a, c] * sqrt(va * vc) - wcov(mt[, a], mt[, c])) / ess
    if (!is.finite(ru)) ru <- 0
    ru <- min(max(ru, -1), 1)
    Ru[a, c] <- Ru[c, a] <- 2 * sin(pi * ru / 6)
  }
  Ru <- .nl_nearest_correlation(Ru)

  # The within-cell coupling carries the posterior's LOCAL dependence, and the
  # sign of each pair's conditional dependence given the other axes is the
  # sign of the posterior precision's entry. Where the cells' means already
  # correlate more strongly than the target, a pairwise solve can ask for a
  # conditional dependence of the opposite sign inside the cell: a dependence
  # the posterior does not have, which on a curved posterior read the log
  # product 19.6% wide instead of 1.8% (gcol33/tulpa#861). Such a pair is made
  # conditionally independent inside the cell. The CONDITIONAL sign is the one
  # to hold: with three axes a pair's marginal within-cell coupling can
  # legitimately oppose the target's (-0.6, -0.6, -0.2 couples the last pair at
  # +0.94), and it is its partial correlation that agrees.
  Qu <- solve(Ru)
  flip <- Qu * P < 0
  diag(flip) <- FALSE
  if (any(flip)) {
    Qu[flip] <- 0
    Qu <- .nl_nearest_correlation(Qu, cor = FALSE)
    Ru <- .nl_nearest_correlation(stats::cov2cor(solve(Qu)))
  }
  R[act, act] <- Ru
  R
}

# The nearest positive-definite matrix to a symmetric one by clipping its
# eigenvalues at `floor`, rescaled to a unit diagonal when `cor` is TRUE.
.nl_nearest_correlation <- function(M, floor = 1e-8, cor = TRUE) {
  ev <- eigen(M, symmetric = TRUE)
  if (min(ev$values) < floor) {
    M <- ev$vectors %*% (pmax(ev$values, floor) * t(ev$vectors))
  }
  if (cor) {
    d <- sqrt(diag(M))
    M <- M / outer(d, d)
  }
  M
}

#' Hyperparameter draws from a nested-Laplace fit
#'
#' @description
#' The hyperparameter half of a draw from the outer-grid mixture: one row per
#' draw, one column per outer-grid axis. A draw's coordinate on each axis is
#' sampled from the within-cell density its cell carries under the fit's own
#' within-cell read, so the columns are a continuous marginal rather than the
#' handful of grid-node values `fit$theta_grid[cells, ]` returns.
#'
#' An axis carrying a declared POINT MASS -- the copy scale's `alpha = 0`, the
#' "no coupling" model whose prior probability is stated rather than read off a
#' node count -- keeps it: draws in that cell are exactly the level, in the
#' proportion the fit reports as `fit$copy_atom$posterior_mass`, and the
#' continuum above it is continuized on its own partition. The level is a model
#' and not a cell representative, so spreading it over a box would both put mass
#' where the axis has none and leave none on the level itself.
#'
#' Reading the node coordinate directly is what
#' [tulpa_posterior_draws()] consumers used to do, and on a 5- to 15-node axis
#' it makes every draw an atom: a truth between two nodes, or past the
#' outermost one, has no draw that can land near it. The continuization is the
#' cell-conditional of the SAME construction the fit reports its interval from
#' (`box_uniform` by default, `chord` when the fit asked for it or its support
#' does not admit the box read), so the draws reproduce the fit's own
#' `theta_ci_lo` / `theta_median` / `theta_ci_hi` to Monte Carlo error.
#'
#' @param fit A nested-Laplace fit ([tulpa_nested_laplace()] or
#'   [tulpa_nested_laplace_joint()]).
#' @param cells Integer vector of outer-grid cell indices, one per draw -- the
#'   `"cells"` attribute of a [tulpa_posterior_draws()] matrix, so the
#'   hyperparameter row and the latent row of a draw come from the same cell.
#'   `NULL` (default) allocates `n` fresh draws across the cells by weight.
#' @param n Number of draws when `cells` is `NULL` (default 1000); ignored
#'   otherwise.
#' @param within Within-cell construction, `"box_uniform"` or `"chord"`.
#'   `NULL` (default) takes the one the fit was read with
#'   (`fit$within_cell_requested`).
#'
#' @return A numeric matrix `[n x n_axes]` with the outer grid's axis names as
#'   columns, or `NULL` when the fit carries no outer-grid axis. Carries
#'   `attr(., "cells")` -- the cell each row's coordinates were drawn in --
#'   plus `attr(., "within_cell")` and `attr(., "within_cell_declined")`, the
#'   per-axis construction that ran and why a requested one did not, and
#'   `attr(., "within_cell_copula")`, the Gaussian-copula correlation matrix
#'   that ties the axes' within-cell draws together so the draws keep the
#'   grid's own correlation between axes (the identity where the grid's axes
#'   are uncorrelated). Each axis's marginal is the same whatever the copula.
#'
#' @seealso [tulpa_posterior_draws()], [tulpa_nested_laplace()]
#' @examples
#' \donttest{
#' set.seed(1)
#' S <- 30L                                   # spatial units in a chain
#' nb <- lapply(seq_len(S), function(s) setdiff(c(s - 1L, s + 1L), c(0L, S + 1L)))
#' nn <- lengths(nb)
#' field <- as.numeric(scale(cumsum(rnorm(S, 0, 0.4))))
#' idx <- rep(seq_len(S), each = 6L); n <- length(idx); x <- rnorm(n)
#' y <- rbinom(n, 1L, plogis(-0.2 + 0.6 * x + field[idx]))
#' prior <- list(type = "icar", n_spatial_units = S, spatial_idx = idx,
#'               adj_row_ptr = c(0L, cumsum(nn)), adj_col_idx = unlist(nb) - 1L,
#'               n_neighbors = nn, tau_grid = c(0.5, 1, 2, 4, 8))
#' fit <- tulpa_nested_laplace(y, rep(1L, n), cbind(1, x), prior = prior,
#'                             family = "binomial",
#'                             control = list(progress = FALSE))
#' th <- tulpa_hyper_draws(fit, n = 200)
#' quantile(th[, "tau"], c(0.025, 0.5, 0.975))
#' }
#' @export
tulpa_hyper_draws <- function(fit, cells = NULL, n = 1000, within = NULL) {
  tg <- .nl_theta_matrix(fit)
  if (is.null(tg) || !is.matrix(tg) || ncol(tg) == 0L || nrow(tg) == 0L) {
    return(NULL)
  }
  w <- fit$weights
  if (is.null(w) || length(w) != nrow(tg) || !any(is.finite(w) & w > 0)) {
    # No stored measure: every coordinate is an atom of the chord partition,
    # which is what an unweighted read of the same node set would take.
    w <- rep(1, nrow(tg))
  }
  if (is.null(cells)) {
    n <- .nl_draws_check_n(n)
    ok <- is.finite(w) & w > 0
    cells <- .nl_mixture_cells(w[ok] / sum(w[ok]), which(ok), n)$row_cells
  } else {
    cells <- as.integer(cells)
    if (anyNA(cells) || any(cells < 1L) || any(cells > nrow(tg))) {
      stop("`cells` must be 1-based outer-grid cell indices (1..",
           nrow(tg), ").", call. = FALSE)
    }
  }

  within  <- .nl_within_cell_mode(within %||% fit$within_cell_requested)
  support <- .nl_node_support(fit$integration, fit$weight_kind)
  outside <- .NL_SUPPORT[[support]]$outside
  # A support that does not admit the requested construction falls back to the
  # chord read exactly as the interval does, and says so per axis.
  req <- if (within %in% .NL_SUPPORT[[support]]$within) within else "chord"
  fell <- if (identical(req, within)) NA_character_ else
    paste0("support_", support)
  # An axis whose support the registry will not name carries NA and mirrors its
  # outer edge in the coordinate the guess picks, exactly as the interval read
  # does; a fit the registry cannot read at all leaves every axis undeclared
  # rather than taking the draw down.
  geo <- tryCatch(.nl_axis_geometry(fit), error = function(e) NULL)
  doms <- geo$domain
  atoms <- geo$atom

  nms <- .nl_axis_names(tg)
  out <- matrix(0.0, length(cells), ncol(tg), dimnames = list(NULL, nms))
  used <- stats::setNames(rep(NA_character_, ncol(tg)), nms)
  decl <- stats::setNames(rep(NA_character_, ncol(tg)), nms)
  geoms <- lapply(seq_len(ncol(tg)), function(j) {
    dm <- if (length(doms) < j) NA_character_ else doms[[j]]
    at <- if (length(atoms) < j) NA_real_ else atoms[[j]]
    .nl_hyper_axis_geometry(as.numeric(tg[, j]), w, dm, req, outside, at,
                            .nl_axis_cell_rows(tg, j, fit$refining_axis))
  })
  # One uniform per draw and axis, tied across axes by the copula that keeps
  # the grid's own correlation (`.nl_hyper_copula()`).
  cop <- .nl_hyper_copula(tg, w, geoms, doms)
  z <- matrix(stats::rnorm(length(cells) * ncol(tg)), length(cells), ncol(tg))
  u <- stats::pnorm(z %*% chol(cop))
  for (j in seq_len(ncol(tg))) {
    g <- geoms[[j]]
    out[, j] <- .nl_hyper_axis_draw(g, as.numeric(tg[cells, j]), cells, u[, j])
    used[j] <- if (identical(g$kind, "none")) NA_character_ else g$kind
    decl[j] <- if (is.na(g$declined)) fell else g$declined
  }
  attr(out, "cells") <- cells
  attr(out, "within_cell") <- used
  attr(out, "within_cell_declined") <- decl
  attr(out, "within_cell_copula") <- cop
  out
}

# The `"theta"` attribute both `tulpa_posterior_draws()` methods carry: the
# hyperparameter half of the same rows, drawn in the cells the latent half came
# from. Attached rather than left to the caller because the caller's own
# alternative is `theta_grid[cells, ]`, which is the atom this exists to
# replace -- a consumer reading the attribute is correct by default, and one
# calling `tulpa_hyper_draws()` on the `"cells"` attribute gets the same object.
.nl_attach_hyper_draws <- function(out, fit) {
  # Attached without the caller asking, so it runs with the RNG RESTORED: an
  # automatic extra must not move the stream, or every number downstream of an
  # existing `set.seed()` script shifts. The latent draws are formed before it
  # and are bit-for-bit what they were; an explicit `tulpa_hyper_draws()` call
  # consumes the stream normally, as a draw should.
  th <- .with_preserved_seed(
    tryCatch(tulpa_hyper_draws(fit, cells = attr(out, "cells")),
             error = function(e) conditionMessage(e)))
  # A fit with no outer axis has no hyperparameter half, which is not a
  # failure; anything else that stopped the continuization is RECORDED rather
  # than swallowed, because a silently absent attribute sends its reader back
  # to `theta_grid[cells, ]` -- the atom this replaces.
  if (is.character(th)) {
    attr(out, "theta_declined") <- th
    return(out)
  }
  if (is.null(th)) return(out)
  attr(out, "theta") <- th
  attr(out, "theta_within_cell") <- attr(th, "within_cell")
  attr(out, "theta_within_cell_declined") <- attr(th, "within_cell_declined")
  out
}
