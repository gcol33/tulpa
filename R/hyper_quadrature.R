# Quadrature weights for the outer hyperparameter grid.
#
# The outer level integrates the Laplace-approximate marginal likelihood against
# a prior measure on the hyperparameters. The nodes are a quadrature rule for
# that measure, not the measure itself: the prior is declared before the fit and
# the nodes only control how accurately it is integrated. Weighting every node
# equally makes the two indistinguishable, so densifying or extending a grid
# after seeing the data silently reweights the prior.
#
# Each axis therefore carries a per-level weight built from the node spacing on
# its integration coordinate (log for a log-scale axis, natural otherwise), and
# an axis with a zero node splits its mass between a declared atom at zero and a
# continuum holding the rest. Node placement then changes only the quadrature
# error.

# Integration coordinate. A log-scale axis is spaced and integrated in log, so
# a flat measure there is a 1/theta density on the natural scale.
.hyper_axis_coord <- function(v, spec) {
  if (isTRUE(spec$log_scale)) log(v) else as.numeric(v)
}

# Carry a declared log-density to the coordinate the outer grid integrates on.
#
# `coord` names the coordinate `fn` is a density on. A density on the natural
# scale of a log-scale axis meets cell widths measured in log, so it picks up
# the change of variables `p(x) dx = p(x) x d(log x)`; a density already on the
# integration coordinate is carried through as written, and on a linear axis the
# two coincide. Every path that turns a declared density into a weight -- the
# axis quadrature here, the generic driver's `log_marginal` fold, the joint
# driver's -- goes through this, so the rule is stated once.
#
# A value the density cannot score (an error, a non-finite return) is -Inf: the
# level carries no prior mass.
.hyper_prior_carry <- function(x, fn, log_scale = FALSE, coord = "natural") {
  x <- as.numeric(x)
  ld <- vapply(x, function(v) {
    out <- tryCatch(fn(v), error = function(e) NA_real_)
    if (length(out) != 1L || !is.finite(out)) NA_real_ else as.numeric(out)
  }, numeric(1))
  if (identical(coord, "natural") && isTRUE(log_scale)) {
    lx <- suppressWarnings(log(x))
    ld <- ld + ifelse(is.finite(lx), lx, -Inf)
  }
  ld[is.na(ld)] <- -Inf
  ld
}

# Is this level the axis's point mass rather than a point of its continuum?
# A zero on a log-scale axis that declares an `atom_mass`; nothing otherwise.
.hyper_is_atom_level <- function(x, spec) {
  if (is.null(spec$atom_mass) || !isTRUE(spec$log_scale)) {
    return(rep(FALSE, length(x)))
  }
  as.numeric(x) == 0
}

# Node cell widths on the integration coordinate, clipped to a fixed interval.
#
# Each node owns the half-interval to its neighbour on either side; the outermost
# nodes own out to the declared bounds. On an evenly spaced grid whose bounds sit
# half a step beyond the end nodes this returns equal widths, which is the rule
# the engine has always applied, so an unrefined grid integrates exactly what it
# did before. Subdividing an interval splits that node's width instead of adding
# weight, which is what makes refinement leave the measure alone.
.hyper_cell_widths <- function(u, lo, hi) {
  K <- length(u)
  if (K < 1L) return(numeric(0))
  if (K == 1L) return(1)
  edges <- c(lo, (u[-K] + u[-1L]) / 2, hi)
  diff(edges)
}

# Default bounds for a node set with no declared support: half a step beyond the
# outermost nodes, so an evenly spaced grid keeps equal weights.
.hyper_default_coord_bounds <- function(u) {
  K <- length(u)
  if (K < 2L) return(c(u[1L] - 0.5, u[1L] + 0.5))
  c(u[1L] - (u[2L] - u[1L]) / 2, u[K] + (u[K] - u[K - 1L]) / 2)
}

# The bare axis name. A multi-block grid prefixes each column with the block
# that owns it (`b1.rho_car`), and every declaration keyed by axis name is keyed
# on the name without that prefix.
.hyper_axis_bare <- function(name) {
  sub("^b[0-9]+[.]", "", as.character(name %||% ""))
}

# Natural-scale domain of one axis as `list(bounds, open)`, or NULL where
# nothing is declared and the axis is treated as unbounded.
#
# Two declarations can reach the same axis and both are claims about the same
# set, so the answer is their INTERSECTION rather than a precedence between
# them. `.NL_AXIS_DOMAIN` states what the axis NAME fixes on every path that
# uses it; a spec's own `bounds` states what the block that built the spec knows
# in addition, which is how the BYM2 mixing weight gets its (0, 1) where the
# name `rho` alone can only say the upper end. Intersecting keeps both true
# statements and can only tighten, which is the safe direction for an interval
# a support is refused outside of.
#
# A finite endpoint of a spec's `bounds` is treated as OPEN, for the same reason
# every entry of the registry is: a declared natural support ends where the
# parameterisation degenerates -- a zero scale, a singular `Q` -- so the
# endpoint itself is not a value the fit can be evaluated at.
.hyper_axis_domain <- function(spec) {
  bare <- .hyper_axis_bare(spec[["name"]])
  d <- if (nzchar(bare)) .NL_AXIS_DOMAIN[[bare]] else NULL
  if (is.null(d) && isTRUE(.hyper_axis_scale(bare))) {
    d <- .NL_AXIS_DOMAIN[[".positive"]]
  }
  lo <- -Inf; hi <- Inf; open <- c(TRUE, TRUE); declared <- FALSE
  if (!is.null(d)) {
    lo <- d$bounds[1L]; hi <- d$bounds[2L]; open <- d$open; declared <- TRUE
  }
  b <- spec[["bounds"]]
  if (!is.null(b) && length(b) == 2L && !anyNA(b)) {
    b <- as.numeric(b)
    if (b[1L] > lo) { lo <- b[1L]; open[1L] <- TRUE }
    if (b[2L] < hi) { hi <- b[2L]; open[2L] <- TRUE }
    declared <- TRUE
  }
  if (!declared || (!is.finite(lo) && !is.finite(hi))) return(NULL)
  list(bounds = c(lo, hi), open = open)
}

# Close a node set's outer cells inside the axis's declared domain.
#
# `bd` is the outer edge pair on the axis's integration coordinate and `u` the
# nodes on that same coordinate. An edge half a node step beyond the outermost
# node is a property of the node SPACING and knows nothing about where the
# parameter stops existing, so on a grid graded toward a boundary it steps past
# it: the proper-CAR nodes `c(0.5, 0.8, 0.95, 0.99)` close at `rho = 1.01`,
# which is not a correlation, and a sampler told to target the same measure
# takes that as the flat prior's support (gcol33/tulpa#657).
#
# An edge outside the domain is replaced by the midpoint between the outermost
# node and the boundary, on the same coordinate. That is the half-step rule
# again with the boundary standing in for the next node, so the edge is never
# moved FURTHER out than the naive one, and on an open boundary it lands
# strictly inside: the outermost node is inside, so the midpoint of the two is.
# On a closed boundary the edge is the boundary itself. Exact equality with an
# open boundary is therefore a violation and is pulled in; equality with a
# closed one stands.
#
# A node already outside the declared domain contradicts the declaration. The
# convention there is the one `.nl_cell_partition()` takes on the reporting
# side: set the declaration aside rather than move the data, so the naive edge
# stands. That is decided PER SIDE, because the two ends are separate claims and
# a grid contradicting one says nothing about the other -- a proper-CAR axis laid
# on an adjacency eigenvalue interval reaches below zero, which contradicts a
# lower bound of 0 while leaving 1 as true an upper bound as it was.
.hyper_domain_clamp <- function(bd, u, spec) {
  dom <- .hyper_axis_domain(spec)
  if (is.null(dom) || length(u) < 1L) return(bd)
  b <- suppressWarnings(.hyper_axis_coord(dom$bounds, spec))
  if (length(b) != 2L || anyNA(b)) return(bd)
  inside <- function(x, k) {
    if (!is.finite(b[k])) return(rep(TRUE, length(x)))
    if (k == 1L) {
      if (dom$open[1L]) x > b[1L] else x >= b[1L]
    } else {
      if (dom$open[2L]) x < b[2L] else x <= b[2L]
    }
  }
  ends <- c(u[1L], u[length(u)])
  for (k in 1:2) {
    if (!is.finite(b[k]) || !all(inside(u, k)) || inside(bd[k], k)) next
    e <- if (dom$open[k]) (ends[k] + b[k]) / 2 else b[k]
    if (!is.finite(e)) e <- ends[k]
    bd[k] <- e
  }
  bd
}

# The `bounds` an engine-named axis carries on its spec: the natural support
# its NAME fixes, and nothing more. Read by the joint spec builder so the spec
# and the support rule cannot come to hold two different name-keyed claims
# about the same axis -- `rho` names a BYM2 mixing weight on (0, 1) in one
# family and an AR1 or multi-output cross-field correlation reaching below zero
# in two others, so a spec declaring (0, 1) by name alone refuses a grid those
# families lay legitimately.
.hyper_spec_bounds <- function(bare) {
  d <- .hyper_axis_domain(list(name = bare))
  if (is.null(d)) NULL else d$bounds
}

# Coordinate bounds the cells of one node set tile: the half-node-step
# extrapolation, closed inside the axis's declared domain. The ONE construction
# behind both the level weights and the reported support, so the interval the
# prior is normalised over and the interval a sampler is told the quadrature
# reached are the same interval.
.hyper_axis_coord_bounds <- function(u, spec) {
  .hyper_domain_clamp(.hyper_default_coord_bounds(u), u, spec)
}

# Prior weight per level on one axis.
#
# The rule is cell width times declared density on the integration coordinate.
# With the default flat density over a declared span the widths tile the span
# and the weights reduce to the even spacing the engine has always used. With a
# proper density the widths carry the quadrature and the density carries the
# prior, so extending or densifying the nodes changes only how well the same
# measure is integrated.
#
# `atom_mass` is the prior probability of the zero level on an axis that carries
# one. It is declared, so it does not move when refinement changes how many
# continuum nodes there are, and the continuum carries `1 - atom_mass`.
#
# `absolute = TRUE` returns each level's ABSOLUTE measure instead: its cell width
# on the integration coordinate times the declared density there, with no
# normalisation over the grid, and the atom at its declared probability. That is
# the measure the evidence sums against (`.nl_outer_log_evidence()`); the
# weights are the same up to the constant the normalisation removes.
.hyper_axis_level_weights <- function(levels, spec, atom_mass = NULL,
                                      close_domain = TRUE, absolute = FALSE) {
  .hyper_axis_measure(levels, spec, atom_mass, close_domain = close_domain,
                      absolute = absolute)$w
}

# The level weights of one axis together with the pieces a refined fibre needs
# to measure a node the levels do not contain: the continuum nodes and the cell
# edges they own on the integration coordinate, and `unit(x)`, the weight a
# unit of coordinate width carries at continuum node `x` under the same density,
# span and normalisation the level weights were built with -- so for every
# continuum level `w == width * unit(level)`. `unit` is NULL where the axis has
# no continuum to extend (a single level, or an atom alone).
.hyper_axis_measure <- function(levels, spec, atom_mass = NULL,
                                close_domain = TRUE, absolute = FALSE) {
  levels <- sort(unique(as.numeric(levels)))
  K <- length(levels)
  none <- list(w = numeric(0), levels = levels, x = numeric(0),
               edges = numeric(0), unit = NULL, slab = NULL)
  if (K == 0L) return(none)
  nm <- format(levels, digits = 15)
  if (K == 1L) {
    none$w <- stats::setNames(1, nm)
    return(none)
  }

  is_atom <- isTRUE(spec$log_scale) & levels == 0
  has_atom <- any(is_atom) && !is.null(atom_mass)
  if (any(is_atom) && !has_atom) {
    stop(sprintf("Axis '%s' carries a zero level but no declared atom mass.",
                 spec$name), call. = FALSE)
  }

  w <- numeric(K)
  cont <- !is_atom
  if (!any(cont)) {
    w[is_atom] <- 1
    none$w <- stats::setNames(w, nm)
    return(none)
  }

  # A flat measure on a log axis is improper, so either a span or a proper
  # density has to make it one. `slab_bounds` declares the span; a declared
  # density needs none and lets the nodes reach as far as the data asks.
  slab <- spec$slab_bounds
  if (!is.null(slab)) {
    inside <- cont & levels >= slab[1L] & levels <= slab[2L]
    if (!any(inside)) {
      stop(sprintf("Axis '%s': no node lies inside `slab_bounds`.", spec$name),
           call. = FALSE)
    }
    cont <- inside
  }

  x  <- levels[cont]
  u  <- .hyper_axis_coord(x, spec)
  bd <- if (is.null(slab)) .hyper_default_coord_bounds(u)
        else sort(.hyper_axis_coord(slab, spec))
  # The outer cells are closed inside the axis's declared domain whichever
  # rule placed them, so the measure is never normalised over a region the
  # parameter does not live on and the weights cannot disagree with the
  # support about where the outermost cell ends.
  #
  # `close_domain = FALSE` leaves the naive half-step mirror standing. That is
  # not a second measure to integrate against -- nothing reports it and no
  # posterior is normalised over it -- it is the measure a boundary DETECTOR
  # reads, which must not weaken because the axis's support happens to close
  # where it is looking (`.nl_axis_marginal_w(measure = "span")`).
  if (close_domain) bd <- .hyper_domain_clamp(bd, u, spec)
  width <- .hyper_cell_widths(u, bd[1L], bd[2L])

  dens <- spec$slab_log_density
  if (is.null(dens)) {
    # Flat over the declared span: the widths themselves are the shape. As an
    # absolute measure a declared span is a uniform prior on it, unless the axis
    # carries a density folded into the marginal instead (`log_prior`), whose
    # cells are measured by their widths alone.
    cw <- width
    per_width <- function(xx) rep(1, length(xx))
    if (absolute && !is.null(slab) && is.null(spec$log_prior)) {
      cw <- width / diff(bd)
      per_width <- function(xx) rep(1 / diff(bd), length(xx))
    }
  } else {
    # Declared density, carried to the coordinate the widths are measured on.
    ld <- .hyper_prior_carry(x, dens, spec$log_scale,
                             spec$slab_log_density_coord %||% "natural")
    cw <- width * exp(ld)
    cw[!is.finite(cw)] <- 0
    per_width <- function(xx) {
      d <- exp(.hyper_prior_carry(xx, dens, spec$log_scale,
                                  spec$slab_log_density_coord %||% "natural"))
      d[!is.finite(d)] <- 0
      d
    }
  }
  # The density shapes the continuum; it does not set how much of the axis the
  # continuum holds. That is `1 - atom_mass`, declared before the fit, so the
  # shape is normalised and the split cannot move when nodes are added, when a
  # grid stops short of the density's tail, or when the density is rescaled.
  scale <- 1
  if (!absolute) {
    tot <- sum(cw)
    if (is.finite(tot) && tot > 0) {
      cw <- cw / tot
      scale <- 1 / tot
    } else {
      cw <- rep(1 / length(cw), length(cw))
      scale <- 1 / sum(width)
      per_width <- function(xx) rep(1, length(xx))
    }
  }

  if (has_atom && absolute) {
    a <- as.numeric(atom_mass)
    w[is_atom] <- a
    w[cont]    <- (1 - a) * cw
    scale <- (1 - a) * scale
  } else if (has_atom) {
    a <- as.numeric(atom_mass)
    if (!is.finite(a) || a < 0 || a >= 1) {
      stop(sprintf("Axis '%s': `atom_mass` must lie in [0, 1).", spec$name),
           call. = FALSE)
    }
    w[is_atom] <- a * .hyper_atom_fold_scale(x, cw, spec)
    w[cont]    <- (1 - a) * cw
    scale <- (1 - a) * scale
  } else {
    w[cont] <- cw
  }
  list(w = stats::setNames(w, nm), levels = levels, x = x,
       edges = c(bd[1L], (u[-length(u)] + u[-1L]) / 2, bd[2L]),
       unit = function(xx) scale * per_width(xx), slab = slab)
}

# Scale the atom is weighed against the continuum on.
#
# An axis's `log_prior` is a density on its continuum, and the driver folds it
# into `log_marginal`, so every continuum cell picks up `exp(lp)` downstream
# while the atom -- which is not a point of that continuum, and is why the
# axis declares its prior probability rather than integrating it -- picks up
# nothing. Weighing the atom at the continuum's density-weighted mean puts the
# two on one scale, so the declared split is the split the fit integrates
# whatever the density is (gcol33/tulpa#624, gcol33/tulpa#626).
#
# 1 where the axis declares no such density, and where the density leaves the
# whole continuum with no mass -- there the atom holds the axis on its own.
.hyper_atom_fold_scale <- function(x, cw, spec) {
  fn <- spec$log_prior
  if (is.null(fn)) return(1)
  lp <- .hyper_prior_carry(x, fn, spec$log_scale,
                           spec$log_prior_coord %||% "integration")
  s <- sum(cw * exp(lp))
  if (!is.finite(s) || s <= 0) 1 else s
}

# Prior probability of the copy scale's "no coupling" point mass: equal odds on
# a coupled and an uncoupled field, stated rather than inherited from a node
# count. Fixed, so refining the copy axis cannot move it.
.TULPA_COPY_ATOM_MASS <- 1 / 2

# Proper density for the copy scale's continuum: an exponential, the penalized
# complexity prior for a scale parameter with its base model at zero (Simpson et
# al. 2017). The rate is read off the grid the caller declared, by putting 5 % of
# the prior mass above its largest node, so the prior is fixed before the fit and
# is weakly informative relative to the range the caller thought plausible.
#' The copy-scale axis's PC-prior density
#'
#' The proper density tulpa uses for the copy scale's continuum: an
#' exponential, the penalized-complexity prior for a scale parameter with its
#' base model at zero (Simpson et al. 2017). The rate is set by putting 5% of
#' the prior mass above `upper`, so a caller that reads this off a fit's own
#' declared grid gets the exact rate the outer integration used -- rather
#' than restating a PC-prior rate that could silently drift from it.
#'
#' @param upper The largest declared node of the copy-scale axis.
#' @return A function `log p(x) = log(lambda) - lambda * x`, or `NULL` if
#'   `upper` is not a finite positive number.
#' @seealso [tulpa_hyper_check_copy_slab()], [tulpa_joint_axis_specs_from_grid()]
#' @examples
#' lp <- tulpa_hyper_copy_slab_density(upper = 2)
#' lp(c(0.1, 1, 2))
#' @export
tulpa_hyper_copy_slab_density <- function(upper) .hyper_copy_slab_density(upper)

.hyper_copy_slab_density <- function(upper) {
  upper <- as.numeric(upper)
  if (!is.finite(upper) || upper <= 0) return(NULL)
  lambda <- -log(0.05) / upper
  function(x) log(lambda) - lambda * x
}

# Integration coordinate of an engine axis, by name: TRUE where the grid is laid
# out log-spaced, FALSE where it is laid out evenly, and NA for a name this table
# does not cover.
#
# An axis's coordinate is a property of how its grid was built, so it is declared
# rather than read off the node values -- inferring it from the spacing would let
# the data choose the measure. A caller that builds its own grid declares its own
# specs (`.nl_st_axis_specs()` is the spatiotemporal one); this table is for the
# axes the joint and single-block dispatchers name.
#
# A BOUNDED axis -- a correlation, a mixing weight -- is integrated on its
# NATURAL coordinate, and that is a statement about the measure rather than
# about the nodes. A scale has no natural unit, so its non-informative measure
# is the multiplicative one and it is spaced and integrated in log; a
# correlation's domain is bounded and its endpoints are models in their own
# right (independence at one end, an intrinsic field at the other), so the
# uniform measure on that domain is proper and is what the flat default means.
# The logit measure is improper on the same domain and puts as much prior mass
# on the last percent below 1 -- where a proper-CAR field is numerically
# intrinsic -- as on the whole middle of the range.
#
# Node PLACEMENT is a separate choice and is not evidence about the coordinate:
# `.hyper_axis_level_weights()` gives every node the width of the cell it owns,
# so a grid graded toward a boundary, where the inner marginal changes fastest,
# integrates the same declared measure as an evenly spaced one. What the
# outermost cell is closed with is the axis's declared DOMAIN
# (`.hyper_domain_clamp()`), not an extrapolation of the node spacing.
#
# NA is the safe answer, not an error: an axis nobody has classified carries no
# quadrature weight, so its nodes stay equally weighted.
.hyper_axis_scale <- function(bare) {
  if (startsWith(bare, "rho")) return(FALSE)
  if (bare %in% c("sigma", "sigma2", "alpha", "tau", "range", "lengthscale", "ell",
                  "s1", "s2", "phi", "phi_gp") ||
      startsWith(bare, "sigma") || startsWith(bare, "tau") ||
      startsWith(bare, "phi_")) return(TRUE)
  NA
}

# Natural-scale support of one axis's continuum: the interval its levels tile on
# the integration coordinate. `slab_bounds` where the axis declares a span, half
# a node step beyond the outermost levels otherwise. This is the region the outer
# quadrature actually reaches, so a sampler asked to target the same measure is
# bounded here and not at the outermost node.
#
# Either rule is closed inside the axis's declared domain, so a bounded axis --
# a correlation, a mixing weight, a probability -- never reports a support
# containing a value its parameter cannot take. The invariant is the one
# `.nl_cell_edges()` states on the reporting side: whenever every level lies
# inside a declared domain, both ends of the returned interval do too.
#
# Returns NULL for an axis with fewer than two continuum levels, which is the
# pinned case the caller leaves out of the sampled vector.
.hyper_axis_support <- function(levels, spec) {
  levels <- sort(unique(as.numeric(levels)))
  levels <- levels[is.finite(levels)]
  cont <- if (isTRUE(spec$log_scale)) levels[levels > 0] else levels
  if (length(cont) < 2L) return(NULL)
  u <- .hyper_axis_coord(cont, spec)
  if (!is.null(spec$slab_bounds)) {
    nat <- sort(as.numeric(spec$slab_bounds))
    if (sum(cont >= nat[1L] & cont <= nat[2L]) < 2L) return(NULL)
    bd <- sort(.hyper_axis_coord(nat, spec))
    cl <- .hyper_domain_clamp(bd, u, spec)
    # A declared span already inside the domain is returned exactly as
    # declared, rather than round-tripped through the integration coordinate.
    if (identical(cl, bd)) return(nat)
    return(if (isTRUE(spec$log_scale)) exp(cl) else cl)
  }
  bd <- .hyper_axis_coord_bounds(u, spec)
  if (isTRUE(spec$log_scale)) exp(bd) else bd
}

# The support of every axis in `specs` that carries one, as a named list of
# natural-scale intervals. Stored on the fit so a second engine reads the span
# this fit integrated rather than rebuilding it from a grid that refinement may
# since have extended.
#
# `refining` is the per-cell tag `.hyper_log_quad_weights()` takes. An axis
# carrying slice cells reports the region the cell-by-cell measure integrates
# (`.hyper_refined_axis_support()`); every other axis, and every axis of a grid
# with no slice cells, the support of its levels. An axis with no declared
# coordinate has no measure to read a region off and keeps the latter.
#' Per-axis integrated support of an outer grid
#'
#' The natural-scale support of every axis in `specs` that carries one, as a
#' named list of intervals -- the region each axis's outer-grid measure
#' actually integrates, accounting for refinement slice cells. Intended for
#' recovering an axis's integrated span from a settled grid, e.g. so a
#' sampled-hyperparameter prior can be derived from what the outer
#' integration used rather than restated alongside it.
#'
#' @param theta_grid A named `[n_cells x n_axes]` matrix (see
#'   [tulpa_theta_matrix()]).
#' @param specs Per-axis spec list, e.g. from
#'   [tulpa_joint_axis_specs_from_grid()].
#' @param refining Optional per-cell refinement-slice tag vector (see
#'   [tulpa_hyper_slice_home()]).
#' @return A named list of natural-scale `c(lo, hi)` intervals, one per axis
#'   in `specs` that carries a declared coordinate, or `NULL` if `theta_grid`
#'   or `specs` is `NULL`.
#' @seealso [tulpa_joint_axis_specs_from_grid()], [tulpa_hyper_slice_home()]
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
#' tg <- tulpa_theta_matrix(fit)
#' tulpa_hyper_grid_supports(tg, tulpa_joint_axis_specs_from_grid(tg))
#' }
#' @export
tulpa_hyper_grid_supports <- function(theta_grid, specs, refining = NULL) {
  .hyper_grid_supports(theta_grid, specs, refining = refining)
}

.hyper_grid_supports <- function(theta_grid, specs, refining = NULL) {
  if (is.null(theta_grid) || is.null(specs)) return(NULL)
  theta_grid <- as.matrix(theta_grid)
  axis_names <- colnames(theta_grid)
  home <- .hyper_slice_home(refining, nrow(theta_grid))
  sliced <- unique(home[nzchar(home)])
  out <- list()
  for (spec in specs) {
    a <- spec$name
    if (!a %in% axis_names) next
    sup <- if (a %in% sliced && !isTRUE(spec$unweighted)) {
      .hyper_refined_axis_support(theta_grid, spec, home)
    } else {
      .hyper_axis_support(theta_grid[, a], spec)
    }
    if (!is.null(sup)) out[[a]] <- sup
  }
  if (length(out) == 0L) return(NULL)
  out
}

# Declared atom mass for one axis, or NULL where the axis carries no point mass.
.hyper_axis_atom_mass <- function(spec) {
  if (is.null(spec$atom_mass)) return(NULL)
  as.numeric(spec$atom_mass)
}

# Per-cell log quadrature weight over the whole product grid: the sum across
# axes of the log weight of the level that cell sits at.
#
# Returns a zero vector when no axis contributes, so a caller can add it
# unconditionally.
#
# `refining` is the per-cell tag the refinement passes leave (`""` for a cell of
# the base tensor, `<axis>` or `consistency_<axis>` for a slice cell placed on
# that axis). A slice cell is one point evaluation at `(pt, z*)`: a new level
# `pt` on its axis at ONE combination `z*` of the others. The product rule above
# is a tensor measure, correct only while every level of an axis appears in
# every row of the others, so a grid carrying slice cells is measured cell by
# cell instead (`.hyper_refined_log_quad()`): each cell owns a box, the boxes
# partition what the base tensor's boxes partitioned, and a slice cell's box is
# carved out of the base boxes of its own row. A grid with no slice cells takes
# the product rule unchanged.
.hyper_log_quad_weights <- function(theta_grid, specs, close_domain = TRUE,
                                    absolute = FALSE, refining = NULL) {
  if (is.null(theta_grid) || is.null(specs)) return(NULL)
  theta_grid <- as.matrix(theta_grid)
  n <- nrow(theta_grid)
  if (n == 0L) return(numeric(0))
  home <- .hyper_slice_home(refining, n)
  if (any(nzchar(home))) {
    return(.hyper_refined_log_quad(theta_grid, specs, home,
                                   close_domain = close_domain,
                                   absolute = absolute))
  }
  axis_names <- colnames(theta_grid)
  groups <- .hyper_logchol_groups(specs, axis_names)
  group_cols <- .hyper_logchol_group_cols(groups)
  out <- numeric(n)
  for (spec in specs) {
    a <- spec$name
    if (!a %in% axis_names || a %in% group_cols) next
    if (isTRUE(spec$unweighted)) {
      # An axis with no declared coordinate has no cell width to measure it by.
      if (absolute) out <- out + NA_real_
      next
    }
    v <- as.numeric(theta_grid[, a])
    lw <- .hyper_axis_level_weights(v, spec, .hyper_axis_atom_mass(spec),
                                    close_domain = close_domain,
                                    absolute = absolute)
    levels <- sort(unique(v))
    idx <- match(v, levels)
    contrib <- log(as.numeric(lw)[idx])
    contrib[!is.finite(contrib)] <- -Inf
    out <- out + contrib
  }
  .hyper_add_logchol_measure(out, theta_grid, groups, close_domain, absolute)
}

# The free-covariance blocks among a grid's axes: each complete set of
# log-Cholesky columns (`.hp_logchol_block_cols()`) whose specs the per-axis
# builder left unclassified, with the design its specs declare
# (`logchol_design`, stamped by `.joint_axis_specs_from_grid()`). Their cells are
# measured as one block, on the coordinates that design names.
.hyper_logchol_groups <- function(specs, axis_names) {
  unw <- Filter(function(s) isTRUE(s$unweighted) && s$name %in% axis_names,
                specs)
  names_unw <- vapply(unw, function(s) s$name, character(1))
  cand <- names_unw[.hp_is_logchol_col(.hyper_axis_bare(names_unw))]
  out <- list()
  for (pre in unique(.hp_col_prefix_vec(cand))) {
    cols <- .hp_logchol_block_cols(axis_names, pre)
    if (is.null(cols)) next
    design <- unw[[match(cols[1L], names_unw)]]$logchol_design
    out[[length(out) + 1L]] <- list(cols = cols, design = design)
  }
  out
}

.hyper_logchol_group_cols <- function(groups) {
  unlist(lapply(groups, function(g) g$cols))
}

# Add each free-covariance block's per-cell log measure to `out`. A block whose
# declared grid is a tensor in no coordinates the engine integrates on has no
# measure, and as an absolute measure that is NA, the same answer an
# unclassified axis gives.
.hyper_add_logchol_measure <- function(out, theta_grid, groups, close_domain,
                                       absolute) {
  for (g in groups) {
    lm <- .hyper_logchol_log_measure(theta_grid[, g$cols, drop = FALSE],
                                     g$design,
                                     close_domain = close_domain,
                                     absolute = absolute)
    if (is.null(lm)) {
      if (absolute) out <- out + NA_real_
    } else {
      out <- out + lm
    }
  }
  out
}

# Per-cell log measure of one free-covariance block: the product of its cell
# widths in the coordinates of its declared `design` (`.hp_logchol_design()`),
# log sigma on each standard deviation and the natural value on the correlation
# for the two-field default, the log-Cholesky columns themselves otherwise --
# the coordinates `.hp_logchol_log_density()` puts the prior on. The widths are
# read off the levels the rows of `M` take in those coordinates, so a subset of
# the declared tensor is measured on its own levels as every other axis is.
# NULL when the block has no design.
.hyper_logchol_log_measure <- function(M, design, close_domain = TRUE,
                                       absolute = FALSE) {
  if (is.null(design$design)) return(NULL)
  C <- signif(.hp_logchol_coords(M, design), 12L)
  axes <- if (identical(design$design, "sd_rho")) {
    list(list(spec = list(name = "sigma", log_scale = TRUE), nat = exp),
         list(spec = list(name = "sigma", log_scale = TRUE), nat = exp),
         list(spec = list(name = "logchol_rho", log_scale = FALSE,
                          bounds = c(-1, 1)), nat = identity))
  } else {
    rep(list(list(spec = list(name = "logchol", log_scale = FALSE),
                  nat = identity)), ncol(C))
  }
  out <- numeric(nrow(C))
  for (j in seq_len(ncol(C))) {
    x <- axes[[j]]$nat(C[, j])
    lev <- sort(unique(x))
    lw <- .hyper_axis_level_weights(lev, axes[[j]]$spec,
                                    close_domain = close_domain,
                                    absolute = absolute)
    contrib <- log(as.numeric(lw)[match(x, lev)])
    contrib[!is.finite(contrib)] <- -Inf
    out <- out + contrib
  }
  out
}

# The axis each cell was placed on by a refinement pass, `""` for a base cell.
# A `refining` vector of the wrong length describes some other grid and is an
# error rather than a guess.
#' Per-cell refinement-slice tag of an outer grid
#'
#' The axis each cell of a nested-Laplace outer grid was placed on by a
#' refinement pass, `""` for a base (unrefined) tensor cell. Used to tell a
#' base grid apart from its refinement slices wherever a reader needs to
#' restrict to one or the other, e.g. rebuilding axis specs from a grid's
#' declared nodes only (see [tulpa_joint_axis_specs_from_grid()]).
#'
#' @param refining The `refining` tag vector stored on a fit (`NULL` for a
#'   grid with no refinement).
#' @param n Number of grid cells; `refining`, if not `NULL`, must have this
#'   length.
#' @return A character vector of length `n`: `""` for a base cell, the axis
#'   name for a refinement-slice cell.
#' @seealso [tulpa_hyper_grid_supports()], [tulpa_joint_axis_specs_from_grid()]
#' @examples
#' tulpa_hyper_slice_home(NULL, 3)
#' tulpa_hyper_slice_home(c("", "", "sigma"), 3)
#' @export
tulpa_hyper_slice_home <- function(refining, n) .hyper_slice_home(refining, n)

.hyper_slice_home <- function(refining, n) {
  if (is.null(refining)) return(rep("", n))
  if (length(refining) != n) {
    stop(sprintf("`refining` has %d entries for a grid of %d cells.",
                 length(refining), n), call. = FALSE)
  }
  out <- sub("^consistency_", "", as.character(refining))
  out[is.na(out)] <- ""
  out
}

# Is a cell a base-tensor cell or a slice cell on `axis`? The cells a refinement
# pass on `axis` may anchor at, so every slice cell's coordinates off its own
# axis are levels of the base tensor.
.hyper_slice_anchor_ok <- function(refining, axis, n) {
  home <- .hyper_slice_home(refining, n)
  !nzchar(home) | home == axis
}

# Cell-by-cell measure of a grid that carries refinement slice cells.
#
# The base tensor's cells own boxes, the product of their per-axis level cells.
# Slice cells on axis j at row r (the base coordinates off j) re-tile that ONE
# row along j: the fibre's nodes are the base levels plus the row's slice
# points, its interior edges the midpoints between them, and its outer edges the
# wider of the base edges and the fibre's own half-step mirror
# (`.hyper_fibre_tiling()`, which `.hyper_refined_axis_support()` reads the
# integrated span from). The row integrates the base span together with its
# slice nodes' refined cells. In the fibre each base node keeps the part `R_j`
# of its base cell `B_j` its refined cell still covers, plus the part of its
# refined cell past the base span on a side where a slice node lies beyond it;
# each slice node owns its refined cell, which lies in one or two base cells
# and, past the base span, in the extension region. The pieces tile the row's
# region with no gap.
#
# Inside the box `B` of a base cell c0 a point x lies outside `R_j` on some set
# `T` of axes. For `T` empty it belongs to c0; for `T = {j}` to the j-slice whose
# fibre cell covers `x_j`, the only claimant there, because a j-slice's box off j
# is its row's base box. Where refinements on several axes meet (`|T| > 1`) each
# of them has an equal claim, and the region is split equally between them. With
# `f_k = |R_k| / |B_k|` the share of c0's box a j-slice piece of width `w` holds
# is therefore
#
#   (w / |B_j|) * integral_0^1 prod_{k != j} (f_k + (1 - f_k) t) dt,
#
# and c0 keeps `prod_k f_k`; summed over the families these are the whole box, so
# the base tensor's mass is conserved whatever is inserted.
#
# Past the base span the same rule decides. The base node of c0 closes an
# extended row on axis k when its row along k reaches past the base edge on the
# node's side. c0's extended box is, on each axis, `B_k` together with `P_k`, the
# region that row reaches past the edge (`span_ext`), empty on an axis where the
# node closes no extended row. Along each axis the row through c0 splits that
# interval into the part c0's node owns, `R_k` and its extension `e_k`, and the
# slice cells' parts, and a point of the extended box is c0's where every axis
# gives c0's node and is split equally among the slices naming it otherwise. A
# slice cell is therefore nearest past an edge of another axis exactly as it is
# inside the base box. With `e_k` and `p_k = |P_k|` in units of `|B_k|`, c0
# carries
#
#   prod_k (f_k + e_k),
#
# and a j-slice piece of width `w` in c0's extended box on axis j, inside the
# base span or past it, carries relative to c0's base box off j
#
#   w * integral_0^1 prod_{k != j} (f_k + e_k + (1 + p_k - f_k - e_k) t) dt,
#
# which is the in-box share above wherever c0 closes no extended row. The
# corner past two edges is c0's where both axes give its node, a slice's where
# one axis gives that slice, and half each where both give slices. Every
# extended box is tiled exactly, so the measure integrates the union of the base
# cells' extended boxes, whose extent along each axis is the span
# `.hyper_refined_axis_support()` reports. Along each axis a width is
# converted to a weight at the owning node through the axis's `unit()`, the
# conversion its level weights were built with, so a grid with no slice cells
# reproduces the product rule.
#
# Every slice cell must sit on base levels off its own axis -- the anchoring
# rule `.hyper_slice_anchor_ok()` enforces -- or its row has no base box.
.hyper_refined_log_quad <- function(theta_grid, specs, home, close_domain,
                                    absolute) {
  n <- nrow(theta_grid)
  axis_names <- colnames(theta_grid)
  base <- !nzchar(home)
  if (!any(base)) {
    stop("A refined outer grid carries no base-tensor cells to measure against.",
         call. = FALSE)
  }
  unknown <- setdiff(unique(home[!base]), axis_names)
  if (length(unknown)) {
    stop(sprintf("Refinement slice cells name axes the grid does not carry: %s.",
                 paste(unknown, collapse = ", ")), call. = FALSE)
  }
  groups <- .hyper_logchol_groups(specs, axis_names)
  group_cols <- .hyper_logchol_group_cols(groups)
  if (length(intersect(unique(home[!base]), group_cols))) {
    stop("Refinement slice cells sit on a free-covariance block's axes, which ",
         "are measured as one block.", call. = FALSE)
  }
  out <- .hyper_add_logchol_measure(numeric(n), theta_grid, groups,
                                    close_domain, absolute)
  measures <- list()
  for (spec in specs) {
    a <- spec$name
    if (!a %in% axis_names || a %in% group_cols) next
    if (isTRUE(spec$unweighted)) {
      if (absolute) out <- out + NA_real_
      next
    }
    v <- as.numeric(theta_grid[, a])
    m <- .hyper_axis_measure(v[base], spec, .hyper_axis_atom_mass(spec),
                             close_domain = close_domain, absolute = absolute)
    idx <- match(v, m$levels)
    off <- home != a & is.na(idx)
    if (any(off)) {
      stop(sprintf(paste0("Refinement slice cell %d sits off the base levels ",
                          "on axis '%s'."), which(off)[1L], a), call. = FALSE)
    }
    contrib <- log(as.numeric(m$w)[idx])
    contrib[home == a] <- 0
    contrib[!is.finite(contrib)] <- -Inf
    out <- out + contrib
    m$spec <- spec
    measures[[a]] <- m
  }

  key_of <- .hyper_row_key
  cell_vals <- function(i) stats::setNames(as.numeric(theta_grid[i, ]),
                                           axis_names)

  # Re-tile every row that carries slice cells, per axis.
  fib <- list()
  slice_pieces <- vector("list", n)
  for (a in intersect(names(measures), unique(home[!base]))) {
    m <- measures[[a]]
    if (length(m$x) < 2L || is.null(m$unit)) {
      stop(sprintf(paste0("Axis '%s' carries refinement slice cells but no ",
                          "continuum of base levels to re-tile."), a),
           call. = FALSE)
    }
    xb <- m$x
    eb <- m$edges
    fib[[a]] <- list()
    rows <- .hyper_slice_rows(theta_grid, home, a)
    for (r in names(rows)) {
      s_idx <- rows[[r]]
      pts <- as.numeric(theta_grid[s_idx, a])
      tl <- .hyper_fibre_tiling(m, pts, close_domain)
      ok <- tl$ok
      bw <- diff(eb)
      key <- sprintf("%.17g", xb)
      fib[[a]][[r]] <- list(f = stats::setNames(tl$retained, key),
                            e = stats::setNames(tl$base_ext / bw, key),
                            p = stats::setNames(tl$span_ext / bw, key))
      for (k in seq_along(s_idx)) {
        i <- s_idx[k]
        if (!ok[k]) {
          slice_pieces[[i]] <- list(axis = a, in_base = numeric(0),
                                    ext = 0, x = pts[k])
          next
        }
        cl <- tl$cell[k, ]
        ov <- pmax(0, pmin(cl[["hi"]], eb[-1L]) -
                      pmax(cl[["lo"]], eb[-length(eb)]))
        slice_pieces[[i]] <- list(
          axis = a, x = pts[k],
          in_base = stats::setNames(ov, sprintf("%.17g", xb)),
          ext = max(0, (cl[["hi"]] - cl[["lo"]]) - sum(ov)),
          ext_node = if (cl[["hi"]] > eb[length(eb)]) xb[length(xb)] else xb[1L],
          bw = bw)
      }
    }
  }

  # A base coordinate's retained fraction on axis `k` (`part = "f"`), the
  # extension piece its node owns past the base span (`part = "e"`), or the
  # whole of its row's region past the base edge that node closes (`part =
  # "p"`), the last two in units of its base cell width: 1, 0 and 0 unless its
  # row along `k` was re-tiled.
  row_part <- function(vals, k, part) {
    none <- if (identical(part, "f")) 1 else 0
    fk <- fib[[k]]
    if (is.null(fk)) return(none)
    rr <- fk[[key_of(vals, k)]]
    if (is.null(rr)) return(none)
    f <- rr[[part]][sprintf("%.17g", vals[[k]])]
    if (length(f) != 1L || is.na(f)) none else as.numeric(f)
  }
  frac <- function(vals, k) row_part(vals, k, "f")
  refined_axes <- names(fib)

  # A base cell keeps, along each axis, the part of its extended interval its
  # node owns, `prod_k (f_k + e_k)` of its base box.
  for (i in which(base)) {
    vals <- cell_vals(i)
    es <- vapply(refined_axes, function(k) row_part(vals, k, "e"), numeric(1))
    if (all(es == 0)) {
      for (k in refined_axes) {
        out[i] <- out[i] + log(frac(vals, k))
      }
    } else {
      fs <- vapply(refined_axes, function(k) frac(vals, k), numeric(1))
      out[i] <- out[i] + log(prod(fs + es))
    }
  }

  # What a slice piece on axis `a` holds of base cell `c0`'s extended box off
  # `a`, in units of its base box there: along each other axis the node owns
  # `f + e` of the extended interval `1 + p` and the slice cells the rest.
  piece_share <- function(c0, a) {
    others <- setdiff(refined_axes, a)
    fs <- vapply(others, function(k) frac(c0, k), numeric(1))
    es <- vapply(others, function(k) row_part(c0, k, "e"), numeric(1))
    ps <- vapply(others, function(k) row_part(c0, k, "p"), numeric(1))
    .hyper_corner_share(fs + es, (1 + ps) - fs - es)
  }

  for (i in which(!base)) {
    pc <- slice_pieces[[i]]
    a <- pc$axis
    unit_a <- measures[[a]]$unit(pc$x)
    total <- pc$ext
    if (total > 0) {
      c0 <- cell_vals(i)
      c0[[a]] <- pc$ext_node
      total <- total * piece_share(c0, a)
    }
    if (length(pc$in_base)) {
      vals <- cell_vals(i)
      for (lev in names(pc$in_base)) {
        w <- pc$in_base[[lev]]
        if (w <= 0) next
        c0 <- vals
        c0[[a]] <- measures[[a]]$x[match(lev, sprintf("%.17g", measures[[a]]$x))]
        total <- total + w * piece_share(c0, a)
      }
    }
    lw <- log(total * unit_a)
    out[i] <- out[i] + (if (is.finite(lw)) lw else -Inf)
  }
  out
}

# The key naming a cell's row along axis `drop`: its coordinates on every other
# axis.
.hyper_row_key <- function(vals, drop) {
  keep <- names(vals) != drop
  paste0("row:", paste(sprintf("%.17g", vals[keep]), collapse = "|"))
}

# The slice cells on axis `a`, grouped by the row of the base tensor each sits
# in, as a named list of cell indices in first-appearance order.
.hyper_slice_rows <- function(theta_grid, home, a) {
  sl <- which(home == a)
  axis_names <- colnames(theta_grid)
  keys <- vapply(sl, function(i) {
    .hyper_row_key(stats::setNames(as.numeric(theta_grid[i, ]), axis_names), a)
  }, character(1))
  out <- lapply(unique(keys), function(r) sl[keys == r])
  names(out) <- unique(keys)
  out
}

# One row of a refined axis re-tiled: the continuum nodes of the base measure
# `m` (`.hyper_axis_measure()`, with `m$spec` attached) joined to that row's
# slice points `pts`, and what each node of the row owns of the region the row
# integrates.
#
# `ok` marks the slice points the continuum admits: finite, positive on a
# log-scale axis, inside a declared span. The fibre cells are the joined nodes'
# nearest-node cells on the integration coordinate: interior edges at the
# midpoints between neighbours, outer edges the wider of the base measure's
# outer edges and the joined nodes' own half-step mirror, closed inside the
# axis's domain when `close_domain`; under a declared span the outer edges are
# the base measure's, which already close on it.
#
# The row integrates `region`: the base span together with the fibre cells of
# its admitted slice points. Every node owns its fibre cell inside `region`, and
# a base node, which owns its base cell's box on every other axis, is held to
# its own base cell inside the base span. Those pieces tile `region` with no gap
# and no overlap. A base node's fibre cell reaches past the base span only on a
# side where a slice point lies beyond it, up to the midpoint towards that
# point; on a side with no slice point beyond, its cell ends at the fibre's
# mirror edge, outside `region`.
#
#   cell      two-column matrix (`lo`, `hi`), one row per entry of `pts`: the
#             fibre cell that point owns, NA where the point is not admitted
#   retained  per base node, the fraction of its base cell its fibre cell covers
#   base_ext  per base node, the width it owns past the base span. A width at
#             the rounding of the edge arithmetic is zero: the midpoint towards
#             an evenly spaced extension point and the base edge are one number
#             reached by two computations.
#   span_ext  per base node, the width of `region` past the base edge that node
#             closes: its own `base_ext` together with the slice cells beyond
#             it. Zero for a node that closes no edge.
#   region    c(lo, hi), the interval the row integrates
.hyper_fibre_tiling <- function(m, pts, close_domain) {
  spec <- m$spec
  eb <- m$edges
  ok <- is.finite(pts) & !(isTRUE(spec$log_scale) & pts <= 0)
  if (!is.null(m$slab)) ok <- ok & pts >= m$slab[1L] & pts <= m$slab[2L]
  xf <- sort(unique(c(m$x, pts[ok])))
  uf <- .hyper_axis_coord(xf, spec)
  bd <- if (!is.null(m$slab)) {
    c(eb[1L], eb[length(eb)])
  } else {
    bf <- .hyper_default_coord_bounds(uf)
    if (close_domain) bf <- .hyper_domain_clamp(bf, uf, spec)
    c(min(eb[1L], bf[1L]), max(eb[length(eb)], bf[2L]))
  }
  ef <- c(bd[1L], (uf[-length(uf)] + uf[-1L]) / 2, bd[2L])
  lo <- ef[-length(ef)]
  hi <- ef[-1L]
  jf <- match(pts, xf)
  jf[!ok] <- NA_integer_
  cell <- cbind(lo = lo[jf], hi = hi[jf])

  region <- c(eb[1L], eb[length(eb)])
  if (any(ok)) {
    region <- c(min(region[1L], cell[ok, "lo"]),
                max(region[2L], cell[ok, "hi"]))
  }

  ib <- match(m$x, xf)
  f <- (pmin(hi[ib], eb[-1L]) - pmax(lo[ib], eb[-length(eb)])) / diff(eb)
  f[!is.finite(f)] <- 1
  ext <- pmax(0, pmin(hi[ib], region[2L]) - eb[length(eb)]) +
         pmax(0, eb[1L] - pmax(lo[ib], region[1L]))
  resolution <- 16 * .Machine$double.eps * max(abs(ef[is.finite(ef)]), 1)
  ext[ext <= resolution] <- 0
  nb <- length(m$x)
  span_ext <- numeric(nb)
  span_ext[nb] <- max(0, region[2L] - eb[length(eb)])
  span_ext[1L] <- span_ext[1L] + max(0, eb[1L] - region[1L])

  list(ok = ok, x = xf, edges = ef, cell = cell,
       retained = pmin(pmax(f, 0), 1), base_ext = ext, span_ext = span_ext,
       region = region)
}

# Natural-scale support of an axis of a refined grid: the interval spanning the
# region its cell-by-cell measure (`.hyper_refined_log_quad()`) integrates.
#
# The base cells tile the support of the base levels, which is the unrefined
# rule applied to the node set the axis was declared on; slice cells on another
# axis sit on those levels and add nothing here. In a row re-tiled along this
# axis (`.hyper_fibre_tiling()`) the integrated `region` is the base span
# together with the cells its admitted slice points own, and the pieces its
# nodes own tile it without a gap. The support is the base support widened to
# every row's region.
#
# A densified row therefore leaves the span the declared nodes had, however
# many nodes it carries. NULL where the base levels carry fewer than two
# continuum nodes.
.hyper_refined_axis_support <- function(theta_grid, spec, home) {
  a <- spec$name
  v <- as.numeric(theta_grid[, a])
  base <- !nzchar(home)
  sup <- .hyper_axis_support(v[base], spec)
  if (is.null(sup) || !any(home == a)) return(sup)
  m <- .hyper_axis_measure(v[base], spec, .hyper_axis_atom_mass(spec))
  if (length(m$x) < 2L) return(sup)
  m$spec <- spec
  eb <- m$edges
  lo <- eb[1L]
  hi <- eb[length(eb)]
  for (s_idx in .hyper_slice_rows(theta_grid, home, a)) {
    rg <- .hyper_fibre_tiling(m, v[s_idx], close_domain = TRUE)$region
    lo <- min(lo, rg[1L])
    hi <- max(hi, rg[2L])
  }
  nat <- function(u) if (isTRUE(spec$log_scale)) exp(u) else u
  c(if (lo < eb[1L]) nat(lo) else sup[1L],
    if (hi > eb[length(eb)]) nat(hi) else sup[2L])
}

# integral_0^1 prod_k (f_k + g_k t) dt, `g` defaulting to `1 - f`: what a slice
# piece on one axis holds of the region it shares when, along each other axis k,
# the base node owns a width `f_k` of that region and slice cells own `g_k`, each
# corner piece split equally among the slices claiming it. `t^s` integrates to
# `1 / (s + 1)`, the equal share among a slice and `s` others. With `f + g = 1`
# it is a share of the region.
.hyper_corner_share <- function(f, g = 1 - f) {
  if (!length(f) || all(f == 1 & g == 0)) return(1)
  coef <- 1
  for (k in seq_along(f)) {
    coef <- c(coef * f[[k]], 0) + c(0, coef * g[[k]])
  }
  sum(coef / seq_along(coef))
}

# Accepted shapes for the copy scale's continuum measure. "exponential" is the
# penalized-complexity density above; "flat" makes the axis flat in log alpha
# over the declared span, the measure the other log-scale axes carry.
.TULPA_COPY_SLAB_CHOICES <- c("exponential", "flat")

#' Validate/default a copy-scale slab measure choice
#'
#' Validates a `copy_slab` argument -- `"exponential"` (the default) or
#' `"flat"`, the two continuum measures tulpa supports for a copy scale's
#' non-atom mass -- defaulting `NULL` to `"exponential"` and erroring on
#' anything else. Intended so a consumer package's own `copy_slab` argument
#' stays in sync with tulpa's own accepted choices rather than restating
#' them.
#'
#' @param x A `copy_slab` value: `NULL`, `"exponential"`, or `"flat"`.
#' @return `x`, defaulted to `"exponential"` when `NULL`.
#' @seealso [tulpa_hyper_copy_slab_density()]
#' @examples
#' tulpa_hyper_check_copy_slab("exponential")
#' try(tulpa_hyper_check_copy_slab("uniform"))
#' @export
tulpa_hyper_check_copy_slab <- function(x) .hyper_check_copy_slab(x)

.hyper_check_copy_slab <- function(x) {
  if (is.null(x)) return("exponential")
  if (!is.character(x) || length(x) != 1L || is.na(x) ||
      !x %in% .TULPA_COPY_SLAB_CHOICES) {
    stop(sprintf("`copy_slab` must be one of %s.",
                 paste(sprintf('"%s"', .TULPA_COPY_SLAB_CHOICES),
                       collapse = " or ")),
         call. = FALSE)
  }
  x
}

# Per-cell log quadrature weight for a grid whose axis specs the caller does not
# hold. Every path that turns `log_marginal` into posterior weights goes through
# here, so the prior mass a cell carries is decided in one place: on an evenly
# spaced grid the weights are equal and this is the rule the engine has always
# applied, and on an uneven one it is the spacing that differs rather than the
# measure.
#
# Specs rebuilt from the grid are rebuilt from its BASE cells when `refining`
# marks slice cells: an axis's declared node set is what fixes a prior read off
# it (the copy scale's exponential rate is set by its largest node), and a node a
# refinement pass appended was not declared.
#' Per-cell log quadrature weight of an outer grid
#'
#' The per-cell log quadrature weight (prior mass) of a nested-Laplace outer
#' grid: every path that turns a fit's `log_marginal` into posterior cell
#' weights goes through this one rule, so the prior mass a cell carries is
#' decided in one place. Rebuilds axis specs from the grid's own columns when
#' `specs` is not supplied. Intended for reconstructing a fit's outer-grid
#' posterior weights (with [tulpa_theta_matrix()] and
#' [tulpa_normalise_weights_safe()]) when the fit doesn't already carry
#' `fit$log_quad`.
#'
#' @param theta_grid A named `[n_cells x n_axes]` matrix (see
#'   [tulpa_theta_matrix()]).
#' @param specs Optional pre-built per-axis spec list; `NULL` rebuilds it
#'   from `theta_grid`'s columns via [tulpa_joint_axis_specs_from_grid()].
#' @param copy_slab `"exponential"` or `"flat"`; the copy-scale axis's
#'   continuum measure (see `?tulpa_joint_axis_specs_from_grid`).
#' @param close_domain Whether an unbounded axis's outer cells are closed at
#'   the grid's own edge rather than left open to infinity.
#' @param folded_axes Optional names of axes folded onto `[0, Inf)` (e.g. a
#'   correlation axis reflected at 0).
#' @param refining Optional refinement-slice tag vector (see
#'   [tulpa_hyper_slice_home()]); when supplied, specs are rebuilt from the
#'   grid's base (non-slice) cells only.
#' @return Numeric vector of per-cell log quadrature weights, length
#'   `nrow(theta_grid)`, or `NULL` if `theta_grid` has no axis names.
#' @seealso [tulpa_theta_matrix()], [tulpa_normalise_weights_safe()],
#'   [tulpa_joint_axis_specs_from_grid()]
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
#' tg <- tulpa_theta_matrix(fit)
#' lq <- tulpa_grid_log_quad(tg)
#' # The fit's own cell weights, rebuilt from its log marginals:
#' tulpa_normalise_weights_safe(fit$log_marginal, log_quad = lq)
#' }
#' @export
tulpa_grid_log_quad <- function(theta_grid, specs = NULL,
                                 copy_slab = "exponential",
                                 close_domain = TRUE, folded_axes = NULL,
                                 refining = NULL) {
  .nl_grid_log_quad(theta_grid, specs = specs, copy_slab = copy_slab,
                    close_domain = close_domain, folded_axes = folded_axes,
                    refining = refining)
}

.nl_grid_log_quad <- function(theta_grid, specs = NULL,
                              copy_slab = "exponential",
                              close_domain = TRUE, folded_axes = NULL,
                              refining = NULL) {
  if (is.null(theta_grid) || is.null(colnames(theta_grid))) return(NULL)
  theta_grid <- as.matrix(theta_grid)
  if (is.null(specs)) {
    base <- !nzchar(.hyper_slice_home(refining, nrow(theta_grid)))
    specs <- .joint_axis_specs_from_grid(theta_grid[base, , drop = FALSE],
                                         copy_slab = copy_slab,
                                         folded_axes = folded_axes)
  }
  .hyper_log_quad_weights(theta_grid, specs, close_domain = close_domain,
                          refining = refining)
}
