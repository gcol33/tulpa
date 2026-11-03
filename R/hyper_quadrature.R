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

# Log of the prior measure one unit of integration-coordinate width carries at
# the natural-scale points `x`, up to the axis's normalising constant: the
# declared continuum density (`slab_log_density`) carried to that coordinate,
# flat where the axis declares none, and `-Inf` outside a declared support
# (`slab_bounds`), where the prior puts no mass. A level's quadrature weight is
# its cell width times this, so the posterior density the outer grid integrates
# is the log marginal plus this term on every axis.
.hyper_axis_log_unit <- function(x, spec) {
  x <- as.numeric(x)
  dens <- spec$slab_log_density
  out <- if (is.null(dens)) rep(0, length(x)) else
    .hyper_prior_carry(x, dens, spec$log_scale,
                       spec$slab_log_density_coord %||% "natural")
  slab <- spec$slab_bounds
  if (!is.null(slab)) out[!(x >= slab[1L] & x <= slab[2L])] <- -Inf
  out
}

# Log posterior density of an outer grid, per unit of a proposal coordinate `u`,
# up to a constant, at the natural-scale points `theta` (one column per axis)
# whose log marginal (with any folded hyperprior) is `lm`. `tags` name each
# axis's proposal coordinate (`.joint_pareto_fwd()`), `u` the points on it, and
# `specs` the axes' measures by column name. The density on the integration
# coordinate `v` is the log marginal plus the measure per unit width
# (`.hyper_axis_log_unit()`), and the change to `u` adds `log |dv / du|`:
# zero on a log-scale axis read in log and on a natural one read as itself, the
# transform's own Jacobian on a natural axis read on logit or atanh, and
# `log |d log theta / du|` on a log-scale axis read on another coordinate, as
# the copy scale is on `identity`. An axis `specs` does not describe is read as
# integrated flat on the coordinate its tag names (log for `log`, natural
# otherwise).
.hyper_outer_log_density_u <- function(lm, theta, u, tags, specs = NULL) {
  out <- as.numeric(lm)
  cn <- colnames(theta)
  if (length(specs)) {
    names(specs) <- vapply(specs, function(s) s$name %||% "", character(1))
  }
  for (j in seq_len(ncol(theta))) {
    th <- as.numeric(theta[, j])
    sp <- if (!is.null(cn) && cn[j] %in% names(specs)) specs[[cn[j]]] else
      list(name = cn[j], log_scale = identical(tags[j], "log"))
    # `.joint_pareto_inv()`'s Jacobian is the one onto the coordinate the tag
    # names (log for `log`), so `d theta / du` adds `u` on a log tag.
    log_dth_du <- .joint_pareto_inv(tags[j], u[, j])$log_jac +
      if (identical(tags[j], "log")) u[, j] else 0
    log_dv_dth <- if (isTRUE(sp$log_scale)) {
      suppressWarnings(ifelse(th > 0, -log(th), -Inf))
    } else 0
    out <- out + .hyper_axis_log_unit(th, sp) + log_dv_dth + log_dth_du
  }
  out[is.na(out)] <- -Inf
  out
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
# extrapolation, widened to the declared span (`.hyper_span_coord_bounds()`) and
# closed inside the axis's declared domain. The ONE construction behind both
# the level weights and the reported support, so the interval the prior is
# normalised over and the interval a sampler is told the quadrature reached are
# the same interval.
.hyper_axis_coord_bounds <- function(u, spec, declared = NULL) {
  .hyper_domain_clamp(.hyper_span_coord_bounds(u, spec, declared), u, spec)
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
                                      close_domain = TRUE, absolute = FALSE,
                                      declared = NULL) {
  .hyper_axis_measure(levels, spec, atom_mass, close_domain = close_domain,
                      absolute = absolute, declared = declared)$w
}

# Half-node-step bounds of a node set on the integration coordinate, widened to
# those of the levels the axis was DECLARED on. A refinement pass adds levels
# inside a declared span as well as past it, and a level laid next to an outer
# node shortens that node's own half-step mirror; the span the axis was declared
# over does not move because a node was added inside it. `declared` holds the
# natural-scale levels of the cells no pass added (NULL, or the same levels, on
# a grid nothing refined).
.hyper_span_coord_bounds <- function(u, spec, declared = NULL) {
  bd <- .hyper_default_coord_bounds(u)
  if (!length(declared)) return(bd)
  d <- sort(unique(as.numeric(declared)))
  d <- d[is.finite(d) & !.hyper_is_atom_level(d, spec)]
  if (isTRUE(spec$log_scale)) d <- d[d > 0]
  if (length(d) < 2L) return(bd)
  dbd <- .hyper_default_coord_bounds(.hyper_axis_coord(d, spec))
  c(min(bd[1L], dbd[1L]), max(bd[2L], dbd[2L]))
}

# The level weights of one axis together with the continuum nodes and the cell
# edges they own on the integration coordinate, and `unit(x)`, the weight a
# unit of coordinate width carries at continuum node `x` under the same density,
# span and normalisation the level weights were built with -- so for every
# continuum level `w == width * unit(level)`. `unit` is NULL where the axis has
# no continuum (a single level, or an atom alone). `declared` is
# `.hyper_span_coord_bounds()`'s.
.hyper_axis_measure <- function(levels, spec, atom_mass = NULL,
                                close_domain = TRUE, absolute = FALSE,
                                declared = NULL) {
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
  bd <- if (is.null(slab)) .hyper_span_coord_bounds(u, spec, declared)
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
  dens <- spec$slab_log_density
  # A proper continuum beside a point mass at zero lives on the whole of
  # (0, Inf), and the atom is its limit there, so the lowest continuum cell
  # reaches down to the atom rather than stopping half a node step below its
  # node. Its width on the log coordinate is infinite; its prior mass is not,
  # and it is integrated exactly (`.hyper_floor_cell_mass()`). Closing it at
  # the half step instead drops the prior mass of (0, edge), which is where the
  # likelihood of an uncoupled field is highest: on an 8 x 8 ICAR fit with the
  # copy scale truly zero the atom read 0.90 of the posterior against 0.83 on a
  # grid laid down to 1e-4.
  floor_cell <- .hyper_continuum_reaches_atom(levels, spec, atom_mass)
  if (floor_cell) bd[1L] <- -Inf
  width <- .hyper_cell_widths(u, bd[1L], bd[2L])

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
    per_width <- function(xx) {
      d <- exp(.hyper_axis_log_unit(xx, spec))
      d[!is.finite(d)] <- 0
      d
    }
    cw <- width * per_width(x)
    if (floor_cell) {
      cw[1L] <- .hyper_floor_cell_mass(
        if (length(u) > 1L) (u[1L] + u[2L]) / 2 else bd[2L], dens, spec)
    }
    cw[!is.finite(cw)] <- 0
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

# Does the continuum of this axis reach down to its point mass? The rule
# `.hyper_axis_measure()` applies: a log-scale axis with a zero level and a
# declared atom mass, whose continuum carries a proper density and no declared
# span. One predicate, read by the measure and by the reported support.
.hyper_continuum_reaches_atom <- function(levels, spec,
                                         atom_mass = spec$atom_mass) {
  isTRUE(spec$log_scale) && !is.null(atom_mass) &&
    any(as.numeric(levels) == 0) &&
    is.null(spec$slab_bounds) && !is.null(spec$slab_log_density)
}

# Prior mass of the lowest continuum cell of an axis whose continuum reaches its
# atom: the declared density integrated from zero to the cell's upper edge
# `edge_u` (on the integration coordinate). Integrated on the coordinate the
# density is declared on -- the natural scale from 0, or the log coordinate from
# -Inf -- where the cell is a finite interval or an integrable tail.
#
# `log_tilt`, a function of the natural-scale value, multiplies the density by
# `exp(log_tilt(x))` inside the integral: the likelihood shape the cell is scored
# with (`.hyper_floor_cell_log_tilt()`).
.hyper_floor_cell_mass <- function(edge_u, dens, spec, log_tilt = NULL) {
  coord <- spec$slab_log_density_coord %||% "natural"
  tilt <- if (is.null(log_tilt)) function(a) 0 else log_tilt
  if (identical(coord, "natural")) {
    f <- function(x) exp(.hyper_prior_carry(x, dens, FALSE) + tilt(x))
    hi <- if (isTRUE(spec$log_scale)) exp(edge_u) else edge_u
    lo <- 0
  } else {
    f <- function(x) exp(.hyper_prior_carry(x, dens, FALSE) + tilt(exp(x)))
    hi <- edge_u
    lo <- -Inf
  }
  m <- tryCatch(stats::integrate(f, lo, hi, rel.tol = 1e-10)$value,
                error = function(e) NA_real_)
  if (!is.finite(m) || m < 0) NA_real_ else m
}

# Per-cell log factor that scores the lowest continuum cell of an axis whose
# continuum reaches its atom (`.hyper_continuum_reaches_atom()`) from the
# likelihood at both of its ends, 0 on every other cell.
#
# That cell runs from the atom at zero to the edge above its node `x1`, and the
# measure gives it the declared prior mass of the whole of (0, edge). Scored at
# `x1` alone, the likelihood there stands for the whole stretch down to zero,
# where the inner marginal is the atom's own: a coupled field whose likelihood
# falls away towards zero has the stretch over-weighted, and how much depends on
# where `x1` happens to sit. Within each row of the other axes the log
# likelihood between the atom and `x1` is the polynomial in the natural-scale
# value through the atom's (`lambda_0`), the node's (`lambda_1`) and, where its
# cell is solved, the next node's (`lambda_2`): the copy scale multiplies the
# field inside the linear predictor, so the log likelihood is locally a
# quadratic in it, as in a regression coefficient, and a chord through the two
# ends alone lies below a concave one. From `x1` to the edge the cell keeps the
# score at its node, the box every other cell is scored by. With `q` that
# polynomial,
#
#   factor = [int_0^x1 p(a) exp(q(a) - lambda_1) da + int_x1^edge p(a) da]
#            / int_0^edge p(a) da,
#
# `p` the declared slab density. A density folded into the marginal
# (`log_prior`) reaches the continuum's cells and not the atom's, so it is taken
# off before the levels are compared. A row with no atom cell, or whose node
# cell carries no finite marginal, keeps the score at `x1` (factor 1).
#
# A cell the cheap-pass screen pruned carries `-Inf` in `log_marginal`: its mass
# is negligible, its likelihood is not. Where `log_marginal` carries the screen's
# own values (attribute `screen`, the fit's `prune_cheap_log_marginal`), such a
# level's ordinate is the screen's value carried onto the solved scale by the
# row's node `x1`, `screen - screen[x1] + log_marginal[x1]`: within a row the two
# differ by the folded hyperprior of the other axes and the screen's Newton
# decrement.
.hyper_floor_cell_log_tilt <- function(theta_grid, a, spec, log_marginal,
                                       atom_mass, close_domain = TRUE,
                                       declared = NULL) {
  v <- as.numeric(theta_grid[, a])
  out <- numeric(length(v))
  m <- .hyper_axis_measure(v, spec, atom_mass, close_domain = close_domain,
                           declared = declared)
  if (!length(m$x)) return(out)
  xs <- c(0, m$x[seq_len(min(2L, length(m$x)))])
  dens <- spec$slab_log_density
  m_edge <- .hyper_floor_cell_mass(m$edges[2L], dens, spec)
  u1 <- .hyper_axis_coord(xs[2L], spec)
  m_x1 <- .hyper_floor_cell_mass(u1, dens, spec)
  if (!is.finite(m_edge) || m_edge <= 0 || !is.finite(m_x1)) return(out)
  fold <- if (is.null(spec$log_prior)) c(0, 0, 0)[seq_along(xs)] else
    c(0, .hyper_prior_carry(xs[-1L], spec$log_prior, spec$log_scale,
                            spec$log_prior_coord %||% "integration"))
  key <- .hyper_row_key(as.matrix(theta_grid),
                        which(colnames(theta_grid) != a))
  scr <- attr(log_marginal, "screen")
  lm <- as.numeric(log_marginal)
  if (length(scr) != length(lm)) scr <- NULL
  probe <- seq(0, xs[2L], length.out = 201L)
  for (k in which(v == xs[2L])) {
    at <- vapply(xs, function(x) {
      i <- which(key == key[k] & v == x)
      if (length(i) == 1L) i else NA_integer_
    }, integer(1))
    if (is.na(at[1L])) next
    lam <- lm[at]
    if (!is.null(scr) && is.finite(lm[k]) && is.finite(scr[k])) {
      cut <- !is.na(at) & lam %in% -Inf & is.finite(scr[at])
      lam[cut] <- scr[at][cut] - scr[k] + lm[k]
    }
    lam <- lam - fold
    if (is.na(lam[1L]) || lam[1L] == Inf || !is.finite(lam[2L])) next
    use <- c(TRUE, TRUE, if (length(xs) > 2L) is.finite(lam[3L]))
    q <- .hyper_lagrange(xs[use], lam[use] - lam[2L])
    off <- max(q(probe), 0)
    low <- .hyper_floor_cell_mass(u1, dens, spec,
                                  log_tilt = function(x) q(x) - off)
    if (!is.finite(low)) next
    out[k] <- .tulpa_logsumexp(c(log(low) + off,
                                 log(max(m_edge - m_x1, 0)))) - log(m_edge)
  }
  out
}

# The interpolating polynomial through the points (`x`, `y`), as a function of
# a numeric vector. A `-Inf` ordinate (a level whose cell carries no likelihood)
# is reachable only at `x[1]`, and makes the function `-Inf` below `x[2]`.
.hyper_lagrange <- function(x, y) {
  force(x); force(y)
  if (!is.finite(y[1L])) {
    rest <- .hyper_lagrange(x[-1L], y[-1L])
    return(function(z) ifelse(z < x[2L], -Inf, rest(z)))
  }
  function(z) {
    out <- 0
    for (i in seq_along(x)) {
      li <- 1
      for (j in seq_along(x)[-i]) li <- li * (z - x[j]) / (x[i] - x[j])
      out <- out + y[i] * li
    }
    out
  }
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
#' The copy-scale axis's slab measure
#'
#' @description
#' `tulpa_hyper_copy_slab_density()` returns the proper density tulpa uses for
#' the copy scale's continuum: an exponential, the penalized-complexity prior
#' for a scale parameter with its base model at zero (Simpson et al. 2017).
#' The rate is set by putting 5% of the prior mass above `upper`, so a caller
#' that reads this off a fit's own declared grid gets the exact rate the outer
#' integration used -- rather than restating a PC-prior rate that could
#' silently drift from it.
#'
#' `tulpa_hyper_check_copy_slab()` validates a `copy_slab` argument --
#' `"exponential"` (the default) or `"flat"`, the two continuum measures tulpa
#' supports for a copy scale's non-atom mass -- defaulting `NULL` to
#' `"exponential"` and erroring on anything else. Intended so a consumer
#' package's own `copy_slab` argument stays in sync with tulpa's own accepted
#' choices rather than restating them.
#'
#' @param upper The largest declared node of the copy-scale axis.
#' @param x A `copy_slab` value: `NULL`, `"exponential"`, or `"flat"`.
#' @return `tulpa_hyper_copy_slab_density()`: a function
#'   `log p(x) = log(lambda) - lambda * x`, or `NULL` if `upper` is not a
#'   finite positive number. `tulpa_hyper_check_copy_slab()`: `x`, defaulted
#'   to `"exponential"` when `NULL`.
#' @seealso [tulpa_joint_axis_specs_from_grid()]
#' @examples
#' lp <- tulpa_hyper_copy_slab_density(upper = 2)
#' lp(c(0.1, 1, 2))
#' tulpa_hyper_check_copy_slab("exponential")
#' try(tulpa_hyper_check_copy_slab("uniform"))
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
# a node step beyond the outermost levels otherwise, and from zero where the
# continuum reaches its atom (`.hyper_continuum_reaches_atom()`). This is the
# region the outer quadrature actually reaches, so a sampler asked to target the
# same measure is bounded here and not at the outermost node.
#
# Either rule is closed inside the axis's declared domain, so a bounded axis --
# a correlation, a mixing weight, a probability -- never reports a support
# containing a value its parameter cannot take. The invariant is the one
# `.nl_cell_edges()` states on the reporting side: whenever every level lies
# inside a declared domain, both ends of the returned interval do too.
#
# Returns NULL for an axis with fewer than two continuum levels, which is the
# pinned case the caller leaves out of the sampled vector. `declared` is
# `.hyper_span_coord_bounds()`'s.
.hyper_axis_support <- function(levels, spec, declared = NULL) {
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
  bd <- .hyper_axis_coord_bounds(u, spec, declared)
  # A continuum reaching its atom integrates down to zero, so that is where its
  # support starts.
  if (.hyper_continuum_reaches_atom(levels, spec)) bd[1L] <- -Inf
  if (isTRUE(spec$log_scale)) exp(bd) else bd
}

# The support of every axis in `specs` that carries one, as a named list of
# natural-scale intervals. Stored on the fit so a second engine reads the span
# this fit integrated rather than rebuilding it from a grid that refinement may
# since have extended.
#
# `refining` is the per-cell tag `.hyper_log_quad_weights()` takes, and the
# support is the one the level weights are built over: the levels' own, widened
# to the span the axis was declared over.
#' Per-axis integrated support of an outer grid
#'
#' The natural-scale support of every axis in `specs` that carries one, as a
#' named list of intervals -- the region each axis's outer-grid measure
#' actually integrates, including levels a refinement pass added. Intended for
#' recovering an axis's integrated span from a settled grid, e.g. so a
#' sampled-hyperparameter prior can be derived from what the outer
#' integration used rather than restated alongside it.
#'
#' @param theta_grid A named `[n_cells x n_axes]` matrix (see
#'   [tulpa_theta_matrix()]).
#' @param specs Per-axis spec list, e.g. from
#'   [tulpa_joint_axis_specs_from_grid()].
#' @param refining Optional per-cell refinement tag vector (see
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
  declared <- !nzchar(.hyper_slice_home(refining, nrow(theta_grid)))
  out <- list()
  for (spec in specs) {
    a <- spec$name
    if (!a %in% axis_names) next
    sup <- .hyper_axis_support(theta_grid[, a], spec,
                               declared = theta_grid[declared, a])
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
# `refining` is the per-cell tag the refinement passes leave: `""` for a cell
# of the grid as declared, the axis name for a cell a pass added. A pass adds
# LEVELS, each laid in every row of the other axes that holds the posterior, so
# the product rule measures the grid and a row left out lacks only cells of
# negligible mass; the tags say which levels were declared,
# and the outer cells keep the span those levels were declared over
# (`.hyper_span_coord_bounds()`).
#
# `log_marginal`, where the cells have been solved, scores the lowest continuum
# cell of a copy scale reaching its atom from both of its ends
# (`.hyper_floor_cell_log_tilt()`). Without it -- a ranking taken before any cell
# is solved -- that cell carries its prior mass alone.
.hyper_log_quad_weights <- function(theta_grid, specs, close_domain = TRUE,
                                    absolute = FALSE, refining = NULL,
                                    log_marginal = NULL) {
  if (is.null(theta_grid) || is.null(specs)) return(NULL)
  theta_grid <- as.matrix(theta_grid)
  n <- nrow(theta_grid)
  if (n == 0L) return(numeric(0))
  declared <- !nzchar(.hyper_slice_home(refining, n))
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
                                    absolute = absolute,
                                    declared = v[declared])
    levels <- sort(unique(v))
    idx <- match(v, levels)
    contrib <- log(as.numeric(lw)[idx])
    contrib[!is.finite(contrib)] <- -Inf
    if (!is.null(log_marginal) && length(log_marginal) == n &&
        .hyper_continuum_reaches_atom(levels, spec,
                                      .hyper_axis_atom_mass(spec))) {
      contrib <- contrib + .hyper_floor_cell_log_tilt(
        theta_grid, a, spec, log_marginal, .hyper_axis_atom_mass(spec),
        close_domain = close_domain, declared = v[declared])
    }
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

# The axis whose refinement added each cell, `""` for a cell of the grid as
# declared. A `refining` vector of the wrong length describes some other grid
# and is an error rather than a guess.
#' Per-cell refinement tag of an outer grid
#'
#' The axis whose refinement pass added each cell of a nested-Laplace outer
#' grid, `""` for a cell of the grid as declared. A pass adds levels to an
#' axis, laid in every row of the others that holds the posterior; the tag
#' tells the declared levels apart wherever a reader needs them alone, e.g.
#' rebuilding axis specs from a grid's declared nodes (see
#' [tulpa_joint_axis_specs_from_grid()]), whose prior must not move with the
#' nodes a pass added.
#'
#' @param refining The `refining` tag vector stored on a fit (`NULL` for a
#'   grid with no refinement).
#' @param n Number of grid cells; `refining`, if not `NULL`, must have this
#'   length.
#' @return A character vector of length `n`: `""` for a declared cell, the
#'   axis name for a cell a refinement pass added.
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
  out <- as.character(refining)
  out[is.na(out)] <- ""
  out
}

# Accepted shapes for the copy scale's continuum measure. "exponential" is the
# penalized-complexity density above; "flat" makes the axis flat in log alpha
# over the declared span, the measure the other log-scale axes carry.
.TULPA_COPY_SLAB_CHOICES <- c("exponential", "flat")

#' @rdname tulpa_hyper_copy_slab_density
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
# Specs rebuilt from the grid are rebuilt from its DECLARED cells when
# `refining` marks cells a pass added: an axis's declared node set is what fixes
# a prior read off it (the copy scale's exponential rate is set by its largest
# node), and a node a refinement pass added was not declared.
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
#' @param refining Optional refinement tag vector (see
#'   [tulpa_hyper_slice_home()]); when supplied, specs are rebuilt from the
#'   grid's declared cells only.
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
