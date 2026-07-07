#' Construct a spatial adjacency graph for areal models
#'
#' @description
#' Build the symmetric adjacency matrix that [spatial()] and [spatial_car()]
#' consume, from a common spatial layout, instead of hand-coding it. The model
#' still receives an explicit graph (`spatial(graph = g$adjacency, ...)`), so the
#' graph stays inspectable before fitting -- the engine never guesses
#' connectivity from coordinates silently.
#'
#' `adjacency()` is a single front-door verb that dispatches on the kind of
#' input:
#' \describe{
#'   \item{a `data.frame`}{cell centroids on a regular grid -- queen (edge or
#'     corner) or rook (edge only) contiguity over the lattice. This covers both
#'     plain coordinate grids and rasterised cells passed as a table of
#'     centres.}
#'   \item{an `sf` object}{polygon contiguity from shared boundaries (queen =
#'     share a point or edge; rook = share an edge). Requires \pkg{sf}.}
#'   \item{a `SpatRaster` (\pkg{terra}) or `stars` object}{raster cells -- the
#'     lattice of (non-`NA`) cell centres, queen or rook. Requires \pkg{terra}
#'     or \pkg{stars} respectively.}
#' }
#'
#' The result is a `tulpa_adjacency` object holding the matrix plus the cell
#' identifier for each node, so a model can pass `g$adjacency` while the
#' observation data is remapped to 1-based node indices with [node_index()].
#'
#' @param x The spatial layout: a `data.frame` of centroids, an `sf` polygon
#'   layer, or a raster (`SpatRaster` / `stars`).
#' @param type Contiguity rule: `"queen"` (default; neighbours share an edge or
#'   a corner) or `"rook"` (neighbours share an edge only). For polygons,
#'   `"touches"` is an alias of `"queen"`.
#' @param x_coord,y_coord For the `data.frame` method, the names of the
#'   coordinate columns holding the cell centres (default `"x"` and `"y"`).
#' @param id For the `data.frame` and `sf` methods, an optional column naming
#'   each cell's identifier. Node `i` of the graph corresponds to `id`'s value
#'   in row `i`; pass this column to [node_index()] to translate the observation
#'   data's cell column into node indices. Default `NULL` uses the row position
#'   (`1:n`) as the identifier.
#' @param order For grid / raster layouts, the neighbourhood order (ring count):
#'   `order = 1` (default) is first-order contiguity (queen = 8 neighbours, rook
#'   = 4); `order = k` extends the stencil to the k-th ring, so queen keeps every
#'   cell within Chebyshev distance `k` (`(2k + 1)^2 - 1` neighbours: 8, 24, 48,
#'   ... for `k = 1, 2, 3`) and rook every cell within Manhattan distance `k`
#'   (`2k(k + 1)` neighbours: 4, 12, 24, ...). Any positive integer is allowed,
#'   so the neighbourhood is fully settable. Ignored by the `sf` (polygon)
#'   method, which is first-order contiguity only.
#' @param tolerance For grid / raster layouts, the per-offset neighbour distance
#'   cut-off as a multiple of the inferred cell size: a candidate at a lattice
#'   offset is kept when its true centre distance is at most `tolerance` times
#'   that offset's expected distance. The default `1.5` admits a snapped
#'   neighbour while rejecting one much farther than its lattice slot implies;
#'   raise it only for irregularly spaced centroids. Neighbourhood extent is set
#'   by `order`, not by `tolerance`.
#' @param offsets Advanced, grid / raster layouts only: a custom neighbour
#'   stencil as a two-column integer matrix or a list of length-2 `c(dx, dy)`
#'   lattice offsets (cell-step units, origin excluded). Supplying it overrides
#'   `type` and `order` and builds exactly that stencil, so any neighbourhood is
#'   expressible (anisotropic, ring-only, off-axis). An ICAR / CAR field is
#'   *undirected*, so the graph must be symmetric: an asymmetric stencil (e.g.
#'   `list(c(0, 1), c(-1, 0), c(1, 0))` = up / left / right but not down) is
#'   symmetrized to an undirected graph, with a message, since a directed
#'   neighbourhood cannot be represented by an undirected field. Default `NULL`
#'   uses the `type` / `order` stencil.
#' @param na_rm For raster layouts, drop cells whose value is `NA` before
#'   building the graph (default `TRUE`), so the nodes are the cells that carry
#'   data.
#' @param ... Passed to methods.
#'
#' @return A `tulpa_adjacency` object: a list with
#'   \describe{
#'     \item{`adjacency`}{the `[n x n]` symmetric sparse adjacency matrix
#'       (`dgCMatrix`, 0/1, zero diagonal) to pass as `graph` / `adjacency`.}
#'     \item{`ids`}{the cell identifier for each node, in node order (length
#'       `n`).}
#'     \item{`n`}{the number of nodes.}
#'     \item{`cellsize`}{the inferred cell size `c(x, y)` for grid / raster
#'       layouts, or `NA` for polygons.}
#'     \item{`type`}{the contiguity rule used.}
#'   }
#'
#' @seealso [node_index()] to map cell identifiers to node indices,
#'   [check_adjacency()] to validate a hand-built matrix, [spatial()] and
#'   [spatial_car()] which consume the graph.
#'
#' @examples
#' # A 3 x 3 regular grid of cell centres
#' grid <- expand.grid(x = 1:3, y = 1:3)
#' grid$cell <- paste0("c", seq_len(nrow(grid)))
#'
#' g <- adjacency(grid, x_coord = "x", y_coord = "y", id = "cell",
#'                type = "queen")
#' g
#' g$adjacency
#'
#' # Second-order (24-neighbour) queen contiguity: any order is settable
#' g2 <- adjacency(grid, id = "cell", order = 2)
#'
#' # Advanced: a custom stencil (symmetrized for the undirected field)
#' g3 <- adjacency(grid, id = "cell", offsets = list(c(1, 0), c(0, 1)))
#'
#' # Use it in a model: graph stays explicit and inspectable
#' # spatial(graph = g$adjacency, formula = ~ 1 || cell_idx)
#'
#' # Remap observation data (original cell ids -> 1:n node indices) by key
#' obs <- data.frame(cell = c("c5", "c1", "c5", "c9"))
#' obs$cell_idx <- node_index(g, obs$cell)
#' obs
#'
#' @name adjacency
#' @export
adjacency <- function(x, ...) {
  UseMethod("adjacency")
}

#' @rdname adjacency
#' @export
adjacency.default <- function(x, ...) {
  stop("adjacency() has no method for an object of class ",
       paste(dQuote(class(x)), collapse = "/"), ". Supply a data.frame of ",
       "centroids, an 'sf' polygon layer, or a raster ('SpatRaster' / ",
       "'stars').", call. = FALSE)
}

#' @rdname adjacency
#' @export
adjacency.data.frame <- function(x, type = c("queen", "rook"),
                                  x_coord = "x", y_coord = "y", id = NULL,
                                  order = 1L, tolerance = 1.5, offsets = NULL,
                                  ...) {
  type <- match.arg(type)
  assert_columns_exist(c(x_coord, y_coord), x, role = "Coordinate")
  coords <- as.matrix(x[, c(x_coord, y_coord), drop = FALSE])
  storage.mode(coords) <- "double"
  if (anyNA(coords)) {
    stop("Coordinate columns contain missing values.", call. = FALSE)
  }
  ids <- .adj_resolve_ids(id, x, n = nrow(coords))

  if (anyDuplicated(paste(coords[, 1L], coords[, 2L], sep = "\r"))) {
    stop("Duplicated cell centroids: each row must be a distinct grid cell.",
         call. = FALSE)
  }

  built <- .lattice_adjacency(coords, type = type, order = order,
                              tolerance = tolerance, offsets = offsets)
  .new_tulpa_adjacency(built$W, ids = ids, cellsize = built$cellsize,
                       type  = if (is.null(offsets)) type else "custom",
                       coords = coords,
                       order = if (is.null(offsets)) order else NA_integer_)
}

#' @rdname adjacency
#' @export
adjacency.sf <- function(x, type = c("queen", "rook", "touches"),
                         id = NULL, ...) {
  if (!requireNamespace("sf", quietly = TRUE)) {
    stop("Polygon adjacency needs the 'sf' package. Install it with ",
         "install.packages(\"sf\").", call. = FALSE)
  }
  type <- match.arg(type)
  geom_type <- if (type == "rook") "rook" else "queen"

  ids <- .adj_resolve_ids(id, x, n = nrow(x))

  # DE-9IM contiguity: queen = boundaries meet in a point or a line with
  # disjoint interiors ("F***T****"); rook = boundaries meet in a line
  # ("F***1****"). The interior-disjoint constraint excludes self-matches.
  pattern <- if (geom_type == "rook") "F***1****" else "F***T****"
  nb <- sf::st_relate(x, x, pattern = pattern, sparse = TRUE)

  n <- length(nb)
  ii <- rep.int(seq_len(n), lengths(nb))
  jj <- unlist(nb, use.names = FALSE)
  W <- .adj_from_edges(ii, jj, n)
  .new_tulpa_adjacency(W, ids = ids, cellsize = c(x = NA_real_, y = NA_real_),
                       type = geom_type, coords = NULL)
}

#' @rdname adjacency
#' @export
adjacency.SpatRaster <- function(x, type = c("queen", "rook"),
                                 order = 1L, tolerance = 1.5, offsets = NULL,
                                 na_rm = TRUE, ...) {
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("Raster adjacency for a 'SpatRaster' needs the 'terra' package.",
         call. = FALSE)
  }
  type <- match.arg(type)
  cells <- seq_len(terra::ncell(x))
  if (isTRUE(na_rm)) {
    vals <- terra::values(x[[1L]], mat = FALSE)
    cells <- cells[!is.na(vals)]
  }
  if (length(cells) == 0L) {
    stop("Raster has no non-NA cells to build a graph over.", call. = FALSE)
  }
  xy <- terra::xyFromCell(x, cells)
  built <- .lattice_adjacency(xy, type = type, order = order,
                              tolerance = tolerance, offsets = offsets)
  .new_tulpa_adjacency(built$W, ids = cells, cellsize = built$cellsize,
                       type  = if (is.null(offsets)) type else "custom",
                       coords = xy,
                       order = if (is.null(offsets)) order else NA_integer_)
}

#' @rdname adjacency
#' @export
adjacency.stars <- function(x, type = c("queen", "rook"),
                            order = 1L, tolerance = 1.5, offsets = NULL,
                            na_rm = TRUE, ...) {
  if (!requireNamespace("stars", quietly = TRUE) ||
      !requireNamespace("sf", quietly = TRUE)) {
    stop("Raster adjacency for a 'stars' object needs the 'stars' and 'sf' ",
         "packages.", call. = FALSE)
  }
  type <- match.arg(type)
  # stars registers st_as_sf() as an S3 method on the sf generic.
  centres <- sf::st_coordinates(sf::st_as_sf(x, as_points = TRUE,
                                             na.rm = isTRUE(na_rm)))
  xy <- as.matrix(centres[, c("X", "Y"), drop = FALSE])
  if (nrow(xy) == 0L) {
    stop("Raster has no non-NA cells to build a graph over.", call. = FALSE)
  }
  built <- .lattice_adjacency(xy, type = type, order = order,
                              tolerance = tolerance, offsets = offsets)
  .new_tulpa_adjacency(built$W, ids = seq_len(nrow(xy)),
                       cellsize = built$cellsize,
                       type  = if (is.null(offsets)) type else "custom",
                       coords = xy,
                       order = if (is.null(offsets)) order else NA_integer_)
}


# ---- shared lattice core -------------------------------------------------

# Lattice-offset stencil for k-th order contiguity on a regular grid. queen keeps
# every cell within Chebyshev distance `order` ((2k+1)^2 - 1 neighbours: 8, 24,
# 48, ...); rook keeps every cell within Manhattan distance `order` (2k(k+1)
# neighbours: 4, 12, 24, ...). order = 1 recovers the classic 8 / 4 stencils.
# Returns a list of length-2 integer offset vectors, origin excluded.
.stencil_offsets <- function(type, order) {
  order <- as.integer(order)
  if (length(order) != 1L || is.na(order) || order < 1L) {
    stop("`order` must be a single positive integer (the neighbourhood ring ",
         "count).", call. = FALSE)
  }
  grid <- expand.grid(dx = -order:order, dy = -order:order,
                      KEEP.OUT.ATTRS = FALSE)
  dx <- grid$dx; dy <- grid$dy
  dist <- if (type == "rook") abs(dx) + abs(dy) else pmax(abs(dx), abs(dy))
  keep <- dist >= 1L & dist <= order
  Map(function(a, b) c(as.integer(a), as.integer(b)), dx[keep], dy[keep])
}

# Validate a user-supplied custom stencil (advanced `offsets` argument): a
# two-column integer matrix or a list of length-2 c(dx, dy) lattice offsets.
# Returns a list of integer length-2 vectors, origin dropped.
.check_offsets <- function(offsets) {
  if (is.matrix(offsets)) {
    if (ncol(offsets) != 2L) {
      stop("`offsets` matrix must have two columns (dx, dy).", call. = FALSE)
    }
    offsets <- lapply(seq_len(nrow(offsets)), function(i) offsets[i, ])
  } else if (!is.list(offsets)) {
    stop("`offsets` must be a two-column integer matrix or a list of length-2 ",
         "c(dx, dy) vectors.", call. = FALSE)
  }
  offsets <- lapply(offsets, function(o) {
    if (length(o) != 2L) {
      stop("each `offsets` entry must be length 2 (dx, dy).", call. = FALSE)
    }
    oi <- as.integer(round(as.numeric(o)))
    if (anyNA(oi)) stop("`offsets` has non-numeric / NA entries.", call. = FALSE)
    oi
  })
  offsets <- offsets[vapply(offsets, function(o) !all(o == 0L), logical(1))]
  if (!length(offsets)) {
    stop("`offsets` has no non-origin entries.", call. = FALSE)
  }
  offsets
}

# Build queen / rook contiguity over centroids on a (near-)regular grid.
# Snaps each centroid to an integer lattice index from the inferred per-axis
# step, hashes (ix, iy) -> node for O(n) neighbour lookup, then for each node
# tests the stencil `type` and `order` imply (see .stencil_offsets) and keeps an
# edge when the true centre distance is within `tolerance` of that offset's
# expected distance. Returns the symmetric 0/1 sparse W and the inferred cell
# size. Single source for the data.frame and raster methods.
.lattice_adjacency <- function(coords, type, order = 1L, tolerance = 1.5,
                               offsets = NULL) {
  if (!is.finite(tolerance) || tolerance <= 0) {
    stop("`tolerance` must be a positive finite scalar.", call. = FALSE)
  }
  X <- coords[, 1L]
  Y <- coords[, 2L]
  n <- length(X)

  cs_x <- .infer_step(X)
  cs_y <- .infer_step(Y)
  cs <- min(c(cs_x, cs_y), na.rm = TRUE)
  if (!is.finite(cs) || cs <= 0) {
    stop("Could not infer a cell size from the coordinates (need at least two ",
         "distinct grid lines on one axis).", call. = FALSE)
  }
  # Single row or single column: that axis has no step; snap it to a constant.
  if (!is.finite(cs_x)) cs_x <- cs
  if (!is.finite(cs_y)) cs_y <- cs

  ix <- round((X - min(X)) / cs_x)
  iy <- round((Y - min(Y)) / cs_y)
  key <- paste(ix, iy, sep = ",")
  node_of <- stats::setNames(seq_len(n), key)

  custom  <- !is.null(offsets)
  offsets <- if (custom) .check_offsets(offsets) else .stencil_offsets(type, order)

  from <- integer(0)
  to <- integer(0)
  for (off in offsets) {
    # Expected centre distance for this lattice step (anisotropy-safe: a rook
    # neighbour along the wider axis is `cs_x` away, the narrower one `cs_y`).
    # `tolerance` is the allowed multiple of it, so a snapped neighbour that is
    # actually much farther than its lattice slot implies is rejected.
    expected <- sqrt((off[1L] * cs_x)^2 + (off[2L] * cs_y)^2)
    band <- tolerance * expected
    nbr_key <- paste(ix + off[1L], iy + off[2L], sep = ",")
    j <- node_of[nbr_key]
    present <- !is.na(j)
    if (!any(present)) next
    i_idx <- which(present)
    j_idx <- as.integer(j[present])
    d <- sqrt((X[i_idx] - X[j_idx])^2 + (Y[i_idx] - Y[j_idx])^2)
    keep <- d <= band
    if (!any(keep)) next
    from <- c(from, i_idx[keep])
    to <- c(to, j_idx[keep])
  }

  W <- .adj_from_edges(from, to, n)
  # An ICAR / CAR field is undirected, so the graph must be symmetric. The built-
  # in stencils come in +/- pairs and are already symmetric; a custom stencil may
  # not be, so symmetrize it (union with the transpose) and report any reverse
  # edges added -- a directed neighbourhood cannot be represented as an
  # undirected field.
  if (custom) {
    n_before <- Matrix::nnzero(W)
    W <- W + Matrix::t(W)
    if (length(W@x)) W@x[] <- 1
    n_added <- Matrix::nnzero(W) - n_before
    if (n_added > 0L) {
      message(sprintf(paste0(
        "adjacency(): custom stencil was not symmetric; symmetrized to an ",
        "undirected graph for the ICAR/CAR field (%d reverse edge(s) added). A ",
        "directed neighbourhood cannot be represented by an undirected field."),
        n_added))
    }
  }
  list(W = W, cellsize = c(x = cs_x, y = cs_y))
}

# Infer the lattice step on one axis: the smallest positive gap between
# distinct coordinate values, ignoring floating-point duplicates. Returns NA
# when the axis has a single distinct value.
.infer_step <- function(v) {
  u <- sort(unique(v))
  if (length(u) < 2L) return(NA_real_)
  g <- diff(u)
  rng <- u[length(u)] - u[1L]
  eps <- if (rng > 0) 1e-8 * rng else 0
  g <- g[g > eps]
  if (length(g) == 0L) return(NA_real_)
  min(g)
}

# Assemble a symmetric 0/1 dgCMatrix from directed edge lists. Both directions
# are supplied by the callers (offset stencils come in +/- pairs; st_relate is
# symmetric), so duplicates are collapsed and the result is binarised.
.adj_from_edges <- function(from, to, n) {
  if (length(from) == 0L) {
    return(methods::as(Matrix::sparseMatrix(i = integer(0), j = integer(0),
                                             dims = c(n, n)), "CsparseMatrix"))
  }
  W <- Matrix::sparseMatrix(i = from, j = to, x = 1, dims = c(n, n),
                            use.last.ij = TRUE)
  W <- methods::as(W, "CsparseMatrix")
  W <- methods::as(W, "generalMatrix")
  if (length(W@x)) W@x[] <- 1
  W
}

.adj_resolve_ids <- function(id, data, n) {
  if (is.null(id)) return(seq_len(n))
  if (length(id) != 1L || !is.character(id)) {
    stop("`id` must be a single column name.", call. = FALSE)
  }
  assert_columns_exist(id, data, role = "Identifier")
  ids <- data[[id]]
  if (anyNA(ids)) {
    stop("`id` column contains missing values.", call. = FALSE)
  }
  if (anyDuplicated(ids)) {
    stop("`id` column has duplicate values: each row must be a distinct cell.",
         call. = FALSE)
  }
  ids
}

.new_tulpa_adjacency <- function(W, ids, cellsize, type, coords, order = 1L) {
  report <- .validate_adjacency(W, ids = ids)
  if (!report$square || !report$symmetric || !report$zero_diag) {
    stop("Internal error: constructed adjacency is not a valid graph.",
         call. = FALSE)
  }
  if (report$n_isolated > 0L) {
    message(sprintf(
      "adjacency(): %d isolated node(s) with no neighbours (e.g. %s). An ICAR/CAR field is improper on disconnected nodes.",
      report$n_isolated,
      paste(utils::head(ids[report$isolated], 3L), collapse = ", ")))
  }
  structure(
    list(adjacency = W,
         ids       = ids,
         n         = nrow(W),
         cellsize  = cellsize,
         type      = type,
         order     = as.integer(order),
         coords    = coords),
    class = c("tulpa_adjacency", "list"))
}


# ---- accessors / validation ----------------------------------------------

#' Map cell identifiers to graph node indices
#'
#' @description
#' Translate a vector of original cell identifiers into the 1-based node indices
#' of a `tulpa_adjacency` graph, by key (not by row order). Use it to add the
#' node-index column the model's spatial grouping bar needs to the observation
#' data, which typically has many rows per cell and a different row order than
#' the graph.
#'
#' @param graph A `tulpa_adjacency` object from [adjacency()].
#' @param ids A vector of cell identifiers to look up (matched against
#'   `graph$ids`).
#'
#' @return An integer vector the same length as `ids`, giving each one's node
#'   index in `graph` (`NA` for identifiers absent from the graph).
#'
#' @seealso [adjacency()].
#'
#' @examples
#' grid <- expand.grid(x = 1:3, y = 1:3)
#' grid$cell <- paste0("c", seq_len(nrow(grid)))
#' g <- adjacency(grid, id = "cell")
#'
#' obs <- data.frame(cell = c("c5", "c1", "c9"))
#' obs$cell_idx <- node_index(g, obs$cell)
#' obs
#'
#' @export
node_index <- function(graph, ids) {
  if (!inherits(graph, "tulpa_adjacency")) {
    stop("`graph` must be a tulpa_adjacency object from adjacency().",
         call. = FALSE)
  }
  idx <- match(ids, graph$ids)
  if (anyNA(idx)) {
    n_missing <- sum(is.na(idx))
    warning(sprintf(
      "node_index(): %d identifier(s) not found in the graph; returned NA.",
      n_missing), call. = FALSE)
  }
  idx
}

#' Validate a spatial adjacency matrix
#'
#' @description
#' Check that a hand-built adjacency matrix is a well-formed spatial graph
#' before passing it to [spatial()] / [spatial_car()]: square, symmetric, zero
#' on the diagonal, 0/1 valued, and free of isolated nodes. [adjacency()] runs
#' the same checks on the graphs it constructs.
#'
#' @param adjacency A matrix (dense or sparse `Matrix`).
#' @param ids Optional cell identifiers; if supplied, their length must match
#'   `nrow(adjacency)` and they must be unique.
#'
#' @return Invisibly, a `tulpa_adjacency_check` list with the per-check results
#'   (`square`, `symmetric`, `zero_diag`, `binary`, isolated-node indices, edge
#'   count) and an overall `ok` flag. Issues are reported via `warning()` and
#'   printed; the function does not stop, so every problem surfaces in one pass.
#'
#' @seealso [adjacency()] to construct a graph, [node_index()].
#'
#' @examples
#' adj <- matrix(0, 4, 4)
#' for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1
#' check_adjacency(adj)
#'
#' @export
check_adjacency <- function(adjacency, ids = NULL) {
  if (!is.matrix(adjacency) && !inherits(adjacency, "Matrix")) {
    stop("`adjacency` must be a matrix (dense or sparse).", call. = FALSE)
  }
  report <- .validate_adjacency(adjacency, ids = ids)
  if (!report$square) {
    warning("Adjacency matrix is not square (", report$nrow, " x ",
            report$ncol, ").", call. = FALSE)
  } else {
    if (!report$symmetric) {
      warning("Adjacency matrix is not symmetric (max |W - t(W)| = ",
              signif(report$asymmetry, 3), ").", call. = FALSE)
    }
    if (!report$zero_diag) {
      warning("Adjacency matrix has non-zero diagonal entries (self-loops): ",
              report$n_self, " node(s).", call. = FALSE)
    }
    if (!report$binary) {
      warning("Adjacency matrix has entries other than 0/1 (weighted graph).",
              call. = FALSE)
    }
    if (report$n_isolated > 0L) {
      warning(report$n_isolated, " isolated node(s) with no neighbours; an ",
              "ICAR/CAR field is improper on disconnected nodes.",
              call. = FALSE)
    }
  }
  if (!is.null(report$id_error)) warning(report$id_error, call. = FALSE)
  print(report)
  invisible(report)
}

# Shared validator: pure (no warnings / printing), returns a report list used by
# both check_adjacency() and the adjacency() constructors.
.validate_adjacency <- function(W, ids = NULL) {
  nr <- nrow(W)
  nc <- ncol(W)
  square <- nr == nc
  out <- list(square = square, nrow = nr, ncol = nc)

  if (!square) {
    out$symmetric <- FALSE
    out$zero_diag <- FALSE
    out$binary <- FALSE
    out$n_self <- NA_integer_
    out$asymmetry <- NA_real_
    out$n_isolated <- NA_integer_
    out$isolated <- integer(0)
    out$n_edges <- NA_integer_
    out$id_error <- .adj_check_ids(ids, nr)
    out$ok <- FALSE
    class(out) <- c("tulpa_adjacency_check", "list")
    return(out)
  }

  Wm <- methods::as(methods::as(W, "CsparseMatrix"), "generalMatrix")
  asym <- max(abs(Wm - Matrix::t(Wm)))
  out$asymmetry <- asym
  out$symmetric <- asym <= 1e-8 * max(1, max(abs(Wm)))

  diag_vals <- Matrix::diag(Wm)
  out$n_self <- sum(diag_vals != 0)
  out$zero_diag <- out$n_self == 0L

  nz <- Wm@x
  out$binary <- length(nz) == 0L || all(nz %in% c(0, 1))

  deg <- Matrix::rowSums(Wm != 0)
  out$isolated <- which(deg == 0)
  out$n_isolated <- length(out$isolated)
  out$n_edges <- sum(Matrix::tril(Wm) != 0)

  out$id_error <- .adj_check_ids(ids, nr)
  out$ok <- out$symmetric && out$zero_diag && out$binary &&
    out$n_isolated == 0L && is.null(out$id_error)
  class(out) <- c("tulpa_adjacency_check", "list")
  out
}

.adj_check_ids <- function(ids, n) {
  if (is.null(ids)) return(NULL)
  if (length(ids) != n) {
    return(sprintf("`ids` length (%d) does not match the number of nodes (%d).",
                   length(ids), n))
  }
  if (anyDuplicated(ids)) return("`ids` has duplicate values.")
  NULL
}


# ---- print methods -------------------------------------------------------

#' @export
print.tulpa_adjacency <- function(x, ...) {
  cat("<tulpa_adjacency>\n")
  ord <- x$order %||% 1L
  contig <- if (isTRUE(ord > 1L)) sprintf("order-%d %s", ord, x$type) else
    sprintf("%s", x$type)
  cat(sprintf("  nodes: %d  edges: %d  (%s contiguity)\n",
              x$n, sum(Matrix::tril(x$adjacency) != 0), contig))
  deg <- Matrix::rowSums(x$adjacency != 0)
  cat(sprintf("  neighbours per node: min %d, mean %.2f, max %d\n",
              min(deg), mean(deg), max(deg)))
  if (all(is.finite(x$cellsize))) {
    cat(sprintf("  cell size: x = %s, y = %s\n",
                signif(x$cellsize[["x"]], 4), signif(x$cellsize[["y"]], 4)))
  }
  n_iso <- sum(deg == 0)
  if (n_iso > 0L) cat(sprintf("  isolated nodes: %d\n", n_iso))
  cat("  pass $adjacency to spatial(graph = ) and node_index() to remap data\n")
  invisible(x)
}

#' @export
print.tulpa_adjacency_check <- function(x, ...) {
  cat("<adjacency check>\n")
  ok_mark <- function(v) if (isTRUE(v)) "ok" else "FAIL"
  cat(sprintf("  square:    %s (%d x %d)\n", ok_mark(x$square), x$nrow, x$ncol))
  if (x$square) {
    cat(sprintf("  symmetric: %s\n", ok_mark(x$symmetric)))
    cat(sprintf("  zero diag: %s\n", ok_mark(x$zero_diag)))
    cat(sprintf("  binary:    %s\n", ok_mark(x$binary)))
    cat(sprintf("  isolated:  %d node(s)\n", x$n_isolated))
    cat(sprintf("  edges:     %d\n", x$n_edges))
  }
  if (!is.null(x$id_error)) cat(sprintf("  ids:       %s\n", x$id_error))
  cat(sprintf("  overall:   %s\n", if (isTRUE(x$ok)) "ok" else "issues found"))
  invisible(x)
}
