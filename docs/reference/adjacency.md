# Construct a spatial adjacency graph for areal models

Build the symmetric adjacency matrix that
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) and
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
consume, from a common spatial layout, instead of hand-coding it. The
model still receives an explicit graph
(`spatial(graph = g$adjacency, ...)`), so the graph stays inspectable
before fitting – the engine never guesses connectivity from coordinates
silently.

`adjacency()` is a single front-door verb that dispatches on the kind of
input:

- a `data.frame`:

  cell centroids on a regular grid – queen (edge or corner) or rook
  (edge only) contiguity over the lattice. This covers both plain
  coordinate grids and rasterised cells passed as a table of centres.

- an `sf` object:

  polygon contiguity from shared boundaries (queen = share a point or
  edge; rook = share an edge). Requires sf.

- a `SpatRaster` (terra) or `stars` object:

  raster cells – the lattice of (non-`NA`) cell centres, queen or rook.
  Requires terra or stars respectively.

The result is a `tulpa_adjacency` object holding the matrix plus the
cell identifier for each node, so a model can pass `g$adjacency` while
the observation data is remapped to 1-based node indices with
[`node_index()`](https://gillescolling.com/tulpa/reference/node_index.md).

## Usage

``` r
adjacency(x, ...)

# Default S3 method
adjacency(x, ...)

# S3 method for class 'data.frame'
adjacency(
  x,
  type = c("queen", "rook"),
  x_coord = "x",
  y_coord = "y",
  id = NULL,
  order = 1L,
  tolerance = 1.5,
  offsets = NULL,
  ...
)

# S3 method for class 'sf'
adjacency(x, type = c("queen", "rook", "touches"), id = NULL, ...)

# S3 method for class 'SpatRaster'
adjacency(
  x,
  type = c("queen", "rook"),
  order = 1L,
  tolerance = 1.5,
  offsets = NULL,
  na_rm = TRUE,
  ...
)

# S3 method for class 'stars'
adjacency(
  x,
  type = c("queen", "rook"),
  order = 1L,
  tolerance = 1.5,
  offsets = NULL,
  na_rm = TRUE,
  ...
)
```

## Arguments

- x:

  The spatial layout: a `data.frame` of centroids, an `sf` polygon
  layer, or a raster (`SpatRaster` / `stars`).

- ...:

  Passed to methods.

- type:

  Contiguity rule: `"queen"` (default; neighbours share an edge or a
  corner) or `"rook"` (neighbours share an edge only). For polygons,
  `"touches"` is an alias of `"queen"`.

- x_coord, y_coord:

  For the `data.frame` method, the names of the coordinate columns
  holding the cell centres (default `"x"` and `"y"`).

- id:

  For the `data.frame` and `sf` methods, an optional column naming each
  cell's identifier. Node `i` of the graph corresponds to `id`'s value
  in row `i`; pass this column to
  [`node_index()`](https://gillescolling.com/tulpa/reference/node_index.md)
  to translate the observation data's cell column into node indices.
  Default `NULL` uses the row position (`1:n`) as the identifier.

- order:

  For grid / raster layouts, the neighbourhood order (ring count):
  `order = 1` (default) is first-order contiguity (queen = 8 neighbours,
  rook = 4); `order = k` extends the stencil to the k-th ring, so queen
  keeps every cell within Chebyshev distance `k` (`(2k + 1)^2 - 1`
  neighbours: 8, 24, 48, ... for `k = 1, 2, 3`) and rook every cell
  within Manhattan distance `k` (`2k(k + 1)` neighbours: 4, 12, 24,
  ...). Any positive integer is allowed, so the neighbourhood is fully
  settable. Ignored by the `sf` (polygon) method, which is first-order
  contiguity only.

- tolerance:

  For grid / raster layouts, the per-offset neighbour distance cut-off
  as a multiple of the inferred cell size: a candidate at a lattice
  offset is kept when its true centre distance is at most `tolerance`
  times that offset's expected distance. The default `1.5` admits a
  snapped neighbour while rejecting one much farther than its lattice
  slot implies; raise it only for irregularly spaced centroids.
  Neighbourhood extent is set by `order`, not by `tolerance`.

- offsets:

  Advanced, grid / raster layouts only: a custom neighbour stencil as a
  two-column integer matrix or a list of length-2 `c(dx, dy)` lattice
  offsets (cell-step units, origin excluded). Supplying it overrides
  `type` and `order` and builds exactly that stencil, so any
  neighbourhood is expressible (anisotropic, ring-only, off-axis). An
  ICAR / CAR field is *undirected*, so the graph must be symmetric: an
  asymmetric stencil (e.g. `list(c(0, 1), c(-1, 0), c(1, 0))` = up /
  left / right but not down) is symmetrized to an undirected graph, with
  a message, since a directed neighbourhood cannot be represented by an
  undirected field. Default `NULL` uses the `type` / `order` stencil.

- na_rm:

  For raster layouts, drop cells whose value is `NA` before building the
  graph (default `TRUE`), so the nodes are the cells that carry data.

## Value

A `tulpa_adjacency` object: a list with

- `adjacency`:

  the `[n x n]` symmetric sparse adjacency matrix (`dgCMatrix`, 0/1,
  zero diagonal) to pass as `graph` / `adjacency`.

- `ids`:

  the cell identifier for each node, in node order (length `n`).

- `n`:

  the number of nodes.

- `cellsize`:

  the inferred cell size `c(x, y)` for grid / raster layouts, or `NA`
  for polygons.

- `type`:

  the contiguity rule used.

## See also

[`node_index()`](https://gillescolling.com/tulpa/reference/node_index.md)
to map cell identifiers to node indices,
[`check_adjacency()`](https://gillescolling.com/tulpa/reference/check_adjacency.md)
to validate a hand-built matrix,
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) and
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md)
which consume the graph.

## Examples

``` r
# A 3 x 3 regular grid of cell centres
grid <- expand.grid(x = 1:3, y = 1:3)
grid$cell <- paste0("c", seq_len(nrow(grid)))

g <- adjacency(grid, x_coord = "x", y_coord = "y", id = "cell",
               type = "queen")
g
#> <tulpa_adjacency>
#>   nodes: 9  edges: 20  (queen contiguity)
#>   neighbours per node: min 3, mean 4.44, max 8
#>   cell size: x = 1, y = 1
#>   pass $adjacency to spatial(graph = ) and node_index() to remap data
g$adjacency
#> 9 x 9 sparse Matrix of class "dgCMatrix"
#>                        
#>  [1,] . 1 . 1 1 . . . .
#>  [2,] 1 . 1 1 1 1 . . .
#>  [3,] . 1 . . 1 1 . . .
#>  [4,] 1 1 . . 1 . 1 1 .
#>  [5,] 1 1 1 1 . 1 1 1 1
#>  [6,] . 1 1 . 1 . . 1 1
#>  [7,] . . . 1 1 . . 1 .
#>  [8,] . . . 1 1 1 1 . 1
#>  [9,] . . . . 1 1 . 1 .

# Second-order (24-neighbour) queen contiguity: any order is settable
g2 <- adjacency(grid, id = "cell", order = 2)

# Advanced: a custom stencil (symmetrized for the undirected field)
g3 <- adjacency(grid, id = "cell", offsets = list(c(1, 0), c(0, 1)))
#> adjacency(): custom stencil was not symmetric; symmetrized to an undirected graph for the ICAR/CAR field (12 reverse edge(s) added). A directed neighbourhood cannot be represented by an undirected field.

# Use it in a model: graph stays explicit and inspectable
# spatial(graph = g$adjacency, formula = ~ 1 || cell_idx)

# Remap observation data (original cell ids -> 1:n node indices) by key
obs <- data.frame(cell = c("c5", "c1", "c5", "c9"))
obs$cell_idx <- node_index(g, obs$cell)
obs
#>   cell cell_idx
#> 1   c5        5
#> 2   c1        1
#> 3   c5        5
#> 4   c9        9
```
