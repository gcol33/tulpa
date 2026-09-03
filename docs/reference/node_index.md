# Map cell identifiers to graph node indices

Translate a vector of original cell identifiers into the 1-based node
indices of a `tulpa_adjacency` graph, by key (not by row order). Use it
to add the node-index column the model's spatial grouping bar needs to
the observation data, which typically has many rows per cell and a
different row order than the graph.

## Usage

``` r
node_index(graph, ids)
```

## Arguments

- graph:

  A `tulpa_adjacency` object from
  [`adjacency()`](https://gillescolling.com/tulpa/reference/adjacency.md).

- ids:

  A vector of cell identifiers to look up (matched against `graph$ids`).

## Value

An integer vector the same length as `ids`, giving each one's node index
in `graph` (`NA` for identifiers absent from the graph).

## See also

[`adjacency()`](https://gillescolling.com/tulpa/reference/adjacency.md).

## Examples

``` r
grid <- expand.grid(x = 1:3, y = 1:3)
grid$cell <- paste0("c", seq_len(nrow(grid)))
g <- adjacency(grid, id = "cell")

obs <- data.frame(cell = c("c5", "c1", "c9"))
obs$cell_idx <- node_index(g, obs$cell)
obs
```
