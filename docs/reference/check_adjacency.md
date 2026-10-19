# Validate a spatial adjacency matrix

Check that a hand-built adjacency matrix is a well-formed spatial graph
before passing it to
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) /
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md):
square, finite, symmetric, zero on the diagonal, 0/1 valued, free of
negative weights and of isolated nodes, and connected.
[`adjacency()`](https://gillescolling.com/tulpa/reference/adjacency.md)
runs the same checks on the graphs it constructs. The spatial
constructors refuse a graph with a missing, non-finite or negative
entry.

## Usage

``` r
check_adjacency(adjacency, ids = NULL)

# S3 method for class 'tulpa_adjacency_check'
print(x, ...)
```

## Arguments

- adjacency:

  A matrix (dense or sparse `Matrix`).

- ids:

  Optional cell identifiers; if supplied, their length must match
  `nrow(adjacency)` and they must be unique.

- x:

  A `tulpa_adjacency_check` report.

- ...:

  Ignored.

## Value

Invisibly, a `tulpa_adjacency_check` list with the per-check results
(`square`, `finite`, `symmetric`, `zero_diag`, `binary`, `nonneg`,
isolated-node indices, `n_components` and `component_sizes`, edge count)
and an overall `ok` flag. Issues are reported via
[`warning()`](https://rdrr.io/r/base/warning.html) and printed; the
function does not stop, so every problem surfaces in one pass.

## See also

[`adjacency()`](https://gillescolling.com/tulpa/reference/adjacency.md)
to construct a graph,
[`node_index()`](https://gillescolling.com/tulpa/reference/node_index.md).

## Examples

``` r
adj <- matrix(0, 4, 4)
for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1
check_adjacency(adj)
#> <adjacency check>
#>   square:    ok (4 x 4)
#>   symmetric: ok
#>   zero diag: ok
#>   binary:    ok
#>   isolated:  0 node(s)
#>   components: 1
#>   edges:     3
#>   overall:   ok
```
