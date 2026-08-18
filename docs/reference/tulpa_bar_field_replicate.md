# Replicate an areal graph across the levels of a factor (replicated CAR)

Build the block-diagonal Kronecker graph `I_L (x) Q` and the
level-offset node index for a replicated areal field: one independent
copy of the graph per level of a `by` factor, all sharing one precision.
This is the graph-side counterpart to
[`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md)
– that helper expands the coefficient columns and is graph-agnostic,
while replication needs the graph, so it is a sibling rather than a new
argument. A downstream package composes the two (column expansion x
replication) from the one implementation rather than re-deriving the
Kronecker remap.

## Usage

``` r
tulpa_bar_field_replicate(adjacency, node, by)
```

## Arguments

- adjacency:

  Symmetric adjacency matrix of the base graph (`[n_node x n_node]`,
  dense or sparse).

- node:

  Integer vector of 1-based graph-node indices, one per observation (the
  resolved bar right-hand side).

- by:

  A vector of the same length as `node` giving each observation's
  replication level; coerced to a factor. With `L` distinct levels the
  field is replicated `L` times.

## Value

A list:

- `adjacency`:

  the `[L*n_node x L*n_node]` block-diagonal Kronecker adjacency
  `I_L (x) Q` (the base graph for `L == 1`).

- `index`:

  integer vector, one per observation: the node index offset into its
  level's copy (`node + (level - 1) * n_node`).

- `n_levels`:

  the number of replication levels `L`.

- `n_nodes`:

  the base graph node count `n_node`.

- `levels`:

  the factor levels of `by`, in replicate order.

## See also

[`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md)
for the coefficient-column expansion,
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) for
the inline areal field constructor whose `by =` argument this powers.

## Examples

``` r
adj <- matrix(0, 4, 4)
for (i in 1:3) adj[i, i + 1] <- adj[i + 1, i] <- 1
node <- rep(1:4, times = 2)
lev  <- rep(c("a", "b"), each = 4)
rep_info <- tulpa_bar_field_replicate(adj, node, lev)
dim(rep_info$adjacency)   # 8 x 8 (I_2 (x) Q)
#> [1] 8 8
rep_info$index            # level b nodes offset by 4
#> [1] 1 2 3 4 5 6 7 8
```
