# Connected components of an adjacency graph, as a list of node-index vectors.

Iterative depth-first search, so it carries no recursion-depth risk on a
large map. Returns what `graph_partition`
(`inst/include/tulpa/graph_components.h`) returns – the actual component
MEMBERSHIP, not just a count – so a genuine disconnected map (a mainland
plus islands) pins each component's constant over that component's real
nodes rather than an equal-size contiguous split. Takes the dense
adjacency rather than a sparse one because symmetric sparse storage
(`dsCMatrix`) keeps a single triangle, which would put an edge's two
endpoints in different components.

## Usage

``` r
.graph_components(adjacency)
```
