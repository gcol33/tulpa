# Connected components of an adjacency graph, as a list of node-index vectors

Breadth-first over the sparse structure of W + W'
(`.adj_component_labels`, shared with the adjacency validator), so an
edge stored in one triangle only still joins its endpoints and no dense
copy is made. Returns what `graph_partition`
(`inst/include/tulpa/graph_components.h`) returns – the actual component
MEMBERSHIP, not just a count – so a genuine disconnected map (a mainland
plus islands) pins each component's constant over that component's real
nodes rather than an equal-size contiguous split. Components are ordered
by their lowest node, members increasing.

## Usage

``` r
.graph_components(adjacency)
```
