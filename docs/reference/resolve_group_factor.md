# Resolve a parsed RE spec to a grouping factor

Three paths, mirroring `resolve_group_rhs`:

- single column name: take `data[[name]]`.

- multiple column names (nested `a/b`): `interaction(data[names])`.

- language expression: evaluate against `data` (e.g., `factor(g)`).

## Usage

``` r
resolve_group_factor(re_spec, data, env)
```
