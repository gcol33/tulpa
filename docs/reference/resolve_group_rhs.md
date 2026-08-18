# Resolve the RHS of a bar term into one or more group specs

Three cases:

- simple name: `g` -\> one spec with `group_vars = "g"`.

- nested `/` : `a/b` -\> two specs (`"a"`, `c("a","b")`).

- other call: `factor(g)` or `g1:g2` -\> one spec carrying the language
  object in `group_expr` so we evaluate it at build time instead of
  guessing column names.

## Usage

``` r
resolve_group_rhs(rhs)
```

## Arguments

- rhs:

  A language object (RHS of `|` or `||`).

## Value

List of group specs.
