# Parse a single random effect bar term

Takes a `|` or `||` language object and extracts the grouping
variable(s), effect terms (intercept, slopes), and correlation
structure. The bar operator drives the `correlated` flag: `|` -\> `TRUE`
(LKJ-Cholesky on the joint slope vector), `||` -\> `FALSE` (diagonal
covariance, one sigma per coefficient). This matches lme4 / glmmTMB and
lets downstream packages branch on `correlated` to choose between
independent-sigma and Cholesky parameterizations.

## Usage

``` r
parse_bar_term(bar_term)
```

## Arguments

- bar_term:

  A language object: `|`(lhs, rhs) or `||`(lhs, rhs)

## Value

A list of RE specs (one per grouping level). Each spec has:

- `group_var`: character, display label (colon-joined for nested)

- `group_vars`: character vector, each element a column name

- `group_expr`: language or NULL (set when grouping is a non-name expr)

- `slope_terms`: list of language objects (slope LHS terms)

- `has_intercept`: logical

- `correlated`: logical (TRUE for `|`, FALSE for `||`)

- `original`: the original bar language object

## Details

Nested grouping `(1 | a/b)` is expanded into one spec per level (a, then
a:b). `||` is preserved as a single spec rather than split into multiple
`|` bars, so the original user intent (one logical RE term with diagonal
covariance) round-trips through the parsed object.
