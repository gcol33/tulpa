# Expand a varying-coefficient bar into per-column field specs

Expand the left-hand side of an lme4-style varying-coefficient bar
(`~ 1 + w || node`) against a data frame into one descriptor per design
matrix column. This is the same expansion
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) and
the inline
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
field constructor use internally to turn a bar into one CAR / temporal
field per design column, exposed so a downstream package can reuse the
one implementation rather than re-parsing the bar grammar.

The bar's right-hand side names the node index (the graph node for a
spatial field, the time index for a temporal field); it is not expanded
but is returned as the `node` attribute. The left-hand side is expanded
with
[`stats::model.matrix()`](https://rdrr.io/r/stats/model.matrix.html):
the intercept column (`1`) is the unweighted (all-ones) field, and each
covariate column is a varying coefficient whose per-observation weight
is that column's design value (`0 +` drops the intercept).

## Usage

``` r
tulpa_bar_field_specs(formula, data)
```

## Arguments

- formula:

  A one-sided formula carrying a single grouping bar, e.g.
  `~ 1 + w || cell`. Use
  [`tulpa_is_spatial_bar()`](https://gillescolling.com/tulpa/reference/tulpa_is_spatial_bar.md)
  to test first. The right-hand side must be a single bare column naming
  the node index; nesting (`a / b`), interaction (`a:b`), or expression
  grouping is rejected.

- data:

  A data frame whose columns the bar left-hand side and the node index
  refer to. The per-column weight vectors are evaluated against it, so
  they have `nrow(data)` entries.

## Value

A list with one element per design-matrix column, each a list:

- `column_name`:

  character; `"Intercept"` for the intercept column, otherwise the
  [`model.matrix()`](https://rdrr.io/r/stats/model.matrix.html) column
  name (e.g. `"w"`).

- `weight`:

  a numeric vector of length `nrow(data)` for a covariate column (the
  per-observation design value scaling that field), or `NULL` for the
  intercept column (which is the all-ones, unweighted field).

- `is_intercept`:

  logical; `TRUE` for the intercept column.

The list carries two attributes: `node` (character, the node-index
column named by the bar right-hand side) and `correlated` (logical,
`TRUE` for a single `|`, `FALSE` for a double `||`).

## See also

[`tulpa_is_spatial_bar()`](https://gillescolling.com/tulpa/reference/tulpa_is_spatial_bar.md)
for the recognizer,
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) for
the inline areal field constructor that consumes this expansion.

## Examples

``` r
d <- data.frame(cell = rep(1:5, each = 4), w = rnorm(20))

# Intercept plus a varying slope on w
specs <- tulpa_bar_field_specs(~ 1 + w || cell, d)
length(specs)                 # 2
specs[[1]]$column_name        # "Intercept"
is.null(specs[[1]]$weight)    # TRUE (unweighted field)
specs[[2]]$column_name        # "w"
identical(specs[[2]]$weight, d$w)  # TRUE
attr(specs, "node")           # "cell"
```
