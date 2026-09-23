# Recognize an inline varying-coefficient bar

Test whether a one-sided formula carries a single varying-coefficient
grouping bar of the form `~ 1 + w || node` (independent fields, `||`) or
`~ 1 + w | node` (correlated fields, `|`). This is the same grammar
[`spatial()`](https://gillescolling.com/tulpa/reference/spatial.md) and
the inline
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
field constructor accept, and the grammar
[`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md)
expands. A downstream package can use it to branch on "is this term a
spatial / temporal varying-coefficient bar?" before calling
[`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md).

## Usage

``` r
tulpa_is_spatial_bar(x)
```

## Arguments

- x:

  A one-sided formula (e.g. `~ 1 + w || node`) or the bar language
  object itself (`quote(1 + w || node)`).

## Value

A single logical: `TRUE` when `x` is (or wraps) a `|` / `||` bar,
`FALSE` otherwise (a plain formula such as `~ 1 + w`, a non-bar term, or
a non-formula / non-language input).

## See also

[`tulpa_bar_field_specs()`](https://gillescolling.com/tulpa/reference/tulpa_bar_field_specs.md)
for expanding the bar into per-column field specs.

## Examples

``` r
tulpa_is_spatial_bar(~ 1 + w || cell)   # TRUE
#> [1] TRUE
tulpa_is_spatial_bar(~ 1 + w | cell)    # TRUE (correlated)
#> [1] TRUE
tulpa_is_spatial_bar(~ 1 + w)           # FALSE (no bar)
#> [1] FALSE
```
