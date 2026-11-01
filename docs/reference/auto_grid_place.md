# May the placement pass move a marked outer-grid axis?

Reads back what
[`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)'s
`place` argument recorded, so a wrapper that rebuilds a value
([`as.numeric()`](https://rdrr.io/r/base/numeric.html) drops every
attribute) can re-apply both halves of the declaration rather than only
the provenance half.

## Usage

``` r
auto_grid_place(x)
```

## Arguments

- x:

  Any object.

## Value

`FALSE` when `x` was marked `auto_grid(place = FALSE)`, `TRUE` otherwise
– including for a value carrying no mark at all, which the engine holds
because it reads as a pin rather than because it asked to be held.

## See also

[`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md),
[`is_auto_grid()`](https://gillescolling.com/tulpa/reference/is_auto_grid.md)

## Examples

``` r
auto_grid_place(auto_grid(c(0.5, 1, 2)))
#> [1] TRUE
auto_grid_place(auto_grid(c(0.5, 1, 2), place = FALSE))
#> [1] FALSE
```
