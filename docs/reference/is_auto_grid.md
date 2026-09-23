# Is an outer-grid setting marked as a default?

Is an outer-grid setting marked as a default?

## Usage

``` r
is_auto_grid(x)
```

## Arguments

- x:

  Any object.

## Value

`TRUE` when `x` carries the
[`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)
marker. This is the PROVENANCE question – whose choice the nodes are –
and is `TRUE` whether or not the mark also asked for them to be
integrated as written; that is
[`auto_grid_place()`](https://gillescolling.com/tulpa/reference/auto_grid_place.md).

## See also

[`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md),
[`auto_grid_place()`](https://gillescolling.com/tulpa/reference/auto_grid_place.md)

## Examples

``` r
is_auto_grid(auto_grid(c(0.5, 1, 2)))
#> [1] TRUE
is_auto_grid(c(0.5, 1, 2))
#> [1] FALSE
```
