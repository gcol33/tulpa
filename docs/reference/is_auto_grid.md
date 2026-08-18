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
marker.

## See also

[`auto_grid()`](https://gillescolling.com/tulpa/reference/auto_grid.md)

## Examples

``` r
is_auto_grid(auto_grid(c(0.5, 1, 2)))
#> [1] TRUE
is_auto_grid(c(0.5, 1, 2))
#> [1] FALSE
```
