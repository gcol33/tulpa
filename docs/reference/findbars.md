# Find all bar terms in a formula's parse tree

Recursively walks the formula AST and collects all `|` and `||` nodes
found inside parentheses. These are the random effect specifications.

## Usage

``` r
findbars(term)
```

## Arguments

- term:

  A language object (formula term)

## Value

A list of language objects, each a `|` or `||` call

## Examples

``` r
findbars(y ~ x + (1 | g) + (1 + x | h))
#> [[1]]
#> 1 | g
#> 
#> [[2]]
#> 1 + x | h
#> 
```
