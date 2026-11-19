# Bar terms of a formula's parse tree

`findbars()` recursively walks the formula AST and collects all `|` and
`||` nodes found inside parentheses. These are the random effect
specifications.

`nobars()` recursively rewrites the formula AST, removing any `|` or
`||` nodes found inside parentheses. It returns the fixed-effects-only
formula.

## Usage

``` r
findbars(term)

nobars(term)
```

## Arguments

- term:

  A language object (formula term)

## Value

`findbars()`: a list of language objects, each a `|` or `||` call.
`nobars()`: a language object with all bar terms removed, or NULL if
nothing remains.

## Examples

``` r
findbars(y ~ x + (1 | g) + (1 + x | h))
#> [[1]]
#> 1 | g
#> 
#> [[2]]
#> 1 + x | h
#> 
nobars(y ~ x + (1 | g) + (1 + x | h))
#> y ~ x
```
