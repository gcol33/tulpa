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
