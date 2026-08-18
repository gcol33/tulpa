# Remove all bar terms from a formula's parse tree

Recursively rewrites the formula AST, removing any `|` or `||` nodes
found inside parentheses. Returns the fixed-effects-only formula.

## Usage

``` r
nobars(term)
```

## Arguments

- term:

  A language object (formula term)

## Value

A language object with all bar terms removed, or NULL if nothing remains
