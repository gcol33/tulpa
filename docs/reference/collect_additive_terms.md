# Collect additive terms from a + chain

Recursively flattens `a + b + c` into `list(a, b, c)`.

## Usage

``` r
collect_additive_terms(expr)
```

## Arguments

- expr:

  A language object

## Value

A list of language objects (individual terms)
