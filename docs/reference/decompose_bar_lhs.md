# Decompose the LHS of a bar term into intercept flag + slope language objects

Walks the additive chain of the LHS and separates numeric intercept
indicators (0, 1, -1) from slope term expressions. Returns language
objects, not deparsed strings.

## Usage

``` r
decompose_bar_lhs(lhs)
```

## Arguments

- lhs:

  A language object (LHS of a bar term)

## Value

A list with `has_intercept` (logical) and `slope_terms` (list of
language objects)
