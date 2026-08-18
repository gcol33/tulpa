# Find all latent(...) calls in a formula's parse tree

Recursively walks the formula AST and collects all `latent(...)` calls.
The matched calls are returned unevaluated;
[`tulpa_parse_formula()`](https://gillescolling.com/tulpa/reference/tulpa_parse_formula.md)
resolves them in the formula's environment.

## Usage

``` r
find_latent_terms(term)
```

## Arguments

- term:

  A language object (formula term)

## Value

A list of language objects, each a `latent(...)` call
