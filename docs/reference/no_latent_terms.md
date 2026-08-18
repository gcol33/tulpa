# Remove all latent(...) calls from a formula's parse tree

Mirror of
[`nobars()`](https://gillescolling.com/tulpa/reference/nobars.md) for
`latent(...)` calls.

## Usage

``` r
no_latent_terms(term)
```

## Arguments

- term:

  A language object (formula term)

## Value

A language object with all `latent(...)` calls removed, or NULL if
nothing remains.
