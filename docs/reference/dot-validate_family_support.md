# Validate the response against a family's support.

The one support rule set, shared by the
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) front
door and by the engine fitters
([`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md),
[`tulpa_laplace_beta()`](https://gillescolling.com/tulpa/reference/tulpa_laplace_beta.md),
[`tulpa_nuts_beta()`](https://gillescolling.com/tulpa/reference/tulpa_nuts_beta.md)).
Holding it in a single function is what keeps a boundary response
rejected with the same message wherever it enters: a per-fitter copy is
only as good as the set of fitters that carry it, and
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) reaches
beta-with-RE through a sampler that carried none.

## Usage

``` r
.validate_family_support(family, y, n_trials = NULL, zi = FALSE)
```

## Arguments

- family:

  Family identifier, canonical or aliased.

- y:

  Response vector, or `NULL` (a model with no response passes).

- n_trials:

  Binomial denominators, when known. Supplied, the binomial rule is
  `0 <= y <= n_trials`; absent, the response is the 0/1 form.

- zi:

  Whether a zero-inflation component is fitted alongside the family,
  passed through to the count rule for the hurdle case.

## Details

Counts delegate to
[`.validate_family_counts()`](https://gillescolling.com/tulpa/reference/dot-validate_family_counts.md);
the continuous families and the binomial denominator add their rules
here.
