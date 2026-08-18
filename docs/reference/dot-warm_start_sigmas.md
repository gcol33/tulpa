# Per-term random-effect SDs from a warm-start source fit

An EB fit estimated them, so they are read off `map`. A Laplace fit
conditioned on values the caller supplied, so those are passed in.

## Usage

``` r
.warm_start_sigmas(fit, sigma_re, n_terms)
```

## Arguments

- fit:

  A fitted `tulpa_fit`.

- sigma_re:

  Fallback SDs, used when the fit did not estimate them.

- n_terms:

  Number of random-effect terms.
