# Split a Laplace / EB mode vector into its semantic blocks

Returns the count-side fixed effects, the zero-inflation coefficients,
and the per-term random-effect deviations, so the caller places each
into the sampler's corresponding block rather than relying on the two
flat vectors agreeing end to end (they do not, under zero inflation).

## Usage

``` r
.warm_start_blocks(fit, re_terms)
```

## Arguments

- fit:

  A fitted `tulpa_fit` from `mode = "eb"` or `mode = "laplace"`.

- re_terms:

  The random-effect terms of the model being sampled, in
  [`tulpa_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_laplace.md)
  form. Taken from the caller rather than from the fit: `re_layout` is
  attached by the front door, so a source fit built directly by
  [`tulpa_eb()`](https://gillescolling.com/tulpa/reference/tulpa_eb.md)
  does not carry one, and the model being sampled is the authority on
  the block shapes in any case.
