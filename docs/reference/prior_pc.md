# Penalized complexity (PC) prior

Specify a PC prior for a positive parameter (typically a standard
deviation). PC priors shrink toward simpler models by penalizing
deviation from a base model.

## Usage

``` r
prior_pc(U = 1, alpha = 0.01)
```

## Arguments

- U:

  Upper bound. P(x \> U) = alpha.

- alpha:

  Tail probability. Default 0.01.

## Value

A `tulpa_prior` object

## Details

The PC prior is specified via: P(sigma \> U) = alpha

- U is the upper bound you consider "large"

- alpha is the probability of exceeding it

This implies an exponential prior with rate = -log(alpha) / U.

## References

Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H.
(2017). Penalising model component complexity: A principled, practical
approach to constructing priors. Statistical Science, 32(1), 1-28.

## Examples

``` r
prior_pc(U = 1, alpha = 0.01)    # P(sigma > 1) = 0.01
#> PC prior: P(x > 1.00) = 0.010
#>   => Exponential(4.605)
prior_pc(U = 0.5, alpha = 0.05)  # Tighter, P(sigma > 0.5) = 0.05
#> PC prior: P(x > 0.50) = 0.050
#>   => Exponential(5.991)
```
