# Prior distributions

Constructors for the priors a
[`tulpa_priors()`](https://gillescolling.com/tulpa/reference/tulpa_priors.md)
specification takes, one per distribution:

- `prior_normal()`: a normal prior.

- `prior_half_normal()`: a half-normal prior (truncated at 0) for a
  positive parameter.

- `prior_half_cauchy()`: a half-Cauchy prior for a positive parameter.
  It has heavier tails than the half-normal and is often used for
  variance parameters.

- `prior_gamma()`: a gamma prior for a positive parameter.

- `prior_exponential()`: an exponential prior for a positive parameter.

- `prior_beta()`: a beta prior for a parameter bounded in (0, 1).

- `prior_pc()`: a penalized complexity (PC) prior for a positive
  parameter, typically a standard deviation. PC priors shrink toward
  simpler models by penalizing deviation from a base model.

## Usage

``` r
prior_normal(mean = 0, sd = 2.5)

prior_half_normal(sd = 1)

prior_half_cauchy(scale = 2.5)

prior_gamma(shape = 2, rate = 0.1)

prior_exponential(rate = 1)

prior_beta(alpha = 1, beta = 1)

prior_pc(U = 1, alpha = 0.01)
```

## Arguments

- mean:

  Prior mean of `prior_normal()`. Default 0.

- sd:

  Standard deviation of `prior_normal()`, scale of
  `prior_half_normal()`. Must be positive.

- scale:

  Scale of `prior_half_cauchy()`. Must be positive.

- shape:

  Shape (alpha) of `prior_gamma()`. Must be positive.

- rate:

  Rate of `prior_gamma()` (beta) or `prior_exponential()` (lambda). Must
  be positive.

- alpha:

  First shape parameter of `prior_beta()`, or the tail probability of
  `prior_pc()`, `P(x > U) = alpha` (default 0.01). Must be positive; the
  tail probability must be below 1.

- beta:

  Second shape parameter of `prior_beta()`. Must be positive.

- U:

  Upper bound of `prior_pc()`: `P(x > U) = alpha`.

## Value

A `tulpa_prior` object.

## Details

The gamma prior has mean `shape / rate` and variance `shape / rate^2`.

The beta prior's shape parameters set its form:

- alpha = beta = 1: Uniform(0, 1)

- alpha = beta = 2: Symmetric, peaked at 0.5

- alpha \> beta: Skewed toward 1

- alpha \< beta: Skewed toward 0

The PC prior is specified via `P(sigma > U) = alpha`: `U` is the upper
bound you consider "large" and `alpha` the probability of exceeding it.
This implies an exponential prior with rate `-log(alpha) / U`.

## References

Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sorbye, S. H.
(2017). Penalising model component complexity: A principled, practical
approach to constructing priors. Statistical Science, 32(1), 1-28.

## Examples

``` r
prior_normal(0, 2.5)
#> Normal(0.00, 2.50)
prior_normal(0, 1)
#> Normal(0.00, 1.00)
prior_half_normal(1)
#> Half-Normal(1.00)
prior_half_cauchy(2.5)
#> Half-Cauchy(2.50)
prior_gamma(2, 0.1)  # Mean = 20, weakly informative
#> Gamma(2.00, 0.10)  [mean = 20.00]
prior_gamma(1, 1)    # Exponential(1)
#> Gamma(1.00, 1.00)  [mean = 1.00]
prior_exponential(1)
#> Exponential(1.00)  [mean = 1.00]
prior_beta(1, 1)   # Uniform
#> Beta(1.00, 1.00)  [mean = 0.50]
prior_beta(2, 2)   # Symmetric, centered at 0.5
#> Beta(2.00, 2.00)  [mean = 0.50]
prior_beta(5, 2)   # Skewed toward 1 (for high autocorrelation)
#> Beta(5.00, 2.00)  [mean = 0.71]
prior_pc(U = 1, alpha = 0.01)    # P(sigma > 1) = 0.01
#> PC prior: P(x > 1.00) = 0.010
#>   => Exponential(4.605)
prior_pc(U = 0.5, alpha = 0.05)  # Tighter, P(sigma > 0.5) = 0.05
#> PC prior: P(x > 0.50) = 0.050
#>   => Exponential(5.991)
```
