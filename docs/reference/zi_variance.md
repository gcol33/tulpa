# Response variance of the zero-inflated mixture.

Law of total variance over the mixture indicator: with probability `pi`
the response is a structural zero, otherwise it is a base-family draw.

## Usage

``` r
zi_variance(eta, logit_zi, family, n_trials = NULL, phi = 1, phi2 = NULL)
```
