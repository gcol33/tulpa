# Score of the zero-inflated mixture in the two-process eta ordering.

Returns an `n x 2` matrix with columns `count` (d loglik / d eta_count)
and `zi` (d loglik / d logit_zi).

## Usage

``` r
zi_score_eta(eta, logit_zi, y, family, n_trials = NULL, phi = 1, phi2 = NULL)
```
