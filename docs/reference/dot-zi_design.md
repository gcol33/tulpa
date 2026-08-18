# Build the zero-inflation design matrix from `ziformula`.

A one-sided formula giving the fixed effects of the structural-zero
logit. `~ 1` is the constant-probability model. Random effects in the ZI
predictor are not supported: the compiled kernels share the
random-effect block into the count process only, so a bar here would be
silently dropped.

## Usage

``` r
.zi_design(ziformula, data, n_obs)
```
