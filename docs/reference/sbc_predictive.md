# Predictive shapes an SBC fitter reports

The tagged representations
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) reads. A
`fitter` (or a posterior-SBC `arms`) callback returns a named list of
ARMS, each a named list over quantities, and each entry is one of these
– the shape the backend actually reports for that quantity. Everything
downstream (the PIT, the CRPS, drawing from a predictive) dispatches on
the `kind` tag, so a new backend shape is one entry in three switches
rather than a parallel scorer.

## Usage

``` r
sbc_mixture(mu, var, w = NULL)

sbc_normal(mean, sd)

sbc_discrete(support, probs)

sbc_rank(rank, n_ref)

sbc_draws(x)
```

## Arguments

- mu, var, w:

  Component means, variances and weights. `w` defaults to equal weights
  and is normalized.

- mean, sd:

  Mean and standard deviation of a single Gaussian.

- support, probs:

  Finite support and its probabilities, normalized.

- rank, n_ref:

  The rank in `0:n_ref` of the truth among `n_ref` reference values, and
  that reference count.

- x:

  Posterior draws.

## Value

A list carrying a `kind` tag and that shape's parameters.

## Details

These are the extension point, not alternative front doors:
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) is the verb,
and these are the argument type it consumes.

`sbc_mixture()` is what an outer hyperparameter grid defines for a fixed
effect – component `k` is `N(mu_k, var_k)` with weight `w_k`, which is
exactly the mixture a nested-Laplace fit reports. `sbc_normal()` is the
one-component case, with its own constructor so a collapsed-moment read
says what it is. `sbc_discrete()` is a distribution on a finite support,
which is what a discrete hyperparameter grid defines for its own axis.
`sbc_rank()` is a rank of the truth among `n_ref` reference values,
which is what a joint log-likelihood comparison against posterior draws
produces – it needs no entry in the simulator's `theta`, since the
comparison against the truth already happened when the rank was formed.
`sbc_draws()` is for a backend reporting no analytic marginal.

The last three have ATOMS, so their PIT is randomized within the atom by
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md); reading a
rank against a continuous uniform is the classic silent SBC bug.

## See also

[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md)

## Examples

``` r
sbc_normal(0.3, 0.1)
#> $kind
#> [1] "mixture"
#> 
#> $mu
#> [1] 0.3
#> 
#> $var
#> [1] 0.01
#> 
#> $w
#> [1] 1
#> 
sbc_mixture(mu = c(0, 1), var = c(1, 4), w = c(0.7, 0.3))
#> $kind
#> [1] "mixture"
#> 
#> $mu
#> [1] 0 1
#> 
#> $var
#> [1] 1 4
#> 
#> $w
#> [1] 0.7 0.3
#> 
sbc_discrete(support = c(0.5, 1, 2), probs = c(0.2, 0.5, 0.3))
#> $kind
#> [1] "discrete"
#> 
#> $support
#> [1] 0.5 1.0 2.0
#> 
#> $probs
#> [1] 0.2 0.5 0.3
#> 
```
