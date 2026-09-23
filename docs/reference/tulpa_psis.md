# Pareto-smoothed importance sampling

Smooths a set of importance ratios by replacing the largest weights with
the order statistics of a generalized-Pareto fit to their upper tail,
and returns the Pareto shape diagnostic `pareto_k` together with the
smoothed (normalized) log weights and their importance-sampling
effective sample size. `pareto_k` estimates the number of finite moments
of the raw weight distribution: `< 0.5` is good (finite variance). The
usable upper boundary is sample-size dependent,
`min(1 - 1/log10(S), 0.7)` for `S` draws (Vehtari et al. 2024): about
0.565 at `S = 200`, reaching the 0.7 cap only past `S` ~ 2154. Above it
the proposal cannot be reliably corrected to the target.

## Usage

``` r
tulpa_psis(log_ratios, tail_points = NULL)
```

## Arguments

- log_ratios:

  Numeric vector of (unnormalized) log importance ratios
  `log p_target(x) - log q_proposal(x)` evaluated at draws `x ~ q`.

- tail_points:

  Number of upper-tail order statistics for the generalized-Pareto fit,
  or `NULL` (default) for the automatic PSIS rule
  `ceil(min(0.2 * S, 3 * sqrt(S)))`. An explicit value is an expert
  tail-threshold control, capped at `floor(0.2 * S)` so the fit stays an
  extreme tail; it is NOT a precision knob (raise the draw count for a
  tighter shape estimate).

## Value

A list with `pareto_k` (the tail shape, `NA` if the sample is too small
to fit), `is_ess` (importance-sampling effective sample size,
`1 / sum(w^2)` on the normalized smoothed weights), `log_weights` (the
normalized smoothed log weights), `tail_len` (the tail size used), and
`tail_smoothed` (`FALSE` when the tail kept its raw log ratios because
the generalized-Pareto fit was not attempted or returned a shape / scale
the quantile function is undefined at; `pareto_k` then reports the
attempted fit and does not describe the returned weights).

## References

Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance
sampling. *JMLR* 25(72):1-58.

## See also

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
for the fit-level diagnostic front door.

## Examples

``` r
set.seed(1)
# Well-behaved importance ratios: k-hat is small.
ps <- tulpa_psis(rnorm(2000))
ps$pareto_k
#> [1] 0.3195261
ps$is_ess
#> [1] 686.8121
```
