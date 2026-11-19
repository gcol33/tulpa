# Multi-scale temporal structure

Specify a temporal random effect that decomposes variation into separate
scales: a smooth trend, an optional seasonal cycle, and a short-term
component. Each scale uses its own prior, letting slow and fast dynamics
be modelled jointly.

## Usage

``` r
temporal_multiscale(
  time_var,
  trend = c("rw2", "rw1", "none"),
  seasonal = NULL,
  short_term = c("ar1", "iid", "none"),
  shared = NULL,
  parameterization = c("auto", "noncentered", "centered")
)

# S3 method for class 'tulpa_temporal_multiscale'
print(x, ...)
```

## Arguments

- time_var:

  Single character string naming the time variable in the data.

- trend:

  Prior for the smooth long-term trend. One of `"rw2"`, `"rw1"`, or
  `"none"`.

- seasonal:

  Optional integer period (`>= 2`) of a seasonal cycle, e.g. `12` for
  monthly data with an annual cycle. `NULL` (default) omits the seasonal
  component.

- short_term:

  Prior for the short-term component. One of `"ar1"`, `"iid"`, or
  `"none"`.

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

- parameterization:

  How the exact sampler (`mode = "hmc"`) samples the components.
  `"auto"` scales field amplitudes using prior and observation
  information. `"noncentered"` samples the trend walk's level, its slope
  (RW2) and its innovations divided by the trend SD, the seasonal walk's
  level and its innovations divided by the seasonal SD, and the
  short-term component's innovations divided by its SD, and builds the
  fields from them; `"centered"` samples the fields' values. Both are
  the same model. The nested-Laplace and Gibbs routes do not read it,
  and the other sampler backends sample the fields' values.

- x:

  A `tulpa_temporal_multiscale` specification.

- ...:

  Ignored.

## Value

A `tulpa_temporal_multiscale` object.

## Details

At least one of `trend`, `seasonal`, or `short_term` must be active.

`mode = "auto"`, `"structured"` and `"nested_laplace"` fit the field by
nested Laplace, one block per component: the trend an RW1 / RW2 walk
over the time points, the seasonal cycle a cyclic RW1 over the period
(time point `t` reads position `(t - 1) mod seasonal + 1`), the
short-term component an AR1 or iid block. Each component's SD (and the
AR1 correlation) is its own axis of the outer grid. The walks carry a
sum-to-zero restriction, so the intercept holds the level; the trend is
told apart from the short-term component by its structure, since a walk
leaves a smooth path unpenalised where an AR1 or iid block shrinks every
time point to zero. Every route uses the same priors: a PC prior with
`P(sigma > 1) = 0.01` on each component's SD and a Beta(2, 2) on
`(rho + 1) / 2`.
[`temporal()`](https://gillescolling.com/tulpa/reference/temporal.md)
returns each component's posterior. The exact sampler (`mode = "hmc"`)
samples the trend, the seasonal and the short-term component non-centred
by default: under the centred parameterization the spread of each walk's
differences follows its SD, and the spread of the short-term values the
short-term SD, each a funnel whose divergent transitions concentrate at
the smallest variances. `mode = "gibbs"` runs the binomial Polya-Gamma
sampler for a field with an RW1 trend (or none). The field is one walk
per component shared by every observation.

## See also

[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for single-scale temporal priors.

## Examples

``` r
# Trend + annual seasonal cycle + AR1 short-term component on monthly data
temporal_multiscale("month", trend = "rw2", seasonal = 12, short_term = "ar1")
#> tulpa Multi-Scale temporal specification
#> =========================================
#> 
#> Time variable: month 
#> 
#> Components:
#>   Trend: RW2 (second-order random walk, smooth) 
#>   Seasonal: Period = 12 (cyclic RW1)
#>   Short-term: AR(1) (autocorrelated residuals) 
#> 
#> Shared: Yes (enters both processes) 
```
