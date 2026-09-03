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
  group_var = NULL,
  shared = NULL
)
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

- group_var:

  Optional character string naming a grouping variable for
  group-specific temporal effects.

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

## Value

A `tulpa_temporal_multiscale` object.

## Details

At least one of `trend`, `seasonal`, or `short_term` must be active.

## See also

[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for single-scale temporal priors.

## Examples

``` r
# Trend + annual seasonal cycle + AR1 short-term component on monthly data
temporal_multiscale("month", trend = "rw2", seasonal = 12, short_term = "ar1")
```
