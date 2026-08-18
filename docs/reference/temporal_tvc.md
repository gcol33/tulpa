# Time-varying coefficient structure

Specify a time-varying coefficient (TVC): one or more fixed-effect
coefficients are allowed to evolve over time, with the evolution
governed by a temporal prior (`rw1`, `rw2`, `ar1`, or a GP).

## Usage

``` r
temporal_tvc(
  time_var,
  terms = 1,
  structure = c("rw1", "rw2", "ar1", "gp"),
  group_var = NULL,
  shared = NULL
)
```

## Arguments

- time_var:

  Single character string naming the time variable in the data.

- terms:

  Which coefficients vary over time. A formula, an integer vector of
  design-matrix column indices, or a character vector of term names.
  Default `1` (the intercept).

- structure:

  Temporal prior governing how the coefficients evolve. One of `"rw1"`,
  `"rw2"`, `"ar1"`, or `"gp"`.

- group_var:

  Optional character string naming a grouping variable for
  group-specific time-varying coefficients.

- shared:

  Whether the effect is shared across processes in a multi-process
  model. `NULL` (default) shares it; `FALSE` fits process-specific
  effects and emits a warning.

## Value

A `tulpa_tvc` object.

## See also

[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for the underlying temporal priors.

## Examples

``` r
# Intercept that drifts as a first-order random walk over year
temporal_tvc("year", structure = "rw1")
#> tulpa temporally-varying coefficients
#> ======================================
#> 
#> Time variable: year 
#> Structure: RW1 (first-order random walk) 
#> Shared: Yes (enters both processes) 
#> 
#> Terms: columns  1 
```
