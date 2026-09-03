# RW1 temporal structure (First-order Random Walk)

Specify a first-order random walk temporal random effect. RW1 penalizes
first differences, so adjacent time points are smoothed toward each
other: `phi[t] - phi[t-1] ~ N(0, sigma^2)`.

## Usage

``` r
temporal_rw1(time_var, group_var = NULL, cyclic = FALSE, shared = NULL)
```

## Arguments

- time_var:

  A formula (`~ time`) or single character string naming the time
  variable in the data.

- group_var:

  Optional formula (`~ g`) or character string naming a grouping
  variable. When supplied, a separate random walk is fit per group;
  `NULL` (default) fits a single walk shared across all observations.

- cyclic:

  Logical. If `TRUE`, the random walk wraps around so the last time
  point is a neighbour of the first (cyclic boundary, e.g. month of
  year). Default `FALSE`.

- shared:

  Whether the temporal effect is shared across processes in a
  multi-process model. `NULL` (default) shares the effect; `FALSE` fits
  process-specific effects and emits a warning about unshared
  confounding.

## Value

A `tulpa_temporal` object.

## Details

The precision matrix is rank `T - 1` (one constraint needed). RW1 is the
least smooth of the random-walk priors; for smoother trends see
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
and for a stationary alternative see
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md).

## See also

[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md)
for other temporal priors.

## Examples

``` r
# Create temporal RW1 specification
temporal_rw1("year")
temporal_rw1("month", cyclic = TRUE)
```
