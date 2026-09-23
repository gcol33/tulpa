# Run an expression under a chosen symplectic integrator

Evaluate `expr` with the integrator set to `name`, then restore whatever
was selected before – on an error as well as on success. The integrator
selection is process-global, so a bare
[`tulpa_integrator()`](https://gillescolling.com/tulpa/reference/tulpa_integrator.md)
call leaves every later fit in the session on the new scheme, and an
error between setting and restoring leaves it there permanently.

## Usage

``` r
with_tulpa_integrator(name, expr, mts_substeps = 4L)
```

## Arguments

- name:

  Integrator name, as for
  [`tulpa_integrator()`](https://gillescolling.com/tulpa/reference/tulpa_integrator.md).

- expr:

  Expression to evaluate. Evaluated in the caller's environment.

- mts_substeps:

  Inner prior-force substeps for `"mts"` (default 4).

## Value

The value of `expr`.

## See also

[`tulpa_integrator()`](https://gillescolling.com/tulpa/reference/tulpa_integrator.md)

## Examples

``` r
with_tulpa_integrator("yoshida4", tulpa_integrator())
#> [1] "yoshida4"
tulpa_integrator()   # unchanged
#> [1] "leapfrog"
```
