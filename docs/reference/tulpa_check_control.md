# Validate a `control = list()` surface against its canonical key set

Front-door fitters across the `tulpa*` ecosystem carry their perf /
numerical / tuning knobs in a single `control` list (design principle
6). Without a name check a misspelled knob is a silent no-op –
`control$adaptve_grid` fits the default and reports nothing. This
validates the names a caller supplied against the whitelist of what the
target fitter actually reads, and errors listing the allowed set.

## Usage

``` r
tulpa_check_control(control, allowed, where)
```

## Arguments

- control:

  The `control` list to validate. `NULL` and empty lists pass.

- allowed:

  Character vector of accepted knob names.

- where:

  Name of the calling fitter, used in the error message.

## Value

`invisible(NULL)`, called for the side effect of erroring on an unknown
or unnamed knob.

## Details

Consumer packages (`tulpaRatio`, `tulpaObs`) call this with their own
key registry rather than reimplementing the check.

## Examples

``` r
tulpa_check_control(list(max_iter = 50), c("max_iter", "tol"), "my_fit")
try(tulpa_check_control(list(max_itr = 50), c("max_iter", "tol"), "my_fit"))
#> Error : Unknown control knob(s) for my_fit(): 'max_itr'.
#> Allowed: max_iter, tol.
```
