# Default cache directory for `tgmrf_cpp()`-compiled DLLs

Reports the per-user directory under
`tools::R_user_dir("tulpa", "cache")` where
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
keeps the objects it compiles. Reporting the path does not create it:
the directory appears the first time a compile needs it, and
[`tulpa_cache_clear()`](https://gillescolling.com/tulpa/reference/tulpa_cache_clear.md)
removes what it holds.

## Usage

``` r
tulpa_cache_dir(create = FALSE)
```

## Arguments

- create:

  Create the directory if it is missing. `FALSE` (the default) only
  reports the path.

## Value

Absolute path (character) to the cache directory.

## See also

[`tulpa_cache_clear()`](https://gillescolling.com/tulpa/reference/tulpa_cache_clear.md)
to remove cached objects.

## Examples

``` r
# Where compiled tgmrf_cpp() blocks are cached. Reporting creates nothing.
tulpa_cache_dir()
```
