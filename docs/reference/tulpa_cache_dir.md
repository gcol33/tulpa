# Default cache directory for `tgmrf_cpp()`-compiled DLLs

Returns a per-user cache directory under
`tools::R_user_dir("tulpa", "cache")` and creates it if missing.

## Usage

``` r
tulpa_cache_dir()
```

## Value

Absolute path (character) to the cache directory.
