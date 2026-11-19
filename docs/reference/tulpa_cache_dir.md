# Cache of `tgmrf_cpp()`-compiled DLLs

`tulpa_cache_dir()` reports the per-user directory under
`tools::R_user_dir("tulpa", "cache")` where
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
keeps the objects it compiles. Reporting the path does not create it:
the directory appears the first time a compile needs it.

`tulpa_cache_clear()` deletes cached compilation output under
`tulpa_cache_dir()`. The cache holds build artefacts only – a removed
entry is rebuilt by the next
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
call on the same source – so it can be emptied at any time.

## Usage

``` r
tulpa_cache_dir(create = FALSE)

tulpa_cache_clear(older_than = 0)
```

## Arguments

- create:

  Create the directory if it is missing. `FALSE` (the default) only
  reports the path.

- older_than:

  Remove only entries whose last modification is more than this many
  days ago. `0` (the default) removes every entry.

## Value

`tulpa_cache_dir()`: the absolute path (character) to the cache
directory. `tulpa_cache_clear()`: the number of entries removed,
invisibly.

## Examples

``` r
# Where compiled tgmrf_cpp() blocks are cached. Reporting creates nothing.
tulpa_cache_dir()
#> [1] "C:\\Users\\Gilles Colling\\AppData\\Local/R/cache/R/tulpa"

# Clearing deletes the caller's own cached builds, so it runs only in an
# interactive session.
if (interactive()) {
  tulpa_cache_clear(older_than = 30)  # keep the last 30 days
  tulpa_cache_clear()                 # empty the cache
}
```
