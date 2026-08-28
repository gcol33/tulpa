# Remove compiled blocks from the tgmrf_cpp() cache

Deletes cached compilation output under
[`tulpa_cache_dir()`](https://gillescolling.com/tulpa/reference/tulpa_cache_dir.md).
The cache holds build artefacts only – a removed entry is rebuilt by the
next
[`tgmrf_cpp()`](https://gillescolling.com/tulpa/reference/tgmrf_cpp.md)
call on the same source – so it can be emptied at any time.

## Usage

``` r
tulpa_cache_clear(older_than = 0)
```

## Arguments

- older_than:

  Remove only entries whose last modification is more than this many
  days ago. `0` (the default) removes every entry.

## Value

The number of entries removed, invisibly.

## See also

[`tulpa_cache_dir()`](https://gillescolling.com/tulpa/reference/tulpa_cache_dir.md)
for the location.

## Examples

``` r
if (FALSE) { # \dontrun{
# Deletes the caller's own cached builds, so it is not run on check.
tulpa_cache_clear()              # empty the cache
tulpa_cache_clear(older_than = 30)  # keep the last 30 days
} # }
```
