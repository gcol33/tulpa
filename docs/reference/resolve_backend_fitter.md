# Resolve the R fitter function for a backend (errors if unreachable).

Looks up the fitter *name* string in the registry and resolves it
lazily, so the registry stays independent of source-file load order.

## Usage

``` r
resolve_backend_fitter(backend)
```
