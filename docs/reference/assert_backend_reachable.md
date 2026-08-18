# Error if a selected backend has no R-level fitter.

Enforces the registry honesty contract: a backend may ship a C++ kernel
reachable from model packages (via `LinkingTo: tulpa`) yet have no R
entry point. Such a backend must never be *silently* selectable from R –
selecting it errors with a precise message naming the C-ABI symbol,
rather than pretending to dispatch.

## Usage

``` r
assert_backend_reachable(backend)
```
