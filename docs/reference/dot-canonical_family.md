# Resolve a family spelling to its canonical registry name.

A no-op for a canonical name, for a `<family>_<link>` code, and for
anything unrecognized, which reaches `.family_or_stop()` and errors
there against the canonical list.

## Usage

``` r
.canonical_family(family)
```

## Arguments

- family:

  Family identifier as supplied by the caller.
