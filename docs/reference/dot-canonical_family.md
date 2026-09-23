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

## Details

Case is normalized as a last resort, after the exact spelling and the
alias table both miss: `stats::Gamma()$family` is `"Gamma"`,
capitalized, and `.family_object_to_name()` lowercases it for the
`family = Gamma(...)` object form, so the equivalent string form
(`family = "Gamma"`) has to reach the same registry entry rather than
being refused as unknown (gcol33/tulpa#806). Applied to a
`<family>_<link>` code too (`"Gamma_log"` -\> `"gamma_log"`), never to a
name that is already recognized as it stands – so a registry name that
happens to collide case-insensitively with another spelling is never
silently rewritten.
