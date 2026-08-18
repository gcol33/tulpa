# Temporal structure specifications for tulpa

Functions to specify temporal random effects for tulpa models. Temporal
effects are shared between processes by default, which helps prevent
bias from temporally-structured unmeasured confounders.

## Value

The temporal constructors documented in this family
([`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md),
[`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md),
[`temporal_ar1()`](https://gillescolling.com/tulpa/reference/temporal_ar1.md),
and the others) each return a `tulpa_temporal` specification object to
pass to the `temporal` argument of
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).
