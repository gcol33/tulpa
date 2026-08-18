# Spatial structure specifications for tulpa

Functions to specify spatial random effects for tulpa models. Spatial
effects are shared between processes by default, which helps prevent
bias from spatially-structured unmeasured confounders.

## Value

The spatial constructors documented in this family
([`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
[`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md),
[`spatial_gp()`](https://gillescolling.com/tulpa/reference/spatial_gp.md),
and the others) each return a `tulpa_spatial` (or related) specification
object to pass to the `spatial` argument of
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).
