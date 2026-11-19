# Latent factor specification

Latent factors are unmeasured drivers shared by the responses of a
model: each unit carries a score on every factor and each response
carries its own loading on it. The specification built by
[`latent_factor()`](https://gillescolling.com/tulpa/reference/latent_factor.md)
is read by
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) through
its `factors =` argument (a long-format joint model, one response level
per row) and by model packages built on tulpa, such as tulpaRatio, whose
two-process models load the factors on the numerator and the
denominator.

## Value

[`latent_factor()`](https://gillescolling.com/tulpa/reference/latent_factor.md)
returns a `tulpa_latent` specification.
