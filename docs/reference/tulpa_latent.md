# Latent Factor Specification for Unmeasured Confounders

Specify latent factors to capture shared unmeasured confounders between
model processes. Latent factors are particularly useful when you suspect
that both processes are driven by common unmeasured variables.

## Value

The constructor documented in this family
([`latent_factor()`](https://gillescolling.com/tulpa/reference/latent_factor.md))
returns a `tulpa_latent` specification object consumed by ratio /
multi-arm model packages built on tulpa (e.g. tulpaRatio), where the
shared factor enters two or more linear predictors and cancels in the
derived ratio. It is a multi-arm construct: the single-response
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) front
door does not read it (for an in-formula latent Gaussian block on a
single response, use `latent(tgmrf(...))`).
