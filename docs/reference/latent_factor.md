# Create a latent factor specification

Define latent factors for capturing unmeasured shared structure between
all processes. Factors are observation-level random effects that enter
both linear predictors when `shared = TRUE` (default).

## Usage

``` r
latent_factor(
  n_factors = 1L,
  prior = NULL,
  shared = TRUE,
  constraint = c("sum_to_zero", "first_zero"),
  scale = TRUE
)
```

## Arguments

- n_factors:

  Integer; number of latent factors. Default is 1. More factors capture
  more complex unmeasured structure but increase computational cost and
  risk overfitting.

- prior:

  Prior for factor standard deviations. Default is a PC prior with
  P(sigma \> 1) = 0.01, which shrinks toward simpler models.

- shared:

  Logical; if TRUE (default), latent factors enter both all process
  linear predictors identically. If FALSE, factors only affect the first
  process.

- constraint:

  Identifiability constraint for factors:

  - `"sum_to_zero"` (default): Factors sum to zero across observations

  - `"first_zero"`: First observation's factor is fixed to zero

- scale:

  Logical; if TRUE (default), factor loadings are standardized to have
  unit variance before applying sigma.

## Value

A `tulpa_latent` object consumed by ratio / multi-arm model packages
built on tulpa (e.g. tulpaRatio); not read by the single-response
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) front
door (use `latent(tgmrf(...))` for a single-response latent Gaussian
block).

## Details

### Why Use Latent Factors?

When modeling multiple processes, they often share unmeasured
confounders. For example, in relative abundance data, both the focal
species count and total count might be affected by:

- Observer skill (unmeasured)

- Local microhabitat conditions (unmeasured)

- Weather on sampling day (unmeasured)

Without accounting for these shared drivers, estimates can be biased.
Latent factors capture this shared structure without requiring the
confounders to be measured.

### Mathematical Model

For observation i with K latent factors, on each of two model arms
(processes 1 and 2):

\$\$\eta^{(1)}\_i = X^{(1)}\_i \beta^{(1)} + \sum\_{k=1}^{K} f\_{ik}
\sigma_k + \ldots\$\$ \$\$\eta^{(2)}\_i = X^{(2)}\_i \beta^{(2)} +
\sum\_{k=1}^{K} f\_{ik} \sigma_k + \ldots\$\$

where:

- \\f\_{ik} \sim N(0, 1)\\ are standardized factor scores

- \\\sigma_k\\ are factor standard deviations with PC prior

- Identifiability: \\\sum_i f\_{ik} = 0\\ for each k

Because factors enter both linear predictors identically (when shared),
they cancel in derived quantities (e.g., ratios, differences):

\$\$\eta^{(1)}\_i - \eta^{(2)}\_i\$\$

This means factors capture shared effects that would otherwise bias
derived quantities.

### Choosing n_factors

- Start with `n_factors = 1` for simple unmeasured confounding

- Use `n_factors = 2-3` if you suspect multiple independent confounders

- More than 3 factors is rarely needed and risks overfitting

- The PC prior provides regularization, shrinking unneeded factors
  toward zero

### Relationship to Random Effects

Latent factors differ from random effects in several ways:

- Random effects are grouped (e.g., site-level), factors are
  observation-level

- Random effects require grouping structure, factors don't

- Factors capture residual correlation not explained by observed
  predictors

You can use both together: random effects for known grouping, factors
for residual unmeasured confounding.

## See also

[`prior_pc()`](https://gillescolling.com/tulpa/reference/prior_pc.md)
for prior specification;
[`latent()`](https://gillescolling.com/tulpa/reference/latent.md) /
[`tgmrf()`](https://gillescolling.com/tulpa/reference/tgmrf.md) for a
single-response in-formula latent Gaussian block

## Examples

``` r
# Basic latent factor (single shared factor)
latent_factor()

# Two latent factors
latent_factor(n_factors = 2)

# Custom prior (more regularization)
latent_factor(n_factors = 1, prior = prior_pc(U = 0.5, alpha = 0.01))

# Numerator-only factor (not shared)
latent_factor(n_factors = 1, shared = FALSE)

# The spec is consumed by a ratio / multi-arm model package (tulpaRatio owns
# the two-arm `species | total ~ ...` formula and the negbin/negbin ratio
# family); it is that package's fitter, not the single-response tulpa() front
# door, that reads the `latent_factor()` object.
```
