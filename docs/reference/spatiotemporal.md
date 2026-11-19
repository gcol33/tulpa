# Spatiotemporal interaction specification

A Knorr-Held (2000) space-time interaction for
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md): one
latent value \\\delta\_{st}\\ per area and time, added to the linear
predictor of every observation at that area and time. Pass the spec as
`tulpa(..., spatiotemporal = spatiotemporal(type = "IV"))`; it rides the
nested-Laplace path beside the spatial and temporal main effects, with
the interaction precision \\\tau\\ integrated on the outer grid.

## Usage

``` r
spatiotemporal(
  spatial = NULL,
  temporal = NULL,
  type = c("IV", "I", "II", "III", "iid", "separable"),
  tau_grid = NULL,
  shared = NULL
)
```

## Arguments

- spatial:

  The areal spatial spec whose graph structures the interaction
  ([`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
  [`spatial_bym2()`](https://gillescolling.com/tulpa/reference/spatial_bym2.md)
  or `list(type = "icar", adjacency = W)`). `NULL` (default) takes the
  fit's own `spatial =` spec. The interaction uses the graph's intrinsic
  CAR structure \\R_s = D - W\\ whatever the main effect's family.

- temporal:

  The temporal spec structuring the interaction in time,
  [`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)
  or
  [`temporal_rw2()`](https://gillescolling.com/tulpa/reference/temporal_rw2.md)
  (cyclic allowed). `NULL` (default) takes the fit's own `temporal =`
  spec.

- type:

  Interaction type:

  - `"I"` (or `"iid"`): unstructured, \\R = I_T \otimes I_S\\.

  - `"II"`: each area its own temporal walk, \\R = R_t \otimes I_S\\.

  - `"III"`: each time its own spatial ICAR, \\R = I_T \otimes R_s\\.

  - `"IV"` (or `"separable"`): \\R = R_t \otimes R_s\\.

- tau_grid:

  Optional outer-grid nodes for the interaction precision \\\tau\\; the
  default is the engine's intrinsic-field precision axis.

- shared:

  Not used.

## Value

A `tulpa_spatiotemporal` spec.

## Details

The interaction's prior is \\\delta \sim N(0, (\tau R)^-)\\ conditioned
on \\A \delta = 0\\, where the rows of \\A\\ span the null space of
\\R\\: for type II a walk's null space per area (its level, and its
linear trend under an acyclic RW2), for type III a sum over each
connected component of the graph at every time, for type IV both. These
are hard linear constraints: the inner Newton solves on the constraint
set and the Laplace approximation integrates over the \\ST - r\\
dimensional subspace, with the log-density normalised by the generalized
determinant \\\|R\|\_+\\. The interaction therefore carries no area
level and no time level (and, under RW2, no area-specific trend) that
the main effects already model.

The cells are laid out area-fastest: cell \\(s, t)\\ is coordinate
\\(t - 1) S + s\\. A time whose rows all have an `NA` response is still
a node of the walk, so the interaction forecasts it.
[`spatiotemporal_effects()`](https://gillescolling.com/tulpa/reference/spatiotemporal_effects.md)
reads the posterior back per cell.

## References

Knorr-Held, L. (2000). Bayesian modelling of inseparable space-time
variation in disease risk. Statistics in Medicine, 19(17-18), 2555-2567.

Goicoa, T., Adin, A., Ugarte, M. D., & Hodges, J. S. (2018). In
spatio-temporal disease mapping models, identifiability constraints
affect PQL and INLA results. Stochastic Environmental Research and Risk
Assessment, 32, 749-770.

## See also

[`spatiotemporal_effects()`](https://gillescolling.com/tulpa/reference/spatiotemporal_effects.md),
[`spatial_car()`](https://gillescolling.com/tulpa/reference/spatial_car.md),
[`temporal_rw1()`](https://gillescolling.com/tulpa/reference/temporal_rw1.md)

## Examples

``` r
# \donttest{
set.seed(1)
S <- 8; n_t <- 6
W <- matrix(0, S, S)
for (i in 1:(S - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
d <- expand.grid(region = 1:S, year = 1:n_t)
d$y <- rpois(nrow(d), exp(1 + 0.3 * sin(d$region) + 0.1 * d$year))
fit <- tulpa(y ~ 1 + spatial(region), data = d, family = "poisson",
             spatial = list(type = "icar", adjacency = W),
             temporal = temporal_rw1("year"),
             spatiotemporal = spatiotemporal(type = "IV"),
             mode = "nested_laplace", control = list(progress = FALSE))
head(spatiotemporal_effects(fit, format = "summary"))
#> Spatiotemporal interaction posterior summary: 8 locations x 6 times
#>   s t         mean       q2.5          q50     q97.5        sd
#> 1 1 1 -0.148479588 -0.6176273 -0.105694509 0.1434200 0.1954687
#> 2 2 1 -0.128295811 -0.5207575 -0.092275053 0.1079577 0.1625603
#> 3 3 1 -0.081686523 -0.4010702 -0.053782120 0.1389099 0.1343069
#> 4 4 1 -0.006839270 -0.2556486 -0.004630852 0.2596565 0.1196976
#> 5 5 1  0.003773783 -0.2673289  0.003364371 0.2519359 0.1206082
#> 6 6 1  0.091613505 -0.1232738  0.060303019 0.4290250 0.1388464
# }
```
