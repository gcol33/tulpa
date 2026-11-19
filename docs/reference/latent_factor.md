# Create a latent factor specification

Define latent factors for unmeasured structure shared by the responses
of a model. A unit `l` carries scores \\u\_{l f} \sim N(0, 1)\\ on
factors \\f = 1, \ldots, K\\, and response `s` loads on them with
loadings \\\lambda\_{s f}\\, so its linear predictor gains \\\sum_f
\lambda\_{s f} u\_{l f}\\. The factors add the covariance \\\Lambda
\Lambda^\top\\ between the linear predictors of the responses of one
unit.

## Usage

``` r
latent_factor(
  n_factors = 1L,
  unit = NULL,
  response = NULL,
  prior = NULL,
  shared = TRUE,
  constraint = NULL
)

# S3 method for class 'tulpa_latent'
print(x, ...)
```

## Arguments

- n_factors:

  Integer; number of latent factors. Default is 1.

- unit:

  For [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)'s
  `factors =` argument: the name of the column of `data` holding the
  unit that carries the scores (e.g. `"site"`).

- response:

  For [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)'s
  `factors =` argument: the name of the column of `data` holding the
  response level each row belongs to (e.g. `"species"`). Each level has
  its own fixed effects and loadings.

- prior:

  Prior of the factor loadings, or `NULL` for the fitter's default.
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  carries a normal prior centred at zero on every loading, folded onto
  the positive half-line on the diagonal, and takes
  `prior_normal(0, sd)` (default `sd = 1`); a model package may read
  other priors (tulpaRatio defaults to a PC prior with P(loading \> 1) =
  0.01 on the diagonal loadings).

- shared:

  Logical; if TRUE (default), the factors load on every process of the
  model. If FALSE, they load on the first process only (a two-process
  model package's numerator);
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  requires TRUE.

- constraint:

  `NULL` (default) or the identifiability constraint a sampling fitter
  pins the scores with: `"sum_to_zero"` (each factor's scores sum to
  zero across units) or `"first_zero"` (the first unit's scores are
  zero). [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  requires `NULL`: its scores carry a proper N(0, 1) prior and each
  response's intercept absorbs their level.

- x:

  A `tulpa_latent` specification.

- ...:

  Ignored.

## Value

A `tulpa_latent` object.

## Details

### Identification

The product \\\lambda\_{s f} u\_{l f}\\ is unchanged by a rescaling, a
sign flip and, for several factors, a rotation of the scores against the
loadings. The unit variance of the scores fixes the scale; a loading
matrix that is lower-triangular with a positive diagonal fixes the sign
and the rotation. \\\Lambda \Lambda^\top\\ is invariant to all three, so
it is the quantity to compare across fits.

### Choosing n_factors

With `S` responses, \\\Lambda\\ lower-triangular has \\S K - K (K - 1) /
2\\ free loadings against the \\S (S + 1) / 2\\ entries of a covariance
between the responses, so at most `S` factors fit. A two-process model
is full rank at `n_factors = 2`.

### Relationship to Random Effects

A random intercept is one response's deviation per group; a factor is a
deviation per unit that every response reads with its own loading, so it
carries the residual correlation between responses that the covariates
do not explain.

## See also

[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
(`factors =`),
[`latent_factors()`](https://gillescolling.com/tulpa/reference/latent_factors.md)
for the scores,
[`prior_normal()`](https://gillescolling.com/tulpa/reference/prior_distributions.md)

## Examples

``` r
# Two factors over sites, one response level per species
latent_factor(n_factors = 2, unit = "site", response = "species")
#> Latent factor specification
#> ===========================
#> 
#> Number of factors: 2 
#> Unit column: site 
#> Response column: species 
#> Shared: Yes (loads on all processes) 
#> Constraint: none (proper score prior) 
#> 
#> Loading prior:
#>   the fitter's default

# A tighter loading prior
latent_factor(unit = "site", response = "species",
              prior = prior_normal(0, 0.5))
#> Latent factor specification
#> ===========================
#> 
#> Number of factors: 1 
#> Unit column: site 
#> Response column: species 
#> Shared: Yes (loads on all processes) 
#> Constraint: none (proper score prior) 
#> 
#> Loading prior:
#>   Normal(0.00, 0.50)

# \donttest{
set.seed(1)
n_site <- 60; n_sp <- 4
u <- rnorm(n_site)
lambda <- c(0.8, 0.5, -0.6, 0.3)
d <- expand.grid(site = factor(seq_len(n_site)),
                 species = factor(paste0("sp", seq_len(n_sp))))
d$x <- rnorm(n_site)[d$site]
d$y <- rpois(nrow(d), exp(0.5 + 0.3 * d$x +
                          lambda[d$species] * u[d$site]))
fit <- tulpa(y ~ x, data = d, family = "poisson",
             factors = latent_factor(unit = "site", response = "species"))
fit$loadings
#>        factor1
#> sp1  0.7542586
#> sp2  0.3246828
#> sp3 -0.4679399
#> sp4  0.1682484
# }
```
