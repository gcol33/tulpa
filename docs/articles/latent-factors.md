# Latent factors

This article fits counts of eight species at the same 120 sites with two
latent site factors: unmeasured site properties that every species
responds to, each species with its own loadings. The data are in long
format, one row per site and species, and the factors enter through the
`factors =` argument of
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md).
Installation is covered in [Get
started](https://gillescolling.com/tulpa/articles/tulpa.md).

``` r

library(tulpa)
```

## The model

Index the species by $`k = 1, \ldots, K`$, the sites by
$`s = 1, \ldots, S`$ and the factors by $`f = 1, \ldots, F`$. Let
$`y_{ks}`$ denote the count of species $`k`$ at site $`s`$ and $`x_s`$ a
measured site covariate. Each species has its own intercept and slope,
and reads the site scores $`u_{sf}`$ through its own loadings
$`\lambda_{kf}`$:

``` math
y_{ks} \sim \mathrm{Poisson}(\mu_{ks}), \qquad
\log \mu_{ks} = \beta_{k0} + \beta_{k1} x_s + \sum_{f} \lambda_{kf} u_{sf},
\qquad u_{sf} \sim \mathcal{N}(0, 1).
```

The product $`\lambda_{kf} u_{sf}`$ is unchanged by three moves:
multiplying the scores by $`c`$ and dividing the loadings by $`c`$,
flipping the sign of both, and, with two or more factors, rotating the
scores against the loadings. The unit variance of the scores fixes the
first. The other two are fixed by the loading matrix
$`\Lambda = (\lambda_{kf})`$ being lower-triangular with a positive
diagonal: factor $`f`$ does not load on species $`1, \ldots, f - 1`$ and
loads positively on species $`f`$. A shift of every score by $`a`$ moves
species $`k`$ by $`a \lambda_{kf}`$, which its intercept absorbs.

The factors add the covariance $`\Lambda \Lambda^\top`$ between the
linear predictors of the species at one site: the residual covariance
left after the covariate. A rotation of the factors leaves
$`\Lambda \Lambda^\top`$ unchanged, so it is the quantity that does not
depend on which species anchor the factors.

The loadings carry the prior $`\mathcal{N}(0, \sigma_\lambda^2)`$,
folded onto the positive half-line on the diagonal, with
$`\sigma_\lambda = 1`$ unless `prior_normal(0, sd)` says otherwise. The
fit is a joint nested Laplace with one response arm per species; the
scores are integrated out site by site by a Laplace approximation inside
the inner solve, so the loadings and the coefficients sit at the mode of
their posterior with the scores marginalised.

## Simulating data

The community has $`K = 8`$ species at $`S = 120`$ sites and two
factors. Species 1 does not load on factor 2, so the true loading matrix
is itself lower-triangular here; the comparison below reads
$`\Lambda \Lambda^\top`$, which would not need that.

``` r

set.seed(4)
S <- 120L; K <- 8L
Lam <- cbind(c(0.9, 0.7, -0.6, 0.4, 0.3, 0.0, -0.8, 0.5),
             c(0.0, 0.8, 0.5, -0.6, 0.6, 0.4, 0.2, -0.3))
U  <- cbind(rnorm(S), rnorm(S))
b0 <- c(1, 0.5, 1.2, 0.2, 0.8, 1.5, 0.3, 0.6)
b1 <- c(0.3, -0.2, 0.5, 0, 0.4, -0.3, 0.2, 0.1)
x  <- rnorm(S)
```

``` r

d <- expand.grid(site = factor(seq_len(S)),
                 species = factor(paste0("sp", seq_len(K))))
k <- as.integer(d$species); s <- as.integer(d$site)
d$x <- x[s]
d$y <- rpois(nrow(d), exp(b0[k] + b1[k] * d$x + rowSums(Lam[k, ] * U[s, ])))
head(d, 3)
#>   site species           x y
#> 1    1     sp1 -0.98394416 4
#> 2    2     sp1 -0.02270056 2
#> 3    3     sp1 -0.88455021 7
```

## Fitting

[`latent_factor()`](https://gillescolling.com/tulpa/reference/latent_factor.md)
names the two columns that give the data its shape: the unit carrying
the scores and the response level each row belongs to.

``` r

spec <- latent_factor(n_factors = 2, unit = "site", response = "species")
spec
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
```

``` r

fit <- tulpa(y ~ x, data = d, family = "poisson", factors = spec)
fit
#> tulpa fit: latent-factor model (2 factors over 120 units, 8 response levels)
#>   N = 960  family = poisson  fixed effects = 16 
#> 
#> Loadings (posterior mean):
#>     factor1 factor2
#> sp1   0.813   0.000
#> sp2   0.718   0.936
#> sp3  -0.544   0.504
#> sp4   0.336  -0.661
#> sp5   0.428   0.596
#> sp6  -0.009   0.443
#> sp7  -0.681   0.168
#> sp8   0.515  -0.367
#> 
#> Fixed effects (posterior mean):
#> sp1:(Intercept)           sp1:x sp2:(Intercept)           sp2:x sp3:(Intercept) 
#>          1.0140          0.2559          0.3900         -0.0699          0.9974 
#>           sp3:x sp4:(Intercept)           sp4:x sp5:(Intercept)           sp5:x 
#>          0.5256          0.2149          0.0879          0.7911          0.2843 
#> sp6:(Intercept)           sp6:x sp7:(Intercept)           sp7:x sp8:(Intercept) 
#>          1.4540         -0.3079          0.2766          0.1167          0.6202 
#>           sp8:x 
#>         -0.0052
```

The order of the levels of `species` decides which species anchor the
factors: `sp1` for factor 1 and `sp2` for factor 2. Put first a species
that is common and clearly responds to the factors; reorder with
`factor(..., levels = )` otherwise.

The coefficients of every species, with intervals, come from
[`summary()`](https://rdrr.io/r/base/summary.html):

``` r

head(coef(summary(fit)), 4)
#>                    estimate  std.error       2.5 %    97.5 %
#> sp1:(Intercept)  1.01398360 0.10082585  0.81954195 1.2052681
#> sp1:x            0.25589817 0.09875076  0.05427611 0.4600495
#> sp2:(Intercept)  0.38997674 0.14270699  0.11193041 0.6527773
#> sp2:x           -0.06986492 0.14088700 -0.36366100 0.2070429
```

## Loadings and the covariance between species

`fit$loadings` holds the posterior mean of $`\Lambda`$ (also returned by
`loadings(fit)`), and `fit$loadings_draws` its draws:

``` r

round(cbind(fit$loadings, true1 = Lam[, 1], true2 = Lam[, 2]), 2)
#>     factor1 factor2 true1 true2
#> sp1    0.81    0.00   0.9   0.0
#> sp2    0.72    0.94   0.7   0.8
#> sp3   -0.54    0.50  -0.6   0.5
#> sp4    0.34   -0.66   0.4  -0.6
#> sp5    0.43    0.60   0.3   0.6
#> sp6   -0.01    0.44   0.0   0.4
#> sp7   -0.68    0.17  -0.8   0.2
#> sp8    0.52   -0.37   0.5  -0.3
```

`fit$factor_cov` is the posterior mean of $`\Lambda \Lambda^\top`$,
formed draw by draw so that it carries the uncertainty of the loadings:

``` r

low <- lower.tri(diag(K), diag = TRUE)
C_true <- tcrossprod(Lam)
draws_C <- apply(fit$loadings_draws, 1, function(L) tcrossprod(L)[low])
q_C <- apply(draws_C, 1, quantile, c(0.025, 0.975))
c(relative_error = norm(fit$factor_cov - C_true, "F") / norm(C_true, "F"),
  covered = mean(q_C[1, ] <= C_true[low] & C_true[low] <= q_C[2, ]))
#> relative_error        covered 
#>      0.2213971      1.0000000
```

``` r

plot(C_true[low], fit$factor_cov[low], pch = 19, ylim = range(q_C),
     xlab = "true covariance", ylab = "estimated covariance")
segments(C_true[low], q_C[1, ], C_true[low], q_C[2, ], col = "grey60")
abline(0, 1, lty = 2)
```

![Estimated covariance between species from the two factors, with 95
percent intervals, against the true covariance for the 36 species pairs
and variances, with the identity
line](latent-factors_files/figure-html/omega-plot-1.png)

The printed `relative_error` is the Frobenius norm of the error against
that of the truth, and `covered` the share of the 36 entries whose 95%
interval contains the true value.

## Site scores

[`latent_factors()`](https://gillescolling.com/tulpa/reference/latent_factors.md)
summarises the scores of each site on each factor, from their posterior
given the data:

``` r

sc <- latent_factors(fit)
head(sc, 3)
#>   unit observation factor       mean        sd       q2.5        q50      q97.5
#> 1    1           1      1  0.4179858 0.4306144 -0.4298341  0.4051892  1.2322932
#> 2    2           2      1 -1.0765950 0.4603207 -1.9468314 -1.0785502 -0.1547959
#> 3    3           3      1  0.9274825 0.3595368  0.2637478  0.9318285  1.6479041
```

``` r

u1 <- sc[sc$factor == 1, "mean"]
plot(U[, 1], u1, pch = 19, xlab = "true score, factor 1",
     ylab = "estimated score")
abline(0, 1, lty = 2)
```

![Posterior mean scores of the 120 sites on factor 1 against the true
scores, with the identity
line](latent-factors_files/figure-html/score-plot-1.png)

``` r

cor(U[, 1], u1)
#> [1] 0.9415655
```

Factor 1 is anchored by species 1, whose true loading on factor 2 is
zero, so its scores compare directly with the simulated ones. Factor 2’s
scores are a rotation of the simulated second column whenever the true
loading matrix is not lower-triangular; compare $`\Lambda \Lambda^\top`$
in that case.

## Fitted values and prediction

[`fitted()`](https://rdrr.io/r/stats/fitted.values.html) returns each
row’s mean conditional on its site’s scores, and
[`predict()`](https://rdrr.io/r/stats/predict.html) the fixed-effect
part at new covariate values, for the species named in `newdata`:

``` r

head(cbind(y = d$y, fitted = fitted(fit)), 3)
#>      y   fitted
#> [1,] 4 3.002366
#> [2,] 2 1.145053
#> [3,] 7 4.756802
nd <- data.frame(species = c("sp1", "sp2"), x = 0)
predict(fit, newdata = nd, type = "response")
#> [1] 2.756560 1.476946
```

## Practical guidance

- **Compare $`\Lambda \Lambda^\top`$, not loadings, across fits.** The
  loadings depend on which species anchor the factors; the covariance
  does not.
- **Order the anchors.** Factor $`f`$ is anchored by the $`f`$-th level
  of the response column: choose species that respond to the factors and
  are not rare.
- **At most $`K`$ factors.** Factor $`f`$ does not load on the first
  $`f - 1`$ species, so the fit refuses more factors than species.
- **Lower-level door.** The same model is a list of `"lf"` blocks in the
  `prior` of
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md),
  one block per factor, for arms that are not rows of one long table.

## See also

- [Joint
  models](https://gillescolling.com/tulpa/articles/joint-models.md): two
  response arms sharing one spatial field through a copy scale.
- [`?latent_factor`](https://gillescolling.com/tulpa/reference/latent_factor.md),
  [`?latent_factors`](https://gillescolling.com/tulpa/reference/latent_factors.md)
  and the `factors` argument of
  [`?tulpa`](https://gillescolling.com/tulpa/reference/tulpa.md).
- [`?tulpa_nested_laplace_joint`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md):
  the arm specification and the list-of-blocks `prior`, including the
  `"lf"` block.
