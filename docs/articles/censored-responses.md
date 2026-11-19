# Censored, truncated and zero-inflated responses

A laboratory reports a concentration below its detection limit as “below
the limit”, with no number. A data set compiled from reports that kept
only values under a ceiling has no rows above it at all. This article
fits both kinds of data through a bound term on the left of the
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) formula,
`y | cens(lower, upper)` for censored values and `y | trunc(lb, ub)` for
truncated ones, and sets each against a Gaussian fit that treats the
recorded numbers as exact. It ends with counts that carry more zeros
than a Poisson model allows. Installation is covered in [Get
started](https://gillescolling.com/tulpa/articles/tulpa.md).

1.  **Simulate** log-concentrations with a site effect and known
    coefficients.
2.  **Censor** them at a detection limit and fit
    `y | cens(lower, upper)`.
3.  **Substitute** a value for the censored samples and fit a plain
    Gaussian model, for comparison.
4.  **Truncate** them at a ceiling and fit `y | trunc(ub = )`.
5.  **Repeat** both comparisons on five simulated data sets.
6.  **Add** a zero-inflation model to a count fit with `ziformula`.

## Example

``` r

library(tulpa)
```

### Simulating concentrations

The log-concentration of sample $`i`$ taken at site $`j`$ is

``` math
y^*_{ij} = \beta_0 + \beta_1 x_{ij} + u_j + \varepsilon_{ij}, \qquad
u_j \sim \mathrm{N}(0, \tau^2), \qquad
\varepsilon_{ij} \sim \mathrm{N}(0, \sigma^2),
```

where $`x_{ij}`$ is a covariate measured with each sample, such as the
distance to a road, $`u_j`$ is the effect of site $`j`$ shared by all
its samples, and $`\varepsilon_{ij}`$ is the measurement-level noise.
The truth is $`\beta_0 = 1`$, $`\beta_1 = 0.6`$, $`\tau = 0.4`$ and
$`\sigma = 0.5`$, with 15 samples at each of 40 sites. The value
$`y^*_{ij}`$ is what an ideal instrument would report; the sections
below hide part of it.

``` r

G <- 40L; per <- 15L; N <- G * per
set.seed(2)
d <- data.frame(site = factor(rep(seq_len(G), each = per)), x = rnorm(N))
u <- rnorm(G, 0, 0.4)
d$y_star <- 1 + 0.6 * d$x + u[as.integer(d$site)] + rnorm(N, 0, 0.5)
```

Every fit below carries the site effect as `(1 | site)` and leaves the
residual variance to be estimated.
[`VarCorr()`](https://gillescolling.com/tulpa/reference/VarCorr.md)
reports the site standard deviation $`\tau`$ and the residual standard
deviation $`\sigma`$ of a fit.

### Fitting values below a detection limit

The detection limit is 0.8 on the log scale. A sample below the limit is
known only to lie in $`(-\infty, 0.8]`$; a sample above it is recorded
exactly. `cens()` reads two bound columns: an `NA` bound leaves that
side open, and equal bounds record an exact value.

``` r

lod <- 0.8
below <- d$y_star < lod
d$y     <- ifelse(below, lod, d$y_star)
d$lower <- ifelse(below, NA, d$y_star)
d$upper <- ifelse(below, lod, d$y_star)
mean(below)
#> [1] 0.3733333
```

37% of the samples fall below the limit. For a censored row with bounds
$`(l_i, r_i]`$ and linear predictor $`\eta_i`$, the likelihood is the
probability that the latent value fell in the interval,

``` math
\Pr(l_i < y^*_i \le r_i) =
\Phi\!\left(\frac{r_i - \eta_i}{\sigma}\right) -
\Phi\!\left(\frac{l_i - \eta_i}{\sigma}\right),
```

with $`\Phi`$ the standard normal distribution function, and for an
exact row it is the Gaussian density. The likelihood reads only the
bounds. The response column `y` holds the recorded values, with the
censored rows at the limit.

``` r

ctl <- list(n_threads = 2L, progress = FALSE)
fit_cens <- tulpa(y | cens(lower, upper) ~ x + (1 | site), data = d,
                  family = "gaussian", control = ctl)
summary(fit_cens)
#> tulpa fit  (nested_laplace_joint, tier 2) 
#> Call: tulpa(formula = y | cens(lower, upper) ~ x + (1 | site), data = d, family = "gaussian", control = ctl)
#>   inference mode: structured 
#>   selected because: `cens()` response; the interval_gaussian kernel on the one-arm joint nested-Laplace door, the residual variance an outer-grid axis 
#>   family: interval_gaussian 
#>   n_obs: 600 
#> 
#> Fixed effects:
#>             estimate std.error  2.5 % 97.5 %
#> (Intercept)   1.0124   0.06891 0.8754 1.1468
#> x             0.6255   0.02692 0.5733 0.6789
#>   interval read: mixture_cdf 
#>   interval declined: NA 
#> 
#> Random effects:
#>       term        coef     sd    source
#> 1     site (Intercept) 0.3939 estimated
#> 2 Residual        <NA> 0.5004 estimated
#> 
#> Hyperparameters:
#>  parameter   mean      sd median  ci_lo  ci_hi
#>   b1.sigma 0.3981 0.05182 0.3939 0.3060 0.5168
#>      phi_y 0.2504 0.01925 0.2494 0.2152 0.2913
#> 
#> Reliability (iid draws):
#>   outer pareto_k -0.06378 (good); grid ESS 7.968 of 35 cells; inner skew not computed (no_finite_contribution)
#>   verdict: reliable (both layers good)
```

The random-effects table holds the site and residual standard
deviations. In the hyperparameter table `b1.sigma` is the site standard
deviation and `phi_y` the residual variance $`\sigma^2`$. The
coefficients, the site standard deviation and the residual variance have
intervals that contain their true values.

### Substituting a value for the censored samples

A common shortcut replaces each censored sample with half the detection
limit on the concentration scale, which is the limit minus $`\log 2`$ on
the log scale, and fits the result as exact. A third fit uses the values
with no limit at all. It is the fit the same samples would give without
censoring, and it separates what the censoring costs from the noise of
this one data set.

``` r

d$y_sub <- ifelse(below, lod - log(2), d$y_star)
```

``` r

fit_sub  <- tulpa(y_sub ~ x + (1 | site), data = d, family = "gaussian",
                  control = ctl)
fit_full <- tulpa(y_star ~ x + (1 | site), data = d, family = "gaussian",
                  control = ctl)
```

The table sets the three fits against the truth.

``` r

row_of <- function(fit) {
  vc <- VarCorr(fit)
  c(intercept = coef(fit)[[1]], slope = coef(fit)[[2]],
    site_sd = vc$sd[vc$term == "site"],
    resid_sd = vc$sd[vc$term == "Residual"])
}
tab_cens <- rbind(truth = c(1, 0.6, 0.4, 0.5),
                  no_limit = row_of(fit_full),
                  censored = row_of(fit_cens),
                  substituted = row_of(fit_sub))
round(tab_cens, 3)
#>             intercept slope site_sd resid_sd
#> truth           1.000 0.600   0.400    0.500
#> no_limit        1.018 0.615   0.394    0.491
#> censored        1.012 0.626   0.394    0.500
#> substituted     0.998 0.561   0.350    0.549
```

The censored fit returns a slope of 0.63 against 0.62 without a limit.
The substituted fit gives 0.56, and its residual standard deviation is
0.55 against the true 0.5.

### Fitting values truncated at a ceiling

Now the samples above 1.5 are missing from the data altogether, and
nothing records that they existed. Each remaining row is a draw from the
Gaussian density cut off at the ceiling $`c`$, so its density is the
Gaussian one divided by the mass kept below the ceiling,

``` math
f(y_i) = \frac{\phi\big((y_i - \eta_i)/\sigma\big) / \sigma}
              {\Phi\big((c - \eta_i)/\sigma\big)},
```

with $`\phi`$ the standard normal density. `trunc(ub = 1.5)` declares
the ceiling. A bound is a scalar, as here, or a column with one value
per row.

``` r

d_kept <- d[d$y_star <= 1.5, ]
c(kept = nrow(d_kept), dropped = N - nrow(d_kept))
#>    kept dropped 
#>     412     188
fit_trunc <- tulpa(y_star | trunc(ub = 1.5) ~ x + (1 | site), data = d_kept,
                   family = "gaussian", control = ctl)
fit_kept  <- tulpa(y_star ~ x + (1 | site), data = d_kept,
                   family = "gaussian", control = ctl)
```

``` r

tab_trunc <- rbind(truth = c(1, 0.6, 0.4, 0.5),
                   truncated = row_of(fit_trunc),
                   gaussian = row_of(fit_kept))
round(tab_trunc, 3)
#>           intercept slope site_sd resid_sd
#> truth         1.000 0.600   0.400    0.500
#> truncated     0.989 0.574   0.397    0.490
#> gaussian      0.799 0.434   0.304    0.419
```

The Gaussian fit on the kept rows reads an intercept of 0.8, a slope of
0.43 and a residual standard deviation of 0.42. The truncated fit
returns 0.99, 0.57 and 0.49, against the true 1, 0.6 and 0.5.

### Repeating the comparison on five data sets

One data set can land near the truth or away from it by chance. The loop
below simulates five more data sets with the same truth and fits the
censored, the substituted, the truncated and the kept-rows Gaussian
model to each, recording the slope and the residual standard deviation.

``` r

one_rep <- function(seed) {
  set.seed(seed)
  r <- data.frame(site = d$site, x = rnorm(N))
  u_r <- rnorm(G, 0, 0.4)
  r$y_star <- 1 + 0.6 * r$x + u_r[as.integer(r$site)] + rnorm(N, 0, 0.5)
  b <- r$y_star < lod
  r$y     <- ifelse(b, lod, r$y_star)
  r$lower <- ifelse(b, NA, r$y_star)
  r$upper <- ifelse(b, lod, r$y_star)
  r$y_sub <- ifelse(b, lod - log(2), r$y_star)
  k <- r[r$y_star <= 1.5, ]
  fit <- function(f, data) {
    tulpa(f, data = data, family = "gaussian", control = ctl)
  }
  fits <- list(
    censored      = fit(y | cens(lower, upper) ~ x + (1 | site), r),
    substituted   = fit(y_sub ~ x + (1 | site), r),
    truncated     = fit(y_star | trunc(ub = 1.5) ~ x + (1 | site), k),
    kept_gaussian = fit(y_star ~ x + (1 | site), k))
  t(sapply(fits, function(f) row_of(f)[c("slope", "resid_sd")]))
}
reps <- lapply(11:15, one_rep)
```

``` r

slope <- sapply(reps, function(r) r[, "slope"])
sd_hat <- sapply(reps, function(r) r[, "resid_sd"])
round(slope, 3)
#>                [,1]  [,2]  [,3]  [,4]  [,5]
#> censored      0.592 0.649 0.578 0.598 0.605
#> substituted   0.548 0.539 0.540 0.541 0.524
#> truncated     0.617 0.562 0.534 0.627 0.570
#> kept_gaussian 0.458 0.472 0.403 0.478 0.436
round(cbind(mean_slope = rowMeans(slope), mean_resid_sd = rowMeans(sd_hat)), 3)
#>               mean_slope mean_resid_sd
#> censored           0.604         0.506
#> substituted        0.538         0.545
#> truncated          0.582         0.495
#> kept_gaussian      0.449         0.430
```

Averaged over the five data sets, the censored and truncated fits return
slopes of 0.6 and 0.58 against the true 0.6. The substituted fit
averages 0.54 and the Gaussian fit on truncated data 0.45. The figure
shows the five estimates of each fit.

``` r

op <- par(mar = c(4, 8, 1, 1))
plot(slope, rep(1:4, 5), pch = rep(c(19, 1, 17, 2), 5),
     yaxt = "n", ylim = c(0.5, 4.5), xlab = "slope estimate", ylab = "")
axis(2, at = 1:4, labels = rownames(slope), las = 1)
abline(v = 0.6, lty = 2)
```

![Slope estimates of four fits on five simulated data sets, one row per
fit, with a dashed vertical line at the true slope
0.6](censored-responses_files/figure-html/replicate-plot-1.png)

``` r

par(op)
```

### Right censoring and lower truncation

A right-censored value, known only to exceed a limit $`r`$, is a row
with `lower = r` and `upper = NA`; one bound pair can mix left-censored,
right-censored, interval-censored and exact rows. Truncation from below
is `trunc(lb = )`, and both bounds can be given together. Here only the
samples at or above 0.5 are kept.

``` r

d_lo <- d[d$y_star >= 0.5, ]
fit_lo <- tulpa(y_star | trunc(lb = 0.5) ~ x + (1 | site), data = d_lo,
                family = "gaussian", control = ctl)
round(confint(fit_lo)[, 1:2], 3)
#>             2.5 % 97.5 %
#> (Intercept) 0.839  1.143
#> x           0.588  0.735
```

The fit keeps 449 of the 600 samples, and the table gives the 95%
intervals of its intercept and slope, whose true values are 1 and 0.6.

### Counts with excess zeros

Zero-inflation is a
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) argument
for the count families. With probability $`\pi`$ an observation is a
structural zero, and otherwise it is drawn from the count family;
`ziformula` gives the linear predictor of $`\operatorname{logit} \pi`$.
The counts below are Poisson with intercept 0.2 and slope 0.6, and a
structural-zero probability of $`\operatorname{logit}^{-1}(-0.5)`$,
about 0.38.

``` r

set.seed(3)
n <- 600L
xz <- rnorm(n)
structural <- rbinom(n, 1, plogis(-0.5))
dz <- data.frame(y = ifelse(structural == 1, 0,
                            rpois(n, exp(0.2 + 0.6 * xz))), x = xz)
mean(dz$y == 0)
#> [1] 0.56
```

``` r

fit_zi <- tulpa(y ~ x, data = dz, family = "poisson", ziformula = ~ 1,
                mode = "laplace")
fit_pois <- tulpa(y ~ x, data = dz, family = "poisson", mode = "laplace")
summary(fit_zi)
#> tulpa fit  (laplace, tier 2) 
#> Call: tulpa(formula = y ~ x, data = dz, family = "poisson", mode = "laplace", ziformula = ~1)
#>   inference mode: structured 
#>   selected because: User-specified backend: laplace 
#>   family: poisson 
#>   n_obs: 600 
#> 
#> Fixed effects:
#>                estimate std.error   2.5 %  97.5 %
#> (Intercept)      0.2242   0.06135  0.1040  0.3445
#> x                0.6045   0.04996  0.5066  0.7025
#> zi_(Intercept)  -0.4917   0.12982 -0.7462 -0.2373
#> 
#> Fixed-effect prior: normal(0, 2.5)
coef(fit_pois)
#> (Intercept)           x 
#>  -0.2316630   0.4752005
```

The zero-inflated fit reports the zero model’s intercept as
`zi_(Intercept)` on the logit scale, next to the count coefficients. The
Poisson fit reads the structural zeros as low counts: its intercept is
-0.23 against the true 0.2, and its slope 0.48 against 0.6.

## Limits of the bound terms

- Both take `family = "gaussian"`. A bound is a fixed number on the
  scale of the response, a scalar or one value per row. A censoring
  limit that is itself estimated is outside these terms.
- The residual variance is estimated when `phi` is left out. A supplied
  `phi` is held fixed, and it is the residual variance $`\sigma^2`$, as
  for a plain Gaussian response.
- In a model with several responses, a censored or truncated response is
  an arm of
  [`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
  with `family = "interval_gaussian"` and the fields `lower` and
  `upper`, or `family = "truncated_gaussian"` and `trunc_lower` and
  `trunc_upper`.
- Survival responses with a censoring indicator, such as right-censored
  event times under a Weibull model, are not engine families.

## Where next

- [Joint
  models](https://gillescolling.com/tulpa/articles/joint-models.md):
  several arms sharing one latent field, which a censored arm can join
  like any other.
- [Get started](https://gillescolling.com/tulpa/articles/tulpa.md): the
  [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) front
  door and its families.
- [`?tulpa`](https://gillescolling.com/tulpa/reference/tulpa.md): the
  `cens()` and [`trunc()`](https://rdrr.io/r/base/Round.html) response
  terms and `ziformula`.
