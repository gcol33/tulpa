# Validating calibration: simulation-based calibration

``` r

library(tulpa)
```

## What the instrument reads

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
scores one fit. It asks whether the machinery that produced that
posterior behaved: whether the chains mixed, whether the outer grid
covered the hyperparameter posterior, whether the inner Gaussian was a
reasonable shape for the latent conditional. Those are verdicts on the
procedure at one data set.

Calibration is a different question, and no single fit can answer it. An
inference algorithm is calibrated when its posteriors are
self-consistent across the whole generative model: simulate a truth,
simulate data at that truth, fit, and the truth should land at a
uniformly distributed position inside the posterior it produced. Repeat
that many times and those positions have a distribution you can test.

The usual way to check this is coverage: count how often a 95% interval
contains the truth. That reads one point of the marginal CDF, or two if
you also run a 50% interval. It cannot say whether a posterior is
biased, over-dispersed, under-dispersed or asymmetric, and it is weak
enough that two genuinely different reads of the same fit can score
identically on it.
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) reads the
whole CDF.

The measurement is the probability integral transform. For a scored
quantity whose true value is `theta_0` and whose reported posterior CDF
is `F`, take `u = F(theta_0)`. Under exact inference `u` is exactly
Uniform(0, 1), so the entire empirical CDF of the `u` values across
simulations is the test statistic (Talts et al. 2018).

## A model to validate

A balanced Gaussian random-intercept model: six regions of four
observations each, an intercept and one covariate, a random intercept
per region whose standard deviation `sigma` is drawn from a seven-point
grid.

The design is small on purpose. The `sigma` posterior then spreads
across four or five of the seven grid cells, which is the regime where
the outer-grid mixture posterior for a fixed effect is farthest from the
Gaussian matching its first two moments.

``` r

GRID <- exp(seq(log(0.2), log(1.5), length.out = 7))
PHI  <- 0.7
BETA <- c(-0.2, 0.7)

simulate_one <- function(seed) {
  set.seed(seed)
  sigma  <- GRID[sample.int(length(GRID), 1L)]
  region <- rep(seq_len(6L), each = 4L)
  X      <- cbind(1, rnorm(24L))
  u      <- rnorm(6L, 0, sigma)
  list(y      = as.numeric(X %*% BETA) + u[region] + rnorm(24L, 0, PHI),
       X      = X,
       region = region,
       theta  = c(beta1 = BETA[1], beta2 = BETA[2], sigma = sigma))
}

fit_one <- function(d, diagnose = FALSE) {
  tulpa_nested_laplace(
    y = d$y, n_trials = rep(1L, length(d$y)), X = d$X,
    prior = list(list(type = "iid", obs_idx = d$region,
                      n_units = max(d$region), sigma_grid = GRID)),
    family = "gaussian", phi = PHI,
    control = list(n_threads = 1L, keep_grid_hessians = TRUE,
                   auto_recenter = FALSE, progress = FALSE,
                   diagnose_k = diagnose, diagnose_skew = diagnose))
}
```

`simulate_one()` has to be a pure function of its seed, and it returns
`theta`, the named vector of true values the PIT is taken against.
`auto_recenter = FALSE` is not a speed knob: the argument that the
`sigma` PIT is uniform needs the fitted grid to be the same seven points
the truth was drawn from, and a recentred grid breaks it.

The single-fit reliability diagnostics are off inside the loop. Each one
costs its own batch of inner solves per fit, and the last section is
where they get read, on one fit rather than two hundred.

## Reporting a fit in a shape `sbc()` can read

A fitter callback returns a named list of **arms**, each a named list
over scored quantities, each entry one of the predictive shapes in
[`?sbc_predictive`](https://gillescolling.com/tulpa/reference/sbc_predictive.md).
Everything downstream dispatches on the shape’s `kind` tag, so a backend
reporting something new is one shape rather than a parallel scorer.

Two of them cover a nested-Laplace fit. What the outer grid defines for
a fixed effect is a Gaussian mixture over the cells, and
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
realizes it: each draw picks a cell by its weight, then samples that
cell’s inner Gaussian. [`coef()`](https://rdrr.io/r/stats/coef.html) and
[`vcov()`](https://rdrr.io/r/stats/vcov.html) report that same mixture’s
first two moments, so building an
[`sbc_normal()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
from them is the collapsed read of the identical posterior. For the
hyperparameter the grid defines a distribution on a finite support,
which is
[`sbc_discrete()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md).

``` r

arms <- function(d) {
  fit <- fit_one(d)
  m   <- coef(fit)
  se  <- sqrt(diag(vcov(fit)))
  D   <- tulpa_posterior_draws(fit, n = 2000)
  w   <- fit$weights / sum(fit$weights)

  list(
    mixture = list(
      beta1 = sbc_draws(D[, 1]),
      beta2 = sbc_draws(D[, 2]),
      sigma = sbc_discrete(as.numeric(fit$theta_grid), w)),
    collapsed = list(
      beta1 = sbc_normal(m[1], se[1]),
      beta2 = sbc_normal(m[2], se[2])),
    narrow = list(
      beta1 = sbc_normal(m[1], se[1] / 1.25),
      beta2 = sbc_normal(m[2], se[2] / 1.25)))
}
```

Every arm reads off **one solve per seed**, so an arm-to-arm difference
carries no fit-to-fit noise. `narrow` is a deliberately mis-scaled
posterior, the same moments with the standard deviation divided by 1.25.
A calibration harness that cannot fail is worth nothing, so the run
below has to put it outside the band.

## The prior-predictive experiment

``` r

res <- sbc("prior_predictive",
           simulator  = simulate_one,
           fitter     = arms,
           n_sim      = 200L,
           flat_prior = c("beta1", "beta2"))
res
#> Simulation-based calibration -- prior_predictive, 200 simulations
#>   band: 0.95 simultaneous (equal local levels, exact crossing probability)
#>   CRPS: proper posterior score
#>   proper prior: verified over 10 probed simulations; flat prior asserted for beta1, beta2
#> 
#>        arm quantity   n     ks inside p_unif inside_folded p_folded     crps
#>    mixture    beta1 200 0.0426   TRUE  0.746          TRUE   0.5120 0.173650
#>    mixture    beta2 200 0.0667   TRUE  0.202         FALSE   0.0379 0.095863
#>    mixture    sigma 200 0.0785   TRUE  0.285          TRUE   0.0986 0.106190
#>  collapsed    beta1 200 0.0541   TRUE  0.568          TRUE   0.1180 0.173570
#>  collapsed    beta2 200 0.0713   TRUE  0.397          TRUE   0.0915 0.095918
#>     narrow    beta1 200 0.0679   TRUE  0.335          TRUE   0.3150 0.173820
#>     narrow    beta2 200 0.0522   TRUE  0.569          TRUE   0.5030 0.095476
#>  crps_se
#>  0.01030
#>  0.00442
#>  0.00698
#>  0.01010
#>  0.00439
#>  0.01110
#>  0.00496
#> 
#> 7 of 7 (arm, quantity) reads inside the band; outside the band: mixture/beta2.
```

`flat_prior` is a guard doing its job. Ordinary SBC draws the truth from
the prior, so an improper prior has nothing to draw from, and the
nested-Laplace door puts no prior on the fixed effects.
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) probes the
simulator, sees that `beta1` and `beta2` do not move across simulations,
and refuses to score them unless the caller names them. Naming them
asserts that their flat prior leaves the PIT uniform by a structural
argument; the assertion is checked in both directions and travels on the
result in `res$premises`.

## Three reads of the same PIT sample

`ks` and `p_unif` read the raw PIT ECDF. `inside_folded` and `p_folded`
read `2 |u - 1/2|`, which is also uniform under correct inference. The
fold is where a symmetric error shows: a posterior that is too narrow
pushes PIT mass towards both ends at once, which cancels in the raw ECDF
and accumulates in the folded one. `crps` is the continuous ranked
probability score, closed form for a Gaussian mixture, so the nested
tier’s own posterior is scored with no Monte Carlo.

`inside` is the verdict against a **simultaneous** band. A pointwise
binomial band is not one: at n = 100, holding each order statistic
inside its own 95% interval holds all of them together only 44.71% of
the time. The band here is calibrated by bisection against the exact
crossing probability of the uniform order statistics, so an ECDF
excursion anywhere along the curve is a 0.05-level event.

Discrete quantities are randomized within their atom,
`u = F(theta^-) + V P(theta)`, so `sigma` on its seven-point grid and a
continuous coefficient share one uniform reference and one band. Reading
a rank against a continuous uniform is the classic silent SBC bug, and
it is not something you can opt into here.

## The proper score, paired

`summary(baseline = )` pairs the CRPS seed by seed against one arm.

``` r

summary(res, baseline = "mixture")
#> Simulation-based calibration -- prior_predictive, 200 simulations
#>   CRPS: proper posterior score
#> 
#>        arm quantity   n     ks inside p_unif inside_folded p_folded     crps
#>    mixture    beta1 200 0.0426   TRUE  0.746          TRUE   0.5120 0.173650
#>    mixture    beta2 200 0.0667   TRUE  0.202         FALSE   0.0379 0.095863
#>    mixture    sigma 200 0.0785   TRUE  0.285          TRUE   0.0986 0.106190
#>  collapsed    beta1 200 0.0541   TRUE  0.568          TRUE   0.1180 0.173570
#>  collapsed    beta2 200 0.0713   TRUE  0.397          TRUE   0.0915 0.095918
#>     narrow    beta1 200 0.0679   TRUE  0.335          TRUE   0.3150 0.173820
#>     narrow    beta2 200 0.0522   TRUE  0.569          TRUE   0.5030 0.095476
#>  crps_se
#>  0.01030
#>  0.00442
#>  0.00698
#>  0.01010
#>  0.00439
#>  0.01110
#>  0.00496
#> 
#> Paired CRPS against arm 'mixture' (negative delta = better)
#>  quantity       arm   n      delta     t worse_frac   p_sign
#>     beta1 collapsed 200 -7.729e-05 -0.16      0.620 0.000845
#>     beta1    narrow 200  1.715e-04  0.18      0.375 0.000499
#>     beta2 collapsed 200  5.578e-05  0.27      0.505 0.944000
#>     beta2    narrow 200 -3.865e-04 -0.61      0.385 0.001400
#>   delta with its t is the proper-score verdict; p_sign is a more
#>   powerful detector of a difference but is not itself proper.
```

`delta` with its `t` is the proper-score verdict; a negative `delta` is
the arm scoring better. `p_sign` is more powerful at detecting that two
arms differ at all, but the sign test is not a proper score, so it
cannot rank them.

Expect the mixture and collapsed arms to score close together. They
carry the same first two moments, and the CRPS integrates the whole
squared CDF difference, which those moments dominate. Where they
separate is on the ECDF reads above, which is why both instruments are
worth running.

The CRPS is a proper posterior score here only because the truth is
drawn afresh each simulation. Hold the truth fixed across seeds and the
CRPS-optimal forecast is a point mass at it, so a sharper wrong
posterior wins. That is enforced rather than documented: a fixed-truth
sweep is not offered as an experiment, and `summary(baseline = )`
refuses to rank one.

## The picture

``` r

plot(res, arm = c("mixture", "narrow"), quantity = "beta2")
```

![PIT ECDF difference from uniform against the simultaneous
band](sbc_files/figure-html/plot-raw-1.png)

``` r

plot(res, arm = c("mixture", "narrow"), quantity = "beta2", folded = TRUE)
```

![Folded PIT ECDF difference from uniform against the simultaneous
band](sbc_files/figure-html/plot-folded-1.png)

The plot draws the ECDF **difference** from uniform, so a calibrated
read is a flat line at zero inside the band. The under-dispersed arm
bows away from it, and the bow is larger folded than raw.

## Calibration conditional on an observed data set

The experiment above averages over the prior. A user fitting their own
data asks something narrower: is the inference reliable in the posterior
geometry *this* data set produces. The prior average can miss a defect
confined to a small region of parameter space, and it can flag one the
observed data rules out.

`experiment = "posterior"` answers the narrow question (Sailynoja et
al. 2026, Algorithm 2). Draw `theta'` from `pi(theta | y_obs)`, simulate
a replicate at `theta'`, and take the PIT under the **augmented**
posterior `pi(theta | y, y_obs)`. That is ordinary SBC with
`pi(theta | y_obs)` in the role of the prior, so the same band, the same
folded read and the same proper score all carry over, and it needs no
proper prior at all.

It takes six callbacks.

``` r

d_obs <- simulate_one(99L)

model <- list(
  data_obs = d_obs,

  fit = function(data) fit_one(data),

  draw_theta = function(fit, seed) {
    set.seed(seed)
    b <- tulpa_posterior_draws(fit, n = 1L)
    k <- attr(b, "cells")[1]
    c(beta1 = unname(b[1, 1]), beta2 = unname(b[1, 2]),
      sigma = as.numeric(fit$theta_grid)[k])
  },

  simulate = function(theta, seed) {
    set.seed(seed)
    region <- rep(seq_len(6L), each = 4L)
    X      <- cbind(1, rnorm(24L))
    u      <- rnorm(6L, 0, theta[["sigma"]])
    list(y = as.numeric(X %*% theta[c("beta1", "beta2")]) +
           u[region] + rnorm(24L, 0, PHI),
         X = X, region = region)
  },

  pool = function(obs, rep) list(
    y      = c(obs$y, rep$y),
    X      = rbind(obs$X, rep$X),
    region = as.integer(c(obs$region, rep$region + max(obs$region)))),

  arms = function(fit, data) {
    m  <- coef(fit)
    se <- sqrt(diag(vcov(fit)))
    D  <- tulpa_posterior_draws(fit, n = 2000)
    list(
      mixture = list(
        beta1 = sbc_draws(D[, 1]),
        beta2 = sbc_draws(D[, 2]),
        sigma = sbc_discrete(as.numeric(fit$theta_grid),
                             fit$weights / sum(fit$weights))),
      narrow = list(
        beta1 = sbc_normal(m[1], se[1] / 1.25),
        beta2 = sbc_normal(m[2], se[2] / 1.25)))
  },

  group_ids = function(data) data$region)
```

``` r

pres <- sbc("posterior", model = model, n_sim = 200L)
pres
#> Simulation-based calibration -- posterior, 200 simulations
#>   band: 0.95 simultaneous (equal local levels, exact crossing probability)
#>   CRPS: proper posterior score (updating prior = posterior at y_obs)
#>   pooling: verified; fresh groups: verified (disjoint group labels)
#> 
#>      arm quantity   n     ks inside  p_unif inside_folded p_folded     crps
#>  mixture    beta1 200 0.0474   TRUE 0.53500          TRUE 0.167000 0.086533
#>  mixture    beta2 200 0.0513   TRUE 0.77900          TRUE 0.470000 0.069730
#>  mixture    sigma 200 0.0729   TRUE 0.52100          TRUE 0.431000 0.060628
#>   narrow    beta1 200 0.0612  FALSE 0.01730         FALSE 0.001610 0.086355
#>   narrow    beta2 200 0.0612  FALSE 0.00395         FALSE 0.000447 0.069845
#>  crps_se
#>  0.00485
#>  0.00371
#>  0.00406
#>  0.00526
#>  0.00408
#> 
#> 3 of 5 (arm, quantity) reads inside the band; outside the band: narrow/beta1, narrow/beta2.
```

The driver hands `draw_theta` and `simulate` different seeds, so
`set.seed(seed)` at the top of each is the correct fixture. Giving both
the same seed makes the replicate’s noise a function of the truth, which
is not `p(y | theta')`, and it shows up as a non-uniform PIT with
nothing wrong in the inference under test.

## Two premises, and what checks them

Each premise silently turns the construction into something that is not
SBC, so each has a guard, and each guard’s conclusion travels on the
result.

``` r

str(pres$premises)
#> List of 2
#>  $ pooling     : chr "verified"
#>  $ fresh_groups: chr "verified (disjoint group labels)"
```

**The augmented posterior conditions on both data sets.** Fitting the
replicate alone is ordinary SBC under a hand-made prior.
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) refuses a
`pool()` returning no more than the replicate, or no more than the
observed data.

**The replicate is conditionally independent of `y_obs` given theta.**
The nested tier integrates the random effects out, so `theta` carries no
per-group value, and a replicate drawn on the *same* regions couples the
two data sets through the group effects theta does not describe.
[`simulate()`](https://rdrr.io/r/stats/simulate.html) above draws six
fresh regions and `pool()` offsets their labels past the observed ones.
Supply `group_ids` and the observable half of that is verified, the
labels being disjoint. Omit it and the result records the premise as
unverified rather than assumed. The other half, that the replicate’s
group effects came from the prior rather than conditionally on `y_obs`,
is not visible from outside the callback and is not claimed.

A third requirement is a property of `draw_theta` rather than a guard.
`theta'` has to be a **joint** draw from `pi(theta | y_obs)`. Sampling
`beta` from
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
and `sigma` independently from the grid weights gives the right two
marginals and the wrong joint, and the `sigma` read leaves the band when
you do it. Reading `attr(b, "cells")` takes the coefficient and the
hyperparameter from the same cell, which is what makes the draw joint.

## Where this sits beside the single-fit band

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
on a nested fit reports the outer Pareto-k-hat, the inner skewness
estimate `gamma_3` and the inner importance k-hat, combined into one
reliability band. That band runs on the fit you already have;
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) costs a few
hundred fits. So the band is what you read routinely, and this is what
you reach for when the answer has to hold up.

The cheap one is not a compressed version of the expensive one, and the
two disagree in both directions. Measured over fifteen configurations at
1000 simulations (`dev_notes/issue339/`): a Poisson configuration
carrying an outer k-hat of 0.196 with both inner scores in the `good`
band, the cleanest verdict the band can give, fails calibration at
`p = 2.3e-13` on its intercept. A binomial configuration carrying an
outer k-hat of 1.413, well past the 0.7 escalation threshold, passes at
`p = 0.17`.

So read the shipped band as a screen. A rejection from
[`sbc()`](https://gillescolling.com/tulpa/reference/sbc.md) is the
strong statement: on that same measurement the false-positive rate is
0.0117 against a nominal 0.05, and power reaches 80% at roughly a 10%
dispersion error or a 0.14-SD location bias, which makes a pass the
weaker one.

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
reads both at once when you hand it the calibration result.

``` r

fit <- fit_one(d_obs, diagnose = TRUE)
diagnostics(fit, sbc = res)
#> Nested-Laplace WHOLE-FIT reliability (i.i.d. draws)
#>   two layers: the outer hyperparameter-grid integration, and the inner Gaussian Laplace on the latent field
#>   outer PSIS pareto_k = 0.790 (unreliable); IS-ESS = 63.1
#>   outer grid quadrature ESS = 4.30 of 7 cells (max weight 0.309)
#>   note: outer grid holds material weight on a boundary node of b1.sigma (lower): that axis truncates its own marginal there, whether or not its mode sits on the node -- widen it and refit to see what is outside
#>   note: outer grid axis maximal at its own boundary on b1.sigma:lower (not moved: auto_recenter_disabled): the span does not contain that axis's mode, so its marginal is a truncated tail at any spacing -- widen or pin that axis and refit
#>   note: outer grid resolution could not be scored on b1.sigma (mode_at_edge): 0 of 1 axes carry a cell-width / posterior-SD ratio, so the grid is not established as resolved on the rest -- an axis reading `mode_at_edge` is one whose nodes do not contain its own posterior mode, which no spacing statement can be made about
#>   note: outer grid does not contain its own posterior mode on b1.sigma:lower: that axis's extreme node carries the modal mass, so its reported bound is an extrapolation off the end of the design -- widen the axis rather than adding nodes inside it
#>   inner Laplace max |gamma_3| = 0.000 (good), scored 2/2 latents
#>   inner Laplace importance pareto_k = 0.275 (good), min IS efficiency 1.0000, scored 2/2 latents
#>     the importance weights are uniform on every probed index: the inner
#>     Gaussian reproduces the conditional posterior over the sampled region,
#>     so the shape above describes no correction and is not banded
#>   whole-fit verdict: scoped: outer (hyperparameter) integration flagged
#>   calibration (SBC, prior_predictive): outside the band: mixture/beta2
#>   per-parameter columns: none.
#>     the fit carries no posterior draws, so the per-parameter mean / sd / ESS / rhat columns are empty; tulpa_posterior_draws(fit) samples the retained outer-grid mixture if a sample is wanted
```

## See also

- [`?sbc`](https://gillescolling.com/tulpa/reference/sbc.md) for the
  full callback contract, the guards, and what each column of the report
  is.
- [`?sbc_predictive`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  for the predictive shapes, and which one a backend should report.
- [`vignette("reliability-pareto-k")`](https://gillescolling.com/tulpa/articles/reliability-pareto-k.md)
  for the single-fit reliability band this screens against: the outer
  Pareto-k-hat, `gamma_3`, and the combined verdict.
- [`?tulpa_posterior_draws`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
  for the mixture sampler the fixed-effect arms are built on, and
  [`?tulpa_psis`](https://gillescolling.com/tulpa/reference/tulpa_psis.md)
  for the Pareto-smoothing core the k-hat uses.

## References

Talts, S., Betancourt, M., Simpson, D., Vehtari, A. and Gelman, A.
(2018). Validating Bayesian inference algorithms with simulation-based
calibration. arXiv:1804.06788.

Sailynoja, T., Schmitt, M., Buerkner, P.-C. and Vehtari, A. (2026).
Posterior SBC: simulation-based calibration checking conditional on
data. *Statistics and Computing* **36**, 78.
<doi:10.1007/s11222-026-10825-9>.

Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules,
prediction, and estimation. *Journal of the American Statistical
Association* **102**, 359-378.
