# Simulation-based calibration

Scores whether an inference algorithm's posterior is CALIBRATED, by
reading the whole marginal CDF rather than one or two nominal levels.
Draw a truth from the distribution the fit updates, simulate a data set
at that truth, fit, and take the probability integral transform (PIT) of
the truth under the reported posterior. Under exact inference those PIT
values are exactly Uniform(0, 1), so the entire ECDF is the measurement
(Talts et al. 2018).

## Usage

``` r
sbc(object, ...)

# Default S3 method
sbc(object, ...)

# S3 method for class 'character'
sbc(
  object = c("prior_predictive", "posterior"),
  simulator = NULL,
  fitter = NULL,
  model = NULL,
  n_sim = 100L,
  quantities = NULL,
  flat_prior = character(),
  level = 0.95,
  seed = 0L,
  n_ref = NULL,
  control = list(),
  ...
)

# S3 method for class 'sbc'
summary(object, baseline = NULL, ...)

# S3 method for class 'sbc'
plot(x, arm = NULL, quantity = NULL, folded = FALSE, ...)
```

## Arguments

- object:

  An `sbc` result.

- ...:

  Ignored.

- simulator, fitter:

  The prior-predictive callbacks; see the contract above. Required for
  `experiment = "prior_predictive"`.

- model:

  The list of posterior-SBC callbacks. Required for
  `experiment = "posterior"`.

- n_sim:

  Number of simulations.

- quantities:

  Optional character vector restricting which quantities are scored.
  Default scores every quantity every arm reports.

- flat_prior:

  Character vector of scored quantities held FIXED under a flat prior,
  admitted by a structural argument the caller is asserting.
  Prior-predictive only.

- level:

  Simultaneous band level.

- seed:

  Offset added to the simulation index, so simulation `s` runs the
  callbacks at `seed + s`.

- n_ref:

  Optional; when supplied it is passed to `fitter` (or to `model$arms`)
  as `n_ref`, the number of reference values a rank predictive is formed
  against. The callback must accept it.

- control:

  List of knobs: `progress` (default `FALSE`) and `rand_seed`, the
  pinned stream the within-atom randomizing uniforms come from (default
  the driver's own, so a result is reproducible and the fits are not
  perturbed by asking for the diagnostic).

- baseline:

  Optional arm name. When given, the summary also carries the PAIRED
  CRPS differences of every other arm against it, seed by seed. A
  negative `delta` is the arm scoring better. This is refused on an
  experiment where the CRPS is not a proper posterior score.

- x:

  An `sbc` result.

- arm, quantity:

  Optional character vectors selecting which panels to draw. Default
  draws every (arm, quantity).

- folded:

  Plot the folded PIT instead of the raw one.

## Value

An object of class `sbc`:

- `pit`:

  one row per (simulation, arm, quantity): the truth, the raw and folded
  PIT, the CRPS and the predictive kind.

- `report`:

  one row per (arm, quantity): the KS distance, whether the ECDF stayed
  inside the simultaneous band, the exact uniformity p-value, the same
  three folded, and the mean CRPS with its standard error.

- `bands`:

  the calibrated simultaneous band, by sample size.

- `premises`:

  what each guard concluded.

- `crps_role`:

  the role the CRPS column has under this experiment.

## The two experiments

- `"prior_predictive"`:

  ordinary SBC. `theta ~ p(theta)`, `y ~ p(y | theta)`, fit, PIT. It
  reports self-consistency AVERAGED over the whole generative
  distribution.

- `"posterior"`:

  calibration CONDITIONAL on an observed data set (Sailynoja et al.
  2026, Algorithm 2). `theta' ~ pi(theta | y_obs)`,
  `y ~ pi(y | theta')`, and the PIT is taken under the AUGMENTED
  posterior `pi(theta | y, y_obs)`. That is ordinary SBC with
  `pi(theta | y_obs)` in the role of the prior, so the same band, the
  same folded read and the same proper score all carry over – and it
  needs no proper prior.

A fixed-truth sweep is not an SBC experiment and is not offered here:
its PIT is not uniform under correct inference and its CRPS is a
descriptive loss, not a proper posterior score.

## What is reported

Three reads of the same PIT sample, per (arm, quantity):

- raw:

  the PIT ECDF against an exact SIMULTANEOUS band. A pointwise binomial
  band is not simultaneous – at n = 100, holding each order statistic at
  95 percent holds all of them together at 0.4471 – so the band here is
  calibrated by bisection against the exact crossing probability of the
  uniform order statistics.

- folded:

  `2 |u - 1/2|`, also uniform, and where a symmetric over- or
  under-dispersion shows after cancelling in the raw ECDF.

- CRPS:

  the strictly proper score, closed form for the nested tier's own
  Gaussian mixture, paired seed by seed through
  `summary(x, baseline = )`.

Every discrete PIT (a rank, a grid axis, a draw set) is randomized
within its atom, `u = F(theta^-) + V P(theta)`, so one uniform reference
and one band serve every quantity. Reading `rank / n_ref` against a
continuous uniform is the classic silent SBC bug.

## The callback contract

For `experiment = "prior_predictive"`:

- `simulator(seed)`:

  returns a list carrying `theta`, a named numeric vector of the true
  values of the scored quantities, plus whatever the fitter needs. It
  must be a pure function of its seed.

- `fitter(d)`:

  returns a named list of ARMS, each a named list over quantities, each
  entry a predictive built by
  [`sbc_mixture()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
  and its siblings. Several arms read off one solve per seed, so an
  arm-to-arm difference carries no fit-to-fit noise.

For `experiment = "posterior"`, `model` is a list of six callbacks –
`data_obs`, `fit(data)`, `draw_theta(fit, seed)`,
`simulate(theta, seed)`, `pool(data_obs, replicate)`, `arms(fit, data)`
– plus an optional `group_ids(data)` used to verify the fresh-groups
premise below. The driver hands `draw_theta` and `simulate` DIFFERENT
seeds, so `set.seed(seed)` at the top of each is the correct fixture;
sharing one makes the replicate's noise a function of the truth, which
is not `p(y | theta')`.

## Two guards

The prior-predictive experiment draws the truth from the prior, so an
IMPROPER prior cannot be used – and the nested-Laplace door puts no
prior on the fixed effects. A scored quantity whose truth does not move
across simulations was not drawn from a proper prior, and `sbc()` errors
on it before spending the fits, pointing at `experiment = "posterior"`.
A location parameter whose flat prior leaves the PIT uniform by a
structural argument is admitted through `flat_prior`, which is itself
checked and travels on the result.

The posterior experiment rests on two premises, each of which silently
invalidates the result when broken. The augmented posterior must
condition on BOTH data sets, so a `pool()` returning no more than the
replicate (or no more than the observed data) is refused. And the
replicate must be conditionally independent of the observed data given
theta, which for a hierarchical model whose group effects are integrated
out means FRESH groups. Supply `group_ids` and the observable half of
that is verified – the group LABELS are disjoint – and omit it and the
result records the premise as unverified rather than assumed. The other
half, that the simulator drew those groups' effects from the prior
rather than conditionally on the observed data, is not visible from
outside the callback and is not claimed.

## References

Talts, Betancourt, Simpson, Vehtari & Gelman (2018). Validating Bayesian
inference algorithms with simulation-based calibration.
arXiv:1804.06788.

Sailynoja, Schmitt, Buerkner & Vehtari (2026). Posterior SBC:
simulation-based calibration checking conditional on data. *Statistics
and Computing* 36:78.
[doi:10.1007/s11222-026-10825-9](https://doi.org/10.1007/s11222-026-10825-9)

Grimit, Gneiting, Berrocal & Johnson (2006). The continuous ranked
probability score for circular variables and its application to
mesoscale forecast ensemble verification. *QJRMS* 132(621C):2925-2942.

## See also

[`sbc_mixture()`](https://gillescolling.com/tulpa/reference/sbc_predictive.md)
for the predictive shapes a fitter reports,
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
for the single-fit reliability band that screens what this measures,
[`pit_residuals()`](https://gillescolling.com/tulpa/reference/pit_residuals.md)
for single-fit posterior-predictive PIT residuals (a different
quantity).

## Examples

``` r
# A conjugate normal-normal model, whose posterior is exact, so its PIT must
# be uniform: the harness scoring itself.
sim <- function(seed) {
  set.seed(seed)
  mu <- rnorm(1)
  list(y = rnorm(10L, mu, 1), theta = c(mu = mu))
}
fitter <- function(d) {
  v <- 1 / (1 + length(d$y))
  list(exact  = list(mu = sbc_normal(v * sum(d$y), sqrt(v))),
       narrow = list(mu = sbc_normal(v * sum(d$y), sqrt(v) / 2)))
}
res <- sbc("prior_predictive", simulator = sim, fitter = fitter, n_sim = 60L)
res
summary(res, baseline = "exact")
```
