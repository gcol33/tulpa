# Approximation-reliability diagnostics for a deterministic nested-Laplace fit

**\[deprecated\]**

Use
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md),
which returns this table for any fit whose draws are an i.i.d.
approximation sample. The sections below document that table; they
remain the reference for its columns and attributes.

Per-parameter reliability diagnostics for a fit whose posterior draws
are i.i.d. samples from a deterministic approximation (the
nested-Laplace grid-mixture posterior `sum_k w_k N(mode_k, V_k)`), where
the between-chain Gelman-Rubin Rhat that
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
reports for a chain fit does not apply. This is the accessor that plays
Rhat's role for the deterministic engine: it answers "did the
approximation work", not "did the chains mix".

The headline is a Pareto-smoothed importance-sampling (PSIS) reliability
diagnostic for the OUTER hyperparameter-grid integration. The nested
integrator scores its hyperparameter grid against the exact
inner-Laplace marginal posterior with a generalized-Pareto fit to the
upper tail of the importance ratios
`log p_target(theta) - log q_proposal(theta)` (see
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md));
the resulting tail-shape `pareto_k` is the "did the outer integration
work" number – `k-hat < 0.5` good, `0.5-0.7` usable, `>= 0.7` unreliable
(Vehtari et al. 2024; Yao et al. 2018). It is computed at fit time and
read back here; a fit that did not run the diagnostic, or whose grid
proposal degenerated, reports it as `NA` and is assessed on the grid
quadrature reliability instead.

`pareto_k` scores the outer integration only. A high `pareto_k` on a fit
whose grid quadrature is healthy (`ess_grid` well above 1, largest cell
weight modest) flags outer-integration (CI-width) calibration in the
right-skewed hyperparameter tail and does not by itself invalidate the
point estimates, which the grid quadrature governs.

`outer_regime` qualifies what a high `pareto_k` means, and is the reason
a bare threshold on `pareto_k` is not a reliability verdict. A sharp
hyperparameter posterior collapses the grid onto ~1 cell (`ess_grid`
near 1); the outer integration has then degenerated to a point
evaluation at the modal hyperparameter, so `pareto_k` is scoring how
well a Gaussian at that mode stands in for the hyperparameter marginal,
not how well a grid integrated it. Where the dominant cell is INTERIOR
to the grid the collapse is benign – the grid bracketed the mode, the
estimate is empirical Bayes there, and only integrated hyperparameter
uncertainty is missing. Where it sits at a grid BOUNDARY the grid may
simply be too narrow: `grid_edge_axes` / `grid_edge_sides` name the axes
to widen. On a fit whose `pareto_k` cleared the good band, the outer
diagnostic also fits a skew-normal proposal and reports the marginal's
estimated skewness as `outer_skew_max`, so an inflated k-hat that was
purely the symmetric proposal's mismatch with a skewed
variance-component marginal is both corrected and explained. A
skew-normal has Gaussian tails, so this can never mask a genuinely
heavy-tailed target.

The grid quadrature reliability – the effective sample size
`ess_grid = 1 / sum(w_k^2)` of the outer integration weights and the
largest single cell weight – is always computed from the stored grid: a
grid that collapses onto one cell (`ess_grid` near 1) integrates no
hyperparameter uncertainty, while a spread grid does.

A SEPARATE layer – the inner Gaussian Laplace approximation to the
latent-field conditional posterior `pi(x | theta, y)`, which `pareto_k`
does not cover – is scored by `inner_skew`: the leading-order Edgeworth
skewness estimate `gamma_3` (Rue, Martino & Chopin 2009 Sec 3.2.3) at
the fitted MAP grid cell, computed when `control$diagnose_skew = TRUE`
(the default) on the fitting call. Reading a high `pareto_k` alone as
"the fit is broken" conflates the two layers: an occu_cover batch
flagged 42/78 species "unreliable" on outer k-hat alone when their point
estimates, governed by the healthy inner layer, were fine – the
`reliability` attribute is the combined verdict that names which layer
degrades, if either does.

Each parameter row also carries the rank-normalized split-Rhat and bulk
/ tail effective sample size of the draws (Vehtari et al. 2021). On
i.i.d. draws these sit at `~1.00` and `~n_draws` by construction; they
are reported, clearly as i.i.d.-draw Monte-Carlo diagnostics and not
chain mixing, to document that the reported posterior summaries are not
Monte-Carlo-limited.

A posterior sample is what those per-parameter rows are computed from,
and nothing else here needs one: every reliability quantity above is
read off the fit. So a fit that carries no draws – a default
single-block nested-Laplace fit, whose posterior is the retained
outer-grid mixture rather than a sample – reports the full band with an
empty per-parameter body and `n_draws = NA`, and records why in the
`param_table_declined` attribute.
[`tulpa_posterior_draws()`](https://gillescolling.com/tulpa/reference/tulpa_posterior_draws.md)
samples that mixture where the rows are wanted. The one case that still
returns `NULL` is a fit with neither draws nor any reliability quantity,
such as a plain Laplace fit with no outer grid to score.

## Usage

``` r
laplace_diagnostics(fit, pars = NULL)
```

## Arguments

- fit:

  A `tulpa_fit` (or subclass, e.g. a `tobs_fit`) whose draws are an
  i.i.d. approximation sample (`$draws_kind == "iid"`).

- pars:

  Optional character vector of parameter names to restrict to.

## Value

A data frame with one row per parameter – `parameter`, `mean`, `sd`,
`ess_bulk`, `ess_tail`, `rhat` – carrying attributes:

- `pareto_k`:

  the outer PSIS reliability k-hat (`NA` if not computed).

- `pareto_k_band`:

  `"good"` / `"ok"` / `"unreliable"` / `NA`.

- `pareto_k_declined`, `pareto_k_declined_note`:

  when `pareto_k` is `NA`, WHY: `"not_requested"`, `"not_applicable"`,
  `"unguessable_axis"` (naming the axis – a permanent limitation of that
  family, so read `ess_grid` instead), `"draws_too_few"`,
  `"grid_too_small"`, `"no_varying_axis"`, `"degenerate_proposal"`, or
  `"internal_inconsistency"` (an engine bug worth reporting), plus a
  one-line reading of it.

- `pareto_k_is_ess`:

  importance-sampling ESS on the smoothed weights.

- `ess_grid`, `n_grid`, `rel_ess_grid`, `max_weight`:

  grid quadrature reliability.

- `outer_regime`:

  `"spread"` / `"collapsed_interior"` / `"collapsed_edge"` – whether the
  outer grid integrated hyperparameter uncertainty at all, and if not
  whether its dominant cell is interior (empirical Bayes at the mode:
  point estimates sound, hyperparameter uncertainty not integrated) or
  against a grid boundary (widen it).

- `grid_edge_axes`, `grid_edge_sides`:

  for an edge collapse, the axes the dominant cell sits against and on
  which side.

- `outer_skew_max`:

  largest estimated \|skewness\| of the hyperparameter marginal,
  computed only when the k-hat triggered the skew-normal proposal rescue
  (`NA` means the Gaussian proposal already fit, not "symmetric and
  unchecked").

- `outer_regime_note`:

  a one-line reading of a collapsed regime, or absent on a spread grid.

- `grid_railed_axes`:

  outer axes whose OWN marginal is maximal at one of their own
  endpoints, as `axis:side` – the span does not contain that axis's
  posterior mode, so its marginal is a truncated tail at any spacing.
  Reported whether or not the engine was allowed to, able to, or built
  to move the axis.

- `grid_placement`, `grid_recentred_axes`, `grid_placement_declined`,
  `grid_placement_note`:

  whether the outer grid was re-centred, on which axes, and – when it
  was not – why.

- `scope`:

  the outer diagnostic's scope string.

- `inner_skew_max`:

  the largest `|gamma_3|` among the scored latent indices (`NA` if
  `control$diagnose_skew = FALSE` or nothing scored).

- `inner_skew_band`:

  `"good"` / `"ok"` / `"unreliable"` / `NA`, banded on `inner_skew_max`
  by the general skewness-magnitude convention (Bulmer 1979) – not a
  Rue-Martino-Chopin-specific cutoff.

- `inner_skew_scored`, `inner_skew_probed`:

  how many of the probed latent indices returned a finite `gamma_3` vs
  how many were probed.

- `inner_skew_declined`, `inner_skew_arms_declined`,
  `inner_skew_declined_note`:

  when nothing was scored, WHY: `"coupled_arm"` (STRUCTURAL – the
  coupled arms have neither a per-observation sum nor a cell
  third-derivative tensor to read, so the outer k-hat is the only
  reliability number this fit has), `"curvature3_unavailable"`,
  `"no_finite_contribution"`, `"no_probe_indices"`, `"not_requested"`,
  `"backend_unsupported"`, `"solve_failed"`, or `"not_converged"` (the
  probe re-solve stopped short of a mode, so neither inner score has a
  point to read); the arms (1-based) a joint fit had no oracle for,
  which is also set on a PARTIALLY scored fit; and a one-line reading.

- `inner_pareto_k`, `inner_pareto_k_band`:

  the inner-Laplace importance k-hat over the probed subspace, and its
  band on the same convention as the outer k-hat. Available wherever a
  mode was found, including a fit `gamma_3` cannot score.

- `inner_pareto_k_rel_ess`, `inner_pareto_k_is_ess`:

  the smallest realized importance efficiency and effective sample size
  across the probed indices – how much correcting the inner Gaussian
  actually needs, which is what makes the scale-free shape above
  readable.

- `inner_pareto_k_uniform`:

  `TRUE` when no probed index carried a material correction, i.e. the
  inner Gaussian reproduces the conditional posterior over the sampled
  region.

- `inner_pareto_k_scored`, `inner_pareto_k_probed`:

  how many probed indices returned a finite k-hat vs how many were
  probed.

- `inner_pareto_k_declined`, `inner_pareto_k_declined_note`:

  when it is `NA`, WHY, from the same closed vocabulary the outer k-hat
  uses.

- `reliability`:

  the combined whole-fit verdict: `"reliable"` only when both layers are
  good; otherwise names which layer is scoped or flags both as
  unreliable. The inner layer enters through the worse of its two
  scores, so a fit whose cubic term declined is still assessed.

and a trailing `summary` attribute (a one-row data frame of the headline
numbers) for printing.

## Scope

The PSIS `pareto_k` diagnoses the OUTER (hyperparameter) integration:
whether the Gaussian-proposal-over-grid approximation of the marginal
hyperparameter posterior `p(theta | data)` can be importance-corrected
to the exact inner marginal. This is the dominant approximation in
nested Laplace and the one with an exactly evaluable target. A full
latent-space PSIS against the exact joint posterior `pi(x)` is not
computed: the latent prior marginal
`p(x) = integral p(x | theta) p(theta) dtheta` has no closed form, and
for the marginalized-occupancy / cover-hurdle likelihoods the exact
joint density is evaluable only inside the C++ kernel, so a stored fit
cannot reconstruct it. The grid quadrature reliability is the
complementary stored-fit number.

The inner layer carries a SECOND score, `inner_pareto_k`, which needs no
likelihood derivative at all and therefore answers where `inner_skew`
declines. The inner Gaussian at the fitted hyperparameter is an
importance proposal for the exact conditional posterior, and the joint
density is the target, so PSIS on that ratio scores the inner
approximation directly. It is computed on the same probed indices along
the same conditional-mean curve, one dimension per index, since
importance sampling degrades with dimension and a k-hat over the whole
latent field would report `n_x` rather than the approximation. A Pareto
shape index is scale-free, so it is banded only on indices whose
realized importance efficiency shows a correction worth describing;
`inner_pareto_k_uniform` records that none did, which is what a
well-approximated inner layer looks like.

`inner_skew` diagnoses the INNER (latent-field) Laplace: whether the
Gaussian approximation to `pi(x_i | theta, y)` is itself a good fit, at
each scored latent index `i`. `gamma_3` is exact for a gaussian-family
coefficient (the log-likelihood is exactly quadratic in eta) and
declines to `NA` – never a silently-wrong `0` ("perfectly Gaussian") –
whenever no per-observation third-derivative oracle is available: a
coupled multi-process likelihood (e.g. tulpaObs's `occu_cover`, whose
arms combine non-separably through a `CellCouplingSpec`) or a family
with no registered third derivative. Only the requested latent indices
are scored (every arm's fixed-effects coefficients by default – see
`control$skew_idx`), since each index costs one extra linear solve; the
full latent field is not scored by default on a large spatial field.

## References

Vehtari, Simpson, Gelman, Yao & Gabry (2024). Pareto smoothed importance
sampling. *JMLR* 25(72):1-58.

Yao, Vehtari, Simpson & Gelman (2018). Yes, but did it work?: Evaluating
variational inference. *ICML*, PMLR 80:5581-5590.

Vehtari, Gelman, Simpson, Carpenter & Burkner (2021).
Rank-normalization, folding, and localization: an improved Rhat for
assessing convergence of MCMC. *Bayesian Analysis* 16(2):667-718.

Rue, Martino & Chopin (2009). Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations.
*JRSS-B* 71(2):319-392.

## See also

[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
(the front door, which returns this table for i.i.d. fits),
[`tulpa_psis()`](https://gillescolling.com/tulpa/reference/tulpa_psis.md).

## Examples

``` r
# \donttest{
set.seed(1)
n <- 200L; x <- rnorm(n)
y <- rbinom(n, 1, plogis(-0.2 + 0.6 * x))
# `mode = "laplace"` returns a mode + covariance and carries no draws; a
# sampled deterministic backend is what this table describes.
fit <- tulpa(y ~ x, data.frame(y = y, x = x), family = "binomial",
             mode = "smc")
diagnostics(fit)
#> Nested-Laplace OUTER-integration reliability (i.i.d. draws)
#>   scope: the outer hyperparameter-grid integration; the latent-field
#>   Laplace is a separate, unscored layer.
#>   outer PSIS pareto_k = NA:
#>     this fit's outer grid is not one the k-hat is defined on (this backend has no outer hyperparameter grid)
#>   whole-fit verdict: not computed (outer integration not assessed; inner Laplace not assessed)
#>   2 parameters, 1000 draws; per-parameter rhat / ESS below are
#>   i.i.d.-draw Monte-Carlo diagnostics (not chain mixing).
#> 
#>     parameter       mean        sd ess_bulk ess_tail     rhat
#> 1 (Intercept) -0.3239187 0.1806756 19.74593 40.14727 1.107996
#> 2           x  0.4047345 0.1642653 41.91750 18.53311 1.041535
# }
```
