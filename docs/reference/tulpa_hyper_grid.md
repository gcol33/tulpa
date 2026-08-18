# Outer hyperparameter-grid integration with a user-supplied inner fit

Generic driver for nested-Laplace-style outer integration over a small
hyperparameter block. The user supplies per-axis specs (values +
optional log-prior + log-scale / bounds / refinable metadata) and an
`inner_fit` callback that, at every hyperparameter cell, returns the
inner log marginal and optionally the fixed-effect posterior mode +
marginal covariance. The driver builds the Cartesian outer grid,
normalises the log-marginals to integration weights, reports per-axis
posterior moments and weighted quantiles, and (when the inner fit
supplies them) law-of-total-covariance fixed-effect posterior + mixture
draws.

This factors the per-family outer-grid plumbing in
[`tulpa_nested_laplace()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
/
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
/
[`tulpa_re_cov_nested()`](https://gillescolling.com/tulpa/reference/tulpa_re_cov_nested.md)
into one callback-driven entry point: downstream consumers (occupancy /
N-mixture / cover hurdle families in tulpaObs, custom user families)
drop in their own per-cell inner fit and get the outer integration for
free.

## Usage

``` r
tulpa_hyper_grid(
  hyper_specs,
  inner_fit,
  combine = c("law_of_total_cov", "weighted_mean_only", "none"),
  n_draws = 2000L,
  seed = NULL,
  beta_names = NULL,
  control = list()
)
```

## Arguments

- hyper_specs:

  A list of axis specs. Each spec is either a
  [`hyper_axis_spec()`](https://gillescolling.com/tulpa/reference/hyper_axis_spec.md)
  object or a plain list with the same fields (the driver auto-wraps);
  see
  [`hyper_axis_spec()`](https://gillescolling.com/tulpa/reference/hyper_axis_spec.md).
  The outer grid is the Cartesian product of the per-axis grids; the
  joint log-prior is the sum of per-axis `log_prior` contributions (axes
  with `log_prior = NULL` are flat / improper).

- inner_fit:

  `function(hypers)` returning a list with:

  - `log_marginal` – scalar; the inner-solve log marginal at this cell.
    Non-finite values are mapped to `-Inf` (the cell gets zero weight).

  - `beta_mean` – numeric vector of fixed-effect posterior mean at the
    cell. Required when `combine != "none"`.

  - `beta_cov` – `p x p` numeric matrix of fixed-effect marginal
    covariance at the cell. Required when
    `combine = "law_of_total_cov"`. `hypers` is a named numeric vector
    with the current cell's axis values (names match
    `vapply(hyper_specs, `\[\[`, character(1), "name")`). Errors thrown
    by `inner_fit` are caught and treated as a failed cell
    (`log_marginal = -Inf`, no beta contribution).

- combine:

  How to pool per-cell fixed-effect posteriors into a single posterior
  over the betas. One of:

  - `"law_of_total_cov"` (default) – compute
    `E[Cov(beta | theta)] + Cov(E[beta | theta])` from the per-cell
    `(beta_mean, beta_cov)`; synthesise `n_draws` posterior draws by
    mixture sampling. Requires `beta_mean` and `beta_cov` per cell.

  - `"weighted_mean_only"` – pool only the per-cell `beta_mean` into the
    weighted posterior mean. `beta_cov` is ignored; no draws.

  - `"none"` – do not assemble a fixed-effect posterior. Only the
    hyperparameter posterior is returned. `beta_mean` / `beta_cov` from
    `inner_fit` are ignored if supplied.

- n_draws:

  Number of posterior draws of the fixed effects to synthesise from the
  cell mixture (default 2000). Used only when
  `combine = "law_of_total_cov"`. `0` disables draw synthesis (the
  law-of-total-cov mean and covariance are still returned).

- seed:

  Optional integer seed for the draw synthesis.

- beta_names:

  Optional character vector naming the fixed-effect coordinates. When
  `NULL` (default) the names are taken from the first successful cell's
  `beta_mean` (or `beta1`, `beta2`, ... if it is unnamed).

- control:

  Optional list of refinement / tuning knobs:

  - `adaptive_grid` (`FALSE`) – run the boundary / interior refinement
    pass on every axis whose spec has `refinable = TRUE`. New cells are
    appended along the refining axis paired with the boundary modal
    cell's other-axis values, carrying a marginal-scale calibration so
    they contribute on the right scale.

  - `adaptive_grid_edge_thresh` (`0.02`) – per-axis trigger threshold.

  - `adaptive_grid_max_passes` (`1L`) – cap on refinement passes.

  - `var_of_means_consistency` (`FALSE`) – run a post-integration
    consistency pass: for refinable axes whose joint-grid var-of-means
    undershoots the Laplace-at-mode SD by more than `tolerance`, append
    Laplace-guided slice points at
    `theta_mean +/- {0.7, 1.5} * theta_sd` pinned at the modal cell. One
    kernel call per axis.

  - `var_of_means_tolerance` (`0.7`) – consistency-pass trigger ratio.

## Value

A list of class `c("tulpa_hyper_grid", "tulpa_fit", "list")` with:

- `theta_grid` – numeric matrix `[n_cells x n_axes]` of outer-grid
  hyperparameter values; columns named after the axes.

- `theta_names` – character vector of axis names.

- `log_marginal` – numeric `[n_cells]`; per-cell log integrand
  `inner$log_marginal + log_prior(theta_cell)`. Non-finite / failed
  cells are `-Inf`.

- `log_prior` – numeric `[n_cells]`; per-cell log-prior contribution
  (the sum across axes of `axis$log_prior(theta_cell[axis])`). `0` when
  all axes have `log_prior = NULL`.

- `weights` – numeric `[n_cells]` summing to 1 (or all `NA` with a
  warning when no cell carries finite mass).

- `theta_mean`, `theta_sd` – named numeric vectors; weighted posterior
  mean and SD per axis. SDs refit via the 3-point Laplace-at-mode
  parabola where possible.

- `theta_median`, `theta_ci_lo`, `theta_ci_hi` – named numeric vectors;
  weighted-quantile median and 2.5 / 97.5\\ (the recommended summary for
  right-skewed scale-like axes).

- `beta`, `beta_cov`, `draws` – fixed-effect posterior, present per
  `combine`:

  - `combine = "law_of_total_cov"`: weighted mean, total covariance
    (`E[V] + V[E]`), `[n_draws x p]` mixture draws.

  - `combine = "weighted_mean_only"`: weighted mean only; `beta_cov` and
    `draws` are `NULL`.

  - `combine = "none"`: all three are `NULL`.

- `means`, `param_names`, `process_info`, `n_samples`, `n_params`, `N` –
  the `tulpa_fit` accessor surface, populated when a fixed-effect
  posterior is assembled.

- `hyper_specs` – echoed list of `hyper_axis_spec` objects (the
  normalised form).

- `combine`, `n_failed`, `n_grid`, `refining_axis` – diagnostic fields.

- `adaptive_grid_info`, `var_of_means_consistency_info` – present when
  the corresponding refinement pass fired.

## See also

[`hyper_axis_spec()`](https://gillescolling.com/tulpa/reference/hyper_axis_spec.md)
for the axis-spec constructor;
[`tulpa_nested_laplace_joint()`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace_joint.md)
for the family-specific outer-grid driver that this helper generalises.

## Examples

``` r
if (FALSE) { # \dontrun{
# Integrate an inner fit over a hyperparameter grid: inner_fit(theta) returns
# a per-cell fit and hyper_specs names the axes. tulpa_nested_laplace() is the
# packaged driver built on this.
res <- tulpa_hyper_grid(hyper_specs, inner_fit)
} # }
```
