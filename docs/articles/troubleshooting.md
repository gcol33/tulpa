# Troubleshooting

Each entry below starts from a situation, shows the message tulpa prints
in that situation, and ends on a corrected call that runs. The entries
are grouped by the stage at which the message appears: preparing the
input, setting priors and the dispersion, fitting, and reading how
reliable a fit is. The terms used here (control knob, tier, outer grid,
Pareto k) are defined in
[Concepts](https://gillescolling.com/tulpa/articles/concepts.md).
Installation is covered in [Get
started](https://gillescolling.com/tulpa/articles/tulpa.md).

``` r

library(tulpa)
```

## Example data

Every entry uses the data below. `d` holds 300 Gaussian measurements on
12 groups `g`, simulated from an intercept of 0.5, a slope of 1.2 on
`x`, group offsets with standard deviation 0.6 and residual standard
deviation 0.8. `db` holds a binary response on the same `x` and `g`, and
`dc` a count of successes out of 5 trials. `ds` places binary
observations in 10 regions on a ring, with the adjacency matrix `W`, and
adds a time index `time` and a three-level panel factor `site`.

``` r

n <- 300
x <- rnorm(n)
g <- factor(sample(1:12, n, replace = TRUE))
u <- rnorm(12, sd = 0.6)
d  <- data.frame(y = 0.5 + 1.2 * x + u[g] + rnorm(n, sd = 0.8), x = x, g = g)
db <- data.frame(y = rbinom(n, 1, plogis(-0.3 + x + u[g])), x = x, g = g)
dc <- data.frame(y = rbinom(n, 5, plogis(-0.3 + x)), x = x)

K <- 10
W <- matrix(0, K, K)
for (i in 1:K) { j <- if (i < K) i + 1 else 1; W[i, j] <- W[j, i] <- 1 }
ds <- data.frame(y = rbinom(n, 1, plogis(-0.2 + 0.7 * x)), x = x,
                 region = sample(1:K, n, replace = TRUE),
                 time = sample(1:15, n, replace = TRUE),
                 site = factor(sample(1:3, n, replace = TRUE)))
```

## Input

### A tuning setting passed as an argument

`n_threads`, `n_iter`, `max_iter` and the other settings that change how
the computation runs are not arguments of
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md). Given
as one, the fit stops before any work and says where the setting goes.

``` r

tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
      phi = 0.8^2, n_threads = 2)
#> Error:
#> ! unknown argument(s) to tulpa(): n_threads. Tuning knobs go in `control = list()`.
```

Move it into `control`.

``` r

fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
             phi = 0.8^2, control = list(n_threads = 2))
coef(fit)
#> (Intercept)           x 
#>   0.3484146   1.2811466
```

### A misspelled control knob

A name in `control` that no backend reads is refused, and the message
lists every name
[`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md) accepts.

``` r

tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace",
      phi = 0.8^2, control = list(n_thread = 2))
#> Error:
#> ! Unknown control knob(s) for tulpa(): 'n_thread'.
#> Allowed: adapt_delta, adaptive_grid, adaptive_grid_cutoff, adaptive_grid_edge_thresh, adaptive_grid_max_frac, adaptive_grid_max_passes, adaptive_grid_min_cells, adaptive_grid_stride, alpha, auto_recenter, axis_refine, batch_size, beta_init, ccd_budget, ccd_budget_floor, checkpoint, cila, copy_atom_mass, copy_slab, damping, device, diagnose_k, diagnose_skew, epsilon, ess_adapt_during_warmup, ess_adapt_interval, ess_joint_proposal_sd, ess_joint_sigma_re, ess_threshold, fitted_var, force_sparse, gradient_mode, hessian, inner_factorization, inner_refresh, integration, k_bootstrap, k_conf_bands, k_max_rounds, k_quality, k_refine, k_samples, k_tail_points, k_threads, keep_grid_hessians, L, local_ccd, marginal, marginal_richardson, marginal_step, mass_matrix, max_grid_cells, max_iter, max_sweeps, max_treedepth, mclmc_adjusted, method, mode_find, n_chains, n_draws, n_grid, n_iter, n_particles, n_per_axis, n_quad, n_threads, n_threads_outer, n_threads_scatter, n_warmup, noncenter, outer_design, outer_maxit, outer_reltol, progress, progress.every, progress.file, progress.throttle, prune, prune_log_gap, prune_tol, qmc_nodes, qmc_tol, quad_evidence_tol, quad_max_level, quad_read_max_nodes, quad_read_nodes, quad_tol, re_cov, recenter_pilot, scale, screen_iters, seed, sigma_init, skew_correct, skew_idx, smc_max_moves, smc_min_moves, span, store_Q, subspace_debias, thin, tile_warm, tol, var_of_means_consistency, var_of_means_min_ess, verbose, vi_max_grad_norm, vi_max_iter, vi_mc_samples, vi_patience, vi_tol_grad, vi_tol_rel_elbo, vi_variant, warmup, wbic, within_cell, x_init.
```

The list is long because it covers every backend; the fix is the
spelling `n_threads` used in the previous entry.

### A knob the chosen backend does not read

Once `mode` is fixed, `control` is checked against the backend that will
run. A Laplace fit has no chain, so a chain length is refused, and the
message lists the four knobs the Laplace backend reads.

``` r

tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "laplace",
      sigma_re = 0.6, phi = 0.8^2, control = list(n_iter = 2000))
#> Error:
#> ! Unknown control knob(s) for tulpa[mode = 'laplace'](): 'n_iter'.
#> Allowed: device, max_iter, n_threads, tol.
```

Either drop the knob or pick a backend that reads it. `n_iter` and
`warmup` belong to the samplers.

``` r

fit <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian", mode = "mala",
             sigma_re = 0.6, phi = 0.8^2,
             control = list(n_iter = 2000, warmup = 500))
coef(fit)
#> (Intercept)           x 
#>   0.3184458   1.2235454
```

### A variable the data do not contain

A covariate missing from `data` stops the fit with a message that names
it and lists the columns of `data`. A missing grouping variable is named
too.

``` r

tulpa(y ~ z, data = d, family = "gaussian", mode = "laplace", phi = 0.8^2)
#> Error:
#> ! variable(s) 'z' in the formula are not columns of `data` and are not defined in the formula's environment. `data` has: y, x, g.
tulpa(y ~ x + (1 | h), data = d, family = "gaussian", mode = "laplace",
      phi = 0.8^2)
#> Error:
#> ! Grouping variable 'h' not found in data
```

Check `names(d)` against every variable in the formula, including the
one after the bar.

``` r

names(d)
#> [1] "y" "x" "g"
coef(tulpa(y ~ x + (1 | g), data = d, family = "gaussian",
           mode = "laplace", sigma_re = 0.6, phi = 0.8^2))
#> (Intercept)           x 
#>   0.3334955   1.2215803
```

### Binomial counts without the number of trials

`dc$y` counts successes out of 5. Without `n_trials`, `binomial` expects
a 0/1 response and stops.

``` r

tulpa(y ~ x, data = dc, family = "binomial", mode = "laplace")
#> Error:
#> ! family = 'binomial' with no `n_trials` requires a 0/1 response. Supply `n_trials` for aggregated counts, or write the response as cbind(successes, failures).
```

Pass the trials, one per row.

``` r

coef(tulpa(y ~ x, data = dc, family = "binomial", mode = "laplace",
           n_trials = rep(5, n)))
#> (Intercept)           x 
#>  -0.3830110   0.9943815
```

The slope is close to the 1 the counts were simulated with. [Data
formatting](https://gillescolling.com/tulpa/articles/data-formatting.md)
shows the other response encodings.

### Missing values in the response

A missing response value stops a Laplace fit, and the message counts the
rows and names the path that reads them: on the nested-Laplace path a
row with no response is a forecast, with its likelihood term masked.

``` r

dna <- d
dna$y[3] <- NA
tulpa(y ~ x, data = dna, family = "gaussian", mode = "laplace", phi = 0.8^2)
#> Error:
#> ! The response has 1 NA row(s). A row with no response is read for its linear predictor (a forecast) on the nested-Laplace path, which masks its likelihood term; mode resolved to backend 'laplace'. Use mode = 'nested_laplace', or drop the rows.
```

A model with no latent field has no nested-Laplace path (see [Nested
Laplace on a model with no latent
field](#nested-laplace-on-a-model-with-no-latent-field)), so remove the
incomplete rows, or impute them, before fitting.

``` r

coef(tulpa(y ~ x, data = dna[!is.na(dna$y), ], family = "gaussian",
           mode = "laplace", phi = 0.8^2))
#> (Intercept)           x 
#>   0.3456292   1.2788365
```

### Region codes that do not match the adjacency

An integer region code is a row of the adjacency matrix. `W` has 10
rows, so codes 11 and 12 point outside it.

``` r

ds_bad <- ds
ds_bad$region[1:2] <- c(11, 12)
tulpa(y ~ x + spatial(region), data = ds_bad, family = "binomial",
      spatial = list(type = "icar", adjacency = W), mode = "laplace")
#> Error:
#> ! Spatial group variable 'region' has values outside [1, n_spatial_units = 10]: 11, 12. Integer `group_var` values are treated as 1-based row indices into the adjacency matrix.
```

An adjacency with a one-sided link is refused too.
[`check_adjacency()`](https://gillescolling.com/tulpa/reference/check_adjacency.md)
reports every property the spatial fit needs, and flags the asymmetry
here.

``` r

W_bad <- W
W_bad[1, 2] <- 0
check_adjacency(W_bad)
#> Warning: Adjacency matrix is not symmetric (max |W - t(W)| = 1).
#> <adjacency check>
#>   square:    ok (10 x 10)
#>   symmetric: FAIL
#>   zero diag: ok
#>   binary:    ok
#>   isolated:  0 node(s)
#>   components: 1
#>   edges:     10
#>   overall:   issues found
```

With codes in `1:10` and the symmetric `W`, the fit runs.

``` r

fit_sp <- tulpa(y ~ x + spatial(region), data = ds, family = "binomial",
                spatial = list(type = "icar", adjacency = W),
                mode = "laplace")
coef(fit_sp)
#> (Intercept)           x 
#>  -0.2585208   0.9839458
```

## Priors and dispersion

### A penalised-complexity prior with an invalid tail probability

A penalised-complexity (PC) prior on a standard deviation is set by two
numbers: a scale `U` and the probability `alpha` that the standard
deviation exceeds it. The prior exists only for `U > 0` and `alpha`
strictly between 0 and 1, and each constructor that takes the pair
checks it.

``` r

prior_pc(U = 1, alpha = 1.5)
#> Error:
#> ! prior_pc(): `alpha` is the tail probability P(sigma > U) and must lie in (0, 1); got 1.5.
spatial_gp(~ lon + lat, approx = "hsgp", sigma_prior_alpha = 1)
#> Error:
#> ! gp(approx = "hsgp"): `sigma_prior_alpha` is the tail probability P(sigma > sigma_prior_U) and must lie in (0, 1); got 1.
```

`alpha = 0.05` with `U = 1` reads “a 5% prior probability that the
standard deviation exceeds 1”.

``` r

prior_pc(U = 1, alpha = 0.05)
#> PC prior: P(x > 1.00) = 0.050
#>   => Exponential(2.996)
```

[Priors](https://gillescolling.com/tulpa/articles/priors.md) covers how
to choose `U` and `alpha`.

### A Gaussian fit without `phi`

For `gaussian`, `phi` is the residual variance, and `mode = "laplace"`
and the samplers hold it fixed at the value passed. Left out, it is
fixed at 1 with a warning.

``` r

fit <- tulpa(y ~ x, data = d, family = "gaussian", mode = "laplace")
#> Warning: tulpa(): `phi` not supplied; conditioning on phi = 1 for family =
#> 'gaussian' on backend 'laplace'. Pass `phi` to override, `estimate_phi = TRUE`
#> with mode = 'eb' to estimate it, or mode = 'auto' / 'structured' to integrate
#> it where the call allows.
```

When the residual variance is known, pass it as a variance: here
`0.8^2`, not `0.8`. When it is not known and the model has a random
effect, `mode = "eb"` with `estimate_phi = TRUE` estimates it along with
the random-effect standard deviation.

``` r

fit_eb <- tulpa(y ~ x + (1 | g), data = d, family = "gaussian",
                mode = "eb", estimate_phi = TRUE)
VarCorr(fit_eb)
#>       term        coef        sd    source
#> 1        g (Intercept) 0.4846696 estimated
#> 2 Residual        <NA> 0.7867884 estimated
```

Both standard deviations are labelled `estimated`: 0.48 for the groups
and 0.79 for the residual, against the 0.6 and 0.8 the data were
simulated with.

## Fitting

### A combination `tulpa()` does not fit yet

Some combinations of terms are refused before fitting. A random slope
next to a spatial field is one.

``` r

tulpa(y ~ x + (1 + x | site) + spatial(region), data = ds,
      family = "binomial", spatial = list(type = "icar", adjacency = W))
#> Error:
#> ! Random-slope term(s) together with a spatial field are not supported through tulpa() yet. Use a random intercept (1 | g) alongside the spatial term, or drop the spatial field.
```

A temporal field with one trend per panel group (`group_var`) next to a
spatial field is another.

``` r

tulpa(y ~ x + spatial(region), data = ds, family = "binomial",
      spatial = list(type = "icar", adjacency = W),
      temporal = temporal_rw1("time", group_var = "site"))
#> Error:
#> ! A grouped (panel) temporal field cannot be combined with a spatial or latent(...) field through tulpa() yet (the joint path has no per-group temporal layout). Fit the panel temporal field on its own.
```

Each message names the combination that does fit. With a random
intercept in place of the slope, the spatial model runs.

``` r

fit <- tulpa(y ~ x + (1 | site) + spatial(region), data = ds,
             family = "binomial",
             spatial = list(type = "icar", adjacency = W),
             mode = "laplace", sigma_re = 0.5)
coef(fit)
#> (Intercept)           x 
#>  -0.2589096   0.9905863
```

### Nested Laplace on a model with no latent field

`mode = "nested_laplace"` integrates the hyperparameters of a spatial,
temporal, smooth or custom field. A model with only a random intercept
has none of these, and the message names the modes that fit it.

``` r

tulpa(y ~ x + (1 | g), data = db, family = "binomial",
      mode = "nested_laplace")
#> Error:
#> ! Backend 'nested_laplace' needs at least one `latent(...)` block, s(...) smoother, spatial(col) field, or temporal field. To integrate the random-effect scale of this model use mode = 'structured' (or 'auto'), which fits it with the RE-covariance integrator; mode = 'laplace' conditions on `sigma_re` instead.
```

`mode = "structured"` stays in Tier 2 and integrates the random-effect
standard deviation.

``` r

fit <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
             mode = "structured")
c(backend = fit$backend, tier = fit$inference_tier)
#>         backend            tier 
#> "re_cov_nested"             "2"
```

### An outer grid larger than the cap

A model with a spatial and a temporal field has two hyperparameters, and
its outer grid crosses the two axes, one inner solve per cell.
`control$max_grid_cells` caps the cell count, and a grid past the cap is
refused before any solve, with the axes that produced the count.

``` r

tulpa(y ~ x + spatial(region), data = ds, family = "binomial",
      spatial = list(type = "icar", adjacency = W),
      temporal = temporal_rw1("time"),
      control = list(max_grid_cells = 50, n_threads = 2))
#> Error:
#> ! Multi-block outer grid has 81 cells, past control$max_grid_cells = 50. It crosses b1 (9 rows: tau 9) x b2 (9 rows: tau 9). The cap is read on the grid each pass is about to solve, the declared grid first; a placement or refinement pass that runs afterwards can re-lay the axes, so a fitted grid can hold a different number of cells. Reduce per-block grid sizes.
```

Raise the cap above the count the message gives, or leave it unset.
[`?tulpa_nested_laplace`](https://gillescolling.com/tulpa/reference/tulpa_nested_laplace.md)
documents that an unset cap solves a grid of any size and warns first
when it passes 2048 cells.

``` r

fit_st <- tulpa(y ~ x + spatial(region), data = ds, family = "binomial",
                spatial = list(type = "icar", adjacency = W),
                temporal = temporal_rw1("time"),
                control = list(max_grid_cells = 100, n_threads = 2,
                               progress = FALSE))
coef(fit_st)
#> (Intercept)           x 
#>  -0.2687675   0.9673109
```

## Reliability

### A sampler chain that has not converged

A sampler fit checks its own chain. A short chain on the binary data
leaves too few effective draws, and the fit warns, naming the parameters
and their Rhat and bulk effective sample size.

``` r

fit_short <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                   mode = "mala", sigma_re = 0.6,
                   control = list(n_iter = 150, warmup = 30))
#> Warning: tulpa(): the 'mala' chain has not converged -- x: Rhat 1.07, bulk-ESS
#> 26; (Intercept): Rhat 1.03, bulk-ESS 39. Its estimates and intervals are not
#> yet the posterior. Run longer (control$n_iter / control$warmup), or use mode =
#> 'hmc'; see check_diagnostics().
check_diagnostics(fit_short)
#> Convergence warnings:
#>   - 2 parameter(s) with Rhat > 1.01
#>   - 2 parameter(s) with bulk-ESS < 400
```

[`check_diagnostics()`](https://gillescolling.com/tulpa/reference/check_diagnostics.md)
lists the checks that failed. A longer chain clears them.

``` r

fit_long <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                  mode = "mala", sigma_re = 0.6,
                  control = list(n_iter = 3000, warmup = 1000))
check_diagnostics(fit_long)
#> Convergence checks passed (Rhat, bulk-ESS, divergences).
head(diagnostics(fit_long), 2)
#> Device: inference=cpu
#>     parameter      rhat ess_bulk ess_tail
#> 1 (Intercept) 0.9996312 564.6958 837.3768
#> 2           x 0.9999560 500.8862 614.9376
```

### `diagnostics()` returns nothing for a Laplace fit

A plain Laplace fit computes no reliability quantity, so
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
returns `NULL` with a message.

``` r

fit_lap <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                 mode = "laplace", sigma_re = 0.6)
diagnostics(fit_lap)
#> diagnostics(): the 'laplace' fit carries no posterior draws. It carries a Laplace mode and its precision (`$mode` / `$H_beta`), summarized by summary() / confint(). No approximation-reliability quantity was computed for it either, so there is nothing to report.
#> Device: inference=cpu, dense_spec=cpu
#> NULL
```

`mode = "imh_laplace"` reuses the Laplace Gaussian as the proposal of an
exact Metropolis-Hastings sampler. A high acceptance rate says the
Laplace approximation was close to the posterior, and the fit then
carries a chain that
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
reads.

``` r

fit_imh <- tulpa(y ~ x + (1 | g), data = db, family = "binomial",
                 mode = "imh_laplace", sigma_re = 0.6)
round(fit_imh$mean_accept, 2)
#> [1] 0.57
rbind(laplace = coef(fit_lap), imh = coef(fit_imh))
#>         (Intercept)         x
#> laplace  -0.3792294 0.9210190
#> imh      -0.3925564 0.9602661
```

### Reading the reliability report of a nested-Laplace fit

On a nested-Laplace fit
[`diagnostics()`](https://gillescolling.com/tulpa/reference/diagnostics.md)
prints the reliability of the outer integration over the
hyperparameters, and of the inner Laplace layer where the backend scores
it. The fit below puts a random walk over the 15 time points of `ds`.

``` r

fit_rw <- tulpa(y ~ x, data = ds, family = "binomial",
                temporal = temporal_rw1("time"),
                control = list(n_threads = 2))
#> [nested-laplace] 1/9 cells (11%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 7/7 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
#> [nested-laplace] 1/5 cells (20%) | elapsed 0s | ETA >=0s | 0.00s/cells
#> [nested-laplace] 5/5 cells (100%) | elapsed 0s | ETA done | 0.00s/cells
diagnostics(fit_rw)
#> Device: inference=cpu, laplace_dense=cpu, sparse_cholesky=cpu
#> Nested-Laplace WHOLE-FIT reliability (i.i.d. draws)
#>   two layers: the outer hyperparameter-grid integration, and the inner
#>     Gaussian Laplace on the latent field
#>   outer PSIS pareto_k = -1.408 (good); IS-ESS = 479.1
#>     proposal: grid_mixture; as the backend placed it, 0.494 (good)
#>   outer grid quadrature ESS = 3.39 of 10 cells (max weight 0.441)
#>   hyperparameter integration contributes 0.0-0.0% of the fixed-effect marginal variance
#>   note: outer grid re-centred on tau: the default span did not contain that axis's posterior mode, so the reported grid is not the default one
#>   note: outer grid coarser than its own posterior on tau (cell width / posterior SD = 1.27, both in that axis's own coordinate): the reported interval's endpoints are resolved to within one cell, so part of their width and their realized coverage are properties of where the grid fell rather than of the posterior -- add nodes on that axis to reduce it
#>   inner Laplace max |gamma_3| = 0.124 (good), scored 2/2 latents
#>   inner Laplace importance pareto_k = 0.621 (good), min IS efficiency 0.9960, scored 2/2 latents
#>     the importance weights are uniform on every probed index: the inner
#>     Gaussian reproduces the conditional posterior over the sampled region,
#>     so the shape above describes no correction and is not banded
#>   whole-fit verdict: reliable (both layers good)
#>   per-parameter columns: none.
#>     the fit carries no posterior draws, so the per-parameter mean / sd / per-parameter columns are empty; tulpa_posterior_draws(fit) samples the retained outer-grid mixture if a sample is wanted
```

The lines to read, in order:

- `outer PSIS pareto_k` with its band in parentheses: below 0.5 is good,
  0.5 to 0.7 ok, 0.7 and above unreliable.
- `outer grid quadrature ESS`: how many cells carry the weight. A value
  near 1 means a single cell holds the posterior.
- the `note:` lines, which name an axis whose grid was moved or is
  coarser than its posterior.
- `whole-fit verdict`, which combines the outer and inner layers.

Here the outer k is -1.408, in the good band, and the verdict reads the
fit as reliable, with both layers good. Each number is also an attribute
of the returned object. On the two-field fit `fit_st` of the grid entry
the outer k is `NA`, because it is not defined on a multi-block grid,
and that fit’s verdict says the outer integration was not assessed.

``` r

attr(diagnostics(fit_rw), "pareto_k")
#> Device: inference=cpu, laplace_dense=cpu, sparse_cholesky=cpu
#> [1] -1.407576
attr(diagnostics(fit_st), "pareto_k")
#> Device: inference=cpu, laplace_dense=cpu, sparse_cholesky=cpu, sparse_selected_inverse=cpu
#> [1] NA
```

[Reliability
diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.md)
explains what a k in the ok or unreliable band means and the levers for
it.

## Where next

- [Concepts](https://gillescolling.com/tulpa/articles/concepts.md): the
  terms used in these messages.
- [Data
  formatting](https://gillescolling.com/tulpa/articles/data-formatting.md):
  response encodings, grouping variables and spatial inputs.
- [Reliability
  diagnostics](https://gillescolling.com/tulpa/articles/reliability-pareto-k.md):
  the outer Pareto k in depth.
- [Inference
  modes](https://gillescolling.com/tulpa/articles/inference-modes.md):
  which backend fits which model.
