# Fit a fixed-effect GLM with a model-agnostic sampler kernel

Drives one of tulpa's ModelData sampler kernels – NUTS (`"hmc"`),
elliptical slice sampling (`"ess"`), SGHMC (`"sghmc"`), SGLD (`"sgld"`),
MCLMC (`"mclmc"`), sequential Monte Carlo (`"smc"`), or variational
inference (`"vi"`) – on a fixed-effect GLM. The model (design +
per-observation likelihood) is built once through the same
built-in-family scaffold the single-point Laplace fit uses, so no
likelihood / link logic is duplicated.

## Usage

``` r
tulpa_sample_glmm(
  y,
  n_trials,
  X,
  family,
  backend,
  phi = 1,
  phi2 = NULL,
  offset = NULL,
  fixed_names = NULL,
  re_spec = NULL,
  spatial_spec = NULL,
  temporal_spec = NULL,
  svc_spec = NULL,
  tvc_spec = NULL,
  zi_spec = NULL,
  sigma_re_scale = 2.5,
  sigma_beta = .tulpa_prior_sd("sample_glmm"),
  warm_start = NULL,
  control = list()
)
```

## Arguments

- y:

  Response vector.

- n_trials:

  Binomial denominators (or `NULL` -\> all 1).

- X:

  Fixed-effect design matrix (`nrow(X) == length(y)`).

- family:

  Character family name (see
  [`family_names()`](https://gillescolling.com/tulpa/reference/family_names.md)).

- backend:

  One of `"hmc"`, `"ess"`, `"sghmc"`, `"sgld"`, `"mclmc"`, `"smc"`,
  `"vi"`.

- phi:

  Dispersion/precision passed to the family (held fixed). The kernel
  parameterization: for `gaussian` / `lognormal` this is the residual SD
  (the [`tulpa()`](https://gillescolling.com/tulpa/reference/tulpa.md)
  front door passes `sqrt(phi)`, its `phi` being the variance); for `t`
  the scale.

- phi2:

  Optional second dispersion: the Student-t degrees of freedom
  (`family = "t"`; default 4 when `NULL`).

- offset:

  Optional fixed additive term on the linear predictor
  (`eta = offset + X beta`), length `length(y)`; `NULL` -\> no offset.

- fixed_names:

  Optional fixed-effect names for the draw columns.

- re_spec:

  Optional random-effect spec: a list with `idx` (list of per-term
  1-based group-index vectors), `ngroups`, `ncoefs`, `correlated`
  (per-term), and `Z` (per-term RE design or `NULL`). `NULL` -\> no RE.

- spatial_spec:

  Optional areal spatial spec: a list with `type` (`"icar"`/`"bym2"`),
  `spatial_idx`, `n_spatial_units`, `adj_row_ptr`, `adj_col_idx`,
  `n_neighbors`, and `scale_factor` (BYM2). `NULL` -\> none.

- temporal_spec:

  Optional temporal spec: a list with `type` (`"rw1"`/`"rw2"`/`"ar1"`),
  `time_idx`, `n_times`, `n_groups`, `group_idx`, and `cyclic`. `NULL`
  -\> none.

- sigma_re_scale:

  Half-Cauchy scale for the RE / BYM2 standard-deviation hyperprior
  (sampled jointly with the latent effects).

- warm_start:

  Optional `list(init, inv_metric_diag)` seeding the NUTS kernel: `init`
  an `n_chains x total_params` matrix of initial positions, one row per
  chain, and `inv_metric_diag` a positive vector of length
  `total_params` used as the starting inverse mass (warmup adaptation
  still runs from it). Build it with
  [`.build_warm_start()`](https://gillescolling.com/tulpa/reference/dot-build_warm_start.md)
  against `cpp_tulpa_glmm_layout()` rather than by hand – the entries
  are positional and the layout owns the positions. Only the NUTS/HMC
  kernel takes one; any other backend errors rather than sampling from
  the default start.

- control:

  List of kernel tuning knobs (`n_iter`, `warmup`, `seed`, `sigma_beta`,
  `n_chains`, `max_treedepth`, `adapt_delta`, `epsilon`, `L`,
  `batch_size`, `alpha`, `n_particles`, `n_mcmc_steps`, `ess_threshold`,
  `vi_variant`, `vi_mc_samples`, `vi_max_iter`, `vi_max_grad_norm`,
  `n_draws`, `verbose`, `mass_matrix`).

  `mass_matrix` selects the NUTS/HMC metric: `"diag"` (the default),
  `"dense"`, `"block_diag"`, or `"auto"`. Under `"auto"` the kernel
  reads the parameter layout and gives each correlated hyperparameter
  group its own small dense block – the BYM2 and GP `(log sigma, phi)`
  pairs, the multiscale-temporal variances, a correlated random-slope
  term's Cholesky coordinates – while an ICAR or latent-factor model
  small enough for the `O(p^2)` per-step cost takes a full dense metric.
  `"block_diag"` additionally blocks the temporal-GP and HSGP
  hyperparameter pairs, which `"auto"` leaves to the diagonal. Backends
  other than `"hmc"` carry no metric and reject a non-default value.

  `vi_max_iter` and `vi_mc_samples` both bound a loop that has to run at
  least once – the optimisation loop whose ELBO history is the fit's
  only record, and the reparameterisation average every gradient divides
  by – so values below 1 are rejected. `vi_max_grad_norm` (default 10)
  is the gradient-norm clip applied before every Adam step.

  `epsilon` pins the step size on the stochastic-gradient backends:
  `"sghmc"` runs its warmup step-size adapter only when no `epsilon` is
  supplied, and `"sgld"` runs its polynomial decay `a * (b + t)^-gamma`
  only then, so a supplied value is what the whole run samples at. On
  `"mclmc"` a non-positive `epsilon` selects the kernel's own
  adaptation. `alpha` is the SGHMC friction and `L` its leapfrog count;
  SGLD carries neither.

  The SGHMC discretisation is calibrated for small
  `epsilon^2 * lambda_max`, and inflates every posterior SD above that;
  the acceptance statistic its adapter targets is computed from a
  log-posterior ratio the sampler never accepts or rejects on, so it
  does not measure that error. A sharply informative design wants an
  `epsilon` chosen by hand.

  The elliptical-slice kernel takes four more, all prefixed `ess_` and
  all inert on other backends. Note that `ess_threshold` above is SMC's
  resampling threshold and not one of them – the two unrelated senses of
  "ESS" are why these carry the prefix. `ess_adapt_during_warmup`
  (default `FALSE`) adapts the random-walk proposal SDs on the
  non-Gaussian parameters during warmup and `ess_adapt_interval`
  (default 50) is how many sweeps sit between those updates, so it
  **acts only while adapting**. `ess_joint_sigma_re` toggles the joint
  `(log_sigma_re, re)` rescaling move, which defaults to on whenever a
  random-effect term is present because the two are strongly
  anti-correlated under the centered parameterization and mix poorly
  when moved separately; forcing it off is how one demonstrates that.
  `ess_joint_proposal_sd` (default 0.1) is that move's step.

  The elliptical-slice kernel draws from R's own RNG, so `control$seed`
  does not reach it: [`set.seed()`](https://rdrr.io/r/base/Random.html)
  before the call is what reproduces an ESS run. Every other backend
  carries `control$seed` into its own generator.

## Value

A `tulpa_fit` with `draws`, `means`, `param_names`, the kernel's
diagnostics, and (for `"hmc"`) `chain_id` / `n_chains` so chain
diagnostics apply.
