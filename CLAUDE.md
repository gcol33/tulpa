# TULPA — Templated Unified Library for Posterior Approximation

General-purpose Bayesian hierarchical modelling engine (v0.1.0). Engine
extracted from numdenom, which has since been renamed tulpaRatio.

## Architecture

The hub of a `tulpa*` package ecosystem. The engine owns inference, latent
structure, and the C++ interface; model packages plug an observation
likelihood in via `LikelihoodSpec` and inherit the rest.

- **tulpa** (engine, 0.1.0) — samplers, autodiff, spatial, temporal, priors, formula infrastructure. Imports tulpaMesh for SPDE mesh construction.
- **tulpaRatio** (1.4.3) — ratio, rate, and proportion models (renamed from numdenom).
- **tulpaObs** (0.0.235) — occupancy, N-mixture, and detection models.
- **tulpaGlmm** — RETIRED. Generalized linear mixed models are fitted by the
  engine directly: `tulpa(y ~ x + (1 | g), family =, mode =)` covers the
  families, inference modes and random-effect structure it carried, and the
  machinery that was only ever in tulpaGlmm has been absorbed (warm-starting a
  sampler from a Laplace/EB fit, the marginal-Laplace covariance correction,
  hyperparameter standard errors, empirical-Bayes dispersion estimation, and
  `VarCorr()`). Do not add a GLMM feature there.
- **tulpaMesh** (>= 0.1.3) — constrained Delaunay meshes for SPDE fields (an engine dependency, not a consumer).

Consumers depend on the engine via `Imports: tulpa (>= 0.0.13)` +
`LinkingTo: tulpa` and plug likelihoods in through `LikelihoodSpec`
(see `inst/include/tulpa/likelihood.h`).

## Design Philosophy

**2026 Bayesian engine: nested approximation + debias.**

Decompose the posterior so well-behaved blocks get cheap deterministic
approximations (Laplace, EP, VI, Pathfinder) and exact MCMC corrects only
the residual directions where the approximation is biased.

Positioning:
- INLA: nested approximation, **no** debias — biased on non-Gaussian residuals.
- Stan: exact MCMC, **no** nested approximation — pays full price on every block.
- tulpa: the synthesis.

Canonical compositions:
- IMH-Laplace = Laplace body + MH bias correction
- Pathfinder = L-BFGS mode + Gaussian fit + ELBO scoring
- nested_laplace + CCD = inner Laplace + outer hyperparameter integration

New backends are framed by which layer they add or replace (approximation,
debias, or outer integration), not as standalone alternatives.

## Key Design Principles

1. **Tier system encodes correctness, not hot path.** Tier 1 (exact MCMC), Tier 2 (Laplace), Tier 3 (VI). Hot path is consumer-dependent — for Gaussian-latent hierarchical workloads the typical path is Tier 2 with optional Tier 1 debiasing. Auto mode never silently chooses Tier 3.
2. **Gradient progression N→A→A_r→H** (Tier 1 / NUTS only): Never skip stages. All modes remain available.
3. **Runtime gradient verification**: Before NUTS sampling, verify active
   gradient against numerical. The gate is `ensure_gradient_verified()` inside
   `run_hmc_chain_cpp` -- the one function every NUTS entry reaches, so no door
   can bypass it. It runs once per fit (`g_gradient_verified` travels with
   `GradientModeFitScope`) and skips inside an across-chain OpenMP region, where
   the producer has already run it on the main thread. Placed in the CALLERS it
   reached 2 of 8 entries, missing the C-ABI consumers use for a single chain
   (gcol33/tulpa#684).
4. **Model packages own their likelihood**: tulpa assembles linear predictors; model packages compute log-likelihood.
5. **No copy-paste logic**: Shared sub-computations in helpers, not duplicated across specialized functions. Conventions that keep this single-sourced: log-prior helpers are named `log_prior_*` (e.g. `log_prior_car_proper`, `log_prior_sigma2_pc`); the column-major Rcpp matrix builder `build_matrix_colmajor` is one template over the element type; the spatially-/temporally-varying-coefficient `print`/`summary` methods delegate to `.print_varying_coef` / `.summary_varying_coef` in `R/varying_coef.R`; multi-block prior detection is the single `.is_multi_block_prior` predicate;
   and every `cpp_nested_laplace_*` outer-grid entry hands its shared response
   and control arguments to its driver as one `tulpa::NlEntryInputs`, collected
   by the argument-free `TULPA_NL_ENTRY_INPUTS` macro and consumed by one runner
   per driver (`src/nl_entry_inputs.h`), so a knob added to a driver is added to
   the bundle and the runner rather than to eleven entry tails.
   `test-nl-entry-forwarding.R` asserts every shared argument is observable in
   the fit at all eleven entries; and `check_arg_length` (`laplace_spec_fit.h`)
   is the one place an entry refuses an R argument whose length does not match
   the one it will be indexed by.
6. **Statistical args vs `control` knobs**: front-door fitters (`tulpa()`, `tulpa_nested_laplace()`, `tulpa_nested_laplace_joint()`) carry only statistical arguments in their signature; all perf / numerical / tuning knobs live in a single `control = list()` (e.g. `control$re_cov`, `n_threads`, `max_iter`, `tol`, `adaptive_grid`, `prune`, `integration`, ...). Pre-release: no deprecation shims -- moved knobs hard-error.

## C++ Interface

- Exported headers: `inst/include/tulpa/`
- Model packages use `LinkingTo: tulpa` in DESCRIPTION
- Key types: `ModelData`, `ParamLayout`, `LikelihoodSpec`, `LikelihoodFn<T>`, `ResidualFn`
- **ABI version**: `TULPA_ABI_VERSION` in `model_data.h`. Bump when any exported struct layout
  changes. Model packages auto-check on first NUTS call — mismatch gives a clear error instead
  of segfault. Registered via `R_RegisterCCallable("tulpa", "tulpa_get_abi_version", ...)`.
- **Layout rule**: `ModelData` requires `n_processes > 0` and a `LikelihoodSpec`
  (ratio models live in tulpaRatio via that interface). New fields go in the
  stable sections — never insert before existing fields.
- **Two AD slots on `LikelihoodSpec`, not three** (gcol33/tulpa#493, ABI 42).
  `ll_double` and `ll_arena` only. `resolve_gradient_fn` dispatches on
  `gradient_fn`, then `ll_arena`, then the numerical fallback, and
  `AUTODIFF_FWD` resolves to the arena path, so a forward-dual slot was filled
  by every spec and read by none — six `fwd::Dual` instantiations of likelihood
  kernels that never ran, and an unexercised copy of a density is where a kernel
  falls silently out of step with its siblings. Do not reintroduce one; route
  the mode instead if forward mode is ever wanted.

## Prior anchors, and where a bad one is caught

Every PC prior in the package routes through `pc_prior.h`, whose calibration
`lambda = -log(alpha) / U` exists for `U > 0` and `alpha` in `(0, 1)` only.
Outside that the density is `-Inf` or NaN at every value of the scale, and it
reaches a gradient as a number rather than as a message. `pc_anchors_valid` is
the ONE predicate; the layers differ only in what they can do about it
(gcol33/tulpa#499):

- **R front doors** (`.check_pc_anchors`, `R/validate_helpers.R`) name the
  argument the user set.
- **The sampler entry** (`read_pc_anchors`, `sampler_model_data.h`) names the
  spec, and is what reads an anchor pair off a spec list at all — a pair the
  spec does not carry keeps the `ModelData` default.
- **The templated density** falls back to a flat prior on sigma. It runs inside
  gradient loops and OpenMP regions, where a throw is `std::terminate` rather
  than an R error (gcol33/tulpa#459), so an error is not available to it. The
  callers keep their change-of-variables terms, so an unvalidated path gets
  flat-on-sigma carried correctly to its own coordinate, never a NaN.

`nl_check_positive` / `nl_grid_axes_positive` (`nested_laplace_grid.h`) are the
same convention for a scale or precision that reaches a logarithm — `sigma_re`,
`tau_grid`, `sigma2_grid`, a lengthscale axis — checked at the entry point, the
counterpart of `nl_grid_axis_unit_interval` for the BYM2 mixing weight.

The HSGP, HSGP-ST and TVC scale priors hardcoded `P(sigma > 1) = 0.01` inline;
they now read `ModelData` fields defaulted to exactly that, settable through
`spatial_gp(approx = "hsgp", sigma_prior_U =, sigma_prior_alpha =)` and
`temporal_tvc(sigma_prior_U =, sigma_prior_alpha =)` (gcol33/tulpa#506).

**The two temporal-GP parameterizations are pinned to each other.** They differ
by exactly the forward transform's log-determinant, which is what fixes the
conditional variance both branches use: the non-centered branch reaches it
through the transform's scale `a_t` and the centered branch through
`cond_var_t`, and flooring different quantities leaves them orders of magnitude
apart wherever the floor binds (a long lengthscale on a fine time grid). Both
read one floored `ar1_one_minus_rho2`. `cpp_test_temporal_gp_density`
(`src/test_temporal_gp_fixture.cpp`) drives the shipped density at either
parameterization — the fixture builds the `ModelData` and the layout, not a
second copy of the density — and `test-temporal-gp-parameterization.R` asserts
the identity, including at a configuration where the floor binds at every step.

## Versioning

`0.1.0` is the first CRAN release. Routine work bumps the patch number, which
keeps counting past 9: `0.1.9` -> `0.1.10` -> `0.1.11`. A CRAN resubmission
bumps the patch; a release adding user-visible surface bumps the minor.

## Building

```r
devtools::load_all()
devtools::check(args = "--no-manual")
```

Test profiles (single source of truth `tests/testthat/helper-tiers.R`; tier table
in `tests/testthat/README.md`): `devtools::test()` runs tiers 1 + 2 (structural +
single-fit recovery); `Sys.setenv(TULPA_FAST = "1")` collapses to tier 1 only
(every fit / sampler skips, whole suite in seconds, for plumbing iteration);
`Sys.setenv(TULPA_SLOW_TESTS = "true")` adds tier 3 (samplers, multi-seed
coverage). CRAN runs tier 1 only.

**Release caveats** (the routine `check()` above does not catch these):

- `--no-manual` skips the PDF reference manual, so Rd LaTeX errors stay
  hidden locally. Non-ASCII typographic Unicode in roxygen (arrows, super/
  subscripts, math operators, Greek -- see the ASCII-only rule) only fails on
  `R CMD Rd2pdf` / win-builder / CRAN. Before any release, build the manual
  (drop `--no-manual`) or run `devtools::check_win_devel()`.
- Recovery / coverage tests are `skip_on_cran()`-gated, so a default
  `check()` run exercises plumbing only and a calibration regression passes
  silently. Validate with `NOT_CRAN=true` (recovery) and
  `TULPA_SLOW_TESTS=true` (the 20-seed aggregate coverage gate in
  `test-nested-laplace-recovery.R`) set in the environment.
- Heavy multi-recovery files can SIGKILL (exit 137) under a background test
  harness; run decisive files individually rather than the full suite at once.

**Windows objects carry `-Wa,-mbig-obj`** (`src/Makevars.win`). A plain COFF
object holds 32767 sections, gcc emits one comdat section per template
instantiation, and nothing merges them when the build does not optimize: at
`-O0` `nested_laplace_joint_multi.cpp` assembles 50241 sections and
`aghq_re.cpp` 36959, and the assembler stops with "file too big". That is the
default path -- `pkgbuild::compile_dll(debug = TRUE)` is what
`devtools::load_all()` goes through, and it replaces R's flags with
`-UNDEBUG -Wall -pedantic -g -O0`. The flag selects the bigobj variant and the
ceiling stops binding. Do not drop it and do not read it as a property of one
file: six further translation units sit between 21000 and 31000 sections at
those flags, while at R's own `-O2` the largest of all 109 is 2221, 6.8% of the
ceiling. The flag is therefore CONDITIONAL on a non-optimizing build: it is
added only when `CXXFLAGS` carries none of `-O2` / `-O3` / `-Os` / `-Ofast`,
tested inside a recursive variable so it expands when a compile rule runs,
after `Makeconf` and the user Makevars are read (`R CMD SHLIB` sets `CXXFLAGS`
to `$(CXX17FLAGS)`, so it is the rule's own flags). Unconditional, it drew a
win-builder NOTE: "checking compilation flags used" reads the compile
COMMANDS, not the Makevars text, so living only in `Makevars.win` does not
hide it -- the claim that it did was never measured, and tulpa 0.4.0's first
win-builder r-devel run is what refuted it.

The repo's `.Rprofile` also sets `options(pkg.build_extra_flags = FALSE)`, which
keeps `load_all()` on R's own `-O2` -- the same flags `R CMD INSTALL` uses, and
far faster than a `-g -O0` build whose objects run to 50 MB apiece. R reads it
only when the session STARTS in the package root and startup files are not
skipped, so a build driven from another directory or under `--vanilla` still
takes the debug flags.

## A varying coefficient's level: centre a proper field, pin an intrinsic one

A varying-coefficient term contributes `eta_i += x_i w(s_i)`, so `w -> w + c`
together with `beta -> beta - c` leaves eta EXACTLY unchanged whatever the
covariate. That alias has to go before the field reaches the likelihood, and
which instrument removes it depends on whether the field's own prior is proper
(gcol33/tulpaRatio#25):

- **Intrinsic** (ICAR, RW1, RW2, the interactions built from them): the constant
  direction carries no prior at all, so one has to be supplied. That is
  `sum_to_zero.h`'s augment-and-centre, and `soft_sum_to_zero.h`'s
  `s2z_precision(n)` is the soft form it replaced.
- **Proper** (an NNGP / GP field, a TVC AR1 or GP, a TYPE_I interaction): the constant
  direction already carries a prior, precision `1' Sigma^-1 1`. Nothing needs
  supplying; the direction needs REMOVING from the likelihood, which is what
  centring the field on its way into eta does. `s2z_centre_blocks` is the whole
  instrument, and a penalty on the sum is not a weaker version of it -- it
  stiffens a direction the sampler still has to traverse, at curvature
  `lambda n^2` against the field's own `1 / sigma^2`. There is also no constant
  that would make one right: matched to the field's own prior on the sum,
  `lambda = 1 / (1' Sigma 1)`, it is a term the field prior already carries;
  anything else is a second, unstated prior on the level.

Both SVC parameterizations go through `svc_center_eta` (`hmc_svc_autodiff.h`),
so the centring and the eta it feeds are one function and no path can build eta
from an uncentred field. The stored draws are centred to match
(`hmc_nuts_chain_iter_store.h`), under either parameterization.

**The TVC block takes the same construction**, per (group, term) block:
`tvc_center_eta` (`hmc_tvc.h`) is the one door into `tvc_eta`, and
`tvc_log_prior` adds the augmentation and its one extra rank for the INTRINSIC
structures (`rw1`, `rw2`) and not for the proper ones (`ar1`, `gp`), the split
`tvc_structure_is_intrinsic` names. The stored draws are centred to match.
It was the last block still on `tvc_sum_to_zero_penalty` (deleted), and the
cost was measured under VI rather than under a sampler, which is the asymmetry
to expect: a stiff aliased direction is something a sampler traverses and a
diagonal-ish variational family cannot. On a poisson rw1 TVC at T = 40, the VI
fit read cor 0.344 / field sd 0.040 against a truth of 0.472 and an `rw1`
comparator at 0.842 / 0.460; after the change it reads 0.842 / 0.460, the
comparator's own numbers. The reported ELBO stopped inverting with run length
in the same move (gcol33/tulpa#844).

**The non-centered path is where this is load-bearing, and the centered path is
where it was NOT the defect.** On `w = L z` a penalty on the sum becomes
`-0.5 lambda (v'z)^2` with `v = L'1`, whose stiffness rides the Vecchia cascade
-- large for early-ordered `z`, small for late ones -- so it is anisotropic
against the unit-variance prior on `z` and no diagonal mass matrix reaches it;
that is what collapsed the field amplitude and produced #144's divergence storm.
On the CENTERED path the same penalty was measured and does NOT misbehave:
paired arms on the poisson SVC fixture, 4 chains x 400 iterations, penalty
against centring, `beta_x` 1.661 / 1.654 against a truth of 1.650 at n = 80 and
1.632 / 1.598 against 1.647 at n = 120, Rhat 1.04 / 1.02 and 1.03 / 1.04, no
divergences either way. It is unified because one construction should identify
one alias, not because the other was breaking.

## Boundary: What Belongs in tulpa vs Model Packages

**tulpa owns** (generic, model-agnostic):
- Inference engines: Laplace, EM+Laplace, VI, ESS, NUTS, MI correction, Gibbs correction
- Autodiff: arena, forward, tape
- Latent structure: spatial, temporal, RE, SVC, TVC, ST, latent factors

### Temporal GP (irregularly-spaced times)

`temporal_gp(time_var, cov =, nu =, period =, parameterization =)` is a
continuous-time GP over the DISTINCT time instants, so it is the field for
irregular spacing where RW1/RW2/AR1 assume a grid. It is **sampler-path only**
(`mode = "hmc"` and the other sampler modes): the hyperparameters
`log_sigma2_temporal_gp` / `logit_phi_temporal_gp` are sampled jointly with the
field, and there is no nested-Laplace kernel laying a grid over a dense T x T
Gaussian. It cannot yet share a fit with a spatial or `latent()` block.

The whole density is `compute_temporal_prior()` in `tulpa_priors_temporal.h`,
templated, including the non-centered `z -> f` forward transform (it overwrites
`phi_temporal` in place for the observation loop, so nothing in eta assembly or
the gradient kernels has to know which parameterization ran). Kernels live in
`src/temporal_gp_kernel.h`, templated over the scalar type because
`(sigma2, phi)` are sampled -- a plain-double kernel cannot serve this path,
which is why the untemplated copies deleted in #284 were never wired.

Dispatch is `cov_is_markov()`: exponential (equivalently Matern `nu = 0.5`) is
an Ornstein-Uhlenbeck process, so its joint density factorizes into a
first-order Markov chain and evaluates in O(T) with no matrix; Matern 3/2 and
5/2, Gaussian and periodic have no finite-dimensional state-space form and take
a dense T x T Cholesky. Matern is closed-form at `nu` in {0.5, 1.5, 2.5} only
and R rejects the rest at construction (gcol33/tulpa#288 was those choices being
accepted and then silently run as exponential).

**`temporal_tvc(structure = "gp")` is the same kernel on a COEFFICIENT**
(gcol33/tulpa#847): `eta_i += x_i w(t_i)` where `temporal_gp()` is
`eta_i += f(t_i)`, so it is the TVC structure for irregular spacing rather than
a second spelling of the additive field. Both doors read one
`src/temporal_gp_kernel.h` -- the OU chain (`ou_chain` / `ou_log_density` /
`ou_forward`) and the dense factorization -- one `read_temporal_gp_kernel()` at
the C++ spec boundary and one `.check_temporal_gp_kernel()` in R, so neither can
come to accept a kernel or a smoothness the other refuses. The coefficient
samples `log_sigma2_tvc_gp[j]` / `logit_phi_tvc_gp[j]` per term, on the same
coordinates the additive field samples its pair on, and the field is PROPER so
`tvc_center_eta` removes its level with no augmentation.

**A GP lengthscale starts at `0.2 * sd(time)`, not at its support's midpoint**
(`init_tvc_gp_lengthscale`, `sampler_model_data.h`). The support defaults to
(0.01, 10) and the time values are standardized, so the midpoint is a
lengthscale five times the data's own spread: the dense T x T covariance is
then numerically rank-one and its Cholesky jitter binds at the starting point.
The runtime gradient check's deviation on the lengthscale then orders itself by
kernel smoothness -- Matern 5/2 clean to a ratio of 1.0 then 9.3e-04 at 2.0;
Gaussian clean to 0.25 then 1.2e-03 at 0.5 and 8.3e-03 at 5.0 -- which is a
floor binding, not a wrong derivative. `temporal_gp(parameterization =
"centered")` still starts at the midpoint and reproduces those numbers exactly
(gcol33/tulpa#851).
- ZI/OI parameter-layout hooks only (`ZIType` enum, `has_zi` / `has_oi`); the distribution-specific ZI likelihood math lives in model packages
- Censoring/truncation KERNELS only: `interval_gaussian` / `truncated_gaussian` are generic per-observation likelihood arms that model packages compose. General censored / survival responses (right-censored gaussian/lognormal, Weibull/exponential AFT with a censoring indicator) are an observation process and belong to tulpaObs via `LikelihoodSpec`; the engine does not grow a censoring-indicator front door (decided 2026-07-07, closes the recurring todo item)
- Generic S3 methods operating on posterior draws: coef, confint, vcov, logLik, summary
- Generic diagnostics: moran_i, durbin_watson, tulpa_variogram, compare_models, model_average
- Generic plotting: trace, density, pairs plots of posterior draws
- Rubin's rules pooling
- Parameter back-transformation (logit → probability)

**Model packages own** (e.g., tulpaObs, tulpaRatio):
- Likelihood functions (LikelihoodSpec), including the distribution-specific zero-inflation / hurdle / one-inflation and ratio (num/denom) likelihood math
- E-step weight computation (model-specific latent variable posterior)
- Data structures and encoding (how to map model data → binomial pseudo-data for Laplace)
- Model-specific diagnostics (waicOccu, ppcOccu, pitResiduals)
- Model-specific fitted/residuals/simulate
- Data formatting and simulation functions
- Print methods referencing model-specific parameter names

### EM+Laplace Engine

`tulpa_em_laplace()` is the generic EM driver: per-submodel `family` +
`offset` on the `m_step_encode` return blocks (gcol33/tulpa#3) and the
optional `m_step_extra(fits, weights, ...) -> fits` callback for non-η
parameters fired between M-step and E-step (gcol33/tulpa#4). `correction =
"mi"/"gibbs"` run post-EM multiple-imputation / warm-started Gibbs refits
pooled via `rubins_pool()` (`.mi_correction` / `.gibbs_correction` in
`R/em_correction.R`). An optional `beta_prior = list(mean, sd)` threads a
Gaussian fixed-effect prior into every `tulpa_laplace()` block and into the
MI/Gibbs refits (gcol33/tulpa#27); blocks may override it with their own
`beta_prior` field. See `?tulpa_em_laplace`.

### Random-effect covariance integration (free Sigma for random slopes)

For random-slope terms the engine treats the RE covariance(s) `Sigma`
themselves as the inferred quantity instead of a point estimate -- the
nested-approx + debias philosophy applied to a free `Sigma`. Both fitters
operate on a **list of covariance blocks**: one block per RE term, each either
**full** (correlated, `(1 + x | g)`) or **diagonal** (uncorrelated,
`(1 + x || g)`), with a scalar `(1 | g)` term as the degenerate `c = 1` block.
A single-term model is the length-1 case of the same path -- no special-casing.

- **`tulpa_re_cov_nested()`** (`R/nested_laplace_re_cov.R`) -- nested-Laplace
  integration over the joint `Sigma`. A full block parameterizes `Sigma = L L'`
  in **log-Cholesky** coords (log-diagonal + strict-lower of `L`, `c(c+1)/2`
  params, **general `c`**, always PD); a diagonal block uses `c` log-SD coords.
  Per-block params stack into one integration vector. Nodes are centred at the
  joint marginal-likelihood mode and rotated by the Cholesky of its posterior
  covariance, and each derived quantity (`sigma_i`, `rho_ij` for full blocks,
  `Sigma_ij`) is computed per cell then weighted-quantiled -- the "Marginalize
  Derived Quantities" rule (Bias-2). Inner solve is `tulpa_laplace()` (which is
  already multi-RE, correlated-or-diagonal) at the supplied covariances; outer
  is the `nested_laplace` + CCD recipe. Node layout defaults to CCD (`ccd_grid()`
  + corrected `ccd_weights()`, polynomial in total `k`); tensor product opt-in
  via `integration = "grid"`. Default `log_prior_theta` is the weakly-informative
  PC + LKJ hyperprior built **per block** by `re_cov_pc_lkj_prior()` (LKJ only on
  full blocks; `correlated = FALSE` gives the diagonal log-SD prior) and summed
  over blocks, with the exact change-of-variables Jacobian.
- **`tulpa_re_cov_gibbs()`** (`R/re_cov_gibbs.R`) -- the exact debias (Bias-1):
  Metropolis-within-Gibbs (MH on per-(term,group) `b`/`beta` with Laplace-shaped
  proposals, cross-term eta bookkeeping). `Sigma_m | b_m` is an **exact conjugate
  draw**: full block -> inverse-Wishart on the matrix; diagonal block ->
  per-coordinate scalar inverse-Wishart (== inverse-gamma). Removes the Laplace
  under-dispersion that biases `Sigma` low for binary/low-count small groups.
- **`tulpa_re_aghq()`** (`R/re_aghq.R`) -- a deterministic alternative debias:
  replaces each per-group Laplace integral with `n_quad`-point adaptive
  Gauss-Hermite quadrature (`n_quad = 1` is the joint Laplace; higher reduces
  small-cluster variance attenuation). The integration core is
  **structure-agnostic**: it integrates an abstract per-group conditional
  log-likelihood `ell_g(b)` against `N(b; 0, Sigma)` given a `b`-space oracle,
  so the grouping / quadrature / log-Cholesky `Sigma` / LKJ / marginal-Hessian
  machinery is shared across every structure. Three input forms select it:
  - `make_site(theta)` -- the common **single-arm, per-row-separable** case
    (`ell_g(b) = sum_i log f_i(eta_i + Z_i b)` on one linear predictor): the
    caller supplies the per-observation marginal and its first two eta
    derivatives, and the engine builds the oracle from them and the RE design.
    Handles random slopes / correlated blocks sharing **one** grouping factor.
  - `make_group(theta)` -- the **general / multi-arm** case: the caller supplies
    the per-group oracle directly (`grad_hess(g, b)` -> value/score/data-only
    observed info; `node_ll(g, B)` -> log-lik at the quadrature nodes). Arms,
    designs and observation granularity live entirely in the callback, so
    non-separable units and random effects on several coupled arms at once
    (e.g. a community N-mixture: species priors on BOTH the abundance and the
    detection coefficients, coupled through the latent count) integrate with no
    engine change. `re_terms` then carries only the covariance structure
    (`n_coefs` / `correlated` / `n_groups`); the per-observation `idx` / `Z` are
    optional.
  - `oracle` -- a **prebuilt native (compiled) oracle**, an external pointer to a
    `REGroupOracle` constructed in a consumer package's src/ via
    `LinkingTo: tulpa` against `<tulpa/aghq_oracle.h>`: the engine drives it
    directly with **no per-group / per-node round trip into R**. `re_terms` /
    `theta0` / `Sigma0` must still describe the layout the oracle exposes; the
    integration core is identical to the R-closure path. This is the production
    path for consumer-package community fitters (e.g. tulpaObs's
    `nmix_laplace_re()` passes a native `NMixCommunityOracle`).
  Supply exactly one of `make_site` / `make_group` / `oracle`. Fixed params + log-Cholesky
  `Sigma` coords are optimized jointly on `sum_g log M_g`; SEs from the
  exact-marginal Hessian. Optional `lkj_eta > 1` penalizes a weakly-identified
  correlation off the boundary without shrinking the marginal SDs. Distinct
  from `agq_fit()` (`R/agq.R`), which is intercept-only RE -- but its built-in
  `binomial`/`poisson`/`gaussian` densities are now the **shared compiled GLMM
  oracle** (`cpp_glmm_oracle_make`, `src/glmm_oracle.h`): a single C++ source of
  truth that `agq_fit()` (`Z = 1` intercept), the single-arm `make_site` path
  here, `tulpa_re_cov_nested(n_quad > 1)` and the Gibbs sweep all consume,
  replacing the per-fitter R density closures (`.agq_loglik_elt()` /
  `.agq_score_info()`, removed). The gaussian residual variance is `phi =
  sigma_eps^2`; binomial / poisson ignore `phi`. Recovery / equivalence:
  `test-re-aghq.R` (single-arm), `test-re-aghq-multiarm.R` (make_group == make_site
  at d=1/d=2, two-arm N-mixture oracle FD-checks + end-to-end).

Both summarize through the shared `.re_cov_derived_summary` over the per-block
covariance layout (weighted quantiles == sample quantiles at equal weight) and
expose the generic `tulpa_fit` accessors: each returns `draws` (fixed-effect
posterior -- the nested path mixture-samples `N(beta_k, Vb_k)` over the weighted
nodes, the Gibbs path uses its `beta_draws`) plus `means` / `param_names` /
`process_info`, while the `Sigma` posterior stays in `posterior`. With one block
the parameter names are bare (`sigma_1`, `rho_12`, ...); with several they are
prefixed by the block label (`g.sigma_1`, `h.sigma_1`, ...). `Sigma_mean` is a
matrix for one block, a named list for several. Both also report the **per-group**
posterior through `ranef()` (#264): the Gibbs path records the `b` its sweep
samples (`fit$re`, row-aligned with the `beta` draws, so `posterior_predict()`
picks it up too) and summarizes it empirically; the nested path retains each
node's Gaussian per-group posterior (`fit$re_nodes` / `fit$re_var_nodes`, from
the inner solve's `return_re_cov` blocks) and reports the exact moments and
CDF-inverted quantiles of the weighted mixture via
`.nl_gauss_mixture_summary()` -- the continuous counterpart of
`.nl_wtd_quantile()`, carrying both the within-node curvature and the `Sigma`
uncertainty. The AGHQ inner marginal (`n_quad > 1`) integrates each group out
instead, so that fit carries `ranef_unavailable` (a reason string `ranef()`
errors on) rather than an empty table. Tests: `test-re-cov-nested.R`,
`test-re-cov-gibbs.R`, `test-re-cov-recovery.R`, `test-re-cov-prior.R` (Jacobian
vs finite differences, diagonal + joint priors), `test-ccd-grid.R`,
`test-tulpa-re-cov-frontdoor.R` (single, diagonal, multi-term routing),
`test-ranef-re-cov.R` (per-group reporting, mixture summary vs Monte Carlo,
block ordering, cross-backend agreement).
**Status:** fully wired through the `tulpa()` front door. When any RE term
carries slopes (no scalar `sigma_re` to condition on), `mode = "laplace"`
redirects to `re_cov_nested` (default) or `re_cov_gibbs`
(`control$re_cov = "gibbs"`) and treats **every** RE term as a covariance block
-- correlated, uncorrelated `(... || g)`, multiple terms, and any accompanying
`(1 | g)` (a 1x1 block); nothing is silently conditioned at `sigma_re = 1`.
Under `mode = "laplace"` a plain random-intercept-only model keeps the
scalar-`sigma_re` design path: that mode is the one door that means "condition
on the scale I supplied". The TIER modes do not -- `mode = "structured"` and
`mode = "auto"` integrate an unsupplied RE scale for a scalar `(1 | g)` as much
as for a slope (gcol33/tulpa#787), because neither has a dataset-implied
`sigma_re` to condition on.

### Generic S3 Methods and Diagnostics

Implemented in `R/methods_generic.R` (`coef`, `confint`, `vcov`, `logLik`,
`summary`, `plot`, `tidy`, `glance`, `ranef`), `R/diagnostics_generic.R`
(`compare_models`, `model_average`, `spatial_range`, `temporal_corr`), and
`R/diagnostics_sim.R` (`moran_i`, `durbin_watson`, `tulpa_variogram`,
`pit_residuals`, `test_uniformity`, `test_dispersion`, `test_outliers`,
`test_zero_inflation`, `check_model`).
Model packages inherit via `class = c("model_fit", "tulpa_fit")`.

## Engineering history

Detailed forensic write-ups of specific closed issues — derivations,
measured coverage / k-hat / timing numbers, why a fix was correct rather
than a regression — live in `ENGINEERING_HISTORY.md` (repo root, tracked,
not auto-loaded). Covers: nested-Laplace evidence and hyperpriors,
gamma_3/gamma_1 inner-Laplace reliability, outer Pareto-k-hat and its
candidate dispatch, the fixed-effect mixture read and hyperparameter axis
read, placement/pilot grids, SBC (prior- and posterior-predictive),
per-cell fixed-effect retention, PD enforcement, the Type-IV/RW2 spatial
metrics, and checkpoint/resume. Grep it by issue number (`gcol33/tulpa#NNN`)
or by subsystem before touching any of those areas — the "why" is there,
not re-derivable from the diff alone.

### Matrix CHOLMOD Fix

tulpa's `R_init_tulpa` calls `M_cholmod_start` which requires Matrix's
CHOLMOD stubs. Fixed by adding `@importFrom Matrix sparseMatrix` to
`tulpa-package.R` so Matrix DLL loads before tulpa's init.

## Extensibility: Custom Latent Blocks

This shipped. The real API is `tgmrf()` (R closures for `Q`/`mu`/`log_prior`)
and `tgmrf_cpp()` (a user `.cpp` compiled via `sourceCpp` against
`inst/include/tulpa/`, keyed by SHA256 + ABI), consumed as a block through
`latent()` and fit at any tier via `tulpa_tgmrf(mode = "imh"/"nuts"/"vi"/"nuts_joint")`
or on the nested-Laplace front door via `latent()`. See `?tgmrf_cpp`,
`vignettes/tgmrf.Rmd`, and `inst/examples/`. The names below
(`tulpa_custom_latent()`, `tulpa_fit(..., tier=)`) are the original sketch and
never shipped under those spellings.

For latent structures not provided by the engine (custom GMRFs, novel
spatial priors, exotic temporal kernels), users supply a templated C++
snippet that tulpa compiles on-the-fly. Same machinery as `LikelihoodSpec`
+ `LinkingTo: tulpa`, but with an ad-hoc entry point — no full model
package required.

User writes templated C++ that compiles against tulpa's AD types:

```cpp
template <typename T>
Eigen::SparseMatrix<T> my_Q(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> my_mu(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);

template <typename T>
T my_log_prior(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta);
```

User binds it in R:

```r
custom <- tulpa_custom_latent(
  cpp_file   = "my_block.cpp",
  theta_init = c(1, 1),
  graph      = my_graph
)

fit <- tulpa_fit(y ~ x + latent(custom), data = d, tier = "laplace")
```

tulpa compiles via `Rcpp::sourceCpp` against `inst/include/tulpa/`,
registers the block in the latent-structure registry, and inference
layers pick it up automatically. Because the user code compiles with
tulpa's templated AD types (`A`, `A_r`), the block works under **any
tier including NUTS** — no R callback, no broken gradient chain.

Comparison:
- INLA `rgeneric`: R callback, no AD, Laplace tier only.
- INLA `cgeneric`: C function, no AD, faster but no exact-MCMC support.
- Stan: full DSL + parser + codegen for the entire model.
- TMB: templated C++ snippet, autodiff via CppAD — closest analog.

Cost: extends existing `LinkingTo: tulpa` machinery with a `sourceCpp`-
driven entry point. No DSL, no parser, no codegen.

## Origin

Engine extracted from numdenom (82K lines, faster than Stan on all 18
benchmarks); numdenom was then renamed tulpaRatio as the engine became the
hub of the `tulpa*` ecosystem.
Name: Twin Peaks reference + acronym (Templated Unified Library for Posterior Approximation).
