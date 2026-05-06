# tulpa — Inference-Method Gap List

Source: external review (ChatGPT, 2026-05-06) cross-checked against the
current `src/` and `R/` tree. Each item lists what's already there, what's
missing, and where it would slot in. The refactor punch list is in `TODO.md`;
this file is method-coverage only.

Numbering = priority within tulpa's "fast Bayesian GLMM inference with sparse
structure" framing, not within the reviewer's ordering.

---

## Tier A — Highest payoff for tulpa's stated direction

### A1. Expectation Propagation (EP)
- **Status:** missing entirely.
- **Why it matters:** for latent-Gaussian GLMMs with non-Gaussian likelihoods,
  EP gives a Gaussian posterior approximation that is typically tighter than
  Laplace (matches marginal moments rather than mode + curvature). Pairs
  cleanly with the existing nested-Laplace + CCD hyperparameter loop —
  reuse `R/nested_laplace.R` and `R/ccd_grid.R` essentially as-is, swap
  the inner Laplace-mode call for an EP site update.
- **Where it lives:** new `src/ep_core.cpp` mirroring `laplace_core.cpp`;
  exposed via `LikelihoodSpec` (needs a per-site moment-matching callback
  on top of the existing `EtaWeightsFn` IRLS callback).
- **Tier:** Tier 2 (Structured) — same epistemic class as Laplace.
- **Math-first check:** for Gaussian + Bernoulli/Poisson/binomial sites the
  cavity tilt has closed-form moments via 1-D Gauss–Hermite — no inner
  Newton needed. Confirm before reaching for generic numerical site updates.

### A2. Variational Laplace / Laplace-corrected VI
- **Status:** missing. Mean-field, low-rank, and full-rank VI exist
  (`src/vi_*.h`); Laplace exists; the *combination* (use Laplace at the
  VI mode for moment correction, or use VI to initialise the Laplace
  Newton solve) does not.
- **Why it matters:** cheap upgrade — Tier 3 → Tier 2-ish. VI gives a fast
  starting point; one Laplace-Newton step at the VI mode upgrades the
  covariance. Standard in Bayesian deep learning + neuroimaging (SPM
  uses this exact pattern under the "Variational Laplace" name).
- **Where it lives:** `R/fit_laplace.R` gets an `init = "vi"` path that
  calls the existing VI then hands the mode + scale to the Newton loop.

### A3. Pathfinder / Laplace + importance sampling
- **Status:** missing. We have Laplace mode-finding and we have HMC; we do
  not currently use the Laplace fit as either an HMC warm-start *with
  importance reweighting* or as a Pathfinder-style multi-path ELBO
  minimiser.
- **Why it matters:** Pathfinder is the standard "fast initialisation +
  diagnostic" tool in modern HMC stacks. For Tier 2 → Tier 1 escalation
  it's nearly free given that we already have L-BFGS in
  `src/hmc_gp_lbfgs.h`.
- **Where it lives:** new `R/pathfinder.R` (single-path first; multi-path
  later). Reuses the existing Laplace solver and the existing L-BFGS code.

### A4. Laplace as MCMC proposal (independence MH)
- **Status:** missing. The Laplace fit is currently used as a final
  approximation and as a warm-start for HMC mass-matrix init, but never
  as a proposal distribution inside an MH accept/reject loop.
- **Why it matters:** for moderate-dimensional posteriors that are
  near-Gaussian, an independence MH with Laplace proposal converges in
  hundreds of iterations and is embarrassingly parallel across chains —
  much cheaper than full NUTS. Lowest-effort genuinely-new MCMC kernel
  given existing infrastructure.
- **Where it lives:** new `src/imh_laplace.cpp`. Tier 1 (Exact) backend.

---

## Tier B — Standard MCMC kernels missing from the lineup

### B1. MALA (Metropolis-Adjusted Langevin)
- **Status:** missing. SGLD exists (`src/sghmc_sampler.cpp`) but it is
  *unadjusted* and stochastic-gradient — not the same algorithm.
- **Why it matters:** natural stepping stone between random-walk MH and
  HMC; useful baseline when HMC is overkill or when the target has
  awkward tails. Reuses the existing analytical-gradient infrastructure
  in `src/hmc_gradient_*` — implementation is essentially "leapfrog
  with L=1 + MH step".
- **Where it lives:** new `src/mala_sampler.cpp`. Tier 1.

### B2. Riemannian HMC (full RMHMC, not just SoftAbs)
- **Status:** partial. SoftAbs metric is in `src/hmc_nuts_softabs.cpp`
  but the full RMHMC kernel (position-dependent metric with the
  generalised leapfrog and metric-correction terms) is not.
- **Why it matters:** for hierarchical posteriors with funnels (the case
  HMC is *worst* at), RMHMC genuinely fixes the geometry rather than
  papering over it with mass-matrix adaptation. Hard to get right —
  not the next thing to ship, but worth scoping.
- **Where it lives:** extension of `src/hmc_nuts_softabs.cpp`. Needs an
  AD path for the metric Jacobian.

### B3. Generic blocked Metropolis-within-Gibbs
- **Status:** partial. Component-wise Gibbs exists for ICAR/CAR/BYM2/HSGP
  in `src/gibbs_spatial*`, but there is no *generic* MwG framework that
  takes user-defined blocks and a per-block proposal kernel.
- **Why it matters:** for hierarchical models where some blocks are
  conjugate (Gibbs) and others are not (MH), MwG is still the workhorse.
  Currently if a model doesn't fit one of the spatial Gibbs templates
  the user is forced into full HMC.
- **Where it lives:** new `R/mwg.R` driver + `src/mwg_kernels.h`. Tier 1.

---

## Tier C — Latent-variable inference

### C1. MCEM (Monte Carlo EM)
- **Status:** missing. Only EM + Laplace exists (`R/em_laplace.R`).
- **Why it matters:** when the E-step expectation has no closed form and
  Laplace is too crude, MCEM draws latent variables via a short MCMC
  chain. Slots in beside the existing EM driver as another `e_step_*`
  callback.
- **Where it lives:** `R/em_mc.R` (sibling to `R/em_laplace.R`). Reuses
  the existing E-step / M-step interface.

### C2. SAEM (Stochastic Approximation EM)
- **Status:** missing.
- **Why it matters:** for nonlinear mixed models and other settings
  where the E-step is intractable, SAEM has stronger convergence
  guarantees than MCEM with smaller per-iteration MC samples. Standard
  in pharmacokinetics (NONMEM, Monolix) and ecology mixed-effects
  models.
- **Where it lives:** `R/em_saem.R`. Same E/M scaffolding as `em_laplace`.

### C3. Adaptive Gaussian quadrature (AGQ)
- **Status:** missing.
- **Why it matters:** the gold-standard exact integrator for low-
  dimensional random-effect blocks (e.g. one or two RE per cluster).
  This is what `lme4::glmer(..., nAGQ = N)` does and it materially
  outperforms Laplace for small clusters with non-Gaussian likelihoods.
  Lowest-hanging accuracy upgrade for the Tier-2 path.
- **Where it lives:** `src/agq.cpp` plus an `n_quad` argument on the
  Laplace-spec entry points; falls back to Laplace at `n_quad = 1`.
  Math-first: `statmod::gauss.quad.prob` already gives the nodes/weights;
  the integration is a per-cluster loop over the existing log-likelihood.

---

## Tier D — Particle / pseudo-marginal methods

### D1. Pseudo-marginal Metropolis-Hastings (PMMH)
- **Status:** missing. SMC sampler exists (`src/smc_sampler.cpp`) but
  not the PMMH kernel that *uses* an SMC marginal-likelihood estimate
  inside an outer MH chain.
- **Why it matters:** the standard tool for state-space models with
  intractable marginal likelihood. Less central to GLMM than to
  state-space, so lower priority for tulpa specifically — but the SMC
  building block is already there.
- **Where it lives:** new `src/pmmh.cpp` wrapping `src/smc_sampler.cpp`.

### D2. Particle Gibbs / conditional SMC (PG-CSMC)
- **Status:** missing.
- **Why it matters:** complement to PMMH for the same problem class —
  gives joint-state posterior rather than just hyperparameter posterior.
  Same "secondary priority for tulpa" caveat as D1.

---

## Tier E — Marginal-likelihood / model comparison

### E1. Bridge sampling
- **Status:** missing.
- **Why it matters:** the practical default for marginal likelihood
  estimation given posterior draws — needed for proper Bayes factors
  and model comparison. `R/diagnostics_generic.R` has `compare_models`
  but it relies on WAIC/PSIS-LOO, not marginal likelihood.
- **Where it lives:** new `R/bridge_sampling.R`. Pure-R is fine; no C++
  needed for the standard iterative scheme.

### E2. Thermodynamic integration / power posteriors
- **Status:** missing.
- **Why it matters:** alternative marginal-likelihood estimator,
  particularly when bridge sampling fails (multimodal posteriors,
  high dimensions). Requires running the sampler at multiple
  temperatures — slots into the existing chain runner with a
  temperature parameter on the log-likelihood.
- **Where it lives:** `R/thermodynamic.R` + a `temperature` argument
  on the existing NUTS / MALA / IMH chain entry points.

---

## Tier F — Modern / "2026-flavor" additions

These reframe the package from "exhaustive classical sampler menu" to
"fast sparse GLMM with a compiler-aware autodiff core and amortised
warm-starts". They are the items most likely to make tulpa feel
contemporary rather than complete-but-classical.

### F1. Pathfinder + flow-correction stack
- **Status:** Pathfinder is A3; the *stack* (Pathfinder warm-start →
  flow-based correction → Laplace-spec backbone) is the framing the
  package should aim for end-to-end, not as separate features.
- **Why it matters:** this is the fast-sparse-GLMM recipe that does not
  exist in R today. Stan has Pathfinder, INLA has the Laplace + CCD
  spine, brms has none of the modern compiler stuff. Owning the
  combination is a defensible niche.
- **Where it lives:** crosscut — A3 (Pathfinder), F2 (flow VI), and the
  existing `src/laplace_spec.cpp` form the three layers. No new file
  needed beyond the per-layer additions, but the README story should
  lead with this stack.

### F2. Flow-based VI (normalizing flows)
- **Status:** missing. Existing VI is mean-field / low-rank / full-rank
  Gaussian only (`src/vi_*.h`).
- **Why it matters:** for posteriors with non-Gaussian shape (skew,
  multi-modal, funnel) a small flow (planar / RealNVP / IAF) on top of
  the Laplace base measure captures geometry that Gaussian VI cannot,
  at a fraction of HMC cost. Slots in as a new VI subclass.
- **Where it lives:** new `src/vi_flow.h` + a tiny autodiff path for the
  flow Jacobian. The existing `src/vi_optimizer.h` SGD/Adam loop is
  reusable as-is.
- **Math-first check:** the change-of-variables term is
  `log|det J_f|` — for planar/radial flows this is closed-form scalar;
  no general-purpose Jacobian determinant needed.

### F3. Transport-map Laplace correction
- **Status:** missing.
- **Why it matters:** Laplace gives a Gaussian fit at the mode; a
  low-degree polynomial / triangular transport map post-corrects the
  Gaussian to match higher posterior moments. Cheaper than full HMC,
  tighter than plain Laplace. Pairs with A3 Pathfinder (Pathfinder gives
  the Gaussian, the map polishes it).
- **Where it lives:** new `R/transport_map.R` + `src/transport_map.cpp`.
  Tier 2 (Structured) — same epistemic class as Laplace.

### F4. Amortised Laplace / mode initialisation
- **Status:** missing.
- **Why it matters:** for repeated fits with the same model structure
  but different data (cross-validation, simulation studies, online
  refits), training a small neural net once to predict the Laplace mode
  + Hessian eliminates the inner Newton solve at inference time.
  Standard amortised-inference / SBI move; closest existing tool is
  `sbi` (Python).
- **Where it lives:** new `R/amortise.R` + a torch / RTorch dependency.
  Optional — only relevant for heavy-rerun workflows.

### F5. Adaptive transport HMC / transport-map MCMC
- **Status:** missing. Most relevant of the "2026 MCMC" branch.
- **Why it matters:** generalises B2 (RMHMC). Instead of a
  position-dependent metric, learn a transport map that pulls the
  posterior to a near-isotropic reference — then run vanilla HMC in
  reference space. Avoids the metric-Jacobian terms that make RMHMC
  hard to implement correctly.
- **Where it lives:** sits on top of F3 (transport map). Same map
  fitted once, then reused as either a Laplace correction (F3) or an
  HMC reparameterisation (F5). One implementation, two consumers.

### F6. Manifold methods (geodesic MC, Riemannian MALA)
- **Status:** missing.
- **Why it matters:** for parameters constrained to a manifold (e.g.
  correlation matrices, Stiefel / Grassmann for latent factors), naive
  unconstrained-space sampling distorts the posterior. Geodesic Monte
  Carlo and manifold-MALA fix this. Lower priority than F1–F5 because
  most GLMM parameter spaces are flat.
- **Where it lives:** would extend `src/hmc_nuts_softabs.cpp`. Latent
  factor models (`R/latent.R`) are the most likely consumer.

### F7. EP/VI hybrid via α-divergence
- **Status:** missing. EP is A1; the α-divergence interpolation
  (α=0 → KL → VI; α=1 → reverse-KL → EP) is its own scope.
- **Why it matters:** picks the best Tier-2 / Tier-3 fit per problem
  rather than committing to one. Active research area, modest
  implementation cost on top of A1 EP + existing VI.
- **Where it lives:** parameter on the EP solver in A1.

### F8. GPU-aware sparse autodiff path (compiler-style)
- **Status:** missing. Existing autodiff is CPU + a CUDA NNGP fragment
  (`src/gpu_cuda.h`, `src/gpu_nngp_laplace.h`) — not a compiler-style
  fused-kernel autodiff over sparse linear algebra.
- **Why it matters:** the modern PPL story (NumPyro, BlackJAX, PyMC
  with JAX) is built on autodiff-over-XLA-compiled sparse linear
  algebra, not hand-written gradient kernels. tulpa's hand-written
  vectorised gradient kernels (`src/hmc_gradient_vectorized*`) are
  fast but they are *the* code that does not scale: every new
  likelihood needs a new specialised kernel.
- **Direction:** rather than chasing JAX, the realistic move for an R
  package is to (a) tighten the existing `LikelihoodSpec` autodiff
  contract so model packages get analytical gradients for free via
  the existing tape (`src/autodiff_tape.h`), and (b) extend the CUDA
  fragment to cover the sparse Hessian solve, not just the NNGP block.
  Item #5 in the refactor TODO is the natural starting point.
- **Where it lives:** crosscut. Touch `inst/include/tulpa/likelihood.h`,
  `src/hmc_gradient_dispatch*`, and `src/gpu_cuda.h`.

---

## Recommended ordering

**Strategy: breadth first, benchmark second.** Ship every method in this
file before standing up a calibration harness. The point is to maximise
the experimental surface — once all kernels exist, ablations and
head-to-head benchmarks on hard cases (funnels, sparse Bernoulli,
near-singular spatial) become *picking winners from a buffet*, not
"hopefully NUTS-vs-Laplace is interesting". Without breadth, the
benchmark has nothing to compare.

**Implementation ordering (cheap → expensive, no item gates the next):**

Wave 1 — **shipped 2026-05-06** (6/6, 880/880 tests pass):
- ✅ **E1 Bridge sampling** — `R/bridge_sampling.R`. Conjugate-normal closed-form within 0.01 nats. Tier-agnostic post-hoc estimator, not in tier registry.
- ✅ **A4 IMH Laplace** — `R/imh_laplace.R`. Tier 1. Independence MH with Gaussian proposal centred at the Laplace mode + scale^2 * H^{-1}.
- ✅ **A3 Pathfinder** — `R/pathfinder.R`. Tier 2. Single-path: L-BFGS via `optim` + Gaussian fit at the mode + ELBO scoring. Multi-path is follow-on.
- ✅ **B1 MALA** — `R/mala.R`. Tier 1. Langevin proposal with dual-averaging step-size adaptation toward acceptance 0.574; optional `mass_diag` preconditioner.
- ✅ **C3 AGQ** — `R/agq.R`. Tier 2. One-RE intercept-only GLMM (binomial/poisson/gaussian); Golub-Welsch nodes computed inline (no `statmod` dep).
- ✅ **C1 MCEM** — `R/em_mc.R`. Sibling of `R/em_laplace.R`. Generic Monte-Carlo EM with Booth-Hobert `n_mc_growth` schedule and Rubin's-rules pooling across MC draws.

Wave 2 — moderate, new core but well-scoped:
- **A1 EP** — Tier 2; pairs with existing nested-Laplace + CCD.
- **A2 Variational Laplace** — VI mode → Laplace Newton step.
- **F2 Flow-based VI** — new VI subclass.
- **F3 Transport-map Laplace** — feeds F5.
- **B3 Generic blocked MwG** — generalises the existing spatial Gibbs.
- **C2 SAEM** — sibling of MCEM.
- **E2 Thermodynamic integration** — temperature parameter on chains.

Wave 3 — bigger refactors / second-tier value:
- **B2 Full RMHMC** — extends SoftAbs.
- **F5 Adaptive transport HMC** — sits on F3.
- **F7 EP/VI α-divergence** — extends A1.
- **F8 Sparse-autodiff tightening** — crosscut, touches `LikelihoodSpec`.
- **D1/D2 PMMH / Particle Gibbs** — wraps existing SMC.
- **F4 Amortised Laplace init** — only useful at scale; needs torch/RTorch.
- **F6 Manifold methods** — only useful for constrained-parameter models.

For each item: implement-first, benchmark-against-existing-Tier-1-second,
add a `mode = "<new-backend>"` wiring in `R/inference_modes.R` third.
The `INFERENCE_TIERS` registry in `R/inference_modes.R:63` is the
single source of truth for backend → tier mapping; every new method
must land there.

---

## Out-of-scope notes

- **Amortised inference / SBI more broadly** (neural posterior
  estimation, neural likelihood, neural ratio): F4 is the only piece
  worth pulling in for the GLMM use case. Full SBI is a different
  package — `sbi` exists in Python and the value-add for tulpa is the
  amortisation of *its* expensive inner loop (Laplace mode-find), not
  reimplementing the full SBI stack.
- **JAX / XLA port of the engine**: not realistic for an R package and
  not what F8 is asking for. The realistic version is "make our
  existing CPU+CUDA path scale via `LikelihoodSpec` autodiff so users
  don't write gradient kernels by hand", which is F8 as written.
