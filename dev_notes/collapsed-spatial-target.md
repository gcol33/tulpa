# Collapsed spatial sampling

I use full-field NUTS for Poisson, Bernoulli and other non-Gaussian spatial
likelihoods. `parameterization = "auto"` selects the existing invertible
field transports. The chain evaluates the joint density of the stated
model, including its spatial prior.

For a conditionally Gaussian likelihood, NUTS can integrate the spatial
field analytically. Each retained outer row receives a draw from its exact
Gaussian conditional. R fits carry the complete joint vector in `draws`
and the sampled outer vector in `sampler_draws`. `log_prob`, `energy`,
`final_position` and `inv_metric` refer to the outer marginal chain.
`model_inputs` describes the stored joint draws. Resume inputs use the
outer layout and the original collapsed spatial specification.

Every NUTS/HMC R fit reports `target`, `inference_tier` and `inference_mode`.
The C++ `NUTSResult` carries a `SamplingTarget` enum and `inference_tier`;
`sampling_target_name()` provides its description. The Gaussian spatial
marginal is Tier 1. An uncorrected Laplace spatial marginal is Tier 2 and
the sampler refuses it with guidance to full-field NUTS.

A custom `LikelihoodSpec` must set `gaussian_spatial_marginal = true` to
declare that its log likelihood is quadratic in the field, that all field
dependence enters through eta, and that extra priors are field-independent.
Its double and arena eta-weight callbacks must return the exact score and
observed curvature. Likelihood callbacks and extra priors must supply their
arena derivatives. This declaration is a mathematical contract owned by
the likelihood provider. The built-in Gaussian identity-link likelihood
with fixed residual dispersion supplies it. The integration is exact for
the supplied prior, including an NNGP prior when that is the chosen model.

The field integral at fixed outer parameters has Gaussian precision
`H = I + F' W F`, mean `H^-1 F' W (y - eta_outer)` in prior-whitened
coordinates, and log integral `log_joint(mode) + d/2 log(2*pi) - logdet(H)/2`.
For non-Gaussian likelihoods the same mode-and-curvature expression is a
Laplace approximation. Increasing chain length retains its deterministic
integration error.

I checked the two consumers before selecting the refusal. tulpaRatio's
`R/engine_sampler.R:engine_sampler_inputs()` converts collapsed spatial
requests to the full-field automatic transport. Its direct bridge
`src/spatial_model_data.h:apply_spatial_model_data()` can still set the
collapsed flags, and `tests/testthat/test-specs-spatial-marginal.R` calls
that bridge directly for a binomial likelihood. Those direct non-Gaussian
requests now receive the refusal. Its gradient fixtures also set collapsed
flags. In tulpaObs, I found no assignment that enables a collapsed spatial
field; `src/occu_fit.cpp` only reports the layout flag. Its NUTS runners
construct ordinary joint layouts. The round-22 report retains the search
output and the exact call sites that require rebuilding for ABI 61.

Before editing the sampler, I measured C-ABI Gaussian fits with 3,000 total
iterations, 1,000 warmup iterations, target acceptance 0.95, maximum tree
depth 9, automatic metric and seed 1177. Each field had three observations
per node, two fixed effects and unit residual SD. Both methods received
the same data, design, priors and neighbor graph. I ran one fit at a time.
Timings include setup and conditional field reconstruction. The minimum
bulk ESS among the fixed effects and hyperparameters is reported per
elapsed second. These are single-seed local measurements under shared
machine load.

| Field | Nodes | Full seconds | Collapsed seconds | Full ESS/s | Collapsed ESS/s |
|---|---:|---:|---:|---:|---:|
| ICAR | 8 | 0.33 | 0.48 | 720.6 | 1012.0 |
| BYM2 | 8 | 0.77 | 1.02 | 635.8 | 370.4 |
| GP | 8 | 1.67 | 1.87 | 218.3 | 129.7 |
| ICAR | 32 | 1.10 | 5.82 | 36.6 | 117.0 |
| BYM2 | 32 | 3.75 | 27.64 | 147.7 | 16.3 |
| GP | 32 | 10.55 | 13.48 | 50.8 | 22.3 |

All twelve chains had zero divergences. The ICAR outer chain benefited
from integration on these fixtures. BYM2 and GP favored full-field
sampling in ESS per second. These measurements establish the tested
costs; they do not establish a general scaling result or posterior
agreement for non-Gaussian marginal chains.

The collapsed implementation uses dense field algebra. Each Newton step
assembles a `d x d` precision and factors it; BYM2 has `d = 2S`. Full-field
NUTS evaluates the joint likelihood and prior at each leapfrog step, with
its existing field transports and metric adaptation. Importance correction
would additionally require an inner proposal and M full latent-particle
density evaluations at each retained outer row, followed by the existing
PSIS reliability gate and whole-row resampling. The shared estimator in
`R/tgmrf_is.R` and `.TGMRF_IS` remain unchanged in this fix. I selected the
existing exact joint sampler for non-Gaussian fields because production
consumer requests already use it and its measured cost is affordable on
the common fixtures. A future demand for native non-Gaussian marginal
sampling would require the full particle correction and its joint-moment
and quantile validation.

References and their computational role:

* [Rue, Martino and Chopin (2009)](https://www.statslab.cam.ac.uk/~rjs57/RSS/0708/Rue08.pdf):
  conditional modes, curvature factorizations and Laplace integration.
* [Gomez-Rubio and Rue (2018)](https://doi.org/10.1007/s11222-017-9778-y),
  Section 4.2: deterministic approximate marginal likelihoods define an
  approximate stationary distribution. Each outer proposal requires a
  conditional integration.
* [Vehtari et al. (2024)](https://www.jmlr.org/papers/v25/19-556.html):
  PSIS stabilizes importance ratios and diagnoses their tail reliability.
  Particle density evaluations precede tail fitting and resampling.
* [Margossian et al. (2020)](https://arxiv.org/html/2004.12550): an
  adjoint-differentiated Laplace approximation reduces gradient cost and
  can improve geometry. Their approximation is assessed against full HMC.
* [Alenlov, Doucet and Lindsten (2021)](https://www.jmlr.org/papers/v22/19-486.html):
  pseudo-marginal HMC is possible on an extended state space. Fresh random
  likelihood estimates at each evaluation do not supply this engine's
  deterministic NUTS target. That extended-space construction would require
  a separate sampler implementation.
