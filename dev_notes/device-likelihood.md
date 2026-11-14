# Resident likelihood evaluation

`LikelihoodSpec::device_fn` describes an observation kernel through
`tulpa/device_likelihood.h`. A null slot uses the existing CPU dispatcher.
ABI 64 covers the appended slot. A consumer must rebuild against the engine;
the existing ABI guard rejects an older allocated spec.

The callback receives the model, layout, and response. It always supplies
`variant`, `constants`, and `parameter_indices`. With `initialize = true`, it
also supplies immutable column-major response channels and CUDA source defining
`tulpa_likelihood_row(response, n, i, eta, constants, value, score, curvature)`.
`score` has one entry per predictor and `curvature` is a row-major square
negative-Hessian block. Extra parameter indices add predictor channels after
the model's predictors and its sampler ZI predictor. Their scores supply direct
likelihood derivatives in those parameter coordinates. The combined channel
count is bounded by the engine's eight-predictor observation buffer.

A consumer chooses a distinct variant for its measured workload. The default
variant is uncalibrated. Forced CUDA accepts it; automatic routing uses CPU
until `inst/device/benchmarks.csv` supplies a sustained measured crossover.
`likelihood` measures value and parameter gradient. `likelihood_weights`
measures the Laplace path including predictor, score, and curvature downloads.
Both use the existing policy and the same settings table.
The operation registry stores all required capabilities. Likelihood operations
require both NVRTC and cuBLAS; the shared policy uses CPU in automatic mode
when either is unavailable and reports the library error for forced CUDA.

`device_likelihood.cpp` owns the resident design, response, offsets, projection,
transpose segments, coefficient buffer, and derivative buffers. The family
kernel assembles predictors and evaluates value, score, and curvature together.
Block sums reduce values. cuBLAS computes the dense design transpose product.
One block per sparse transpose segment reduces each reconstructed field score,
with a fixed reduction order and zero floating-point atomic adds. A fit policy
owns workspaces per model, layout, and host worker. Laplace grid cells refresh
their field projection while retaining the observation arrays on the device.
Response preparation caches count-data normalizers on the main thread, so
worker payload preparation performs no R-math calls.

The NUTS path initializes priors and reconstructed fields through the existing
generic posterior. Its observation projection is obtained by evaluating the
existing named-effect and spatiotemporal assembly helpers with affine symbols.
The spatial field is combined in its physical coordinates before this step.
An arena custom backward propagates the reduced scores through the existing
field transformations and priors. Observation densities and fixed-effect
matrix products create no arena nodes on this route. Reconstructions already
expressed as per-observation host vectors retain their reconstruction cost.
Collapsed spatial targets retain their specialized integration path.

Spec Laplace builds its projection from the same templated `compute_eta_spec`
used by CPU evaluation. The Newton loop receives the device predictors and
retains the fused scores and curvature for its next scatter. Its sparse or
dense Hessian assembly and prior assembly use their existing implementations.

The device dispatcher precedes the full-gradient and arena choices after the
coordinate-transform and collapsed-target dispatch. Numerical mode always uses
the CPU double objective. `ensure_gradient_verified()` resolves the active
device gradient, compares it with both existing numerical references, and
selects numerical mode on disagreement. A direct-extra-parameter consumer
fixture and an intentionally incorrect device score exercise this contract.

`count_math.h`, `precision_math.h`, and `precision_coefficients.h` are compiled
by both the host compiler and NVRTC. The runtime compiler receives the installed
header text as virtual headers, including aliases for their relative includes.
`device_math.h` supplies execution annotations and scalar math adapters. The
negative-binomial log-mean density and beta-binomial rising correction are
shared by CPU, arena, and device evaluation. Kernel strings contain no copied
special-function implementation.

The stream benchmark launches independent chains on separate streams and
compares them with a chain dimension in one launch. It shares observation
arrays across chains and records elapsed time including synchronization and
gradient downloads. Production NUTS retains independent chain trajectories;
the device layer's shared-handle lease serializes host workers' evaluations.
The benchmark measures concurrent streams within one lease. A sampler with
lockstep trajectories would require its own scheduling comparison.

`tools/bench-device-likelihood.R` records raw paired timings, numerical scales,
errors, setup time, arena timing, and chain scheduling. It probes actual arena
node use before projecting capacity growth, and reads currently available
host and device memory before each size. Skipped rows retain their reason and
the available byte counts. The measured CPU comparator is the existing fused
per-row closed form with dense and indexed gradient accumulation; the arena
comparator differentiates the built-in observation callback. These are
likelihood evaluations, excluding priors and fit setup.
The standard fixture uses three dense design columns per predictor and five
indexed field groups. Its four-chain scheduling measurements share immutable
observation arrays within one workspace. Production host workers own separate
resident workspaces and use the existing serialized device lease.

## Consumer allocation sites

* tulpaRatio: `src/lik_dispatch.cpp::build_ratio_likelihood_spec`, consumed by
  `src/tulpa_bridge.cpp`; `src/engine_fixture.h` allocates the test fixture.
* tulpaObs: `src/nuts_engine.h` allocates sampler specs; `src/occu_fit.cpp`
  allocates the single-season occupancy spec; `src/occ_nested_likelihood.cpp`
  owns the nested-likelihood bundle's spec.

Their existing assignments leave `device_fn` null after rebuilding. No
consumer repository is changed by this patch.

## References

* Stan Math's [Bernoulli-logit](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/prim/bernoulli_logit_glm_lpmf.hpp),
  [Poisson-log](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/prim/poisson_log_glm_lpmf.hpp),
  [negative-binomial-log](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/prim/neg_binomial_2_log_glm_lpmf.hpp),
  and [normal-identity](https://github.com/stan-dev/math/blob/develop/stan/math/opencl/prim/normal_id_glm_lpdf.hpp)
  implementations: I took the fused predictor/value/score evaluation, reduced
  values, and separate dense transpose-score product. Tulpa's shared precision
  primitives supply its distribution arithmetic.
* [Ciglaric, Cesnovar and Strumbelj, IWOCL 2020](https://www.iwocl.org/wp-content/uploads/iwocl-syclcon-2020-ciglaric-26-slides.pdf):
  I took fused observation work and buffer reuse to reduce launches and
  intermediate global-memory traffic. Compilation uses dev21's NVRTC route.
* [NumPyro MCMC](https://num.pyro.ai/en/stable/mcmc.html): I took the explicit
  distinction between vectorized chains and independently parallel chains.
* [BlackJAX multiple chains](https://blackjax-devs.github.io/blackjax/examples/howto_sample_multiple_chains.html):
  I took the chain dimension and the requirement to measure the scheduling
  choice. Its discussion of variable trajectory lengths limits what a kernel
  benchmark establishes about a full NUTS run.
