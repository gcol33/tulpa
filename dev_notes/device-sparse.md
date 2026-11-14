# Sparse factors

`SparseCholeskySolver` owns policy selection for the existing sparse Laplace
drivers. `analyze()` selects `sparse_cholesky`, variant zero, using the matrix
order and the shared device settings. A CPU choice retains Matrix's CHOLMOD.
A CUDA choice retains a `device::SparseFactor`. cuDSS supplies memory planning
and the ordering used to build the selected-entry pattern. Ordinary CUDA
factorization and selected inversion perform no CHOLMOD analysis or numeric
factorization.

`factorize()` accepts new values on the analyzed pattern. `solve()` accepts
column-major blocks of right-hand sides. `log_determinant()` reads twice the
sum of logarithms of cuDSS's positive Cholesky diagonal. Matching/scaling and
pivot perturbation are disabled. A non-positive minor returns `false`; its
device info is cleared before the next numeric attempt. Failed solves fill
NaN. Device API errors travel through `Status::read` and the enclosing policy.

`selected_inversion_diagonal()` solves blocked identity columns on the device.
`selected_inversion_full()` computes the exact inverse on the symbolic
simplicial fill pattern in the cuDSS ordering. Its elimination tree and row
reaches use structure only. The returned lookup preserves original-coordinate
access. NVRTC kernels construct right-hand sides and gather selected entries.
The dense workspace holds at most 64 columns; the full dense inverse is never
allocated. `device::SparseFactor::inverse_diagonal(indices, out)` also accepts
a subset, including repeated indices, for callers needing fewer marginals.

cuDSS 1.x exposes the factor diagonal through `CUDSS_DATA_DIAG`. Its public
data parameters expose no sparse factor entries and no selected inverse.
Copying the factor for CPU Takahashi and applying device Takahashi to cuDSS's
supernodes therefore have no documented implementation route. The exact unit
solves are compared with CHOLMOD plus the existing CPU Takahashi recursion.
The timing gate covers factorization, log-determinant, solves and inverse
diagonal together. The shared table retains CPU automatic dispatch wherever
that complete CUDA operation has no sustained measured win.

The square-root operations used by Gaussian draws and triangular-whitened
quadratic forms retain their CHOLMOD implementation. When requested from a
CUDA solver, they analyze and factor the current values on CPU once and record
`sparse_square_root=cpu`. Simplicial indefinite LDL/inertia probes use CPU.

## Multi-cell hooks

`device::SparseFactor` in `src/device_sparse.h` is the device implementation
used by the shared solver. `analyze(n, p, i, x, stream)` consumes packed sorted
lower CSC with 32-bit indices and double values. Symmetry lets that same
storage serve as upper CSR. `refactor(x, positive_definite)` replaces values
without re-analysis. `solve(b, x, columns)`, `inverse_entries(p, i, perm, z)`
and `inverse_diagonal(indices, z)` reuse the numeric factor. `analyses()` and
`stream()` expose the lifecycle for a scheduler.

Each factor owns its cuDSS handle, configuration, symbolic/numeric state,
matrix descriptors and device buffers. An omitted stream creates an owned
nonblocking stream; an explicit stream is borrowed and must outlive the
factor. `SparseCholeskySolver::set_stream()` binds a borrowed stream before
analysis, invalidating existing state. One worker owns each factor. Independent
workers may execute different factors concurrently on their own streams.
Methods complete their stream before returning host results; these are
synchronous host hooks. They do not enqueue an entire outer-grid batch.
The native lifecycle fixture also checks a borrowed stream, repeated diagonal
indices, and rejection of a selected-pattern column pointer beyond its buffer.

## Blackwell sanitizer compatibility

NVIDIA acknowledges invalid shared-memory reports from the CUB masked bulk
copy instruction on Blackwell with Compute Sanitizer 13.2 or earlier in
[NVIDIA/cccl#8288](https://github.com/NVIDIA/cccl/issues/8288). The sparse
fixture encounters that report inside cuDSS's analysis kernels. The local
verification uses Compute Sanitizer from CUDA 13.4. The reproducer is
`dev_notes/repro-sanitizer-blackwell.R`; its optional argument is the library
containing this tulpa build. Run it through memcheck with child-process
instrumentation enabled. All engine kernels remain instrumented.

The shared `Context` owns driver stream creation, destruction and synchronization.
Its host upload completes the default-stream transfer before returning, so
nonblocking solver streams consume the uploaded values. The NVRTC cache and
launch lookup use the existing short `Lease`; cuDSS factor/solve calls use
their object-owned handles without holding the global library-handle mutex.

No installed consumer struct changes. The engine ABI stays 63.

## Reproduction

Set `CUDSS_DIR` to a cuDSS 1.x runtime root. Build and install into a private R
library. Run `tools/bench-device.R <output.csv> sparse_cholesky` from the
package root. It checks 1,000 through 100,000-node triangular finite-element
SPDE precision matrices, records paired raw samples in the shared CSV schema,
and writes a `.sparse-stages.csv` file separating analysis, factor/logdet,
solve and inverse. The default thread configuration is retained. Device
equivalence tests print the numerical scale and relative tolerance.

## References

* [cuDSS functions](https://docs.nvidia.com/cuda/cudss/latest/functions.html)
  and [data types](https://docs.nvidia.com/cuda/cudss/latest/types.html):
  analysis/numeric/solve lifetimes, CSR descriptors, stream binding, device
  info and diagonal query. The runtime C ABI declarations follow cuDSS 1.0.0
  headers; headers and binaries are external to the package.
* [CUDA synchronization behavior](https://docs.nvidia.com/cuda/cuda-driver-api/api-sync-behavior.html):
  pageable synchronous host copies can return after staging, before DMA completes.
  The shared upload completion implements the ordering required by independent streams.
* [Rennich, Stosic and Davis](https://doi.org/10.1016/j.parco.2016.06.004):
  resident factors, elimination-tree structure and amortizing GPU transfer
  and launch costs. The Matrix-provided CHOLMOD build remains the CPU solver.
* [Gaedke-Merzhaeuser, van Niekerk, Schenk and Rue, 2023](https://doi.org/10.1007/s11222-022-10192-1):
  independent outer cells and separating sparse factorization, solves and
  selected inversion. The title is *Parallelized integrated nested Laplace
  approximations for fast Bayesian inference*. The later
  [INLA_DIST paper](https://arxiv.org/abs/2303.15254) has Krainski and Janalik
  as coauthors and discusses a blocked GPU solver for structured systems.
  The generic sparse interface here does not assume that block structure.
* Takahashi, Fagan and Chin (1973), *Formation of a sparse bus impedance
  matrix and its application to short circuit study*: the existing CPU
  selected-inverse recurrence is the reference. [Rue and Martino
  (2007)](https://doi.org/10.1016/j.jspi.2006.07.016) connects sparse inverse
  entries to latent-Gaussian marginal variances. Device unit solves compute
  those same inverse entries.
