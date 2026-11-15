# Device operations

The engine loads CUDA libraries at run time. Package installation uses the R
C++ toolchain, without CUDA headers, import libraries, or nvcc. The implementation
is internal in `src/device.h`, `src/device.cpp`, and `src/device_exports.cpp`;
it exposes no device handles through the consumer ABI. R users call `gpu_info()`
and set `control$device`.

## Policy and errors

Each R fitting door opens a device session after validating `control`. An inner
door inherits the enclosing policy when its requested choice is `auto` or agrees
with the enclosing choice. The session restores the enclosing policy on normal
return and on error. Outer-grid workers share the policy's counters and first
failure. The process-wide active policy is an atomic shared pointer, retained
while workers execute.

Numerical helpers use this sequence:

```cpp
auto policy = tulpa::device::current_policy();
tulpa::device::Status status;
bool cuda = policy->select(operation, size, variant, status);
if (!status) { policy->fail(status); return; }
// Execute the selected implementation and read every library status.
policy->fail(status);
if (status) policy->record(operation, cuda);
```

`operations()` owns the operation-to-capabilities registry. An unknown operation
is an error. The `variant` is an exact matrix order for NNGP, the outer worker
width for Laplace batches, and zero for scalar sparse operations and axpy.
An unmeasured operation or variant has an infinite crossover in automatic mode.
Forced CUDA ignores the size gate and requires all of that operation's capabilities.
Host inference remains on CPU. Inside an outer cell, a failed device status
raises an internal `CellFailure`, caught within that worker by `run_cell`.
The same callback runs again from its supplied warm start under a CPU override.
The result carries `device_cpu_recovered` and the policy records a recovery
stage. Outside a cell, the R finalization and session exit read failed statuses
on the main thread. Numerical workers never call R error handlers.

`Policy::select`, `record`, and `fail` are the API later numerical backends call.
The drivers themselves pass through the numerical helper. The fit's
`device$stages` table records operation names and CPU/CUDA call counts.

## Context and resources

`Context::instance()` owns one retained primary context on device zero.
`std::call_once` publishes initialization and its success or failure to all
threads. The primary context interoperates with other CUDA runtime clients.
`Lease` serializes the shared library handles and pushes/pops the calling
thread's current context. `Buffer` owns a device allocation and checks copy
bounds. Destruction releases allocations, library handles, compiled modules,
and the primary-context reference. It preserves the first numerical failure.

Library calls return `Status::read(library, code, operation, detail)`. Library
symbols are resolved through `Context::symbol`; new backends use this loader
and the same checked status convention. `SparseFactor` uses the cuDSS capability
for symbolic analysis, refactorization, solves and exact inverse entries.

The CUDA 13 library names are:

| Library | Windows | Linux |
| --- | --- | --- |
| Driver | nvcuda.dll | libcuda.so.1 |
| cuBLAS | cublas64_13.dll | libcublas.so.13 |
| cuSOLVER | cusolver64_12.dll | libcusolver.so.12 |
| NVRTC | nvrtc64_130_0.dll | libnvrtc.so.13 |
| cuDSS | cudss64_1.dll, cudss64_0.dll | libcudss.so.1, libcudss.so.0 |

Windows searches the normal DLL path and the `bin/x64` and `bin` directories
of `CUDA_PATH` and `CUDSS_DIR`. Linux uses the dynamic loader's library path.
The cuDSS names cover its current and earlier shared-library ABI; the sparse
backend uses the cuDSS 1.x C ABI constants checked against the installed headers.

## Runtime kernels

CUDA sources live in `inst/device/*.cu` and are read at package load. Source
filenames identify entries in `kernel_source(name)`. NVRTC compiles a requested
source with C++17, the selected device's compute capability, and fused
multiply-add contraction disabled. Compilation logs are attached to errors.
`cuModuleLoadData` loads PTX, `cuModuleGetFunction` resolves the entry, and
`Context::launch` launches a `Launch` configuration with caller-supplied
arguments. Its cache is keyed by source text and entry name. The context owns
cached modules, and NVRTC program handles have scoped ownership.

The `axpy` operation consumes this route end to end. It is an internal numerical
primitive and verification fixture. A later family kernel adds a source file
and registry entry and uses the same compile/cache/launch machinery; no
family-specific context member or loader is needed.

## Measurement

`inst/device/benchmarks.csv` is the settings source. It stores paired CPU/CUDA
timings. The harness appends a numbered calibration and retains previous rows.
`.device_thresholds()` selects the latest calibration for each operation and
variant, takes medians
at each measured size, and chooses the first size at which the fastest CUDA
route and every larger measured size win. `stream_seconds` holds the independent
stream measurement; infinity means that alternative was unmeasured. The separate
`batch_threshold` chooses batched launches when they sustain a win over streams.
Forced CUDA exercises batched launches when the caller supplies a batch scope;
the harness supplies width one for its independent-stream comparator. With no
sustained measured win the corresponding crossover is infinite. Both decisions
come from this one table, with no size threshold copied into C++.
Batch preference applies when the scope has multiple workers. With one worker,
both CUDA comparators execute one-cell launches.

`tools/bench-device.R` checks that every registered operation has a runner.
The NNGP runner measures complete conditional-moment batches, including
covariance construction, factorization, checked transfers, factor verification,
and the CPU triangular solves. It asserts fixture dimensions and CPU/CUDA
agreement before timing, warms both routes, alternates their order, and writes
paired samples. The axpy runner includes allocation, copies, kernel execution,
synchronization, and cleanup. Timing uses the native steady clock.

The `sparse_cholesky` gate uses whole Poisson nested-Laplace SPDE fits with
CUDA factorization and the separately selected CPU inverse. Fixture construction
is outside the wall-time measurement. Primitive factor, solve and inverse
durations are retained in a sidecar; one CPU factor plus the inverse calibrates
`sparse_selected_inverse` against the device inverse on its resident factor.
Every native call's per-cell Newton
counts are retained in the whole-fit RDS sidecar.

`laplace_dense` compares CPU, independent per-factor cuSOLVER handles and
nonblocking streams, and coalesced `potrfBatched` launches. Pointer, status,
factor and RHS storage are reused. `laplace_full_csc` measures the same CUDA
factor through the CSC solver interface against its CHOLMOD CPU route. It
covers full patterns and precisions filled from dense Hessian storage. Entries
omitted from the stored pattern remain zero in the device's dense matrix.
The two interfaces have separate measured gates and share `DenseFactor`.
The factor is defined in `device_dense.h` and owns one host/device numerical
state. `DenseCholeskyScratch::factor` serves fixed-hyperparameter spec fits and
outer cells through `DenseFactor::factorize`. A batch scope supplies the
Laplace operation and worker-width variant; the same factor storage reaches
the coalesced factorization and resident solves. CPU factorization uses Eigen
LLT. Other dense workloads retain their covariance, projection and triangular
solve operations on this type.
`laplace_dense_inverse` measures factorization plus inverse diagonals against
CHOLMOD and Takahashi. It uses the same retained dense factor and blocked
triangular solves. A variance-only solver selects this operation for dense
inputs, independently of the Newton factor gate. Its benchmark passes zero
ordinary RHS columns and checks every inverse diagonal. Every timed CUDA trial
rejects CPU recovery; the table records the measured library's MD5 identifier.
`laplace_scatter` measures the weighted Gram
matrix and design-transpose score. Its size is observation count times squared
latent dimension, and its variant is the worker width. Each cell binds its
design once and retains it across Newton iterations. CPU assembly uses cached
column-major vectors; CUDA uses retained buffers and a signed-curvature kernel.
Both routes read scores and weights through the shared likelihood callback.
`SpecObservationState` owns the fused likelihood workspace and its cached
predictors, scores and curvatures together with the cell's dense observation
design. Eligible full patterns assemble the whole observation contribution
through `DenseScatter`. The structured-pattern route uses the shared basis
products for its basis blocks. Each observation term is assembled once;
the existing prior pass follows either layout.
`sparse_uniform` compares independent cuDSS
streams with uniform batches sharing symbolic analysis by their exact CSC
pattern. Uniform-batch log-determinants use a host factor: the installed cuDSS
diagonal query does not provide each slot's diagonal. The host factor is reused
by selected inversion. Each operation's gate comes from the same CSV.

`SparseCholeskySolver::Purpose::SelectedInverse` makes a variance-only solver
select that operation before allocating a device factor. When inverse policy
selects CPU, it factors the converged precision directly through CHOLMOD and
runs the Takahashi recursion. A Newton solver retaining a device factor lazily
creates its host factor only when a host operation needs it.

## Numerical backends

* #1180 provides sparse analysis, factorization, log determinant, solves, and selected
  inverse behind one shared solver interface. It obtains cuDSS symbols and
  buffers from this context, retains symbolic state per sparsity pattern, and
  registers its measured operation and equivalence fixtures.
* #1181 batches outer cells through that solver interface. `BatchScope` groups
  compatible requests while each worker retains its cell's Newton and line-search
  state. A completed cell submits no further requests; the worker takes its next
  cell. Mutable cuSOLVER handles belong to individual streams; shared launches
  use the context lease.
* #1182 provides `DenseFactor` and `DenseBasis` through the existing context
  and status reader. Numerical constants remain in their shared headers.
* #1183 supplies runtime-compiled family sources through `kernel_source` and
  `Context::launch`. `LikelihoodSpec::device_fn` carries the consumer descriptor;
  device gradients pass `ensure_gradient_verified()`.

Each backend adds paired benchmark samples and CPU/CUDA tests before enabling
automatic dispatch. The no-toolkit build and CPU route remain ordinary package
builds.
