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

`operations()` owns the operation-to-capability registry. An unknown operation
is an error. The `variant` is an exact matrix order for NNGP and zero for axpy.
An unmeasured operation or variant has an infinite crossover in automatic mode.
Forced CUDA ignores the size gate and requires that operation's capability.
Host inference remains on CPU. A failed device operation records a status;
it never calls R or throws a library error from a worker. The R finalization
and session exit read this status on the main thread.

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
and the same checked status convention. cuDSS is a loader capability, with
no sparse numerical operation registered yet.

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
backend must check the resolved API version when it adds numerical calls.

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
timings. `.device_thresholds()` groups by operation and variant, takes medians
at each measured size, and chooses the first size at which CUDA and every
larger measured size win. With no sustained measured win the threshold is
infinite. There is no size threshold copied into C++.

`tools/bench-device.R` checks that every registered operation has a runner.
The NNGP runner measures complete conditional-moment batches, including
covariance construction, factorization, checked transfers, factor verification,
and the CPU triangular solves. It asserts fixture dimensions and CPU/CUDA
agreement before timing, warms both routes, alternates their order, and writes
paired samples. The axpy runner includes allocation, copies, kernel execution,
synchronization, and cleanup. Timing uses the native steady clock.

## Subsequent backends

* #1180 adds sparse analysis, factorization, log determinant, solves, and selected
  inverse behind one shared solver interface. It obtains cuDSS symbols and
  buffers from this context, retains symbolic state per sparsity pattern, and
  registers its measured operation and equivalence fixtures.
* #1181 batches outer cells through that solver interface. `Launch::stream`
  supports a future per-stream workspace; concurrent use of mutable cuBLAS or
  cuSOLVER handles needs one handle per stream or the existing lease.
* #1182 adds dense solver and BLAS operations to the registry. It uses the
  existing context and status reader, and keeps numerical constants in their
  existing shared headers.
* #1183 adds runtime-compiled family sources through `kernel_source` and
  `Context::launch`. Its new likelihood ABI slot belongs in the consumer
  interface, and device gradients still pass `ensure_gradient_verified()`.

Each backend adds paired benchmark samples and CPU/CUDA tests before enabling
automatic dispatch. The no-toolkit build and CPU route remain ordinary package
builds.
