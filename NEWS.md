# tulpa NEWS

## 2026-05-12 — ABI v11: caller-supplied inv-mass diagonal for NUTS

* `TULPA_ABI_VERSION` bumped **10 → 11**. Downstream packages must
  rebuild against the v11 headers.
* Registered C-callable `tulpa_run_nuts_generic` (`nuts_api.h`
  `NUTSFn`) gains a new positional parameter `const double*
  inv_metric_diag` immediately before `NUTSResult* result_out`. Pass
  `nullptr` to keep the v10 behaviour (default structural warm-start
  of the mass matrix). Pass a length-`n_params` vector to seed the
  diagonal inverse-mass — useful for warm-starting NUTS from an
  analytical-approximation method (Laplace, VI, etc.).
* `run_hmc_chain_cpp` / `run_hmc_chain` (`hmc_sampler_funcs.h`) take a
  matching trailing `inv_metric_init` `std::vector<double>` (default
  empty). Within `run_hmc_chain_cpp` (`hmc_nuts_chain_setup.h`), a
  non-empty caller diagonal overrides the structural diagonal set by
  `warm_start_mass_matrix`. Values are clamped to `[1e-3, 1e3]`
  before being installed via `mass.set_diagonal`, then
  `find_reasonable_epsilon` re-runs against the seeded metric.
* Mass-matrix *adaptation* is unchanged: the standard dual-averaging
  + expanding-windows path still refines the diagonal during warmup.
  The caller's vector is the *initial* metric, not a frozen one.
* Internal callers of `run_hmc_chain_cpp` (`hmc_nuts_parallel.cpp` ×3,
  `tulpa_generic_sampler.cpp` ×1) continue to pass no `inv_metric_init`
  via the default empty vector; the local forward declaration in
  `tulpa_generic_sampler.cpp` was updated to match the canonical
  declaration's parameter list.
* Downstream rebuild notes: `tulpaGlmm` exercises this end-to-end via
  Day-32's `hmc_warm_start = "laplace"` argument. `tulpaOcc` and
  `tulpaRatio` need to be reinstalled against v11 — both already
  updated to pass `nullptr` for the new parameter (no logic change).

## 2026-05-11 — NNGP Laplace: full off-diagonal precision scatter

* `laplace_mode_gp` (and the spatial-only / ST-combo NNGP entries in
  `nested_laplace.cpp`) now assemble the **full NNGP precision matrix**
  `Λ = (I - A)' D⁻¹ (I - A)` in every Newton iteration, replacing the
  diagonal-on-w approximation that only kept `1/v_i` on the focal
  diagonal of each row.
* What was missed before: the gradient contribution to neighbours
  (`+a_{i,k}·q_i/v_i`), the off-diagonal Hessian entries
  (focal, neighbour_k) and (neighbour_k, neighbour_kp), and the
  pairwise precision between members of every conditioning set. The
  Newton mode for `w` was therefore shrunk toward zero and pointwise
  field recovery on smooth latent fields collapsed (cor ≈ 0).
* New helpers in `gpu_nngp_laplace.h`:
    - `batch_nngp_scatter(..., alpha_out = nullptr)` — backward-compatible
      extra optional output capturing the per-row conditional regression
      weights (already computed internally; just exposed).
    - `apply_nngp_full_prior_dense` — scatters the full precision
      contribution into a dense `(grad, H)` pair via the alpha + cv
      bundle.
    - `apply_nngp_full_prior_sparse` — same, into a `SparseHessianBuilder`.
    - `make_nngp_prior_sparsity_pattern` — emits the `(row, col)` pairs
      required to back the sparse path.
* Wired into the four scatter call sites: `laplace_mode_gp` dense Newton,
  `laplace_mode_gp` sparse Newton (with pattern expansion), the
  spatial-only `cpp_nested_laplace_nngp` scatter lambda, and
  `make_nngp_spatial_ops::add_prior_at_k` (the ST-combo NNGP block).
  Log-prior calls (`log_prior` lambdas) are unchanged — they only need
  `cm` and `cv`, and the existing `batch_nngp_scatter` signature still
  supports that without `alpha_out`.
* Effect (downstream measurement from tulpaGlmm Day-31 smoke):
  `cor(w_mean, f_true)` jumps from near zero to **≈ 0.81 Pearson** on
  a 120-location Poisson + smooth-GP simulation. β recovery unchanged.
* Additive: no `TULPA_ABI_VERSION` bump (still v10). Public shim
  signatures are unchanged; only the inner Laplace scatter is upgraded.
  Downstream packages must rebuild against this commit to pick up the
  new behaviour (no source changes required).

## 2026-05-11 — nested-Laplace ST family: 5 more indexed × indexed combos

* Adds five additional joint spatial × temporal nested-Laplace shims,
  built on the same `run_two_indexed_nested_laplace` driver and joint
  inner Newton introduced earlier today:
    - `tulpa_nested_laplace_st_icar_rw1`
    - `tulpa_nested_laplace_st_icar_rw2`
    - `tulpa_nested_laplace_st_car_proper_rw1`
    - `tulpa_nested_laplace_st_car_proper_rw2`
    - `tulpa_nested_laplace_st_car_proper_ar1`
  Each routes through a per-combo Rcpp entry plus a C-callable
  `_impl` wrapper; matching typedefs + getters live in
  `nested_laplace_api.h`.
* New internal factory pattern `IndexedPriorOps` in `nested_laplace.cpp`
  with per-kind builders `make_icar_ops`, `make_car_proper_ops`,
  `make_rw1_ops`, `make_rw2_ops`, `make_ar1_ops`. The shared
  `run_two_indexed_nested_laplace` driver now consumes
  `std::function`-typed callbacks, so adding the next indexed × indexed
  combination is a few lines of Rcpp glue rather than a re-derivation.
* Refactored `cpp_nested_laplace_st_icar_ar1` to use the new factories
  (identical behavior; just dropped the inline lambdas).
* Additive: no `TULPA_ABI_VERSION` bump (still v9). The new shims are
  resolved via `R_GetCCallable` at first use; downstream packages
  rebuilt against ABI v9 pick them up automatically.

## 2026-05-11 — nested-Laplace joint spatial × temporal (ICAR × AR1)

* New shim `tulpa_nested_laplace_st_icar_ar1` for joint nested-Laplace
  inference with an ICAR spatial field AND an AR1 temporal field in the
  same fit. The joint inner Newton solves over the full latent vector
  `[beta] [re] [w_spatial (n_s)] [w_temporal (n_t)]` at each grid point;
  the cross-block `H[w_s, w_t]` is non-zero, so the two fields cannot be
  Laplace-marginalized separately. The hyperparameter grid is supplied
  caller-side as paired vectors of length `n_grid` (Cartesian product of
  `τ_spatial × τ_temporal × ρ_temporal` built on the R side).
* New C-callable typedef `NestedLaplaceStIcarAr1Fn` + getter
  `get_nested_laplace_st_icar_ar1_fn()` in `nested_laplace_api.h`.
* New internal building blocks in `nested_laplace.cpp`: the templated
  `run_two_indexed_nested_laplace` driver and helpers
  `nl_compute_eta_two_indexed` / `nl_scatter_obs_two_indexed`. These are
  the shared substrate for the remaining 11 (spatial_kind × temporal_kind)
  combinations.
* `TULPA_ABI_VERSION` bumped 8 → 9. Downstream packages must be rebuilt
  against this header set.

## 2026-05-11 — nested-Laplace HSGP returns modes + store_Q

* `tulpa_nested_laplace_hsgp` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The basis-coefficient latent `[beta] [re] [beta_M (n_basis)]` is
  returned per `(σ², ℓ)` grid point, and with `store_Q = 1` the joint Q
  at the mode is retained in the standard `NestedLaplaceShimResult::Q_*_flat`
  slots.
* `cpp_nested_laplace_hsgp` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_hsgp_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceHsgpFn` in `nested_laplace_api.h` is updated to match.
* This unblocks HSGP mixture-of-MVN sampling in tulpaGlmm — the
  observation-level spatial effect `f_i = Σ_j Φ_ij · √S(λ_j; σ²_k, ℓ_k) · β_M_j`
  can be reconstructed caller-side from modes + posterior draws over the
  basis coefficients plus the per-draw grid index.
* `TULPA_ABI_VERSION` bumped 7 → 8. Downstream packages (tulpaGlmm,
  tulpaOcc) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace BYM2 returns modes + store_Q

* `tulpa_nested_laplace_bym2` now sets `store_modes = 1` (was 0) and
  gained a `store_Q` flag matching the rest of the nested-Laplace family.
  The reparameterised latent `[beta] [re] [phi (n_spatial)] [theta
  (n_spatial)]` is returned per grid point, and with `store_Q = 1` the
  joint Q at the mode is retained in the standard
  `NestedLaplaceShimResult::Q_*_flat` slots.
* `cpp_nested_laplace_bym2` gained a trailing `bool store_Q = false`
  argument and now passes `store_modes = true` to the grid driver. The
  C-callable `tulpa_nested_laplace_bym2_impl` signature picks up the
  matching `int store_Q` parameter; the public typedef
  `NestedLaplaceBym2Fn` in `nested_laplace_api.h` is updated to match.
* This unblocks BYM2 mixture-of-MVN sampling in tulpaGlmm — the total
  spatial effect `w_s = σ·(√ρ · scale · φ_s + √(1−ρ) · θ_s)` can be
  reconstructed caller-side from modes + posterior draws over the
  (σ, ρ) grid.
* `TULPA_ABI_VERSION` bumped 6 → 7. Downstream packages (tulpaGlmm,
  tulpaOcc) must be rebuilt against the updated headers.

## 2026-05-11 — nested-Laplace store_Q on RW1/RW2/AR1/CAR_proper

* `tulpa_nested_laplace_rw1`, `tulpa_nested_laplace_rw2`,
  `tulpa_nested_laplace_ar1`, and `tulpa_nested_laplace_car_proper` now
  accept a `store_Q` flag (matching the ICAR shim added in v5). When set
  the shim retains the joint negative-Hessian Q at each grid point's mode
  in `NestedLaplaceShimResult::Q_*_flat`, so downstream packages can draw
  mixture-of-MVN posteriors `sum_k w_k · N(mode_k, Q_k^{-1})` without
  re-doing the Newton assembly R-side.
* The underlying `cpp_nested_laplace_<rw1|rw2|ar1|car_proper>` entries
  gained a trailing `bool store_Q = false` argument. Default is `false`,
  so existing callers that don't ask for Q keep the previous behaviour
  and footprint.
* `TULPA_ABI_VERSION` bumped 5 → 6. Downstream packages (tulpaGlmm,
  tulpaOcc) must be rebuilt against the updated headers.

## 2026-05-06 — Takahashi partial inverse as a registered C-callable

* New free function `tulpa::takahashi_partial_inverse_dense(n, Lp, Li, Lx,
  Z_out)` in `sparse_cholesky.{h,cpp}` runs the Takahashi recursion on a
  caller-supplied lower-triangular `L` (CSC) and writes a dense column-major
  `n*n` `Z` with `Q^{-1}` on `pattern(L + L^T)` and zeros elsewhere. A
  matching `takahashi_partial_inverse_csc` returns just the `Zx` values on
  pattern(L). The existing `SparseCholeskySolver::selected_inversion_diagonal`
  now routes through the new helper so there is one source of truth for the
  recursion (no copy-paste).
* Registered C-callable `tulpa_takahashi_partial_inverse_dense` exposes the
  pure-function variant to downstream packages. Resolved via
  `tulpa::get_takahashi_partial_inverse_dense_fn()` in
  `inst/include/tulpa/sparse_solver_api.h`; the getter `Rf_error`s if the
  symbol is missing (i.e. caller built against newer headers than the loaded
  tulpa).
* No struct layout changes; `TULPA_ABI_VERSION` stays at 4. Downstream
  packages that want the new shim need only rebuild against the updated
  `sparse_solver_api.h`.

## 2026-05-05 — multi-term + slope REs on the spec-Laplace path

* `tulpa_laplace_spec_dense` (and its public C ABI shim) now accepts the
  full multi-term, multi-coefficient RE structure populated by `populate_re`
  in downstream model packages: K = `data.n_re_terms` random-effect terms,
  each with `q_t = re_n_coefs[t]` coefficients per group (intercept-only
  when `q_t == 1`, intercept + slopes when `q_t > 1`), uncorrelated
  (`(x||g)`) or correlated (`(x|g)`) prior covariance. Per-process
  sharing is uniform across terms via `data.sharing.re`.
* The legacy single-term path (`layout.re_start` / `layout.re_end` /
  `log_sigma_re_idx` with `data.n_re_terms == 0`) is preserved and stays
  bit-identical numerically.
* The `result_out->mode` writeout from `tulpa_laplace_spec_dense_impl`
  now concatenates every RE term's block in term order
  (`re_start_multi[t]..re_end_multi[t]`), matching the new
  `SpecLatentLayout` ordering.
* New internal test fixture `cpp_laplace_spec_test_multi_re` exercises
  the multi-term path end-to-end against hand-derived linear-Gaussian
  reference solutions; new tests in `tests/testthat/test-laplace-spec.R`
  cover (a) two crossed intercept-only terms, (b) one correlated random
  slope, (c) one uncorrelated random slope.
* `TULPA_ABI_VERSION` bumped from 3 → 4. Downstream packages that ship
  with the spec-Laplace dispatcher (e.g. tulpaGlmm) must be rebuilt
  against the new headers.
