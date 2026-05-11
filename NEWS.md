# tulpa NEWS

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
