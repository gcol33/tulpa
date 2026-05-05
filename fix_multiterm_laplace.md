# Multi-term + slope random effects on the spec-Laplace path

Date: 2026-05-05.

## 1. The limitation today

`inst/include/tulpa/laplace_spec_api.h:18`:
> At most one iid RE term (layout->has_re + .re_start / .re_end). The RE shares into process k iff data->sharing.re[k] is true.

`inst/include/tulpa/laplace_spec_api.h:25-27`:
> Random-slope / spatial / temporal variants are follow-on work; this entry will reject them with a clear error so callers get a deterministic signal instead of silent miscomputation.

`src/laplace_spec.cpp:11-18` repeats the "at most one iid RE term" contract and explicitly says random-slope is follow-on work. The current implementation reads `layout.re_start / layout.re_end` (single block, scalar precision `tau_re = 1/sigma_re²`) and `re_group_1based[i]` (single per-obs group index); `compute_eta_spec` adds `+u_g` into every shared process and `scatter_spec` does the same.

Downstream `tulpaglmm/src/glmm_laplace.cpp:202-206` rejects multi-term and random-slope re_specs:
```cpp
if (n_terms != 1 || n_coefs_vec[0] != 1) {
    stop("Laplace supports a single intercept-only RE term ...");
}
```

This is what we lift.

## 2. Math: joint posterior for K terms with arbitrary q_t

For K = `n_re_terms` RE terms, term t has:
- coefficient dimension q_t = `re_n_coefs[t]` (q_t = 1 for `(1|g)`, q_t > 1 for slopes)
- group count G_t = `re_n_groups_multi[t]`
- per-obs slope row z_{t,i} ∈ R^{q_t}: for intercept-only z_{t,i} = [1]; with slopes z_{t,i} = [1, x_{t,i,1}, …, x_{t,i,q_t-1}]
- group index g_t(i) ∈ {1, …, G_t} per obs (= `re_group_multi_flat[i*K + t]`)
- correlated flag `corr_t = re_correlated_multi[t]`

Latent vector b_{t,g} ∈ R^{q_t} (centered: spec-Laplace integrates over b directly, mirroring the existing single-term path; HMC uses non-centered z but that's a separate code path).

Per-obs RE contribution at obs i, into process k whose `sharing.re[k] == true`:

    eta_k_i += sum_{t = 0..K-1}  z_{t,i}^T b_{t, g_t(i)}

Prior: b_{t,g} ~ N(0, Σ_t) iid across g, Σ_t fixed by hyperparameters:
- uncorrelated (`(x||g)`):    Σ_t = diag(σ_{t,0}², …, σ_{t, q_t - 1}²)
- correlated (`(x|g)`):       Σ_t = diag(σ_t) · L_t · L_t^T · diag(σ_t),
                              L_t the unit-diagonal Cholesky factor parameterized
                              via tanh of `chol_re_start_multi[t] : chol_re_end_multi[t]`
                              (same parameterization as `tulpa_priors_re.h`).

Per-group precision Q_t = Σ_t^{-1}, q_t × q_t. The full RE prior precision is block-diagonal: blkdiag over (t, g) of Q_t (Q_t reused G_t times).

Log-prior contribution to log-marginal (centered): `−0.5 sum_{t,g} b_{t,g}^T Q_t b_{t,g} + 0.5 sum_t G_t · log|Q_t| − 0.5 (sum_t G_t q_t) log(2π)`.

Gradient wrt latent:

    grad_b log_prior at (t,g)        = −Q_t · b_{t,g}
    +grad_b log_lik at (t,g,c)       = sum_{i: g_t(i)==g}  z_{t,i,c} · sum_{k: shared} grad_eta_{i,k}

Negative-Hessian wrt latent — block (b_{t,g,c}, b_{t',g',c'}):

    likelihood:  sum_{i: g_t(i)==g, g_{t'}(i)==g'}  z_{t,i,c} · z_{t',i,c'} · sum_{k,l shared} w_{i,kl}
    prior     :  +Q_t[c,c']  iff (t == t' and g == g')

Cross-term coupling (t != t') is non-zero only at observations where both terms hit groups (g, g') simultaneously; for crossed `(1|g1) + (1|g2)` this is generally a non-zero block (tested below).

`Q_t` for the uncorrelated case is `diag(1/σ_{t,0}², …, 1/σ_{t,q_t-1}²)` and `log|Q_t| = -2 sum_c log σ_{t,c}`.

For the correlated case, Σ_t = D L L^T D with D = diag(σ_t). Then Q_t = D^{-1} L^{-T} L^{-1} D^{-1} and `log|Q_t| = −2 sum_c log σ_{t,c} − 2 log|det L|`. Since `det L = prod_c L_{cc}` (lower-triangular), `log|det L| = sum_c log L_{cc}`.

We compute Q_t once per Newton iteration (it depends only on hyperparameters, which are pinned during latent optimization, but we keep the recomputation in scatter to mirror the existing pattern).

## 3. ParamLayout

`ParamLayout` already has the multi-term + slope fields populated by `hmc_param_layout.cpp:43-138`. No new fields needed:

- `log_sigma_re_multi[t]` – first sigma slot per term (used for q_t == 1)
- `log_sigma_re_slopes[t][c]` – per-coef sigma slot when `has_re_slopes`
- `re_start_multi[t] / re_end_multi[t]` – latent block for term t (length G_t · q_t, observation-major within group: `re_start_multi[t] + g*q_t + c`)
- `re_n_coefs_multi[t]`, `re_correlated_multi[t]`
- `chol_re_start_multi[t] / chol_re_end_multi[t]` – tanh-Cholesky params for correlated terms

Legacy single-term fields (`log_sigma_re_idx`, `re_start`, `re_end`) are populated to point at term 0 when n_terms == 1, AND when `n_re_terms > 1` they alias term 0 (matches existing convention). The shim's `result_out->mode` writeout currently walks `[re_start, re_end)` only; this needs extending to walk all multi-term blocks in order.

## 4. ModelData

Already populated by `tulpaglmm/src/populate_helpers.h`:

- `n_re_terms`
- `re_group_multi_flat[obs * n_terms + t]` – 1-based group index per (obs, term)
- `re_n_groups_multi[t]`, `re_n_coefs[t]`, `re_n_slopes[t]`
- `re_correlated[t]`, `re_n_chol[t]`
- `re_slope_matrices[t]` – row-major `[N x n_slopes_t]`, contains slope columns only (not the intercept column). For obs i, slope c (c = 1..q_t-1), value is `re_slope_matrices[t][i * n_slopes_t + (c-1)]`.

The legacy single-term mirror (`data.re_group`, `data.n_re_groups`) is set when n_terms == 1 and intercept-only. We continue using that mirror only for back-compat (but the new spec-Laplace path always reads from the multi-term fields).

Sharing: `data.sharing.re[k]` is per-process, applied uniformly to all RE terms (matches HMC's behaviour). No per-term sharing matrix — a future generalization but out of scope.

## 5. SpecLatentLayout extension (in laplace_spec.cpp)

Replace the single-term fields with a `std::vector<ReTermSlot>`:

```cpp
struct ReTermSlot {
    int n_coefs;          // q_t
    int n_groups;         // G_t
    int param_start;      // = layout.re_start_multi[t], absolute
    int latent_offset;    // offset into (β-blocks-then-RE) latent vec for term t's block
    bool correlated;      // == layout.re_correlated_multi[t]
    int chol_start;       // -1 if uncorrelated; else absolute idx
    std::vector<int> sigma_slots;  // length q_t; absolute idx
};

struct SpecLatentLayout {
    int np;
    std::vector<int> beta_start, beta_count, latent_offset; // [np], [np], [np+1] as before
    std::vector<ReTermSlot> re_terms;     // empty when no RE
    int re_offset_total = 0;              // = latent_offset[np] when no RE; else first re term's latent_offset
    int n_x = 0;
};
```

`build_latent_layout`:
- If `layout.has_re` is false → empty re_terms, n_x = sum_beta.
- Else (`data.n_re_terms` ≥ 1, or legacy n_re_terms == 0 with single intercept-only): walk a unified n_terms count from `data.n_re_terms == 0 ? 1 : data.n_re_terms`. For each term t pull:
  - `n_coefs = (layout.re_n_coefs_multi.size() > t) ? layout.re_n_coefs_multi[t] : 1`
  - `n_groups = (data.re_n_groups_multi.size() > t) ? data.re_n_groups_multi[t] : data.n_re_groups`
  - `param_start = (layout.re_start_multi.size() > t) ? layout.re_start_multi[t] : layout.re_start`
  - `correlated = (layout.re_correlated_multi.size() > t) ? layout.re_correlated_multi[t] : false`
  - `chol_start = (layout.chol_re_start_multi.size() > t) ? layout.chol_re_start_multi[t] : -1`
  - `sigma_slots`: from `layout.log_sigma_re_slopes[t]` when populated and length q_t; else `{layout.log_sigma_re_multi[t]}` (q_t = 1) or `{layout.log_sigma_re_idx}` for legacy.

`latent_offset` runs over beta blocks first, then RE blocks in term order, with each term contributing G_t · q_t slots.

## 6. compute_eta_spec generalization

```
re_eff = 0
for t in 0..K-1:
    g = re_group_at(i, t) - 1; if g < 0 continue
    contrib = b[t,g,0]   // intercept
    for c = 1..q_t-1:
        contrib += b[t,g,c] * z_{t,i,c}    // slope c-1 from re_slope_matrices[t][i*n_slopes_t + (c-1)]
    re_eff += contrib    // shared by all RE terms uniformly
for k in 0..np-1:
    eta_k_i = X_k row · beta_k + offset_k_i + (shared(k) ? re_eff : 0)
```

(K == 1, q_0 == 1 reduces to the old single-RE path: re_eff = params[re_start + g_re].)

## 7. scatter_spec generalization

Per-obs:

1. Call `eta_weights_fn` to get `grad_eta[k]`, `neg_hess_eta[k*np + l]`.
2. Compute `s_grad = sum_{k: shared} grad_eta[k]` and `s_hess = sum_{k,l: shared,shared} neg_hess_eta[k*np+l]`.
3. Beta gradient + beta×beta Hessian: unchanged from current code (uses full neg_hess_eta).
4. RE × β cross block: for each shared process l with p_l > 0, weight `w_l = sum_{k: shared} neg_hess_eta[k*np + l]`, and for each term t with active group g_t(i), the row index is `latent_offset_t + g_t(i)*q_t + c`. Contribution to `H[re_row, latent_offset[l] + m] += w_l · z_{t,i,c} · X_l(i,m)`.
5. RE × RE block (term × term):
   - For each term-pair (t, t'), if both have active groups g_t(i), g_{t'}(i): contribution to H is z_{t,i} · z_{t',i}^T · s_hess at the (term=t, group=g_t(i), term'=t', group'=g_{t'}(i)) block. For t == t' and g_t == g_{t'}, on-diagonal block. Else cross block (lower-triangle convention preserved).
6. RE gradient: for each term t with active g, `grad_RE[term=t, g, c] += z_{t,i,c} · s_grad`.

After the obs loop:

- Symmetrize lower → upper.
- β prior: unchanged.
- RE prior block: per term t, build Q_t (q_t × q_t) from σ_t and (if correlated) L_t (computed once from the chol slots via the same tanh parameterization as the HMC path). For each group g: contribute `+Q_t · b_{t,g}` to negative gradient (i.e. `grad += -Q_t b`) and `+Q_t` to the q_t × q_t Hessian sub-block at `latent_offset_t + g*q_t : ... + (g+1)*q_t`.

## 8. log_prior_latent generalization

Similar walk: for each term t,
- compute L_t (if correlated) from chol slots
- `b_{t,g}^T Q_t b_{t,g}` summed over g
- `+0.5 G_t log|Q_t|`, where `log|Q_t| = -2 sum_c log σ_{t,c} - 2 sum_c log L_{t,cc}` (the L term zero when uncorrelated)
- `−0.5 G_t q_t log(2π)`.

These should sum to the existing single-term formula when K = 1, q_0 = 1.

## 9. apply_latent_step

Walk all term blocks `[re_start_multi[t], re_end_multi[t])` instead of just the legacy block.

## 10. Shim mode-output (tulpa_shims_laplace_spec.h)

The current shim writes `result_out->mode = [β_0, β_1, …, β_{np-1}, b_{re_start..re_end}]`. Generalise to walk all term blocks in term order (matches `SpecLatentLayout`). Single-term case is bit-identical because `re_start_multi[0] == re_start` and `re_end_multi[0] == re_end`.

## 11. Test harness fixtures

`cpp_laplace_spec_test_gaussian` and `cpp_laplace_spec_test_gaussian2p` build their own `ParamLayout` directly. They keep working unchanged because the legacy-only layout fields (`re_start`, `re_end`, `log_sigma_re_idx`, `has_re`) are still honored by the unified path. They keep the n_terms == 1 q_t == 1 contract.

For multi-term tests, add a new fixture `cpp_laplace_spec_test_multi_re` that:
- accepts a list of (group_idx, n_groups, n_coefs, sigma_vec, correlated, slope_mat) per term
- builds ParamLayout using `compute_param_layout` (which we know now populates the multi-term fields), or builds it manually for full control
- runs the new path and returns mode + log_marginal

## 12. Migration list

| Caller                                                  | Action                                                                                       |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `src/laplace_spec.cpp::cpp_laplace_spec_test_gaussian`  | No change — single-term intercept-only path stays bit-identical.                              |
| `src/laplace_spec.cpp::cpp_laplace_spec_test_gaussian2p`| No change — same reason.                                                                      |
| `src/tulpa_shims_laplace_spec.h::tulpa_laplace_spec_dense_impl` | Generalise mode output to walk all term blocks.                                       |
| `tulpaglmm/src/glmm_laplace.cpp:192-210`                | NOT EDITED in this task. The `stop()` is now lifted-able by the user; needs a follow-up.      |
| Family-enum laplace path (`tulpa_laplace_mode_dense`)   | Untouched — separate code path with its own multi-term wiring (in `hmc_observation_likelihood.h` and friends). |

## 13. ABI bump

`TULPA_ABI_VERSION` will be bumped from 3 → 4. The struct `ParamLayout` is unchanged in layout (already has the multi-term fields; we just start using them). `ModelData` is unchanged. The bump signals "spec-Laplace contract changed: now reads multi-term fields", so downstream packages compiled against ABI 3 should be rebuilt.

Note: tulpaGlmm uses `check_abi_version()` on first call to `get_laplace_spec_dense_fn()`. Bumping ABI without rebuilding tulpaGlmm will fail at runtime with a clear ABI-mismatch error — which is the desired behavior. The user must rebuild tulpaGlmm against the new tulpa.

## 14. Test plan (added to tests/testthat/test-laplace-spec.R)

a. `(1|g1) + (1|g2)` two crossed intercept-only terms, Gaussian likelihood, hand-computed linear-system solution.
b. `(x|g)` correlated random slope (q=2, intercept + slope x with correlation), Gaussian, hand-computed linear-system solution. With one term the correlated case has L = [[1,0],[ρ,√(1-ρ²)]]; we use the tanh-raw param to pin a known correlation.
c. `(x||g)` uncorrelated random slope (q=2, intercept + slope, diagonal Σ), Gaussian, hand-computed linear-system solution.
d. Single-term intercept-only (regression) — must remain bit-identical to existing fixture.

For each test, the linear-Gaussian problem reduces to solving (X^T X + tau prior precision) x = rhs, exactly as the current `gaussian2p` test does. The reference is built directly from blocks (X1, X2, Z_1 (intercept), Z_2 (slopes)).

## 15. Estimated diff size

| File                                                       | LOC delta |
|------------------------------------------------------------|-----------|
| `src/laplace_spec.cpp` (struct, build_latent_layout, compute_eta, scatter, log_prior, apply_step, new fixture) | +250..+350 |
| `src/tulpa_shims_laplace_spec.h` (mode-output walk)        | +20       |
| `inst/include/tulpa/laplace_spec_api.h` (contract comment) | ±15       |
| `inst/include/tulpa/model_data.h` (ABI bump)               | ±2        |
| `tests/testthat/test-laplace-spec.R` (new tests)           | +200..+250|
| `NEWS.md` (new file)                                       | +20       |
| **Total**                                                  | ~600 LOC  |

Well under the 1000 LOC stop-condition.

## 16. Review

- Math closes: precision Q_t depends only on hyperparams, gradient and Hessian written down explicitly above. Single-term q_0==1 reduces algebraically to existing tau_re scalar.
- Layout encodes every case: ParamLayout already has all the slots we need; we just consume them.
- Footguns:
  - **Centered vs non-centered**: spec-Laplace stays centered (b directly), HMC stays non-centered (z). The same `re_start_multi` slots store *different* quantities depending on path. populate_re writes 0-initialized values which is fine for both.
  - **Sigma slot indexing**: layout has `log_sigma_re_slopes[t][c]` for the slopes case and `log_sigma_re_multi[t]` for the multi-term-no-slopes case. We need to handle both.
  - **Shim mode output ordering**: term-major within RE block (term 0's full block, then term 1's, …) so a downstream caller can walk it predictably. Single-term case unchanged.
  - **Cross-term Hessian**: an obs that hits two terms (e.g. crossed `(1|g1)+(1|g2)`) creates a coupling between b_{0,g1(i)} and b_{1,g2(i)}. We must scatter that, not assume block diagonal across terms.
  - **q_t slope-row construction**: row 0 of z_{t,i} is always 1 (intercept); rows 1..q_t-1 are pulled from `re_slope_matrices[t]`. populate_re only writes slope_matrices when has_intercept is implicit; we must trust `n_coefs[t]` and read `n_slopes_t = q_t - 1` columns from the matrix.
  - **Correlated case**: log-det handling is per-term: log|Σ_t| = 2 sum_c log σ_{t,c} + 2 sum_c log L_{t,cc}. Must mirror the tanh parameterization used in HMC priors_re. We compute L from `chol_re_start_multi[t]` once per scatter call.
