// cell_coupling.h
// Public abstractions for *cell-coupled* per-cell likelihoods on the joint
// nested-Laplace path (gcol33/tulpa#32 Change 2).
//
// ============================================================================
// Why this header exists
// ============================================================================
//
// The existing joint engine (`tulpa_nested_laplace_joint()`) assumes that the
// per-arm data log-likelihood factorises as
//
//     log L_joint(eta) = sum_arm sum_obs log f_arm(y_obs | eta_arm[obs]).
//
// That covers the 2-arm cover-hurdle (zero arm and positive arm each see
// their own data). It does NOT cover families where the same cell's
// contribution couples observations across arms in a non-separable mixture,
// e.g. tulpaObs's occu_cover, whose per-cell density is
//
//     p_obs_c = psi_c * [prod_v Bern(y_det_cv | p_cv) * f_pos(y_cov_cv | mu_cv)]
//             + (1 - psi_c) * 1{all y_det_cv = 0 and all y_cov_cv = 0}.
//
// The latent state psi_c modulates the per-cell likelihood of the other two
// arms in one mixture; no per-obs factorisation exists.
//
// `CellCouplingSpec` is the abstraction the inner Newton's per-cell branch
// dispatches to in place of the arm-separable sum. Downstream packages
// (tulpaObs in particular) supply a compiled subclass of `CellCouplingSpec`
// in their own `src/`, register it from `R_init_<pkg>` against tulpa's
// process-global registry (see `src/cell_coupling_registry.h`), and reference
// it from R via a string name on the joint-fit call.
//
// ============================================================================
// Why virtual dispatch (not a template)
// ============================================================================
//
// `tulpa::REGroupOracle` in `<tulpa/aghq_oracle.h>` already uses virtual
// dispatch for the per-group oracle, and the inner-Newton hot path remains
// fast because the virtual call happens once per *cell*, not per *obs*: each
// `evaluate_cell()` call typically does O(J_c) work over the cell's J_c
// visits, so the dispatch cost is amortised. The same trade-off applies
// here. A template-based design would force every concrete spec to live in a
// header instantiated from the inner Newton's translation unit, which
// prevents downstream packages (which compile their spec in their own DLL)
// from supplying one without recompiling tulpa. Virtual dispatch matches the
// existing oracle pattern and keeps the per-cell branch under
// `LinkingTo: tulpa` + `R_GetCCallable` boundary, no cross-DLL templates.
//
// ============================================================================
// Lifetime / ownership across the R boundary
// ============================================================================
//
// Mirrors `REGroupOracle`: the consumer package heap-allocates one
// `CellCouplingSpec` subclass and hands it to tulpa as a
// `std::shared_ptr<CellCouplingSpec>` via `register_cell_coupling()`. The
// registry retains the shared_ptr for the lifetime of the process; lookups
// return the same shared_ptr by name. Any backing storage the spec reads
// (per-cell visit counts, per-arm response data) must outlive the fit; the
// recommended pattern is to capture it inside the spec subclass at
// construction time so the registry's shared_ptr keeps it alive.

#ifndef TULPA_CELL_COUPLING_H
#define TULPA_CELL_COUPLING_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

struct CellCouplingSpec;

// Function-pointer signature of the `tulpa_register_cell_coupling` registered
// C callable. Consumer packages (e.g. tulpaObs) declare a matching
// `R_GetCCallable("tulpa", "tulpa_register_cell_coupling")` lookup in their
// `R_init_<pkg>` and call through this signature to insert their compiled
// `CellCouplingSpec` subclass into tulpa's process-global registry.
//
// Lives in the public header (rather than `src/cell_coupling_registry.h`) so
// downstream packages linking via `LinkingTo: tulpa` can see it without
// duplicating the typedef. The wire format is locked here.
using RegisterCellCouplingFn = void (*)(const char* name,
                                        std::shared_ptr<CellCouplingSpec> spec);

// Curvature the kernel asks a CellCouplingSpec to write into CellDerivs.
//   Observed : -d^2 log p_cell / d eta d eta (the true Hessian). Default;
//              used for the final mode-pass that feeds log_det_Q and the SEs.
//   Expected : the complete-data expected information (Fisher scoring). PSD by
//              construction; used for the inner Newton step when the caller
//              selects control$hessian = "fisher". A spec that does not
//              implement it MUST ignore the request and write the observed
//              Hessian (the default behaviour).
enum class CurvatureMode { Observed, Expected };

// ----------------------------------------------------------------------------
// CellEtas -- read-only per-cell view of each arm's eta at the rows the inner
// Newton has assigned to this cell.
//
// The inner Newton owns the per-arm eta vectors (one entry per row of the
// arm's data). For a given cell index `c` and arm index `k`, the rows
// belonging to this cell are listed in `arm_rows[k]` (length `n_rows_in_arm(k)`),
// indexing into the arm's full eta vector `arm_eta_ptr[k]`. `eta(k, j)`
// resolves to `arm_eta_ptr[k][arm_rows[k][j]]`.
//
// This is a non-owning view: pointers point into the inner Newton's
// scratch buffers and are only valid for the duration of one
// `evaluate_cell()` call. Specs must not store pointers beyond the call.
// ----------------------------------------------------------------------------
struct CellEtas {
    // Per-arm eta vector pointers. Index k in [0, n_arms()) maps to the arm's
    // full eta buffer (length = the arm's total row count).
    const double* const* arm_eta_ptr = nullptr;

    // Per-arm row indices for this cell. arm_rows[k] is a pointer to an
    // int[arm_row_count[k]] array of 0-based row indices into arm_eta_ptr[k].
    const int* const* arm_rows = nullptr;

    // Per-arm count of rows in this cell. Length n_arms.
    const int* arm_row_count = nullptr;

    int n_arms_ = 0;

    // Batched multi-response (gcol33/tulpa#66). When n_batch_ > 1 each arm's
    // eta buffer holds B species' eta columns laid species-major:
    // arm_eta_ptr[k] + s * arm_eta_stride[k] is species s's column. The B=1
    // path leaves n_batch_ = 1 and arm_eta_stride = nullptr, so eta(k, j) and
    // the (k, j) two-arg accessor below are byte-identical to the pre-batch
    // path. Appended last so existing field offsets are unchanged.
    const int* arm_eta_stride = nullptr;  // per-arm column stride (= total rows)
    int n_batch_ = 1;

    int n_arms() const { return n_arms_; }
    int n_batch() const { return n_batch_; }
    int n_rows_in_arm(int k) const { return arm_row_count[k]; }

    // eta(k, j) -- the eta value at the j-th row of arm k that belongs to
    // this cell. j in [0, n_rows_in_arm(k)). B=1 / first-species view.
    double eta(int k, int j) const {
        return arm_eta_ptr[k][arm_rows[k][j]];
    }

    // eta(k, j, s) -- species s's eta at the j-th row of arm k. For s = 0 and
    // the default (null) stride this reduces exactly to eta(k, j).
    double eta(int k, int j, int s) const {
        const int off = arm_eta_stride ? s * arm_eta_stride[k] : 0;
        return arm_eta_ptr[k][off + arm_rows[k][j]];
    }
};

// ----------------------------------------------------------------------------
// CellResponse -- read-only per-cell view of each arm's response data at
// the rows belonging to this cell.
//
// Storage layout mirrors `JointArm` in `src/laplace_newton_joint.h`: each
// arm carries a per-row `y` (double) and `n_trials` (int) plus a family tag
// string. The view exposes pointers into the arm's full response buffers
// indexed by the same `arm_rows[k]` array used by `CellEtas`, so a spec
// reads `y(k, j)` and `n_trials(k, j)` symmetrically with `etas.eta(k, j)`.
//
// The `family` field carries the per-arm family tag as recorded on the R
// side (typically the marker strings the joint driver passes through, e.g.
// `"occu_cover_psi"` / `"occu_cover_p"` / `"occu_cover_pos"`); specs can use
// it for additional dispatch within their `evaluate_cell()` body.
// ----------------------------------------------------------------------------
struct CellResponse {
    // Per-arm response vector pointers. arm_y[k] is the arm's full y buffer
    // (length = the arm's total row count). May be nullptr for arms whose
    // family carries no observed data (e.g. occu_cover's psi arm).
    const double* const* arm_y = nullptr;

    // Per-arm trial-count vector pointers (binomial denominators). Same
    // indexing as arm_y; may be nullptr.
    const int* const* arm_n_trials = nullptr;

    // Per-arm family tag strings. Length n_arms.
    const char* const* arm_family = nullptr;

    // Per-arm dispersion parameter (gaussian SD, negbin size, lognormal SD,
    // beta precision -- family-specific meaning). Length n_arms.
    const double* arm_phi = nullptr;

    // Per-arm row indices for this cell (same pointer as in CellEtas).
    const int* const* arm_rows = nullptr;

    // Per-arm row count for this cell.
    const int* arm_row_count = nullptr;

    int n_arms_ = 0;

    // Batched multi-response (gcol33/tulpa#66). Mirrors CellEtas: arm_y[k] +
    // s * arm_y_stride[k] is species s's response column; arm_phi_batch (when
    // set) carries per-species dispersion laid [k * n_batch_ + s]. B=1 leaves
    // both null and n_batch_ = 1, so y(k, j) / phi(k) are byte-identical.
    const int*    arm_y_stride   = nullptr;  // per-arm y column stride
    const double* arm_phi_batch  = nullptr;  // [n_arms * n_batch_], or null
    int           n_batch_       = 1;

    int n_arms()  const { return n_arms_; }
    int n_batch() const { return n_batch_; }
    int n_rows_in_arm(int k) const { return arm_row_count[k]; }

    // y(k, j) -- the response value at the j-th row of arm k that belongs
    // to this cell. Undefined if arm_y[k] is nullptr (the spec must know
    // which of its arms carry data). B=1 / first-species view.
    double y(int k, int j) const {
        return arm_y[k][arm_rows[k][j]];
    }

    // y(k, j, s) -- species s's response at the j-th row of arm k. s = 0 with
    // null stride reduces exactly to y(k, j).
    double y(int k, int j, int s) const {
        const int off = arm_y_stride ? s * arm_y_stride[k] : 0;
        return arm_y[k][off + arm_rows[k][j]];
    }

    int n_trials(int k, int j) const {
        return arm_n_trials[k][arm_rows[k][j]];
    }

    const char* family(int k) const { return arm_family[k]; }
    double phi(int k)               const { return arm_phi[k];    }

    // phi(k, s) -- species s's dispersion for arm k. Falls back to the shared
    // arm_phi[k] when no per-species table is supplied.
    double phi(int k, int s) const {
        return arm_phi_batch ? arm_phi_batch[k * n_batch_ + s] : arm_phi[k];
    }
};

// ----------------------------------------------------------------------------
// CellDerivs -- writable per-cell accumulator for the gradient and
// negative-Hessian contributions the spec computes.
//
// For each arm k participating in the coupling, the spec fills:
//
//   grad(k, j)         = d log p_cell / d eta(k, j)
//   neg_hess(k, j, j)  = -d^2 log p_cell / d eta(k, j)^2     (diagonal)
//   cross_hess(k, j,
//              l, m)   = -d^2 log p_cell / d eta(k, j) d eta(l, m)
//                                                            (off-diagonal,
//                                                             k <= l ordering
//                                                             enforced by the
//                                                             integration step)
//
// The integration step (see `cell_coupling_integration.md`) scatters these
// per-cell blocks into the joint gradient / Hessian via the same
// (X, Z, latent-cross) bookkeeping that `scatter_arm_obs_joint_multi()`
// (`src/nested_laplace_joint_multi.h`) uses for the separable path. The
// spec ONLY needs to write eta-space derivatives; everything else (chain
// through X, latent cross-terms, prior) is the kernel's job.
//
// Like `CellEtas`, this is a non-owning view of inner-Newton scratch; the
// kernel pre-allocates these buffers per cell and the spec writes into them.
// Buffer sizes are pre-set by the kernel from `n_rows_in_arm(k)`; the spec
// must not write out of range.
// ----------------------------------------------------------------------------
struct CellDerivs {
    // Per-arm gradient buffers. arm_grad[k] is length n_rows_in_arm(k).
    double* const* arm_grad = nullptr;

    // Per-arm diagonal negative-Hessian buffers. arm_neg_hess_diag[k] is
    // length n_rows_in_arm(k); entry j is -d^2 log p_cell / d eta(k,j)^2.
    double* const* arm_neg_hess_diag = nullptr;

    // Per-arm cross-Hessian buffers, ROW-MAJOR. arm_cross_hess[k][l] is
    // a (n_rows_in_arm(k) * n_rows_in_arm(l))-length buffer storing
    // -d^2 log p_cell / d eta(k,j) d eta(l,m) at index (j * n_rows_in_arm(l) + m).
    // Only k <= l blocks are populated; the integration step symmetrises.
    // Off-diagonal blocks may be nullptr for arm pairs the spec leaves
    // uncoupled at this cell -- the kernel treats nullptr as "this pair
    // contributes nothing here".
    double* const* const* arm_cross_hess = nullptr;

    // Per-arm row count for this cell (same as in CellEtas).
    const int* arm_row_count = nullptr;

    int n_arms_ = 0;

    // Batched multi-response (gcol33/tulpa#66). When n_batch_ > 1 the kernel
    // sizes arm_grad[k] / arm_neg_hess_diag[k] to rc * n_batch_ and the spec
    // writes species s's row j at index [s * rc + j] (species-major). The
    // cross buffers arm_cross_hess[k][l] are sized rc_k * rc_l * n_batch_ with
    // species s's (j, m) entry at [s * (rc_k * rc_l) + j * rc_l + m] (same
    // species only; no cross-species curvature). B=1 leaves n_batch_ = 1 so
    // [0 * rc + j] == [j], byte-identical to the pre-batch layout. Appended
    // last to keep existing field offsets unchanged.
    int n_batch_ = 1;

    // Which curvature the kernel is requesting for this scatter. A spec that
    // implements Fisher scoring branches on it; specs that do not simply
    // ignore it and always write the observed Hessian. Appended last so the
    // vtable and the leading fields are unchanged.
    CurvatureMode curvature = CurvatureMode::Observed;

    // Gradient-only request. When true the kernel will NOT use the Hessian this
    // call produces (it is reusing a cached factorization for this inner Newton
    // step -- the Shamanskii / chord path behind control$inner_refresh), so a
    // spec MAY skip its negative-Hessian work (e.g. the digamma/trigamma terms
    // of a beta arm) and leave `arm_neg_hess_diag` / `arm_cross_hess` at the
    // zeros the kernel pre-fills. The gradient (`arm_grad`) and the returned
    // log-density MUST still be exact -- only the curvature may be omitted. A
    // spec that does not implement this simply ignores it and writes the full
    // Hessian (correct, just no saving). Appended last to keep the leading
    // fields and any existing field offsets unchanged.
    bool grad_only = false;

    // Optional per-arm rank-1 self-cross descriptor (gcol33/tulpaObs#94). When
    // a coupled arm k's (k, k) off-diagonal cross-Hessian is exactly the
    // symmetric rank-1 a * v v^T -- every cross-row second derivative factoring
    // through one scalar, as in the all-undetected occupancy mixture whose
    // density depends on the arm's rows only through the scalar
    // P0 = prod_v (1 - p_v) -- a spec MAY declare it here instead of filling the
    // dense arm_cross_hess[k][k] block. The spec sets
    //     arm_cross_rank1_coef[k] = a          (0 keeps the dense path)
    //     arm_cross_rank1_vec[k]  = v          (length n_rows_in_arm(k))
    // and folds the rank-1's own diagonal into arm_neg_hess_diag[k] (storing the
    // true diagonal minus a * v[r]^2). The kernel scatters the term as one
    // a * u u^T in joint-dof space (u = sum_r v[r] * chain(row_r)) instead of the
    // O(rc^2) dense arm_cross_hess[k][k] loop, collapsing the all-undetected
    // scatter from O(sum rc^2) to O(sum rc). Only the (k, k) self block is
    // rank-1-eligible; cross-arm (k != l) blocks always use arm_cross_hess. The
    // kernel honours the descriptor only on the single-response path
    // (n_batch_ == 1); a batched spec keeps the dense arm_cross_hess path.
    // Buffers are kernel-provided writable scratch (arm_cross_rank1_coef
    // pre-zeroed length n_arms; arm_cross_rank1_vec[k] a writable rc_k buffer);
    // nullptr keeps the dense path, so a spec that does not implement this is
    // unaffected. Appended last to keep existing field offsets unchanged.
    double*        arm_cross_rank1_coef = nullptr;  // [n_arms], 0 = dense path
    double* const* arm_cross_rank1_vec  = nullptr;  // [n_arms] -> [rc_k]

    int n_arms() const { return n_arms_; }
    int n_batch() const { return n_batch_; }
    int n_rows_in_arm(int k) const { return arm_row_count[k]; }
};

// ----------------------------------------------------------------------------
// CellCouplingSpec -- abstract base for a cell-coupled likelihood.
//
// One spec object describes how to evaluate `log p_cell` and its eta-space
// derivatives for every cell of one joint fit. The spec is constructed in
// the consumer package (which captures any per-cell metadata it needs at
// construction time), registered under a name via `register_cell_coupling()`,
// and looked up by the joint driver from a user-supplied
// `cell_coupling = "<name>"` argument.
//
// Concrete subclasses live either in `tulpa/src/` (e.g. the separable
// default in `cell_coupling_separable.h`) or in a consumer package's `src/`
// (e.g. tulpaObs's `occu_cover_lognormal` spec). The registry is shared
// across both.
// ----------------------------------------------------------------------------
struct CellCouplingSpec {
    virtual ~CellCouplingSpec() = default;

    // The 0-based arm indices this spec couples. The kernel uses this to
    // decide which arms route their per-cell contribution through
    // `evaluate_cell()` (the coupled ones) versus the existing
    // arm-separable per-obs path (the rest). Arms NOT in this list keep the
    // current `scatter_arm_obs_joint_multi()` path unchanged.
    virtual std::vector<int> arm_ids() const = 0;

    // Evaluate `log p_cell` at cell `cell_idx` and fill the per-arm
    // gradient / negative-Hessian buffers in `out`. Returns the scalar
    // `log p_cell`; the kernel accumulates it into the joint log-lik.
    //
    // Arguments:
    //   cell_idx -- 0-based cell index, in [0, n_cells).
    //   etas     -- per-arm eta view at this cell's rows.
    //   y_cell   -- per-arm response view at this cell's rows.
    //   out      -- writable per-arm derivative buffers (pre-sized by the
    //                kernel from n_rows_in_arm).
    //
    // The spec MAY assume the buffers in `out` are zeroed on entry; the
    // kernel guarantees this. It MUST NOT touch buffers for arms outside
    // `arm_ids()`.
    virtual double evaluate_cell(int cell_idx,
                                 const CellEtas&     etas,
                                 const CellResponse& y_cell,
                                 CellDerivs&         out) const = 0;

    // Optional human-readable name for diagnostics. Defaults to empty; the
    // registry already uniquely keys specs by their registration string, so
    // this is purely a debugging aid.
    virtual std::string name() const { return std::string(); }

    // Whether `evaluate_cell()` is safe to call concurrently across cells
    // from an OpenMP-parallel cell loop. Mirrors `REGroupOracle::thread_safe()`.
    // Native C++ specs that read only their own const member state should
    // return true; R-closure bridges or specs that mutate scratch must
    // return false (the kernel then falls back to a serial cell loop).
    virtual bool thread_safe() const { return true; }

    // Which (kk, ll) coupled-arm index pairs (kk <= ll, indices into the order
    // returned by arm_ids()) this spec writes into the DENSE per-cell
    // cross-Hessian buffer `out.arm_cross_hess[kk][ll]`. The joint cell loop
    // allocates a dense rc_kk * rc_ll slab ONLY for these pairs; every other
    // pair gets a nullptr buffer, which the spec must then leave untouched.
    //
    // This bounds a cell with J observations on a self-coupled arm to O(J)
    // instead of O(J^2): a self pair (kk, kk) that the spec emits through the
    // rank-1 self-cross descriptor (arm_cross_rank1_*) is simply omitted, and a
    // cross pair the spec never writes (a factorising likelihood) is omitted too.
    // `rank1_self_supported` is true when the calling engine supplies the rank-1
    // descriptor (the single-response path); a spec that needs a dense self block
    // when the rank-1 path is unavailable (e.g. batched) keys on this flag.
    //
    // The default returns every kk <= ll pair, i.e. the historical "allocate all"
    // behaviour, so a spec that does not override this is unaffected. Declared
    // last so the vtable slot is appended (existing slots keep their offsets).
    virtual std::vector<std::pair<int, int>> dense_cross_pairs(
            int n_coupled, bool /*rank1_self_supported*/) const {
        std::vector<std::pair<int, int>> pairs;
        for (int kk = 0; kk < n_coupled; kk++)
            for (int ll = kk; ll < n_coupled; ll++)
                pairs.emplace_back(kk, ll);
        return pairs;
    }
};

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_H
