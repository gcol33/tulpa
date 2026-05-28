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
#include <string>
#include <vector>

namespace tulpa {

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

    int n_arms() const { return n_arms_; }
    int n_rows_in_arm(int k) const { return arm_row_count[k]; }

    // eta(k, j) -- the eta value at the j-th row of arm k that belongs to
    // this cell. j in [0, n_rows_in_arm(k)).
    double eta(int k, int j) const {
        return arm_eta_ptr[k][arm_rows[k][j]];
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

    int n_arms() const { return n_arms_; }
    int n_rows_in_arm(int k) const { return arm_row_count[k]; }

    // y(k, j) -- the response value at the j-th row of arm k that belongs
    // to this cell. Undefined if arm_y[k] is nullptr (the spec must know
    // which of its arms carry data).
    double y(int k, int j) const {
        return arm_y[k][arm_rows[k][j]];
    }

    int n_trials(int k, int j) const {
        return arm_n_trials[k][arm_rows[k][j]];
    }

    const char* family(int k) const { return arm_family[k]; }
    double phi(int k)               const { return arm_phi[k];    }
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

    int n_arms() const { return n_arms_; }
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
};

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_H
