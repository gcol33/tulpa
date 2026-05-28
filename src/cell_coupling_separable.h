// cell_coupling_separable.h
// Default `CellCouplingSpec` for the arm-separable per-obs sum -- the
// likelihood shape every existing joint nested-Laplace consumer uses.
//
// Installed by `src/cell_coupling_registry.h` as the spec returned by
// `lookup_cell_coupling("separable")`. The joint driver falls back to this
// spec when the user does not pass an explicit `cell_coupling = "<name>"`,
// so every existing joint fit (the cover hurdle in particular) stays on
// its current code path.
//
// ============================================================================
// What "separable" means
// ============================================================================
//
// The existing per-obs path computes, at the inner Newton's per-cell loop,
//
//     log p_cell = sum_{arm k in cell c} sum_{obs i in cell c, arm k}
//                       log f_arm(y_i | eta_arm[i]),
//
// with the per-obs gradient and (diagonal in eta) Fisher curvature already
// available via `arm_grad_hess()` (`src/laplace_newton_joint.h`). Every
// production joint fit (the cover hurdle, every single-arm nested-Laplace
// reduction) is arm-separable -- the cross-arm Hessian blocks are zero at
// the obs level, the only coupling between arms is through the shared
// latent field's quadratic prior, which the kernel handles outside the
// per-cell branch.
//
// ============================================================================
// Why this is a sentinel, not a full re-implementation
// ============================================================================
//
// A literal `evaluate_cell()` implementation here would have to duplicate
// the per-obs gradient / Hessian accumulation that
// `scatter_arm_obs_joint_multi()` (`src/nested_laplace_joint_multi.h`)
// already does -- including the `arm_grad_hess()` call that already routes
// through every arm's `LikelihoodSpec` (built-in family or model-supplied),
// the per-arm cross with `X` / RE / shared field bookkeeping, etc. Two
// copies of that logic would drift, contradict the no-copy-paste rule, and
// in steady state add zero capability over the existing path.
//
// Instead, `SeparableCellCoupling` is a *marker* spec: its `arm_ids()`
// returns the empty list (no arm routes through this spec) and its
// `evaluate_cell()` returns 0.0 without touching the derivative buffers.
// The joint driver's per-cell branch reads:
//
//     if (spec->arm_ids().empty()) {
//         // legacy per-obs path
//     } else {
//         spec->evaluate_cell(c, etas, y_cell, derivs);
//     }
//
// so every arm of every fit using the separable default stays on the
// existing per-obs scatter with no behavioural change.
//
// The factory function `make_separable_cell_coupling()` returns a
// `std::shared_ptr<CellCouplingSpec>` so it can be handed to the registry
// (`src/cell_coupling_registry.h`) under the well-known name `"separable"`.

#ifndef TULPA_CELL_COUPLING_SEPARABLE_H
#define TULPA_CELL_COUPLING_SEPARABLE_H

#include <memory>
#include <string>
#include <vector>

#include "tulpa/cell_coupling.h"

namespace tulpa {

// ----------------------------------------------------------------------------
// SeparableCellCoupling -- marker spec for the arm-separable per-obs sum.
//
// `arm_ids()` returns an empty list so the integration step dispatches every
// arm to the existing `scatter_arm_obs_joint_multi()` path. `evaluate_cell()`
// is a no-op (returns 0.0 and writes nothing) and will not be called by the
// integration step in steady state -- it is provided only so the abstract
// base's pure-virtual contract is fulfilled and so a future literal
// separable implementation has a hook to grow into.
// ----------------------------------------------------------------------------
class SeparableCellCoupling final : public CellCouplingSpec {
public:
    std::vector<int> arm_ids() const override {
        return {};  // no arms coupled -> legacy per-obs path handles every arm
    }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas& /*etas*/,
                         const CellResponse& /*y_cell*/,
                         CellDerivs&     /*out*/) const override {
        // Sentinel: the integration step short-circuits on `arm_ids().empty()`
        // and never reaches here in production. The body is left at 0.0 so a
        // misrouted call yields a harmless additive identity instead of NaN.
        return 0.0;
    }

    std::string name() const override { return std::string("separable"); }

    bool thread_safe() const override { return true; }
};

// ----------------------------------------------------------------------------
// Factory: heap-allocate the canonical separable default. Called once from
// `src/cell_coupling_registry.h` at tulpa DLL load (registry pre-registers
// the result under the name "separable").
// ----------------------------------------------------------------------------
inline std::shared_ptr<CellCouplingSpec> make_separable_cell_coupling() {
    return std::make_shared<SeparableCellCoupling>();
}

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_SEPARABLE_H
