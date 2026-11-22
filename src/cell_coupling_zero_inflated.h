// cell_coupling_zero_inflated.h
// The zero-inflated mixture as a cell-coupled likelihood over two joint arms.
//
// A zero-inflated response has two linear predictors per row: the count
// predictor eta (arm 0, the response's own design, fields and random effects)
// and the logit of the structural-zero probability z (arm 1, the ziformula's
// design). The row's density
//
//     y = 0:  log(pi + (1 - pi) p0(eta)),     pi = inv_logit(z)
//     y > 0:  log(1 - pi) + log f(y | eta)
//
// couples the two predictors through its 2 x 2 curvature block. The joint
// driver already carries two arms with their own designs and latent blocks,
// and a cell-coupled spec already scatters a per-cell cross-arm curvature
// through the same X / RE / latent bookkeeping as the per-row path, so the
// mixture is that spec with one row of each arm per cell. Row j of arm 0
// pairs with row j of arm 1 within a cell.
//
// The mixture math is zi::mixture_ll_double / zi::mixture_eta_weights_double
// (builtin_family_zi.h), the one implementation every compiled path reads. The
// count family, its trials and its dispersion are arm 0's; arm 1's response
// is not read. A missing count response (NaN) contributes nothing, as on the
// per-row path.

#ifndef TULPA_CELL_COUPLING_ZERO_INFLATED_H
#define TULPA_CELL_COUPLING_ZERO_INFLATED_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "builtin_family_zi.h"
#include "tulpa/cell_coupling.h"

namespace tulpa {

class ZeroInflatedCellCoupling final : public CellCouplingSpec {
public:
    std::vector<int> arm_ids() const override { return {0, 1}; }

    double evaluate_cell(int                 /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        const int rc = etas.n_rows_in_arm(0);
        const int B  = etas.n_batch();
        const std::string family(y_cell.family(0));
        const double phi2 = std::numeric_limits<double>::quiet_NaN();
        double* cross = out.arm_cross_hess[0][1];
        double cell_ll = 0.0;
        for (int s = 0; s < B; s++) {
            const std::size_t base = (std::size_t) s * rc;
            const double phi = y_cell.phi(0, s);
            for (int j = 0; j < rc; j++) {
                const double y = y_cell.y(0, j, s);
                if (std::isnan(y)) continue;
                const int nt = y_cell.arm_n_trials && y_cell.arm_n_trials[0]
                                   ? y_cell.n_trials(0, j) : 1;
                const double eta = etas.eta(0, j, s);
                const double z   = etas.eta(1, j, s);
                cell_ll += zi::mixture_ll_double(y, nt, eta, z, family, phi,
                                                 phi2);
                double g[2], h[4];
                zi::mixture_eta_weights_double(y, nt, eta, z, family, phi,
                                               phi2, g, h);
                out.arm_grad[0][base + j] = g[0];
                out.arm_grad[1][base + j] = g[1];
                if (out.grad_only) continue;
                out.arm_neg_hess_diag[0][base + j] = h[0];
                out.arm_neg_hess_diag[1][base + j] = h[3];
                if (cross) {
                    cross[(std::size_t) s * rc * rc + (std::size_t) j * rc + j] =
                        h[1];
                }
            }
        }
        return cell_ll;
    }

    std::string name() const override { return std::string("zero_inflated"); }

    bool thread_safe() const override { return true; }

    // Only the count-by-zero cross block is written: within an arm the rows of
    // a cell do not interact.
    std::vector<std::pair<int, int>> dense_cross_pairs(
            int /*n_coupled*/, bool /*rank1_self_supported*/) const override {
        return {{0, 1}};
    }
};

inline std::shared_ptr<CellCouplingSpec> make_zero_inflated_cell_coupling() {
    return std::make_shared<ZeroInflatedCellCoupling>();
}

} // namespace tulpa

#endif // TULPA_CELL_COUPLING_ZERO_INFLATED_H
