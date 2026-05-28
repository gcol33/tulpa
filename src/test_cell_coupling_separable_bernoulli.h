// test_cell_coupling_separable_bernoulli.h
// Test-only CellCouplingSpec that reproduces a single-arm binomial
// (n_trials = 1) per-obs likelihood as a per-cell coupled spec, with
// one row per cell. Used by tests/testthat/test-cell-coupling-recovery.R
// to gate that the per-cell branch's scatter + log-lik path produces
// byte-equivalent (~1e-12) mode + log-marginal output as the existing
// arm-separable per-obs path on the same data.
//
// Lives under src/ (not inst/include/) because it is consumed only
// internally by `tulpa_register_test_separable_bernoulli_coupling` -- the
// Rcpp export in src/test_cell_coupling_register.cpp.

#ifndef TULPA_TEST_CELL_COUPLING_SEPARABLE_BERNOULLI_H
#define TULPA_TEST_CELL_COUPLING_SEPARABLE_BERNOULLI_H

#include "tulpa/cell_coupling.h"
#include <cmath>
#include <string>
#include <vector>

namespace tulpa {

// Single-arm binomial spec, one row per cell. The per-cell density at
// cell c is `Bern(y_c | sigmoid(eta_c))`:
//   log p_cell  = y * log p + (1 - y) * log (1 - p),   p = sigmoid(eta)
//   d/d eta     = y - p
//   -d^2/d eta^2 = p * (1 - p)
class TestSeparableBernoulliCoupling final : public CellCouplingSpec {
public:
    std::vector<int> arm_ids() const override { return {0}; }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        const int rc = etas.n_rows_in_arm(0);
        double cell_ll = 0.0;
        for (int j = 0; j < rc; j++) {
            const double eta = etas.eta(0, j);
            const double y   = y_cell.y(0, j);
            const double p   = 1.0 / (1.0 + std::exp(-eta));

            const double one_m_p = 1.0 - p;
            const double log_p_safe   = (p     > 0.0) ? std::log(p)     : -1e300;
            const double log_1mp_safe = (one_m_p > 0.0) ? std::log(one_m_p) : -1e300;
            cell_ll += y * log_p_safe + (1.0 - y) * log_1mp_safe;

            out.arm_grad[0][j]          = y - p;
            out.arm_neg_hess_diag[0][j] = p * one_m_p;
        }
        return cell_ll;
    }

    std::string name() const override {
        return std::string("test_separable_bernoulli");
    }

    bool thread_safe() const override { return true; }
};

} // namespace tulpa

#endif // TULPA_TEST_CELL_COUPLING_SEPARABLE_BERNOULLI_H
