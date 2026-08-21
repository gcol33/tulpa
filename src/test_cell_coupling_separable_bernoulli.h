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
#include <cstddef>
#include <string>
#include <vector>

namespace tulpa {

// Single-arm binomial spec, one row per cell. The per-cell density at
// cell c is `Bern(y_c | sigmoid(eta_c))`:
//   log p_cell  = y * eta - log(1 + exp(eta)),   p = sigmoid(eta)
//   d/d eta     = y - p
//   -d^2/d eta^2 = p * (1 - p)
//
// The log-density and the sigmoid are branched on the sign of eta exactly as
// log_lik_binomial_kernel / grad_log_lik_binomial do (laplace_likelihoods.cpp),
// so the reference stays exact wherever the path it validates is. Forming
// 1 / (1 + exp(-eta)) and then log(1 - p) instead rounds p to exactly 1 at
// eta above ~37, where log(1 - p) is log(0): the reference would then read
// -Inf (or a sentinel) on a y = 0 row whose true density is -eta.
class TestSeparableBernoulliCoupling final : public CellCouplingSpec {
public:
    std::vector<int> arm_ids() const override { return {0}; }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        const int rc = etas.n_rows_in_arm(0);
        const int B  = etas.n_batch();
        double cell_ll = 0.0;
        // Species-major buffers: species s owns [s * rc, (s + 1) * rc) of the
        // per-arm gradient and curvature. At B = 1 the offset is zero and the
        // three-argument eta / y accessors reduce to the two-argument ones, so
        // the single-response layout is unchanged.
        for (int s = 0; s < B; s++) {
            const std::size_t base = (std::size_t) s * rc;
            for (int j = 0; j < rc; j++) {
                const double eta = etas.eta(0, j, s);
                const double y   = y_cell.y(0, j, s);
                double p, log_lik;
                if (eta > 0.0) {
                    const double e = std::exp(-eta);
                    p       = 1.0 / (1.0 + e);
                    log_lik = y * eta - eta - std::log1p(e);
                } else {
                    const double e = std::exp(eta);
                    p       = e / (1.0 + e);
                    log_lik = y * eta - std::log1p(e);
                }
                cell_ll += log_lik;

                const double one_m_p = 1.0 - p;
                out.arm_grad[0][base + j]          = y - p;
                // On a grad-only step (cached-factor reuse) the kernel discards
                // the Hessian, so leave arm_neg_hess_diag at the pre-filled zero.
                if (!out.grad_only) {
                    out.arm_neg_hess_diag[0][base + j] = p * one_m_p;
                }
            }
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
