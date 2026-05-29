// test_cell_coupling_bivariate_gaussian.h
// Test-only CellCouplingSpec exposing a two-arm bivariate Gaussian per
// cell with one row per arm per cell. Used by
// tests/testthat/test-cell-coupling-cross-hess.R to gate the cross-arm
// Hessian scatter path in src/nested_laplace_joint_multi.h (the
// `arm_cross_hess` writes + scatter_cross_chain_{dense,sparse} helpers).
//
// Per-cell density at cell c:
//   log p_cell = -0.5 * (eta_c - y_c)' Lambda (eta_c - y_c)
// where eta_c = (eta(0, 0), eta(1, 0)), y_c = (y(0, 0), y(1, 0)), and
// Lambda is a fixed 2 x 2 precision matrix supplied at construction.
//
// Closed forms:
//   d/d eta(0, 0) =  -lam00 * (eta_0 - y_0) - lam01 * (eta_1 - y_1)
//   d/d eta(1, 0) =  -lam11 * (eta_1 - y_1) - lam01 * (eta_0 - y_0)
//   -d^2/d eta(0, 0)^2     = lam00
//   -d^2/d eta(1, 0)^2     = lam11
//   -d^2/d eta(0, 0) d eta(1, 0) = lam01
//
// The constant `-0.5 * log(2 pi)^2 + 0.5 * log det Lambda` is added to the
// returned log p_cell so the marginal-likelihood path is comparable across
// Lambda values.

#ifndef TULPA_TEST_CELL_COUPLING_BIVARIATE_GAUSSIAN_H
#define TULPA_TEST_CELL_COUPLING_BIVARIATE_GAUSSIAN_H

#include "tulpa/cell_coupling.h"
#include <cmath>
#include <string>
#include <vector>

namespace tulpa {

class TestBivariateGaussianCoupling final : public CellCouplingSpec {
public:
    TestBivariateGaussianCoupling(double lam00, double lam11, double lam01)
        : lam00_(lam00), lam11_(lam11), lam01_(lam01) {
        double det = lam00_ * lam11_ - lam01_ * lam01_;
        log_const_ = -std::log(2.0 * M_PI)
                     + ((det > 0.0) ? 0.5 * std::log(det) : 0.0);
    }

    std::vector<int> arm_ids() const override { return {0, 1}; }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        const double e0 = etas.eta(0, 0);
        const double e1 = etas.eta(1, 0);
        const double y0 = y_cell.y(0, 0);
        const double y1 = y_cell.y(1, 0);

        const double r0 = e0 - y0;
        const double r1 = e1 - y1;

        out.arm_grad[0][0] = -(lam00_ * r0 + lam01_ * r1);
        out.arm_grad[1][0] = -(lam11_ * r1 + lam01_ * r0);

        out.arm_neg_hess_diag[0][0] = lam00_;
        out.arm_neg_hess_diag[1][0] = lam11_;

        if (out.arm_cross_hess && out.arm_cross_hess[0]
            && out.arm_cross_hess[0][1]) {
            out.arm_cross_hess[0][1][0] = lam01_;
        }

        return -0.5 * (lam00_ * r0 * r0 + lam11_ * r1 * r1
                       + 2.0 * lam01_ * r0 * r1)
               + log_const_;
    }

    std::string name() const override {
        return std::string("test_bivariate_gaussian");
    }

    bool thread_safe() const override { return true; }

private:
    double lam00_;
    double lam11_;
    double lam01_;
    double log_const_;
};

} // namespace tulpa

#endif // TULPA_TEST_CELL_COUPLING_BIVARIATE_GAUSSIAN_H
