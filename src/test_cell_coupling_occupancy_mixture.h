// test_cell_coupling_occupancy_mixture.h
// Test-only CellCouplingSpec exposing a two-arm occupancy mixture: the
// smallest genuinely NON-SEPARABLE per-cell density the engine can fit
// entirely on its own (gcol33/tulpa#300).
//
// ============================================================================
// The density
// ============================================================================
//
// Cell c carries one row on arm 0 (the occupancy arm, eta_a = logit psi_c) and
// J_c rows on arm 1 (the detection arm, eta_v = logit p_cv) with binary
// responses y_cv. The per-cell density is the two-state mixture
//
//   p_cell = psi_c * prod_v Bern(y_cv | p_cv)
//          + (1 - psi_c) * 1{sum_v y_cv = 0}.
//
// Two branches, both exact:
//
//   S_c > 0 (something was detected, so the cell is occupied):
//       log p_cell = log psi_c + sum_v [y log p_v + (1 - y) log q_v],
//     which FACTORISES -- every cross derivative is zero at such a cell.
//
//   S_c = 0 (nothing was detected):
//       log p_cell = log(psi_c * P0_c + 1 - psi_c),   P0_c = prod_v q_cv,
//     which does NOT factorise: the occupancy state modulates the detection
//     arm's contribution inside one logarithm, so d^2 log p_cell / d eta_a
//     d eta_v and d^2 log p_cell / d eta_v d eta_w are both nonzero. This is
//     the shape tulpaObs's occu_cover carries, reduced to two arms and binary
//     detections.
//
// ============================================================================
// Closed forms (all-undetected branch)
// ============================================================================
//
// With psi = sigmoid(eta_a), m = 1 - psi, p_v = sigmoid(eta_v), q_v = 1 - p_v,
// P0 = prod_v q_v and D = psi P0 + m:
//
//   dL/d eta_a       =  psi m (P0 - 1) / D                        =: g_a
//   dL/d eta_v       = -psi P0 p_v / D                            =: g_v
//   d2L/d eta_a^2    =  (P0 - 1) psi m (m - psi) / D  -  g_a^2
//   d2L/d eta_v^2    = -psi P0 p_v (q_v - p_v) / D    -  g_v^2
//   d2L/d eta_a d eta_v = -psi m P0 p_v / D^2
//   d2L/d eta_v d eta_w =  psi m P0 p_v p_w / D^2       (v != w)
//
// `CellDerivs` takes the NEGATIVE second derivatives, so each of the six lines
// above is written with its sign flipped. The (arm 0, arm 1) block and the
// (arm 1, arm 1) off-diagonal block are both filled DENSELY -- the rank-1
// self-cross shortcut the header documents is deliberately not taken here, so
// the finite-difference third-derivative tensor gcol33/tulpa#301 needs has a
// full explicit Hessian to difference.
//
// Registered under the name "test_occupancy_mixture" by
// `cpp_register_test_occupancy_mixture_coupling()` in
// src/test_cell_coupling_register.cpp. Lives under src/ rather than
// inst/include/ because no other package consumes it.

#ifndef TULPA_TEST_CELL_COUPLING_OCCUPANCY_MIXTURE_H
#define TULPA_TEST_CELL_COUPLING_OCCUPANCY_MIXTURE_H

#include "tulpa/cell_coupling.h"
#include "autodiff_utils.h"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

// log sigmoid(x), evaluated on whichever side keeps exp() bounded.
inline double test_occ_log_inv_logit(double x) {
    return (x >= 0.0) ? -std::log1p(std::exp(-x))
                      : x - std::log1p(std::exp(x));
}

class TestOccupancyMixtureCoupling final : public CellCouplingSpec {
public:
    // Arm 0 is the occupancy arm (one row per cell); arm 1 is the detection
    // arm (J_c rows per cell).
    std::vector<int> arm_ids() const override { return {0, 1}; }

    // The occupancy arm holds one row per cell, so its self block has no
    // off-diagonal entry to write and is omitted. The (0, 1) cross block and
    // the (1, 1) detection self block are both dense.
    std::vector<std::pair<int, int>> dense_cross_pairs(
            int n_coupled, bool /*rank1_self_supported*/) const override {
        std::vector<std::pair<int, int>> pairs;
        if (n_coupled < 2) return pairs;
        pairs.emplace_back(0, 1);
        pairs.emplace_back(1, 1);
        return pairs;
    }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        if (etas.n_rows_in_arm(0) < 1) return 0.0;

        const int    J    = etas.n_rows_in_arm(1);
        const double a    = etas.eta(0, 0);
        const double psi  = math::inv_logit(a);
        const double m    = math::inv_logit(-a);   // 1 - psi, evaluated stably

        std::vector<double> p(J), q(J);
        double S = 0.0, log_P0 = 0.0;
        for (int v = 0; v < J; v++) {
            const double e = etas.eta(1, v);
            p[v] = math::inv_logit(e);
            q[v] = math::inv_logit(-e);
            S   += y_cell.y(1, v);
            log_P0 += test_occ_log_inv_logit(-e);
        }

        const bool want_hess = !out.grad_only;

        if (S > 0.0) {
            // Occupied with certainty: the density factorises into the
            // occupancy Bernoulli and the per-visit detection Bernoullis.
            double ll = test_occ_log_inv_logit(a);
            out.arm_grad[0][0] = m;
            if (want_hess) out.arm_neg_hess_diag[0][0] = psi * m;
            for (int v = 0; v < J; v++) {
                const double y = y_cell.y(1, v);
                ll += y * test_occ_log_inv_logit(etas.eta(1, v))
                    + (1.0 - y) * test_occ_log_inv_logit(-etas.eta(1, v));
                out.arm_grad[1][v] = y - p[v];
                if (want_hess) out.arm_neg_hess_diag[1][v] = p[v] * q[v];
            }
            // Cross blocks stay at the zeros the kernel pre-filled.
            return ll;
        }

        // All-undetected: the genuinely coupled branch.
        const double P0 = std::exp(log_P0);
        const double D  = psi * P0 + m;

        const double g_a = psi * m * (P0 - 1.0) / D;
        out.arm_grad[0][0] = g_a;
        for (int v = 0; v < J; v++) {
            out.arm_grad[1][v] = -psi * P0 * p[v] / D;
        }

        if (want_hess) {
            out.arm_neg_hess_diag[0][0] =
                -((P0 - 1.0) * psi * m * (m - psi) / D - g_a * g_a);
            for (int v = 0; v < J; v++) {
                const double g_v = out.arm_grad[1][v];
                out.arm_neg_hess_diag[1][v] =
                    psi * P0 * p[v] * (q[v] - p[v]) / D + g_v * g_v;
            }

            const double cross_scale = psi * m * P0 / (D * D);

            // (arm 0, arm 1): a 1 x J row-major block.
            if (out.arm_cross_hess && out.arm_cross_hess[0]
                && out.arm_cross_hess[0][1]) {
                double* ch = out.arm_cross_hess[0][1];
                for (int v = 0; v < J; v++) ch[v] = cross_scale * p[v];
            }

            // (arm 1, arm 1): the J x J detection self block, written
            // symmetrically. The kernel reads only the strict upper triangle;
            // the lower half is filled so a direct probe of evaluate_cell()
            // sees the whole matrix.
            if (out.arm_cross_hess && out.arm_cross_hess[1]
                && out.arm_cross_hess[1][1]) {
                double* ch = out.arm_cross_hess[1][1];
                for (int v = 0; v < J; v++) {
                    for (int w = v + 1; w < J; w++) {
                        const double h = -cross_scale * p[v] * p[w];
                        ch[(std::size_t)v * J + w] = h;
                        ch[(std::size_t)w * J + v] = h;
                    }
                }
            }
        }

        return std::log(D);
    }

    std::string name() const override {
        return std::string("test_occupancy_mixture");
    }

    bool thread_safe() const override { return true; }
};

} // namespace tulpa

#endif // TULPA_TEST_CELL_COUPLING_OCCUPANCY_MIXTURE_H
