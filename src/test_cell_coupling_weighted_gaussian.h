// test_cell_coupling_weighted_gaussian.h
// Test-only CellCouplingSpec that reads the two pieces of the batched
// CellResponse surface no other in-repo spec touches: the per-species
// dispersion `phi(k, s)` and the per-row trial count `n_trials(k, j)`
// (gcol33/tulpa#592, gcol33/tulpa#591).
//
// The batched driver lays the dispersion table as `buf.phi[k * n_batch + s]`
// and its two readers re-key it independently -- the fused scatter builds
// `arm_phi_batch[kk * B + s]` on the coupled-arm index, `species_cell_loglik`
// builds a length-n_coupled vector on a B = 1 view. Two re-keyings of one
// table, on the curvature side and the objective side of the same fit. Every
// other registered spec is binomial and calls neither accessor, so a
// transposed or arm-id-vs-coupled-index slip moved nothing and no test could
// see it. This spec makes both load-bearing: a species whose phi moves must
// move in BOTH reads, and only that species.
//
// The density is a replicate-weighted gaussian, summed over the spec's arms:
//
//   log p_cell = sum_j n_j * [ -0.5 log(2 pi phi_s) - 0.5 (y_j - eta_j)^2 / phi_s ]
//   d/d eta_j     = n_j * (y_j - eta_j) / phi_s
//   -d^2/d eta_j^2 = n_j / phi_s
//
// `n_j` is the row's trial count read as a replicate weight, which is the
// cheapest way to make `n_trials(k, j)` change the answer on a family that has
// no trials of its own: weight w at dispersion phi is exactly weight 1 at
// dispersion phi / w, so the read is pinned by an identity rather than by a
// tolerance. Before the batched entry filled the buffer, `arm_n_trials[k]` was
// null on every batched fit and this accessor dereferenced it.
//
// The coupled arm ids are a constructor argument so a fixture can couple TWO
// arms at once. That is what makes the `[k * n_batch + s]` layout arbitrable:
// with one arm, or with one species, a transposed index reads the same cell and
// no fixture can tell. With n_arms = 2 and B >= 2 it does not.
//
// Lives under src/ (not inst/include/) because it is consumed only by the Rcpp
// registration export in src/test_cell_coupling_register.cpp.

#ifndef TULPA_TEST_CELL_COUPLING_WEIGHTED_GAUSSIAN_H
#define TULPA_TEST_CELL_COUPLING_WEIGHTED_GAUSSIAN_H

#include "tulpa/cell_coupling.h"
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace tulpa {

class TestWeightedGaussianCoupling final : public CellCouplingSpec {
public:
    explicit TestWeightedGaussianCoupling(std::vector<int> arm_ids)
        : arm_ids_(std::move(arm_ids)) {}

    std::vector<int> arm_ids() const override { return arm_ids_; }

    double evaluate_cell(int             /*cell_idx*/,
                         const CellEtas&     etas,
                         const CellResponse& y_cell,
                         CellDerivs&         out) const override {
        const int n_coupled = etas.n_arms();
        const int B = etas.n_batch();
        const double kLog2Pi = 1.8378770664093454835606594728112;
        double cell_ll = 0.0;
        // `k` is the COUPLED-arm index the cell view is laid on, not the arm
        // id -- which is exactly the re-keying each batched reader performs on
        // the dispersion table.
        for (int k = 0; k < n_coupled; k++) {
            const int rc = etas.n_rows_in_arm(k);
            // Species-major buffers, as in TestSeparableBernoulliCoupling:
            // species s owns [s * rc, (s + 1) * rc) of the per-arm gradient and
            // curvature. At B = 1 the offset is zero and the three-argument
            // accessors reduce to the two-argument ones.
            for (int s = 0; s < B; s++) {
                const std::size_t base = (std::size_t) s * rc;
                // The per-species dispersion. `phi(k, s)` falls back to the
                // shared arm_phi[k] when no batch table is supplied, so this
                // same spec serves the single-species path unchanged.
                const double phi = y_cell.phi(k, s);
                const double inv_phi = 1.0 / phi;
                const double log_phi = std::log(phi);
                for (int j = 0; j < rc; j++) {
                    const double eta = etas.eta(k, j, s);
                    const double y   = y_cell.y(k, j, s);
                    // Shared design: no species index on the trial count.
                    const double w   = (double) y_cell.n_trials(k, j);
                    const double r   = y - eta;

                    cell_ll += w * (-0.5 * (kLog2Pi + log_phi)
                                    - 0.5 * r * r * inv_phi);

                    out.arm_grad[k][base + j] = w * r * inv_phi;
                    // On a grad-only step the kernel discards the Hessian, so
                    // the pre-filled zero is left alone.
                    if (!out.grad_only) {
                        out.arm_neg_hess_diag[k][base + j] = w * inv_phi;
                    }
                }
            }
        }
        return cell_ll;
    }

    std::string name() const override {
        return std::string("test_weighted_gaussian");
    }

    bool thread_safe() const override { return true; }

private:
    std::vector<int> arm_ids_;
};

} // namespace tulpa

#endif // TULPA_TEST_CELL_COUPLING_WEIGHTED_GAUSSIAN_H
