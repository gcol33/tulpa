// st_type_iv_precision.h
// The Type-IV spatiotemporal interaction precision as a MATRIX.
//
// tulpa_priors_st.h owns the Type-IV prior as a QUADRATIC FORM
// (st_kronecker_temporal_quad + st_sum_to_zero_penalty): that is what the
// log-posterior and every autodiff gradient evaluate, and it never needs the
// matrix. The precision-informed mass override does, so this header assembles
// the same operator in sparse form.
//
// Two definitions of one operator can drift, so the matrix is not the
// authority: test-st-iv-precision.R pins x' Q x against the quadratic forms in
// tulpa_priors_st.h on random x, for RW1 and RW2 and for a graph with an
// isolated unit. Change the prior and that test goes red here.
//
// Layout: delta is indexed s * T + t (temporal varies fastest), the convention
// SpatiotemporalData::st_flat and the prior's `delta.data() + s * T_st` reads
// both carry. So the Kronecker order is Q_s (x) Q_t.
#ifndef TULPA_ST_TYPE_IV_PRECISION_H
#define TULPA_ST_TYPE_IV_PRECISION_H

#include <cstddef>
#include <vector>

#include <Eigen/SparseCore>

#include "tulpa/model_data.h"
#include "tulpa/soft_sum_to_zero.h"
#include "tulpa/types.h"

namespace tulpa_st {

using tulpa::ModelData;
using tulpa::TemporalType;

// Number of stored entries the assembly below emits, before duplicate summing.
// Read by the caller as a budget: the two sum-to-zero margins contribute
// S^2 * T and S * T^2, which outgrow the Kronecker term itself on any wide
// field, and a mass-matrix override is not worth an allocation the fit cannot
// afford.
inline std::size_t st_type_iv_triplet_count(const ModelData& data, int S, int T) {
    const auto& st = data.spatiotemporal_data;
    const std::size_t Sz = (std::size_t)S, Tz = (std::size_t)T;

    // Entries st_add_qt_entries emits for one S-block: 4 per first difference
    // (RW1), 9 per second difference (RW2). Both bounded above by the count
    // below, which is what makes this an upper bound rather than an estimate.
    const std::size_t qt_emit =
        (st.temporal_type == TemporalType::RW2) ? (Tz * 9) : (Tz * 4);

    // Blocks written: one diagonal per spatial unit plus one per directed
    // adjacency edge.
    const std::size_t n_blocks =
        Sz + (st.adj_col_idx.empty() ? 0 : st.adj_col_idx.size());

    return n_blocks * qt_emit       // tau * (Q_s (x) Q_t)
         + Sz * Tz                  // diag(h_lik) + ridge
         + Sz * Tz * Tz             // lambda_row * (I_S (x) J_T)
         + Sz * Sz * Tz;            // lambda_col * (J_S (x) I_T)
}

// Q_t on pattern: the RW1 / RW2 precision D' D for the ACYCLIC differences.
// Acyclic unconditionally, because st_kronecker_temporal_quad passes
// cyclic = false to rw1/rw2_quadratic_form whatever
// SpatiotemporalData::temporal_cyclic says. This matches the density that is
// EVALUATED, which is the operator a mass matrix has to be built from.
inline void st_add_qt_entries(TemporalType type, int T,
                              std::vector<Eigen::Triplet<double>>& out,
                              int row_base, int col_base, double scale) {
    if (type == TemporalType::RW1) {
        // sum_{t=1}^{T-1} (a_t - a_{t-1})(b_t - b_{t-1}): D1' D1.
        for (int t = 1; t < T; t++) {
            out.emplace_back(row_base + t,     col_base + t,      scale);
            out.emplace_back(row_base + t - 1, col_base + t - 1,  scale);
            out.emplace_back(row_base + t,     col_base + t - 1, -scale);
            out.emplace_back(row_base + t - 1, col_base + t,     -scale);
        }
    } else if (type == TemporalType::RW2) {
        // sum_{t=2}^{T-1} d2_a[t] d2_b[t] with d2[t] = x_t - 2 x_{t-1} + x_{t-2}:
        // D2' D2, emitted as the outer product of each stencil row.
        if (T < 3) return;
        const double w[3] = {1.0, -2.0, 1.0};   // offsets t-2, t-1, t
        for (int t = 2; t < T; t++) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    out.emplace_back(row_base + t - 2 + a,
                                     col_base + t - 2 + b,
                                     scale * w[a] * w[b]);
                }
            }
        }
    }
}

// Assemble the Gaussian-approximation posterior precision of the Type-IV
// interaction block, in the coordinate the sampler holds:
//
//   Q = kron_scale * (Q_s (x) Q_t)
//     + h_scale * diag(h_lik)
//     + s2z_scale * (lambda_row * (I_S (x) J_T) + lambda_col * (J_S (x) I_T))
//     + ridge * I
//
// The three scales are what separates the two parameterizations. Centered
// (delta sampled): the prior carries tau, the likelihood and the penalty read
// delta itself, so (tau, 1, 1). Non-centered (z sampled, delta = z / sqrt(tau)):
// the prior is tau-free and both the likelihood and the penalty pick up the
// chain rule twice, so (1, 1/tau, 1/tau).
//
// `h_lik` is the per-coordinate eta-space likelihood curvature; empty means a
// prior-only precision. Returns false without touching `Q` when the assembly
// would exceed `max_triplets`.
inline bool st_type_iv_precision(
    const ModelData& data,
    int S, int T,
    double kron_scale,
    double h_scale,
    const std::vector<double>& h_lik,
    double s2z_scale,
    double ridge,
    std::size_t max_triplets,
    Eigen::SparseMatrix<double>& Q
) {
    if (S <= 0 || T <= 0) return false;
    const std::size_t need = st_type_iv_triplet_count(data, S, T);
    if (need > max_triplets) return false;

    const auto& st = data.spatiotemporal_data;
    const int ST = S * T;

    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(need);

    // kron_scale * (Q_s (x) Q_t). Q_s = diag(n_neighbors) - A, read from the
    // same two arrays the prior's quadratic form walks.
    for (int s = 0; s < S; s++) {
        const int deg = st.n_neighbors.empty() ? 0 : st.n_neighbors[s];
        if (deg != 0) {
            st_add_qt_entries(st.temporal_type, T, trip, s * T, s * T,
                              kron_scale * (double)deg);
        }
        if (st.adj_row_ptr.empty()) continue;
        for (int jj = st.adj_row_ptr[s]; jj < st.adj_row_ptr[s + 1]; jj++) {
            const int s2 = st.adj_col_idx[jj] - 1;   // stored 1-based
            if (s2 < 0 || s2 >= S) continue;
            st_add_qt_entries(st.temporal_type, T, trip, s * T, s2 * T,
                              -kron_scale);
        }
    }

    // Likelihood curvature and the numerical ridge, both diagonal.
    const bool have_h = ((int)h_lik.size() == ST);
    for (int k = 0; k < ST; k++) {
        double d = ridge;
        if (have_h) d += h_scale * h_lik[k];
        if (d != 0.0) trip.emplace_back(k, k, d);
    }

    // Soft sum-to-zero on both margins. Each margin's own length sets its
    // precision, which is what s2z_precision's contract asks for.
    if (s2z_scale != 0.0) {
        const double lambda_row = s2z_scale * tulpa::s2z_precision(T);
        const double lambda_col = s2z_scale * tulpa::s2z_precision(S);
        for (int s = 0; s < S; s++) {                 // I_S (x) J_T
            for (int t1 = 0; t1 < T; t1++) {
                for (int t2 = 0; t2 < T; t2++) {
                    trip.emplace_back(s * T + t1, s * T + t2, lambda_row);
                }
            }
        }
        for (int t = 0; t < T; t++) {                 // J_S (x) I_T
            for (int s1 = 0; s1 < S; s1++) {
                for (int s2 = 0; s2 < S; s2++) {
                    trip.emplace_back(s1 * T + t, s2 * T + t, lambda_col);
                }
            }
        }
    }

    Q.resize(ST, ST);
    Q.setFromTriplets(trip.begin(), trip.end());
    Q.makeCompressed();
    return true;
}

}  // namespace tulpa_st

#endif  // TULPA_ST_TYPE_IV_PRECISION_H
