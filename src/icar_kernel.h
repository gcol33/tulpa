// icar_kernel.h
// Shared CAR / ICAR neighbor-loop kernels.
//
// Q(rho) = D - rho*W is the CAR family precision matrix; ICAR is the special
// case rho = 1.0. Three primitives cover every caller in the codebase:
//
//   - car_quad_form(phi, S, ...)     -> phi' Q(rho) phi   (uses adjacency symmetry)
//   - car_apply(phi, out, S, ...)    -> out = Q(rho) phi  (matvec)
//   - car_apply_row(i, phi, ...)     -> (Q(rho) phi)[i]   (single row, no allocation)
//
// All three accept raw int* / double* so they bind equally well to
// std::vector<int>::data() and Rcpp::IntegerVector::begin().

#ifndef TULPA_ICAR_KERNEL_H
#define TULPA_ICAR_KERNEL_H

#include <vector>

namespace tulpa {

// Number of connected components of the adjacency graph (CSR, 0-based
// adj_col_idx). The ICAR precision D - W has rank S - k for k components (one
// constant null direction per component), so the log|tau Q| normalizer uses
// (S - k)/2, not (S - 1)/2; a disconnected graph (e.g. spatial(by=)
// replication) otherwise biases tau upward. Matches the Laplace path's
// n_components. Iterative DFS (an explicit stack, no recursion depth risk).
// col_base is 0 for a 0-based adj_col_idx (the sampler ModelData adjacency) and
// 1 for a 1-based one (the spatiotemporal adjacency).
inline int count_graph_components(
    int S, const int* adj_row_ptr, const int* adj_col_idx, int col_base = 0
) {
    if (S <= 0) return 0;
    std::vector<int> seen(S, 0);
    std::vector<int> stack;
    int k = 0;
    for (int s0 = 0; s0 < S; s0++) {
        if (seen[s0]) continue;
        seen[s0] = 1;
        stack.clear();
        stack.push_back(s0);
        while (!stack.empty()) {
            int s = stack.back();
            stack.pop_back();
            const int row_end = adj_row_ptr[s + 1];
            for (int e = adj_row_ptr[s]; e < row_end; e++) {
                int t = adj_col_idx[e] - col_base;
                if (t >= 0 && t < S && !seen[t]) {
                    seen[t] = 1;
                    stack.push_back(t);
                }
            }
        }
        k++;
    }
    return k;
}

// Visit each of an intrinsic field's `n_components` disjoint, equal-size,
// contiguous connected components, calling comp(comp_start_absolute,
// comp_size). A connected graph has one component spanning [start, start + n)
// (n_components <= 1, byte-identical to a single whole-field pass); a
// replicated field over the block-diagonal I_L (x) Q has L equal-size
// components (the `by =` replicated CAR). The single source of the
// per-component loop so every engine's gradient, Hessian, pattern and
// log-prior treat the null space identically -- the ICAR rank normalizer is
// J - n_components, so the sum-to-zero penalty must pin exactly that many
// directions.
template <typename F>
inline void for_each_icar_component(int start, int n, int n_components, F&& comp) {
    const int L = (n_components > 1) ? n_components : 1;
    const int csize = n / L;
    for (int c = 0; c < L; c++) comp(start + c * csize, csize);
}

// (Q(rho) phi)[i] = n_neighbors[i] * phi[i] - rho * sum_{j ~ i} phi[j].
inline double car_apply_row(
    int i,
    const double* phi,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double rho = 1.0
) {
    double neighbor_sum = 0.0;
    const int row_end = adj_row_ptr[i + 1];
    for (int k = adj_row_ptr[i]; k < row_end; k++) {
        neighbor_sum += phi[adj_col_idx[k]];
    }
    return n_neighbors[i] * phi[i] - rho * neighbor_sum;
}

// out[i] = (Q(rho) phi)[i] for i in [0, S).
inline void car_apply(
    const double* phi, double* out, int S,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double rho = 1.0
) {
    for (int i = 0; i < S; i++) {
        out[i] = car_apply_row(i, phi, adj_row_ptr, adj_col_idx, n_neighbors, rho);
    }
}

// phi' Q(rho) phi = sum_i n_neighbors[i] * phi[i]^2 - 2*rho * sum_{i<j~i} phi[i] * phi[j].
inline double car_quad_form(
    const double* phi, int S,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double rho = 1.0
) {
    double quad = 0.0;
    for (int i = 0; i < S; i++) {
        quad += n_neighbors[i] * phi[i] * phi[i];
        const int row_end = adj_row_ptr[i + 1];
        for (int k = adj_row_ptr[i]; k < row_end; k++) {
            int j = adj_col_idx[k];
            if (j > i) {
                quad -= 2.0 * rho * phi[i] * phi[j];
            }
        }
    }
    return quad;
}

} // namespace tulpa

#endif // TULPA_ICAR_KERNEL_H
