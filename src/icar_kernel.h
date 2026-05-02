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

namespace tulpa {

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
