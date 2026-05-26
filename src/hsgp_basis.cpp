// hsgp_basis.cpp
// R-facing builder for the 2D HSGP Laplacian basis.
//
// Thin wrapper over tulpa_hsgp::setup_hsgp_2d (src/hmc_hsgp.h) so the front
// door can assemble the (phi_basis, lambda_eig) pair the nested-Laplace HSGP
// kernel (cpp_nested_laplace_hsgp) consumes, WITHOUT reimplementing the basis
// math in R. setup_hsgp_2d is the single source of truth for the
// eigenfunctions / eigenvalues (Riutort-Mayol et al. 2023); the R test fixture
// make_hsgp_basis_2d in test-nested-laplace-gp.R is an independent oracle that
// must continue to agree with it.
//
// Lives in its own translation unit because hmc_hsgp.h pulls in RcppEigen for
// its matvec helpers; nested_laplace.cpp intentionally includes only the
// Eigen-free hmc_hsgp_kernels.h and must stay that way.

// [[Rcpp::depends(RcppEigen)]]

#include "hmc_hsgp.h"   // tulpa_hsgp::setup_hsgp_2d, tulpa::HSGPData
#include <Rcpp.h>
#include <vector>

// Build the 2D HSGP basis at the supplied coordinates.
//
// coords : n_obs x 2 matrix of (already prepared) coordinates. setup_hsgp_2d
//          recenters and scales internally to [-L, L]; any standardization
//          (e.g. scale_coords) is applied by the caller before this point.
// m      : basis functions per dimension (total M = m * m).
// c      : boundary factor (L = c * range / 2, floored at 0.1).
//
// Returns list(phi_basis = n_obs x M, lambda_eig = length-M eigenvalues),
// ordered identically so column j of phi_basis pairs with lambda_eig[j].
// [[Rcpp::export]]
Rcpp::List cpp_hsgp_basis_2d(const Rcpp::NumericMatrix& coords, int m, double c) {
    const int n_obs = coords.nrow();
    if (coords.ncol() != 2) {
        Rcpp::stop("cpp_hsgp_basis_2d: coords must have exactly 2 columns (2D HSGP).");
    }
    if (n_obs < 1) {
        Rcpp::stop("cpp_hsgp_basis_2d: coords must have at least one row.");
    }
    if (m < 1) {
        Rcpp::stop("cpp_hsgp_basis_2d: m (basis functions per dimension) must be >= 1.");
    }

    // setup_hsgp_2d expects a flattened [x1, y1, x2, y2, ...] coordinate vector.
    std::vector<double> flat(2 * static_cast<std::size_t>(n_obs));
    for (int i = 0; i < n_obs; ++i) {
        flat[2 * i]     = coords(i, 0);
        flat[2 * i + 1] = coords(i, 1);
    }

    tulpa::HSGPData data;
    tulpa_hsgp::setup_hsgp_2d(flat, n_obs, m, c, /*shared=*/true, data);

    const int M = data.m_total;
    Rcpp::NumericMatrix phi_basis(n_obs, M);
    for (int i = 0; i < n_obs; ++i) {
        for (int j = 0; j < M; ++j) {
            // phi_flat is row-major [n_obs x M].
            phi_basis(i, j) = data.phi_flat[static_cast<std::size_t>(i) * M + j];
        }
    }

    Rcpp::NumericVector lambda_eig(M);
    for (int j = 0; j < M; ++j) {
        lambda_eig[j] = data.eigenvalues[j];
    }

    return Rcpp::List::create(
        Rcpp::_["phi_basis"]  = phi_basis,
        Rcpp::_["lambda_eig"] = lambda_eig
    );
}
