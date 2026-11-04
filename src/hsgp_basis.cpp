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
// Lives in its own translation unit because hmc_hsgp.h pulls in RcppEigen;
// nested_laplace.cpp intentionally includes only the Eigen-free
// hmc_hsgp_kernels.h and must stay that way.

// [[Rcpp::depends(RcppEigen)]]

#include "linalg_fast.h"  // require_coords_2col
#include "hmc_hsgp.h"   // tulpa_hsgp::setup_hsgp_2d, tulpa::HSGPData
#include "hsgp_spectral.h"
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

    tulpa_linalg::require_coords_2col(coords, "cpp_hsgp_basis_2d");

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

namespace {

std::vector<double> hsgp_flat_coords(const Rcpp::NumericMatrix& coords,
                                     const char* what) {
    if (coords.ncol() != 2) Rcpp::stop("%s: coords must have exactly 2 columns.", what);
    tulpa_linalg::require_coords_2col(coords, what);
    const int n = coords.nrow();
    std::vector<double> flat(2 * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        flat[2 * i]     = coords(i, 0);
        flat[2 * i + 1] = coords(i, 1);
    }
    return flat;
}

} // namespace

// The HSGP domain a coordinate set implies: the centre of its bounding box and
// the half-widths L = c * range / 2. Zero-width coordinates use a finite
// auxiliary interval. With expanded=false the last two entries are the raw
// coordinate half-widths. A basis on the expanded domain
// at any other coordinates (cpp_hsgp_basis_2d_domain) is the same basis the
// coordinate set was fitted with.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hsgp_domain_2d(const Rcpp::NumericMatrix& coords, double c,
                                     bool expanded = true) {
    if (coords.nrow() < 1) Rcpp::stop("cpp_hsgp_domain_2d: coords must have at least one row.");
    if (!(c >= 1.0)) Rcpp::stop("cpp_hsgp_domain_2d: the boundary factor c must be >= 1.");
    const std::vector<double> flat = hsgp_flat_coords(coords, "cpp_hsgp_domain_2d");
    double xc, yc, L1, L2, S1, S2;
    tulpa_hsgp::hsgp_center_L_2d(flat, coords.nrow(), c, xc, yc, L1, L2, &S1, &S2);
    return Rcpp::NumericVector::create(xc, yc, expanded ? L1 : S1, expanded ? L2 : S2);
}

// The 2D HSGP basis at `coords` on a GIVEN domain c(x_center, y_center, L1, L2)
// (cpp_hsgp_domain_2d). Returns list(phi_basis = n x m^2, lambda_eig = m^2),
// in the column order of cpp_hsgp_basis_2d.
// [[Rcpp::export]]
Rcpp::List cpp_hsgp_basis_2d_domain(const Rcpp::NumericMatrix& coords, int m,
                                    const Rcpp::NumericVector& domain) {
    if (m < 1) Rcpp::stop("cpp_hsgp_basis_2d_domain: m must be >= 1.");
    if (domain.size() != 4 || !(domain[2] > 0.0) || !(domain[3] > 0.0)) {
        Rcpp::stop("cpp_hsgp_basis_2d_domain: domain must be c(x_center, "
                   "y_center, L1, L2) with positive half-widths.");
    }
    const int n = coords.nrow();
    const std::vector<double> flat = hsgp_flat_coords(coords, "cpp_hsgp_basis_2d_domain");
    tulpa::HSGPData data;
    tulpa_hsgp::hsgp_fill_basis_2d(flat, n, m, domain[0], domain[1], domain[2],
                                   domain[3], /*shared=*/true, data);
    const int M = data.m_total;
    Rcpp::NumericMatrix phi(n, M);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < M; ++j)
            phi(i, j) = data.phi_flat[static_cast<std::size_t>(i) * M + j];
    return Rcpp::List::create(
        Rcpp::_["phi_basis"]  = phi,
        Rcpp::_["lambda_eig"] = Rcpp::NumericVector(data.eigenvalues.begin(),
                                                    data.eigenvalues.end()));
}

// sqrt of the spectral density of `kernel` (0 squared exponential, 1 Matern
// 3/2, 2 Matern 5/2) at each eigenvalue, the per-coefficient loading scale of
// an HSGP latent z ~ N(0, I) at (sigma2, lengthscale):
// f(s) = sum_j phi_j(s) sqrt(S(lambda_j)) z_j.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hsgp_sqrt_spectral(const Rcpp::NumericVector& lambda_eig,
                                           double sigma2, double lengthscale,
                                           int kernel) {
    Rcpp::NumericVector out(lambda_eig.size());
    tulpa::priors::hsgp_sqrt_spectral_2d(
        tulpa::priors::hsgp_kernel_from_code(kernel, "cpp_hsgp_sqrt_spectral"),
        sigma2, lengthscale, lambda_eig.begin(), lambda_eig.size(), out.begin());
    return out;
}
