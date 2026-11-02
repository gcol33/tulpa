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

// Grid-marginalised HSGP field at new coordinates (predict() kriging).
//
// The HSGP field is f(x) = Phi(x) * (sqrt(S_k) .* beta_k), where beta_k is the
// per-grid-cell latent (N(0, I) prior) at the cell's (sigma2, lengthscale) and
// S_k is the spectral density at that cell. The posterior field marginalises
// over the hyperparameter grid: f(x) = sum_k w_k * Phi(x) * (sqrt(S_k) .* beta_k)
// (weighted, NOT a plug-in at the posterior-mean hyperparameters -- the
// "marginalise derived quantities" rule). The basis at the new coordinates uses
// the TRAINING centering / boundary (derived from coords_train + c), so it is
// consistent with the fitted basis.
//
// coords_train : n_train x 2, the coordinates the field was fit on (fixes the
//                centering + boundary L).
// coords_new   : n_new x 2, prediction coordinates.
// m, c         : HSGP basis functions per dimension and boundary factor.
// beta_grid    : n_grid x M matrix of per-cell latent means (M = m * m).
// sigma2_grid, lengthscale_grid, weights : length n_grid.
//
// Returns the length-n_new posterior-mean field.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hsgp_field_predict(
    const Rcpp::NumericMatrix& coords_train,
    const Rcpp::NumericMatrix& coords_new,
    int m, double c,
    const Rcpp::NumericMatrix& beta_grid,
    const Rcpp::NumericVector& sigma2_grid,
    const Rcpp::NumericVector& lengthscale_grid,
    const Rcpp::NumericVector& weights
) {
    if (coords_train.ncol() != 2 || coords_new.ncol() != 2) {
        Rcpp::stop("cpp_hsgp_field_predict: coords must have exactly 2 columns.");
    }
    const int n_train = coords_train.nrow();
    const int n_new   = coords_new.nrow();
    const int M       = m * m;
    const int n_grid  = beta_grid.nrow();
    if (beta_grid.ncol() != M) {
        Rcpp::stop("cpp_hsgp_field_predict: beta_grid must have m*m columns.");
    }
    if (sigma2_grid.size() != n_grid || lengthscale_grid.size() != n_grid ||
        weights.size() != n_grid) {
        Rcpp::stop("cpp_hsgp_field_predict: grid vectors must match nrow(beta_grid).");
    }

    // Training-consistent centering + boundary from the training extent.
    std::vector<double> flat_train(2 * static_cast<std::size_t>(n_train));
    for (int i = 0; i < n_train; ++i) {
        flat_train[2 * i]     = coords_train(i, 0);
        flat_train[2 * i + 1] = coords_train(i, 1);
    }
    double x_center, y_center, L1, L2;
    tulpa_hsgp::hsgp_center_L_2d(flat_train, n_train, c, x_center, y_center, L1, L2);

    // Basis at the new coordinates under that centering.
    std::vector<double> flat_new(2 * static_cast<std::size_t>(n_new));
    for (int i = 0; i < n_new; ++i) {
        flat_new[2 * i]     = coords_new(i, 0);
        flat_new[2 * i + 1] = coords_new(i, 1);
    }
    tulpa::HSGPData data;
    tulpa_hsgp::hsgp_fill_basis_2d(flat_new, n_new, m, x_center, y_center,
                                   L1, L2, /*shared=*/true, data);

    // Accumulate sum_k w_k * Phi_new * (sqrt(S_k) .* beta_k).
    Rcpp::NumericVector field(n_new, 0.0);
    std::vector<double> scaled_beta(M);
    std::vector<double> f_k;
    for (int k = 0; k < n_grid; ++k) {
        const double s2 = sigma2_grid[k];
        const double ls = lengthscale_grid[k];
        for (int j = 0; j < M; ++j) {
            const double S = tulpa_hsgp::spectral_density_se(data.eigenvalues[j], s2, ls);
            scaled_beta[j] = std::sqrt(S) * beta_grid(k, j);
        }
        f_k.assign(n_new, 0.0);
        for (int i = 0; i < n_new; ++i) {
            const double* phi_row = &data.phi_flat[static_cast<std::size_t>(i) * M];
            double acc = 0.0;
            for (int j = 0; j < M; ++j) acc += phi_row[j] * scaled_beta[j];
            f_k[i] = acc;
        }
        const double w = weights[k];
        for (int i = 0; i < n_new; ++i) field[i] += w * f_k[i];
    }
    return field;
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
// the half-widths L = c * range / 2 (floored at 0.1), the rule setup_hsgp_2d
// applies. Returns c(x_center, y_center, L1, L2). A basis built on this domain
// at any other coordinates (cpp_hsgp_basis_2d_domain) is the same basis the
// coordinate set was fitted with.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hsgp_domain_2d(const Rcpp::NumericMatrix& coords, double c) {
    if (coords.nrow() < 1) Rcpp::stop("cpp_hsgp_domain_2d: coords must have at least one row.");
    if (!(c >= 1.0)) Rcpp::stop("cpp_hsgp_domain_2d: the boundary factor c must be >= 1.");
    const std::vector<double> flat = hsgp_flat_coords(coords, "cpp_hsgp_domain_2d");
    double xc, yc, L1, L2;
    tulpa_hsgp::hsgp_center_L_2d(flat, coords.nrow(), c, xc, yc, L1, L2);
    return Rcpp::NumericVector::create(xc, yc, L1, L2);
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

// sqrt of the squared-exponential spectral density at each eigenvalue, the
// per-coefficient loading scale of an HSGP latent z ~ N(0, I) at
// (sigma2, lengthscale): f(s) = sum_j phi_j(s) sqrt(S(lambda_j)) z_j.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hsgp_sqrt_spectral(const Rcpp::NumericVector& lambda_eig,
                                           double sigma2, double lengthscale) {
    Rcpp::NumericVector out(lambda_eig.size());
    for (int j = 0; j < lambda_eig.size(); ++j) {
        const double S = tulpa_hsgp::spectral_density_se(lambda_eig[j], sigma2,
                                                         lengthscale);
        out[j] = std::sqrt(S > 0.0 ? S : 0.0);
    }
    return out;
}
