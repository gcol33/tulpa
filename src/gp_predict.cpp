// gp_predict.cpp
// R-facing NNGP conditional-mean kriging for a fitted GP / NNGP nested-Laplace
// field. predict.tulpa_fit() calls this to interpolate the posterior-mean field
// to new coordinates.
//
// The field is grid-marginalised: for each hyperparameter grid cell the field
// at the new location is the NNGP conditional mean given the fitted field at
// that location's nearest training locations, cond_mean = c' C^{-1} w_nb, using
// the SAME covariance kernel (tulpa_svc::compute_cov) and the SAME conditional
// solver (tulpa_nngp::cond_moments) the fit used. The per-cell means are then
// weighted by the grid posterior weights (marginalised, not a plug-in at the
// posterior mean). Reuses the single-source covariance + conditional kernels so
// prediction and fitting agree.

// [[Rcpp::depends(RcppEigen)]]

#include "hmc_svc.h"     // tulpa_svc::compute_cov, tulpa::CovType
#include "nngp_cond.h"   // tulpa_nngp::cond_moments, VarFloor
#include "linalg_fast.h" // tulpa_linalg::coords_dist
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

// new_coords    : n_new x d prediction coordinates.
// unique_coords : n_loc x d fitted (unique) location coordinates. The
//                 coordinate dimension is whatever the caller supplied and
//                 must match the fit, which is dimension-general -- a
//                 predictor pinned to two columns would read a different
//                 metric than the fit it predicts from.
// field_grid    : n_grid x n_loc, per-cell posterior-mean field at the unique
//                 locations (in unique_coords row order).
// sigma2_grid, phi_grid, weights : length n_grid hyperparameter grid + weights.
// nn            : number of nearest neighbours to condition on.
// cov_type      : covariance kernel code (tulpa::CovType: 0 exp, 1 matern32,
//                 2 gaussian, 3 spherical).
//
// Returns the length-n_new posterior-mean field.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_gp_field_predict(
    const Rcpp::NumericMatrix& new_coords,
    const Rcpp::NumericMatrix& unique_coords,
    const Rcpp::NumericMatrix& field_grid,
    const Rcpp::NumericVector& sigma2_grid,
    const Rcpp::NumericVector& phi_grid,
    const Rcpp::NumericVector& weights,
    int nn, int cov_type
) {
    if (new_coords.ncol() < 1 || new_coords.ncol() != unique_coords.ncol()) {
        Rcpp::stop("cpp_gp_field_predict: prediction and fitted coordinates "
                   "must have the same number of columns (>= 1).");
    }
    const int n_new  = new_coords.nrow();
    const int n_loc  = unique_coords.nrow();
    const int n_grid = field_grid.nrow();
    if (field_grid.ncol() != n_loc) {
        Rcpp::stop("cpp_gp_field_predict: field_grid cols must equal nrow(unique_coords).");
    }
    if (sigma2_grid.size() != n_grid || phi_grid.size() != n_grid ||
        weights.size() != n_grid) {
        Rcpp::stop("cpp_gp_field_predict: grid vectors must match nrow(field_grid).");
    }
    const int m = std::min(nn, n_loc);
    if (m < 1) Rcpp::stop("cpp_gp_field_predict: need at least one neighbour.");
    const tulpa::CovType ct = static_cast<tulpa::CovType>(cov_type);
    const double jitter    = 1e-6;
    const double var_floor = 1e-8;

    Rcpp::NumericVector out(n_new, 0.0);
    std::vector<std::pair<double, int>> dists(n_loc);
    std::vector<int> nb(m);
    std::vector<double> C(static_cast<std::size_t>(m) * m), c_vec(m), w_nb(m);

    for (int i = 0; i < n_new; ++i) {
        for (int j = 0; j < n_loc; ++j) {
            dists[j] = std::make_pair(
                tulpa_linalg::coords_dist(new_coords, i, unique_coords, j), j);
        }
        std::partial_sort(dists.begin(), dists.begin() + m, dists.end());
        for (int a = 0; a < m; ++a) nb[a] = dists[a].second;

        double acc = 0.0;
        for (int k = 0; k < n_grid; ++k) {
            const double s2  = sigma2_grid[k];
            const double phi = phi_grid[k];
            for (int a = 0; a < m; ++a) {
                for (int b = 0; b < m; ++b) {
                    if (a == b) { C[static_cast<std::size_t>(a) * m + b] = s2; continue; }
                    C[static_cast<std::size_t>(a) * m + b] =
                        tulpa_svc::compute_cov(
                            tulpa_linalg::coords_dist(unique_coords, nb[a],
                                                      unique_coords, nb[b]),
                            s2, phi, ct);
                }
                c_vec[a] = tulpa_svc::compute_cov(dists[a].first, s2, phi, ct);
                w_nb[a]  = field_grid(k, nb[a]);
            }
            double cm, cv;
            const bool ok = tulpa_nngp::cond_moments<double>(
                C, c_vec, w_nb, m, s2, jitter, var_floor,
                tulpa_nngp::VarFloor::Clamp, cm, cv);
            if (ok) acc += weights[k] * cm;
        }
        out[i] = acc;
    }
    return out;
}
