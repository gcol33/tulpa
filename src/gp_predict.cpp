// gp_predict.cpp
// The GP / NNGP field of a fitted nested-Laplace model at new coordinates,
// one row per supplied field vector.
//
// Each row is a value of the latent field at the fitted (unique) locations --
// a posterior draw, or a cell's inner-Laplace mode -- together with the outer
// cell its hyperparameters come from. At a new location the field is the NNGP
// predictive given the row's field at that location's nearest fitted
// locations (Datta et al. 2016, section 2.2):
//
//   w* | w ~ N(alpha' w_nb, sigma2 + jitter - c' alpha),
//   alpha = (C + jitter I)^-1 c,
//
// under the covariance kernel (tulpa_svc::compute_cov) and the conditional
// solver (tulpa_nngp::cond_weights) the density uses. New locations are
// conditionally independent given the fitted field, so a row of standard
// normal `noise` turns the conditional means into a joint draw at every new
// location; a zero row returns the conditional means. A new coordinate that
// coincides with a fitted location takes that location's value.
//
// The kriging weights depend on the location and the cell only, so they are
// solved once per (location, cell) and applied to every row of that cell.

// [[Rcpp::depends(RcppEigen)]]

#include "hmc_svc.h"     // tulpa_svc::compute_cov, tulpa::CovType
#include "nngp_cond.h"   // tulpa_nngp::cond_weights, VarFloor
#include "linalg_fast.h" // tulpa_linalg::coords_dist
#include <Rcpp.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

// new_coords    : n_new x d prediction coordinates.
// unique_coords : n_loc x d fitted (unique) location coordinates, in the
//                 column order of `field_rows`. The coordinate dimension is
//                 whatever the fit used.
// field_rows    : R x n_loc field values at the fitted locations.
// cell          : length R, the 0-based outer cell of each row.
// sigma2_cell, phi_cell : the field variance and range of every outer cell.
// nn            : number of nearest neighbours to condition on.
// cov_type      : covariance kernel code (tulpa::CovType).
// noise         : R x n_new standard normal deviates (zeros: the means).
//
// Returns the R x n_new field at the new coordinates.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_gp_field_rows(
    const Rcpp::NumericMatrix& new_coords,
    const Rcpp::NumericMatrix& unique_coords,
    const Rcpp::NumericMatrix& field_rows,
    const Rcpp::IntegerVector& cell,
    const Rcpp::NumericVector& sigma2_cell,
    const Rcpp::NumericVector& phi_cell,
    int nn, int cov_type,
    const Rcpp::NumericMatrix& noise
) {
    if (new_coords.ncol() < 1 || new_coords.ncol() != unique_coords.ncol()) {
        Rcpp::stop("cpp_gp_field_rows: prediction and fitted coordinates "
                   "must have the same number of columns (>= 1).");
    }
    const int n_new = new_coords.nrow();
    const int n_loc = unique_coords.nrow();
    const int R     = field_rows.nrow();
    const int K     = sigma2_cell.size();
    if (field_rows.ncol() != n_loc) {
        Rcpp::stop("cpp_gp_field_rows: field_rows must have nrow(unique_coords) columns.");
    }
    if (cell.size() != R || phi_cell.size() != K) {
        Rcpp::stop("cpp_gp_field_rows: `cell` must have one entry per row and the "
                   "cell vectors one entry per cell.");
    }
    if (noise.nrow() != R || noise.ncol() != n_new) {
        Rcpp::stop("cpp_gp_field_rows: `noise` must be nrow(field_rows) x nrow(new_coords).");
    }
    for (int r = 0; r < R; ++r) {
        if (cell[r] < 0 || cell[r] >= K) {
            Rcpp::stop("cpp_gp_field_rows: cell index %d of row %d is outside [0, %d).",
                       cell[r], r + 1, K);
        }
    }
    const int m = std::min(nn, n_loc);
    if (m < 1) Rcpp::stop("cpp_gp_field_rows: need at least one neighbour.");
    const tulpa::CovType ct = static_cast<tulpa::CovType>(cov_type);
    const double jitter    = 1e-6;
    const double var_floor = 1e-8;

    // Rows grouped by cell, so each (location, cell) pair is solved once.
    std::vector<std::vector<int>> rows_of(K);
    for (int r = 0; r < R; ++r) rows_of[cell[r]].push_back(r);

    Rcpp::NumericMatrix out(R, n_new);
    std::vector<std::pair<double, int>> dists(n_loc);
    std::vector<int> nb(m);
    std::vector<double> C(static_cast<std::size_t>(m) * m), c_vec(m), alpha;

    for (int i = 0; i < n_new; ++i) {
        for (int j = 0; j < n_loc; ++j) {
            dists[j] = std::make_pair(
                tulpa_linalg::coords_dist(new_coords, i, unique_coords, j), j);
        }
        std::partial_sort(dists.begin(), dists.begin() + m, dists.end());
        for (int a = 0; a < m; ++a) nb[a] = dists[a].second;
        if (dists[0].first == 0.0) {
            for (int r = 0; r < R; ++r) out(r, i) = field_rows(r, nb[0]);
            continue;
        }

        for (int k = 0; k < K; ++k) {
            if (rows_of[k].empty()) continue;
            const double s2  = sigma2_cell[k];
            const double phi = phi_cell[k];
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
            }
            double cv = 0.0;
            if (!tulpa_nngp::cond_weights<double>(C, c_vec, m, s2, jitter,
                                                  var_floor,
                                                  tulpa_nngp::VarFloor::Clamp,
                                                  alpha, cv)) {
                Rcpp::stop("cpp_gp_field_rows: the neighbour covariance of new "
                           "location %d is not positive definite at cell %d.",
                           i + 1, k + 1);
            }
            const double sd = std::sqrt(cv);
            for (int r : rows_of[k]) {
                double mean = 0.0;
                for (int a = 0; a < m; ++a) mean += alpha[a] * field_rows(r, nb[a]);
                out(r, i) = mean + sd * noise(r, i);
            }
        }
    }
    return out;
}
