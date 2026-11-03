// varying_coef_draws.cpp
// ----------------------------------------------------------------------------
// The spatially- and temporally-varying coefficient fields of a ModelData
// sampler fit, per posterior draw, at the training design or at new
// coordinates / times.
//
// A draw row is the full parameter vector, so the field the likelihood read at
// that draw is what initialize_generic_state() reconstructs from it: the HSGP
// basis expansion, the non-centered NNGP transform, and the centring that
// removes each block's level from eta. Reading the field from there, rather
// than reshaping draw columns, is what makes the read independent of how a
// backend stored them (a basis coefficient, a whitened z, an uncentred w).
//
// At new coordinates / times the field is drawn from its conditional given the
// draw's own field and hyperparameters:
//
//   * NNGP SVC: the NNGP predictive at the nn nearest fitted locations, under
//     the kernel, nugget and variance floor the density runs (Datta et al.
//     2016, section 2.2).
//   * HSGP SVC: the basis re-evaluated at the new coordinates with the
//     training centring and boundary; the field, and with it the level the
//     centring removed, is a deterministic function of the draw's basis
//     coefficients.
//   * TVC rw1: Brownian bridge between fitted levels, Brownian increments past
//     either end (the random walk is the Brownian motion read at integer
//     positions). rw2 / ar1: the recursion run forward past the last level and
//     backward before the first.
//   * TVC gp: the GP conditional at any time; on the exponential kernel the
//     Markov chain conditions on the bracketing instants only, which is exact.
//
// The stored field is CENTRED (its level aliases the fixed effect and is
// removed from eta). For an intrinsic field (rw1 / rw2) every operation above
// is equivariant in the level, so the centred field is predicted directly. A
// proper NNGP / GP / AR1 field's conditional mean is NOT equivariant -- far from the data it
// reverts to zero, not to the centred field's mean -- so its level is first
// drawn from its exact conditional: the likelihood sees only w_c, and under the
// prior N(0, Q^-1)
//
//     mbar | w_c ~ N(-1'Q w_c / 1'Q 1, 1 / 1'Q 1),
//
// after which the uncentred w = w_c + mbar is predicted and the level removed
// again, so the predicted coefficient is on the same footing as the stored one.
// ----------------------------------------------------------------------------

#include <Rcpp.h>
#include "fixed_effect_prior.h"
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "hmc_sampler.h"           // ModelData / ParamLayout
#include "laplace_spec_fit.h"      // as_offset_vec
#include "log_post_impl.h"         // GenericLogPostState, initialize_generic_state
#include "sampler_model_data.h"    // build_sampler_model_inputs
#include "hmc_hsgp.h"              // hsgp_center_L_2d, hsgp_fill_basis_2d
#include "hmc_svc.h"               // compute_cov, kSvcJitter, kSvcVarFloor
#include "hmc_temporal.h"          // ar1_cross_form
#include "nngp_cond.h"             // cond_weights, nngp_row_neighbours
#include "temporal_gp_kernel.h"    // temporal_cov, temporal_cov_chol, ou_chain
#include "tulpa_priors_svc.h"      // svc_hsgp_term_field
#include "tulpa_priors_tvc.h"      // tvc_ar1_rho_from, tvc_gp_phi_from

namespace tulpa_vc {

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;
using tulpa::TemporalType;
using tulpa::math::safe_exp;

// Nugget the temporal-GP density factorizes its covariance with
// (temporal_cov_chol), carried onto the new instant's own variance so the
// conditional is the one of the joint matrix the density used.
constexpr double kTvcGpNugget = 1e-8;

inline double dist_2d(const std::vector<double>& flat, int a,
                      double x, double y) {
    const double dx = flat[2 * (std::size_t)a] - x;
    const double dy = flat[2 * (std::size_t)a + 1] - y;
    return std::sqrt(dx * dx + dy * dy);
}

inline double dist_2d(const std::vector<double>& flat, int a, int b) {
    return dist_2d(flat, a, flat[2 * (std::size_t)b], flat[2 * (std::size_t)b + 1]);
}

// The level of a proper field from its conditional given the centred values.
inline double draw_level(double one_Q_one, double one_Q_wc) {
    return -one_Q_wc / one_Q_one + R::norm_rand() / std::sqrt(one_Q_one);
}

// 1'Q 1 and 1'Q w for the NNGP precision Q = (I - A)' D^-1 (I - A) the SVC
// density factorizes: one Vecchia factor per location, in the fitted order,
// with the density's own neighbour tables, nugget and floor.
inline void svc_nngp_level_forms(const tulpa::SVCData& sv, const double* w,
                                 double sigma2, double phi,
                                 double& one_Q_one, double& one_Q_w) {
    const int N = sv.n_obs, nn = sv.nn;
    const int n_order = (int)sv.nn_order.size();
    one_Q_one = 0.0;
    one_Q_w = 0.0;
    std::vector<double> C, c_vec, alpha;
    std::vector<int> nb_loc;
    for (int i = 0; i < N; i++) {
        const int loc = sv.nn_order[i];
        const int n_nb = (i == 0) ? 0 : tulpa_nngp::nngp_row_neighbours(
            sv.nn_idx.data() + (std::size_t)i * nn, 1, nn, n_order);
        double r1 = 1.0, rw = w[loc], d = sigma2;
        if (n_nb > 0) {
            nb_loc.resize(n_nb);
            for (int k = 0; k < n_nb; k++) {
                nb_loc[k] = sv.nn_order[sv.nn_idx[(std::size_t)i * nn + k] - 1];
            }
            C.assign((std::size_t)n_nb * n_nb, 0.0);
            c_vec.resize(n_nb);
            for (int a = 0; a < n_nb; a++) {
                c_vec[a] = tulpa_svc::compute_cov(
                    sv.nn_dist[(std::size_t)i * nn + a], sigma2, phi, sv.cov_type);
                for (int b = 0; b < n_nb; b++) {
                    C[(std::size_t)a * n_nb + b] = (a == b) ? sigma2 :
                        tulpa_svc::compute_cov(dist_2d(sv.coords, nb_loc[a], nb_loc[b]),
                                               sigma2, phi, sv.cov_type);
                }
            }
            if (!tulpa_nngp::cond_weights(C, c_vec, n_nb, sigma2,
                                          tulpa_svc::kSvcJitter,
                                          tulpa_svc::kSvcVarFloor,
                                          tulpa_nngp::VarFloor::Blend, alpha, d)) {
                Rcpp::stop("svc prediction: a fitted neighbour covariance is not "
                           "positive definite at this draw.");
            }
            for (int k = 0; k < n_nb; k++) {
                r1 -= alpha[k];
                rw -= alpha[k] * w[nb_loc[k]];
            }
        }
        one_Q_one += r1 * r1 / d;
        one_Q_w   += r1 * rw / d;
    }
}

// Prediction neighbours of each new coordinate among the fitted locations, by
// distance; `exact` is the fitted location a new coordinate sits on, or -1.
struct PredNeighbours {
    std::vector<std::vector<int>> idx;
    std::vector<std::vector<double>> dist;
    std::vector<int> exact;
};

inline PredNeighbours svc_pred_neighbours(const tulpa::SVCData& sv,
                                          const Rcpp::NumericMatrix& nc, int nn) {
    const int N = sv.n_obs, n_new = nc.nrow();
    const int m = std::max(1, std::min(nn, N));
    PredNeighbours out;
    out.idx.resize(n_new);
    out.dist.resize(n_new);
    out.exact.assign(n_new, -1);
    std::vector<std::pair<double, int>> d(N);
    for (int p = 0; p < n_new; p++) {
        for (int i = 0; i < N; i++) d[i] = {dist_2d(sv.coords, i, nc(p, 0), nc(p, 1)), i};
        std::partial_sort(d.begin(), d.begin() + m, d.end());
        if (d[0].first == 0.0) out.exact[p] = d[0].second;
        for (int a = 0; a < m; a++) {
            out.idx[p].push_back(d[a].second);
            out.dist[p].push_back(d[a].first);
        }
    }
    return out;
}

// One draw of the NNGP SVC fields at the new coordinates, written to
// row[j * n_new + p].
inline void svc_nngp_predict_row(const tulpa::SVCData& sv, const PredNeighbours& pn,
                                 const std::vector<double>& w_flat,
                                 const std::vector<double>& sigma2,
                                 const std::vector<double>& phi, double* row) {
    const int N = sv.n_obs, n_new = (int)pn.exact.size();
    std::vector<double> C, c_vec, alpha;
    for (int j = 0; j < sv.n_svc; j++) {
        const double* w = w_flat.data() + (std::size_t)j * N;
        double q = 0.0, b = 0.0;
        svc_nngp_level_forms(sv, w, sigma2[j], phi[j], q, b);
        const double mbar = draw_level(q, b);
        for (int p = 0; p < n_new; p++) {
            double& out = row[(std::size_t)j * n_new + p];
            if (pn.exact[p] >= 0) { out = w[pn.exact[p]]; continue; }
            const int m = (int)pn.idx[p].size();
            C.assign((std::size_t)m * m, 0.0);
            c_vec.resize(m);
            for (int a = 0; a < m; a++) {
                c_vec[a] = tulpa_svc::compute_cov(pn.dist[p][a], sigma2[j], phi[j],
                                                  sv.cov_type);
                for (int bb = 0; bb < m; bb++) {
                    C[(std::size_t)a * m + bb] = (a == bb) ? sigma2[j] :
                        tulpa_svc::compute_cov(
                            dist_2d(sv.coords, pn.idx[p][a], pn.idx[p][bb]),
                            sigma2[j], phi[j], sv.cov_type);
                }
            }
            double v = 0.0;
            if (!tulpa_nngp::cond_weights(C, c_vec, m, sigma2[j],
                                          tulpa_svc::kSvcJitter,
                                          tulpa_svc::kSvcVarFloor,
                                          tulpa_nngp::VarFloor::Blend, alpha, v)) {
                Rcpp::stop("svc prediction: a prediction neighbour covariance is "
                           "not positive definite at this draw.");
            }
            double mean = 0.0;
            for (int a = 0; a < m; a++) mean += alpha[a] * (w[pn.idx[p][a]] + mbar);
            out = mean + std::sqrt(v) * R::norm_rand() - mbar;
        }
    }
}

// The HSGP basis at new coordinates under the TRAINING centring and boundary.
// Outside [-L, L] the Laplacian eigenfunctions are periodic continuations that
// vanish on the boundary, so they approximate no GP there and a coordinate
// beyond it is refused rather than evaluated.
inline tulpa::HSGPData svc_hsgp_basis_at(const ModelData& data,
                                         const Rcpp::NumericMatrix& nc) {
    const auto& sv = data.svc_data;
    double xc, yc, L1, L2;
    tulpa_hsgp::hsgp_center_L_2d(sv.coords, sv.n_obs, data.svc_hsgp_boundary_factor,
                                 xc, yc, L1, L2);
    const int n_new = nc.nrow();
    std::vector<double> flat(2 * (std::size_t)n_new);
    int n_out = 0;
    for (int p = 0; p < n_new; p++) {
        flat[2 * p] = nc(p, 0);
        flat[2 * p + 1] = nc(p, 1);
        if (std::abs(nc(p, 0) - xc) > L1 || std::abs(nc(p, 1) - yc) > L2) n_out++;
    }
    if (n_out > 0) {
        Rcpp::stop("svc prediction: %d of %d new coordinate(s) lie outside the HSGP "
                   "domain (the fitted extent widened by c_boundary = %g), where "
                   "the basis approximates no Gaussian process. Refit with a "
                   "larger `c_boundary` in spatial_svc(), or use approx = \"nngp\".",
                   n_out, n_new, data.svc_hsgp_boundary_factor);
    }
    tulpa::HSGPData basis;
    tulpa_hsgp::hsgp_fill_basis_2d(flat, n_new, data.svc_hsgp_m_per_dim, xc, yc,
                                   L1, L2, /*shared=*/true, basis);
    return basis;
}

// A new time for a TVC block: the fitted level it sits on (`exact`, 0-based) or
// -1, the 0-based grid position for rw1 / rw2 / ar1 (fractional between levels
// for rw1), or the time value on the scale the kernel measures for gp, and the
// block's group.
struct TvcQuery {
    std::vector<int> exact;
    std::vector<double> pos;
    std::vector<int> group;
};

// rw1 / rw2 / ar1: one draw of every query of group g from the path `wc`
// (length T) of one group. Positions past either end are walked in order, so
// queries of one draw lie on one path. `centred` marks a path stored with its
// level removed (a varying coefficient's), whose proper (AR1) level is drawn
// from its conditional before the walk and removed again after; an additive
// field's path is stored as it is. With `noise` false every innovation is
// zero, which is the conditional mean of the draw.
inline void temporal_path_predict(TemporalType st, const double* wc, int T,
                                  double tau, double rho, const TvcQuery& q,
                                  int g, bool centred, bool noise,
                                  double* out_row) {
    const int n_new = (int)q.exact.size();
    const double sd_step = 1.0 / std::sqrt(tau);
    auto z = [noise]() { return noise ? R::norm_rand() : 0.0; };
    double mbar = 0.0;
    std::vector<double> w(wc, wc + T);
    if (centred && st == TemporalType::AR1) {
        std::vector<double> ones(T, 1.0);
        const double one_Q_one = tau * tulpa_temporal::ar1_cross_form(
            ones.data(), ones.data(), T, rho);
        const double one_Q_w = tau * tulpa_temporal::ar1_cross_form(
            ones.data(), wc, T, rho);
        mbar = noise ? draw_level(one_Q_one, one_Q_w) : -one_Q_w / one_Q_one;
        for (double& v : w) v += mbar;
    }

    std::vector<std::pair<double, int>> fwd, bwd;
    for (int p = 0; p < n_new; p++) {
        if (q.group[p] != g) continue;
        if (q.exact[p] >= 0) { out_row[p] = wc[q.exact[p]]; continue; }
        const double s = q.pos[p];
        if (s > T - 1) {
            fwd.push_back({s - (T - 1), p});
        } else if (s < 0) {
            bwd.push_back({-s, p});
        } else {
            // Interior, rw1 only: the Brownian bridge between the two levels.
            const int k = (int)std::floor(s);
            const double u = s - k;
            const double mean = (1.0 - u) * w[k] + u * w[k + 1];
            out_row[p] = mean + std::sqrt(u * (1.0 - u)) * sd_step * z();
        }
    }

    // Walk outward from one end. `x1` is the end level, `x0` the one inside it
    // (read by rw2's second difference).
    auto walk = [&](std::vector<std::pair<double, int>>& qs, double x1, double x0) {
        std::sort(qs.begin(), qs.end());
        double h_done = 0.0;
        for (const auto& hq : qs) {
            const double h = hq.first;
            if (st == TemporalType::RW1) {
                x1 += std::sqrt(h - h_done) * sd_step * z();
                h_done = h;
            } else {
                while (h_done + 0.5 < h) {
                    const double next = (st == TemporalType::RW2)
                        ? 2.0 * x1 - x0 + sd_step * z()
                        : rho * x1 + sd_step * z();
                    x0 = x1;
                    x1 = next;
                    h_done += 1.0;
                }
            }
            out_row[hq.second] = x1 - mbar;
        }
    };
    if (!fwd.empty()) walk(fwd, w[T - 1], T > 1 ? w[T - 2] : w[T - 1]);
    if (!bwd.empty()) walk(bwd, w[0], T > 1 ? w[1] : w[0]);
}

// The kernel of a continuous-time GP path: its fitted instants (on the scale
// the kernel measures) and covariance.
struct TemporalGpKernel {
    const std::vector<double>* tt;
    int T;
    tulpa::TemporalCovType cov_type;
    double nu;
    double period;
};

// gp: one draw of every query of group g from the path `wc` of one group. The
// exponential kernel is Markov, so a new instant conditions on the fitted
// instants bracketing it and the level forms read the chain's tridiagonal
// precision; any other kernel conditions on every fitted instant through the
// dense factor. `centred` and `noise` as in temporal_path_predict().
inline void temporal_gp_path_predict(const TemporalGpKernel& kg,
                                     const double* wc, double sigma2,
                                     double phi, const TvcQuery& q, int g,
                                     bool centred, bool noise,
                                     double* out_row) {
    const int T = kg.T, n_new = (int)q.exact.size();
    const std::vector<double>& tt = *kg.tt;
    const bool markov = tulpa_temporal_gp::cov_is_markov(kg.cov_type, kg.nu);
    auto kern = [&](double d) {
        return tulpa_temporal_gp::temporal_cov(d, sigma2, phi, kg.cov_type, kg.nu,
                                               kg.period);
    };

    double one_Q_one = 0.0, one_Q_w = 0.0;
    std::vector<double> L, Kinv1, Kinvw;
    if (markov) {
        std::vector<double> rho, omr2;
        tulpa_temporal_gp::ou_chain(tt, T, phi, rho, omr2);
        one_Q_one = 1.0 / sigma2;
        one_Q_w = wc[0] / sigma2;
        for (int t = 1; t < T; t++) {
            const double v = sigma2 * omr2[t - 1];
            const double r1 = 1.0 - rho[t - 1];
            one_Q_one += r1 * r1 / v;
            one_Q_w += r1 * (wc[t] - rho[t - 1] * wc[t - 1]) / v;
        }
    } else {
        if (!tulpa_temporal_gp::temporal_cov_chol(tt, T, sigma2, phi, kg.cov_type,
                                                  kg.nu, kg.period, L)) {
            Rcpp::stop("tvc prediction: the fitted GP covariance is not positive "
                       "definite at this draw.");
        }
        std::vector<double> ones(T, 1.0), y, wv(wc, wc + T);
        tulpa_nngp::solve_lower(L, T, ones, y);
        tulpa_nngp::solve_upper(L, T, y, Kinv1);
        for (int t = 0; t < T; t++) {
            one_Q_one += Kinv1[t];
            one_Q_w += Kinv1[t] * wc[t];
        }
    }
    const double mbar = !centred ? 0.0
        : (noise ? draw_level(one_Q_one, one_Q_w) : -one_Q_w / one_Q_one);

    std::vector<double> C, c_vec, alpha, y;
    std::vector<int> nb;
    for (int p = 0; p < n_new; p++) {
        if (q.group[p] != g) continue;
        if (q.exact[p] >= 0) { out_row[p] = wc[q.exact[p]]; continue; }
        const double s = q.pos[p];
        if (markov) {
            nb.clear();
            const int hi = (int)(std::upper_bound(tt.begin(), tt.end(), s) - tt.begin());
            if (hi > 0) nb.push_back(hi - 1);
            if (hi < T) nb.push_back(hi);
        } else {
            nb.resize(T);
            for (int t = 0; t < T; t++) nb[t] = t;
        }
        const int m = (int)nb.size();
        c_vec.resize(m);
        for (int a = 0; a < m; a++) c_vec[a] = kern(std::abs(s - tt[nb[a]]));
        double v = 0.0;
        if (markov) {
            C.assign((std::size_t)m * m, 0.0);
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                    C[(std::size_t)a * m + b] = kern(std::abs(tt[nb[a]] - tt[nb[b]]));
            if (!tulpa_nngp::cond_weights(C, c_vec, m, sigma2,
                                          kTvcGpNugget, 0.0,
                                          tulpa_nngp::VarFloor::Clamp, alpha, v)) {
                Rcpp::stop("tvc prediction: a bracketing GP covariance is not "
                           "positive definite at this draw.");
            }
        } else {
            tulpa_nngp::solve_lower(L, T, c_vec, y);
            tulpa_nngp::solve_upper(L, T, y, alpha);
            double kKk = 0.0;
            for (int t = 0; t < T; t++) kKk += c_vec[t] * alpha[t];
            v = std::max(sigma2 + kTvcGpNugget - kKk, 0.0);
        }
        double mean = 0.0;
        for (int a = 0; a < m; a++) mean += alpha[a] * (wc[nb[a]] + mbar);
        out_row[p] = mean + std::sqrt(v) * (noise ? R::norm_rand() : 0.0) - mbar;
    }
}

} // namespace tulpa_vc

// The varying-coefficient fields at each posterior draw.
//
// With no new coordinates / times, `svc` is S x (n_svc * n_units), term-major,
// the field at every fitted location (NNGP) or row (HSGP), and `tvc` is
// S x (n_groups * n_tvc * n_times) in the block layout the eta assembly reads.
// With `svc_coords_new` (n_new x 2, on the fitted coordinate scale) `svc` is
// S x (n_svc * n_new); with `tvc_new` (list: exact, pos, group, all length
// n_new and 0-based) `tvc` is S x (n_tvc * n_new). `svc_nn` is the number of
// fitted locations an NNGP prediction conditions on.
// [[Rcpp::export]]
Rcpp::List cpp_varying_coef_draws(
    Rcpp::NumericMatrix draws,
    Rcpp::NumericVector y,
    Rcpp::IntegerVector n_trials,
    Rcpp::NumericMatrix X,
    std::string family,
    double phi = 1.0,
    Rcpp::Nullable<Rcpp::NumericVector> sigma_beta = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::List> re_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> spatial_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> temporal_spec = R_NilValue,
    double sigma_re_scale = 2.5,
    double phi2 = NA_REAL,
    Rcpp::Nullable<Rcpp::List> svc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::List> zi_spec = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericMatrix> svc_coords_new = R_NilValue,
    Rcpp::Nullable<Rcpp::List> tvc_new = R_NilValue,
    int svc_nn = 15
) {
    const double sigma_beta_value =
        tulpa::read_sigma_beta(sigma_beta);
    using namespace tulpa_vc;
    const int N = y.size();
    tulpa::SamplerModelInputs in;
    std::vector<double> offset = tulpa::as_offset_vec(offset_nullable, N);
    tulpa::build_sampler_model_inputs(
        in, y, n_trials, X, family, phi, phi2, sigma_beta_value, offset,
        sigma_re_scale, re_spec, spatial_spec, temporal_spec, svc_spec,
        tvc_spec, zi_spec);
    const ModelData& data = in.data;
    const ParamLayout& layout = in.layout;
    const int D = layout.total_params;
    if (draws.ncol() != D) {
        Rcpp::stop("cpp_varying_coef_draws: the draws carry %d columns but the "
                   "model lays out %d parameters.", draws.ncol(), D);
    }
    const int S = draws.nrow();

    const bool has_svc = data.has_svc && layout.has_svc && data.svc_data.n_svc > 0;
    const bool has_tvc = data.has_tvc && layout.has_tvc && data.tvc_data.n_tvc > 0;
    const auto& sv = data.svc_data;
    const auto& tv = data.tvc_data;

    // New SVC coordinates: the NNGP prediction neighbours or the HSGP basis,
    // fixed across draws.
    const bool svc_at_new = has_svc && svc_coords_new.isNotNull();
    Rcpp::NumericMatrix nc;
    PredNeighbours pn;
    tulpa::HSGPData new_basis;
    int n_svc_new = 0;
    if (svc_at_new) {
        nc = Rcpp::NumericMatrix(svc_coords_new.get());
        if (nc.ncol() != 2) Rcpp::stop("svc prediction: coordinates need 2 columns.");
        n_svc_new = nc.nrow();
        if (data.svc_is_hsgp) new_basis = svc_hsgp_basis_at(data, nc);
        else pn = svc_pred_neighbours(sv, nc, svc_nn);
    }

    const bool tvc_at_new = has_tvc && tvc_new.isNotNull();
    TvcQuery tq;
    if (tvc_at_new) {
        Rcpp::List tl(tvc_new.get());
        tq.exact = Rcpp::as<std::vector<int>>(tl["exact"]);
        tq.pos = Rcpp::as<std::vector<double>>(tl["pos"]);
        tq.group = Rcpp::as<std::vector<int>>(tl["group"]);
        if (tq.pos.size() != tq.exact.size() || tq.group.size() != tq.exact.size()) {
            Rcpp::stop("tvc prediction: exact, pos and group must have one entry "
                       "per new row.");
        }
        for (std::size_t p = 0; p < tq.exact.size(); p++) {
            if (tq.group[p] < 0 || tq.group[p] >= tv.n_groups ||
                tq.exact[p] >= tv.n_times) {
                Rcpp::stop("tvc prediction: new row %d addresses no fitted block.",
                           (int)p + 1);
            }
        }
    }

    const int svc_units = !has_svc ? 0 : (svc_at_new ? n_svc_new : sv.n_obs);
    const int n_tvc_new = (int)tq.exact.size();
    const int tvc_cols = !has_tvc ? 0 :
        (tvc_at_new ? tv.n_tvc * n_tvc_new : tv.n_groups * tv.n_tvc * tv.n_times);
    Rcpp::NumericMatrix svc_out(S, has_svc ? sv.n_svc * svc_units : 0);
    Rcpp::NumericMatrix tvc_out(S, tvc_cols);

    std::vector<double> params(D), row_svc, row_tvc;
    std::vector<double> s2(has_svc ? sv.n_svc : 0), ph(has_svc ? sv.n_svc : 0);
    const bool tvc_gp = has_tvc && tv.structure == TemporalType::GP;
    for (int s = 0; s < S; s++) {
        for (int j = 0; j < D; j++) params[j] = draws(s, j);
        tulpa::GenericLogPostState<double> state;
        tulpa::initialize_generic_state(params, data, layout, state);

        if (has_svc) {
            row_svc.assign((std::size_t)sv.n_svc * svc_units, 0.0);
            if (!svc_at_new) {
                if ((int)state.svc_w.size() != sv.n_svc * sv.n_obs) {
                    Rcpp::stop("cpp_varying_coef_draws: the SVC field was not "
                               "reconstructed at draw %d.", s + 1);
                }
                row_svc = state.svc_w;
            } else {
                for (int j = 0; j < sv.n_svc; j++) {
                    s2[j] = safe_exp(params[layout.log_sigma2_svc_start + j]);
                    ph[j] = safe_exp(params[layout.log_phi_svc_start + j]);
                }
                if (data.svc_is_hsgp) {
                    // The basis field is a deterministic function of the draw,
                    // so its level is too: the mean of the uncentred field over
                    // the fitted rows, which the centred field the likelihood
                    // read has removed.
                    const int M = data.svc_hsgp_data.m_total;
                    const auto kernel = static_cast<tulpa::priors::HsgpKernel>(
                        data.svc_hsgp_data.kernel);
                    std::vector<double> f_train(sv.n_obs);
                    for (int j = 0; j < sv.n_svc; j++) {
                        const double* beta_j = &params[layout.svc_w_start + j * M];
                        tulpa::priors::svc_hsgp_term_field(
                            kernel, data.svc_hsgp_data.phi_flat,
                            data.svc_hsgp_data.eigenvalues, sv.n_obs, M,
                            s2[j], ph[j], beta_j, f_train.data());
                        double level = 0.0;
                        for (double v : f_train) level += v;
                        level /= sv.n_obs;
                        double* out_j = row_svc.data() + (std::size_t)j * n_svc_new;
                        tulpa::priors::svc_hsgp_term_field(
                            kernel, new_basis.phi_flat, new_basis.eigenvalues, n_svc_new, M,
                            s2[j], ph[j], beta_j, out_j);
                        for (int p = 0; p < n_svc_new; p++) out_j[p] -= level;
                    }
                } else {
                    svc_nngp_predict_row(sv, pn, state.svc_w, s2, ph, row_svc.data());
                }
            }
            for (std::size_t k = 0; k < row_svc.size(); k++) svc_out(s, k) = row_svc[k];
        }

        if (has_tvc) {
            if (!tvc_at_new) {
                for (int k = 0; k < tvc_cols; k++) tvc_out(s, k) = state.tvc_w[k];
            } else {
                row_tvc.assign((std::size_t)tvc_cols, 0.0);
                std::vector<char> group_used(tv.n_groups, 0);
                for (int g : tq.group) group_used[g] = 1;
                for (int j = 0; j < tv.n_tvc; j++) {
                    double* out_j = row_tvc.data() + (std::size_t)j * n_tvc_new;
                    for (int g = 0; g < tv.n_groups; g++) {
                        if (!group_used[g]) continue;
                        const double* wc = state.tvc_w.data() +
                            tulpa_tvc::tvc_block_offset(tv, g, j);
                        if (tvc_gp) {
                            const double sg2 = safe_exp(
                                params[layout.log_sigma2_tvc_gp_start + j]);
                            const double phj = tulpa::priors::tvc_gp_phi_from(
                                params[layout.logit_phi_tvc_gp_start + j], data);
                            const TemporalGpKernel kg{&tv.time_values, tv.n_times,
                                                      tv.cov_type, tv.nu, tv.period};
                            temporal_gp_path_predict(kg, wc, sg2, phj, tq, g,
                                                     true, true, out_j);
                        } else {
                            const double tau = safe_exp(
                                params[layout.log_tau_tvc_start + j]);
                            const double rho = (tv.structure == TemporalType::AR1)
                                ? tulpa::priors::tvc_ar1_rho_from(
                                      params[layout.logit_rho_tvc_start + j])
                                : 0.0;
                            temporal_path_predict(tv.structure, wc, tv.n_times,
                                                  tau, rho, tq, g, true, true,
                                                  out_j);
                        }
                    }
                }
                for (int k = 0; k < tvc_cols; k++) tvc_out(s, k) = row_tvc[k];
            }
        }
    }

    return Rcpp::List::create(
        Rcpp::_["svc"] = has_svc ? Rcpp::wrap(svc_out) : R_NilValue,
        Rcpp::_["tvc"] = has_tvc ? Rcpp::wrap(tvc_out) : R_NilValue);
}

// An additive temporal field (rw1 / rw2 / ar1 / gp) at new times, per row of
// `field` (R x n_groups * n_times, group-major, the field at its fitted
// instants under one posterior draw or one cell's latent draw), with that
// row's hyperparameters: `tau` / `rho` for a walk, `sigma2` / `phi` for a GP
// (on `time_values` with kernel `cov_type`, `nu`, `period`). `query` is the
// list .temporal_query() builds (exact, pos, group; 0-based). A row's field is
// read as stored -- an additive field is not centred -- and `draw` false gives
// the conditional mean.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_temporal_field_rows(
    Rcpp::NumericMatrix field, int n_times, std::string structure,
    Rcpp::NumericVector tau, Rcpp::NumericVector rho,
    Rcpp::NumericVector sigma2, Rcpp::NumericVector phi,
    Rcpp::NumericVector time_values, int cov_type, double nu, double period,
    Rcpp::List query, bool draw) {
    using namespace tulpa_vc;
    TvcQuery q;
    q.exact = Rcpp::as<std::vector<int>>(query["exact"]);
    q.pos = Rcpp::as<std::vector<double>>(query["pos"]);
    q.group = Rcpp::as<std::vector<int>>(query["group"]);
    const int R = field.nrow(), n_new = (int)q.exact.size();
    if (n_times < 1 || field.ncol() % n_times != 0) {
        Rcpp::stop("cpp_temporal_field_rows: the field's %d columns are not "
                   "whole groups of %d times.", field.ncol(), n_times);
    }
    const int G = field.ncol() / n_times;
    for (int p = 0; p < n_new; p++) {
        if (q.group[p] < 0 || q.group[p] >= G || q.exact[p] >= n_times) {
            Rcpp::stop("cpp_temporal_field_rows: new row %d addresses no "
                       "fitted block.", p + 1);
        }
    }
    const bool gp = (structure == "gp");
    TemporalType st = TemporalType::RW1;
    if (structure == "rw2") st = TemporalType::RW2;
    else if (structure == "ar1") st = TemporalType::AR1;
    else if (!gp && structure != "rw1") {
        Rcpp::stop("cpp_temporal_field_rows: unknown structure '%s'.",
                   structure.c_str());
    }
    const int n_hyper = gp ? sigma2.size() : tau.size();
    if (n_hyper != R || (!gp && st == TemporalType::AR1 && rho.size() != R)) {
        Rcpp::stop("cpp_temporal_field_rows: one hyperparameter value per row "
                   "is needed.");
    }
    std::vector<double> tt(time_values.begin(), time_values.end());
    if (gp && (int)tt.size() != n_times) {
        Rcpp::stop("cpp_temporal_field_rows: %d time values for %d times.",
                   (int)tt.size(), n_times);
    }
    const TemporalGpKernel kg{&tt, n_times,
                              static_cast<tulpa::TemporalCovType>(cov_type),
                              nu, period};
    std::vector<char> group_used(G, 0);
    for (int g : q.group) group_used[g] = 1;
    Rcpp::NumericMatrix out(R, n_new);
    std::vector<double> row(field.ncol()), out_row(n_new);
    for (int r = 0; r < R; r++) {
        for (int k = 0; k < field.ncol(); k++) row[k] = field(r, k);
        std::fill(out_row.begin(), out_row.end(), 0.0);
        for (int g = 0; g < G; g++) {
            if (!group_used[g]) continue;
            const double* wc = row.data() + (std::size_t)g * n_times;
            if (gp) {
                temporal_gp_path_predict(kg, wc, sigma2[r], phi[r], q, g,
                                         false, draw, out_row.data());
            } else {
                temporal_path_predict(st, wc, n_times, tau[r],
                                      st == TemporalType::AR1 ? rho[r] : 0.0,
                                      q, g, false, draw, out_row.data());
            }
        }
        for (int p = 0; p < n_new; p++) out(r, p) = out_row[p];
    }
    return out;
}
