// nmix_community_oracle.cpp
// Out-of-line bodies for NMixCommunityOracle (declared in nmix_community_oracle.h).
// Kept out of the header so a translation unit that only calls the oracle (the EM
// driver nmix_community_em.cpp, and the engine shim aghq_re.cpp) does not
// re-instantiate the per-site Eigen assembly inline -- which overflows MinGW g++
// under -O2. The per-site marginal math is nmix_kernel.h (the single source).

#include "nmix_community_oracle.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tulpa {

NMixCommunityOracle::NMixCommunityOracle(const Rcpp::IntegerVector& y,
                                         const Rcpp::IntegerVector& site_idx,
                                         const Rcpp::IntegerVector& species_idx,
                                         const Rcpp::NumericMatrix& X_lambda,
                                         const Rcpp::NumericMatrix& X_p,
                                         int n_sites, int n_species, int K_max,
                                         bool nb) {
    p_lam    = X_lambda.ncol();
    p_p      = X_p.ncol();
    d        = p_lam + p_p;
    is_nb    = nb;
    n_theta  = d + (nb ? 1 : 0);     // NB carries log_r as the (d+1)-th theta entry
    n_groups = n_species;
    mu       = Eigen::VectorXd::Zero(d);

    Xlam.resize(n_sites, p_lam);
    for (int i = 0; i < n_sites; ++i)
        for (int c = 0; c < p_lam; ++c) Xlam(i, c) = X_lambda(i, c);

    // Group the long-form rows by (species, site), preserving input order.
    const int n_obs = y.size();
    std::vector<std::vector<std::vector<int>>> rows(
        n_species, std::vector<std::vector<int>>(n_sites));
    for (int r = 0; r < n_obs; ++r)
        rows[species_idx[r] - 1][site_idx[r] - 1].push_back(r);

    sp_sites.assign(n_species, std::vector<SiteRec>());
    for (int s = 0; s < n_species; ++s) {
        sp_sites[s].reserve(n_sites);
        for (int i = 0; i < n_sites; ++i) {
            SiteRec rec;
            rec.site = i;
            const std::vector<int>& rr = rows[s][i];
            const int J = (int)rr.size();
            std::vector<int> yv(J);
            rec.Xp.resize(J, p_p);
            for (int j = 0; j < J; ++j) {
                yv[j] = y[rr[j]];
                for (int c = 0; c < p_p; ++c) rec.Xp(j, c) = X_p(rr[j], c);
            }
            rec.cache = nmix_precompute_site(yv.data(), J, K_max);
            sp_sites[s].push_back(std::move(rec));
        }
    }
}

NMixCommunityOracle::SpeciesEval
NMixCommunityOracle::eval_species(int g, const double* b,
                                  bool want_negH, bool want_fisher) const {
    SpeciesEval e;
    e.grad = Eigen::VectorXd::Zero(d);
    if (want_negH)   e.negH   = Eigen::MatrixXd::Zero(d, d);
    if (want_fisher) e.fisher = Eigen::MatrixXd::Zero(d, d);

    Eigen::VectorXd coef(d);
    for (int i = 0; i < d; ++i) coef(i) = mu(i) + b[i];

    std::vector<double> eta_p;
    for (const SiteRec& rec : sp_sites[g]) {
        const int J = rec.cache.n_visits;
        double eta_lam = 0.0;
        for (int c = 0; c < p_lam; ++c) eta_lam += Xlam(rec.site, c) * coef(c);
        eta_lam = clamp30(eta_lam);

        eta_p.assign(J, 0.0);
        for (int j = 0; j < J; ++j) {
            double v = 0.0;
            for (int c = 0; c < p_p; ++c) v += rec.Xp(j, c) * coef(p_lam + c);
            eta_p[j] = clamp30(v);
        }

        const NMixSiteResult res =
            compute_nmix_site_cached(rec.cache, eta_p.data(), eta_lam, r);
        e.logL += res.log_lik;

        // Score: lambda block += Xlam_i * grad_eta_lambda; p block += Xp * grad_eta_p.
        for (int c = 0; c < p_lam; ++c)
            e.grad(c) += Xlam(rec.site, c) * res.grad_eta_lambda;
        for (int j = 0; j < J; ++j)
            for (int c = 0; c < p_p; ++c)
                e.grad(p_lam + c) += rec.Xp(j, c) * res.grad_eta_p[j];
        // Dispersion: log_r enters every site directly (design = 1), so its score
        // is the per-site d ell/d log_r summed over the species' sites.
        if (is_nb) e.grad_logr += res.grad_theta;

        if (!want_negH && !want_fisher) continue;

        // Per-site eta-space blocks (coords: 0 = lambda, 1..J = visits).
        const int dd = 1 + J;
        Eigen::MatrixXd Bobs, Bfis;
        if (want_negH)   { Bobs = Eigen::MatrixXd::Zero(dd, dd); Bobs(0, 0) = res.info_eta_lambda; }
        if (want_fisher) { Bfis = Eigen::MatrixXd::Zero(dd, dd); Bfis(0, 0) = res.info_eta_lambda; }
        for (int j = 0; j < J; ++j) {
            if (want_negH)   Bobs(1 + j, 1 + j) = res.info_eta_p[j];
            if (want_fisher) Bfis(1 + j, 1 + j) = res.info_eta_p[j];
        }
        if (want_negH && J > 0) {
            Eigen::VectorXd vv(dd);
            vv(0) = -res.score_wt_lambda;
            for (int j = 0; j < J; ++j) {
                const double pj = (eta_p[j] > 0.0)
                    ? 1.0 / (1.0 + std::exp(-eta_p[j]))
                    : std::exp(eta_p[j]) / (1.0 + std::exp(eta_p[j]));
                vv(1 + j) = pj;
            }
            Bobs.noalias() -= res.var_N * (vv * vv.transpose());
        }

        // Design map Z_i: eta coord 0 -> Xlam_i over the lambda coefs,
        // eta coord 1+j -> Xp row j over the p coefs.
        Eigen::MatrixXd Zi = Eigen::MatrixXd::Zero(dd, d);
        for (int c = 0; c < p_lam; ++c) Zi(0, c) = Xlam(rec.site, c);
        for (int j = 0; j < J; ++j)
            for (int c = 0; c < p_p; ++c) Zi(1 + j, p_lam + c) = rec.Xp(j, c);

        const Eigen::MatrixXd Zt = Zi.transpose();
        if (want_negH) {
            const Eigen::MatrixXd BZ = Bobs * Zi;
            e.negH.noalias() += Zt * BZ;
        }
        if (want_fisher) {
            const Eigen::MatrixXd BZ = Bfis * Zi;
            e.fisher.noalias() += Zt * BZ;
        }
    }
    return e;
}

void NMixCommunityOracle::grad_hess(int g, const double* b, double& logL,
                                    double* grad, double* negH) const {
    const SpeciesEval e = eval_species(g, b, /*want_negH=*/true, /*want_fisher=*/false);
    logL = e.logL;
    for (int i = 0; i < d; ++i) grad[i] = e.grad(i);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) negH[(std::size_t)i * d + j] = e.negH(i, j);
}

void NMixCommunityOracle::node_ll(int g, const double* B, int n_nodes,
                                  double* out) const {
    Eigen::VectorXd coef(d);
    std::vector<double> eta_p;
    for (int k = 0; k < n_nodes; ++k) {
        const double* bk = B + (std::size_t)k * d;
        for (int i = 0; i < d; ++i) coef(i) = mu(i) + bk[i];
        double ll = 0.0;
        for (const SiteRec& rec : sp_sites[g]) {
            const int J = rec.cache.n_visits;
            double eta_lam = 0.0;
            for (int c = 0; c < p_lam; ++c) eta_lam += Xlam(rec.site, c) * coef(c);
            eta_lam = clamp30(eta_lam);
            eta_p.assign(J, 0.0);
            for (int j = 0; j < J; ++j) {
                double v = 0.0;
                for (int c = 0; c < p_p; ++c) v += rec.Xp(j, c) * coef(p_lam + c);
                eta_p[j] = clamp30(v);
            }
            ll += compute_nmix_site_cached(rec.cache, eta_p.data(), eta_lam, r).log_lik;
        }
        out[k] = ll;
    }
}

void NMixCommunityOracle::theta_score(int g, const double* b,
                                      double* dl_dtheta) const {
    const SpeciesEval e = eval_species(g, b, /*want_negH=*/false, /*want_fisher=*/false);
    for (int i = 0; i < d; ++i) dl_dtheta[i] = e.grad(i);
    if (is_nb) dl_dtheta[d] = e.grad_logr;   // d ell_g / d log_r (global dispersion)
}

bool NMixCommunityOracle::newton_hess(int g, const double* b, double* H) const {
    const SpeciesEval e = eval_species(g, b, /*want_negH=*/false, /*want_fisher=*/true);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j) H[(std::size_t)i * d + j] = e.fisher(i, j);
    return true;
}

}  // namespace tulpa
