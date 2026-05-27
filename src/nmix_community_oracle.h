// nmix_community_oracle.h
// Native compiled REGroupOracle for the community / multispecies N-mixture
// (spAbundance msNMix): per-species Royle (2004) N-mixture with Gaussian
// community hyperpriors on the per-species abundance / detection coefficients.
// The grouping factor is the species; the per-group RE vector is
// b_s = (b_lambda_s, b_p_s) (dimension d = p_lambda + p_p), entering the
// abundance and detection linear predictors as coef = mu + b_s where mu is the
// fixed community mean (the engine's theta).
//
// This is the compiled counterpart of the R make_group oracle .nmix_re_oracle()
// in R/nmix_laplace_re.R: it computes, per species, the marginal value / score
// / observed-information by summing the per-site N-mixture marginal
// (nmix_kernel.h) over the species' sites and sandwiching the per-site eta-space
// blocks with the abundance / detection designs. Routing tulpa_nmix_laplace_re
// through this native oracle removes the per-group / per-node round trip into R
// the RClosureOracle bridge incurs, so the integration math stays the single
// shared engine (aghq_re_core) but the per-species likelihood is compiled.
//
// The per-site marginal factorizes over sites given the coefficients, so the
// per-site accumulation here reproduces the whole-species marginal exactly; the
// curvature is the design-sandwiched per-site observed-information block
//   B_i = diag(I^lambda_i, I^p_ij) - Var(N_i|y_i) v_i v_i',
//   v_i = (-score_wt_lambda_i, p_i1, ..., p_iJ)
// (Louis 1982; the abundance/detection coupling). The complete-data Fisher (the
// PSD Newton curvature for the mode-find, where the observed info is indefinite
// away from the mode) is the same sandwich with the diagonal Fisher only.
//
// Poisson abundance only (the kernel's NB path is a follow-up: thread an
// r / log_r dispersion through theta and the dispersion coupling).

#ifndef TULPA_NMIX_COMMUNITY_ORACLE_H
#define TULPA_NMIX_COMMUNITY_ORACLE_H

#include "tulpa/aghq_oracle.h"
#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tulpa {

struct NMixCommunityOracle : REGroupOracle {
    int p_lam = 0, p_p = 0;
    Eigen::MatrixXd Xlam;                 // n_sites x p_lambda (shared across species)
    Eigen::VectorXd mu;                   // active community means (theta), length d

    // Per species, per site: the cached Poisson marginal (lgamma precompute) and
    // the detection design rows for that site's visits, in input order.
    struct SiteRec {
        int site = 0;                     // 0-based row into Xlam
        NMixSiteCache cache;              // eta-independent lgamma terms
        Eigen::MatrixXd Xp;               // J_i x p_p
    };
    std::vector<std::vector<SiteRec>> sp_sites;   // [n_species][n_sites]

    NMixCommunityOracle(const Rcpp::IntegerVector& y,
                        const Rcpp::IntegerVector& site_idx,
                        const Rcpp::IntegerVector& species_idx,
                        const Rcpp::NumericMatrix& X_lambda,
                        const Rcpp::NumericMatrix& X_p,
                        int n_sites, int n_species, int K_max) {
        p_lam    = X_lambda.ncol();
        p_p      = X_p.ncol();
        d        = p_lam + p_p;
        n_theta  = d;
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

    void rebind(const double* theta) override {
        for (int i = 0; i < d; ++i) mu(i) = theta[i];
    }

    // Per-species marginal at coef = mu + b: value, b-score, data observed info
    // (negH), and the PSD complete-data Fisher. One site loop is the single
    // source for grad_hess / newton_hess / theta_score (they select fields).
    struct SpeciesEval {
        double logL = 0.0;
        Eigen::VectorXd grad;             // d ell_g / db  (== d ell_g / d theta)
        Eigen::MatrixXd negH;             // -d^2 ell_g / db^2 (marginal observed info)
        Eigen::MatrixXd fisher;           // complete-data Fisher (PSD)
    };

    static inline double clamp30(double e) {
        return e < -30.0 ? -30.0 : (e > 30.0 ? 30.0 : e);
    }

    SpeciesEval eval_species(int g, const double* b) const {
        SpeciesEval e;
        e.grad   = Eigen::VectorXd::Zero(d);
        e.negH   = Eigen::MatrixXd::Zero(d, d);
        e.fisher = Eigen::MatrixXd::Zero(d, d);

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
                compute_nmix_site_cached(rec.cache, eta_p.data(), eta_lam);
            e.logL += res.log_lik;

            // Score: lambda block += Xlam_i * grad_eta_lambda; p block += Xp * grad_eta_p.
            for (int c = 0; c < p_lam; ++c)
                e.grad(c) += Xlam(rec.site, c) * res.grad_eta_lambda;
            for (int j = 0; j < J; ++j)
                for (int c = 0; c < p_p; ++c)
                    e.grad(p_lam + c) += rec.Xp(j, c) * res.grad_eta_p[j];

            // Per-site eta-space blocks (coords: 0 = lambda, 1..J = visits).
            const int dd = 1 + J;
            Eigen::MatrixXd Bobs = Eigen::MatrixXd::Zero(dd, dd);   // marginal observed info
            Eigen::MatrixXd Bfis = Eigen::MatrixXd::Zero(dd, dd);   // complete-data Fisher
            Bobs(0, 0) = res.info_eta_lambda;
            Bfis(0, 0) = res.info_eta_lambda;
            for (int j = 0; j < J; ++j) {
                Bobs(1 + j, 1 + j) = res.info_eta_p[j];
                Bfis(1 + j, 1 + j) = res.info_eta_p[j];
            }
            if (J > 0) {
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

            e.negH.noalias()   += Zi.transpose() * Bobs * Zi;
            e.fisher.noalias() += Zi.transpose() * Bfis * Zi;
        }
        return e;
    }

    void grad_hess(int g, const double* b, double& logL,
                   double* grad, double* negH) const override {
        const SpeciesEval e = eval_species(g, b);
        logL = e.logL;
        for (int i = 0; i < d; ++i) grad[i] = e.grad(i);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) negH[(std::size_t)i * d + j] = e.negH(i, j);
    }

    void node_ll(int g, const double* B, int n_nodes, double* out) const override {
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
                ll += compute_nmix_site_cached(rec.cache, eta_p.data(), eta_lam).log_lik;
            }
            out[k] = ll;
        }
    }

    // theta = mu enters as coef = mu + b, so d ell_g / d theta == d ell_g / db.
    void theta_score(int g, const double* b, double* dl_dtheta) const override {
        const SpeciesEval e = eval_species(g, b);
        for (int i = 0; i < d; ++i) dl_dtheta[i] = e.grad(i);
    }

    // PSD complete-data Fisher for the safeguarded Newton (the marginal observed
    // info `negH` can be indefinite away from the mode for a latent-N marginal).
    bool newton_hess(int g, const double* b, double* H) const override {
        const SpeciesEval e = eval_species(g, b);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) H[(std::size_t)i * d + j] = e.fisher(i, j);
        return true;
    }

    bool thread_safe() const override { return true; }
};

}  // namespace tulpa

#endif  // TULPA_NMIX_COMMUNITY_ORACLE_H
