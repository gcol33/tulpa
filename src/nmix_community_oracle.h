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
// per-site accumulation reproduces the whole-species marginal exactly; the
// curvature is the design-sandwiched per-site observed-information block
//   B_i = diag(I^lambda_i, I^p_ij) - Var(N_i|y_i) v_i v_i',
//   v_i = (-score_wt_lambda_i, p_i1, ..., p_iJ)
// (Louis 1982; the abundance/detection coupling). The complete-data Fisher (the
// PSD Newton curvature for the mode-find, where the observed info is indefinite
// away from the mode) is the same sandwich with the diagonal Fisher only.
//
// The substantive method bodies live in nmix_community_oracle.cpp (declared
// here): they instantiate a large amount of Eigen template code, and keeping
// them out-of-line means a TU that only *calls* the oracle (e.g. the EM driver
// nmix_community_em.cpp) does not re-instantiate the per-site assembly inline,
// which overflows the MinGW g++ compiler under -O2.
//
// Poisson OR negative-binomial abundance. NB adds a GLOBAL dispersion size r
// (shared across species, not a per-species RE) carried as the (d+1)-th theta
// entry log_r; n_theta is then d+1 so the engine's theta-gradient picks up the
// log_r row, while the per-species RE dimension stays d. Poisson is the
// r = +Inf limit. The per-site NB marginal / score / dispersion machinery is
// nmix_kernel.h (the single source).

#ifndef TULPA_NMIX_COMMUNITY_ORACLE_H
#define TULPA_NMIX_COMMUNITY_ORACLE_H

#include "tulpa/aghq_oracle.h"
#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <limits>
#include <vector>

namespace tulpa {

struct NMixCommunityOracle : REGroupOracle {
    int p_lam = 0, p_p = 0;
    Eigen::MatrixXd Xlam;                 // n_sites x p_lambda (shared across species)
    Eigen::VectorXd mu;                   // active community means (theta), length d
    bool   is_nb = false;                 // negative-binomial abundance
    double r     = std::numeric_limits<double>::infinity();  // active NB size (Poisson: +Inf)

    // Per species, per site: the cached Poisson marginal (lgamma precompute) and
    // the detection design rows for that site's visits, in input order.
    struct SiteRec {
        int site = 0;                     // 0-based row into Xlam
        NMixSiteCache cache;              // eta-independent lgamma terms
        Eigen::MatrixXd Xp;               // J_i x p_p
    };
    std::vector<std::vector<SiteRec>> sp_sites;   // [n_species][n_sites]

    // Per-species value / score, and optionally the marginal observed info
    // (negH) and / or the PSD complete-data Fisher; one site loop is the single
    // source for grad_hess / newton_hess / theta_score.
    struct SpeciesEval {
        double logL = 0.0;
        Eigen::VectorXd grad;             // d ell_g / db  (== d ell_g / d mu)
        Eigen::MatrixXd negH;             // -d^2 ell_g / db^2 (marginal observed info)
        Eigen::MatrixXd fisher;           // complete-data Fisher (PSD)
        double grad_logr = 0.0;           // d ell_g / d log_r summed over sites (NB only)
    };

    NMixCommunityOracle(const Rcpp::IntegerVector& y,
                        const Rcpp::IntegerVector& site_idx,
                        const Rcpp::IntegerVector& species_idx,
                        const Rcpp::NumericMatrix& X_lambda,
                        const Rcpp::NumericMatrix& X_p,
                        int n_sites, int n_species, int K_max,
                        bool nb = false);

    static inline double clamp30(double e) {
        return e < -30.0 ? -30.0 : (e > 30.0 ? 30.0 : e);
    }

    SpeciesEval eval_species(int g, const double* b,
                             bool want_negH = true, bool want_fisher = true) const;

    void rebind(const double* theta) override {
        for (int i = 0; i < d; ++i) mu(i) = theta[i];
        if (is_nb) r = std::exp(theta[d]);   // theta[d] = log_r (global dispersion)
    }
    void grad_hess(int g, const double* b, double& logL,
                   double* grad, double* negH) const override;
    void node_ll(int g, const double* B, int n_nodes, double* out) const override;
    void theta_score(int g, const double* b, double* dl_dtheta) const override;
    bool newton_hess(int g, const double* b, double* H) const override;
    bool thread_safe() const override { return true; }
};

}  // namespace tulpa

#endif  // TULPA_NMIX_COMMUNITY_ORACLE_H
