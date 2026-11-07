// aghq_re_oracles.h
// Concrete REGroupOracle implementations in the engine. RClosureOracle bridges
// arbitrary R make_site / make_group closures into the engine; the native
// compiled SingleArmSeparable oracle backs the built-in single-arm GLMM family.
// Model-specific compiled oracles (e.g. tulpaObs's NMixCommunityOracle) live in
// the consumer package's src/ and reach the engine across the R boundary as an
// XPtr<tulpa::REGroupOracle> -- the AGHQ core drives any subclass through the
// virtual interface in <tulpa/aghq_oracle.h>.

#ifndef TULPA_AGHQ_RE_ORACLES_H
#define TULPA_AGHQ_RE_ORACLES_H

#include "tulpa/aghq_oracle.h"
#include <Rcpp.h>
#include <cstddef>
#include <vector>

namespace tulpa {

// builder(theta) -> list(grad_hess = function(g, b) -> list(logL, grad, negH),
//                        node_ll   = function(g, B) -> numeric over node rows,
//                        joint     = optional function(g, b) -> list(logL, grad,
//                                    negH, theta_grad, theta_info, cross)).
// Group index passed to R is 1-based. `joint` supplies, in one call, the theta
// score d ell_g / d theta, the theta-space observed information
// -d^2 ell_g / d theta^2 and the cross block -d^2 ell_g / d theta db (n_theta x
// d) beside grad_hess's b-space value / score / information; with it the bridge
// has a genuine theta_score, and tulpa_re_em(theta_update = "latent") runs its
// joint Newton on one R call per group. Not thread-safe: R is single-threaded.
struct RClosureOracle : REGroupOracle {
    Rcpp::Function builder;
    Rcpp::Function gh_fn;     // current bound grad_hess (placeholder until rebind)
    Rcpp::Function nll_fn;    // current bound node_ll
    Rcpp::Function joint_fn;  // current bound joint (valid when has_joint)
    bool has_joint = false;

    // Optional PSD mode-find curvature, cached from the most recent grad_hess
    // call (the R oracle may return a `fisher` block alongside `negH`). The
    // mode-find calls grad_hess then newton_hess at the same (g, b), so the
    // cache is valid for that lookup.
    mutable std::vector<double> fisher_buf;
    mutable bool fisher_present = false;

    RClosureOracle(Rcpp::Function builder_, int ng, int d_, int nth)
        : builder(builder_), gh_fn(builder_), nll_fn(builder_), joint_fn(builder_) {
        n_groups = ng; d = d_; n_theta = nth;
    }

    void rebind(const double* theta) override {
        Rcpp::NumericVector th(theta, theta + n_theta);
        Rcpp::List o = builder(th);
        gh_fn  = Rcpp::as<Rcpp::Function>(o["grad_hess"]);
        nll_fn = Rcpp::as<Rcpp::Function>(o["node_ll"]);
        has_joint = o.containsElementNamed("joint") && !Rf_isNull(o["joint"]);
        if (has_joint) joint_fn = Rcpp::as<Rcpp::Function>(o["joint"]);
    }

    // The `joint` closure at (g, b): b-space value / score / information plus the
    // theta score, theta information and theta-b cross block (all data-only).
    void joint(int g, const double* b, double& logL, double* grad, double* negH,
               double* tgrad, double* tinfo, double* cross) const {
        Rcpp::NumericVector bb(b, b + d);
        Rcpp::List r = joint_fn(g + 1, bb);
        logL = Rcpp::as<double>(r["logL"]);
        Rcpp::NumericVector gv = r["grad"], tg = r["theta_grad"];
        Rcpp::NumericMatrix H = r["negH"], A = r["theta_info"], X = r["cross"];
        for (int i = 0; i < d; ++i) grad[i] = gv[i];
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) negH[(std::size_t)i * d + j] = H(i, j);
        for (int i = 0; i < n_theta; ++i) tgrad[i] = tg[i];
        for (int i = 0; i < n_theta; ++i)
            for (int j = 0; j < n_theta; ++j) tinfo[(std::size_t)i * n_theta + j] = A(i, j);
        for (int i = 0; i < n_theta; ++i)
            for (int j = 0; j < d; ++j) cross[(std::size_t)j * n_theta + i] = X(i, j);
    }

    void grad_hess(int g, const double* b, double& logL,
                   double* grad, double* negH) const override {
        Rcpp::NumericVector bb(b, b + d);
        Rcpp::List r = gh_fn(g + 1, bb);
        logL = Rcpp::as<double>(r["logL"]);
        Rcpp::NumericVector gv = r["grad"];
        for (int i = 0; i < d; ++i) grad[i] = gv[i];
        Rcpp::NumericMatrix H = r["negH"];
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) negH[(std::size_t)i * d + j] = H(i, j);
        if (r.containsElementNamed("fisher")) {
            Rcpp::NumericMatrix F = r["fisher"];
            if (fisher_buf.size() != (std::size_t)d * d)
                fisher_buf.assign((std::size_t)d * d, 0.0);
            for (int i = 0; i < d; ++i)
                for (int j = 0; j < d; ++j) fisher_buf[(std::size_t)i * d + j] = F(i, j);
            fisher_present = true;
        } else {
            fisher_present = false;
        }
    }

    void node_ll(int g, const double* B, int nN, double* out) const override {
        Rcpp::NumericMatrix Bm(nN, d);
        for (int k = 0; k < nN; ++k)
            for (int j = 0; j < d; ++j) Bm(k, j) = B[(std::size_t)k * d + j];
        Rcpp::NumericVector v = nll_fn(g + 1, Bm);
        for (int k = 0; k < nN; ++k) out[k] = v[k];
    }

    bool newton_hess(int, const double*, double* H) const override {
        if (!fisher_present) return false;
        for (std::size_t k = 0; k < fisher_buf.size(); ++k) H[k] = fisher_buf[k];
        return true;
    }

    void theta_score(int g, const double* b, double* dl) const override {
        if (!has_joint) return;
        double l = 0.0;
        std::vector<double> gr(d), nh((std::size_t)d * d), A((std::size_t)n_theta * n_theta),
            X((std::size_t)n_theta * d);
        joint(g, b, l, gr.data(), nh.data(), dl, A.data(), X.data());
    }
    bool has_theta_score() const override { return has_joint; }
    bool thread_safe() const override { return false; }
};

} // namespace tulpa

#endif // TULPA_AGHQ_RE_ORACLES_H
