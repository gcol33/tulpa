// aghq_re_oracles.h
// Concrete REGroupOracle implementations. RClosureOracle bridges arbitrary R
// make_site / make_group closures into the engine (the path that keeps every
// existing tulpa_re_aghq caller working unchanged); the native compiled oracles
// (SingleArmSeparable, NMixCommunity) are added in Phase B.

#ifndef TULPA_AGHQ_RE_ORACLES_H
#define TULPA_AGHQ_RE_ORACLES_H

#include "tulpa/aghq_oracle.h"
#include <Rcpp.h>
#include <cstddef>
#include <vector>

namespace tulpa {

// builder(theta) -> list(grad_hess = function(g, b) -> list(logL, grad, negH),
//                        node_ll   = function(g, B) -> numeric over node rows).
// Group index passed to R is 1-based. theta_score is unused (the A-C optimizer
// is FD over the objective). Not thread-safe: R is single-threaded.
struct RClosureOracle : REGroupOracle {
    Rcpp::Function builder;
    Rcpp::Function gh_fn;     // current bound grad_hess (placeholder until rebind)
    Rcpp::Function nll_fn;    // current bound node_ll

    // Optional PSD mode-find curvature, cached from the most recent grad_hess
    // call (the R oracle may return a `fisher` block alongside `negH`). The
    // mode-find calls grad_hess then newton_hess at the same (g, b), so the
    // cache is valid for that lookup.
    mutable std::vector<double> fisher_buf;
    mutable bool fisher_present = false;

    RClosureOracle(Rcpp::Function builder_, int ng, int d_, int nth)
        : builder(builder_), gh_fn(builder_), nll_fn(builder_) {
        n_groups = ng; d = d_; n_theta = nth;
    }

    void rebind(const double* theta) override {
        Rcpp::NumericVector th(theta, theta + n_theta);
        Rcpp::List o = builder(th);
        gh_fn  = Rcpp::as<Rcpp::Function>(o["grad_hess"]);
        nll_fn = Rcpp::as<Rcpp::Function>(o["node_ll"]);
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

    void theta_score(int, const double*, double*) const override {}  // unused (FD optimizer)
    bool thread_safe() const override { return false; }
};

} // namespace tulpa

#endif // TULPA_AGHQ_RE_ORACLES_H
