// aghq_re.cpp
// Rcpp entry points for the unified AGHQ random-effect-covariance engine. The R
// optimize-family wrappers (tulpa_re_aghq, agq_fit, tulpa_nmix_laplace_re) build
// an oracle, optimize the compiled objective with stats::optim, and extract
// BLUPs / Sigma / SEs at the optimum. The numeric core lives in the headers
// (aghq_re_core.h); these are thin marshalling shims.

#include "aghq_re_core.h"
#include "aghq_re_oracles.h"
#include "nmix_community_oracle.h"
#include <Rcpp.h>
#include <vector>

using namespace Rcpp;
using namespace tulpa;

static std::vector<ReCovBlock> parse_blocks(const IntegerVector& nc,
                                            const LogicalVector& full, int& d) {
    std::vector<ReCovBlock> blocks;
    d = 0;
    for (int m = 0; m < nc.size(); ++m) {
        blocks.emplace_back((int)nc[m], (bool)full[m]);
        d += (int)nc[m];
    }
    return blocks;
}

// Build an R-closure-backed oracle (the bridge for arbitrary make_site/make_group
// likelihoods). Returns an external pointer the objective / extractor reuse.
// [[Rcpp::export]]
SEXP cpp_aghq_make_rclosure_oracle(Function builder, int n_groups, int d, int n_theta) {
    return XPtr<REGroupOracle>(new RClosureOracle(builder, n_groups, d, n_theta), true);
}

// Native compiled community / multispecies N-mixture oracle (msNMix). Groups are
// species; the per-group RE is (b_lambda, b_p); theta is the community means.
// Returns an external pointer the optimize driver consumes exactly like the
// R-closure bridge, but with no per-group / per-node round trip into R.
// [[Rcpp::export]]
SEXP cpp_nmix_community_oracle(IntegerVector y, IntegerVector site_idx,
                              IntegerVector species_idx, NumericMatrix X_lambda,
                              NumericMatrix X_p, int n_sites, int n_species,
                              int K_max) {
    return XPtr<REGroupOracle>(
        new NMixCommunityOracle(y, site_idx, species_idx, X_lambda, X_p,
                                n_sites, n_species, K_max), true);
}

// AGHQ ML-II objective sum_g log M_g + LKJ at par = [theta ; log-Chol Sigma].
// Returns a large finite penalty on a failed solve so stats::optim rejects it.
// [[Rcpp::export]]
double cpp_aghq_objective(NumericVector par, SEXP oracle, IntegerVector nc,
                          LogicalVector full, int n_quad, double lkj_eta) {
    XPtr<REGroupOracle> orc(oracle);
    int d; std::vector<ReCovBlock> blocks = parse_blocks(nc, full, d);
    AghqGrid grid = aghq_build_grid(d, n_quad);
    Eigen::VectorXd pe(par.size());
    for (int i = 0; i < par.size(); ++i) pe(i) = par(i);
    AghqValueGrad r = aghq_objective_grad(*orc, pe, blocks, grid, lkj_eta, /*want_grad=*/false);
    return r.ok ? r.f : -1e10;
}

// Per-group posterior modes + marginal variances at the optimum (BLUPs).
// [[Rcpp::export]]
List cpp_aghq_blups(NumericVector par, SEXP oracle, IntegerVector nc, LogicalVector full) {
    XPtr<REGroupOracle> orc(oracle);
    int d; std::vector<ReCovBlock> blocks = parse_blocks(nc, full, d);
    const int nth = orc->n_theta, ng = orc->n_groups;
    Eigen::VectorXd pe(par.size());
    for (int i = 0; i < par.size(); ++i) pe(i) = par(i);
    const Eigen::VectorXd theta = pe.head(nth);
    const Eigen::VectorXd eta   = pe.tail(pe.size() - nth);

    std::vector<Eigen::MatrixXd> Ls = recov_theta_to_L(eta, blocks);
    Eigen::MatrixXd Sig = recov_block_diag_sigma(Ls, d);
    Eigen::LLT<Eigen::MatrixXd> lltS(Sig);
    const Eigen::MatrixXd P = lltS.solve(Eigen::MatrixXd::Identity(d, d));

    orc->rebind(theta.data());
    NumericMatrix BHAT(ng, d), BVAR(ng, d);
    for (int g = 0; g < ng; ++g) {
        GroupMode m = aghq_group_mode(*orc, g, P);
        Eigen::LLT<Eigen::MatrixXd> lltN(m.negH);
        const Eigen::MatrixXd C = lltN.solve(Eigen::MatrixXd::Identity(d, d));
        for (int j = 0; j < d; ++j) { BHAT(g, j) = m.b(j); BVAR(g, j) = C(j, j); }
    }
    return List::create(_["bhat"] = BHAT, _["bvar"] = BVAR);
}
