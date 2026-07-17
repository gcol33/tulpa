// aghq_re.cpp
// Rcpp entry points for the unified AGHQ random-effect-covariance engine. The R
// optimize-family wrappers (tulpa_re_aghq, agq_fit) build an oracle, optimize
// the compiled objective with stats::optim, and extract BLUPs / Sigma / SEs at
// the optimum. The numeric core lives in the headers (aghq_re_core.h); these
// are thin marshalling shims.
//
// Consumer-package oracles (e.g. tulpaObs's NMixCommunityOracle) construct
// their XPtr<tulpa::REGroupOracle> in their own src/ and pass it across the R
// boundary into tulpa_re_aghq() -- the engine drives any REGroupOracle through
// the virtual interface declared in <tulpa/aghq_oracle.h>.

#include "aghq_re_core.h"
#include "aghq_re_oracles.h"
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

// AGHQ ML-II objective sum_g log M_g + LKJ at par = [theta ; log-Chol Sigma].
// Returns a large finite penalty on a failed solve so stats::optim rejects it.
// `n_quad` is per covariance block (length 1 broadcasts to every block, length
// nc.size() gives each block its own node count); see aghq_nq_per_axis.
// [[Rcpp::export]]
double cpp_aghq_objective(NumericVector par, SEXP oracle, IntegerVector nc,
                          LogicalVector full, IntegerVector n_quad, double lkj_eta) {
    XPtr<REGroupOracle> orc(oracle);
    int d; std::vector<ReCovBlock> blocks = parse_blocks(nc, full, d);
    std::vector<int> nqb(n_quad.begin(), n_quad.end());
    AghqGrid grid = aghq_build_grid(aghq_nq_per_axis(blocks, nqb));
    Eigen::VectorXd pe(par.size());
    for (int i = 0; i < par.size(); ++i) pe(i) = par(i);
    AghqValueGrad r = aghq_objective_grad(*orc, pe, blocks, grid, lkj_eta, /*want_grad=*/false);
    return r.ok ? r.f : -1e10;
}

// AGHQ objective AND its analytic gradient w.r.t. par = [theta ; log-Chol Sigma]
// in one group sweep. The gradient is the Fisher-identity gradient of the TRUE
// marginal (theta: posterior-weighted theta-score; Sigma: the moment-matching
// residual mapped to log-Cholesky coords); it omits the node-placement
// derivatives, which are O(AGHQ truncation), so it agrees with the finite
// difference of cpp_aghq_objective only as n_quad grows -- matching to ~1e-6 by
// n_quad = 9 and diverging at n_quad = 1 (the pure-Laplace curvature term). The
// analytic-gradient optimize path (n_quad > 1) consumes this; the FD path uses
// cpp_aghq_objective. `ok = FALSE` flags a failed solve; `grad` is then zeroed.
// [[Rcpp::export]]
List cpp_aghq_objective_grad(NumericVector par, SEXP oracle, IntegerVector nc,
                             LogicalVector full, IntegerVector n_quad, double lkj_eta) {
    XPtr<REGroupOracle> orc(oracle);
    int d; std::vector<ReCovBlock> blocks = parse_blocks(nc, full, d);
    std::vector<int> nqb(n_quad.begin(), n_quad.end());
    AghqGrid grid = aghq_build_grid(aghq_nq_per_axis(blocks, nqb));
    Eigen::VectorXd pe(par.size());
    for (int i = 0; i < par.size(); ++i) pe(i) = par(i);
    AghqValueGrad r = aghq_objective_grad(*orc, pe, blocks, grid, lkj_eta, /*want_grad=*/true);
    NumericVector grad(r.grad.size());
    for (int i = 0; i < (int)r.grad.size(); ++i) grad(i) = r.grad(i);
    return List::create(_["f"]    = r.ok ? r.f : -1e10,
                        _["grad"] = grad,
                        _["ok"]   = r.ok);
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
