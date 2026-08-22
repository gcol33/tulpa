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

// Per-group posterior modes + marginal variances at the optimum (BLUPs), plus
// the mode/theta cross-Hessian (the "Bf" block: -d^2 ell_g/d theta db at the
// mode) when the oracle can supply it. `bcross` is an (ng x nth x d) array
// with `bcross(g, , )` the group's n_theta x d block; `bcross_available` flags
// whether the oracle's theta_score is a genuine implementation (the R-closure
// bridge is not, so bcross is filled with NA rather than a silent 0 there).
//
// `bcov` is the (ng x d x d) array with `bcov(g, , )` the group's FULL joint
// posterior covariance Cov(b_g | y) across every RE term sharing that
// grouping factor -- not just the per-term diagonal `bvar` already returns.
// When a group carries more than one RE term (e.g. an abundance-arm block
// and a detection-arm block on the same species), `m.negH` is already the
// joint (d x d) Hessian of the group's COMBINED mode-finding solve (every
// term's coefficients are found together, one Newton step over the whole
// group vector), so `C = solve(negH)` already carries the cross-term
// covariance -- it was simply discarded before this change (only `C(j, j)`
// reached R). `bvar` is kept unchanged (a block-diagonal caller needs no new
// wiring); `bcov` is the superset a caller doing a joint (theta, b_g) draw
// across correlated terms needs, the same way `bcross`'s un-sliced (nth x d)
// shape already is per group.
//
// `group_ok` is the per-group solve status: FALSE where the mode search or the
// factorization of the group's penalized precision failed, in which case that
// group's `bhat` / `bvar` / `bcov` / `bcross` slots are NA. Both are read
// through the same aghq_group_solve() the AGHQ objective uses, so the two
// cannot disagree about what counts as a failed solve.
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
    // A coordinate whose assembled variance is not positive is carried by the
    // absolute PD backstop, so the covariance reported for it did not come
    // from the parameter. That is reported rather than inherited: it is the
    // one case where the jitter IS the number being read (gcol33/tulpa#595).
    std::vector<char> jitter_floored;
    Eigen::MatrixXd Sig = recov_block_diag_sigma(Ls, d, &jitter_floored);
    const AghqSigmaFactor sf = aghq_sigma_factor(Sig, d);
    if (!sf.ok)
        stop("cpp_aghq_blups: the covariance at the supplied parameter is not "
             "positive definite (its Cholesky failed), so no BLUP can be "
             "extracted.");
    LogicalVector JIT(d);
    for (int i = 0; i < d; ++i) JIT(i) = (jitter_floored[i] != 0);

    orc->rebind(theta.data());
    const bool cross_ok = orc->has_theta_score() && nth > 0;
    NumericMatrix BHAT(ng, d), BVAR(ng, d);
    NumericVector BCROSS((std::size_t)ng * nth * d);
    BCROSS.attr("dim") = IntegerVector::create(ng, nth, d);
    if (!cross_ok) std::fill(BCROSS.begin(), BCROSS.end(), NA_REAL);
    NumericVector BCOV((std::size_t)ng * d * d);
    BCOV.attr("dim") = IntegerVector::create(ng, d, d);
    LogicalVector GOK(ng);
    for (int g = 0; g < ng; ++g) {
        const AghqGroupSolve gs = aghq_group_solve(*orc, g, sf.P, d,
                                                   /*want_chol=*/false);
        GOK(g) = gs.ok;
        // A group whose mode search failed, or whose penalized precision would
        // not factor, has no posterior mode and no covariance. Its slots stay NA
        // so the caller drops it, rather than carrying the numbers a failed
        // decomposition's solve happens to return.
        if (!gs.ok) {
            for (int j = 0; j < d; ++j) {
                BHAT(g, j) = NA_REAL; BVAR(g, j) = NA_REAL;
                for (int k = 0; k < d; ++k)
                    BCOV(g + (std::size_t)ng * (j + (std::size_t)d * k)) = NA_REAL;
            }
            if (cross_ok)
                for (int j = 0; j < nth; ++j)
                    for (int k = 0; k < d; ++k)
                        BCROSS(g + (std::size_t)ng * (j + (std::size_t)nth * k)) = NA_REAL;
            continue;
        }
        for (int j = 0; j < d; ++j) {
            BHAT(g, j) = gs.mode.b(j); BVAR(g, j) = gs.C(j, j);
            for (int k = 0; k < d; ++k)
                BCOV(g + (std::size_t)ng * (j + (std::size_t)d * k)) = gs.C(j, k);
        }
        if (cross_ok) {
            const Eigen::MatrixXd Bf =
                aghq_group_cross_hess(*orc, g, gs.mode.b);   // nth x d
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < d; ++k)
                    BCROSS(g + (std::size_t)ng * (j + (std::size_t)nth * k)) = Bf(j, k);
        }
    }
    return List::create(_["bhat"] = BHAT, _["bvar"] = BVAR, _["bcov"] = BCOV,
                        _["bcross"] = BCROSS, _["bcross_available"] = cross_ok,
                        _["group_ok"] = GOK,
                        _["sigma_jitter_floored"] = JIT);
}
