// aghq_re_core.h
// The shared inner engine: integrate an abstract per-group conditional
// log-likelihood ell_g(b; theta) against N(b; 0, Sigma) by adaptive
// Gauss-Hermite quadrature, and return the ML-II objective sum_g log M_g (+ LKJ)
// together with its ANALYTIC gradient w.r.t. (theta, log-Cholesky Sigma coords).
// C++ port + generalization of the integration core in R/re_aghq.R; driven only
// through REGroupOracle, so every structure (single-arm GLMM, multi-arm
// N-mixture, R-closure bridge) shares this one implementation.
//
// Gradient (Fisher's identity):
//   d log M_g / d theta = sum_k wt_k * (data theta-score at node b_k)
//   d/dSigma sum_g log M_g = 0.5 Q (sum_g R_g - G Sigma) Q,  R_g = sum_k wt_k b_k b_k'
// mapped to log-Cholesky coords by recov_block_grad (re_cov_chol.h). No
// node-placement derivatives (those are O(AGHQ truncation error)).

#ifndef TULPA_AGHQ_RE_CORE_H
#define TULPA_AGHQ_RE_CORE_H

#include "tulpa/aghq_oracle.h"
#include "aghq_re_mode.h"
#include "tulpa/gauss_hermite.h"
#include "re_cov_chol.h"
#include <RcppEigen.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <limits>

namespace tulpa {

using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Tensor quadrature grid (probabilist's GHQ), precomputed once per fit. Each axis
// carries its own node count, so a heterogeneous stack of covariance blocks can
// use fewer nodes on the cheap (scalar nuisance) blocks than on the correlated
// coefficient blocks. `Q` is the node count of axis 0 (equal to every axis in
// the uniform case).
struct AghqGrid {
    int d = 0, Q = 0, n_nodes = 0;
    Eigen::MatrixXd Znodes;   // n_nodes x d (node coordinates)
    Eigen::VectorXd logw;     // n_nodes (sum of per-axis log weights)
    Eigen::VectorXd z2;       // n_nodes (sum of squared node coords)
};

// Per-axis node counts from a per-block request. `nq_block` is either length 1
// (broadcast to every block) or length blocks.size() (one node count per block);
// every axis of block m gets nq_block[m]. This maps a per-covariance-block order
// onto the flat axis layout the tensor grid iterates.
inline std::vector<int> aghq_nq_per_axis(const std::vector<ReCovBlock>& blocks,
                                         const std::vector<int>& nq_block) {
    const bool broadcast = (nq_block.size() == 1);
    std::vector<int> per_axis;
    for (size_t mi = 0; mi < blocks.size(); ++mi) {
        const int q = broadcast ? nq_block[0] : nq_block[mi];
        for (int c = 0; c < blocks[mi].nc; ++c) per_axis.push_back(q);
    }
    return per_axis;
}

// Tensor grid from a per-axis node-count vector (length d). When every axis holds
// the same count this reproduces the uniform Q^d grid byte-for-byte (same node
// order, weights, z2); a scalar request routes here via a length-1 broadcast.
inline AghqGrid aghq_build_grid(const std::vector<int>& nq_per_axis) {
    const int d = (int)nq_per_axis.size();
    std::unordered_map<int, GaussHermite> rules;   // one Golub-Welsch solve per distinct count
    std::vector<const GaussHermite*> gh(d);
    std::vector<Eigen::VectorXd> logwk(d);
    std::vector<int> Qc(d);
    long n = 1;
    for (int c = 0; c < d; ++c) {
        const int nq = nq_per_axis[c];
        auto it = rules.find(nq);
        if (it == rules.end()) it = rules.emplace(nq, gauss_hermite_prob(nq)).first;
        gh[c]    = &it->second;
        logwk[c] = it->second.weights.array().log();
        Qc[c]    = (int)it->second.nodes.size();
        n *= Qc[c];
    }
    AghqGrid g;
    g.d = d; g.Q = (d > 0 ? Qc[0] : 0); g.n_nodes = (int)n;
    g.Znodes.resize(n, d);
    g.logw.resize(n);
    g.z2.resize(n);
    std::vector<int> idx(d, 0);                 // mixed radix, axis 0 fastest (expand.grid order)
    for (long r = 0; r < n; ++r) {
        double lw = 0.0, z2s = 0.0;
        for (int c = 0; c < d; ++c) {
            const double z = gh[c]->nodes(idx[c]);
            g.Znodes(r, c) = z;
            lw += logwk[c](idx[c]);
            z2s += z * z;
        }
        g.logw(r) = lw;
        g.z2(r) = z2s;
        for (int c = 0; c < d; ++c) { if (++idx[c] < Qc[c]) break; idx[c] = 0; }
    }
    return g;
}

struct AghqValueGrad {
    double f = -std::numeric_limits<double>::infinity();  // objective to MAXIMIZE
    Eigen::VectorXd grad;                                 // d f / d[theta, eta]
    bool ok = false;
};

// Objective + analytic gradient at par = [theta (n_theta) ; eta (sum block k)].
// `orc` is rebound to theta internally. Empty groups contribute exactly 0 to
// both objective and gradient (no special-casing needed).
// want_grad = false returns only the objective f (skips the theta-score node
// sweep, the R_g accumulation, and the block-gradient assembly) -- the fast path
// the FD optimizer uses. want_grad = true additionally returns the analytic
// gradient of the TRUE marginal (Fisher identity), which is consistent with the
// computed objective only as n_quad grows (verified: matches FD to ~1e-6 at
// n_quad=9, diverges at n_quad=1 where the Laplace curvature term is missing).
// Used for diagnostics / the future
// AD-exact-gradient optimizer; the A-C optimizer is FD over f.
inline AghqValueGrad aghq_objective_grad(REGroupOracle& orc,
                                         const Eigen::VectorXd& par,
                                         const std::vector<ReCovBlock>& blocks,
                                         const AghqGrid& grid,
                                         double lkj_eta,
                                         bool want_grad = true) {
    const int d   = grid.d;
    const int nth = orc.n_theta;
    const int K   = recov_total_k(blocks);
    AghqValueGrad R;
    R.grad = Eigen::VectorXd::Zero(nth + K);

    const Eigen::VectorXd theta = par.head(nth);
    const Eigen::VectorXd eta   = par.tail(K);

    std::vector<Eigen::MatrixXd> Ls = recov_theta_to_L(eta, blocks);
    Eigen::MatrixXd Sig = recov_block_diag_sigma(Ls, d);
    Eigen::LLT<Eigen::MatrixXd> lltS(Sig);
    if (lltS.info() != Eigen::Success) return R;
    const Eigen::MatrixXd P = lltS.solve(Eigen::MatrixXd::Identity(d, d));
    const double logdetS =
        2.0 * Eigen::MatrixXd(lltS.matrixL()).diagonal().array().log().sum();

    orc.rebind(theta.data());

    const int nN = grid.n_nodes;
    Eigen::MatrixXd sumR = Eigen::MatrixXd::Zero(d, d);
    Eigen::VectorXd gtheta = Eigen::VectorXd::Zero(nth);
    double total = 0.0;

    std::vector<double> hvals(nN), tsc(std::max(nth, 1));
    for (int g = 0; g < orc.n_groups; ++g) {
        GroupMode m = aghq_group_mode(orc, g, P);
        if (!m.ok) return R;
        // C = (-H_g)^{-1}; Lc lower with Lc Lc' = C.
        Eigen::LLT<Eigen::MatrixXd> lltN(m.negH);
        if (lltN.info() != Eigen::Success) return R;
        const Eigen::MatrixXd C = lltN.solve(Eigen::MatrixXd::Identity(d, d));
        Eigen::LLT<Eigen::MatrixXd> lltC(C);
        if (lltC.info() != Eigen::Success) return R;
        const Eigen::MatrixXd Lc = lltC.matrixL();

        // Nodes b_k = b_hat + Lc z_k (row-major for the oracle's node_ll).
        RowMat B = grid.Znodes * Lc.transpose();
        B.rowwise() += m.b.transpose();

        orc.node_ll(g, B.data(), nN, hvals.data());

        // terms_k = logw_k + ell_g(b_k) - 0.5 b_k' P b_k + 0.5 z_k'z_k
        const Eigen::VectorXd quad = (B * P).cwiseProduct(B).rowwise().sum();
        Eigen::VectorXd terms(nN);
        for (int k = 0; k < nN; ++k)
            terms(k) = grid.logw(k) + hvals[k] - 0.5 * quad(k) + 0.5 * grid.z2(k);
        const double mx = terms.maxCoeff();
        const double lse = mx + std::log((terms.array() - mx).exp().sum());
        if (!std::isfinite(lse)) return R;

        total += -0.5 * logdetS + Lc.diagonal().array().log().sum() + lse;

        const Eigen::VectorXd wt = (terms.array() - lse).exp();   // posterior weights, sum 1
        if (want_grad) {
            sumR.noalias() += B.transpose() * wt.asDiagonal() * B;    // R_g
            if (nth > 0) {
                for (int k = 0; k < nN; ++k) {
                    orc.theta_score(g, B.data() + (size_t)k * d, tsc.data());
                    gtheta += wt(k) * Eigen::Map<Eigen::VectorXd>(tsc.data(), nth);
                }
            }
        }
    }

    R.f = total + recov_lkj_penalty(Ls, blocks, lkj_eta);
    R.ok = true;
    if (!want_grad) return R;

    if (nth > 0) R.grad.head(nth) = gtheta;

    // Sigma gradient per block. Sig is block-diagonal so P's diagonal blocks are
    // the per-block inverses; slice rather than re-invert.
    const double G = (double)orc.n_groups;
    int posK = 0, posD = 0;
    for (size_t mi = 0; mi < blocks.size(); ++mi) {
        const ReCovBlock& b = blocks[mi];
        const int nc = b.nc;
        const Eigen::MatrixXd Qm    = P.block(posD, posD, nc, nc);
        const Eigen::MatrixXd Sm    = Sig.block(posD, posD, nc, nc);
        const Eigen::MatrixXd sumRm = sumR.block(posD, posD, nc, nc);
        const Eigen::MatrixXd Smat  = 0.5 * Qm * (sumRm - G * Sm) * Qm;
        recov_block_grad(Smat, Ls[mi], b, lkj_eta, R.grad.data() + nth + posK);
        posD += nc; posK += b.k;
    }
    return R;
}

} // namespace tulpa

#endif // TULPA_AGHQ_RE_CORE_H
