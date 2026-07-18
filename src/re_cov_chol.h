// re_cov_chol.h
// Multi-block random-effect covariance in log-Cholesky coordinates, C++ port of
// the .re_cov_* packing in R/nested_laplace_re_cov.R. Each RE term is one block,
// either FULL (correlated, Sigma = L L', c(c+1)/2 log-Cholesky params: log(L_ii)
// on the diagonal, raw L_ij below, in column-major lower-triangular order) or
// DIAGONAL (uncorrelated, c log-SD params). This is the hot-path packing the
// AGHQ objective + gradient use; the cold-path R helpers (.re_cov_derived_summary)
// stay in R.
//
// Also provides the analytic Sigma-gradient mapping: given a block's "score
// matrix" S = 0.5 Q (sum_g R_g - G Sigma) Q  (Q = Sigma^{-1}, R_g the AGHQ
// posterior 2nd moment), the gradient of sum_g log M_g w.r.t. the block's
// log-Cholesky coordinates is the Frobenius contraction <2 S L, dL/dcoord>, plus
// the LKJ penalty gradient. Setting S = 0 reproduces the EM stationary point
// Sigma = (1/G) sum_g R_g.

#ifndef TULPA_RE_COV_CHOL_H
#define TULPA_RE_COV_CHOL_H

#include <RcppEigen.h>
#include <vector>
#include <cmath>

namespace tulpa {

struct ReCovBlock {
    int  nc   = 1;       // coefficients in the block
    bool full = false;   // correlated (full Sigma) vs diagonal
    int  k    = 1;       // packed parameter count: full -> nc(nc+1)/2, else nc

    ReCovBlock() = default;
    ReCovBlock(int nc_, bool full_)
        : nc(nc_), full(full_ && nc_ > 1),
          k(full ? nc_ * (nc_ + 1) / 2 : nc_) {}
};

inline int recov_total_k(const std::vector<ReCovBlock>& blocks) {
    int k = 0;
    for (const auto& b : blocks) k += b.k;
    return k;
}
inline int recov_total_d(const std::vector<ReCovBlock>& blocks) {
    int d = 0;
    for (const auto& b : blocks) d += b.nc;
    return d;
}

// One block's k coordinates -> lower Cholesky factor L (nc x nc). Column-major
// lower-triangular order, matching .re_logchol_to_L.
inline Eigen::MatrixXd recov_block_L(const double* th, const ReCovBlock& b) {
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(b.nc, b.nc);
    if (b.full) {
        int idx = 0;
        for (int j = 0; j < b.nc; ++j)
            for (int i = j; i < b.nc; ++i)
                L(i, j) = (i == j) ? std::exp(th[idx++]) : th[idx++];
    } else {
        for (int i = 0; i < b.nc; ++i) L(i, i) = std::exp(th[i]);
    }
    return L;
}

// Stacked theta -> per-block Cholesky factors.
inline std::vector<Eigen::MatrixXd>
recov_theta_to_L(const Eigen::VectorXd& theta,
                 const std::vector<ReCovBlock>& blocks) {
    std::vector<Eigen::MatrixXd> out;
    out.reserve(blocks.size());
    int pos = 0;
    for (const auto& b : blocks) {
        out.push_back(recov_block_L(theta.data() + pos, b));
        pos += b.k;
    }
    return out;
}

// Block-diagonal Sigma (d x d) from per-block factors, with a tiny PD jitter.
inline Eigen::MatrixXd recov_block_diag_sigma(const std::vector<Eigen::MatrixXd>& Ls,
                                              int d, double jitter = 1e-10) {
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(d, d);
    int pos = 0;
    for (const auto& L : Ls) {
        const int nc = (int)L.rows();
        S.block(pos, pos, nc, nc) = L * L.transpose();
        pos += nc;
    }
    S.diagonal().array() += jitter;
    return S;
}

// LKJ penalty value: sum over FULL blocks of (eta-1)(2 sum log L_ii - 2 sum log
// sigma_i), sigma_i = sqrt(Sigma_ii). Matches lkj_logprior in R/re_aghq.R.
inline double recov_lkj_penalty(const std::vector<Eigen::MatrixXd>& Ls,
                                const std::vector<ReCovBlock>& blocks,
                                double lkj_eta) {
    if (lkj_eta == 1.0) return 0.0;
    double s = 0.0;
    for (size_t m = 0; m < blocks.size(); ++m) {
        if (!blocks[m].full) continue;
        const Eigen::MatrixXd& L = Ls[m];
        double sum_logLii = 0.0, sum_logsig = 0.0;
        for (int i = 0; i < L.rows(); ++i) {
            sum_logLii += std::log(L(i, i));
            sum_logsig += 0.5 * std::log(std::max(L.row(i).squaredNorm(), 1e-24));
        }
        s += (lkj_eta - 1.0) * (2.0 * sum_logLii - 2.0 * sum_logsig);
    }
    return s;
}

// Gradient of (sum_g log M_g + LKJ) w.r.t. one block's k log-Cholesky coords.
//   Smat = 0.5 * Q * (sum_g R_g[block] - G * Sigma[block]) * Q   (symmetric)
//   L    = the block's Cholesky factor
// Writes b.k entries into grad_out (column-major lower-tri order).
inline void recov_block_grad(const Eigen::MatrixXd& Smat, const Eigen::MatrixXd& L,
                             const ReCovBlock& b, double lkj_eta, double* grad_out) {
    const Eigen::MatrixXd gradL = 2.0 * Smat * L;   // d(sum log M)/dL
    const bool lkj = (lkj_eta != 1.0) && b.full;    // LKJ only on full blocks
    Eigen::VectorXd Sii;
    if (lkj) {
        Sii.resize(b.nc);
        for (int i = 0; i < b.nc; ++i) Sii(i) = std::max(L.row(i).squaredNorm(), 1e-24);
    }
    if (b.full) {
        int idx = 0;
        for (int j = 0; j < b.nc; ++j)
            for (int i = j; i < b.nc; ++i) {
                double g;
                if (i == j) {
                    const double Lii = L(i, j);
                    g = gradL(i, j) * Lii;                       // chain through log L_ii
                    if (lkj) g += (lkj_eta - 1.0) * (2.0 - 2.0 * Lii * Lii / Sii(i));
                } else {
                    g = gradL(i, j);
                    if (lkj) g += (lkj_eta - 1.0) * (-2.0 * L(i, j) / Sii(i));
                }
                grad_out[idx++] = g;
            }
    } else {
        for (int i = 0; i < b.nc; ++i)
            grad_out[i] = gradL(i, i) * L(i, i);                 // chain through log sigma_i
    }
}

} // namespace tulpa

#endif // TULPA_RE_COV_CHOL_H
