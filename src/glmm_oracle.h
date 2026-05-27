// glmm_oracle.h
// Native single-arm GLMM oracle: the per-group conditional log-likelihood
// ell_g(b) = sum_{i in g} log f(eta_i + Z_i b) for a per-row-separable GLMM,
// with eta_i = (X beta)_i + offset_i. One compiled REGroupOracle implementation
// shared by the optimize family and the Gibbs sweep, so the GLMM family density
// (binomial / poisson / gaussian / neg-binomial-2) has a single source of truth.
//
// `offset` is the per-observation contribution of every OTHER random-effect
// block (the cross-block coupling for a multi-term Gibbs sweep); the fixed
// effects X beta are owned by rebind(theta = beta). A single-block model leaves
// the offset at zero.

#ifndef TULPA_GLMM_ORACLE_H
#define TULPA_GLMM_ORACLE_H

#include "tulpa/aghq_oracle.h"
#include <RcppEigen.h>
#include <R.h>
#include <Rmath.h>
#include <cstddef>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>

namespace tulpa {

enum class GLMMFamily { Binomial, Poisson, Gaussian, NegBin };

inline GLMMFamily glmm_family_from_string(const std::string& f) {
    if (f == "binomial")                          return GLMMFamily::Binomial;
    if (f == "poisson")                           return GLMMFamily::Poisson;
    if (f == "gaussian")                          return GLMMFamily::Gaussian;
    if (f == "neg_binomial_2" || f == "negbin")   return GLMMFamily::NegBin;
    throw std::runtime_error("tulpa GLMM oracle: unsupported family '" + f + "'.");
}

// Per-observation conditional log-density and its first two derivatives in the
// linear predictor eta. The full normalizing constants are kept (so `l` is a
// true log-density); only differences / curvature enter the MH ratio and the
// marginal, but keeping them makes the routine self-contained and reusable.
//   phi: gaussian residual VARIANCE (sd = sqrt(phi)); neg-binomial size r.
struct GLMMElt { double l, d1, d2; };

inline GLMMElt glmm_elt(GLMMFamily fam, double eta, double y,
                        double n_trials, double phi) {
    switch (fam) {
    case GLMMFamily::Binomial: {
        const double l1p = (eta > 0.0) ? eta + std::log1p(std::exp(-eta))
                                       : std::log1p(std::exp(eta));
        const double mu  = 1.0 / (1.0 + std::exp(-eta));
        const double l   = y * eta - n_trials * l1p + R::lchoose(n_trials, y);
        return { l, y - n_trials * mu, -n_trials * mu * (1.0 - mu) };
    }
    case GLMMFamily::Poisson: {
        const double lam = std::exp(eta);
        const double l   = y * eta - lam - R::lgammafn(y + 1.0);
        return { l, y - lam, -lam };
    }
    case GLMMFamily::Gaussian: {
        const double r = y - eta;
        const double l = -0.5 * std::log(2.0 * M_PI) - 0.5 * std::log(phi)
                         - 0.5 * r * r / phi;
        return { l, r / phi, -1.0 / phi };
    }
    case GLMMFamily::NegBin: {
        const double rsz = phi;                 // size
        const double mu  = std::exp(eta);
        const double rm  = rsz + mu;
        const double l   = R::lgammafn(y + rsz) - R::lgammafn(rsz)
                           - R::lgammafn(y + 1.0)
                           + rsz * std::log(rsz / rm) + y * std::log(mu / rm);
        const double d1  = y - (rsz + y) * mu / rm;
        const double d2  = -(rsz + y) * rsz * mu / (rm * rm);
        return { l, d1, d2 };
    }
    }
    return { 0.0, 0.0, 0.0 };   // unreachable
}

// ---------------------------------------------------------------------------
// SingleArmGLMMOracle. Owns one RE block's design (Z, group map) plus the
// shared fixed-effect design X; eta_i = (X beta)_i + offset_i + (Z_i b). The
// engine adds the Sigma^{-1} prior curvature, so negH is the data-only observed
// information sum_i (-d2_i) z_i z_i' (PSD for every GLMM family here, so the
// mode-find Newton drives on negH directly -- newton_hess stays the default).
// ---------------------------------------------------------------------------
struct SingleArmGLMMOracle : REGroupOracle {
    GLMMFamily fam;
    double phi;
    Eigen::MatrixXd X;                 // n x p fixed-effect design
    Eigen::MatrixXd Z;                 // n x nc this block's RE design
    Eigen::VectorXd y;                 // n
    Eigen::VectorXd ntrials;           // n (binomial; 1 otherwise)
    std::vector<std::vector<int>> rows_by_g;   // 0-based obs indices per group

    Eigen::VectorXd xb;                // X beta, set by rebind
    Eigen::VectorXd offset;            // other blocks' RE contribution (n), default 0

    SingleArmGLMMOracle(GLMMFamily fam_, double phi_,
                        Eigen::MatrixXd X_, Eigen::MatrixXd Z_,
                        const Eigen::VectorXi& idx /*1-based group per obs*/,
                        int ng, Eigen::VectorXd y_, Eigen::VectorXd ntrials_)
        : fam(fam_), phi(phi_), X(std::move(X_)), Z(std::move(Z_)),
          y(std::move(y_)), ntrials(std::move(ntrials_)) {
        n_groups = ng;
        d        = static_cast<int>(Z.cols());
        n_theta  = static_cast<int>(X.cols());
        const int n = static_cast<int>(X.rows());
        xb     = Eigen::VectorXd::Zero(n);
        offset = Eigen::VectorXd::Zero(n);
        rows_by_g.assign(ng, std::vector<int>());
        for (int i = 0; i < idx.size(); ++i) {
            const int g = idx(i) - 1;
            if (g >= 0 && g < ng) rows_by_g[g].push_back(i);
        }
    }

    void rebind(const double* theta) override {
        Eigen::Map<const Eigen::VectorXd> beta(theta, n_theta);
        xb.noalias() = X * beta;
    }

    // Set the per-observation offset (the other blocks' RE contribution); pass
    // nullptr to clear it.
    void set_offset(const double* off) {
        if (off == nullptr) { offset.setZero(); return; }
        for (int i = 0; i < offset.size(); ++i) offset(i) = off[i];
    }

    // out[i] = Z_i . b_{group(i)} -- this block's contribution to the shared
    // linear predictor (length n). Drives the sweep's cross-block bookkeeping.
    void re_contribution(const Eigen::MatrixXd& B, Eigen::VectorXd& out) const {
        out.setZero(X.rows());
        for (int g = 0; g < n_groups; ++g) {
            const Eigen::RowVectorXd bg = B.row(g);
            for (int i : rows_by_g[g]) out(i) = Z.row(i).dot(bg.transpose());
        }
    }

    void grad_hess(int g, const double* b, double& logL,
                   double* grad, double* negH) const override {
        Eigen::Map<const Eigen::VectorXd> bv(b, d);
        logL = 0.0;
        Eigen::VectorXd gg = Eigen::VectorXd::Zero(d);
        Eigen::MatrixXd HH = Eigen::MatrixXd::Zero(d, d);
        for (int i : rows_by_g[g]) {
            const Eigen::RowVectorXd zr = Z.row(i);
            const double eta = xb(i) + offset(i) + zr.dot(bv);
            const GLMMElt e = glmm_elt(fam, eta, y(i), ntrials(i), phi);
            logL += e.l;
            gg.noalias() += e.d1 * zr.transpose();
            HH.noalias() += (-e.d2) * (zr.transpose() * zr);
        }
        for (int j = 0; j < d; ++j) grad[j] = gg(j);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j) negH[(std::size_t)i * d + j] = HH(i, j);
    }

    void node_ll(int g, const double* B, int nN, double* out) const override {
        for (int k = 0; k < nN; ++k) {
            Eigen::Map<const Eigen::VectorXd> bv(B + (std::size_t)k * d, d);
            double s = 0.0;
            for (int i : rows_by_g[g]) {
                const double eta = xb(i) + offset(i) + Z.row(i).dot(bv);
                s += glmm_elt(fam, eta, y(i), ntrials(i), phi).l;
            }
            out[k] = s;
        }
    }

    void theta_score(int g, const double* b, double* dl) const override {
        Eigen::Map<const Eigen::VectorXd> bv(b, d);
        Eigen::VectorXd s = Eigen::VectorXd::Zero(n_theta);
        for (int i : rows_by_g[g]) {
            const double eta = xb(i) + offset(i) + Z.row(i).dot(bv);
            const double d1 = glmm_elt(fam, eta, y(i), ntrials(i), phi).d1;
            s.noalias() += d1 * X.row(i).transpose();
        }
        for (int j = 0; j < n_theta; ++j) dl[j] = s(j);
    }

    bool thread_safe() const override { return true; }
};

} // namespace tulpa

#endif // TULPA_GLMM_ORACLE_H
