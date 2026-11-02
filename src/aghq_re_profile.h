// aghq_re_profile.h
// The fixed-effect profile of the AGHQ marginal at one random-effect covariance:
// the inner solve tulpa_re_cov_nested() runs at every integration node, every
// outer optimizer step and every outer Pareto-k proposal draw when n_quad > 1.
//
// At a fixed covariance point eta (log-Cholesky coordinates) the fixed effects
// beta are integrated by a Laplace step over
//
//   F(beta) = sum_g log M_g(beta, eta) + log N(beta; m, s^2),
//
// the AGHQ group marginals (aghq_objective_grad) plus the Gaussian fixed-effect
// prior when one is set. beta_hat maximizes F by Newton steps on the analytic
// beta-gradient (the Fisher-identity score of aghq_re_core.h); the curvature H
// is a central difference of that gradient, symmetrised -- the same Hessian
// stats::optimHess() forms from a gradient function -- and the integrated value
// is
//
//   log p(y | eta) = F(beta_hat) + (p / 2) log(2 pi) - (1 / 2) log |H|.
//
// The analytic gradient omits the node-placement derivatives (O(AGHQ truncation),
// see aghq_re_core.h), so near the maximum its Newton direction and F can
// disagree. A step is taken only where it raises F; the loop stops when the
// predicted gain g' H^-1 g / 2 or the realized one falls below F's relative
// tolerance (1e-10), or when no step along the direction raises F -- the point
// where the two disagree, within that truncation of F's maximum.

#ifndef TULPA_AGHQ_RE_PROFILE_H
#define TULPA_AGHQ_RE_PROFILE_H

#include "aghq_re_core.h"
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tulpa {

struct AghqProfile {
    Eigen::VectorXd beta;
    Eigen::MatrixXd H;
    double log_marginal = -std::numeric_limits<double>::infinity();
    bool ok = false;
};

// Gaussian fixed-effect prior; `on = false` for none (F is the marginal alone).
struct AghqBetaPrior {
    bool on = false;
    Eigen::VectorXd mean, sd;
};

inline AghqProfile aghq_profile_beta(REGroupOracle& orc,
                                     const Eigen::VectorXd& eta,
                                     const std::vector<ReCovBlock>& blocks,
                                     const AghqGrid& grid,
                                     const Eigen::VectorXd& beta0,
                                     const AghqBetaPrior& prior,
                                     int max_iter, double hess_step) {
    const int p = static_cast<int>(beta0.size());
    const int K = static_cast<int>(eta.size());
    const double LOG_2PI = std::log(2.0 * M_PI);
    AghqProfile out;

    struct Eval { double F = 0.0; Eigen::VectorXd g; bool ok = false; };
    Eigen::VectorXd par(p + K);
    par.tail(K) = eta;
    auto eval = [&](const Eigen::VectorXd& b) {
        Eval e;
        par.head(p) = b;
        const AghqValueGrad r = aghq_objective_grad(orc, par, blocks, grid,
                                                    /*lkj_eta=*/1.0,
                                                    /*want_grad=*/true);
        if (!r.ok || !std::isfinite(r.f)) return e;
        e.F = r.f;
        e.g = r.grad.head(p);
        if (prior.on) {
            for (int j = 0; j < p; ++j) {
                const double s = prior.sd(j), z = (b(j) - prior.mean(j)) / s;
                e.F += -0.5 * z * z - std::log(s) - 0.5 * LOG_2PI;
                e.g(j) -= z / s;
            }
        }
        e.ok = std::isfinite(e.F) && e.g.allFinite();
        return e;
    };
    // H = -(dg / dbeta), central differences, symmetrised.
    auto curvature = [&](const Eigen::VectorXd& b, Eigen::MatrixXd& H) {
        H.resize(p, p);
        for (int j = 0; j < p; ++j) {
            Eigen::VectorXd bp = b, bm = b;
            bp(j) += hess_step;
            bm(j) -= hess_step;
            const Eval ep = eval(bp), em = eval(bm);
            if (!ep.ok || !em.ok) return false;
            H.col(j) = -(ep.g - em.g) / (2.0 * hess_step);
        }
        H = (0.5 * (H + H.transpose())).eval();
        return H.allFinite();
    };

    Eigen::VectorXd beta = beta0;
    Eval cur = eval(beta);
    if (!cur.ok) return out;
    Eigen::MatrixXd H;
    bool H_at_beta = false;   // H was formed at the current beta
    for (int it = 0; it < max_iter; ++it) {
        if (!curvature(beta, H)) return out;
        H_at_beta = true;
        Eigen::LLT<Eigen::MatrixXd> llt(H);
        Eigen::VectorXd step;
        if (llt.info() == Eigen::Success) {
            step = llt.solve(cur.g);
        } else {
            // Away from the maximum the curvature need not be positive
            // definite; a gradient step scaled by the largest curvature keeps
            // the iteration moving uphill until it is.
            const double scale = std::max(H.cwiseAbs().maxCoeff(), 1.0);
            step = cur.g / scale;
        }
        // Converged when the predicted gain g' H^-1 g / 2 is below the relative
        // tolerance on F that the BFGS read used (reltol 1e-10).
        const double ftol = 1e-10 * (std::fabs(cur.F) + 1e-10);
        const double decrement = cur.g.dot(step);
        if (std::isfinite(decrement) && decrement >= 0.0 &&
            0.5 * decrement < ftol) break;
        double t = 1.0;
        bool moved = false;
        for (int half = 0; half < 30; ++half, t *= 0.5) {
            const Eval nx = eval(beta + t * step);
            if (nx.ok && nx.F > cur.F) {
                beta += t * step;
                H_at_beta = false;
                const double gain = nx.F - cur.F;
                cur = nx;
                moved = gain >= ftol;
                break;
            }
        }
        if (!moved) break;
    }
    if (!H_at_beta && !curvature(beta, H)) return out;
    Eigen::LLT<Eigen::MatrixXd> lltH(H);
    if (lltH.info() != Eigen::Success) return out;
    const double logdet =
        2.0 * Eigen::MatrixXd(lltH.matrixL()).diagonal().array().log().sum();
    out.beta = beta;
    out.H = H;
    out.log_marginal = cur.F + 0.5 * p * LOG_2PI - 0.5 * logdet;
    out.ok = std::isfinite(out.log_marginal) && beta.allFinite();
    return out;
}

} // namespace tulpa

#endif // TULPA_AGHQ_RE_PROFILE_H
