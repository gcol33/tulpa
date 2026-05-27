// aghq_re_mode.h
// Shared per-group posterior mode-find for the random-effect engine. ONE source
// for the Newton that both the AGHQ marginal (node centring) and the Gibbs
// sweep (proposal shape) use, so neither re-implements it.
//
// Solves, for group g, the penalized stationarity
//   argmax_b [ ell_g(b) - 0.5 b' P b ],   P = Sigma^{-1},
// and returns the mode b_hat together with the penalized precision
// negH = (-d^2 ell_g/db^2)(b_hat) + P (the observed information used by the
// Laplace/AGHQ marginal). This is the C++ port of grp_mode() in R/re_aghq.R.
//
// Globally convergent safeguarded Newton: the step curvature is the penalized
// precision with its eigenvalues floored to a PD ascent direction, and an
// Armijo backtrack on the penalized objective accepts only increases. This is
// needed because a latent-variable marginal (the N-mixture, whose per-site
// observed information carries the -Var[N|y] coupling) can be INDEFINITE away
// from the mode; a plain Newton there takes non-ascent steps and diverges. At
// the mode itself the observed information is PSD (it is a local max), so the
// returned negH is valid for the marginal. For a GLMM (already-PD negH) the
// flooring is a no-op and the full Newton step is accepted unchanged.

#ifndef TULPA_AGHQ_RE_MODE_H
#define TULPA_AGHQ_RE_MODE_H

#include "tulpa/aghq_oracle.h"
#include <RcppEigen.h>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <cmath>

namespace tulpa {

struct GroupMode {
    Eigen::VectorXd b;       // posterior mode b_hat (length d)
    Eigen::MatrixXd negH;    // penalized precision: data negH(b_hat) + P (d x d)
    bool ok = true;          // false if the solve failed irrecoverably
};

// orc must already be rebind()-ed to the active theta. P is Sigma^{-1} (d x d).
inline GroupMode aghq_group_mode(const REGroupOracle& orc, int g,
                                 const Eigen::MatrixXd& P,
                                 int max_it = 50, double tol = 1e-9) {
    const int d = orc.d;
    GroupMode out;
    out.b = Eigen::VectorXd::Zero(d);

    std::vector<double> grad(d), negH(d * d);
    double logL = 0.0;

    // Evaluate score / observed info at b and return the penalized objective
    // q(b) = ell_g(b) - 0.5 b' P b (the quantity the mode maximizes).
    auto eval = [&](const Eigen::VectorXd& b,
                    Eigen::VectorXd& g_out, Eigen::MatrixXd& H_out) -> double {
        orc.grad_hess(g, b.data(), logL, grad.data(), negH.data());
        g_out = Eigen::Map<Eigen::VectorXd>(grad.data(), d);
        // negH is symmetric; storage order is irrelevant for a symmetric matrix.
        H_out = Eigen::Map<Eigen::MatrixXd>(negH.data(), d, d);
        return logL - 0.5 * b.dot(P * b);
    };

    Eigen::VectorXd gv(d);
    Eigen::MatrixXd H(d, d);
    std::vector<double> fish((std::size_t)d * d);
    double q_cur = eval(out.b, gv, H);
    for (int it = 0; it < max_it; ++it) {
        const Eigen::VectorXd g_pen = gv - P * out.b;          // penalized score

        // Step curvature: prefer the true observed-info Hessian (quadratic
        // convergence -> a precise mode) when it is PD, which it is at and near
        // the mode. Where it is indefinite (a latent-variable marginal away from
        // the mode), fall back to the oracle's PSD Fisher so the step still
        // ascends. The Fisher was cached by the grad_hess in the last eval, at
        // this same out.b.
        Eigen::MatrixXd H_pen = H + P;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H_pen);
        if (es.info() != Eigen::Success) { out.ok = false; break; }
        const double pd_tol = 1e-10 * std::max(1.0, es.eigenvalues().cwiseAbs().maxCoeff());
        if (es.eigenvalues().minCoeff() <= pd_tol &&
            orc.newton_hess(g, out.b.data(), fish.data())) {
            H_pen = Eigen::Map<Eigen::MatrixXd>(fish.data(), d, d) + P;
            es.compute(H_pen);
            if (es.info() != Eigen::Success) { out.ok = false; break; }
        }

        // Modified-Newton direction: reflect negative eigenvalues (use |lambda|)
        // and floor tiny ones, so the step stays bounded and ascends q even for
        // an indefinite curvature. (A tiny positive floor instead of reflection
        // would blow the step up along a near-zero / negative direction.)
        Eigen::VectorXd evals = es.eigenvalues();
        const double fl = 1e-8 * std::max(1.0, evals.cwiseAbs().maxCoeff());
        for (int i = 0; i < d; ++i) evals(i) = std::max(std::abs(evals(i)), fl);
        const Eigen::VectorXd step =
            es.eigenvectors() *
            (es.eigenvectors().transpose() * g_pen).cwiseQuotient(evals);
        if (!step.allFinite()) { out.ok = false; break; }

        // Armijo backtrack on the penalized objective (step is an ascent dir).
        const double armijo = 1e-4 * g_pen.dot(step);
        double scale = 1.0;
        bool improved = false;
        Eigen::VectorXd b_try;
        double q_try = q_cur;
        while (scale > 1e-4) {
            b_try = out.b + scale * step;
            if (b_try.allFinite()) {
                q_try = eval(b_try, gv, H);
                if (q_try >= q_cur + scale * armijo) { improved = true; break; }
            }
            scale *= 0.5;
        }
        if (!improved) { eval(out.b, gv, H); break; }  // restore gv/H at the mode
        out.b = b_try; q_cur = q_try;                   // gv/H already at b_try
        if ((scale * step).cwiseAbs().maxCoeff() < tol) break;
    }
    // Observed information at the mode (gv/H hold the values at out.b).
    out.negH = H + P;
    return out;
}

} // namespace tulpa

#endif // TULPA_AGHQ_RE_MODE_H
