// re_em.cpp
// Laplace-EM for grouped random effects on a per-group oracle (REGroupOracle):
// the n_quad = 1 sibling of the AGHQ engine (aghq_re.cpp), sharing its per-group
// mode solve (aghq_group_solve). Each iteration
//   1. solves every group's posterior mode b_g and covariance C_g = H_g^{-1},
//      H_g = negH_g(b_g) + Sigma^{-1};
//   2. updates each covariance block in closed form,
//      Sigma_m = mean_g(b_gm b_gm' + C_gm), diagonal for an uncorrelated block,
//      eigenvalues floored at sigma_floor^2;
//   3. moves theta by BFGS on the Laplace marginal at that Sigma,
//      L(theta) = sum_g [ell_g(b_g) - b_g'P b_g / 2 - log|H_g| / 2]
//                 - G log|Sigma| / 2 - prec |theta|^2 / 2.
// The theta gradient is exact by the envelope theorem (b_g is stationary):
//   dL/dtheta_k = sum_g [ s_gk - tr(C_g dH_g/dtheta_k) / 2 ] - prec theta_k,
// with s_gk the data theta-score at b_g and dH_g/dtheta_k the TOTAL derivative
// of the penalized precision along db_g/dtheta_k = C_g d(grad_b ell_g)/dtheta_k.
// Every piece is read through grad_hess alone, by central differences in theta
// at the mode (no mode re-solve), so it holds for any oracle, native or the
// R-closure bridge, without a theta_score implementation. The per-group loops
// run on `n_threads` threads only when the oracle declares thread_safe(); each
// group writes its own slots and the sums run in group order afterwards, so the
// result does not depend on the thread count.

#include "aghq_re_core.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <vector>
#include <cmath>
#include <limits>

using namespace Rcpp;
using namespace tulpa;

namespace {

struct EmBlocks {
    std::vector<int> nc, off;
    std::vector<bool> full;
    int d = 0;
};

struct EmSweep {
    double L = -std::numeric_limits<double>::infinity();
    std::vector<Eigen::VectorXd> b;
    std::vector<Eigen::MatrixXd> C;
    std::vector<char> ok;
    bool all_ok = false;
};

Eigen::MatrixXd em_block_diag(const std::vector<Eigen::MatrixXd>& S,
                              const EmBlocks& bl) {
    Eigen::MatrixXd Sig = Eigen::MatrixXd::Zero(bl.d, bl.d);
    for (size_t m = 0; m < S.size(); ++m)
        Sig.block(bl.off[m], bl.off[m], bl.nc[m], bl.nc[m]) = S[m];
    return Sig;
}

// All group modes at theta (already bound) and precision P; the Laplace
// marginal L excluding the theta prior. `warm`, when it holds a previous sweep,
// starts each group's Newton at that sweep's mode. The group loop runs in
// parallel on an oracle that declares itself thread-safe; each group writes
// only its own slots, so the result does not depend on the thread count.
EmSweep em_sweep(const REGroupOracle& orc, const AghqSigmaFactor& sf, int d,
                 int n_threads, const EmSweep* warm = nullptr) {
    const int ng = orc.n_groups;
    EmSweep s;
    s.b.resize(ng); s.C.resize(ng); s.ok.assign(ng, 0);
    std::vector<double> lm(ng, 0.0);
    const int nt = orc.thread_safe() ? std::max(1, n_threads) : 1;
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
#endif
    for (int g = 0; g < ng; ++g) {
        std::vector<double> grad(d), negH((size_t)d * d);
        const Eigen::VectorXd* b0 =
            (warm && (int)warm->ok.size() == ng && warm->ok[g]) ? &warm->b[g] : nullptr;
        const AghqGroupSolve gs = aghq_group_solve(orc, g, sf.P, d, false, b0);
        if (!gs.ok) continue;
        double logL = 0.0;
        orc.grad_hess(g, gs.mode.b.data(), logL, grad.data(), negH.data());
        Eigen::LLT<Eigen::MatrixXd> llt(gs.mode.negH);
        if (llt.info() != Eigen::Success) continue;
        const double ldH =
            2.0 * Eigen::MatrixXd(llt.matrixL()).diagonal().array().log().sum();
        lm[g] = logL - 0.5 * gs.mode.b.dot(sf.P * gs.mode.b) - 0.5 * ldH;
        s.b[g] = gs.mode.b; s.C[g] = gs.C; s.ok[g] = 1;
    }
    s.all_ok = true;
    double L = -0.5 * ng * sf.logdet;
    for (int g = 0; g < ng; ++g) {
        if (!s.ok[g]) { s.all_ok = false; break; }
        L += lm[g];
    }
    s.L = s.all_ok ? L : -std::numeric_limits<double>::infinity();
    return s;
}

// Envelope gradient of L in theta at the modes of `s` (orc bound to theta on
// entry and on exit).
Eigen::VectorXd em_theta_grad(REGroupOracle& orc, const Eigen::VectorXd& theta,
                              const EmSweep& s, double prec, int n_threads) {
    const int ng = orc.n_groups, d = orc.d, nth = orc.n_theta;
    Eigen::VectorXd grad = -prec * theta;
    const int nt = orc.thread_safe() ? std::max(1, n_threads) : 1;
    (void)nt;
    // Per coordinate: pass 1 at (theta +- h e_k, b_g) gives the data score and
    // d(grad_b ell)/dtheta_k; pass 2 at (theta +- h e_k, b_g +- h db_k) gives
    // the total derivative of the precision.
    std::vector<double> lp(ng), lmn(ng);
    std::vector<Eigen::VectorXd> gp(ng), gm(ng), db(ng);
    std::vector<Eigen::MatrixXd> Hp(ng), Hm(ng);
    for (int k = 0; k < nth; ++k) {
        const double h = 1e-5 * std::max(1.0, std::abs(theta(k)));
        Eigen::VectorXd tp = theta, tm = theta;
        tp(k) += h; tm(k) -= h;
        // pass(sign, shift): grad_hess at (theta + sign h e_k, b_g + shift db_g).
        auto pass = [&](const Eigen::VectorXd& t, double shift,
                        std::vector<double>* L_out, std::vector<Eigen::VectorXd>* G_out,
                        std::vector<Eigen::MatrixXd>* H_out) {
            orc.rebind(t.data());
#ifdef _OPENMP
            #pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
#endif
            for (int g = 0; g < ng; ++g) {
                std::vector<double> gr(d), nh((size_t)d * d);
                const Eigen::VectorXd bb = (shift == 0.0) ? s.b[g]
                                           : Eigen::VectorXd(s.b[g] + shift * db[g]);
                double l = 0.0;
                orc.grad_hess(g, bb.data(), l, gr.data(), nh.data());
                if (L_out) (*L_out)[g] = l;
                if (G_out) (*G_out)[g] = Eigen::Map<Eigen::VectorXd>(gr.data(), d);
                if (H_out) (*H_out)[g] = Eigen::Map<Eigen::MatrixXd>(nh.data(), d, d);
            }
        };
        pass(tp, 0.0, &lp, &gp, nullptr);
        pass(tm, 0.0, &lmn, &gm, nullptr);
        for (int g = 0; g < ng; ++g)
            db[g] = s.C[g] * ((gp[g] - gm[g]) / (2.0 * h));
        pass(tp, h, nullptr, nullptr, &Hp);
        pass(tm, -h, nullptr, nullptr, &Hm);
        double gk = 0.0;
        for (int g = 0; g < ng; ++g) {
            const double score = (lp[g] - lmn[g]) / (2.0 * h);
            const Eigen::MatrixXd dH = (Hp[g] - Hm[g]) / (2.0 * h);
            gk += score - 0.5 * (s.C[g].cwiseProduct(dH)).sum();
        }
        grad(k) += gk;
    }
    orc.rebind(theta.data());
    return grad;
}

}  // namespace

// [[Rcpp::export]]
List cpp_re_em(SEXP oracle, NumericVector theta0, List Sigma0,
               IntegerVector nc, LogicalVector full,
               double theta_prior_prec, double sigma_floor, double tol,
               int max_iter, int theta_iter, int n_threads) {
    XPtr<REGroupOracle> orc(oracle);
    const int ng = orc->n_groups, nth = orc->n_theta;
    EmBlocks bl;
    for (int m = 0; m < nc.size(); ++m) {
        bl.off.push_back(bl.d);
        bl.nc.push_back(nc[m]);
        bl.full.push_back((bool)full[m]);
        bl.d += nc[m];
    }
    const int d = bl.d;
    if (d != orc->d) stop("cpp_re_em: the blocks span %d coordinates, the oracle %d.", d, orc->d);
    if ((int)theta0.size() != nth) stop("cpp_re_em: theta0 has length %d, the oracle %d.",
                                        (int)theta0.size(), nth);
    std::vector<Eigen::MatrixXd> S(nc.size());
    for (int m = 0; m < nc.size(); ++m) {
        NumericMatrix Sm = Sigma0[m];
        S[m] = Eigen::Map<Eigen::MatrixXd>(REAL(Sm), nc[m], nc[m]);
    }
    Eigen::VectorXd theta(nth);
    for (int i = 0; i < nth; ++i) theta(i) = theta0[i];
    const double prec = theta_prior_prec;
    auto prior = [&](const Eigen::VectorXd& t) { return -0.5 * prec * t.squaredNorm(); };

    auto factor = [&]() { return aghq_sigma_factor(em_block_diag(S, bl), d); };
    AghqSigmaFactor sf = factor();
    if (!sf.ok) stop("cpp_re_em: the starting covariance is not positive definite.");
    orc->rebind(theta.data());
    EmSweep sw = em_sweep(*orc, sf, d, n_threads);
    if (!sw.all_ok)
        return List::create(_["ok"] = false, _["reason"] = "start");

    std::vector<double> trace{sw.L + prior(theta)};
    bool converged = false;
    int it = 0;
    for (it = 1; it <= max_iter; ++it) {
        // Closed-form covariance update from the current posterior moments.
        for (size_t m = 0; m < S.size(); ++m) {
            const int o = bl.off[m], c = bl.nc[m];
            Eigen::MatrixXd Sm = Eigen::MatrixXd::Zero(c, c);
            for (int g = 0; g < ng; ++g) {
                const Eigen::VectorXd bg = sw.b[g].segment(o, c);
                Sm += bg * bg.transpose() + sw.C[g].block(o, o, c, c);
            }
            Sm /= ng;
            if (!bl.full[m]) Sm = Eigen::MatrixXd(Sm.diagonal().asDiagonal());
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (Sm + Sm.transpose()));
            Eigen::VectorXd ev = es.eigenvalues().cwiseMax(sigma_floor * sigma_floor);
            S[m] = es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
        }
        sf = factor();
        if (!sf.ok) break;
        sw = em_sweep(*orc, sf, d, n_threads, &sw);
        if (!sw.all_ok) break;

        // BFGS on theta at this Sigma, started from the identity scaled by the
        // first gradient.
        if (nth > 0) {
            Eigen::VectorXd gcur = em_theta_grad(*orc, theta, sw, prec, n_threads);
            Eigen::MatrixXd Binv = Eigen::MatrixXd::Identity(nth, nth) /
                                   std::max(1.0, gcur.cwiseAbs().maxCoeff());
            double fcur = sw.L + prior(theta);
            for (int j = 0; j < theta_iter; ++j) {
                Eigen::VectorXd step = Binv * gcur;
                double scale = 1.0;
                bool moved = false;
                Eigen::VectorXd tnew; EmSweep snew;
                while (scale > 1e-6) {
                    tnew = theta + scale * step;
                    orc->rebind(tnew.data());
                    snew = em_sweep(*orc, sf, d, n_threads, &sw);
                    if (snew.all_ok && snew.L + prior(tnew) >= fcur + 1e-4 * scale * gcur.dot(step)) {
                        moved = true; break;
                    }
                    scale *= 0.5;
                }
                if (!moved) { orc->rebind(theta.data()); break; }
                const Eigen::VectorXd gnew = em_theta_grad(*orc, tnew, snew, prec, n_threads);
                const Eigen::VectorXd sv = tnew - theta, yv = gcur - gnew;
                const double fnew = snew.L + prior(tnew);
                theta = tnew; sw = snew;
                const double sy = sv.dot(yv);
                if (sy > 1e-12) {
                    const double rho = 1.0 / sy;
                    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nth, nth);
                    Binv = (I - rho * sv * yv.transpose()) * Binv *
                           (I - rho * yv * sv.transpose()) + rho * sv * sv.transpose();
                }
                const double df = fnew - fcur;
                gcur = gnew; fcur = fnew;
                if (gcur.cwiseAbs().maxCoeff() < 1e-6 || std::abs(df) < 1e-10 * (1.0 + std::abs(fcur)))
                    break;
            }
        }
        const double Lnew = sw.L + prior(theta);
        const double Lold = trace.back();
        trace.push_back(Lnew);
        if (std::abs(Lnew - Lold) < tol * (1.0 + std::abs(Lold))) { converged = true; break; }
    }

    // Fixed-parameter covariance: inverse of the theta curvature of L at the
    // final Sigma, the central-difference Jacobian of the envelope gradient.
    Eigen::MatrixXd Hth = Eigen::MatrixXd::Zero(nth, nth);
    bool hess_ok = sw.all_ok;
    for (int k = 0; k < nth && hess_ok; ++k) {
        const double h = 1e-4 * std::max(1.0, std::abs(theta(k)));
        Eigen::VectorXd tp = theta, tm = theta;
        tp(k) += h; tm(k) -= h;
        orc->rebind(tp.data());
        EmSweep sp = em_sweep(*orc, sf, d, n_threads, &sw);
        orc->rebind(tm.data());
        EmSweep sm = em_sweep(*orc, sf, d, n_threads, &sw);
        if (!sp.all_ok || !sm.all_ok) { hess_ok = false; break; }
        const Eigen::VectorXd gp = em_theta_grad(*orc, tp, sp, prec, n_threads);
        const Eigen::VectorXd gm = em_theta_grad(*orc, tm, sm, prec, n_threads);
        Hth.col(k) = -(gp - gm) / (2.0 * h);
    }
    orc->rebind(theta.data());
    sw = em_sweep(*orc, sf, d, n_threads, &sw);
    Hth = 0.5 * (Hth + Hth.transpose());

    NumericMatrix TH(nth, nth);
    for (int i = 0; i < nth; ++i) for (int j = 0; j < nth; ++j) TH(i, j) = Hth(i, j);
    List Sout(S.size());
    for (size_t m = 0; m < S.size(); ++m) {
        NumericMatrix Sm(bl.nc[m], bl.nc[m]);
        for (int i = 0; i < bl.nc[m]; ++i) for (int j = 0; j < bl.nc[m]; ++j) Sm(i, j) = S[m](i, j);
        Sout[m] = Sm;
    }
    NumericMatrix BHAT(ng, d);
    NumericVector BCOV((size_t)ng * d * d);
    BCOV.attr("dim") = IntegerVector::create(ng, d, d);
    LogicalVector GOK(ng);
    for (int g = 0; g < ng; ++g) {
        GOK(g) = (bool)sw.ok[g];
        for (int j = 0; j < d; ++j) {
            BHAT(g, j) = sw.ok[g] ? sw.b[g](j) : NA_REAL;
            for (int k = 0; k < d; ++k)
                BCOV(g + (size_t)ng * (j + (size_t)d * k)) = sw.ok[g] ? sw.C[g](j, k) : NA_REAL;
        }
    }
    NumericVector th(nth);
    for (int i = 0; i < nth; ++i) th(i) = theta(i);
    return List::create(_["ok"] = true, _["theta"] = th, _["Sigma_list"] = Sout,
                        _["theta_hessian"] = TH, _["theta_hessian_ok"] = hess_ok,
                        _["log_marginal"] = sw.all_ok ? sw.L : NA_REAL,
                        _["trace"] = wrap(trace), _["bhat"] = BHAT, _["bcov"] = BCOV,
                        _["group_ok"] = GOK, _["converged"] = converged,
                        _["n_iter"] = std::min(it, max_iter));
}
