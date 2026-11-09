// re_em.cpp
// Laplace-EM for grouped random effects on a per-group oracle (REGroupOracle):
// the n_quad = 1 sibling of the AGHQ engine (aghq_re.cpp), sharing its per-group
// mode solve (aghq_group_solve). Each iteration
//   1. solves every group's posterior mode b_g and covariance C_g = H_g^{-1},
//      H_g = negH_g(b_g) + Sigma^{-1};
//   2. updates each covariance block in closed form,
//      Sigma_m = mean_g(b_gm b_gm' + C_gm), diagonal for an uncorrelated block,
//      eigenvalues floored at sigma_floor^2;
//   3. moves theta, by one of two rules.
//
// theta_update = "marginal": BFGS on the Laplace marginal at that Sigma,
//   L(theta) = sum_g [ell_g(b_g) - b_g'P b_g / 2 - log|H_g| / 2]
//              - G log|Sigma| / 2 - theta' D theta / 2,
// D = diag(prior precision). The theta gradient is exact by the envelope
// theorem (b_g is stationary):
//   dL/dtheta_k = sum_g [ s_gk - tr(C_g dH_g/dtheta_k) / 2 ] - D_k theta_k,
// with s_gk the data theta-score at b_g and dH_g/dtheta_k the TOTAL derivative
// of the penalized precision along db_g/dtheta_k = C_g d(grad_b ell_g)/dtheta_k.
// Every piece is read through grad_hess alone, by central differences in theta
// at the mode (no mode re-solve), so it holds for any oracle. The theta
// covariance is the inverse curvature of L.
//
// theta_update = "latent": theta is a latent Gaussian with prior N(0, D^-1),
// solved jointly with every b_g by a damped Newton on the penalized joint
// log-density F(theta, b) = sum_g ell_g(b_g) - sum_g b_g'P b_g / 2 -
// theta'D theta / 2. The (theta, b) Hessian is an arrowhead (theta couples to
// every group, the groups not to each other), so each step folds the b blocks
// into the Schur complement S = A_tt + D - sum_g X_g C_g X_g' (A_tt the theta
// information, X_g the theta-b cross information) and back-substitutes. A step
// is accepted on a backtracking line search, escalating to a Levenberg ridge on
// S; an iteration that accepts no step at any damping is reported (`stuck`)
// instead of read as convergence. The theta covariance is S^-1, the marginal
// covariance of theta under the joint Gaussian approximation. This is the
// treatment of fixed effects as latent Gaussian variables (INLA): it needs only
// first and second derivatives of ell_g, where the marginal rule's exact
// gradient differentiates the per-group Hessian once more. It needs the
// oracle's theta score (an R-closure oracle supplies it through `joint`).
//
// The per-group loops run on `n_threads` threads only when the oracle declares
// thread_safe(); each group writes its own slots and the sums run in group
// order afterwards, so the result does not depend on the thread count.

#include "aghq_re_core.h"
#include "aghq_re_oracles.h"
#include <tulpa/nested_progress.h>
#include <tulpa/re_cov_update.h>
#include <Rcpp.h>
#include <RcppEigen.h>
#include <memory>
#include <string>
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

// Closed-form covariance update from the posterior moments (b_g, C_g).
void em_sigma_update(std::vector<Eigen::MatrixXd>& S, const EmBlocks& bl,
                     const std::vector<Eigen::VectorXd>& b,
                     const std::vector<Eigen::MatrixXd>& C, double sigma_floor) {
    const int ng = (int)b.size();
    for (size_t m = 0; m < S.size(); ++m) {
        const int o = bl.off[m], c = bl.nc[m];
        Eigen::MatrixXd Sm = Eigen::MatrixXd::Zero(c, c);
        for (int g = 0; g < ng; ++g) {
            const Eigen::VectorXd bg = b[g].segment(o, c);
            Sm += bg * bg.transpose() + C[g].block(o, o, c, c);
        }
        S[m] = re_cov_update(Sm / ng, bl.full[m], sigma_floor);
    }
}

// AGHQ moment step at the converged latent-rule mode. Each group's posterior
// second moment E[b_g b_g' | y] is integrated by adaptive Gauss-Hermite
// quadrature on the tensor grid the AGHQ engine uses, centred at b_g and scaled
// by chol(C_g); the covariance update takes the group mean of those moments in
// place of the Laplace b_g b_g' + C_g, which removes the Laplace small-cluster
// attenuation of the variance components. Each C_g is then reprojected at the
// new precision through its likelihood curvature negH_g = C_g^-1 - P_old, and
// the theta precision with it, S_new = S + sum_g X_g (C_g_old - C_g_new) X_g'
// (X_g is likelihood curvature, independent of Sigma), so Sigma, C_g and S
// describe one fit. Returns false, leaving every input untouched, when a
// factorization fails or a quadrature sum is not finite.
bool em_aghq_debias(REGroupOracle& orc, const EmBlocks& bl, int n_quad,
                    double sigma_floor, const std::vector<Eigen::VectorXd>& b,
                    std::vector<Eigen::MatrixXd>& C, const std::vector<Eigen::MatrixXd>& X,
                    std::vector<Eigen::MatrixXd>& S, Eigen::MatrixXd& Sth,
                    const AghqSigmaFactor& sf_old) {
    const int ng = orc.n_groups, d = bl.d;
    const AghqGrid grid = aghq_build_grid(std::vector<int>(d, n_quad));
    const int nN = grid.n_nodes;
    std::vector<Eigen::MatrixXd> Rg(ng);
    std::vector<double> hv(nN);
    for (int g = 0; g < ng; ++g) {
        Eigen::LLT<Eigen::MatrixXd> lltC(0.5 * (C[g] + C[g].transpose()));
        if (lltC.info() != Eigen::Success) return false;
        const Eigen::MatrixXd Lc = lltC.matrixL();
        RowMat B = grid.Znodes * Lc.transpose();
        B.rowwise() += b[g].transpose();
        orc.node_ll(g, B.data(), nN, hv.data());
        const Eigen::VectorXd quad = (B * sf_old.P).cwiseProduct(B).rowwise().sum();
        Eigen::VectorXd terms(nN);
        for (int k = 0; k < nN; ++k)
            terms(k) = grid.logw(k) + hv[k] - 0.5 * quad(k) + 0.5 * grid.z2(k);
        const double mx = terms.maxCoeff();
        const double lse = mx + std::log((terms.array() - mx).exp().sum());
        if (!std::isfinite(lse)) return false;
        const Eigen::VectorXd wt = (terms.array() - lse).exp();
        Rg[g] = B.transpose() * wt.asDiagonal() * B;
    }
    std::vector<Eigen::MatrixXd> Snew(S.size());
    for (size_t m = 0; m < S.size(); ++m) {
        const int o = bl.off[m], c = bl.nc[m];
        Eigen::MatrixXd Mm = Eigen::MatrixXd::Zero(c, c);
        for (int g = 0; g < ng; ++g) Mm += Rg[g].block(o, o, c, c);
        Snew[m] = re_cov_update(Mm / ng, bl.full[m], sigma_floor);
    }
    const AghqSigmaFactor sf_new = aghq_sigma_factor(em_block_diag(Snew, bl), d);
    if (!sf_new.ok) return false;
    std::vector<Eigen::MatrixXd> Cnew(ng);
    Eigen::MatrixXd Snew_th = Sth;
    for (int g = 0; g < ng; ++g) {
        Eigen::FullPivLU<Eigen::MatrixXd> luC(C[g]);
        if (!luC.isInvertible()) return false;
        const Eigen::MatrixXd negH = Eigen::MatrixXd(luC.inverse()) - sf_old.P;
        Eigen::FullPivLU<Eigen::MatrixXd> luN(negH + sf_new.P);
        if (!luN.isInvertible()) return false;
        Cnew[g] = luN.inverse();
        if (X[g].rows() > 0) Snew_th += X[g] * (C[g] - Cnew[g]) * X[g].transpose();
    }
    if (!Snew_th.allFinite()) return false;
    S = Snew; C = Cnew; Sth = Snew_th;
    return true;
}

// log|M| of a symmetric matrix through its LDLT; -Inf when not positive
// definite.
double em_logdet(const Eigen::MatrixXd& M) {
    Eigen::LDLT<Eigen::MatrixXd> ldlt(0.5 * (M + M.transpose()));
    if (ldlt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
    const Eigen::VectorXd dv = ldlt.vectorD();
    if ((dv.array() <= 0.0).any()) return -std::numeric_limits<double>::infinity();
    return dv.array().log().sum();
}

// All group modes at theta (already bound) and precision P; the Laplace
// marginal L excluding the theta prior. `warm`, when it holds a previous sweep,
// starts each group's Newton at that sweep's mode.
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
                              const EmSweep& s, const Eigen::VectorXd& prec,
                              int n_threads) {
    const int ng = orc.n_groups, d = orc.d, nth = orc.n_theta;
    Eigen::VectorXd grad = -prec.cwiseProduct(theta);
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

// ---- latent rule ----------------------------------------------------------

// One group's derivative blocks at (theta, b) for the joint Newton.
struct GroupJoint {
    double l = 0.0;
    Eigen::VectorXd gb, gt;          // d ell/db (d), d ell/dtheta (nth)
    Eigen::MatrixXd Hbb, Htt, Htb;   // -d2 ell/db2, -d2 ell/dtheta2, -d2 ell/dtheta db
};

// Every group's blocks at the bound theta. An R-closure oracle with a `joint`
// closure answers in one call per group; otherwise grad_hess gives the b
// blocks, theta_score the theta score, aghq_group_cross_hess the cross block
// (central difference of the theta score over b), theta_obs_info the theta
// information -- or, where the oracle does not supply it, a central difference
// of the theta score over theta (which rebinds, and so runs as its own passes).
std::vector<GroupJoint> em_joint_blocks(REGroupOracle& orc,
                                        const Eigen::VectorXd& theta,
                                        const std::vector<Eigen::VectorXd>& b,
                                        int n_threads) {
    const int ng = orc.n_groups, d = orc.d, nth = orc.n_theta;
    std::vector<GroupJoint> J(ng);
    const RClosureOracle* rc = dynamic_cast<const RClosureOracle*>(&orc);
    if (rc && rc->has_joint) {
        for (int g = 0; g < ng; ++g) {
            GroupJoint& j = J[g];
            j.gb.resize(d); j.gt.resize(nth);
            j.Hbb.resize(d, d); j.Htt.resize(nth, nth); j.Htb.resize(nth, d);
            rc->joint(g, b[g].data(), j.l, j.gb.data(), j.Hbb.data(), j.gt.data(),
                      j.Htt.data(), j.Htb.data());
        }
        return J;
    }
    const int nt = orc.thread_safe() ? std::max(1, n_threads) : 1;
    (void)nt;
    std::vector<char> need_tt(ng, 0);
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
#endif
    for (int g = 0; g < ng; ++g) {
        GroupJoint& j = J[g];
        std::vector<double> gr(d), nh((size_t)d * d), ts(nth), A((size_t)nth * nth);
        orc.grad_hess(g, b[g].data(), j.l, gr.data(), nh.data());
        j.gb = Eigen::Map<Eigen::VectorXd>(gr.data(), d);
        j.Hbb = Eigen::Map<Eigen::MatrixXd>(nh.data(), d, d);
        orc.theta_score(g, b[g].data(), ts.data());
        j.gt = Eigen::Map<Eigen::VectorXd>(ts.data(), nth);
        j.Htb = aghq_group_cross_hess(orc, g, b[g]);
        if (orc.theta_obs_info(g, b[g].data(), A.data()))
            j.Htt = Eigen::Map<Eigen::MatrixXd>(A.data(), nth, nth);
        else
            need_tt[g] = 1;
    }
    bool any_tt = false;
    for (int g = 0; g < ng; ++g) if (need_tt[g]) { any_tt = true; J[g].Htt.setZero(nth, nth); }
    if (any_tt) {
        for (int k = 0; k < nth; ++k) {
            const double h = 1e-5 * std::max(1.0, std::abs(theta(k)));
            Eigen::VectorXd tp = theta, tm = theta;
            tp(k) += h; tm(k) -= h;
            std::vector<Eigen::VectorXd> sp(ng), sm(ng);
            for (int sgn = 0; sgn < 2; ++sgn) {
                orc.rebind(sgn == 0 ? tp.data() : tm.data());
#ifdef _OPENMP
                #pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
#endif
                for (int g = 0; g < ng; ++g) {
                    if (!need_tt[g]) continue;
                    std::vector<double> ts(nth);
                    orc.theta_score(g, b[g].data(), ts.data());
                    (sgn == 0 ? sp : sm)[g] = Eigen::Map<Eigen::VectorXd>(ts.data(), nth);
                }
            }
            for (int g = 0; g < ng; ++g)
                if (need_tt[g]) J[g].Htt.col(k) = -(sp[g] - sm[g]) / (2.0 * h);
        }
        orc.rebind(theta.data());
        for (int g = 0; g < ng; ++g)
            if (need_tt[g]) J[g].Htt = 0.5 * (J[g].Htt + J[g].Htt.transpose());
    }
    return J;
}

// Penalized joint log-density F at (theta, b); orc is rebound to theta.
double em_joint_F(REGroupOracle& orc, const Eigen::VectorXd& theta,
                  const std::vector<Eigen::VectorXd>& b, const Eigen::MatrixXd& P,
                  const Eigen::VectorXd& prec, int n_threads) {
    const int ng = orc.n_groups, d = orc.d;
    orc.rebind(theta.data());
    std::vector<double> lg(ng, 0.0);
    const int nt = orc.thread_safe() ? std::max(1, n_threads) : 1;
    (void)nt;
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic) num_threads(nt) if (nt > 1)
#endif
    for (int g = 0; g < ng; ++g) {
        double v = 0.0;
        orc.node_ll(g, b[g].data(), 1, &v);
        lg[g] = v - 0.5 * b[g].dot(P * b[g]);
    }
    (void)d;
    double F = -0.5 * theta.dot(prec.cwiseProduct(theta));
    for (int g = 0; g < ng; ++g) F += lg[g];
    return F;
}

struct LatentMode {
    Eigen::VectorXd theta;
    std::vector<Eigen::VectorXd> b;
    std::vector<Eigen::MatrixXd> C, X;  // posterior covariances, cross blocks
    Eigen::MatrixXd S;                  // Schur complement (theta precision)
    double F = -std::numeric_limits<double>::infinity();
    bool stuck = false, ok = true;
};

// One group's contribution to the joint negative Hessian of F,
//   G_g = [Htt_g, Htb_g; Htb_g', Hbb_g + P],
// made positive semi-definite where it is not (Nocedal & Wright 2006,
// sec. 3.4, eigenvalue modification by absolute value). A group whose
// log-likelihood is not concave at the current point (a half-normal distance
// marginal far out along its lambda-sigma ridge) otherwise yields a Newton
// direction that DESCENDS its own log-likelihood; the joint line search can
// still accept it, since it scores the sum over groups, and the group is
// carried away from its mode and held there. A block that is already PSD is
// returned untouched, so a concave problem takes exactly the plain Newton step.
// Returns true when the block was modified.
bool em_group_psd(Eigen::MatrixXd& Htt, Eigen::MatrixXd& Htb, Eigen::MatrixXd& HbbP) {
    const int nth = (int)Htt.rows(), d = (int)HbbP.rows();
    const int n = nth + d;
    if (n == 0) return false;
    Eigen::MatrixXd G(n, n);
    G.topLeftCorner(nth, nth) = Htt;
    G.topRightCorner(nth, d) = Htb;
    G.bottomLeftCorner(d, nth) = Htb.transpose();
    G.bottomRightCorner(d, d) = HbbP;
    G = 0.5 * (G + G.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    if (es.info() != Eigen::Success) return false;
    const Eigen::VectorXd ev = es.eigenvalues();
    const double scale = ev.cwiseAbs().maxCoeff();
    if (!(ev.minCoeff() < -1e-10 * std::max(scale, 1.0))) return false;
    const Eigen::MatrixXd& V = es.eigenvectors();
    G = V * ev.cwiseAbs().asDiagonal() * V.transpose();
    Htt = G.topLeftCorner(nth, nth);
    Htb = G.topRightCorner(nth, d);
    HbbP = G.bottomRightCorner(d, d);
    return true;
}

// Damped joint Newton on F(theta, b) at fixed precision P.
LatentMode em_latent_mode(REGroupOracle& orc, Eigen::VectorXd theta,
                          std::vector<Eigen::VectorXd> b, const Eigen::MatrixXd& P,
                          const Eigen::VectorXd& prec, int newton_max, int n_threads) {
    const int ng = orc.n_groups, d = orc.d, nth = orc.n_theta;
    LatentMode out;
    out.C.resize(ng); out.X.resize(ng);
    double Fcur = em_joint_F(orc, theta, b, P, prec, n_threads);
    if (!std::isfinite(Fcur)) { out.ok = false; out.theta = theta; out.b = b; return out; }
    std::vector<Eigen::VectorXd> gbp(ng);
    for (int it = 0; it < newton_max; ++it) {
        orc.rebind(theta.data());
        const std::vector<GroupJoint> J = em_joint_blocks(orc, theta, b, n_threads);
        Eigen::MatrixXd A = prec.asDiagonal();
        Eigen::VectorXd gf = -prec.cwiseProduct(theta);
        for (int g = 0; g < ng; ++g) {
            Eigen::MatrixXd Htt = J[g].Htt, Htb = J[g].Htb;
            Eigen::MatrixXd Cg_inv = J[g].Hbb + P;
            em_group_psd(Htt, Htb, Cg_inv);
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Cg_inv);
            out.C[g] = lu.isInvertible() ? Eigen::MatrixXd(lu.inverse())
                       : Eigen::MatrixXd(Eigen::CompleteOrthogonalDecomposition<
                             Eigen::MatrixXd>(Cg_inv).pseudoInverse());
            out.X[g] = Htb;
            gbp[g] = J[g].gb - P * b[g];
            A += Htt;
            gf += J[g].gt;
        }
        Eigen::MatrixXd S = A;
        Eigen::VectorXd rhs = gf;
        for (int g = 0; g < ng; ++g) {
            const Eigen::MatrixXd M = out.X[g] * out.C[g];
            S -= M * out.X[g].transpose();
            rhs -= M * gbp[g];
        }
        out.S = S;
        bool ok = false;
        double step = 1.0, lambda = 0.0;
        Eigen::VectorXd du, tn;
        std::vector<Eigen::VectorXd> db(ng), bn(ng);
        double Fn = Fcur;
        for (int damp = 0; damp <= 6 && !ok; ++damp) {
            Eigen::MatrixXd Sd = S;
            if (lambda > 0.0) Sd.diagonal().array() += lambda;
            Eigen::FullPivLU<Eigen::MatrixXd> lu(Sd);
            if (lu.isInvertible()) {
                du = lu.solve(rhs);
            } else {
                Sd.diagonal().array() += std::max(lambda, 1e-6);
                du = Eigen::FullPivLU<Eigen::MatrixXd>(Sd).solve(rhs);
            }
            for (int g = 0; g < ng; ++g)
                db[g] = out.C[g] * (gbp[g] - out.X[g].transpose() * du);
            step = 1.0;
            for (int ls = 0; ls < 25; ++ls) {
                tn = theta + step * du;
                for (int g = 0; g < ng; ++g) bn[g] = b[g] + step * db[g];
                Fn = em_joint_F(orc, tn, bn, P, prec, n_threads);
                if (std::isfinite(Fn) && Fn >= Fcur - 1e-8) { ok = true; break; }
                step *= 0.5;
            }
            if (!ok) lambda = (lambda == 0.0) ? 1e-4 : lambda * 10.0;
        }
        if (!ok) { out.stuck = true; break; }
        double delta = (step * du).cwiseAbs().maxCoeff();
        for (int g = 0; g < ng; ++g)
            delta = std::max(delta, (step * db[g]).cwiseAbs().maxCoeff());
        if (nth == 0 && d == 0) delta = 0.0;
        theta = tn; b = bn; Fcur = Fn;
        if (delta < 1e-7) break;
    }
    orc.rebind(theta.data());
    out.theta = theta; out.b = b; out.F = Fcur;
    return out;
}

// Laplace log-marginal of a latent-rule mode at covariance factor sf, with
// the theta prior.
double em_latent_logml(const LatentMode& m, const AghqSigmaFactor& sf,
                       const Eigen::VectorXd& prec) {
    const int ng = (int)m.b.size();
    double acc = m.F;   // sum_g [ell_g - b'Pb/2] - theta'D theta/2
    for (int g = 0; g < ng; ++g) acc += 0.5 * em_logdet(m.C[g]);
    return acc - 0.5 * ng * sf.logdet;
}

std::unique_ptr<tulpa_progress::GridProgress> em_progress(const List& pg, int total) {
    const bool on = pg.containsElementNamed("progress") && as<bool>(pg["progress"]);
    const std::string file = pg.containsElementNamed("file") ? as<std::string>(pg["file"]) : "";
    if (!on && file.empty()) return nullptr;
    const int every = pg.containsElementNamed("every") ? as<int>(pg["every"]) : 0;
    const double thr = pg.containsElementNamed("throttle") ? as<double>(pg["throttle"]) : 2.0;
    const std::string label = pg.containsElementNamed("label") ? as<std::string>(pg["label"]) : "re-em";
    return std::unique_ptr<tulpa_progress::GridProgress>(
        new tulpa_progress::GridProgress(label, total, every, thr, file, on, "iter"));
}

}  // namespace

// [[Rcpp::export]]
List cpp_re_em(SEXP oracle, NumericVector theta0, List Sigma0,
               IntegerVector nc, LogicalVector full,
               NumericVector theta_prior_prec, double sigma_floor, double tol,
               int max_iter, int theta_iter, int n_threads,
               int theta_mode, SEXP b0, int newton_max, List progress,
               int aghq_debias) {
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
    if ((int)theta_prior_prec.size() != nth)
        stop("cpp_re_em: theta_prior_prec has length %d, theta %d.",
             (int)theta_prior_prec.size(), nth);
    const bool latent = theta_mode == 1;
    {
        // An R-closure oracle learns which closures it carries when bound.
        std::vector<double> t0(theta0.begin(), theta0.end());
        orc->rebind(t0.data());
    }
    if (latent && nth > 0 && !orc->has_theta_score())
        stop("cpp_re_em: the latent theta update needs the oracle's theta score.");
    std::vector<Eigen::MatrixXd> S(nc.size());
    for (int m = 0; m < nc.size(); ++m) {
        NumericMatrix Sm = Sigma0[m];
        S[m] = Eigen::Map<Eigen::MatrixXd>(REAL(Sm), nc[m], nc[m]);
    }
    Eigen::VectorXd theta(nth), prec(nth);
    for (int i = 0; i < nth; ++i) { theta(i) = theta0[i]; prec(i) = theta_prior_prec[i]; }
    auto prior = [&](const Eigen::VectorXd& t) { return -0.5 * t.dot(prec.cwiseProduct(t)); };

    // Optional warm start for the group modes (n_groups x d).
    EmSweep warm0;
    const bool has_b0 = !Rf_isNull(b0);
    if (has_b0) {
        NumericMatrix B0(b0);
        if (B0.nrow() != ng || B0.ncol() != d)
            stop("cpp_re_em: b0 is %d x %d, expected %d x %d.", B0.nrow(), B0.ncol(), ng, d);
        warm0.b.resize(ng); warm0.ok.assign(ng, 1);
        for (int g = 0; g < ng; ++g) {
            warm0.b[g].resize(d);
            for (int j = 0; j < d; ++j) warm0.b[g](j) = B0(g, j);
        }
    }

    auto factor = [&]() { return aghq_sigma_factor(em_block_diag(S, bl), d); };
    AghqSigmaFactor sf = factor();
    if (!sf.ok) stop("cpp_re_em: the starting covariance is not positive definite.");
    std::unique_ptr<tulpa_progress::GridProgress> prog = em_progress(progress, max_iter);

    std::vector<double> trace;
    bool converged = false, stuck = false;
    int it = 0;
    EmSweep sw;
    Eigen::MatrixXd Hth = Eigen::MatrixXd::Zero(nth, nth);
    bool hess_ok = false, debias_ok = false;
    std::vector<Eigen::MatrixXd> Xout;

    if (latent) {
        std::vector<Eigen::VectorXd> b(ng);
        for (int g = 0; g < ng; ++g)
            b[g] = has_b0 ? warm0.b[g] : Eigen::VectorXd(Eigen::VectorXd::Zero(d));
        LatentMode lm;
        double Lprev = -std::numeric_limits<double>::infinity();
        for (it = 1; it <= max_iter; ++it) {
            lm = em_latent_mode(*orc, theta, b, sf.P, prec, newton_max, n_threads);
            if (!lm.ok) break;
            theta = lm.theta; b = lm.b;
            const double L = em_latent_logml(lm, sf, prec);
            trace.push_back(L);
            em_sigma_update(S, bl, lm.b, lm.C, sigma_floor);
            sf = factor();
            if (prog) prog->tick();
            if (!sf.ok) break;
            const double rel = std::abs(L - Lprev) / (std::abs(Lprev) + 1.0);
            if (it > 1 && rel < tol && !lm.stuck) { converged = true; break; }
            Lprev = L;
        }
        if (prog) prog->finish();
        if (!sf.ok || !lm.ok) return List::create(_["ok"] = false, _["reason"] = "start");
        // Final mode at the converged covariance.
        lm = em_latent_mode(*orc, theta, b, sf.P, prec, newton_max, n_threads);
        if (!lm.ok) return List::create(_["ok"] = false, _["reason"] = "start");
        stuck = lm.stuck;
        if (stuck) converged = false;
        theta = lm.theta;
        sw.b = lm.b; sw.C = lm.C; sw.ok.assign(ng, 1); sw.all_ok = true;
        sw.L = em_latent_logml(lm, sf, prec) - prior(theta);
        Hth = 0.5 * (lm.S + lm.S.transpose());
        Xout = lm.X;
        if (aghq_debias > 1) {
            debias_ok = em_aghq_debias(*orc, bl, aghq_debias, sigma_floor, sw.b, sw.C,
                                       Xout, S, Hth, sf);
            Hth = 0.5 * (Hth + Hth.transpose());
        }
        hess_ok = Hth.allFinite();
    } else {
        orc->rebind(theta.data());
        sw = em_sweep(*orc, sf, d, n_threads, has_b0 ? &warm0 : nullptr);
        if (!sw.all_ok)
            return List::create(_["ok"] = false, _["reason"] = "start");
        trace.push_back(sw.L + prior(theta));
        for (it = 1; it <= max_iter; ++it) {
            em_sigma_update(S, bl, sw.b, sw.C, sigma_floor);
            sf = factor();
            if (!sf.ok) break;
            sw = em_sweep(*orc, sf, d, n_threads, &sw);
            if (!sw.all_ok) break;

            // BFGS on theta at this Sigma, started from the identity scaled by
            // the first gradient.
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
            if (prog) prog->tick();
            if (std::abs(Lnew - Lold) < tol * (1.0 + std::abs(Lold))) { converged = true; break; }
        }
        if (prog) prog->finish();

        // Fixed-parameter covariance: inverse of the theta curvature of L at the
        // final Sigma, the central-difference Jacobian of the envelope gradient.
        hess_ok = sw.all_ok;
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
    }

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
    List out = List::create(_["ok"] = true, _["theta_hessian"] = TH,
                            _["theta_hessian_ok"] = hess_ok,
                            _["log_marginal"] = sw.all_ok ? sw.L : NA_REAL,
                            _["trace"] = wrap(trace), _["bhat"] = BHAT, _["bcov"] = BCOV,
                            _["group_ok"] = GOK, _["converged"] = converged,
                            _["stuck"] = stuck, _["n_iter"] = std::min(it, max_iter));
    out["aghq_debias_ok"] = debias_ok;
    NumericVector th(nth);
    for (int i = 0; i < nth; ++i) th(i) = theta(i);
    out["theta"] = th;
    out["Sigma_list"] = Sout;
    if (latent) {
        NumericVector BX((size_t)ng * nth * d);
        BX.attr("dim") = IntegerVector::create(ng, nth, d);
        for (int g = 0; g < ng; ++g)
            for (int i = 0; i < nth; ++i)
                for (int j = 0; j < d; ++j)
                    BX(g + (size_t)ng * (i + (size_t)nth * j)) = Xout[g](i, j);
        out["bcross"] = BX;
    }
    return out;
}

// The shared covariance update (tulpa/re_cov_update.h) for an R caller.
// [[Rcpp::export]]
NumericMatrix cpp_re_cov_update(NumericMatrix M, bool full, double floor_sd) {
    const int c = M.nrow();
    if (M.ncol() != c) stop("cpp_re_cov_update: M must be square.");
    const Eigen::MatrixXd S = re_cov_update(
        Eigen::Map<Eigen::MatrixXd>(REAL(M), c, c), full, floor_sd);
    NumericMatrix out(c, c);
    for (int i = 0; i < c; ++i) for (int j = 0; j < c; ++j) out(i, j) = S(i, j);
    return out;
}
