// latent_factor_marginal.h
//
// Laplace marginalisation of latent-factor scores inside the joint inner solve.
//
// A latent-factor block contributes eta_i += sum_f u_{l(i), f} lambda_{k(i), f}
// (one block per factor f, all blocks reading the same unit l(i)). The scores
// u_l in R^F are many (one vector per unit) and each is informed only by the
// rows of its own unit, so their posterior spread is a fixed fraction of their
// prior spread however many units there are. The joint mode over (u, theta) --
// theta being every other latent coordinate (loadings, fixed effects, random
// effects, other blocks) -- ignores that spread: given the loadings it
// regresses each arm on the shrunk modal scores, which inflates the loadings
// by the ratio of the score variance to the variance of the score modes. The
// bias does not shrink as units are added (the number of scores grows with
// them), and without an anchor on a loading the joint mode has no finite
// scale at all.
//
// The target is instead the posterior of theta with the scores integrated out:
//
//   g(theta) = f(u_hat(theta), theta) - 1/2 sum_l log det H_l(u_hat(theta), theta)
//
// with f the log joint, u_hat(theta) the conditional score modes and H_l the
// F x F negative Hessian of f in the scores of unit l. H_l is block-diagonal
// across units because no row reads two units, so the integral factorises
// into one F-dimensional Laplace integral per unit.
//
// The inner Newton maximises f(x) + c' x with the score coordinates of c zero:
// its stationary point solves f_u = 0 (the conditional score modes) and
// f_theta + c = 0. Setting c to the gradient of the log-determinant term,
//
//   c_j = -1/2 sum_l tr(H_l^-1 dH_l / dtheta_j),
//
// where the derivative is TOTAL, through u_hat(theta) as well, makes the
// stationary point the mode of g. c depends on x, so the solve is repeated
// with c re-evaluated at its mode until the mode stops moving.
//
// The theta precision of the score-marginal posterior is -d^2 g, the Schur
// complement of the observed Hessian H minus D = dc / dtheta (the curvature of
// the log-determinant term). The final solve carries D on the fixed effects
// of the reached arms and the loadings (factor_curvature_coords), so the
// Gaussian N(x_hat, P^-1), P = H - D on that block, has the Laplace
// conditional of the scores and that theta precision, and the log marginal
// f - 1/2 log det P the inner Laplace reports is the Laplace approximation of
// the score-marginal evidence. D is left out along the coordinates of other
// latent blocks and random effects, where the log-determinant depends on
// theta only through the rows' third derivatives.
//
// Per row i of arm k reading unit l, with a_i in R^F the row's score loadings
// (a_{i,f} = lambda_{k,f} d_f), W_i = -l''(eta_i), W'_i = dW_i / deta = -l'''(eta_i)
// and l'_i the score:
//
//   H_l   = I + sum_i W_i a_i a_i'           M_l = H_l^-1
//   q_i   = a_i' M_l a_i
//   t_l   = sum_i W'_i q_i a_i               v_l = M_l t_l
//   s_i   = -1/2 (W'_i q_i - W_i v_l' a_i)
//
//   c    += s_i * d eta_i / d theta               (every non-score coordinate)
//   c[lambda_{k,f}] += -1/2 d_f (2 W_i (M_l a_i)_f + v_{l,f} l'_i)
//
// The first term carries the dependence of H_l on eta_i and of u_hat on
// theta through the row; the second the direct dependence of a_i on its
// loading. d eta_i / d lambda_{k,f} = u_{l,f} d_f is part of the first term.

#ifndef TULPA_LATENT_FACTOR_MARGINAL_H
#define TULPA_LATENT_FACTOR_MARGINAL_H

#include "inner_laplace_skew.h"
#include "joint_hessian_pattern.h"
#include "laplace_core.h"
#include "laplace_newton_joint.h"
#include "latent_block.h"
#include "nested_laplace_joint_core.h"
#include "sparse_hessian.h"
#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace tulpa {

// Ceiling on the tilt fixed-point rounds of one inner solve. Each round moves
// the mode by a fixed fraction of the previous move (the fraction of the
// loadings' information the scores' spread removes), so a solve that has not
// settled well inside this many rounds is not converging.
constexpr int kFactorMarginalMaxRounds = 500;

// The latent-factor blocks of a block list, in block order: entry f is the
// block of factor f.
inline std::vector<int> factor_block_indices(const std::vector<LatentBlock>& blocks) {
    std::vector<int> out;
    for (int b = 0; b < static_cast<int>(blocks.size()); b++)
        if (blocks[b].contrib_kind == BlockContribKind::BILINEAR_FACTOR)
            out.push_back(b);
    return out;
}

// Reads the factor blocks' slots of one row. `u_slot[f] < 0` where factor f
// does not reach the row. Returns the row's unit (0-based) or -1.
struct FactorRow {
    std::vector<int>    u_slot, lambda_slot;
    std::vector<double> d;
    int unit = -1;
};

inline void read_factor_row(int i, int k_arm,
                            const std::vector<LatentBlock>& blocks,
                            const std::vector<int>& fb,
                            const std::vector<std::vector<double>>& d_eff,
                            FactorRow& r) {
    const int F = static_cast<int>(fb.size());
    r.u_slot.assign(F, -1);
    r.lambda_slot.assign(F, -1);
    r.d.assign(F, 0.0);
    r.unit = -1;
    for (int f = 0; f < F; f++) {
        const LatentBlock& blk = blocks[fb[f]];
        const auto s = blk.obs_factor_lambda(i, k_arm);
        if (s.first < 0 || s.second < 0) continue;
        const double d = d_eff[fb[f]][k_arm] * block_row_weight(blk, i, k_arm);
        if (d == 0.0) continue;
        r.u_slot[f] = s.first;
        r.lambda_slot[f] = s.second;
        r.d[f] = d;
        if (r.unit < 0) r.unit = s.first - blk.start;
    }
}

// Observed-curvature term of the bilinear product: d^2 eta / du dlambda = d,
// so the negative Hessian gains -l'_i d at (u, lambda). The Gauss-Newton
// scatter carries only the W_i a a' part; the Laplace approximation at the mode
// needs the observed matrix.
inline void add_factor_observed_cross(
    const Rcpp::NumericVector&               x,
    const std::vector<Rcpp::NumericVector>&  etas,
    const std::vector<JointArm>&             arms,
    const std::vector<ArmSpecView>&          views,
    const std::vector<LatentBlock>&          blocks,
    const std::vector<int>&                  fb,
    const std::vector<std::vector<double>>&  d_eff,
    const std::vector<bool>&                 skip_arm,
    SparseHessianBuilder&                    H
) {
    (void) x;
    FactorRow r;
    const int F = static_cast<int>(fb.size());
    for (int k = 0; k < static_cast<int>(arms.size()); k++) {
        if (k < static_cast<int>(skip_arm.size()) && skip_arm[k]) continue;
        for (int i = 0; i < arms[k].N; i++) {
            read_factor_row(i, k, blocks, fb, d_eff, r);
            if (r.unit < 0) continue;
            const GradHess gh = arm_grad_hess(views[k], i, etas[k][i]);
            for (int f = 0; f < F; f++) {
                if (r.u_slot[f] < 0) continue;
                H.add(r.u_slot[f], r.lambda_slot[f], -gh.grad * r.d[f]);
            }
        }
    }
}

// Per-row likelihood curvature of the factor rows and the per-unit inverse
// score precisions M_l = H_l^-1, H_l = I + sum_i W_i a_i a_i'. `l3` null skips
// the third derivative (Wp left 0).
struct FactorRowVals { double W, Wp, lp; };
struct FactorUnitState {
    std::vector<std::vector<FactorRowVals>> vals;
    std::vector<double>                     M;
};

inline bool factor_unit_state(
    const Rcpp::NumericVector&               x,
    const std::vector<Rcpp::NumericVector>&  etas,
    const std::vector<JointArm>&             arms,
    const std::vector<ArmSpecView>&          views,
    const std::vector<Curvature3Oracle>*     l3,
    const std::vector<LatentBlock>&          blocks,
    const std::vector<int>&                  fb,
    const std::vector<std::vector<double>>&  d_eff,
    const std::vector<bool>&                 skip_arm,
    FactorUnitState&                         st
) {
    const int F  = static_cast<int>(fb.size());
    const int S  = blocks[fb[0]].factor_n_latent;
    const int nA = static_cast<int>(arms.size());
    st.vals.assign(nA, {});
    std::vector<double> Hs(static_cast<std::size_t>(S) * F * F, 0.0);
    for (int l = 0; l < S; l++)
        for (int f = 0; f < F; f++) Hs[(static_cast<std::size_t>(l) * F + f) * F + f] = 1.0;
    FactorRow r;
    std::vector<double> a(F);
    for (int k = 0; k < nA; k++) {
        if (k < static_cast<int>(skip_arm.size()) && skip_arm[k]) continue;
        st.vals[k].assign(arms[k].N, FactorRowVals{0.0, 0.0, 0.0});
        for (int i = 0; i < arms[k].N; i++) {
            read_factor_row(i, k, blocks, fb, d_eff, r);
            if (r.unit < 0) continue;
            const double eta = etas[k][i];
            const GradHess gh = arm_grad_hess(views[k], i, eta);
            double l3v = 0.0;
            if (l3) {
                l3v = (*l3)[k].scalar ? (*l3)[k].scalar(i, eta)
                                      : std::numeric_limits<double>::quiet_NaN();
            }
            if (!std::isfinite(gh.grad) || !std::isfinite(gh.neg_hess) ||
                !std::isfinite(l3v)) return false;
            st.vals[k][i] = FactorRowVals{gh.neg_hess, -l3v, gh.grad};
            for (int f = 0; f < F; f++)
                a[f] = r.u_slot[f] < 0 ? 0.0 : x[r.lambda_slot[f]] * r.d[f];
            double* Hl = &Hs[static_cast<std::size_t>(r.unit) * F * F];
            for (int f = 0; f < F; f++)
                for (int g = 0; g < F; g++) Hl[f * F + g] += gh.neg_hess * a[f] * a[g];
        }
    }
    st.M.assign(Hs.size(), 0.0);
    for (int l = 0; l < S; l++) {
        Eigen::Map<const Eigen::MatrixXd> Hl(&Hs[static_cast<std::size_t>(l) * F * F], F, F);
        Eigen::LLT<Eigen::MatrixXd> llt(Hl);
        if (llt.info() != Eigen::Success) return false;
        Eigen::Map<Eigen::MatrixXd> Ml(&st.M[static_cast<std::size_t>(l) * F * F], F, F);
        Ml = llt.solve(Eigen::MatrixXd::Identity(F, F));
    }
    return true;
}

// The tilt c(x) described above, written into `tilt` (length n_x, zero on the
// score coordinates). Returns false when a likelihood value, curvature or
// third derivative it needs is not finite.
inline bool factor_score_marginal_tilt(
    const Rcpp::NumericVector&               x,
    const std::vector<Rcpp::NumericVector>&  etas,
    const std::vector<JointArm>&             arms,
    const std::vector<ParsedArm>&            parsed,
    const std::vector<ArmSpecView>&          views,
    const std::vector<Curvature3Oracle>&     l3,
    const std::vector<LatentBlock>&          blocks,
    const std::vector<int>&                  fb,
    const std::vector<std::vector<double>>&  d_eff,
    const std::vector<bool>&                 skip_arm,
    int                                      k_grid,
    std::vector<double>&                     tilt
) {
    const int F  = static_cast<int>(fb.size());
    const int S  = blocks[fb[0]].factor_n_latent;
    const int nA = static_cast<int>(arms.size());
    const int B  = static_cast<int>(blocks.size());
    std::fill(tilt.begin(), tilt.end(), 0.0);

    FactorUnitState st;
    if (!factor_unit_state(x, etas, arms, views, &l3, blocks, fb, d_eff,
                           skip_arm, st)) return false;
    const auto& vals = st.vals;
    const auto& Ms   = st.M;
    FactorRow r;
    std::vector<double> a(F);
    auto row_loadings = [&](const FactorRow& rr) {
        for (int f = 0; f < F; f++)
            a[f] = rr.u_slot[f] < 0 ? 0.0 : x[rr.lambda_slot[f]] * rr.d[f];
    };
    // Pass 2: t_l = sum_i W'_i q_i a_i.
    std::vector<double> t(static_cast<std::size_t>(S) * F, 0.0);
    std::vector<double> Ma(F);
    auto apply_M = [&](int l) {
        const double* Ml = &Ms[static_cast<std::size_t>(l) * F * F];
        for (int f = 0; f < F; f++) {
            double s = 0.0;
            for (int g = 0; g < F; g++) s += Ml[f * F + g] * a[g];
            Ma[f] = s;
        }
    };
    for (int k = 0; k < nA; k++) {
        if (k < static_cast<int>(skip_arm.size()) && skip_arm[k]) continue;
        for (int i = 0; i < arms[k].N; i++) {
            read_factor_row(i, k, blocks, fb, d_eff, r);
            if (r.unit < 0) continue;
            row_loadings(r);
            apply_M(r.unit);
            double q = 0.0;
            for (int f = 0; f < F; f++) q += a[f] * Ma[f];
            for (int f = 0; f < F; f++)
                t[static_cast<std::size_t>(r.unit) * F + f] += vals[k][i].Wp * q * a[f];
        }
    }
    std::vector<double> v(t.size(), 0.0);
    for (int l = 0; l < S; l++) {
        const double* Ml = &Ms[static_cast<std::size_t>(l) * F * F];
        for (int f = 0; f < F; f++) {
            double s = 0.0;
            for (int g = 0; g < F; g++) s += Ml[f * F + g] * t[static_cast<std::size_t>(l) * F + g];
            v[static_cast<std::size_t>(l) * F + f] = s;
        }
    }

    // Pass 3: scatter s_i along the row's Jacobian over the non-score
    // coordinates, plus the loadings' direct term.
    std::vector<double> d_row(B);
    std::vector<std::pair<int,double>> multi_scratch;
    std::vector<double> basis_scratch;
    std::vector<char> is_score_block(B, 0);
    for (int f = 0; f < F; f++) is_score_block[fb[f]] = 1;
    for (int k = 0; k < nA; k++) {
        if (k < static_cast<int>(skip_arm.size()) && skip_arm[k]) continue;
        const ParsedArm& pa = parsed[k];
        for (int b = 0; b < B; b++) d_row[b] = d_eff[b][k];
        for (int i = 0; i < arms[k].N; i++) {
            read_factor_row(i, k, blocks, fb, d_eff, r);
            if (r.unit < 0) continue;
            const FactorRowVals& rv = vals[k][i];
            row_loadings(r);
            apply_M(r.unit);
            const double* vl = &v[static_cast<std::size_t>(r.unit) * F];
            double q = 0.0, va = 0.0;
            for (int f = 0; f < F; f++) { q += a[f] * Ma[f]; va += vl[f] * a[f]; }
            const double s = -0.5 * (rv.Wp * q - rv.W * va);

            for (int j = 0; j < pa.p; j++) tilt[pa.beta_start + j] += s * pa.X(i, j);
            if (pa.n_re_groups > 0) {
                const int g = static_cast<int>(pa.re_idx[i]) - 1;
                if (g >= 0 && g < pa.n_re_groups) tilt[pa.re_start + g] += s;
            }
            for_each_row_block_latent(
                i, k, k_grid, blocks, d_row, x.begin(), multi_scratch,
                basis_scratch,
                [](int) { return false; },
                [&](int b, int latent, double w) {
                    if (is_score_block[b] && latent < blocks[b].start +
                                                      blocks[b].factor_n_latent)
                        return;
                    tilt[latent] += s * w;
                });
            for (int f = 0; f < F; f++) {
                if (r.u_slot[f] < 0) continue;
                tilt[r.lambda_slot[f]] += -0.5 * r.d[f] *
                    (2.0 * rv.W * Ma[f] + vl[f] * rv.lp);
            }
        }
    }
    for (double c : tilt) if (!std::isfinite(c)) return false;
    return true;
}

// Curvature of the score log-determinant along `coords`
// (factor_curvature_coords): D = dc / dtheta, the TOTAL derivative of the tilt,
// so the scores move with theta along du_hat / dtheta = -H_uu^-1 H_u,theta. The
// theta precision of the score-marginal posterior is the Schur complement of
// the observed Hessian minus D; the joint precision carrying it is the
// observed Hessian with D subtracted from its (coords x coords) block. Column
// a is a central difference of the tilt along e_{coords[a]} + du_hat, one
// pair of tilt evaluations per coordinate. Written symmetrised, row-major,
// |coords| x |coords|. Returns false when a tilt evaluation does.
template <typename TiltFn>
inline bool factor_score_marginal_curvature(
    const Rcpp::NumericVector&               x,
    const std::vector<Rcpp::NumericVector>&  etas,
    const std::vector<JointArm>&             arms,
    const std::vector<ParsedArm>&            parsed,
    const std::vector<ArmSpecView>&          views,
    const std::vector<LatentBlock>&          blocks,
    const std::vector<int>&                  fb,
    const std::vector<std::vector<double>>&  d_eff,
    const std::vector<bool>&                 skip_arm,
    const std::vector<int>&                  coords,
    TiltFn&&                                 tilt_at,
    std::vector<double>&                     D
) {
    const int F  = static_cast<int>(fb.size());
    const int S  = blocks[fb[0]].factor_n_latent;
    const int nA = static_cast<int>(arms.size());
    const int nc = static_cast<int>(coords.size());
    const int n_x = static_cast<int>(x.size());
    FactorUnitState st;
    if (!factor_unit_state(x, etas, arms, views, nullptr, blocks, fb, d_eff,
                           skip_arm, st)) return false;
    std::vector<int> pos(n_x, -1);
    for (int a = 0; a < nc; a++) pos[coords[a]] = a;

    // h[(a * S + l) * F + f] = H_{u_{l,f}, coords[a]} (negative Hessian).
    std::vector<double> h(static_cast<std::size_t>(nc) * S * F, 0.0);
    FactorRow r;
    std::vector<double> av(F);
    auto add_h = [&](int a, int l, double w_row, double J) {
        double* hl = &h[(static_cast<std::size_t>(a) * S + l) * F];
        for (int f = 0; f < F; f++) hl[f] += w_row * av[f] * J;
    };
    for (int k = 0; k < nA; k++) {
        if (k < static_cast<int>(skip_arm.size()) && skip_arm[k]) continue;
        const ParsedArm& pa = parsed[k];
        for (int i = 0; i < arms[k].N; i++) {
            read_factor_row(i, k, blocks, fb, d_eff, r);
            if (r.unit < 0) continue;
            const FactorRowVals& rv = st.vals[k][i];
            for (int f = 0; f < F; f++)
                av[f] = r.u_slot[f] < 0 ? 0.0 : x[r.lambda_slot[f]] * r.d[f];
            for (int j = 0; j < pa.p; j++) {
                const int a = pos[pa.beta_start + j];
                if (a >= 0) add_h(a, r.unit, rv.W, pa.X(i, j));
            }
            for (int f = 0; f < F; f++) {
                if (r.u_slot[f] < 0) continue;
                const int a = pos[r.lambda_slot[f]];
                if (a < 0) continue;
                add_h(a, r.unit, rv.W, x[r.u_slot[f]] * r.d[f]);
                h[(static_cast<std::size_t>(a) * S + r.unit) * F + f] -= rv.lp * r.d[f];
            }
        }
    }

    constexpr double kStep = 1e-5;
    D.assign(static_cast<std::size_t>(nc) * nc, 0.0);
    std::vector<double> xp(x.begin(), x.end()), xm(x.begin(), x.end());
    std::vector<double> cp(n_x), cm(n_x), du(static_cast<std::size_t>(S) * F);
    for (int a = 0; a < nc; a++) {
        for (int l = 0; l < S; l++) {
            const double* Ml = &st.M[static_cast<std::size_t>(l) * F * F];
            const double* hl = &h[(static_cast<std::size_t>(a) * S + l) * F];
            for (int f = 0; f < F; f++) {
                double s = 0.0;
                for (int g = 0; g < F; g++) s += Ml[f * F + g] * hl[g];
                du[static_cast<std::size_t>(l) * F + f] = -s;
            }
        }
        std::copy(x.begin(), x.end(), xp.begin());
        std::copy(x.begin(), x.end(), xm.begin());
        xp[coords[a]] += kStep;
        xm[coords[a]] -= kStep;
        for (int f = 0; f < F; f++) {
            const int u0 = blocks[fb[f]].start;
            for (int l = 0; l < S; l++) {
                const double d = kStep * du[static_cast<std::size_t>(l) * F + f];
                xp[u0 + l] += d;
                xm[u0 + l] -= d;
            }
        }
        if (!tilt_at(xp, cp) || !tilt_at(xm, cm)) return false;
        for (int b = 0; b < nc; b++)
            D[static_cast<std::size_t>(b) * nc + a] =
                (cp[coords[b]] - cm[coords[b]]) / (2.0 * kStep);
    }
    for (int a = 0; a < nc; a++)
        for (int b = 0; b < a; b++) {
            const double s = 0.5 * (D[static_cast<std::size_t>(a) * nc + b] +
                                    D[static_cast<std::size_t>(b) * nc + a]);
            D[static_cast<std::size_t>(a) * nc + b] = s;
            D[static_cast<std::size_t>(b) * nc + a] = s;
        }
    return true;
}
// Orient every factor so its first arm's loading is positive, at the first
// row of that arm reading the factor. (u_f, lambda_f) -> (-u_f, -lambda_f)
// leaves eta, the log joint and the score-marginal objective unchanged.
// Returns true when a factor was flipped.
inline bool orient_factor_blocks(std::vector<double>& x,
                                 const std::vector<LatentBlock>& blocks,
                                 const std::vector<int>& fb) {
    bool flipped = false;
    for (int b : fb) {
        const LatentBlock& blk = blocks[b];
        const int lam0 = blk.start + blk.factor_n_latent;
        if (x[lam0] >= 0.0) continue;
        for (int j = blk.start; j < blk.start + blk.size; j++) x[j] = -x[j];
        flipped = true;
    }
    return flipped;
}

// A start for a factor whose loadings are all zero: u = 0, lambda = 0 is a
// stationary point of the log joint (every score and loading gradient carries
// the other factor of the product), so the solve would not leave it. The
// factor's first arm takes loading 1, which makes the score gradient the
// residual of that arm; distinct factors start from distinct arms.
inline void start_factor_blocks(std::vector<double>& x,
                                const std::vector<LatentBlock>& blocks,
                                const std::vector<int>& fb) {
    for (int b : fb) {
        const LatentBlock& blk = blocks[b];
        const int lam0 = blk.start + blk.factor_n_latent;
        bool all_zero = true;
        for (int j = lam0; j < blk.start + blk.size; j++)
            if (x[j] != 0.0) { all_zero = false; break; }
        if (all_zero) x[lam0] = 1.0;
    }
}

// Proximal term -rho / 2 |x - center|^2 the tilted inner objective carries
// while the fixed point is sought; rho = 0 is the plain tilted objective.
// The term vanishes at a fixed point (the next mode equals the centre), so it
// changes how the iteration reaches the score-marginal mode, not the mode.
struct FactorProx {
    double              rho = 0.0;
    std::vector<double> center;
};

// Repeats `solve(warm, full)` -- an inner Laplace solve of
// f + tilt' x - prox.rho / 2 |x - prox.center|^2 reading the caller's `tilt`
// and `prox` -- with the tilt re-evaluated at each mode, until the oriented
// mode moves by no more than `tol` in any coordinate.
//
// The plain iteration (rho = 0) takes the maximiser of the tilted objective
// as the next point. Where the log joint is nearly flat along a direction the
// tilt pushes on (the scale and the mixing of several factors are identified
// by the score prior alone), that maximiser can lie far from the current
// point, outside the region where the tilt evaluated here describes the
// score-marginal objective: the iteration oscillates with growing steps, or
// the inner solve fails. A round whose solve does not converge, or whose step
// is longer than the last, is redone from the same point with the proximal
// weight raised tenfold (from 1), and the weight is kept from then on. The
// proximal iteration solves (H + rho I) dx = grad g where the plain one
// solves H dx = grad g, so a large enough rho damps the directions in which H
// understates the curvature of g. Then one full solve at rho = 0 from the fixed point
// carries the read-outs. The returned log marginal has the tilt term removed,
// so it is the Laplace log marginal of the model; `converged` also requires
// the fixed point.
template <typename SolveFn, typename TiltFn>
inline LaplaceResult solve_factor_score_marginal(
    SolveFn&&                         solve,
    TiltFn&&                          tilt_at,
    std::vector<double>&              tilt,
    FactorProx&                       prox,
    const std::vector<double>&        prev_mode,
    int                               n_x,
    const std::vector<LatentBlock>&   blocks,
    const std::vector<int>&           fb,
    double                            tol,
    int                               max_rounds
) {
    std::vector<double> warm = (static_cast<int>(prev_mode.size()) == n_x)
                               ? prev_mode : std::vector<double>(n_x, 0.0);
    start_factor_blocks(warm, blocks, fb);
    orient_factor_blocks(warm, blocks, fb);
    std::fill(tilt.begin(), tilt.end(), 0.0);
    std::vector<double> tilt_next(n_x, 0.0);
    prox.rho = 0.0;
    prox.center = warm;

    constexpr double kRhoMax = 1e8;
    LaplaceResult res;
    bool fixed_point = false, tilt_ok = true;
    double last_step = std::numeric_limits<double>::infinity();
    for (int round = 0; round < max_rounds; round++) {
        prox.center = warm;
        res = solve(warm, false);
        std::vector<double> x = res.mode;
        const bool solved = std::isfinite(res.log_marginal) && res.converged &&
                            static_cast<int>(x.size()) == n_x;
        double step = std::numeric_limits<double>::infinity();
        if (solved) {
            orient_factor_blocks(x, blocks, fb);
            step = 0.0;
            for (int j = 0; j < n_x; j++)
                step = std::max(step, std::fabs(x[j] - warm[j]));
        }
        // Round 0 runs at tilt 0 and lands on the joint mode, so its step
        // says nothing about contraction.
        if (!solved || (round > 1 && step > last_step)) {
            if (prox.rho >= kRhoMax) break;
            prox.rho = std::max(1.0, 10.0 * prox.rho);
            continue;
        }
        if (round > 0) last_step = step;
        warm.swap(x);
        if (!tilt_at(warm, tilt_next)) { tilt_ok = false; break; }
        tilt.swap(tilt_next);
        if (round > 0 && step <= tol) { fixed_point = true; break; }
    }
    prox.rho = 0.0;
    prox.center = warm;
    if (tilt_ok) res = solve(warm, true);
    if (static_cast<int>(res.mode.size()) != n_x) {
        res.converged = false;
        return res;
    }
    double tx = 0.0;
    for (int j = 0; j < n_x; j++) tx += tilt[j] * res.mode[j];
    res.log_marginal -= tx;
    res.converged = res.converged && fixed_point;
    return res;
}
} // namespace tulpa

#endif // TULPA_LATENT_FACTOR_MARGINAL_H
