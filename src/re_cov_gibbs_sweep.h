// re_cov_gibbs_sweep.h
// Oracle-driven Metropolis-within-Gibbs sweep for random-effect covariances.
//
// This is the C++ port of the per-sweep structure of `tulpa_re_cov_gibbs`
// (R/re_cov_gibbs.R). The estimator is unchanged -- it is a pure
// interpreter-speed rewire so the per-group / per-beta likelihood has a single
// source of truth (the REGroupOracle) instead of the duplicated R
// `.re_obs_loglik`.
//
// Targets the exact joint posterior p(beta, {b_m}, {Sigma_m} | y) by:
//   * beta | b, Sigma   -- random-walk Metropolis, proposal shape = Cholesky of
//                          the inverse theta-curvature (sum_g theta_obs_info,
//                          inverted) or a supplied pilot Hessian. The MH ratio
//                          is sum_g grad_hess(g, b_g).logL evaluated at the
//                          current b under the current vs proposed beta (via
//                          oracle.rebind(beta)), plus the Gaussian beta prior.
//   * b_{m,g} | beta, Sigma -- per-(block, group) random-walk Metropolis. The
//                          proposal shape is chol((negH)^-1) at the pilot mode
//                          from aghq_group_mode(...). The MH ratio is the group
//                          log-lik (grad_hess(g, b).logL) + the Gaussian RE
//                          log-prior -0.5 b' Q b.
//   * Sigma_m | b_m     -- exact conjugate inverse-Wishart per block (full =>
//                          matrix IW; diagonal => per-coordinate scalar IW),
//                          via the Bartlett sampler ported from .rinvwishart.
//
// Robbins-Monro proposal-scale adaptation runs during burn-in; thinning and
// recording mirror the R targets/decay exactly. All randomness goes through the
// package RNG (R::rnorm / R::rchisq / R::unif_rand) so results are reproducible
// from R's set.seed.
//
// Cross-block eta coupling. The R sampler holds ONE shared linear predictor and,
// when updating block m, uses the offset `base = xb + (re_total - re_contrib_m)`
// -- every other block's RE contribution plus the fixed effects, held fixed. The
// shared REGroupOracle interface (inst/include/tulpa/aghq_oracle.h) owns the
// design for ONE per-group conditional log-likelihood ell_g(b) and has no slot
// for such an external per-observation offset. To stay strictly inside the
// allowed files (the shared header must not change), the multi-block coupling is
// expressed through a thin local extension `GibbsREGroupOracle` declared HERE,
// which adds set_offset(). A single-block model never needs it (offset is just
// the fixed effects, which rebind() already owns) and can be driven through a
// plain REGroupOracle. See the deliverable note: if the shared interface is to
// support multi-block Gibbs natively, add an offset hook (e.g.
// `virtual void set_eta_offset(const double*) {}`) to REGroupOracle.

#ifndef TULPA_RE_COV_GIBBS_SWEEP_H
#define TULPA_RE_COV_GIBBS_SWEEP_H

#include "tulpa/aghq_oracle.h"
#include "aghq_re_mode.h"
#include <RcppEigen.h>
#include <Rcpp.h>          // R::rnorm, R::rchisq, R::unif_rand
#include <vector>
#include <cmath>
#include <stdexcept>

namespace tulpa {

// ---------------------------------------------------------------------------
// Optional oracle extension for multi-block coupling.
//
// A block's per-group conditional log-likelihood depends on the OTHER blocks'
// random-effect contributions only through a fixed per-observation offset
// (the R `base` vector). An oracle that participates in a multi-block sweep
// implements this to receive that offset before each block update. Single-block
// models can ignore it (use a plain REGroupOracle).
// ---------------------------------------------------------------------------
struct GibbsREGroupOracle : public REGroupOracle {
    // off has one entry per observation the oracle owns (its own indexing); the
    // sweep computes it as fixed-effects + sum of all OTHER blocks' RE
    // contributions. Default no-op so a single-block oracle need not implement it.
    virtual void set_eta_offset(const double* /*off*/) {}
};

// ---------------------------------------------------------------------------
// Native inverse-Wishart via Bartlett (port of .rinvwishart in R).
//
//   Sigma ~ IW(df, Lambda)  <=>  Sigma^{-1} ~ Wishart(df, Lambda^{-1}).
// Sample the Wishart by Bartlett: Lambda^{-1} = C C'; A lower-triangular with
// A_ii = sqrt(chisq_{df - i + 1}), A_ij ~ N(0,1) for i > j; W = (C A)(C A)';
// then Sigma = W^{-1}. Requires df > p - 1 (a 1x1 block / scalar inverse-gamma
// needs only df > 0). Uses R::rchisq / R::rnorm for reproducibility.
// ---------------------------------------------------------------------------
inline Eigen::MatrixXd rinvwishart(double df, const Eigen::MatrixXd& Lambda) {
    const int p = static_cast<int>(Lambda.rows());
    if (df <= p - 1) {
        throw std::runtime_error(
            "inverse-Wishart df must exceed p - 1.");
    }
    // V = Lambda^{-1} (SPD); C = chol-lower(V) so V = C C'.
    Eigen::LLT<Eigen::MatrixXd> lltLam(Lambda);
    if (lltLam.info() != Eigen::Success)
        throw std::runtime_error("inverse-Wishart Lambda not SPD.");
    // V = Lambda^{-1}
    Eigen::MatrixXd V = lltLam.solve(Eigen::MatrixXd::Identity(p, p));
    V = 0.5 * (V + V.transpose());
    Eigen::LLT<Eigen::MatrixXd> lltV(V);
    if (lltV.info() != Eigen::Success)
        throw std::runtime_error("inverse-Wishart V = Lambda^{-1} not SPD.");
    Eigen::MatrixXd C = lltV.matrixL();   // lower; V = C C'

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p, p);
    for (int i = 0; i < p; ++i) {
        A(i, i) = std::sqrt(R::rchisq(df - i));   // chisq_{df - (i+1) + 1} = df - i
        for (int j = 0; j < i; ++j) A(i, j) = R::rnorm(0.0, 1.0);
    }
    Eigen::MatrixXd CA = C * A;
    Eigen::MatrixXd W  = CA * CA.transpose();      // Wishart(df, V) = Sigma^{-1}
    W = 0.5 * (W + W.transpose());
    Eigen::LLT<Eigen::MatrixXd> lltW(W);
    if (lltW.info() != Eigen::Success)
        throw std::runtime_error("inverse-Wishart W not SPD.");
    Eigen::MatrixXd S = lltW.solve(Eigen::MatrixXd::Identity(p, p));
    return 0.5 * (S + S.transpose());
}

// ---------------------------------------------------------------------------
// Symmetrize + eigen-floor a proposal covariance to SPD, return lower Cholesky
// (the random-walk proposal shape). Port of .re_chol_spd in R.
// ---------------------------------------------------------------------------
inline Eigen::MatrixXd chol_spd(const Eigen::MatrixXd& S_in, double floor = 1e-8) {
    Eigen::MatrixXd S = 0.5 * (S_in + S_in.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    Eigen::VectorXd d = es.eigenvalues().cwiseMax(floor);
    Eigen::MatrixXd M = es.eigenvectors() * d.asDiagonal()
                        * es.eigenvectors().transpose();
    M = 0.5 * (M + M.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(M);
    return llt.matrixL();
}

// ---------------------------------------------------------------------------
// Per-block covariance + conjugate prior layout (mirrors .re_cov_block_layout
// + .re_gibbs_block_prior semantics).
//   full  = TRUE  -> correlated (nc x nc) inverse-Wishart, prior IW(nu0, Lambda0).
//   full  = FALSE -> diagonal: nc independent scalar inverse-Wisharts
//                    (== inverse-gamma), prior shape nu0, per-coord scales lambda0.
// ---------------------------------------------------------------------------
struct CovBlock {
    int nc = 1;
    bool full = false;
    int n_groups = 0;
    double nu0 = 0.0;             // IW df (full) or scalar IW shape (diagonal)
    Eigen::MatrixXd Lambda0;      // full: nc x nc prior scale
    Eigen::VectorXd lambda0;      // diagonal: per-coordinate prior scale
};

// Exact conjugate draw of one block's Sigma given its random effects B_m
// (G_m x nc). Port of .re_gibbs_draw_sigma.
inline Eigen::MatrixXd draw_block_sigma(const Eigen::MatrixXd& B_m,
                                        const CovBlock& bl) {
    const int G  = static_cast<int>(B_m.rows());
    const int nc = bl.nc;
    if (bl.full) {
        Eigen::MatrixXd Lam = bl.Lambda0 + B_m.transpose() * B_m;  // + sum_g b b'
        return rinvwishart(bl.nu0 + G, Lam);
    }
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nc, nc);
    for (int i = 0; i < nc; ++i) {
        Eigen::MatrixXd Lam1(1, 1);
        Lam1(0, 0) = bl.lambda0(i) + B_m.col(i).squaredNorm();
        Eigen::MatrixXd Si = rinvwishart(bl.nu0 + G, Lam1);
        S(i, i) = Si(0, 0);
    }
    return S;
}

// ---------------------------------------------------------------------------
// Sweep configuration. n_theta on the oracle is the fixed-effect dimension p
// (beta); the sweep samples beta = theta.
// ---------------------------------------------------------------------------
struct GibbsConfig {
    int n_iter   = 2000;   // recorded post-burn-in sweeps
    int n_burnin = 1000;   // burn-in sweeps (proposal-scale adaptation)
    int thin     = 1;
    Eigen::VectorXd beta_prior_mean;   // length p
    Eigen::VectorXd beta_prior_sd;     // length p
};

// ---------------------------------------------------------------------------
// Recorded output. beta_draws is (n_kept x p); Sigma_draws[k][m] is the m-th
// block's Sigma at recorded sweep k. accept_beta / accept_b are the recorded
// acceptance rates (matching res$accept in R).
// ---------------------------------------------------------------------------
struct GibbsOutput {
    int n_kept = 0;
    Eigen::MatrixXd beta_draws;                              // n_kept x p
    std::vector<std::vector<Eigen::MatrixXd>> Sigma_draws;   // [k][m]
    double accept_beta = 0.0;
    double accept_b    = 0.0;
    Eigen::VectorXd s_beta_final;   // 1 entry: final adapted beta scale
    Eigen::VectorXd s_b_final;      // per-block final adapted b scale
};

// ---------------------------------------------------------------------------
// Per-block runtime state held across the sweep.
// ---------------------------------------------------------------------------
struct BlockState {
    Eigen::MatrixXd B;                          // G_m x nc current RE values
    std::vector<Eigen::MatrixXd> L_g;           // per-group proposal Cholesky (nc x nc)
    Eigen::MatrixXd Q;                           // Sigma_m^{-1}
    double s_b;                                  // adapted RW scale
    double tgt_b;                                // target acceptance
};

// Inverse of an SPD matrix via Cholesky.
inline Eigen::MatrixXd spd_inverse(const Eigen::MatrixXd& S) {
    const int n = static_cast<int>(S.rows());
    Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (S + S.transpose()));
    return llt.solve(Eigen::MatrixXd::Identity(n, n));
}

// ---------------------------------------------------------------------------
// The driver.
//
// oracles[m] is the per-block oracle (one per covariance block). Each owns the
// per-group conditional log-likelihood ell_g(b) of its block, with the FIXED
// EFFECTS already wired through rebind(beta); the cross-block coupling offset is
// pushed via set_eta_offset() before that block's b-update. For a single block,
// a plain REGroupOracle is sufficient (cast to GibbsREGroupOracle is allowed
// because set_eta_offset is a no-op there). All oracles share the same beta of
// length p = oracles[0]->n_theta, and the same group-to-observation map is owned
// inside each oracle.
//
// re_contrib_fn(m, B_m) -> per-observation contribution of block m to the shared
//   linear predictor (length N), and obs_index_fn maps the block's group-major
//   observation order. To keep the header self-contained and avoid duplicating
//   the design here, the offset bookkeeping is delegated to two callbacks the
//   harness/driver supplies:
//     set_block_offset(m): push the current cross-block offset into oracles[m].
//     after_block_update(m, B_m): tell the coupling bookkeeper block m changed.
//   Single-block sweeps pass no-ops.
//
// Sigma0[m] is the initial per-block Sigma; b0[m] the initial G_m x nc RE
// values; Lg0[m] the per-group initial proposal Cholesky factors; beta0 the
// initial fixed effects; L_beta the beta proposal Cholesky.
// ---------------------------------------------------------------------------
template <typename SetOffsetFn, typename AfterUpdateFn>
inline GibbsOutput run_gibbs_sweep(
        std::vector<GibbsREGroupOracle*>& oracles,
        const std::vector<CovBlock>& blocks,
        const std::vector<Eigen::MatrixXd>& b0,
        const std::vector<std::vector<Eigen::MatrixXd>>& Lg0,
        const std::vector<Eigen::MatrixXd>& Sigma0,
        const Eigen::VectorXd& beta0,
        const Eigen::MatrixXd& L_beta,
        const GibbsConfig& cfg,
        SetOffsetFn set_block_offset,        // void(int m)
        AfterUpdateFn after_block_update) {  // void(int m, const Eigen::MatrixXd& B_m)

    const int M = static_cast<int>(blocks.size());
    const int p = static_cast<int>(beta0.size());

    Eigen::VectorXd beta = beta0;

    std::vector<BlockState> st(M);
    for (int m = 0; m < M; ++m) {
        st[m].B   = b0[m];
        st[m].L_g = Lg0[m];
        st[m].Q   = spd_inverse(Sigma0[m]);
        st[m].s_b = 2.4 / std::sqrt(static_cast<double>(blocks[m].nc));
        st[m].tgt_b = (blocks[m].nc > 1) ? 0.234 : 0.44;
    }
    std::vector<Eigen::MatrixXd> Sigma_cur = Sigma0;

    double s_beta   = 2.4 / std::sqrt(static_cast<double>(p));
    double tgt_beta = (p > 1) ? 0.234 : 0.44;

    const int n_sweep = cfg.n_burnin + cfg.n_iter;
    const int thin    = std::max(1, cfg.thin);

    GibbsOutput out;
    // keep_at: seq(n_burnin+1, n_sweep, by=thin)  (1-based in R; here 0-based loop)
    std::vector<char> keep(n_sweep, 0);
    for (int s = cfg.n_burnin; s < n_sweep; s += thin) keep[s] = 1;
    int n_kept = 0;
    for (int s = 0; s < n_sweep; ++s) if (keep[s]) ++n_kept;
    out.n_kept = n_kept;
    out.beta_draws.resize(n_kept, p);
    out.Sigma_draws.assign(n_kept, std::vector<Eigen::MatrixXd>(M));

    long acc_beta_rec = 0, acc_b_rec = 0, n_b_rec = 0;
    int kept = 0;

    // Scratch buffers for oracle calls.
    std::vector<double> gscratch, hscratch;

    for (int sweep = 1; sweep <= n_sweep; ++sweep) {
        const bool adapting = sweep <= cfg.n_burnin;
        const double gamma_t = 1.0 / std::sqrt(static_cast<double>(sweep));

        // --- beta | b, Sigma : RW Metropolis --------------------------------
        // ll(beta) = sum_m sum_g ell_g(b_g; beta) + Gaussian beta prior.
        // beta enters every block's ell through rebind(beta); the current b is
        // held fixed. Evaluate at current and proposed beta.
        auto beta_loglik = [&](const Eigen::VectorXd& bvec) -> double {
            double ll = 0.0;
            for (int m = 0; m < M; ++m) {
                set_block_offset(m);                 // push cross-block offset
                oracles[m]->rebind(bvec.data());
                const int nc = blocks[m].nc;
                const int G  = blocks[m].n_groups;
                double logL; std::vector<double> g(nc), h(nc * nc);
                for (int gg = 0; gg < G; ++gg) {
                    Eigen::VectorXd bg = st[m].B.row(gg).transpose();
                    oracles[m]->grad_hess(gg, bg.data(), logL, g.data(), h.data());
                    ll += logL;
                }
            }
            // Gaussian beta prior.
            for (int j = 0; j < p; ++j) {
                ll += R::dnorm(bvec(j), cfg.beta_prior_mean(j),
                               cfg.beta_prior_sd(j), 1);
            }
            return ll;
        };

        double ll_cur = beta_loglik(beta);
        Eigen::VectorXd z(p);
        for (int j = 0; j < p; ++j) z(j) = R::rnorm(0.0, 1.0);
        Eigen::VectorXd beta_prop = beta + s_beta * (L_beta * z);
        double ll_prop = beta_loglik(beta_prop);
        bool acc_beta = std::log(R::unif_rand()) < (ll_prop - ll_cur);
        if (acc_beta) beta = beta_prop;
        // Leave every oracle rebind()-ed to the accepted beta for the b-update.
        for (int m = 0; m < M; ++m) { set_block_offset(m); oracles[m]->rebind(beta.data()); }
        if (adapting)
            s_beta = std::exp(std::log(s_beta) + gamma_t * ((acc_beta ? 1.0 : 0.0) - tgt_beta));

        // --- b_{m,g} | beta, Sigma : per-(block, group) RW Metropolis -------
        long n_acc_b = 0, n_prop_b = 0;
        for (int m = 0; m < M; ++m) {
            const int nc = blocks[m].nc;
            const int G  = blocks[m].n_groups;
            set_block_offset(m);
            oracles[m]->rebind(beta.data());
            const Eigen::MatrixXd& Q = st[m].Q;
            long acc_m = 0;
            double logL_cur, logL_prop;
            std::vector<double> g(nc), h(nc * nc);
            for (int gg = 0; gg < G; ++gg) {
                Eigen::VectorXd bg = st[m].B.row(gg).transpose();
                oracles[m]->grad_hess(gg, bg.data(), logL_cur, g.data(), h.data());
                double ll_g_cur = logL_cur - 0.5 * (bg.transpose() * Q * bg)(0, 0);

                Eigen::VectorXd zz(nc);
                for (int c = 0; c < nc; ++c) zz(c) = R::rnorm(0.0, 1.0);
                Eigen::VectorXd bg_prop = bg + st[m].s_b * (st[m].L_g[gg] * zz);

                oracles[m]->grad_hess(gg, bg_prop.data(), logL_prop, g.data(), h.data());
                double ll_g_prop = logL_prop
                                   - 0.5 * (bg_prop.transpose() * Q * bg_prop)(0, 0);

                if (std::log(R::unif_rand()) < (ll_g_prop - ll_g_cur)) {
                    st[m].B.row(gg) = bg_prop.transpose();
                    ++acc_m;
                }
            }
            after_block_update(m, st[m].B);   // refresh coupling bookkeeping
            if (adapting)
                st[m].s_b = std::exp(std::log(st[m].s_b)
                              + gamma_t * (static_cast<double>(acc_m) / G - st[m].tgt_b));
            n_acc_b  += acc_m;
            n_prop_b += G;
        }

        // --- Sigma_m | b_m : exact conjugate draw per block -----------------
        for (int m = 0; m < M; ++m) {
            Sigma_cur[m] = draw_block_sigma(st[m].B, blocks[m]);
            st[m].Q      = spd_inverse(Sigma_cur[m]);
        }

        // --- record ----------------------------------------------------------
        if (!adapting && keep[sweep - 1]) {
            out.beta_draws.row(kept) = beta.transpose();
            for (int m = 0; m < M; ++m) out.Sigma_draws[kept][m] = Sigma_cur[m];
            acc_beta_rec += acc_beta ? 1 : 0;
            acc_b_rec    += n_acc_b;
            n_b_rec      += n_prop_b;
            ++kept;
        }
    }

    out.accept_beta = (n_kept > 0) ? static_cast<double>(acc_beta_rec) / n_kept : 0.0;
    out.accept_b    = (n_b_rec > 0) ? static_cast<double>(acc_b_rec) / n_b_rec
                                    : std::numeric_limits<double>::quiet_NaN();
    out.s_beta_final = Eigen::VectorXd::Constant(1, s_beta);
    out.s_b_final.resize(M);
    for (int m = 0; m < M; ++m) out.s_b_final(m) = st[m].s_b;
    return out;
}

// ---------------------------------------------------------------------------
// Convenience: build the per-group proposal Cholesky factors L_g from the
// shared per-group Newton mode-find. For each group g, aghq_group_mode returns
// the penalized precision negH = data-negH(b_hat) + Sigma^{-1}; the proposal
// shape is chol((negH)^-1) -- exactly the per-group posterior covariance block
// the R pilot takes from tulpa_laplace's return_re_cov. ONE mode-find source.
//
// orc must be rebind()-ed (and offset-set) to the pilot theta beforehand.
// ---------------------------------------------------------------------------
inline std::vector<Eigen::MatrixXd> pilot_group_proposals(
        const REGroupOracle& orc, const Eigen::MatrixXd& Sigma) {
    const int G = orc.n_groups;
    const int d = orc.d;
    Eigen::MatrixXd P = spd_inverse(Sigma);   // Sigma^{-1}
    std::vector<Eigen::MatrixXd> Lg(G);
    for (int g = 0; g < G; ++g) {
        GroupMode gm = aghq_group_mode(orc, g, P);
        Eigen::MatrixXd cov = spd_inverse(gm.negH);   // (negH)^{-1}
        Lg[g] = chol_spd(cov);
        (void)d;
    }
    return Lg;
}

} // namespace tulpa

#endif // TULPA_RE_COV_GIBBS_SWEEP_H
