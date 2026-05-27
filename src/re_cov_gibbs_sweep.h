// re_cov_gibbs_sweep.h
// Cohesive C++ Metropolis-within-Gibbs sweep for random-effect covariances Sigma
// (the engine behind tulpa_re_cov_gibbs). It samples the exact joint posterior
// p(beta, {b_m}, {Sigma_m} | y) with one likelihood source -- the native
// SingleArmGLMMOracle (glmm_oracle.h) -- replacing the duplicated R per-row
// density. The estimator is unchanged from the R sampler:
//
//   * beta | b, Sigma      -- random-walk Metropolis. The total data log-lik is
//                             evaluated over ONE block's group partition (every
//                             grouping factor partitions all observations), with
//                             the other blocks' RE contributions carried in that
//                             block's per-observation offset; plus a Gaussian
//                             beta prior. Proposal shape = supplied L_beta.
//   * b_{m,g} | beta, Sigma -- per-(block, group) random-walk Metropolis. eta =
//                             X beta + (other blocks' RE) + Z_m b; the MH ratio
//                             is the group log-lik + the Gaussian RE log-prior
//                             -0.5 b' Q_m b. Proposal shape = supplied per-group
//                             Cholesky L_g (the pilot Laplace posterior block).
//   * Sigma_m | b_m         -- exact conjugate inverse-Wishart per block (full =>
//                             matrix IW; diagonal => per-coordinate scalar IW).
//
// The shared linear predictor xb + sum_m re_contrib_m is maintained incrementally
// so each block update sees the others held fixed (the cross-block eta coupling
// the R sampler did via `base = xb + (re_total - re_contrib_m)`). Robbins-Monro
// proposal-scale adaptation runs during burn-in. All randomness goes through R's
// RNG (R::rnorm / R::rchisq / R::unif_rand), so set.seed in R reproduces a run.

#ifndef TULPA_RE_COV_GIBBS_SWEEP_H
#define TULPA_RE_COV_GIBBS_SWEEP_H

#include "glmm_oracle.h"
#include <RcppEigen.h>
#include <Rcpp.h>          // R::rnorm, R::rchisq, R::unif_rand, R::dnorm
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace tulpa {

// ---------------------------------------------------------------------------
// Native inverse-Wishart via Bartlett (port of .rinvwishart in R).
//   Sigma ~ IW(df, Lambda)  <=>  Sigma^{-1} ~ Wishart(df, Lambda^{-1}).
// Sample the Wishart by Bartlett: Lambda^{-1} = C C'; A lower-triangular with
// A_ii = sqrt(chisq_{df - i + 1}), A_ij ~ N(0,1) for i > j; W = (C A)(C A)';
// then Sigma = W^{-1}. Requires df > p - 1 (a 1x1 block needs only df > 0).
// ---------------------------------------------------------------------------
inline Eigen::MatrixXd rinvwishart(double df, const Eigen::MatrixXd& Lambda) {
    const int p = static_cast<int>(Lambda.rows());
    if (df <= p - 1)
        throw std::runtime_error("inverse-Wishart df must exceed p - 1.");
    Eigen::LLT<Eigen::MatrixXd> lltLam(Lambda);
    if (lltLam.info() != Eigen::Success)
        throw std::runtime_error("inverse-Wishart Lambda not SPD.");
    Eigen::MatrixXd V = lltLam.solve(Eigen::MatrixXd::Identity(p, p));   // Lambda^{-1}
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

// Inverse of an SPD matrix via Cholesky.
inline Eigen::MatrixXd spd_inverse(const Eigen::MatrixXd& S) {
    const int n = static_cast<int>(S.rows());
    Eigen::LLT<Eigen::MatrixXd> llt(0.5 * (S + S.transpose()));
    return llt.solve(Eigen::MatrixXd::Identity(n, n));
}

// ---------------------------------------------------------------------------
// Per-block covariance + conjugate prior (mirrors .re_cov_block_layout +
// .re_gibbs_block_prior). full = TRUE -> correlated nc x nc inverse-Wishart
// IW(nu0, Lambda0); full = FALSE -> nc independent scalar inverse-Wisharts
// (== inverse-gamma), shape nu0, per-coordinate scales lambda0.
// ---------------------------------------------------------------------------
struct CovBlock {
    int nc = 1;
    bool full = false;
    int n_groups = 0;
    double nu0 = 0.0;
    Eigen::MatrixXd Lambda0;      // full
    Eigen::VectorXd lambda0;      // diagonal
};

// Exact conjugate draw of one block's Sigma given its random effects B_m
// (G_m x nc). Port of .re_gibbs_draw_sigma.
inline Eigen::MatrixXd draw_block_sigma(const Eigen::MatrixXd& B_m,
                                        const CovBlock& bl) {
    const int G  = static_cast<int>(B_m.rows());
    const int nc = bl.nc;
    if (bl.full) {
        Eigen::MatrixXd Lam = bl.Lambda0 + B_m.transpose() * B_m;   // + sum_g b b'
        return rinvwishart(bl.nu0 + G, Lam);
    }
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(nc, nc);
    for (int i = 0; i < nc; ++i) {
        Eigen::MatrixXd Lam1(1, 1);
        Lam1(0, 0) = bl.lambda0(i) + B_m.col(i).squaredNorm();
        S(i, i) = rinvwishart(bl.nu0 + G, Lam1)(0, 0);
    }
    return S;
}

// ---------------------------------------------------------------------------
// Sweep configuration / recorded output.
// ---------------------------------------------------------------------------
struct GibbsConfig {
    int n_iter   = 2000;
    int n_burnin = 1000;
    int thin     = 1;
    Eigen::VectorXd beta_prior_mean;   // length p
    Eigen::VectorXd beta_prior_sd;     // length p
};

struct GibbsOutput {
    int n_kept = 0;
    Eigen::MatrixXd beta_draws;                              // n_kept x p
    std::vector<std::vector<Eigen::MatrixXd>> Sigma_draws;   // [k][m]
    double accept_beta = 0.0;
    double accept_b    = 0.0;
};

// ---------------------------------------------------------------------------
// The driver. oracles[m] owns block m's design (Z_m, group map) and the shared
// X; all blocks span the same n observations. b0[m] / Lg0[m] / Sigma0[m] are the
// per-block initial RE, per-group proposal Cholesky factors and initial Sigma;
// beta0 / L_beta the initial fixed effects and their proposal Cholesky.
// ---------------------------------------------------------------------------
inline GibbsOutput run_glmm_gibbs(
        std::vector<SingleArmGLMMOracle*>& oracles,
        const std::vector<CovBlock>& blocks,
        const std::vector<Eigen::MatrixXd>& b0,
        const std::vector<std::vector<Eigen::MatrixXd>>& Lg0,
        const std::vector<Eigen::MatrixXd>& Sigma0,
        const Eigen::VectorXd& beta0,
        const Eigen::MatrixXd& L_beta,
        const GibbsConfig& cfg) {

    const int M = static_cast<int>(blocks.size());
    const int p = static_cast<int>(beta0.size());
    const int n = static_cast<int>(oracles[0]->X.rows());

    Eigen::VectorXd beta = beta0;

    // Per-block RE state, proposal shapes, precisions, adapted scales.
    std::vector<Eigen::MatrixXd> B = b0;
    std::vector<std::vector<Eigen::MatrixXd>> L_g = Lg0;
    std::vector<Eigen::MatrixXd> Q(M);
    std::vector<Eigen::MatrixXd> Sigma_cur = Sigma0;
    std::vector<double> s_b(M), tgt_b(M);
    for (int m = 0; m < M; ++m) {
        Q[m]     = spd_inverse(Sigma0[m]);
        s_b[m]   = 2.4 / std::sqrt(static_cast<double>(blocks[m].nc));
        tgt_b[m] = (blocks[m].nc > 1) ? 0.234 : 0.44;
    }

    double s_beta   = 2.4 / std::sqrt(static_cast<double>(p));
    double tgt_beta = (p > 1) ? 0.234 : 0.44;

    // Shared linear predictor bookkeeping: re_contrib[m] is block m's per-obs RE
    // contribution; re_total their sum. offset for block m is re_total minus its
    // own contribution (the other blocks, held fixed).
    std::vector<Eigen::VectorXd> re_contrib(M);
    Eigen::VectorXd re_total = Eigen::VectorXd::Zero(n);
    for (int m = 0; m < M; ++m) {
        oracles[m]->re_contribution(B[m], re_contrib[m]);
        re_total += re_contrib[m];
    }
    for (int m = 0; m < M; ++m) oracles[m]->rebind(beta.data());

    auto push_offset = [&](int m) {
        Eigen::VectorXd off = re_total - re_contrib[m];
        oracles[m]->set_offset(off.data());
    };

    const int n_sweep = cfg.n_burnin + cfg.n_iter;
    const int thin    = std::max(1, cfg.thin);
    std::vector<char> keep(n_sweep, 0);
    for (int s = cfg.n_burnin; s < n_sweep; s += thin) keep[s] = 1;
    int n_kept = 0;
    for (int s = 0; s < n_sweep; ++s) if (keep[s]) ++n_kept;

    GibbsOutput out;
    out.n_kept = n_kept;
    out.beta_draws.resize(n_kept, p);
    out.Sigma_draws.assign(n_kept, std::vector<Eigen::MatrixXd>(M));

    long acc_beta_rec = 0, acc_b_rec = 0, n_b_rec = 0;
    int kept = 0;

    std::vector<double> gscr(0), hscr(0);

    for (int sweep = 1; sweep <= n_sweep; ++sweep) {
        const bool adapting  = sweep <= cfg.n_burnin;
        const double gamma_t = 1.0 / std::sqrt(static_cast<double>(sweep));

        // --- beta | b, Sigma : RW Metropolis --------------------------------
        // Total data log-lik via block 0's group partition (offset carries the
        // other blocks' RE; X beta via rebind), plus the Gaussian beta prior.
        push_offset(0);
        const int nc0 = blocks[0].nc;
        gscr.assign(nc0, 0.0); hscr.assign((std::size_t)nc0 * nc0, 0.0);
        auto beta_loglik = [&](const Eigen::VectorXd& bvec) -> double {
            oracles[0]->rebind(bvec.data());
            double ll = 0.0, logL;
            for (int g = 0; g < blocks[0].n_groups; ++g) {
                Eigen::VectorXd bg = B[0].row(g).transpose();
                oracles[0]->grad_hess(g, bg.data(), logL, gscr.data(), hscr.data());
                ll += logL;
            }
            for (int j = 0; j < p; ++j)
                ll += R::dnorm(bvec(j), cfg.beta_prior_mean(j), cfg.beta_prior_sd(j), 1);
            return ll;
        };

        double ll_cur = beta_loglik(beta);
        Eigen::VectorXd z(p);
        for (int j = 0; j < p; ++j) z(j) = R::rnorm(0.0, 1.0);
        Eigen::VectorXd beta_prop = beta + s_beta * (L_beta * z);
        double ll_prop = beta_loglik(beta_prop);
        const bool acc_beta = std::log(R::unif_rand()) < (ll_prop - ll_cur);
        if (acc_beta) beta = beta_prop;
        for (int m = 0; m < M; ++m) oracles[m]->rebind(beta.data());
        if (adapting)
            s_beta = std::exp(std::log(s_beta)
                       + gamma_t * ((acc_beta ? 1.0 : 0.0) - tgt_beta));

        // --- b_{m,g} | beta, Sigma : per-(block, group) RW Metropolis -------
        long n_acc_b = 0, n_prop_b = 0;
        for (int m = 0; m < M; ++m) {
            const int nc = blocks[m].nc;
            const int G  = blocks[m].n_groups;
            push_offset(m);
            const Eigen::MatrixXd& Qm = Q[m];
            std::vector<double> g(nc), h((std::size_t)nc * nc);
            long acc_m = 0;
            double logL_cur, logL_prop;
            for (int gg = 0; gg < G; ++gg) {
                Eigen::VectorXd bg = B[m].row(gg).transpose();
                oracles[m]->grad_hess(gg, bg.data(), logL_cur, g.data(), h.data());
                const double ll_g_cur = logL_cur - 0.5 * (bg.transpose() * Qm * bg)(0, 0);

                Eigen::VectorXd zz(nc);
                for (int c = 0; c < nc; ++c) zz(c) = R::rnorm(0.0, 1.0);
                Eigen::VectorXd bg_prop = bg + s_b[m] * (L_g[m][gg] * zz);

                oracles[m]->grad_hess(gg, bg_prop.data(), logL_prop, g.data(), h.data());
                const double ll_g_prop = logL_prop
                                  - 0.5 * (bg_prop.transpose() * Qm * bg_prop)(0, 0);

                if (std::log(R::unif_rand()) < (ll_g_prop - ll_g_cur)) {
                    B[m].row(gg) = bg_prop.transpose();
                    ++acc_m;
                }
            }
            // Refresh this block's contribution and the running total.
            Eigen::VectorXd nc_contrib;
            oracles[m]->re_contribution(B[m], nc_contrib);
            re_total += nc_contrib - re_contrib[m];
            re_contrib[m] = nc_contrib;
            if (adapting)
                s_b[m] = std::exp(std::log(s_b[m])
                           + gamma_t * (static_cast<double>(acc_m) / G - tgt_b[m]));
            n_acc_b  += acc_m;
            n_prop_b += G;
        }

        // --- Sigma_m | b_m : exact conjugate draw per block -----------------
        for (int m = 0; m < M; ++m) {
            Sigma_cur[m] = draw_block_sigma(B[m], blocks[m]);
            Q[m]         = spd_inverse(Sigma_cur[m]);
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
    return out;
}

} // namespace tulpa

#endif // TULPA_RE_COV_GIBBS_SWEEP_H
