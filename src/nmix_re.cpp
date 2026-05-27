// nmix_re.cpp
// Community / multispecies N-mixture (spAbundance msNMix) by Laplace-EM.
//
// Per species s = 1..S, site i, visit j:
//   N_{s,i}        ~ Poisson(lambda_{s,i})
//   y_{s,i,j} | N  ~ Binomial(N_{s,i}, p_{s,i,j})
//   log lambda_{s,i} = X_lambda_i . (mu_lambda + b_lambda_s)
//   logit p_{s,i,j}  = X_p_{ij}   . (mu_p      + b_p_s)
//   b_lambda_s ~ N(0, Sigma_lambda),  b_p_s ~ N(0, Sigma_p)     (community RE)
//
// The latent N_{s,i} integrates out per species-site in closed form via the
// shared per-site kernel (nmix_kernel.h, compute_nmix_site). The per-species
// coefficient deviations b_s = (b_lambda_s, b_p_s) are the random effects; their
// Gaussian community priors pin (mu_lambda, mu_p) as the fixed effects, so --
// unlike the intrinsic spatial fields -- no sum-to-zero constraint is needed.
//
// This is the C++ fast path for the community fit (gcol33/tulpa#31). The R
// adapter that wraps tulpa_nmix_site_marginal() into tulpa_re_aghq()'s make_group
// integrator is correct but pays R-interpreter overhead in the marginal
// optimizer's per-group Newton loop; this routine runs the same Laplace-EM in
// compiled code.
//
// ALGORITHM (mirrors the EM-rate / observed-info split nmix_spatial.cpp uses):
//   E-step  joint mode of (mu, {b_s}) by block-coordinate Newton. The mode-find
//           curvature is the complete-data Fisher (PSD, EM-rate, block-diagonal
//           across arms); species are conditionally independent given mu.
//   M-step  Sigma_lambda = mean_s [ b_lambda_s b_lambda_s' + Cov(b_lambda_s|y) ],
//           Sigma_p likewise, with Cov(b_s|y) = (H_bb_s + Sigma^{-1})^{-1}.
//   SEs     at convergence the marginal information for mu is the Schur
//           complement of the b-block in the joint OBSERVED-information Hessian
//           (the Var[N|y] rank-1 correction is added there, per Louis 1982):
//             I(mu) = sum_s A_s + tau I - sum_s A_s (A_s + Sigma^{-1})^{-1} A_s,
//           A_s the per-species observed-info coefficient curvature.
//
// Poisson first cut (r = +Inf). A global NB size is a follow-up: it adds a
// scalar dispersion to mu and threads compute_nmix_site's r argument.

#include "nmix_kernel.h"
#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

namespace {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// One site of one species: the site row (into X_lambda), the visit rows (into
// X_p), and the precomputed lgamma cache for that species-site's counts (so the
// EM's repeated evaluations skip the lgamma recompute).
struct SiteBlock {
  int site;                       // 0-based row of X_lambda
  std::vector<int> rows;          // 0-based rows of X_p (this site's visits)
  tulpa::NMixSiteCache cache;     // eta-independent marginal terms (Poisson)
};

// Per-species data curvature accumulated at a coefficient pair.
struct Accum {
  VectorXd grad;            // d = p_lam + p_p, score wrt the coefficient vector
  MatrixXd H;               // d x d, curvature (Fisher or observed, see use_obs)
  double   loglik;
};

// Accumulate one species' marginal log-lik, coefficient-space score, and
// curvature at coefficients (coef_lam, coef_p) = (mu + b) for that species.
// `use_obs = false` -> complete-data Fisher (PSD, block-diagonal, for the Newton
// mode-finder); `use_obs = true` -> marginal observed information (with the
// -Var[N|y] v v' rank-1 coupling between the arms, for the final SEs).
Accum accumulate_species(const std::vector<SiteBlock>& blocks,
                         const MatrixXd& Xl, const MatrixXd& Xp,
                         const VectorXd& coef_lam, const VectorXd& coef_p,
                         bool use_obs) {
  const int p_lam = (int)Xl.cols();
  const int p_p   = (int)Xp.cols();
  const int d     = p_lam + p_p;
  Accum acc;
  acc.grad = VectorXd::Zero(d);
  acc.H    = MatrixXd::Zero(d, d);
  acc.loglik = 0.0;

  std::vector<double> eta_p;
  for (const SiteBlock& blk : blocks) {
    const int i = blk.site;
    const int J = (int)blk.rows.size();
    const double eta_lambda = Xl.row(i).dot(coef_lam);
    eta_p.resize(J);
    for (int j = 0; j < J; ++j) eta_p[j] = Xp.row(blk.rows[j]).dot(coef_p);
    const tulpa::NMixSiteResult res =
      tulpa::compute_nmix_site_cached(blk.cache, eta_p.data(), eta_lambda);
    acc.loglik += res.log_lik;

    // Score: d logL / d coef = X' (eta-score).
    acc.grad.head(p_lam) += Xl.row(i).transpose() * res.grad_eta_lambda;
    // Complete-data Fisher (block-diagonal across arms).
    acc.H.topLeftCorner(p_lam, p_lam).noalias() +=
      res.info_eta_lambda * (Xl.row(i).transpose() * Xl.row(i));
    for (int j = 0; j < J; ++j) {
      const int r = blk.rows[j];
      acc.grad.tail(p_p) += Xp.row(r).transpose() * res.grad_eta_p[j];
      acc.H.bottomRightCorner(p_p, p_p).noalias() +=
        res.info_eta_p[j] * (Xp.row(r).transpose() * Xp.row(r));
    }
    if (use_obs && res.var_N > 0.0) {
      // Marginal observed info = complete-data Fisher - Var[N|y] u u',
      // u = (-score_wt_lambda X_lambda_i ; sum_j p_ij X_p_ij). The cross-arm
      // block of u u' is the abundance/detection coupling through latent N.
      VectorXd u(d);
      u.head(p_lam) = -res.score_wt_lambda * Xl.row(i).transpose();
      VectorXd up = VectorXd::Zero(p_p);
      for (int j = 0; j < J; ++j) {
        const int r = blk.rows[j];
        const double pj = 1.0 / (1.0 + std::exp(-eta_p[j]));
        up += pj * Xp.row(r).transpose();
      }
      u.tail(p_p) = up;
      acc.H.noalias() -= res.var_N * (u * u.transpose());
    }
  }
  return acc;
}

// Symmetric inverse of a small PD matrix via LLT, with a tiny ridge fallback.
MatrixXd safe_inverse(const MatrixXd& M) {
  const int d = (int)M.rows();
  Eigen::LLT<MatrixXd> llt(M);
  if (llt.info() == Eigen::Success)
    return llt.solve(MatrixXd::Identity(d, d));
  MatrixXd Mr = M;
  double md = M.diagonal().cwiseAbs().mean();
  if (!(md > 0)) md = 1.0;
  Mr.diagonal().array() += 1e-8 * md;
  return Mr.ldlt().solve(MatrixXd::Identity(d, d));
}

double logdet_spd(const MatrixXd& M) {
  Eigen::LLT<MatrixXd> llt(M);
  if (llt.info() != Eigen::Success) return std::numeric_limits<double>::quiet_NaN();
  double ld = 0.0;
  const MatrixXd& L = llt.matrixL();
  for (int i = 0; i < L.rows(); ++i) ld += std::log(L(i, i));
  return 2.0 * ld;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nmix_laplace_re(
    Rcpp::IntegerVector y,
    Rcpp::IntegerVector site_idx,        // 1-based, length n_obs
    Rcpp::IntegerVector species_idx,     // 1-based, length n_obs
    Rcpp::NumericMatrix X_lambda,        // n_sites x p_lam
    Rcpp::NumericMatrix X_p,             // n_obs   x p_p
    int n_sites, int n_species,
    Rcpp::NumericMatrix Sigma_lambda_init,
    Rcpp::NumericMatrix Sigma_p_init,
    Rcpp::NumericVector mu_lambda_init,
    Rcpp::NumericVector mu_p_init,
    int K_max,
    int max_iter = 100, double tol = 1e-6,
    int inner_max = 50, double inner_tol = 1e-8,
    double sigma_beta = 100.0, bool verbose = false) {

  const int n_obs = y.size();
  const int p_lam = X_lambda.ncol();
  const int p_p   = X_p.ncol();
  const int d     = p_lam + p_p;
  const int S     = n_species;
  const double tau_beta = 1.0 / (sigma_beta * sigma_beta);

  const Eigen::Map<MatrixXd> Xl(X_lambda.begin(), n_sites, p_lam);
  const Eigen::Map<MatrixXd> Xp(X_p.begin(), n_obs, p_p);
  std::vector<int> yv(y.begin(), y.end());

  // ---- group observations by species, then by site ----
  std::vector<std::vector<SiteBlock>> by_species(S);
  {
    // species -> (site -> rows). Sites are dense 0..n_sites-1.
    std::vector<std::vector<std::vector<int>>> tmp(
      S, std::vector<std::vector<int>>(n_sites));
    for (int o = 0; o < n_obs; ++o) {
      const int s = species_idx[o] - 1;
      const int i = site_idx[o] - 1;
      if (s < 0 || s >= S)        Rcpp::stop("species_idx out of range.");
      if (i < 0 || i >= n_sites)  Rcpp::stop("site_idx out of range.");
      tmp[s][i].push_back(o);
    }
    for (int s = 0; s < S; ++s)
      for (int i = 0; i < n_sites; ++i)
        if (!tmp[s][i].empty()) {
          std::vector<int> rows = std::move(tmp[s][i]);
          std::vector<int> yblk(rows.size());
          for (size_t j = 0; j < rows.size(); ++j) yblk[j] = yv[rows[j]];
          tulpa::NMixSiteCache cache =
            tulpa::nmix_precompute_site(yblk.data(), (int)rows.size(), K_max);
          by_species[s].push_back(SiteBlock{i, std::move(rows), std::move(cache)});
        }
  }

  // ---- initialize ----
  VectorXd mu(d);
  for (int j = 0; j < p_lam; ++j) mu[j] = mu_lambda_init[j];
  for (int j = 0; j < p_p; ++j)   mu[p_lam + j] = mu_p_init[j];

  MatrixXd Sig_l = Eigen::Map<MatrixXd>(Sigma_lambda_init.begin(), p_lam, p_lam);
  MatrixXd Sig_p = Eigen::Map<MatrixXd>(Sigma_p_init.begin(), p_p, p_p);

  std::vector<VectorXd> b(S, VectorXd::Zero(d));   // per-species RE deviation

  auto block_prec = [&](const MatrixXd& Sl, const MatrixXd& Sp) {
    MatrixXd P = MatrixXd::Zero(d, d);
    P.topLeftCorner(p_lam, p_lam)     = safe_inverse(Sl);
    P.bottomRightCorner(p_p, p_p)     = safe_inverse(Sp);
    return P;
  };

  bool converged = false;
  int n_iter = 0;
  std::vector<Accum> acc_s(S);   // last-iteration per-species Fisher accumulators

  for (int it = 0; it < max_iter; ++it) {
    n_iter = it + 1;
    const MatrixXd P = block_prec(Sig_l, Sig_p);

    // ---- E-step: joint mode of (mu, {b_s}) by block-coordinate Newton ----
    for (int inner = 0; inner < inner_max; ++inner) {
      double max_step = 0.0;
      // (a) per-species b update (each species independent given mu)
      for (int s = 0; s < S; ++s) {
        const VectorXd coef = mu + b[s];
        Accum acc = accumulate_species(by_species[s], Xl, Xp,
                                       coef.head(p_lam), coef.tail(p_p),
                                       /*use_obs=*/false);
        acc_s[s] = acc;
        const VectorXd g = acc.grad - P * b[s];
        const MatrixXd Hb = acc.H + P;
        const VectorXd step = safe_inverse(Hb) * g;
        b[s] += step;
        max_step = std::max(max_step, step.cwiseAbs().maxCoeff());
      }
      // (b) mu update (pooled over species at current b)
      VectorXd g_mu = VectorXd::Zero(d);
      MatrixXd H_mu = MatrixXd::Zero(d, d);
      for (int s = 0; s < S; ++s) {
        const VectorXd coef = mu + b[s];
        Accum acc = accumulate_species(by_species[s], Xl, Xp,
                                       coef.head(p_lam), coef.tail(p_p),
                                       /*use_obs=*/false);
        acc_s[s] = acc;
        g_mu += acc.grad;
        H_mu += acc.H;
      }
      g_mu -= tau_beta * mu;
      H_mu.diagonal().array() += tau_beta;
      const VectorXd step_mu = safe_inverse(H_mu) * g_mu;
      mu += step_mu;
      max_step = std::max(max_step, step_mu.cwiseAbs().maxCoeff());
      if (max_step < inner_tol) break;
    }

    // ---- M-step: EM update of the community covariances ----
    const MatrixXd Pm = block_prec(Sig_l, Sig_p);
    MatrixXd Sl_new = MatrixXd::Zero(p_lam, p_lam);
    MatrixXd Sp_new = MatrixXd::Zero(p_p, p_p);
    for (int s = 0; s < S; ++s) {
      const MatrixXd cov_s = safe_inverse(acc_s[s].H + Pm);   // Cov(b_s | y)
      const VectorXd bl = b[s].head(p_lam);
      const VectorXd bp = b[s].tail(p_p);
      Sl_new += bl * bl.transpose() + cov_s.topLeftCorner(p_lam, p_lam);
      Sp_new += bp * bp.transpose() + cov_s.bottomRightCorner(p_p, p_p);
    }
    Sl_new /= (double)S;
    Sp_new /= (double)S;

    const double dSig = std::max((Sl_new - Sig_l).cwiseAbs().maxCoeff(),
                                 (Sp_new - Sig_p).cwiseAbs().maxCoeff());
    Sig_l = Sl_new;
    Sig_p = Sp_new;
    if (verbose) {
      Rcpp::Rcout << "iter " << n_iter << "  dSigma=" << dSig << "\n";
    }
    if (dSig < tol) { converged = true; break; }
  }

  // ---- final pass: observed-info marginal Hessian for mu, Laplace logLik ----
  const MatrixXd P = block_prec(Sig_l, Sig_p);
  const double logdetP = logdet_spd(P);
  MatrixXd I_mu = MatrixXd::Zero(d, d);
  I_mu.diagonal().array() += tau_beta;
  double loglik_marg = 0.0;
  MatrixXd blup_l(S, p_lam), blup_p(S, p_p);
  for (int s = 0; s < S; ++s) {
    const VectorXd coef = mu + b[s];
    Accum acc = accumulate_species(by_species[s], Xl, Xp,
                                   coef.head(p_lam), coef.tail(p_p),
                                   /*use_obs=*/true);
    const MatrixXd A  = acc.H;            // observed-info coefficient curvature
    const MatrixXd Bb = A + P;            // b-block of the joint Hessian
    const MatrixXd Bbinv = safe_inverse(Bb);
    I_mu += A - A * Bbinv * A;            // Schur complement contribution
    // Laplace marginal log-lik for this species.
    const double ldH = logdet_spd(Bb);
    loglik_marg += acc.loglik - 0.5 * b[s].dot(P * b[s])
                   + 0.5 * logdetP - 0.5 * ldH;
    blup_l.row(s) = b[s].head(p_lam).transpose();
    blup_p.row(s) = b[s].tail(p_p).transpose();
  }
  const MatrixXd vcov = safe_inverse(I_mu);

  Rcpp::NumericVector mu_lambda(p_lam), mu_p(p_p);
  for (int j = 0; j < p_lam; ++j) mu_lambda[j] = mu[j];
  for (int j = 0; j < p_p; ++j)   mu_p[j] = mu[p_lam + j];

  // Pack vcov, Sigma blocks, BLUPs as R matrices.
  Rcpp::NumericMatrix vcov_out(d, d);
  for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j) vcov_out(i, j) = vcov(i, j);
  Rcpp::NumericMatrix Sl_out(p_lam, p_lam), Sp_out(p_p, p_p);
  for (int i = 0; i < p_lam; ++i) for (int j = 0; j < p_lam; ++j) Sl_out(i, j) = Sig_l(i, j);
  for (int i = 0; i < p_p; ++i) for (int j = 0; j < p_p; ++j) Sp_out(i, j) = Sig_p(i, j);
  Rcpp::NumericMatrix bl_out(S, p_lam), bp_out(S, p_p);
  for (int s = 0; s < S; ++s) {
    for (int j = 0; j < p_lam; ++j) bl_out(s, j) = blup_l(s, j);
    for (int j = 0; j < p_p; ++j)   bp_out(s, j) = blup_p(s, j);
  }

  return Rcpp::List::create(
    Rcpp::Named("mu_lambda")    = mu_lambda,
    Rcpp::Named("mu_p")         = mu_p,
    Rcpp::Named("vcov")         = vcov_out,
    Rcpp::Named("Sigma_lambda") = Sl_out,
    Rcpp::Named("Sigma_p")      = Sp_out,
    Rcpp::Named("b_lambda")     = bl_out,
    Rcpp::Named("b_p")          = bp_out,
    Rcpp::Named("log_lik")      = loglik_marg,
    Rcpp::Named("converged")    = converged,
    Rcpp::Named("n_iter")       = n_iter,
    Rcpp::Named("K_max")        = K_max
  );
}
