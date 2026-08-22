// re_cov_gibbs.cpp
// Rcpp entry point for the Metropolis-within-Gibbs random-effect-covariance
// sampler (tulpa_re_cov_gibbs). The R wrapper does the pilot Laplace solve
// (starting values + proposal shapes + initial Sigma) and the posterior summary;
// this runs the hot sweep loop in C++ with a single likelihood source
// (SingleArmGLMMOracle), so the per-row family density is no longer duplicated
// in R.

#include "re_cov_gibbs_sweep.h"
#include <RcppEigen.h>
#include <memory>
#include <vector>
#include <string>

using namespace Rcpp;
using namespace tulpa;

// Build a native single-arm GLMM oracle (XPtr<REGroupOracle>) for the AGHQ
// engine: the per-group conditional likelihood of a separable GLMM with RE
// design Z over one shared grouping factor. theta = beta (the fixed effects).
// Reused by the deterministic-integrate driver (single-factor tulpa_re_cov_nested)
// so its inner solve shares the one compiled family density. Z is the stacked
// RE design (n x sum_block n_coefs); the block split lives in the objective's
// Sigma packing, not the oracle.
// [[Rcpp::export]]
SEXP cpp_glmm_oracle_make(std::string family, double phi,
                          NumericVector y, NumericVector n_trials,
                          NumericMatrix X, NumericMatrix Z,
                          IntegerVector idx, int n_groups) {
    const GLMMFamily fam = glmm_family_from_string(family);
    Eigen::MatrixXd Xe = as<Eigen::MatrixXd>(X);
    Eigen::MatrixXd Ze = as<Eigen::MatrixXd>(Z);
    Eigen::VectorXi ix = as<Eigen::VectorXi>(idx);
    Eigen::VectorXd ye = as<Eigen::VectorXd>(y);
    Eigen::VectorXd nt = as<Eigen::VectorXd>(n_trials);
    return XPtr<REGroupOracle>(
        new SingleArmGLMMOracle(fam, phi, Xe, Ze, ix, n_groups, ye, nt), true);
}

// Shape check for one block-spec field. `rows`/`cols` are what arrived,
// `want_rows`/`want_cols` what the sweep indexes it by. A vector field passes
// its length as `rows` and 1 as `cols`.
static void check_block_dims(int m, const char* field,
                             R_xlen_t rows, R_xlen_t cols,
                             int want_rows, int want_cols) {
    if ((int)rows != want_rows || (int)cols != want_cols) {
        stop("cpp_re_cov_gibbs_sweep: block %d field `%s` is %d x %d, "
             "expected %d x %d.", m + 1, field, (int)rows, (int)cols,
             want_rows, want_cols);
    }
}

// blocks: list of per-block specs, each a list with
//   Z (n x nc), idx (n, 1-based), nc, full (logical), n_groups, nu0,
//   Lambda0 (nc x nc, full only) or lambda0 (nc, diagonal only),
//   b0 (G x nc), Lg0 (list of nc x nc), Sigma0 (nc x nc).
// [[Rcpp::export]]
List cpp_re_cov_gibbs_sweep(std::string family, double phi,
                            NumericVector y, NumericVector n_trials,
                            NumericMatrix X, List blocks,
                            NumericVector beta0, NumericMatrix L_beta,
                            int n_iter, int n_burnin, int thin,
                            NumericVector beta_prior_mean,
                            NumericVector beta_prior_sd) {

    const GLMMFamily fam = glmm_family_from_string(family);
    const int M = blocks.size();
    if (M < 1) stop("cpp_re_cov_gibbs_sweep: `blocks` must hold at least one block.");
    Eigen::MatrixXd Xe = as<Eigen::MatrixXd>(X);
    Eigen::VectorXd ye = as<Eigen::VectorXd>(y);
    Eigen::VectorXd nt = as<Eigen::VectorXd>(n_trials);
    const int n = static_cast<int>(Xe.rows());
    const int p = static_cast<int>(Xe.cols());

    std::vector<std::unique_ptr<SingleArmGLMMOracle>> owned;
    std::vector<SingleArmGLMMOracle*> oracles;
    std::vector<CovBlock> cb(M);
    std::vector<Eigen::MatrixXd> b0(M), Sigma0(M);
    std::vector<std::vector<Eigen::MatrixXd>> Lg0(M);

    owned.reserve(M);
    for (int m = 0; m < M; ++m) {
        List bm = blocks[m];
        Eigen::MatrixXd Z = as<Eigen::MatrixXd>(bm["Z"]);
        Eigen::VectorXi idx = as<Eigen::VectorXi>(bm["idx"]);
        const int nc = as<int>(bm["nc"]);
        const bool full = as<bool>(bm["full"]);
        const int ng = as<int>(bm["n_groups"]);

        if (nc < 1) {
            stop("cpp_re_cov_gibbs_sweep: block %d has nc = %d; it must be >= 1.",
                 m + 1, nc);
        }
        if (ng < 1) {
            stop("cpp_re_cov_gibbs_sweep: block %d has n_groups = %d; it must be >= 1.",
                 m + 1, ng);
        }
        check_block_dims(m, "Z", Z.rows(), Z.cols(), n, nc);

        owned.emplace_back(new SingleArmGLMMOracle(fam, phi, Xe, Z, idx, ng, ye, nt));
        oracles.push_back(owned.back().get());

        cb[m].nc = nc; cb[m].full = full; cb[m].n_groups = ng;
        cb[m].nu0 = as<double>(bm["nu0"]);
        if (full) {
            cb[m].Lambda0 = as<Eigen::MatrixXd>(bm["Lambda0"]);
            check_block_dims(m, "Lambda0", cb[m].Lambda0.rows(),
                             cb[m].Lambda0.cols(), nc, nc);
        } else {
            cb[m].lambda0 = as<Eigen::VectorXd>(bm["lambda0"]);
            check_block_dims(m, "lambda0", cb[m].lambda0.size(), 1, nc, 1);
        }

        b0[m]     = as<Eigen::MatrixXd>(bm["b0"]);
        Sigma0[m] = as<Eigen::MatrixXd>(bm["Sigma0"]);
        check_block_dims(m, "b0", b0[m].rows(), b0[m].cols(), ng, nc);
        check_block_dims(m, "Sigma0", Sigma0[m].rows(), Sigma0[m].cols(), nc, nc);
        List lg = bm["Lg0"];
        if (lg.size() != ng) {
            stop("cpp_re_cov_gibbs_sweep: block %d Lg0 has %d entries, expected "
                 "n_groups = %d.", m + 1, (int)lg.size(), ng);
        }
        Lg0[m].resize(lg.size());
        for (int g = 0; g < lg.size(); ++g) {
            Lg0[m][g] = as<Eigen::MatrixXd>(lg[g]);
            check_block_dims(m, "Lg0[[g]]", Lg0[m][g].rows(), Lg0[m][g].cols(),
                             nc, nc);
        }
    }

    GibbsConfig cfg;
    cfg.n_iter = n_iter; cfg.n_burnin = n_burnin; cfg.thin = thin;
    cfg.beta_prior_mean = as<Eigen::VectorXd>(beta_prior_mean);
    cfg.beta_prior_sd   = as<Eigen::VectorXd>(beta_prior_sd);

    Eigen::VectorXd b0v = as<Eigen::VectorXd>(beta0);
    Eigen::MatrixXd Lb  = as<Eigen::MatrixXd>(L_beta);

    if ((int)b0v.size() != p) {
        stop("cpp_re_cov_gibbs_sweep: beta0 has length %d, expected ncol(X) = %d.",
             (int)b0v.size(), p);
    }
    if (Lb.rows() != p || Lb.cols() != p) {
        stop("cpp_re_cov_gibbs_sweep: L_beta is %d x %d, expected %d x %d.",
             (int)Lb.rows(), (int)Lb.cols(), p, p);
    }
    if ((int)cfg.beta_prior_mean.size() != p ||
        (int)cfg.beta_prior_sd.size() != p) {
        stop("cpp_re_cov_gibbs_sweep: beta_prior_mean (%d) and beta_prior_sd (%d) "
             "must both have length ncol(X) = %d.",
             (int)cfg.beta_prior_mean.size(), (int)cfg.beta_prior_sd.size(), p);
    }

    GibbsOutput out = run_glmm_gibbs(oracles, cb, b0, Lg0, Sigma0, b0v, Lb, cfg);

    // Sigma_draws -> list (length n_kept) of list (length M) of matrices.
    List sig(out.n_kept);
    for (int k = 0; k < out.n_kept; ++k) {
        List per(M);
        for (int m = 0; m < M; ++m) per[m] = wrap(out.Sigma_draws[k][m]);
        sig[k] = per;
    }

    return List::create(
        _["beta_draws"]  = wrap(out.beta_draws),
        _["re_draws"]    = wrap(out.re_draws),
        _["Sigma_draws"] = sig,
        _["accept_beta"] = out.accept_beta,
        _["accept_b"]    = out.accept_b,
        _["n_kept"]      = out.n_kept);
}
