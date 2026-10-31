// nested_laplace_joint_core.cpp
// Definition of make_single_arm, declared in nested_laplace_joint_core.h.

#include <RcppEigen.h>
#include "nested_laplace_joint_core.h"

namespace tulpa {

void make_single_arm(
    std::vector<ParsedArm>& parsed,
    std::vector<JointArm>& arms,
    const Rcpp::NumericMatrix& X,
    const Rcpp::NumericVector& re_idx,
    const Rcpp::IntegerVector& spatial_idx,
    int p, int n_re_groups, double sigma_re,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const std::string& family, double phi, int N,
    Rcpp::Nullable<Rcpp::NumericVector> offset_nullable,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable) {
    if ((int)X.nrow() != N) {
        Rcpp::stop("nrow(X) (%d) must equal n_obs (%d).", (int)X.nrow(), N);
    }
    if ((int)y.size() != N) {
        Rcpp::stop("length(y) (%d) must equal n_obs (%d).", (int)y.size(), N);
    }
    if ((int)n_trials.size() != N) {
        Rcpp::stop("length(n_trials) (%d) must equal n_obs (%d).",
                   (int)n_trials.size(), N);
    }
    // The list-parsing path checks both of these per arm; a kernel that builds
    // its single arm here indexes them exactly the same way.
    if ((int)re_idx.size() != N) {
        Rcpp::stop("length(re_idx) (%d) must equal n_obs (%d).",
                   (int)re_idx.size(), N);
    }
    if ((int)spatial_idx.size() != N) {
        Rcpp::stop("length(spatial_idx) (%d) must equal n_obs (%d).",
                   (int)spatial_idx.size(), N);
    }

    parsed.resize(1);
    ParsedArm& pa = parsed[0];
    pa.X           = X;
    pa.re_idx      = re_idx;
    pa.spatial_idx = spatial_idx;
    pa.p           = p;
    pa.n_re_groups = n_re_groups;
    pa.sigma_re    = sigma_re;
    pa.beta_start  = 0;
    pa.re_start    = p;
    pa.tau_re      = (n_re_groups > 0)
                     ? 1.0 / (sigma_re * sigma_re + RE_VARIANCE_FLOOR)
                     : 0.0;
    if (offset_nullable.isNotNull()) {
        Rcpp::NumericVector off(offset_nullable);
        if ((int)off.size() != N) {
            Rcpp::stop("length(offset) (%d) must equal n_obs (%d).",
                       (int)off.size(), N);
        }
        pa.offset = off;
    }

    arms.resize(1);
    JointArm& a = arms[0];
    a.y        = y;
    a.n_trials = n_trials;
    a.family   = family;
    a.phi      = phi;
    a.N        = N;
    if (weights_nullable.isNotNull()) {
        Rcpp::NumericVector w(weights_nullable);
        if ((int)w.size() != N) {
            Rcpp::stop("length(weights) (%d) must equal n_obs (%d).",
                       (int)w.size(), N);
        }
        a.weights = w;
    }
}

} // namespace tulpa
