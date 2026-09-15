// sampler_log_prob.h
// The log posterior every ModelData sampler reports per stored draw.
//
// NUTS records, at each retained position, the value of the target its fused
// gradient pass evaluates: the generic log posterior on the sampler's own
// unconstrained coordinates, with every change-of-variables Jacobian the priors
// carry. tulpa_hmc::compute_log_post is its double-precision form. The other kernels driven through the same
// ModelData (ESS, SGHMC, SGLD, MCLMC, SMC, VI) sample -- or approximate -- that
// same target, so their per-draw log posterior is that function evaluated at
// each draw. Evaluating it here for all of them, rather than trusting whatever
// running value each kernel happens to keep, is what makes the reported
// quantity one definition across backends.

#ifndef TULPA_SAMPLER_LOG_PROB_H
#define TULPA_SAMPLER_LOG_PROB_H

#include <Rcpp.h>

#include <atomic>
#include <cstddef>
#include <string>
#include <vector>

#include "hmc_sampler.h"
#include "omp_threads.h"

namespace tulpa {

// Log posterior at each row of `draws` (n_draws x layout.total_params). Rows are
// independent, so they are evaluated across a thread team; each row copies its
// own position, and the double evaluator carries no tape. A raise from inside
// the evaluator is caught per row and re-raised on the calling thread after the
// region.
inline Rcpp::NumericVector sampler_log_prob_rows(const Rcpp::NumericMatrix& draws,
                                                 const ModelData& data,
                                                 const ParamLayout& layout) {
    const int n = draws.nrow();
    const int D = layout.total_params;
    if (n > 0 && draws.ncol() != D) {
        Rcpp::stop("sampler_log_prob_rows: draws have %d columns but the layout "
                   "has %d parameters.", (int)draws.ncol(), D);
    }
    std::vector<double> out(static_cast<std::size_t>(n), NA_REAL);
    if (n == 0) return Rcpp::wrap(out);

    const double* src = draws.begin();
    std::atomic<bool> failed{false};
    std::string err;
    auto body = [&](int s) {
        if (failed.load()) return;
        try {
            std::vector<double> q(static_cast<std::size_t>(D));
            for (int j = 0; j < D; j++) {
                q[j] = src[static_cast<std::size_t>(j) * n + s];
            }
            out[s] = tulpa_hmc::compute_log_post(q, data, layout);
        } catch (const std::exception& e) {
            if (!failed.exchange(true)) err = e.what();
        } catch (...) {
            failed.exchange(true);
        }
    };
    tulpa_parallel_for(tulpa_omp_team_size(n), n, body);
    if (failed.load()) {
        Rcpp::stop("per-draw log posterior failed: %s",
                   err.empty() ? "unknown error" : err.c_str());
    }
    return Rcpp::wrap(out);
}

}  // namespace tulpa

#endif  // TULPA_SAMPLER_LOG_PROB_H
