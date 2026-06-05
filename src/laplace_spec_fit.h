// laplace_spec_fit.h
// Marshalling from the standalone single-point Laplace R-export inputs into the
// spec solver (laplace_mode_spec_dense_solve, laplace_spec.cpp). Single source
// of truth for the ModelData / ParamLayout the family-enum single-point fitters
// used to build inline: cpp_laplace_fit / _spatial / _bym2 all share the
// single-process built-in-family + optional single iid RE setup here, then add
// their own GMRF blocks. cpp_laplace_fit_multi_re's multi-term marshalling lives
// in laplace_core.cpp (its only caller); the correlated-covariance conversion it
// needs (pack -> spec log-Cholesky params) is shared from here.

#ifndef TULPA_LAPLACE_SPEC_FIT_H
#define TULPA_LAPLACE_SPEC_FIT_H

#include "tulpa/likelihood.h"
#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "laplace_builtin_family_spec.h"
#include "laplace_core.h"            // LaplaceResult
#include "laplace_re_priors.h"       // BetaPrior
#include "laplace_spec_solve.h"      // laplace_mode_spec_dense_solve
#include <Rcpp.h>
#include <cmath>
#include <string>
#include <vector>

namespace tulpa {

// Extract an optional offset() vector from a Nullable<NumericVector> into a
// std::vector<double>, validating its length against N. A null argument yields
// an empty vector (treated as "no offset" by the eta assemblers). Single source
// of truth for the offset marshalling shared by the single-point Laplace and
// ModelData-sampler R exports.
inline std::vector<double> as_offset_vec(
    const Rcpp::Nullable<Rcpp::NumericVector>& offset_nullable, int N
) {
    if (offset_nullable.isNull()) return {};
    Rcpp::NumericVector o(offset_nullable);
    if ((int)o.size() != N) {
        Rcpp::stop("offset length (%d) must equal the number of observations (%d).",
                   (int)o.size(), N);
    }
    return std::vector<double>(o.begin(), o.end());
}

// Spec-solver inputs for a single-process built-in-family fit with an optional
// single iid RE term, kept alive together: data borrows spec & resp (and resp
// borrows the response arrays), so the whole struct must outlive the solve.
// resp.y points into the caller's R vector -- keep that alive too.
struct SpecFamilyInputs {
    LikelihoodSpec        spec;
    BuiltinFamilyResponse resp;
    std::vector<int>      n_trials;     // stable storage behind resp.n_trials
    ModelData             data;
    ParamLayout           layout;
    std::vector<int>      re_group;     // 1-based per-obs RE group, empty if none
};

// Build the [beta(p) | RE(G) | block(n_block_latent) | log_sigma_re] layout +
// ModelData for the iid-RE built-in-family exports. n_block_latent reserves the
// GMRF block region between the RE effects and the (held-fixed) log_sigma_re
// hyperparameter slot, so the contiguous-latent contract the spec solver asserts
// (block.start == compacted latent offset == p + G) holds. The caller fills the
// blocks separately and passes them to laplace_mode_spec_dense_solve.
//
// `y` aliases the caller's NumericVector (no copy): resp.y points into it, so it
// must outlive the solve. re_group_1based is the per-obs 1-based RE index (length
// N when n_re_groups > 0, else empty). weights (length N) is optional per-obs
// likelihood weight, borrowed. offset (length N) is an optional fixed additive
// term on the linear predictor (eta = offset + X beta + ...), copied into the
// single process's ProcessData::offset and read by compute_eta_spec /
// precompute_generic_fixed_eta; nullptr leaves it empty (treated as zero).
inline void build_spec_family_inputs(
    SpecFamilyInputs& in,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const std::vector<int>& re_group_1based,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi, double sigma_beta,
    int n_block_latent,
    const double* weights = nullptr,
    const double* offset = nullptr
) {
    const int N = y.size();
    const int p = X.ncol();
    const bool has_re = (n_re_groups > 0) && ((int)re_group_1based.size() == N);

    ProcessData proc;
    proc.p = p;
    proc.X_flat.resize((size_t)N * p);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < p; j++)
            proc.X_flat[(size_t)i * p + j] = X(i, j);
    if (offset != nullptr) proc.offset.assign(offset, offset + N);

    in.spec          = builtin_family_spec(family);
    in.n_trials.assign(n_trials.begin(), n_trials.end());
    in.resp.y        = y.begin();
    in.resp.n_trials = in.n_trials.data();
    in.resp.N        = N;
    in.resp.family   = family;
    in.resp.phi      = phi;
    in.resp.weights  = weights;

    in.data = ModelData();
    in.data.n_processes         = 1;
    in.data.processes.push_back(proc);
    in.data.N                   = N;
    in.data.sigma_beta          = sigma_beta;
    in.data.likelihood_spec     = &in.spec;
    in.data.model_response_data = &in.resp;
    in.data.sharing.init(1);
    if (has_re) {
        in.data.n_re_groups = n_re_groups;
        in.data.re_group.assign(re_group_1based.begin(), re_group_1based.end());
        in.re_group.assign(re_group_1based.begin(), re_group_1based.end());
    }

    in.layout = ParamLayout();
    in.layout.process_beta_start.push_back(0);
    in.layout.process_beta_count.push_back(p);
    int next = p;
    if (has_re) {
        in.layout.has_re   = true;
        in.layout.re_start = next;          // RE param_start == compacted offset
        next += n_re_groups;
        in.layout.re_end   = next;
    }
    next += n_block_latent;                 // GMRF block region [p+G, p+G+M)
    if (has_re) in.layout.log_sigma_re_idx = next++;  // hyperparam AFTER latent
    in.layout.total_params = next;
}

// Convert a random-effect term's `pack` (the value the family-enum kernel
// consumed) into the spec RE parameterization (log-SD slots + tanh-Cholesky raw
// values). Single source for the exact Sigma the spec solver must reconstruct.
//
//   diagonal   (correlated = false): pack is length q of marginal SDs ->
//       log_sigma[c] = log(pack[c]); tanh_raw empty.
//   correlated (correlated = true) : pack is the column-major lower-triangular
//       Cholesky of Sigma (Sigma = L L', as packed by .re_cov_spec). Decompose
//       Sigma = D R D with D = diag(sd), R the correlation matrix: the spec
//       stores log(sd) and the strict-lower of the correlation Cholesky
//       L_R = D^{-1} L through atanh (build_chol_L re-applies tanh and recovers
//       the diagonal as sqrt(1 - rowsumsq), exactly inverting this). The
//       resulting Q = (D L_R L_R' D)^{-1} and log|Q| match the family-enum
//       kernel's to machine precision.
//
// log_sigma is filled with q entries; tanh_raw with q(q-1)/2 entries
// (row-major strict-lower: (1,0),(2,0),(2,1),...), matching build_chol_L's order.
inline void pack_to_spec_re_params(
    const double* pack, int q, bool correlated,
    std::vector<double>& log_sigma,    // out, length q
    std::vector<double>& tanh_raw      // out, length q(q-1)/2 (empty if !correlated)
) {
    log_sigma.assign(q, 0.0);
    tanh_raw.clear();
    if (!correlated || q == 1) {
        for (int c = 0; c < q; c++) log_sigma[c] = std::log(pack[c]);
        return;
    }
    // Unpack the column-major lower-triangular Cholesky L (Sigma = L L').
    std::vector<double> L((size_t)q * q, 0.0);
    {
        int idx = 0;
        for (int c = 0; c < q; c++)
            for (int r = c; r < q; r++)
                L[(size_t)r * q + c] = pack[idx++];
    }
    // Marginal SDs: sd_r = sqrt(Sigma_rr) = sqrt(sum_{k<=r} L[r,k]^2).
    std::vector<double> sd(q, 0.0);
    for (int r = 0; r < q; r++) {
        double s = 0.0;
        for (int k = 0; k <= r; k++) s += L[(size_t)r * q + k] * L[(size_t)r * q + k];
        sd[r] = std::sqrt(s);
        log_sigma[r] = std::log(sd[r]);
    }
    // Correlation Cholesky L_R = D^{-1} L (row r scaled by 1/sd_r); its strict
    // lower entries are the tanh of the stored raw values.
    tanh_raw.reserve((size_t)q * (q - 1) / 2);
    for (int row = 1; row < q; row++) {
        for (int col = 0; col < row; col++) {
            double l_rc = L[(size_t)row * q + col] / sd[row];
            // Guard the atanh domain against a (near-)singular correlation.
            if (l_rc >  0.999999) l_rc =  0.999999;
            if (l_rc < -0.999999) l_rc = -0.999999;
            tanh_raw.push_back(std::atanh(l_rc));
        }
    }
}

} // namespace tulpa

#endif // TULPA_LAPLACE_SPEC_FIT_H
