// laplace_spec_fit.cpp
// Definitions of the spec-fit input marshalling declared in laplace_spec_fit.h.

#include <RcppEigen.h>
#include "laplace_spec_fit.h"

namespace tulpa {

const SubspaceDebiasOptions* unwrap_debias(
    const Rcpp::Nullable<Rcpp::List>& debias,
    SubspaceDebiasOptions& storage
) {
    if (debias.isNull()) return nullptr;
    Rcpp::List spec(debias);
    if (!spec.containsElementNamed("idx")) return nullptr;
    Rcpp::IntegerVector idx_r = spec["idx"];
    if (idx_r.size() == 0) return nullptr;
    storage.idx.resize(idx_r.size());
    for (int k = 0; k < idx_r.size(); k++) {
        if (idx_r[k] == NA_INTEGER || idx_r[k] < 1) {
            Rcpp::stop("debias$idx[%d] is %s; latent indices are 1-based.",
                       k + 1,
                       idx_r[k] == NA_INTEGER ? "NA"
                                              : std::to_string(idx_r[k]).c_str());
        }
        storage.idx[k] = idx_r[k] - 1;
    }
    if (spec.containsElementNamed("n_iter")) storage.n_iter = Rcpp::as<int>(spec["n_iter"]);
    if (spec.containsElementNamed("warmup")) storage.n_warmup = Rcpp::as<int>(spec["warmup"]);
    if (spec.containsElementNamed("thin"))   storage.thin = Rcpp::as<int>(spec["thin"]);
    return &storage;
}

void build_spec_family_inputs(
    SpecFamilyInputs& in,
    const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const std::vector<int>& re_group_1based,
    int n_re_groups, double sigma_re,
    const std::string& family, double phi, double sigma_beta,
    int n_block_latent,
    const double* weights,
    const double* offset) {
    const int N = y.size();
    const int p = X.ncol();
    check_arg_length(X.nrow(), N, "nrow(X)", "length(y)");
    check_arg_length(n_trials.size(), N, "length(n_trials)", "length(y)");
    // A mismatched grouping vector would otherwise drop the whole RE term and
    // return a converged fixed-effects fit for a model nobody asked for.
    if (n_re_groups > 0 && (int)re_group_1based.size() != N) {
        Rcpp::stop("length(re_idx) (%d) must equal length(y) (%d) when "
                   "n_re_groups (%d) is positive.",
                   (int)re_group_1based.size(), N, n_re_groups);
    }
    // Every caller writes log(sigma_re) into the params vector, so a
    // non-positive scale starts the solve from a non-finite point.
    if (n_re_groups > 0 && (!(sigma_re > 0.0) || !std::isfinite(sigma_re))) {
        Rcpp::stop("sigma_re (%g) must be finite and positive.", sigma_re);
    }
    // The group index is 1-based with 0 meaning "no random effect on this row";
    // the RE scatter reads re_group[i] - 1 with only the lower bound tested.
    if (n_re_groups > 0) {
        for (int i = 0; i < N; i++) {
            const int g = re_group_1based[i];
            if (g < 0 || g > n_re_groups) {
                Rcpp::stop("re_idx[%d] (%d) must be 0 (no random effect) or a "
                           "1-based index in [1, %d].", i + 1, g, n_re_groups);
            }
        }
    }
    const bool has_re = (n_re_groups > 0);

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
    in.resp.prepare();

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

} // namespace tulpa
