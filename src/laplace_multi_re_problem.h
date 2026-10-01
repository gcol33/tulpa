// laplace_multi_re_problem.h
// The multi-term random-effect Laplace problem, marshalled once from its R
// inputs: the ModelData / ParamLayout / LikelihoodSpec / response the spec
// solver consumes, and the storage those borrow from.
//
// Every argument here describes the MODEL -- response, designs, grouping,
// family, priors -- and none of it moves with the random-effect covariance, so
// a caller that evaluates one model at many covariances builds one problem and
// asks it for a parameter vector per covariance (params_at). The single-point
// fit (cpp_laplace_fit_multi_re) is the one-covariance case of the same object.
//
// The object borrows into itself (data.likelihood_spec, data.model_response_data,
// resp.y, resp.n_trials, resp.weights), so it is neither copyable nor movable.

#ifndef TULPA_LAPLACE_MULTI_RE_PROBLEM_H
#define TULPA_LAPLACE_MULTI_RE_PROBLEM_H

#include <Rcpp.h>
#include "laplace_core.h"
#include "laplace_re_priors.h"
#include "laplace_spec_fit.h"
#include "re_structure.h"
#include <string>
#include <vector>

namespace tulpa {

class MultiReProblem {
public:
    // `re_sigma_list` fixes each term's covariance FORM (a packed Cholesky of
    // length q(q+1)/2 marks a correlated term, q marginal SDs a diagonal one);
    // its values are read only by params_at.
    MultiReProblem(
        Rcpp::NumericVector y, Rcpp::IntegerVector n,
        Rcpp::NumericMatrix X,
        Rcpp::List re_idx_list,
        Rcpp::IntegerVector re_ngroups,
        Rcpp::List re_sigma_list,
        const std::string& family, double phi,
        Rcpp::Nullable<Rcpp::List> re_Z_list,
        Rcpp::Nullable<Rcpp::IntegerVector> re_ncoefs,
        Rcpp::Nullable<Rcpp::NumericVector> weights,
        Rcpp::Nullable<Rcpp::NumericVector> offset,
        Rcpp::Nullable<Rcpp::NumericVector> beta_prior_mean,
        Rcpp::Nullable<Rcpp::NumericVector> beta_prior_sd,
        double phi2,
        Rcpp::Nullable<Rcpp::NumericMatrix> X_zi,
        double zi_prior_sd
    ) : y_(y) {
        const int N = y.size();
        const int p = X.ncol();
        K_ = re_ngroups.size();

        // This entry marshals ModelData by hand rather than through
        // build_spec_family_inputs, so the length checks that path carries are
        // made here. Everything below indexes by N or by K, both read off
        // arguments the caller supplies separately.
        check_arg_length(X.nrow(), N, "nrow(X)", "length(y)");
        check_arg_length(n.size(), N, "length(n)", "length(y)");
        check_arg_length(re_sigma_list.size(), K_, "length(re_sigma_list)",
                         "length(re_ngroups)");

        // --- Zero inflation: the spec-Laplace shim passes 0.0 for the
        //     `logit_zi` callback argument, so the ZI linear predictor is
        //     carried as a second PROCESS (eta[1] = X_zi beta_zi) rather than
        //     through data.zi_type's side channel. data.zi_type therefore stays
        //     NONE here -- it means "the side channel is live", which is the
        //     sampler paths' mechanism, not this one. The beta block becomes
        //     [beta_count | beta_zi]. ---
        Rcpp::NumericMatrix X_zi_mat;
        int p_zi = 0;
        if (X_zi.isNotNull()) {
            X_zi_mat = Rcpp::as<Rcpp::NumericMatrix>(X_zi);
            if (X_zi_mat.nrow() != N) {
                Rcpp::stop("X_zi has %d rows but y has length %d.",
                           (int)X_zi_mat.nrow(), N);
            }
            p_zi = X_zi_mat.ncol();
        }
        const bool has_zi = (p_zi > 0);
        if (has_zi && !zi::compiled_zi_supported(family)) {
            Rcpp::stop("family '%s' has no compiled zero-inflated kernel "
                       "(supported: %s).",
                       family.c_str(),
                       zi::compiled_zi_supported_families().c_str());
        }
        p_total_ = p + p_zi;

        // --- Optional per-coef Gaussian fixed-effect prior (mean / sd -> tau). ---
        bool has_bp = false;
        if (beta_prior_sd.isNotNull()) {
            Rcpp::NumericVector sd = Rcpp::as<Rcpp::NumericVector>(beta_prior_sd);
            if ((int)sd.size() != p) {
                Rcpp::stop("beta_prior_sd has length %d but X has %d columns.",
                           (int)sd.size(), p);
            }
            bp_.tau.resize(p);
            for (int j = 0; j < p; j++) {
                if (!(sd[j] > 0.0) || ISNAN(sd[j])) {
                    Rcpp::stop("beta_prior_sd[%d] must be positive (Inf allowed = no penalty).", j + 1);
                }
                // sd = +Inf -> tau = 0 (no penalty); BetaPrior::tau_at returns
                // it and the spec beta-prior treats tau = 0 as a no-op.
                bp_.tau[j] = R_finite(sd[j]) ? 1.0 / (sd[j] * sd[j]) : 0.0;
            }
            has_bp = true;
        }
        if (beta_prior_mean.isNotNull()) {
            Rcpp::NumericVector mn = Rcpp::as<Rcpp::NumericVector>(beta_prior_mean);
            if ((int)mn.size() != p) {
                Rcpp::stop("beta_prior_mean has length %d but X has %d columns.",
                           (int)mn.size(), p);
            }
            bp_.mean.assign(mn.begin(), mn.end());
            has_bp = true;
        }
        // The beta prior is supplied for the count block only; the ZI block is
        // appended with a weakly-informative N(0, zi_prior_sd^2) matching
        // ModelData::zi_prior_sd, which keeps the logit identified when a level
        // has no zeros (where the likelihood alone would send beta_zi to -Inf).
        // It is applied whenever the ZI block exists: it is the ZI block's own
        // prior, not an extension of the caller's count-block prior, and the
        // sampler paths carry it unconditionally through
        // ModelData::zi_prior_sd. Materializing `tau` here means the count block
        // keeps DEFAULT_TAU_BETA explicitly rather than through the empty-vector
        // fallback, which cannot represent two different precisions.
        if (has_zi) {
            if (!(zi_prior_sd > 0.0) || ISNAN(zi_prior_sd)) {
                Rcpp::stop("zi_prior_sd must be positive (Inf allowed = no penalty).");
            }
            const double tau_zi = R_finite(zi_prior_sd)
                ? 1.0 / (zi_prior_sd * zi_prior_sd) : 0.0;
            if (bp_.tau.empty()) bp_.tau.assign(p, DEFAULT_TAU_BETA);
            bp_.tau.resize(p_total_, tau_zi);
            if (!bp_.mean.empty()) bp_.mean.resize(p_total_, 0.0);
        }
        use_bp_ = has_bp || has_zi;

        // --- Per-obs likelihood weights (borrowed; the family adapter scales
        //     the score + Fisher weight). Stable storage outlives the solve. ---
        const double* w_ptr = nullptr;
        if (weights.isNotNull()) {
            Rcpp::NumericVector wv = Rcpp::as<Rcpp::NumericVector>(weights);
            check_arg_length(wv.size(), N, "length(weights)", "length(y)");
            w_store_.assign(wv.begin(), wv.end());
            w_ptr = w_store_.data();
        }

        // --- n_coefs per term (default 1 = intercept only). ---
        ncoefs_.assign(K_, 1);
        if (re_ncoefs.isNotNull()) {
            Rcpp::IntegerVector nc = Rcpp::as<Rcpp::IntegerVector>(re_ncoefs);
            check_arg_length(nc.size(), K_, "length(re_ncoefs)",
                             "length(re_ngroups)");
            for (int t = 0; t < K_; t++) ncoefs_[t] = nc[t];
        }

        // --- Process design + built-in family spec + response. ---
        ProcessData proc;
        proc.p = p;
        proc.X_flat.resize((size_t)N * p);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < p; j++)
                proc.X_flat[(size_t)i * p + j] = X(i, j);
        if (offset.isNotNull()) {
            Rcpp::NumericVector ov = Rcpp::as<Rcpp::NumericVector>(offset);
            check_arg_length(ov.size(), N, "length(offset)", "length(y)");
            proc.offset.assign(ov.begin(), ov.end());
        }

        ProcessData proc_zi;
        if (has_zi) {
            proc_zi.p = p_zi;
            proc_zi.X_flat.resize((size_t)N * p_zi);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < p_zi; j++)
                    proc_zi.X_flat[(size_t)i * p_zi + j] = X_zi_mat(i, j);
        }

        spec_ = builtin_family_spec(family, has_zi);
        n_trials_.assign(n.begin(), n.end());
        resp_.y        = y_.begin();
        resp_.n_trials = n_trials_.data();
        resp_.N        = N;
        resp_.family   = family;
        resp_.phi      = phi;
        resp_.phi2     = phi2;   // NA_REAL is a NaN => family default (e.g. t df = 4)
        resp_.weights  = w_ptr;
        resp_.prepare();

        data_.n_processes         = has_zi ? 2 : 1;
        data_.processes.push_back(proc);
        if (has_zi) data_.processes.push_back(proc_zi);
        data_.N                   = N;
        data_.sigma_beta          = 100.0;   // default ridge (tau = 1e-4); overridden by bp
        data_.likelihood_spec     = &spec_;
        data_.model_response_data = &resp_;
        data_.sharing.init(data_.n_processes);
        // Random effects enter the count predictor only. A ZI predictor with
        // its own random effects is a separate feature; sharing the count RE
        // into it would silently impose equal effects on both, which is not
        // the model.
        if (has_zi) data_.sharing.re[1] = false;

        // --- Multi-term RE structure (shared marshalling, re_structure.h). ---
        data_.re_parameterization = 1;       // unused on the centered Laplace path
        // Correlated when a packed Sigma-Cholesky (length q(q+1)/2) is
        // supplied; otherwise diagonal (length-q marginal SDs).
        std::vector<bool> corr_flags(K_, false);
        for (int t = 0; t < K_; t++) {
            Rcpp::NumericVector sig = Rcpp::as<Rcpp::NumericVector>(re_sigma_list[t]);
            corr_flags[t] = (ncoefs_[t] > 1) &&
                ((int)sig.size() == ncoefs_[t] * (ncoefs_[t] + 1) / 2);
        }
        populate_re_structure(
            data_, N, re_idx_list,
            std::vector<int>(re_ngroups.begin(), re_ngroups.end()),
            ncoefs_, re_Z_list, corr_flags);

        // --- ParamLayout: [beta | sigma slots | chol slots | RE effects], the
        //     schema build_latent_layout reads (mirrors hmc_param_layout). ---
        layout_.process_beta_start.push_back(0);
        layout_.process_beta_count.push_back(p);
        if (has_zi) {
            layout_.process_beta_start.push_back(p);
            layout_.process_beta_count.push_back(p_zi);
        }
        int next = p_total_;
        layout_.has_re                   = (K_ > 0);   // fixed-effects-only when no RE terms
        layout_.has_re_slopes            = data_.has_re_slopes;
        layout_.has_re_correlated_slopes = data_.has_re_correlated_slopes;
        layout_.log_sigma_re_multi.resize(K_);
        layout_.log_sigma_re_slopes.resize(K_);
        layout_.re_start_multi.resize(K_);
        layout_.re_end_multi.resize(K_);
        layout_.re_n_coefs_multi.resize(K_);
        layout_.re_correlated_multi.resize(K_);
        layout_.chol_re_start_multi.assign(K_, -1);
        layout_.chol_re_end_multi.assign(K_, -1);
        for (int t = 0; t < K_; t++) {
            const int q = ncoefs_[t];
            layout_.re_n_coefs_multi[t]    = q;
            layout_.re_correlated_multi[t] = data_.re_correlated[t];
            layout_.log_sigma_re_slopes[t].resize(q);
            for (int c = 0; c < q; c++) layout_.log_sigma_re_slopes[t][c] = next++;
            layout_.log_sigma_re_multi[t]  = layout_.log_sigma_re_slopes[t][0];
        }
        for (int t = 0; t < K_; t++) {
            if (data_.re_n_chol[t] > 0) {
                layout_.chol_re_start_multi[t] = next;
                next += data_.re_n_chol[t];
                layout_.chol_re_end_multi[t] = next;
            }
        }
        for (int t = 0; t < K_; t++) {
            layout_.re_start_multi[t] = next;
            next += data_.re_n_groups_multi[t] * data_.re_n_coefs[t];
            layout_.re_end_multi[t] = next;
        }
        layout_.log_sigma_re_idx = K_ > 0 ? layout_.log_sigma_re_multi[0] : -1;
        layout_.re_start = K_ > 0 ? layout_.re_start_multi[0] : -1;
        layout_.re_end   = K_ > 0 ? layout_.re_end_multi[0]   : -1;
        layout_.total_params = next;
    }

    MultiReProblem(const MultiReProblem&) = delete;
    MultiReProblem& operator=(const MultiReProblem&) = delete;

    const ModelData&   data()   const { return data_; }
    const ParamLayout& layout() const { return layout_; }
    const LikelihoodSpec& spec() const { return spec_; }
    const void* response() const { return &resp_; }
    const BetaPrior* beta_prior() const { return use_bp_ ? &bp_ : nullptr; }

    // The solver's parameter vector at one covariance: the hyperparameter slots
    // from each term's `pack` (marginal SDs or a packed Sigma-Cholesky, converted
    // to the spec log-Cholesky parameterization), the latent slots from the
    // optional warm start (mode order [beta | per-term RE effects]), zero
    // otherwise. Errors through R, so it is called outside any parallel region.
    std::vector<double> params_at(
        Rcpp::List re_sigma_list,
        Rcpp::Nullable<Rcpp::NumericVector> x_init
    ) const {
        check_arg_length(re_sigma_list.size(), K_, "length(re_sigma_list)",
                         "length(re_ngroups)");
        std::vector<double> params(layout_.total_params, 0.0);
        std::vector<double> log_sigma, tanh_raw;
        for (int t = 0; t < K_; t++) {
            Rcpp::NumericVector sig = Rcpp::as<Rcpp::NumericVector>(re_sigma_list[t]);
            const int q = ncoefs_[t];
            const R_xlen_t need = data_.re_correlated[t]
                ? (R_xlen_t)q * (q + 1) / 2
                : (R_xlen_t)q;
            if (sig.size() != need) {
                Rcpp::stop("re_sigma_list[[%d]] must have length %d for a %s "
                           "term with %d coefficient(s), got %d.",
                           t + 1, (int)need,
                           data_.re_correlated[t] ? "correlated" : "diagonal",
                           q, (int)sig.size());
            }
            const std::string label =
                "re_sigma_list[[" + std::to_string(t + 1) + "]]";
            pack_to_spec_re_params(sig.begin(), q, data_.re_correlated[t],
                                   log_sigma, tanh_raw, label.c_str());
            for (int c = 0; c < q; c++)
                params[layout_.log_sigma_re_slopes[t][c]] = log_sigma[c];
            if (data_.re_n_chol[t] > 0)
                for (int j = 0; j < data_.re_n_chol[t]; j++)
                    params[layout_.chol_re_start_multi[t] + j] = tanh_raw[j];
        }
        if (x_init.isNotNull()) {
            Rcpp::NumericVector xi = Rcpp::as<Rcpp::NumericVector>(x_init);
            // The warm start is [beta | per-term RE effects]; the RE half is
            // only known once populate_re_structure has resolved each term's
            // groups and coefficients, which is why the length is checked here.
            R_xlen_t need = p_total_;
            for (int t = 0; t < K_; t++)
                need += (R_xlen_t)data_.re_n_groups_multi[t] * data_.re_n_coefs[t];
            check_arg_length(xi.size(), need, "length(x_init)",
                             "the latent dimension");
            int off = 0;
            for (int j = 0; j < p_total_; j++) params[j] = xi[off++];
            for (int t = 0; t < K_; t++) {
                const int sz = data_.re_n_groups_multi[t] * data_.re_n_coefs[t];
                for (int j = 0; j < sz; j++)
                    params[layout_.re_start_multi[t] + j] = xi[off++];
            }
        }
        return params;
    }

private:
    Rcpp::NumericVector y_;
    int K_ = 0;
    int p_total_ = 0;
    std::vector<int> ncoefs_;
    BetaPrior bp_;
    bool use_bp_ = false;
    std::vector<double> w_store_;
    std::vector<int> n_trials_;
    LikelihoodSpec spec_;
    BuiltinFamilyResponse resp_;
    ModelData data_;
    ParamLayout layout_;
};

} // namespace tulpa

#endif // TULPA_LAPLACE_MULTI_RE_PROBLEM_H
