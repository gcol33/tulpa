// tulpa_spde_sampler.cpp
//
// SPDE-via-NUTS sampler. Conditions on a fixed Matern parameter pair
// (kappa, tau_spde) and samples the (beta, w_mesh, log_phi) latent block
// jointly via tulpa's full NUTS backend.
//
// Phase 1 architecture (gcol33/tulpa#b): the SPDE field is a first-class
// SpatialType::SPDE — w_mesh lives in ParamLayout::spde_w_start..end and
// the Q-prior + A-projection are wired into compute_log_post_generic via
// tulpa_priors_spde.h and add_generic_spatial_effect. The likelihood
// callback is a plain GLM: it only sees eta (which already contains the
// A * w contribution) and reads log_phi from the extra-params slot.
//
// Layout:
//   params[0 .. p)                  beta (process 0)
//   params[spde_w_start .. spde_w_end)  w_mesh (mesh-node effects)
//   params[extra_offset]            log_phi (sampled jointly)
//
// Hyperparameter integration over (kappa, tau_spde) is left to an outer
// loop (the existing nested-Laplace grid in cpp_nested_laplace_spde, or
// a future joint-NUTS variant — gcol33/tulpa#a — that extends arena AD
// with a sparse-Cholesky adjoint). Within a single call, (kappa, tau)
// are fixed; Q is built once and cached on ModelData::spde_data.

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include "hmc_sampler.h"
#include "spde_qbuilder.h"
#include "spde_nc_apply.h"
#include "tulpa/likelihood.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

namespace {

// Family enum local to this TU — selects the per-obs log-likelihood and
// the role of `log_phi`. Kept C++-local because the user-facing R wrapper
// translates a string into this code. The conventions match
// laplace_family_link.h (neg_binomial_2 size, Gamma shape, Beta precision).
enum class SpdeFamily : int {
    GAUSSIAN = 0,    // log_phi = log(sigma);          y ~ N(mu, sigma^2)
    POISSON  = 1,    // log_phi unused;                y ~ Poisson(exp(eta))
    BINOMIAL = 2,    // log_phi unused;                y ~ Binomial(n_trials, sigmoid(eta))
    GAMMA    = 3,    // log_phi = log(shape);          y ~ Gamma(shape, shape/mu), mu = exp(eta)
    NEGBIN   = 4,    // log_phi = log(size r);         var = mu + mu^2/r, mu = exp(eta)
    BETA     = 5     // log_phi = log(precision phi);  mu = sigmoid(eta), a = mu*phi, b = (1-mu)*phi
};

// =====================================================================
// SpdeGlmData: response + per-family precomputed constants. Spatial
// structure (Q, A, FEM matrices) now lives on ModelData::spde_data;
// the likelihood callback no longer touches it.
// =====================================================================
struct SpdeGlmData {
    SpdeFamily family;
    std::vector<double> y;
    std::vector<int>    n_trials;     // binomial only
    std::vector<double> log_y_fact;   // log(y!) for poisson / negbin constant
    std::vector<double> log_choose;   // lchoose(n, y) for binomial constant
    std::vector<double> log_y;        // log(y) for gamma / beta
    std::vector<double> log_1my;      // log(1 - y) for beta

    // log_phi prior: only used by Gaussian (log_sigma), Gamma (log_shape),
    // NegBin (log_size), Beta (log_precision). The R wrapper picks a
    // sensible default per-family; for Poisson / Binomial log_phi is
    // pinned tightly and ignored downstream.
    double log_phi_prior_sd = 3.0;
};

// =====================================================================
// Per-observation log-likelihood (templated for AD).
//
// Phase 1: eta[0] already carries beta + spatial (A*w) + offset from
// compute_log_post_generic — we just dispatch on family. log_phi is the
// only extra param this likelihood owns; it sits at layout.extra_offset.
// =====================================================================
template<typename T>
T spde_glm_likelihood(
    int i,
    const T* eta,
    const T& /*logit_zi*/,
    const T& /*logit_oi*/,
    const std::vector<T>& params,
    const ModelData& /*data*/,
    const ParamLayout& layout,
    const void* model_data
) {
    using std::exp;
    using std::log;
    using std::lgamma;

    const auto* sd = static_cast<const SpdeGlmData*>(model_data);
    const T eta_i = eta[0];

    switch (sd->family) {
        case SpdeFamily::GAUSSIAN: {
            T log_sigma = params[layout.extra_offset];
            T resid     = T(sd->y[i]) - eta_i;
            T inv_sigma = exp(T(0.0) - log_sigma);
            return T(0.0) - log_sigma - T(0.5) * resid * resid * inv_sigma * inv_sigma;
        }
        case SpdeFamily::POISSON: {
            T lambda = exp(eta_i);
            return T(sd->y[i]) * eta_i - lambda - T(sd->log_y_fact[i]);
        }
        case SpdeFamily::BINOMIAL: {
            T one_plus_exp = T(1.0) + exp(eta_i);
            return T(sd->log_choose[i]) + T(sd->y[i]) * eta_i
                 - T(static_cast<double>(sd->n_trials[i])) * log(one_plus_exp);
        }
        case SpdeFamily::GAMMA: {
            T log_phi = params[layout.extra_offset];
            T phi     = exp(log_phi);
            T neg_eta = T(0.0) - eta_i;
            return phi * log_phi - lgamma(phi)
                 + (phi - T(1.0)) * T(sd->log_y[i])
                 - phi * eta_i
                 - phi * T(sd->y[i]) * exp(neg_eta);
        }
        case SpdeFamily::NEGBIN: {
            T log_phi = params[layout.extra_offset];
            T phi     = exp(log_phi);
            T mu      = exp(eta_i);
            T y_i     = T(sd->y[i]);
            return lgamma(y_i + phi) - lgamma(phi)
                 - T(sd->log_y_fact[i])
                 + phi * log_phi
                 - (y_i + phi) * log(mu + phi)
                 + y_i * eta_i;
        }
        case SpdeFamily::BETA: {
            T log_phi = params[layout.extra_offset];
            T phi     = exp(log_phi);
            T mu      = T(1.0) / (T(1.0) + exp(T(0.0) - eta_i));
            T a       = mu * phi;
            T b       = (T(1.0) - mu) * phi;
            return lgamma(phi) - lgamma(a) - lgamma(b)
                 + (a - T(1.0)) * T(sd->log_y[i])
                 + (b - T(1.0)) * T(sd->log_1my[i]);
        }
    }
    return T(0.0);
}

// =====================================================================
// log_phi prior: Normal(0, sd_log_phi). Returned up to additive constants.
// The w_mesh prior moved to tulpa_priors_spde.h and is now wired through
// the structured HMC path (compute_spde_prior in initialize_generic_state).
// =====================================================================
static double spde_glm_log_phi_prior_double(
    const std::vector<double>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    const auto* sd = static_cast<const SpdeGlmData*>(model_data);
    double log_phi = params[layout.extra_offset];
    double sd_lp   = sd->log_phi_prior_sd;
    return -0.5 * (log_phi / sd_lp) * (log_phi / sd_lp);
}

static tulpa::arena::Var spde_glm_log_phi_prior_arena(
    const std::vector<tulpa::arena::Var>& params,
    const ParamLayout& layout,
    const void* model_data
) {
    using tulpa::arena::Var;
    const auto* sd = static_cast<const SpdeGlmData*>(model_data);
    Var log_phi = params[layout.extra_offset];
    double sd_lp = sd->log_phi_prior_sd;
    Var z = log_phi * Var(1.0 / sd_lp);
    return Var(-0.5) * z * z;
}

}  // anonymous namespace

namespace tulpa_hmc {
    HMCResultCpp run_hmc_chain_cpp(
        const std::vector<double>& q_init,
        const ModelData& data,
        const ParamLayout& layout,
        int n_iter, int n_warmup, int L, int chain_id,
        unsigned int seed, bool verbose, int max_treedepth,
        MassMatrixType metric_type, double adapt_delta, int riemannian,
        const std::vector<double>& inv_metric_init);
}

// =====================================================================
// Rcpp entry point.
// =====================================================================
// [[Rcpp::export]]
Rcpp::List cpp_tulpa_fit_spde_nuts(
    Rcpp::NumericVector y_r,
    Rcpp::IntegerVector n_trials_r,
    Rcpp::NumericMatrix X_r,
    Rcpp::NumericVector A_x, Rcpp::IntegerVector A_i, Rcpp::IntegerVector A_p,
    int n_obs, int n_mesh,
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x, Rcpp::IntegerVector G1_i, Rcpp::IntegerVector G1_p,
    double kappa, double tau_spde,
    std::string family,
    int alpha = 2,
    double nu = -1.0,
    double sigma_beta = 10.0,
    double log_phi_prior_sd = 3.0,
    double log_phi_init = 0.0,
    int n_iter = 2000,
    int n_warmup = 1000,
    int max_treedepth = 10,
    double adapt_delta = 0.8,
    int seed = 42,
    bool verbose = false,
    Rcpp::Nullable<Rcpp::NumericVector> rational_poles_nullable = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> rational_weights_nullable = R_NilValue,
    bool joint_hypers = false,
    double prior_range_0     = -1.0,
    double prior_range_alpha = -1.0,
    double prior_sigma_0     = -1.0,
    double prior_sigma_alpha = -1.0,
    double log_kappa_init    = 0.0,
    double log_tau_init      = 0.0,
    Rcpp::Nullable<Rcpp::NumericVector> Q_precomp_x = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> Q_precomp_i = R_NilValue,
    Rcpp::Nullable<Rcpp::IntegerVector> Q_precomp_p = R_NilValue
) {
    const int N = y_r.size();
    const int p = X_r.ncol();
    if (N != n_obs) Rcpp::stop("length(y) must equal n_obs");
    if (N == 0 || p == 0) Rcpp::stop("y and X must be non-empty");
    if (n_mesh <= 0) Rcpp::stop("n_mesh must be positive");

    SpdeFamily fam;
    if      (family == "gaussian")       fam = SpdeFamily::GAUSSIAN;
    else if (family == "poisson")        fam = SpdeFamily::POISSON;
    else if (family == "binomial")       fam = SpdeFamily::BINOMIAL;
    else if (family == "gamma")          fam = SpdeFamily::GAMMA;
    else if (family == "neg_binomial_2") fam = SpdeFamily::NEGBIN;
    else if (family == "beta")           fam = SpdeFamily::BETA;
    else Rcpp::stop("Unsupported family for tulpa_nuts_spde: '%s'. "
                    "Supported: gaussian, poisson, binomial, gamma, "
                    "neg_binomial_2, beta.", family.c_str());

    // Family-specific input validation.
    if (fam == SpdeFamily::POISSON || fam == SpdeFamily::NEGBIN) {
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] < 0.0 ||
                std::abs(y_r[i] - std::round(y_r[i])) > 1e-9) {
                Rcpp::stop("%s family requires non-negative integer y",
                           family.c_str());
            }
        }
    } else if (fam == SpdeFamily::BINOMIAL) {
        if (n_trials_r.size() != N) {
            Rcpp::stop("binomial family requires n_trials of length n_obs");
        }
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] < 0.0 || y_r[i] > n_trials_r[i] ||
                std::abs(y_r[i] - std::round(y_r[i])) > 1e-9) {
                Rcpp::stop("binomial family requires y in [0, n_trials]");
            }
        }
    } else if (fam == SpdeFamily::GAMMA) {
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] <= 0.0) {
                Rcpp::stop("gamma family requires strictly positive y");
            }
        }
    } else if (fam == SpdeFamily::BETA) {
        for (int i = 0; i < N; i++) {
            if (!R_finite(y_r[i]) || y_r[i] <= 0.0 || y_r[i] >= 1.0) {
                Rcpp::stop("beta family requires y strictly in (0, 1)");
            }
        }
    }

    // PC prior anchors. Joint mode requires all four anchors strictly
    // positive (P(range < r_0) = alpha_r, P(sigma > s_0) = alpha_s). If
    // any anchor is non-positive, the prior is improper-flat (floor-check
    // / gradient-verification only — never production sampling).
    if (joint_hypers) {
        const bool pc_disabled = (prior_range_0     <= 0.0 ||
                                   prior_range_alpha <= 0.0 ||
                                   prior_range_alpha >= 1.0 ||
                                   prior_sigma_0     <= 0.0 ||
                                   prior_sigma_alpha <= 0.0 ||
                                   prior_sigma_alpha >= 1.0);
        if (pc_disabled) {
            Rcpp::warning("joint_hypers=TRUE with an incomplete PC prior "
                          "(one or more of prior_range_0, prior_range_alpha, "
                          "prior_sigma_0, prior_sigma_alpha is non-positive "
                          "or out of (0,1)). The hyper prior reverts to "
                          "improper-flat; this is intended only for "
                          "gradient-verification runs, not production.");
        }
    }

    // Build SPDE Q at the supplied (kappa, tau_spde). Q + per-row A get
    // marshalled into ModelData::spde_data so the structured HMC path
    // owns them; nothing SPDE-specific stays in the likelihood spec.
    //
    // Joint-NUTS mode skips this: Q is rebuilt per gradient evaluation
    // inside SpdeNcTransform from the FEM matrices, and the cached Q on
    // SpdeModelData is unused by compute_spde_prior(joint=true). The
    // rational poles/weights are still plumbed through to SpdeModelData
    // so the lazy NC-transform builder picks them up.
    tulpa::SpdeQBuilder qb;
    std::vector<double> rat_poles, rat_weights;
    const bool use_rational = rational_poles_nullable.isNotNull() &&
                               rational_weights_nullable.isNotNull();
    if (use_rational) {
        rat_poles   = Rcpp::as<std::vector<double>>(rational_poles_nullable);
        rat_weights = Rcpp::as<std::vector<double>>(rational_weights_nullable);
        if (rat_poles.size() != rat_weights.size() || rat_poles.empty()) {
            Rcpp::stop("rational_poles / rational_weights must be non-empty "
                       "and of equal length.");
        }
    }
    // Precomputed-Q path (fractional-nu fixed-hyper, gcol33/tulpa#85): the
    // current operator-based rational field (BRASIL roots, #71) assembles its
    // precision Q = Pl' C^{-1} Pl in R (.spde_assemble_at), which the stale
    // poles/weights QBuilder rebuild cannot reproduce. When the caller passes
    // the precomputed Q (and A carries the rational A_eff = A Pr), use it
    // directly: the sampler then draws the auxiliary weights x ~ N(0, Q^{-1})
    // with eta = X beta + A_eff x, the field being u = Pr x (mapped back in R).
    const bool use_precomp_Q = Q_precomp_x.isNotNull() &&
                               Q_precomp_i.isNotNull() && Q_precomp_p.isNotNull();
    if (use_precomp_Q && joint_hypers) {
        Rcpp::stop("Precomputed Q is a fixed-hyper path; joint_hypers must be "
                   "FALSE (the rational roots are not differentiable in kappa).");
    }
    if (!joint_hypers && !use_precomp_Q) {
        qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);
        if (use_rational) {
            qb.rebuild_rational(kappa, tau_spde, rat_poles, rat_weights);
        } else {
            qb.rebuild(kappa, tau_spde, alpha);
        }
    }

    tulpa::ARows a_rows = tulpa::build_A_rows(N, n_mesh, A_x, A_i, A_p);

    // Per-family precomputed constants.
    SpdeGlmData sd;
    sd.family = fam;
    sd.y.assign(y_r.begin(), y_r.end());
    if (fam == SpdeFamily::BINOMIAL) {
        sd.n_trials.assign(n_trials_r.begin(), n_trials_r.end());
        sd.log_choose.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_choose[i] = R::lchoose(static_cast<double>(sd.n_trials[i]), sd.y[i]);
        }
    }
    if (fam == SpdeFamily::POISSON || fam == SpdeFamily::NEGBIN) {
        sd.log_y_fact.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_y_fact[i] = R::lgammafn(sd.y[i] + 1.0);
        }
    }
    if (fam == SpdeFamily::GAMMA || fam == SpdeFamily::BETA) {
        sd.log_y.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_y[i] = std::log(sd.y[i]);
        }
    }
    if (fam == SpdeFamily::BETA) {
        sd.log_1my.resize(N);
        for (int i = 0; i < N; i++) {
            sd.log_1my[i] = std::log(1.0 - sd.y[i]);
        }
    }
    sd.log_phi_prior_sd = log_phi_prior_sd;

    // LikelihoodSpec wiring: just log_phi as the single extra param. The
    // w_mesh block is now a first-class layout slot (SpatialType::SPDE).
    tulpa::LikelihoodSpec spec;
    spec.name              = "spde_" + family;
    spec.n_processes       = 1;
    spec.ll_double         = spde_glm_likelihood<double>;
    spec.ll_arena          = spde_glm_likelihood<tulpa::arena::Var>;
    spec.ll_fwd            = spde_glm_likelihood<::fwd::Dual>;
    spec.n_extra_params    = 1;  // log_phi
    spec.extra_prior       = spde_glm_log_phi_prior_double;
    spec.extra_prior_arena = spde_glm_log_phi_prior_arena;

    ModelData data;
    data.N           = N;
    data.n_processes = 1;
    data.sigma_beta  = sigma_beta;

    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.resize(static_cast<size_t>(N) * p);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < p; j++) {
            proc.X_flat[i * p + j] = X_r(i, j);
        }
    }
    data.processes.push_back(proc);
    data.model_response_data = &sd;
    data.likelihood_spec     = &spec;
    data.sharing.init(1);

    data.zi_type     = tulpa::ZIType::NONE;
    data.p_zi        = 0;
    data.p_oi        = 0;
    data.zi_prior_sd = 1.0;
    data.oi_prior_sd = 1.0;

    // Wire the SPDE block onto ModelData. compute_param_layout sees
    // spatial_type == SPDE and allocates the w_mesh slot; the structured
    // HMC log_post adds -0.5 w' Q w + A*w eta contribution automatically.
    data.spatial_type = tulpa::SpatialType::SPDE;
    data.has_spde     = true;
    auto& sm = data.spde_data;
    sm.n_mesh   = n_mesh;
    // Matern smoothness: caller passes it explicitly (e.g. nu=0.5 for the
    // rational alpha=1.5 case). The legacy fallback nu = alpha - 1 only
    // applies when nu is left unset by the caller (nu < 0), and is
    // exact only for integer-alpha cases.
    sm.nu       = (nu > 0.0) ? nu : static_cast<double>(alpha) - 1.0;
    sm.kappa    = kappa;
    sm.tau_spde = tau_spde;
    sm.alpha    = alpha;
    sm.C0_diag.assign(C0_diag.begin(), C0_diag.end());
    sm.G1_x.assign(G1_x.begin(), G1_x.end());
    sm.G1_i.assign(G1_i.begin(), G1_i.end());
    sm.G1_p.assign(G1_p.begin(), G1_p.end());
    sm.a_rows   = std::move(a_rows);
    if (!joint_hypers) {
        if (use_precomp_Q) {
            Rcpp::NumericVector qx(Q_precomp_x);
            Rcpp::IntegerVector qi(Q_precomp_i), qp(Q_precomp_p);
            sm.Q_x.assign(qx.begin(), qx.end());
            sm.Q_i.assign(qi.begin(), qi.end());
            sm.Q_p.assign(qp.begin(), qp.end());
        } else {
            sm.Q_p.assign(qb.Q_p.begin(), qb.Q_p.end());
            sm.Q_i.assign(qb.Q_i.begin(), qb.Q_i.end());
            sm.Q_x.assign(qb.Q_x.begin(), qb.Q_x.end());
        }
    }
    // Rational coefficients are needed in both modes: fixed-hyper Laplace
    // uses them to rebuild Q at each outer-grid point, and joint-NUTS uses
    // them inside SpdeNcTransform on every gradient call.
    sm.rational_poles   = std::move(rat_poles);
    sm.rational_weights = std::move(rat_weights);
    // log|Q| is constant under fixed hypers and cancels in NUTS, so we
    // leave it at 0.0 — Phase 2 will compute it for prior-comparable
    // log-posteriors when joint hypers move.
    sm.log_det_Q = 0.0;

    // Joint-NUTS state. Activates the non-centered z-parameterisation
    // and the PC prior on (range, sigma); compute_param_layout will then
    // reserve log_kappa_spde_idx / log_tau_spde_idx after the z block.
    sm.joint_hypers      = joint_hypers;
    sm.prior_range_0     = prior_range_0;
    sm.prior_range_alpha = prior_range_alpha;
    sm.prior_sigma_0     = prior_sigma_0;
    sm.prior_sigma_alpha = prior_sigma_alpha;

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);
    int n_params = layout.total_params;

    std::vector<double> init(n_params, 0.0);
    init[layout.extra_offset] = log_phi_init;
    if (joint_hypers) {
        if (layout.log_kappa_spde_idx < 0 || layout.log_tau_spde_idx < 0) {
            Rcpp::stop("Internal error: joint_hypers=true but param layout "
                       "did not reserve log_kappa / log_tau slots. This is a "
                       "bug in compute_param_layout.");
        }
        init[layout.log_kappa_spde_idx] = log_kappa_init;
        init[layout.log_tau_spde_idx]   = log_tau_init;
    }

    tulpa_hmc::HMCResultCpp result = tulpa_hmc::run_hmc_chain_cpp(
        init, data, layout,
        n_iter, n_warmup,
        0,                                  // L=0 -> NUTS
        1,                                  // chain_id
        static_cast<unsigned int>(seed),
        verbose,
        max_treedepth,
        tulpa::MassMatrixType::DIAG,
        adapt_delta,
        0,                                  // riemannian off
        std::vector<double>{}
    );

    int n_sample = result.n_sample;
    Rcpp::NumericMatrix draws(n_sample, n_params);
    for (int s = 0; s < n_sample; s++) {
        const double* row = result.sample_row(s);
        for (int j = 0; j < n_params; j++) {
            draws(s, j) = row[j];
        }
    }

    // Column names. In joint mode the SPDE block is z (non-centered), in
    // legacy mode it is w directly. The transformed w (joint mode) lives
    // in a separate w_draws matrix below.
    const std::string spde_block_name = joint_hypers ? "z" : "w";
    Rcpp::CharacterVector col_names(n_params);
    for (int j = 0; j < p; j++) {
        col_names[j] = "beta[" + std::to_string(j + 1) + "]";
    }
    for (int m = 0; m < n_mesh; m++) {
        col_names[layout.spde_w_start + m] =
            spde_block_name + "[" + std::to_string(m + 1) + "]";
    }
    if (joint_hypers) {
        col_names[layout.log_kappa_spde_idx] = "log_kappa";
        col_names[layout.log_tau_spde_idx]   = "log_tau";
    }
    col_names[layout.extra_offset] = "log_phi";
    Rcpp::colnames(draws) = col_names;

    Rcpp::NumericVector means(n_params, 0.0);
    for (int s = 0; s < n_sample; s++) {
        for (int j = 0; j < n_params; j++) {
            means[j] += draws(s, j) / n_sample;
        }
    }
    means.names() = col_names;

    // Joint-mode post-processing: transform z -> w per draw via the
    // cached SpdeNcTransform (built lazily during sampling), and derive
    // (range, sigma) on the natural scale from (log_kappa, log_tau).
    //
    // Matern map (d=2, nu = alpha - 1):
    //   range = sqrt(8 nu) / kappa
    //   sigma = 1 / (sqrt(4 pi) * kappa * tau)
    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("draws")       = draws,
        Rcpp::Named("means")       = means,
        Rcpp::Named("n_samples")   = n_sample,
        Rcpp::Named("n_params")    = n_params,
        Rcpp::Named("p")           = p,
        Rcpp::Named("n_mesh")      = n_mesh,
        Rcpp::Named("kappa")       = kappa,
        Rcpp::Named("tau_spde")    = tau_spde,
        Rcpp::Named("family")      = family,
        Rcpp::Named("log_prob")    = Rcpp::wrap(result.log_prob),
        Rcpp::Named("accept_prob") = Rcpp::wrap(result.accept_prob),
        Rcpp::Named("divergent")   = Rcpp::wrap(result.divergent),
        Rcpp::Named("treedepth")   = Rcpp::wrap(result.treedepth),
        Rcpp::Named("sampler")     = result.sampler.empty() ? "nuts" : result.sampler,
        Rcpp::Named("epsilon")     = result.epsilon,
        Rcpp::Named("joint_hypers") = joint_hypers
    );

    if (joint_hypers) {
        constexpr double k_pi = 3.14159265358979323846;
        // Use the caller-supplied nu when present (correct for fractional
        // cases); fall back to alpha - 1 only when nu was left unset.
        const double nu_used     = (nu > 0.0) ? nu
                                              : static_cast<double>(alpha) - 1.0;
        const double sqrt_8nu    = std::sqrt(8.0 * nu_used);
        const double sqrt_4pi    = std::sqrt(4.0 * k_pi);

        Rcpp::NumericMatrix w_draws(n_sample, n_mesh);
        Rcpp::NumericVector range_draws(n_sample);
        Rcpp::NumericVector sigma_draws(n_sample);
        Rcpp::NumericVector kappa_draws(n_sample);
        Rcpp::NumericVector tau_draws(n_sample);

        std::vector<double> params_row(n_params);
        std::vector<double> w_out;
        for (int s = 0; s < n_sample; s++) {
            for (int j = 0; j < n_params; j++) params_row[j] = draws(s, j);
            tulpa::apply_spde_nc_transform_double(params_row, data, layout, w_out);
            for (int m = 0; m < n_mesh; m++) w_draws(s, m) = w_out[m];

            const double lk = params_row[layout.log_kappa_spde_idx];
            const double lt = params_row[layout.log_tau_spde_idx];
            const double k_s = std::exp(lk);
            const double t_s = std::exp(lt);
            kappa_draws[s] = k_s;
            tau_draws[s]   = t_s;
            range_draws[s] = sqrt_8nu / k_s;
            sigma_draws[s] = 1.0 / (sqrt_4pi * k_s * t_s);
        }
        Rcpp::CharacterVector w_names(n_mesh);
        for (int m = 0; m < n_mesh; m++) {
            w_names[m] = "w[" + std::to_string(m + 1) + "]";
        }
        Rcpp::colnames(w_draws) = w_names;

        out["w_draws"]     = w_draws;
        out["range_draws"] = range_draws;
        out["sigma_draws"] = sigma_draws;
        out["kappa_draws"] = kappa_draws;
        out["tau_draws"]   = tau_draws;
    }

    return out;
}
