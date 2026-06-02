// hmc_sampler_decls.h
// Fragment of hmc_sampler.h. Self-contained: defines symbols inside
// namespace tulpa_hmc and aliases tulpa:: types into that namespace.
// using-decls for tulpa:: types, enum parsers, ParamLayout / log-post /
// gradient function declarations.
#ifndef TULPA_HMC_SAMPLER_DECLS_H
#define TULPA_HMC_SAMPLER_DECLS_H

#include <string>
#include <vector>

#include "tulpa/model_data.h"
#include "tulpa/param_layout.h"
#include "tulpa/types.h"

namespace tulpa_hmc {

// Import all canonical types from exported tulpa:: headers
using tulpa::ModelData;
using tulpa::ParamLayout;
using tulpa::ProcessData;
using tulpa::SharingSpec;
using tulpa::TemporalType;
using tulpa::TemporalData;
using tulpa::MultiscaleTemporalData;
using tulpa::TemporalGPData;
using tulpa::TemporalCovType;
using tulpa::ZIType;
using tulpa::GPData;
using tulpa::MultiscaleGPData;
using tulpa::CovType;
using tulpa::STType;
using tulpa::SpatiotemporalData;
using tulpa::NonsepType;
using tulpa::SVCData;
using tulpa::HSGPData;
using tulpa::TVCData;
using tulpa::SpatialType;
using tulpa::GradientMode;
using tulpa::MassMatrixType;
using tulpa::GPSolverConfig;
using tulpa::GPSolver;
using tulpa::MSGPSampler;

// Parse gradient mode from string
inline GradientMode parse_gradient_mode(const std::string& mode_str) {
    static const tulpa::EnumEntry<GradientMode> table[] = {
        {"auto", GradientMode::AUTO}, {"AUTO", GradientMode::AUTO},
        {"N", GradientMode::NUMERICAL}, {"numerical", GradientMode::NUMERICAL},
        {"A_t", GradientMode::AUTODIFF_TAPE}, {"autodiff_tape", GradientMode::AUTODIFF_TAPE},
        {"A_r", GradientMode::AUTODIFF_ARENA}, {"arena", GradientMode::AUTODIFF_ARENA},
        {"autodiff_arena", GradientMode::AUTODIFF_ARENA},
        {"A", GradientMode::AUTODIFF_FWD}, {"autodiff", GradientMode::AUTODIFF_FWD},
        {"forward", GradientMode::AUTODIFF_FWD},
        {"H", GradientMode::HANDCODED}, {"handcoded", GradientMode::HANDCODED},
        {"analytical", GradientMode::HANDCODED}
    };
    return tulpa::parse_enum(mode_str, table, GradientMode::AUTO);
}

// Parse metric type from string
inline MassMatrixType parse_metric_type(const std::string& metric_str) {
    static const tulpa::EnumEntry<MassMatrixType> table[] = {
        {"dense", MassMatrixType::DENSE}, {"DENSE", MassMatrixType::DENSE},
        {"block_diag", MassMatrixType::BLOCK_DIAG}, {"BLOCK_DIAG", MassMatrixType::BLOCK_DIAG},
        {"auto", MassMatrixType::AUTO}, {"AUTO", MassMatrixType::AUTO}
    };
    return tulpa::parse_enum(metric_str, table, MassMatrixType::DIAG);
}

// Human-readable metric name for verbose logging
inline const char* metric_name(MassMatrixType t) {
    switch (t) {
        case MassMatrixType::DIAG: return "DIAG";
        case MassMatrixType::DENSE: return "DENSE";
        case MassMatrixType::BLOCK_DIAG: return "BLOCK_DIAG";
        case MassMatrixType::AUTO: return "AUTO";
    }
    return "UNKNOWN";
}

// Global gradient mode + accessors. Defined in hmc_gradient_fallback.cpp.
// Direct access is reserved for the dispatcher and the per-chain warmup
// fallback path; everywhere else should go through set/get.
extern GradientMode g_gradient_mode;
void set_gradient_mode(GradientMode mode);
GradientMode get_gradient_mode();

// Reset VecGradWorkspace cache (for new model fit)
void reset_grad_workspace_cache();

// Function pointer type for gradient computation (eliminates per-call dispatch overhead)
using GradientFn = void(*)(
    const std::vector<double>&, const ModelData&, const ParamLayout&,
    std::vector<double>&, double*);

// Resolve the gradient function pointer once based on mode + model config
GradientFn resolve_gradient_fn(GradientMode mode, const ModelData& data, const ParamLayout& layout);

// ModelData and ParamLayout are now defined in exported headers:
//   inst/include/tulpa/model_data.h   (tulpa::ModelData)
//   inst/include/tulpa/param_layout.h (tulpa::ParamLayout)
// Imported into tulpa_hmc namespace via using declarations above.

ParamLayout compute_param_layout(const ModelData& data);
int get_n_params(const ModelData& data);

// =====================================================================
// Log-posterior computation (with OpenMP parallelization)
// =====================================================================

// Main log-posterior function
// When skip_obs_loop=true, returns only prior+structural terms (O(p+S+T)),
// skipping the O(N) observation loop. Used by fused gradient+log_post computation.
double compute_log_post(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    bool skip_obs_loop = false,
    const double* precomputed_st_log_prior = nullptr,
    const double* precomputed_tgp_log_prior = nullptr
);

// Separable prior + likelihood (gcol33/tulpa#6 prereq).
// Contract: compute_log_post == compute_log_prior + compute_log_lik_only
// to within numerical tolerance for any well-defined params.
// SMC and other tempered samplers consume these as independent callables
// (target = log_prior + beta * log_lik).
double compute_log_prior(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);
double compute_log_lik_only(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout
);

// Gradient computation (with optional fused log-posterior)
// When log_post_out is non-null, the log-posterior is computed alongside
// the gradient in a single pass, avoiding redundant O(N) computation.
void compute_gradient(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
);

// Generic (multi-process) gradient drivers used by the dispatch when the
// model plugs a LikelihoodSpec. Defined in hmc_gradient_fallback.cpp.
//
// Phase D simplification (gcol33/tulpa#15): the legacy ratio
// numerical / autodiff / arena / forward fallbacks and the full set
// of H-mode specialized kernels (composite, hsgp, gp / gp_collapsed,
// icar_collapsed, msgp, svc, tvc, st, temporal_gp, ms_temporal,
// latent) were removed along with the legacy entry points. Downstream
// model packages must route through `data.likelihood_spec`.
void compute_gradient_generic_numerical(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
);

void compute_gradient_generic_arena(
    const std::vector<double>& params,
    const ModelData& data,
    const ParamLayout& layout,
    std::vector<double>& grad,
    double* log_post_out = nullptr
);

// Get RE value for observation (handles non-centered re_param=1 -> sigma*z transform).
inline double re_value_for_eta(
    const double* re,
    int g,
    double sigma_re,
    int re_parameterization
) {
    double val = re[g];
    if (re_parameterization == 1) val *= sigma_re;
    return val;
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_SAMPLER_DECLS_H
