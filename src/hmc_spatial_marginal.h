#ifndef TULPA_HMC_SPATIAL_MARGINAL_H
#define TULPA_HMC_SPATIAL_MARGINAL_H

#include <tulpa/model_data.h>
#include <tulpa/param_layout.h>
#include <vector>
#include <string>
#include <random>

namespace tulpa_hmc {

inline bool has_collapsed_spatial(const tulpa::ParamLayout& layout) {
    return layout.is_icar_collapsed || layout.is_bym2_collapsed || layout.is_gp_collapsed;
}

double compute_collapsed_spatial(const std::vector<double>& params, const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, bool prior_only = false);
double compute_collapsed_spatial_normalizer(const tulpa::ModelData& data,
    const tulpa::ParamLayout& layout, std::vector<std::string>* uncarried);
void compute_gradient_collapsed_spatial(const std::vector<double>& params,
    const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    std::vector<double>& gradient, double* log_post);
void compute_prior_gradient_collapsed_spatial(const std::vector<double>& params,
    const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    std::vector<double>& gradient, double* log_post);
void reconstruct_spatial_point(const std::vector<double>& position,
    const tulpa::ModelData& data, const tulpa::ParamLayout& layout,
    std::mt19937& rng, bool draw, double* structured, double* unstructured);
void reconstruct_collapsed_spatial(const tulpa::ModelData* data, const tulpa::ParamLayout* layout,
    const double* samples, int n_sample, unsigned int seed, bool draw,
    double* structured, double* unstructured);

}
#endif
