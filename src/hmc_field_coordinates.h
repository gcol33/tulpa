#ifndef TULPA_HMC_FIELD_COORDINATES_H
#define TULPA_HMC_FIELD_COORDINATES_H

#include <tulpa/field_coordinates.h>
#include "hmc_sampler.h"
#include <limits>

namespace tulpa_hmc {

using tulpa::FieldCoordinate;
using FieldCoordinates = tulpa::FieldCoordinates<GradientFn>;

tulpa::FieldCoordinateOperator prepare_spatial_gaussian_coordinate(const ModelData& data,
    const ParamLayout& layout);
tulpa::FieldCoordinateOperator prepare_bym2_coordinate(const ModelData& data, const ParamLayout& layout);

inline thread_local const FieldCoordinates* field_coordinates = nullptr;

class FieldCoordinateScope {
    const FieldCoordinates* previous_;
public:
    explicit FieldCoordinateScope(const FieldCoordinates* current)
        : previous_(field_coordinates) { field_coordinates = current; }
    ~FieldCoordinateScope() { field_coordinates = previous_; }
    FieldCoordinateScope(const FieldCoordinateScope&) = delete;
    FieldCoordinateScope& operator=(const FieldCoordinateScope&) = delete;
};

inline FieldCoordinates make_field_coordinates(const std::vector<double>& initial,
                                         const ModelData& data,
                                         const ParamLayout& layout) {
  FieldCoordinateScope raw(nullptr);
  FieldCoordinates out;
  out.gradient = resolve_gradient_fn(g_gradient_mode, data, layout);
  out.prior_gradient = resolve_prior_gradient_fn(g_gradient_mode, data, layout);
  const bool gaussian_spatial = layout.total_params <= DENSE_MAX_PARAMS &&
    layout.is_gp && !layout.is_gp_collapsed && data.gp_parameterization == 1 &&
    data.auto_gp_coordinates && data.conditional_gp_coordinates;
  if (gaussian_spatial) out.operators.push_back(prepare_spatial_gaussian_coordinate(data, layout));
  if (layout.is_bym2 && !layout.is_bym2_collapsed && layout.total_params <= DENSE_MAX_PARAMS)
    out.operators.push_back(prepare_bym2_coordinate(data, layout));
  if (layout.is_gp && data.has_gp && data.gp_range_prior.type == tulpa::RangePriorType::UNIFORM)
    out.ranges.push_back({layout.log_phi_gp_idx, data.gp_range_prior.lower, data.gp_range_prior.upper});
  struct Block {
    int begin, end, scale;
    double sign;
    int width;
    bool constant, linear, standardized, coordinatewise;
  };
  std::vector<Block> blocks;
  auto add = [&](int begin, int end, int scale, double sign, int width = 0,
                 bool constant = false, bool linear = false, bool standardized = false,
                 bool coordinatewise = false) {
    if (begin < 0 || end <= begin || scale < 0) return;
    const int step = coordinatewise ? 1 : end - begin;
    for (int i = begin; i < end; i += step)
      blocks.push_back({i, i + step, scale, sign, coordinatewise ? 0 : width,
                        constant, linear, standardized, coordinatewise});
  };
  if (!gaussian_spatial && data.auto_spatial_coordinates && layout.has_spatial && !layout.is_bym2)
    add(layout.spatial_start, layout.spatial_end, layout.log_tau_spatial_idx, 1);
  if (data.auto_temporal_coordinates && layout.has_temporal && !layout.is_temporal_gp)
    add(layout.temporal_start, layout.temporal_end, layout.log_tau_temporal_idx, 1,
        data.n_times, false, data.temporal_type == TemporalType::RW2 && !data.temporal_cyclic);
  if (!gaussian_spatial && data.auto_gp_coordinates && layout.is_gp && !layout.is_gp_collapsed)
    add(layout.gp_w_start, layout.gp_w_end, layout.log_sigma2_gp_idx,
        1, 0, false, false, true);
  if (data.auto_temporal_gp_coordinates && layout.is_temporal_gp)
    add(layout.temporal_start, layout.temporal_end, layout.log_sigma2_temporal_gp_idx,
        1, 0, false, false, true);
  if (data.auto_svc_coordinates && layout.has_svc && !data.svc_is_hsgp) {
    const int width = (layout.svc_w_end - layout.svc_w_start) / data.svc_data.n_svc;
    for (int j = 0; j < data.svc_data.n_svc; ++j)
      add(layout.svc_w_start + j * width, layout.svc_w_start + (j + 1) * width,
          layout.log_sigma2_svc_start + j, 1, 0, false, false, true);
  }
  if (data.auto_multiscale_coordinates && layout.has_multiscale_temporal) {
    const bool trend_nc = data.ms_trend_parameterization == 1;
    const bool seasonal_nc = data.ms_seasonal_parameterization == 1;
    const bool short_nc = data.ms_short_parameterization == 1;
    const int trend_null = trend_nc ?
        tulpa_temporal::rw_walk_order(data.multiscale_temporal_data.trend_type) : 0;
    add(layout.trend_start + trend_null, layout.trend_end,
        layout.log_sigma2_trend_idx, trend_nc ? 1 : -1,
        trend_nc ? 0 : data.multiscale_temporal_data.n_times, false,
        !trend_nc && data.multiscale_temporal_data.trend_type == TemporalType::RW2,
        trend_nc, trend_nc);
    add(layout.seasonal_start + (seasonal_nc ? 1 : 0), layout.seasonal_end,
        layout.log_sigma2_seasonal_idx, seasonal_nc ? 1 : -1,
        0, false, false, seasonal_nc, seasonal_nc);
    add(layout.short_term_start, layout.short_term_end, layout.log_sigma2_short_idx,
        short_nc ? 1 : -1, 0, false, false, short_nc, short_nc);
  }
  if (layout.has_tvc) {
    const auto& tvc = data.tvc_data;
    for (int g = 0; g < tvc.n_groups; ++g) for (int j = 0; j < tvc.n_tvc; ++j) {
      const int begin = layout.tvc_w_start + (g * tvc.n_tvc + j) * tvc.n_times;
      const bool intrinsic = tvc.structure == TemporalType::RW1 || tvc.structure == TemporalType::RW2;
      add(begin, begin + tvc.n_times, layout.log_tau_tvc_start + j, 1, tvc.n_times,
          intrinsic, tvc.structure == TemporalType::RW2 && !tvc.cyclic);
    }
  }
  if (blocks.empty()) return out;
  // A fixed reference gives every chain and resumed fit the same transport.
  std::vector<double> origin(initial.size(), 0.0);
  for (const auto& block : blocks)
    std::fill(origin.begin() + block.begin, origin.begin() + block.end, 0.0);
  if (!std::isfinite(compute_log_post(origin, data, layout))) return out;
  std::vector<double> score;
  out.gradient(origin, data, layout, score, nullptr);
  const double full = compute_log_post(origin, data, layout);
  const double prior = compute_log_post(origin, data, layout, true);
  // A balanced direction measures one Rayleigh quotient per block. Its fixed
  // curvature sets a coordinate only; the density keeps every prior and
  // likelihood term. Setup takes a constant number of density evaluations
  // per block, independent of its number of coefficients.
  for (const auto& block : blocks) {
    std::vector<double> plus = origin, minus = origin;
    const double h = std::pow(std::numeric_limits<double>::epsilon(), 0.25) *
        std::sqrt(1 + std::abs(full));
    const FieldCoordinate geometry{block.begin, block.end, block.scale, block.sign,
        1, 0, block.width, block.constant, block.linear};
    if (geometry.rank() == 0) continue;
    std::vector<double> direction(origin.size(), 0.0);
    for (int i = block.begin; i < block.end; ++i) direction[i] = i % 2 ? 1 : -1;
    double denominator = 0;
    for (int start = block.begin; start < block.end; start += geometry.width()) {
      const auto null = geometry.null_component(direction, start);
      for (int t = 0; t < geometry.width(); ++t) {
        const int i = start + t;
        const double shift = h * (direction[i] - null.at(t));
        plus[i] += shift;
        minus[i] -= shift;
        denominator += shift * shift;
      }
    }
    if (!(denominator > 0)) continue;
    const double prior_curve = -(compute_log_post(plus, data, layout, true) -
        2 * prior + compute_log_post(minus, data, layout, true)) / denominator;
    const double full_curve = -(compute_log_post(plus, data, layout) -
        2 * full + compute_log_post(minus, data, layout)) / denominator;
    if (!(prior_curve > 0) || !std::isfinite(prior_curve) || !std::isfinite(full_curve))
      continue;
    double score_power = 0.0;
    if (!block.coordinatewise)
      for (int i = block.begin; i < block.end; ++i) score_power += score[i] * score[i];
    const double likelihood_information = std::max(0.0, full_curve - prior_curve) +
        score_power / (block.end - block.begin);
    const double scaled = (block.standardized ? likelihood_information : prior_curve) *
        std::exp(-block.sign * origin[block.scale]);
    const double fixed = block.standardized ? prior_curve : likelihood_information;
    out.entries.push_back({block.begin, block.end, block.scale, block.sign, scaled, fixed,
        block.width, block.constant, block.linear});
  }
  return out;
}

inline void compute_gradient_field_coordinates(
    const std::vector<double>& z, const ModelData& data, const ParamLayout& layout,
    std::vector<double>& gradient, double* log_post, bool prior_only = false) {
    const auto& coordinates = *field_coordinates;
    static thread_local std::vector<double>* workspace = nullptr;
    if (!workspace) workspace = new std::vector<double>();
    auto& x = *workspace;
    const double jacobian = coordinates.forward(z, x);
    if (!std::isfinite(jacobian)) {
        gradient.assign(z.size(), 0.0);
        if (log_post) *log_post = -INFINITY;
        return;
    }
    FieldCoordinateScope raw(nullptr);
    const auto fn = prior_only ? coordinates.prior_gradient : coordinates.gradient;
    fn(x, data, layout, gradient, log_post);
    coordinates.pullback(z, x, gradient);
    if (log_post) *log_post += jacobian;
}

inline void compute_full_gradient_field_coordinates(
    const std::vector<double>& z, const ModelData& data, const ParamLayout& layout,
    std::vector<double>& gradient, double* log_post) {
    compute_gradient_field_coordinates(z, data, layout, gradient, log_post);
}

inline void compute_prior_gradient_field_coordinates(
    const std::vector<double>& z, const ModelData& data, const ParamLayout& layout,
    std::vector<double>& gradient, double* log_post) {
    compute_gradient_field_coordinates(z, data, layout, gradient, log_post, true);
}

} // namespace tulpa_hmc
#endif
