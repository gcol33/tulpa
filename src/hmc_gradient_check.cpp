// hmc_gradient_check.cpp
// Shared gradient diagnostics used by Rcpp-facing HMC entry points.
//
// This file owns the diagnostic-only path. Sampling code should resolve one
// gradient function and call it directly; these checks intentionally compute
// several gradients so tests can compare H, A, A_r, and numerical modes.

#include "hmc_gradient_check.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace tulpa_hmc {

namespace {

double max_relative_gradient_diff(const std::vector<double>& a,
                                  const std::vector<double>& b) {
  double mx = 0.0;
  for (size_t i = 0; i < a.size() && i < b.size(); i++) {
    const double diff = std::abs(a[i] - b[i]);
    const double scale = std::max(1.0, std::max(std::abs(a[i]), std::abs(b[i])));
    mx = std::max(mx, diff / scale);
  }
  return mx;
}

// Autodiff gradients differentiate log_post_impl<T>, while several legacy
// hand-coded paths are checked against compute_log_post(). Pick the numerical
// reference that matches the implementation being inspected.
bool is_autodiff_gradient_fn(GradientFn fn) {
  return fn == &compute_gradient_arena ||
         fn == &compute_gradient_forward ||
         fn == &compute_gradient_autodiff;
}

// Emit compact failure context without flooding test output with every
// parameter. The caller adds layout information when it helps locate the block.
void print_gradient_mismatch(const char* label,
                             const std::vector<double>& lhs,
                             const std::vector<double>& rhs,
                             const char* lhs_name,
                             const char* rhs_name,
                             int n_worst) {
  std::vector<std::pair<double, int>> diffs;
  for (size_t i = 0; i < lhs.size() && i < rhs.size(); i++) {
    const double diff = std::abs(lhs[i] - rhs[i]);
    const double scale = std::max(1.0, std::max(std::abs(lhs[i]), std::abs(rhs[i])));
    diffs.push_back({diff / scale, static_cast<int>(i)});
  }
  std::sort(diffs.begin(), diffs.end(), [](const auto& a, const auto& b) {
    return a.first > b.first;
  });

  Rprintf("  %s (top %d of %d params):\n",
          label, std::min(n_worst, static_cast<int>(diffs.size())),
          static_cast<int>(lhs.size()));
  for (int k = 0; k < std::min(n_worst, static_cast<int>(diffs.size())); k++) {
    const int idx = diffs[k].second;
    Rprintf("    param[%d]: %s=%.8e %s=%.8e  rel_diff=%.2e\n",
            idx, lhs_name, lhs[idx], rhs_name, rhs[idx], diffs[k].first);
  }
}

// Keep this printer broad rather than model-specific: gradient_check_only is
// often used while adding new model combinations, and the active layout flags
// are more useful than a single hard-coded branch name.
void print_gradient_layout_context(const ModelData& data,
                                   const ParamLayout& layout) {
  Rprintf("  Layout: beta_num[%d-%d] beta_denom[%d-%d] re[%d+]\n",
          layout.legacy.beta_num_start,
          layout.legacy.beta_num_start + data.legacy.p_num - 1,
          layout.legacy.beta_denom_start,
          layout.legacy.beta_denom_start + data.legacy.p_denom - 1,
          layout.re_start);
  if (layout.has_spatial) {
    Rprintf("  spatial[%d-%d]\n", layout.spatial_start, layout.spatial_end - 1);
  }
  if (layout.has_temporal) {
    Rprintf("  temporal[%d-%d] log_tau[%d]\n",
            layout.temporal_start, layout.temporal_end - 1,
            layout.log_tau_temporal_idx);
  }
  if (layout.has_spatiotemporal) {
    Rprintf("  ST delta[%d+] tau_st[%d]\n",
            layout.st_delta_start, layout.log_tau_st_idx);
  }
  if (layout.has_tvc) {
    Rprintf("  TVC w[%d+] tau[%d+]\n",
            layout.tvc_w_start, layout.log_tau_tvc_start);
  }
  if (layout.has_svc) {
    Rprintf("  SVC w[%d+] sigma2[%d+] phi[%d+]\n",
            layout.svc_w_start, layout.log_sigma2_svc_start,
            layout.log_phi_svc_start);
  }
  if (layout.has_re_slopes) {
    Rprintf("  has_re_slopes=true n_re_terms=%d\n", data.n_re_terms);
  }
  if (layout.is_temporal_gp) {
    Rprintf("  temporal_gp: sigma2[%d] phi[%d]\n",
            layout.log_sigma2_temporal_gp_idx,
            layout.logit_phi_temporal_gp_idx);
  }
  if (layout.is_multiscale_gp) {
    Rprintf("  MSGP: sigma2_local[%d] phi_local[%d] sigma2_reg[%d] phi_reg[%d]\n",
            layout.log_sigma2_gp_local_idx, layout.log_phi_gp_local_idx,
            layout.log_sigma2_gp_regional_idx, layout.log_phi_gp_regional_idx);
    Rprintf("  MSGP: beta_local[%d-%d] beta_reg[%d-%d]\n",
            layout.gp_local_start, layout.gp_local_end - 1,
            layout.gp_regional_start, layout.gp_regional_end - 1);
  }
  if (layout.has_multiscale_temporal) {
    Rprintf("  ms_temporal\n");
  }
  if (layout.has_latent) {
    Rprintf("  latent\n");
  }
}

}  // namespace

Rcpp::List run_gradient_check_only(const std::vector<double>& q0,
                                   const ModelData& data) {
  ParamLayout layout = compute_param_layout(data);
  const double tol = 1e-4;

  std::vector<double> grad_N;
  compute_gradient_numerical(q0, data, layout, grad_N);

  std::vector<double> grad_N_impl;
  compute_gradient_numerical_impl(q0, data, layout, grad_N_impl);

  double h_vs_n = -1.0;
  GradientFn h_fn = resolve_gradient_fn(GradientMode::HANDCODED, data, layout);
  if (h_fn != &compute_gradient_numerical &&
      h_fn != &compute_gradient_numerical_impl) {
    const auto& ref = is_autodiff_gradient_fn(h_fn) ? grad_N_impl : grad_N;
    std::vector<double> grad_H;
    h_fn(q0, data, layout, grad_H, nullptr);
    h_vs_n = max_relative_gradient_diff(grad_H, ref);
    if (h_vs_n > tol) {
      print_gradient_mismatch(
          "H gradient mismatch", grad_H, ref, "H", "N", 5);
      print_gradient_layout_context(data, layout);
    }
  }

  double ar_vs_n = -1.0;
  GradientFn ar_fn = resolve_gradient_fn(GradientMode::AUTODIFF_ARENA, data, layout);
  if (ar_fn != &compute_gradient_numerical &&
      ar_fn != &compute_gradient_numerical_impl) {
    std::vector<double> grad_Ar;
    ar_fn(q0, data, layout, grad_Ar, nullptr);
    ar_vs_n = max_relative_gradient_diff(grad_Ar, grad_N_impl);
  }

  double a_vs_n = -1.0;
  GradientFn a_fn = resolve_gradient_fn(GradientMode::AUTODIFF_FWD, data, layout);
  if (a_fn != &compute_gradient_numerical &&
      a_fn != &compute_gradient_numerical_impl) {
    std::vector<double> grad_A;
    a_fn(q0, data, layout, grad_A, nullptr);
    a_vs_n = max_relative_gradient_diff(grad_A, grad_N_impl);
  }

  double h_vs_ar = -1.0;
  if (h_vs_n >= 0 && ar_vs_n >= 0) {
    std::vector<double> grad_H2;
    std::vector<double> grad_Ar2;
    h_fn(q0, data, layout, grad_H2, nullptr);
    ar_fn(q0, data, layout, grad_Ar2, nullptr);
    h_vs_ar = max_relative_gradient_diff(grad_H2, grad_Ar2);
    if (h_vs_ar > tol) {
      print_gradient_mismatch(
          "H vs A_r cross-check DIVERGE", grad_H2, grad_Ar2,
          "H", "A_r", 3);
      print_gradient_layout_context(data, layout);
    }
  }

  return Rcpp::List::create(
      Rcpp::Named("h_vs_n") = h_vs_n,
      Rcpp::Named("ar_vs_n") = ar_vs_n,
      Rcpp::Named("a_vs_n") = a_vs_n,
      Rcpp::Named("h_vs_ar") = h_vs_ar,
      Rcpp::Named("tol") = tol,
      Rcpp::Named("h_ok") = (h_vs_n >= 0)
          ? ((h_vs_n < tol) ? true : (h_vs_ar >= 0 && h_vs_ar < tol))
          : NA_LOGICAL,
      Rcpp::Named("ar_ok") = (ar_vs_n >= 0) ? (ar_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("a_ok") = (a_vs_n >= 0) ? (a_vs_n < tol) : NA_LOGICAL,
      Rcpp::Named("n_params") = static_cast<int>(q0.size()));
}

}  // namespace tulpa_hmc
