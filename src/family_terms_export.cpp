// family_terms_export.cpp
// Per-observation (loglik, grad, curvature) probes for the family kernels that
// are maintained in parallel.
//
// The same per-family math lives in several kernels because each serves a
// different backend: laplace_family_link.h (the Laplace/Newton dispatch),
// glmm_oracle.h (the compiled GLMM oracle behind agq_fit / re_aghq / the Gibbs
// sweep), and laplace_likelihoods.cpp (the explicit triplets). They are only
// consistent by convention -- and the conventions genuinely differ:
//
//   * `phi` is the residual SD in laplace_family_link.h and the residual
//     VARIANCE in glmm_oracle.h. The R layer bridges this at each boundary
//     (fit_laplace.R sqrt()s it; agq.R squares sigma_eps). Nothing in C++
//     enforces it, and both readings are finite and well-behaved, so a
//     mismatch surfaces only as two backends fitting different models at
//     phi != 1.
//   * glmm_oracle.h returns the signed d2/deta2; the Laplace side returns the
//     negated Hessian.
//
// These exports give each kernel a per-observation callable surface at a shared
// (y, eta, phi) so test-family-cross-path.R can pin the conventions against
// each other. They exist for testing only.

#include <Rcpp.h>
#include "tulpa_stop.h"
#include <string>
#include <utility>
#include <vector>

#include "laplace_family_link.h"
#include "laplace_family_curvature.h"
#include "laplace_family_zi_curvature.h"
#include "laplace_family_zi_phi.h"
#include "glmm_oracle.h"
#include "laplace_likelihoods.h"
#include "builtin_family_ll_ad.h"
#include "density_probe.h"
#include "probe_lengths.h"

// One scalar per observation from `eval(y, n_trials, eta)`, after the length
// contract of probe_lengths.h; `n_trials` recycles from length 1.
template <typename F>
Rcpp::NumericVector family_probe_vec(const char* fn,
                                     const Rcpp::NumericVector& y,
                                     const Rcpp::IntegerVector& n_trials,
                                     const Rcpp::NumericVector& eta, F eval) {
  const R_xlen_t n = eta.size();
  check_probe_lengths(fn, n, y, n_trials, nullptr);
  const bool recycle_nt = (n_trials.size() == 1);
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; i++) {
    out[i] = eval(y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i]);
  }
  return out;
}

// One row per observation of the struct `eval(y, n_trials, eta, logit_zi)`
// returns, one named column per listed field, after the same length contract
// with logit_zi read to the length of eta.
template <typename S, typename F>
Rcpp::NumericMatrix zi_mixture_probe(
    const char* fn, const Rcpp::NumericVector& y,
    const Rcpp::IntegerVector& n_trials, const Rcpp::NumericVector& eta,
    const Rcpp::NumericVector& logit_zi,
    const std::vector<std::pair<const char*, double S::*>>& cols, F eval) {
  const R_xlen_t n = eta.size();
  check_probe_lengths(fn, n, y, n_trials, &logit_zi);
  const bool recycle_nt = (n_trials.size() == 1);
  const int n_cols = static_cast<int>(cols.size());
  Rcpp::NumericMatrix out(n, n_cols);
  Rcpp::CharacterVector names(n_cols);
  for (int j = 0; j < n_cols; ++j) names[j] = cols[j].first;
  Rcpp::colnames(out) = names;
  for (R_xlen_t i = 0; i < n; i++) {
    const S d = eval(y[i], recycle_nt ? n_trials[0] : n_trials[i], eta[i],
                     logit_zi[i]);
    for (int j = 0; j < n_cols; ++j) out(i, j) = d.*(cols[j].second);
  }
  return out;
}

// Log likelihood at each column of a matrix of observation predictors.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_loglik_batch(Rcpp::NumericVector y,
                                           Rcpp::IntegerVector n_trials,
                                           Rcpp::NumericMatrix eta,
                                           std::string family, double phi) {
    const int n = eta.nrow();
    if (y.size() != n || (n_trials.size() != 1 && n_trials.size() != n)) {
        tulpa::stop("y and n_trials must match the rows of eta (n_trials may be scalar).");
    }
    const auto fam = tulpa::resolve_family(family);
    std::vector<double> constants(n);
    for (int i = 0; i < n; ++i) {
        const int nt = n_trials[n_trials.size() == 1 ? 0 : i];
        constants[i] = tulpa::log_lik_const_for_kind(y[i], nt, fam.kind);
    }
    Rcpp::NumericVector out(eta.ncol());
    for (int s = 0; s < eta.ncol(); ++s) {
        if (s % 256 == 0) Rcpp::checkUserInterrupt();
        for (int i = 0; i < n; ++i) {
            const int nt = n_trials[n_trials.size() == 1 ? 0 : i];
            out[s] += tulpa::log_lik_for_family(y[i], nt, eta(i, s), fam,
                                               phi, NA_REAL, constants[i]);
        }
    }
    return out;
}

// Unary primitive value and derivative on each autodiff scalar.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_precision_ad_math(double x, bool remainder) {
    return density_callback_terms(x, 0, [&](auto v, auto) {
        return remainder ? tulpa::math::stirling_remainder(v) : tulpa::math::log1pmx(v);
    });
}

// Grouped response callbacks on the built-in likelihood spec.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_beta_grouped_terms(Rcpp::NumericVector ys, double eta, double phi) {
    tulpa::BuiltinFamilyResponse r;
    double y = ys.size() ? ys[0] : .5;
    int n = ys.size();
    r.y = &y; r.n_trials = &n; r.N = 1; r.family = "beta"; r.phi = phi;
    r.beta_y.resize(1);
    for (double v : ys) r.beta_y[0].emplace_back(v);
    r.prepare();
    tulpa::ModelData data;
    tulpa::ParamLayout layout;
    std::vector<double> params;
    double grad, hess;
    tulpa::builtin_family_eta_weights(0, &eta, 0, 0, params, data, layout, &r, &grad, &hess);
    const double ll = tulpa::builtin_family_ll_double(0, &eta, 0, 0, params, data, layout, &r);
    fwd::Dual e(eta,1);
    const auto ad = tulpa::builtin_family_ll_ad(0, &e, fwd::Dual(0), fwd::Dual(0),
        std::vector<fwd::Dual>(), data, layout, &r);
    return Rcpp::NumericVector::create(ll, grad, hess, ad.val, ad.grad);
}

// Public density and resolved dispatch at the same predictor.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_precision_density_paths(double y, double eta,
                                                std::string family, double phi) {
    const auto fam = tulpa::resolve_family(family);
    const double mu = family == "beta" ? tulpa::inv_logit_link(eta) : std::exp(eta);
    auto out = Rcpp::NumericVector::create(
        Rcpp::_["resolved"] = tulpa::log_lik_for_family(y, 1, eta, fam, phi,
            NA_REAL, tulpa::log_lik_const_for_kind(y, 1, fam.kind)),
        Rcpp::_["header"] = tulpa::log_lik_mu(y, mu, phi, family, 1),
        Rcpp::_["grouped"] = family == "beta"
            ? tulpa::log_lik_beta_grouped({tulpa::BetaObservation(y)}, eta, phi)
            : tulpa::math::log_lik_gamma(y, phi, mu));
    out.attr("mu") = mu;
    out.attr("positive_logit_mu") = tulpa::math::inv_logit_pos(eta);
    return out;
}

// Terms from the Laplace/Newton family dispatch (laplace_family_link.h).
// `phi` follows that kernel's convention: residual SD for gaussian/lognormal.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_terms(double y, int n_trials, double eta,
                                     std::string family, double phi,
                                     double phi2 = NA_REAL) {
  const tulpa::GradHess gh =
      tulpa::grad_hess_for_family(y, n_trials, eta, family, phi, phi2);
  return Rcpp::NumericVector::create(
      Rcpp::_["log_lik"]   = tulpa::log_lik_for_family(y, n_trials, eta, family,
                                                       phi, phi2),
      Rcpp::_["grad"]      = gh.grad,
      Rcpp::_["neg_hess"]  = gh.neg_hess);
}

// The same observed curvature over a whole vector, which is what
// .family_obs_weight() in R now returns for EVERY family. It used to answer
// from the registry where a family registered an `obs_weight` closure and fall
// back to the EXPECTED weight where none was registered -- a different
// function, and for beta / gamma / inverse_gaussian / beta_binomial / tweedie /
// t one that can even carry the opposite sign (at y = 12, eta = -0.9 the
// beta_binomial observed curvature is -0.125 where the expected weight is
// +0.658). beta_binomial is in .ZI_FAMILIES, so the zero-inflation mixture
// differentiated through the wrong one (gcol33/tulpa#824).
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_weight(const Rcpp::NumericVector& y,
                                          const Rcpp::IntegerVector& n_trials,
                                          const Rcpp::NumericVector& eta,
                                          std::string family, double phi,
                                          double phi2 = NA_REAL) {
  const R_xlen_t n = eta.size();
  if (y.size() != n || n_trials.size() != n) {
    tulpa::stop("cpp_family_obs_weight(): y, n_trials and eta must have "
               "the same length.");
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    out[i] = tulpa::obs_grad_hess_for_family(y[i], n_trials[i], eta[i],
                                             family, phi, phi2).neg_hess;
  }
  return out;
}

// Whether curvature_deta_for_family() is exact for this family, so the exact
// Laplace gradient can refuse rather than optimize a fiction.
// [[Rcpp::export]]
bool cpp_family_has_curvature_derivative(std::string family) {
  return tulpa::has_curvature_derivative(family);
}

// Vectorized over observations. The exact Laplace gradient needs dw/deta at
// every observation of a fit, so the per-element probe above would cost one
// .Call per row; this keeps it to one. `n_trials` is recycled when length 1.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta_vec(Rcpp::NumericVector y,
                                                  Rcpp::IntegerVector n_trials,
                                                  Rcpp::NumericVector eta,
                                                  std::string family, double phi,
                                                  double phi2 = NA_REAL) {
  return family_probe_vec("cpp_family_curvature_deta_vec", y, n_trials, eta,
      [&](double yi, int nt, double ei) {
        return tulpa::curvature_deta_for_family(yi, nt, ei, family, phi, phi2);
      });
}

// Whether curvature_deta2_for_family() is exact for this family, so the exact
// Laplace theta-Hessian can refuse rather than optimize a fiction.
// [[Rcpp::export]]
bool cpp_family_has_curvature_2nd_derivative(std::string family) {
  return tulpa::has_curvature_2nd_derivative(family);
}

// Whether the Newton working weight already IS the observed curvature, so the
// observed-minus-working difference is the zero function. What separates a
// family whose delta is exactly zero from one whose delta merely happens to be
// small on a given fixture.
// [[Rcpp::export]]
bool cpp_family_working_weight_is_observed(std::string family) {
  return tulpa::working_weight_is_observed(family);
}

// Whether the compiled zero-inflation mixture can be built over this base. The
// R front door reads this rather than keeping its own list, so the two cannot
// drift.
// [[Rcpp::export]]
bool cpp_family_compiled_zi_supported(std::string family) {
  return tulpa::zi::compiled_zi_supported(family);
}

// Vectorized observed-minus-working curvature W_obs - w at every observation.
// The exact mode Jacobian dx_hat/dtheta needs the TRUE posterior Hessian
// A' diag(W_obs) A + P, which is the working-weight H_joint plus
// A' diag(W_obs - w) A. Returning the difference lets the caller correct
// H_joint in place rather than rebuild the precision. It is identically zero for
// every family whose working weight already is the observed curvature, and for
// the families that lack an exact observed form obs_grad_hess_for_family
// delegates to the working weight, so this returns zero there too -- the caller
// must gate on has_exact_mode_jacobian, not on the difference being nonzero.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_curvature_delta_vec(Rcpp::NumericVector y,
                                                       Rcpp::IntegerVector n_trials,
                                                       Rcpp::NumericVector eta,
                                                       std::string family, double phi,
                                                       double phi2 = NA_REAL) {
  return family_probe_vec("cpp_family_obs_curvature_delta_vec", y, n_trials, eta,
      [&](double yi, int nt, double ei) {
        const double w_obs =
            tulpa::obs_grad_hess_for_family(yi, nt, ei, family, phi, phi2).neg_hess;
        const double w =
            tulpa::grad_hess_for_family(yi, nt, ei, family, phi, phi2).neg_hess;
        return w_obs - w;
      });
}

// Vectorized d(W_obs - w)/deta, the eta-derivative of the vector above. The
// closed outer Hessian differentiates quantities formed on the TRUE-curvature
// inverse, which needs dH_true/dtheta and so this derivative; without it the
// assembly would differentiate through the working-weight inverse instead, an
// inconsistency that only shows where the two weights differ.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_obs_curvature_delta_deta_vec(
    Rcpp::NumericVector y, Rcpp::IntegerVector n_trials,
    Rcpp::NumericVector eta, std::string family, double phi,
    double phi2 = NA_REAL) {
  return family_probe_vec("cpp_family_obs_curvature_delta_deta_vec", y, n_trials,
      eta, [&](double yi, int nt, double ei) {
        return tulpa::obs_curvature_delta_deta_for_family(yi, nt, ei, family,
                                                          phi, phi2);
      });
}

// Whether the derivative above is exact for this family. False for
// neg_binomial_1, whose observed curvature would need a tetragamma.
// [[Rcpp::export]]
bool cpp_family_has_obs_curvature_delta_derivative(std::string family) {
  return tulpa::has_obs_curvature_delta_derivative(family);
}

// Whether the exact analytic mode Jacobian can be formed for this family, so the
// marginal correction can trust the closed J or fall back to differencing.
// [[Rcpp::export]]
bool cpp_family_has_exact_mode_jacobian(std::string family) {
  return tulpa::has_exact_mode_jacobian(family);
}

// Vectorized over observations, matching cpp_family_curvature_deta_vec: the
// closed-form theta-Hessian needs d2w/deta2 at every observation of a fit.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_family_curvature_deta2_vec(Rcpp::NumericVector y,
                                                   Rcpp::IntegerVector n_trials,
                                                   Rcpp::NumericVector eta,
                                                   std::string family, double phi,
                                                   double phi2 = NA_REAL) {
  return family_probe_vec("cpp_family_curvature_deta2_vec", y, n_trials, eta,
      [&](double yi, int nt, double ei) {
        return tulpa::curvature_deta2_for_family(yi, nt, ei, family, phi, phi2);
      });
}

// The six partials of the zero-inflation mixture's 2 x 2 curvature block, one
// row per observation, columns [dWee_deta, dWee_dz, dWez_deta, dWez_dz,
// dWzz_deta, dWzz_dz]. The exact outer gradient contracts them against the
// three linear-predictor (co)variances to form its dW/dtheta channel.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_curvature_deriv(Rcpp::NumericVector y,
                                                   Rcpp::IntegerVector n_trials,
                                                   Rcpp::NumericVector eta,
                                                   Rcpp::NumericVector logit_zi,
                                                   std::string family, double phi,
                                                   double phi2 = NA_REAL) {
  using S = tulpa::zi::MixtureCurvatureDeriv;
  return zi_mixture_probe<S>("cpp_zi_mixture_curvature_deriv", y, n_trials, eta,
      logit_zi,
      {{"dWee_deta", &S::dWee_deta}, {"dWee_dz", &S::dWee_dz},
       {"dWez_deta", &S::dWez_deta}, {"dWez_dz", &S::dWez_dz},
       {"dWzz_deta", &S::dWzz_deta}, {"dWzz_dz", &S::dWzz_dz}},
      [&](double yi, int nt, double ei, double z) {
        return tulpa::zi::mixture_curvature_deriv(yi, nt, ei, z, family, phi,
                                                  phi2);
      });
}

// Whether the six partials above are exact for this family, so the outer
// optimization can take the analytic gradient or fall back cleanly.
// [[Rcpp::export]]
bool cpp_family_has_zi_curvature_derivative(std::string family) {
  return tulpa::zi::has_zi_curvature_derivative(family);
}

// The SECOND derivatives of the mixture's curvature block, one row per
// observation. Because the block is -Hess(log density), each is a fourth
// derivative of one scalar, so five columns cover all nine second partials --
// named for how many of the four derivatives are in eta. The closed outer
// Hessian contracts them against the same predictor (co)variances the gradient
// uses. Under a hurdle the three mixed columns are identically zero.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_curvature_deriv2(Rcpp::NumericVector y,
                                                    Rcpp::IntegerVector n_trials,
                                                    Rcpp::NumericVector eta,
                                                    Rcpp::NumericVector logit_zi,
                                                    std::string family, double phi,
                                                    double phi2 = NA_REAL) {
  using S = tulpa::zi::MixtureCurvatureDeriv2;
  return zi_mixture_probe<S>("cpp_zi_mixture_curvature_deriv2", y, n_trials, eta,
      logit_zi,
      {{"d4_e4", &S::d4_e4}, {"d4_e3z", &S::d4_e3z}, {"d4_e2z2", &S::d4_e2z2},
       {"d4_ez3", &S::d4_ez3}, {"d4_z4", &S::d4_z4}},
      [&](double yi, int nt, double ei, double z) {
        return tulpa::zi::mixture_curvature_deriv2(yi, nt, ei, z, family, phi,
                                                   phi2);
      });
}

// Whether the closed-form outer Hessian's curvature input is exact under a
// mixture. Narrower than cpp_family_has_zi_curvature_derivative(): the gradient
// differentiates the 2 x 2 block once, the Hessian twice, and the second one
// additionally needs the base family's second eta-derivative for the coupled
// y = 0 branch.
// [[Rcpp::export]]
bool cpp_family_has_zi_curvature_2nd_derivative(std::string family) {
  return tulpa::zi::has_zi_curvature_2nd_derivative(family);
}

// Dispersion derivatives of the mixture's y = 0 branch under GENUINE zero
// inflation (see laplace_family_zi_phi.h). Zero at y != 0 and for a hurdle,
// where the base family's own phi registry is already the mixture's derivative;
// the R assembly adds the two over their disjoint rows.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_zi_mixture_phi_deriv(Rcpp::NumericVector y,
                                             Rcpp::IntegerVector n_trials,
                                             Rcpp::NumericVector eta,
                                             Rcpp::NumericVector logit_zi,
                                             std::string family, double phi,
                                             double phi2 = NA_REAL) {
  using S = tulpa::zi::MixturePhiDeriv;
  return zi_mixture_probe<S>("cpp_zi_mixture_phi_deriv", y, n_trials, eta,
      logit_zi,
      {{"dl_dp", &S::dl_dp}, {"dsc_e_dp", &S::dsc_e_dp},
       {"dsc_z_dp", &S::dsc_z_dp}, {"dWee_dp", &S::dWee_dp},
       {"dWez_dp", &S::dWez_dp}, {"dWzz_dp", &S::dWzz_dp},
       {"dWee_dp_de", &S::dWee_dp_de}, {"dWee_dp_dz", &S::dWee_dp_dz},
       {"dWez_dp_dz", &S::dWez_dp_dz}, {"dWzz_dp_dz", &S::dWzz_dp_dz},
       {"dl_dp2", &S::dl_dp2}, {"dsc_e_dp2", &S::dsc_e_dp2},
       {"dsc_z_dp2", &S::dsc_z_dp2}, {"dWee_dp2", &S::dWee_dp2},
       {"dWez_dp2", &S::dWez_dp2}, {"dWzz_dp2", &S::dWzz_dp2}},
      [&](double yi, int nt, double ei, double z) {
        return tulpa::zi::mixture_phi_deriv(yi, nt, ei, z, family, phi, phi2);
      });
}

// Whether the dispersion coordinate is exact alongside a zero-inflation
// process for this family.
// [[Rcpp::export]]
bool cpp_family_has_zi_phi_deriv(std::string family) {
  return tulpa::zi::has_zi_phi_deriv(family);
}

// Per-observation log-likelihood of the two Gaussian families whose density
// reads a bound rather than only the point response, from the closed forms the
// Laplace kernels solve under: `interval_gaussian` reads (a, b) as the class
// interval (lower == upper finite: the value observed exactly), and
// `truncated_gaussian` reads them as the truncation bounds of a density
// evaluated at `y`. `sigma` is the residual SD, recycled when length 1, so a
// draw-by-observation matrix is read one row at a time.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_bound_gaussian_loglik(const Rcpp::NumericVector& y,
                                              const Rcpp::NumericVector& a,
                                              const Rcpp::NumericVector& b,
                                              const Rcpp::NumericVector& eta,
                                              const Rcpp::NumericVector& sigma,
                                              std::string family) {
  const R_xlen_t n = eta.size();
  if (y.size() != n || a.size() != n || b.size() != n ||
      (sigma.size() != 1 && sigma.size() != n)) {
    tulpa::stop("cpp_bound_gaussian_loglik(): y, a, b and eta must have the "
               "same length, and sigma length 1 or that length.");
  }
  const bool interval = family == "interval_gaussian";
  if (!interval && family != "truncated_gaussian") {
    tulpa::stop("cpp_bound_gaussian_loglik(): family '%s' reads no bound.",
               {family.c_str()});
  }
  Rcpp::NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i) {
    const double s = sigma[sigma.size() == 1 ? 0 : i];
    out[i] = interval
        ? tulpa::log_lik_interval_gaussian(a[i], b[i], eta[i], s)
        : tulpa::log_lik_truncated_gaussian(y[i], a[i], b[i], eta[i], s);
  }
  return out;
}
