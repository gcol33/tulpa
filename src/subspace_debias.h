// Gaussian-shaped subspace Metropolis updates.
//
// The fast method targets the joint density on the Gaussian conditional-mean
// surface x = mode + M u. The exact method augments each symmetric u proposal
// with an independent Gaussian conditional residual r. Its joint proposal is
// q(u' | u) g(r'), so log alpha = log p(x') - log p(x)
// + (r'^T H r' - r^T H r) / 2. Both u and r are retained on rejection.
// The residual is obtained from a full Gaussian draw d by removing M L^-1 d_S;
// its precision energy is the full standardized radius minus ||L^-1 d_S||^2.
// Each retained exact draw contains the full latent vector.

#ifndef TULPA_SUBSPACE_DEBIAS_H
#define TULPA_SUBSPACE_DEBIAS_H

#include "hard_constraints.h"
#include "inner_gaussian_draw.h"
#include "inner_laplace_probe.h"
#include "laplace_cholesky.h"
#include "rwmh.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

template <typename Result, typename... Args>
inline Rcpp::Function subspace_callback(const std::function<Result(Args...)>& fn) {
  using Callback = Rcpp::InternalFunctionWithStdFunction::CppFunctionBaseFromStdFunction<Result, Args...>;
  Rcpp::XPtr<Callback> pointer(new Callback(fn), true);
  Rcpp::Function make = Rcpp::Environment::Rcpp_namespace()["internal_function"];
  return make(pointer);
}

inline double subspace_log_accept(double log_target_new, double log_target_old,
                                  double residual_quad_new, double residual_quad_old) {
  return log_target_new - log_target_old +
    0.5 * (residual_quad_new - residual_quad_old);
}

// A small symmetric matrix held column-major in a flat buffer, so the templated
// Cholesky core (detail::cholesky_factorize_impl_raw) can read it through the
// same chol_at() customization point the dense Hessian storages use.
struct FlatSymMat {
  const double* a = nullptr;
  int n = 0;
};

inline double chol_at(const FlatSymMat& M, int j, int k) {
  return M.a[j + k * M.n];
}

struct SubspaceDebiasOptions {
  std::vector<int> idx;      // 0-based latent indices S
  int n_iter = 2000;         // recorded sweeps (post-warmup, pre-thinning)
  int n_warmup = 1000;       // warmup sweeps, used for scale adaptation
  int thin = 1;
  bool exact = false;
  SEXP importance = R_NilValue;
};

struct SubspaceDebiasOutcome {
  std::vector<int> idx;         // retained indices: S for fast, all for exact
  std::vector<double> draws;    // column-major offsets, n_kept x idx.size()
  std::vector<double> sigma_ss; // q * q, column-major: the inner Laplace's own
                                // marginal covariance of x_S
  int n_kept = 0;
  double accept = std::numeric_limits<double>::quiet_NaN();
  double scale = std::numeric_limits<double>::quiet_NaN();
  // Why nothing was sampled, when nothing was. Empty on success. Vocabulary:
  // "no_probe_indices", "degenerate_proposal" (a probe column failed to solve,
  // or Sigma_SS was not positive definite), "objective_not_finite" (the joint
  // density is not finite at the mode, so no Metropolis ratio exists).
  std::string declined;
  std::vector<double> importance;
  std::vector<double> importance_log_weights;
};

// Build the Gaussian coordinates and run surface or full-latent Metropolis.
//
// `eval_log_joint(x) -> double` is the Newton loop's own penalized-objective
// closure, so this routine carries no likelihood knowledge and works for the
// single-arm and joint loops alike. `x_buf` is caller-supplied scratch sized
// n_x; it must hold `mode` on entry and is restored to it on return. The
// factor is reused as-is, with no refactorization.
//
// Randomness comes from R's RNG (rwmh.h), so set.seed reproduces a run and this
// must not be called from inside a parallel region.
template <typename EvalLogJoint>
inline SubspaceDebiasOutcome compute_subspace_debias(
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    EvalLogJoint eval_log_joint,
    Rcpp::NumericVector& x_buf,
    const SubspaceDebiasOptions& opts,
    const HardConstraintState* hard = nullptr,
    const std::function<void(Rcpp::NumericVector&)>& present_draw = nullptr
) {
  SubspaceDebiasOutcome out;
  out.idx = opts.idx;
  if (opts.exact) {
    out.idx.resize(n_x);
    for (int j = 0; j < n_x; ++j) out.idx[j] = j;
  }
  const int q = static_cast<int>(opts.idx.size());
  if (opts.n_iter < 1 || opts.n_warmup < 0 || opts.thin < 1 ||
      opts.n_iter >= std::numeric_limits<int>::max() - opts.n_warmup) {
    Rcpp::stop("Invalid subspace iteration budget.");
  }
  if (q == 0 && !opts.exact) { out.declined = "no_probe_indices"; return out; }
  if (opts.exact && hard && hard->active()) {
    out.declined = "exact_hard_constraints_unsupported";
    return out;
  }
  std::vector<int> sorted_idx = opts.idx;
  std::sort(sorted_idx.begin(), sorted_idx.end());
  if (std::adjacent_find(sorted_idx.begin(), sorted_idx.end()) != sorted_idx.end() ||
      (!sorted_idx.empty() && (sorted_idx.front() < 0 || sorted_idx.back() >= n_x))) {
    Rcpp::stop("debias$idx must contain distinct indices within the latent vector.");
  }

  // --- the q conditional-mean columns V = Sigma E_S -------------------------
  std::vector<double> V(static_cast<std::size_t>(n_x) * q, 0.0);
  std::vector<double> rhs(n_x, 0.0), v(n_x, 0.0), z_work;
  if (!use_sparse) z_work.assign(n_x, 0.0);
  for (int b = 0; b < q; b++) {
    double sigma_b = 0.0;
    if (!inner_probe_column(n_x, opts.idx[b], chol, sparse_solver, use_sparse,
                            rhs, v, z_work, sigma_b)) {
      out.declined = "degenerate_proposal";
      return out;
    }
    for (int k = 0; k < n_x; k++) V[static_cast<std::size_t>(b) * n_x + k] = v[k];
  }
  // Under hard constraints A x = 0 the inner Gaussian is the one conditioned
  // on the constraint set, Sigma_c = B (B' H B)^-1 B'. Each column above is
  // solved against the live factor (of the anchored H_a where the state
  // anchors its structural rows), and the state's fold of the same factor
  // turns it into the column of Sigma_c, so the surface stays on the set.
  if (hard && hard->active()) {
    for (int b = 0; b < q; b++) {
      if (!hard->cap.fold(V.data() + static_cast<std::size_t>(b) * n_x, n_x)) {
        out.declined = "degenerate_proposal";
        return out;
      }
    }
  }

  // Sigma_SS = V[S, :] -- symmetric by construction; symmetrized against the
  // asymmetry two independent triangular solves leave behind.
  out.sigma_ss.assign(static_cast<std::size_t>(q) * q, 0.0);
  for (int b = 0; b < q; b++) {
    for (int a = 0; a < q; a++) {
      const double vab = V[static_cast<std::size_t>(b) * n_x + opts.idx[a]];
      const double vba = V[static_cast<std::size_t>(a) * n_x + opts.idx[b]];
      out.sigma_ss[a + b * q] = 0.5 * (vab + vba);
    }
  }

  std::vector<double> L(static_cast<std::size_t>(q) * q, 0.0);
  double log_det_ss = 0.0;
  detail::cholesky_factorize_impl_raw(FlatSymMat{out.sigma_ss.data(), q}, q,
                                      L.data(), log_det_ss);
  for (int e = 0; e < q * q; e++) {
    if (!std::isfinite(L[e])) { out.declined = "degenerate_proposal"; return out; }
  }
  for (int a = 0; a < q; a++) {
    if (!(L[a + a * q] > 0.0)) { out.declined = "degenerate_proposal"; return out; }
  }

  // --- M = V L^{-T}, so x(u) = mode + M u and x_S(u) = mode_S + L u ---------
  // Row r of M solves L m = V[r, :]' by forward substitution.
  std::vector<double> M(static_cast<std::size_t>(n_x) * q, 0.0);
  for (int r = 0; r < n_x; r++) {
    for (int a = 0; a < q; a++) {
      double sum = V[static_cast<std::size_t>(a) * n_x + r];
      for (int c = 0; c < a; c++) sum -= L[a + c * q] * M[static_cast<std::size_t>(c) * n_x + r];
      M[static_cast<std::size_t>(a) * n_x + r] = sum / L[a + a * q];
    }
  }

  std::vector<double> residual(n_x, 0.0), residual_prop(n_x, 0.0);
  std::vector<double> eps(n_x), gaussian(n_x), gaussian_u(q);
  double residual_quad = 0.0, proposed_quad = 0.0;
  auto draw_residual = [&]() {
    proposed_quad = 0.0;
    if (q == n_x) return true;
    for (int j = 0; j < n_x; ++j) {
      eps[j] = R::rnorm(0.0, 1.0);
      proposed_quad += eps[j] * eps[j];
    }
    if (!inner_gaussian_transform(n_x, 1, chol, sparse_solver, use_sparse,
                                  eps.data(), gaussian.data())) return false;
    for (int a = 0; a < q; ++a) {
      double sum = gaussian[opts.idx[a]];
      for (int b = 0; b < a; ++b) sum -= L[a + b * q] * gaussian_u[b];
      gaussian_u[a] = sum / L[a + a * q];
      proposed_quad -= gaussian_u[a] * gaussian_u[a];
    }
    for (int j = 0; j < n_x; ++j) {
      double d = gaussian[j];
      for (int a = 0; a < q; ++a) d -= M[a * n_x + j] * gaussian_u[a];
      residual_prop[j] = d;
    }
    for (int j : opts.idx) residual_prop[j] = 0.0;
    proposed_quad = std::max(0.0, proposed_quad);
    return true;
  };

  // --- random-walk Metropolis in u -----------------------------------------
  std::vector<double> u(q, 0.0), u_prop(q, 0.0);
  auto target_at = [&](const std::vector<double>& uu) -> double {
    for (int k = 0; k < n_x; k++) {
      double acc = mode[k] + (opts.exact ? residual_prop[k] : 0.0);
      for (int a = 0; a < q; a++) acc += M[static_cast<std::size_t>(a) * n_x + k] * uu[a];
      x_buf[k] = acc;
    }
    return eval_log_joint(x_buf);
  };

  double lp_cur = target_at(u);
  if (!std::isfinite(lp_cur)) {
    for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];
    out.declined = "objective_not_finite";
    return out;
  }

  const double lp_mode = lp_cur;
  std::vector<double> presented_mode = mode;
  if (opts.exact && present_draw) {
    present_draw(x_buf);
    std::copy(x_buf.begin(), x_buf.end(), presented_mode.begin());
    std::copy(mode.begin(), mode.end(), x_buf.begin());
  }
  if (opts.exact && opts.importance != R_NilValue) {
    double logdet = 0.0;
    if (use_sparse) logdet = sparse_solver.log_determinant();
    else for (int j = 0; j < n_x; ++j) logdet += 2 * std::log(chol.L[j + j * n_x]);
    std::function<Rcpp::NumericMatrix(Rcpp::NumericMatrix)> transform_fn = [&](Rcpp::NumericMatrix Z) {
      Rcpp::NumericMatrix delta(n_x, Z.ncol());
      if (!inner_gaussian_transform(n_x, Z.ncol(), chol, sparse_solver,
                                    use_sparse, Z.begin(), delta.begin())) {
        Rcpp::stop("Exact subspace correction requires an LL' precision factor.");
      }
      return delta;
    };
    Rcpp::Function transform = subspace_callback(transform_fn);
    const double log_g_mode = 0.5 * (logdet - n_x * 1.8378770664093453);
    std::function<Rcpp::NumericVector(Rcpp::NumericMatrix)> log_joint_fn = [&](Rcpp::NumericMatrix W) {
      Rcpp::NumericVector values(W.ncol());
      for (int i = 0; i < W.ncol(); ++i) {
        std::copy(W.begin() + static_cast<std::size_t>(i) * n_x,
                  W.begin() + static_cast<std::size_t>(i + 1) * n_x, x_buf.begin());
        values[i] = eval_log_joint(x_buf) - lp_mode + log_g_mode;
      }
      std::copy(mode.begin(), mode.end(), x_buf.begin());
      return values;
    };
    Rcpp::Function log_joint = subspace_callback(log_joint_fn);
    Rcpp::Function estimate(opts.importance);
    Rcpp::List got = estimate(Rcpp::List::create(
      Rcpp::Named("mean") = mode, Rcpp::Named("transform") = transform,
      Rcpp::Named("log_determinant") = logdet), log_joint);
    for (const char* key : {"log_prob", "particle_ess", "n_particles",
                            "log_estimate_sd", "likelihood_evaluations",
                            "pareto_k", "k_threshold"}) {
      out.importance.push_back(Rcpp::as<double>(got[key]));
    }
    out.importance_log_weights = Rcpp::as<std::vector<double>>(got["log_weights"]);
  }

  double scale = rw_init_scale(std::max(1, q));
  const double target = rw_target_accept(std::max(1, q));
  const int thin = std::max(1, opts.thin);
  const int n_warm = std::max(0, opts.n_warmup);
  const int n_rec = std::max(0, opts.n_iter);
  const int n_sweep = n_warm + n_rec;

  const int n_kept = (n_rec - 1) / thin + 1;
  out.n_kept = n_kept;
  const int n_cols = opts.exact ? n_x : q;
  if (opts.exact) {
    out.idx.resize(n_x);
    for (int j = 0; j < n_x; ++j) out.idx[j] = j;
  }
  out.draws.assign(static_cast<std::size_t>(n_kept) * n_cols, 0.0);

  long acc_rec = 0;
  int kept = 0;
  for (int sweep = 1; sweep <= n_sweep; sweep++) {
    if (sweep % 256 == 0) Rcpp::checkUserInterrupt();
    const bool adapting = sweep <= n_warm;
    for (int a = 0; a < q; a++) u_prop[a] = u[a] + scale * R::rnorm(0.0, 1.0);
    double adapt_accept = 0.0;
    if (opts.exact && adapting && q > 0) {
      // Nuisance rejection limits joint acceptance independently of the u
      // step. Tune u using its conditional acceptance at the retained residual.
      residual_prop = residual;
      const double conditional_lp = target_at(u_prop);
      adapt_accept = std::isfinite(conditional_lp)
          ? std::exp(std::min(0.0, conditional_lp - lp_cur)) : 0.0;
    }
    if (opts.exact && !draw_residual()) {
      std::copy(mode.begin(), mode.end(), x_buf.begin());
      out.draws.clear();
      out.n_kept = 0;
      out.declined = "sparse_factor_not_ll";
      return out;
    }
    const double lp_prop = target_at(u_prop);
    const bool acc = rw_accept(subspace_log_accept(lp_prop, lp_cur,
      opts.exact ? proposed_quad : 0.0, opts.exact ? residual_quad : 0.0));
    if (acc) {
      u = u_prop; lp_cur = lp_prop;
      if (opts.exact) { residual = residual_prop; residual_quad = proposed_quad; }
    }
    if (adapting) {
      if (!opts.exact || q > 0) {
        scale = rw_adapt_scale(scale, rw_adapt_gain(sweep),
          opts.exact ? adapt_accept : (acc ? 1.0 : 0.0), target);
      }
    } else if ((sweep - n_warm - 1) % thin == 0 && kept < n_kept) {
      if (opts.exact) {
        for (int j = 0; j < n_x; ++j) {
          x_buf[j] = mode[j] + residual[j];
          for (int a = 0; a < q; ++a) x_buf[j] += M[a * n_x + j] * u[a];
        }
        if (present_draw) present_draw(x_buf);
        for (int j = 0; j < n_x; ++j)
          out.draws[j * n_kept + kept] = x_buf[j] - presented_mode[j];
      } else {
        for (int a = 0; a < q; a++) {
          double d = 0.0;
          for (int c = 0; c <= a; c++) d += L[a + c * q] * u[c];
          out.draws[static_cast<std::size_t>(a) * n_kept + kept] = d;
        }
      }
      acc_rec += acc ? 1 : 0;
      kept++;
    }
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];   // restore
  out.n_kept = kept;
  // Acceptance is counted over the RECORDED draws, not over every post-warmup
  // proposal, so at thin > 1 it rests on one sweep in `thin`. The estimator is
  // unbiased either way and the reported rate is the one that belongs to the
  // draws returned; the cost is variance, which is why a run reporting a rate
  // far from `target` at a large `thin` is worth re-reading at thin = 1.
  out.accept = (kept > 0) ? static_cast<double>(acc_rec) / kept
                          : std::numeric_limits<double>::quiet_NaN();
  out.scale = scale;
  return out;
}

// Apply convergence and factor guards, then attach samples and IS diagnostics.
template <typename Result, typename EvalLogJoint>
inline void run_subspace_debias(
    Result& result,
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    EvalLogJoint eval_log_joint,
    Rcpp::NumericVector& x_buf,
    const SubspaceDebiasOptions* opts,
    const char* gate_declined = nullptr,
    const HardConstraintState* hard = nullptr,
    const std::function<void(Rcpp::NumericVector&)>& present_draw = nullptr
) {
  if (!opts || (opts->idx.empty() && !opts->exact)) return;
  if (gate_declined || !result.converged) {
    // The surface is anchored at the mode and shaped by the inner Laplace's
    // covariance there, so neither exists at a point the solve stopped short of.
    result.debias_idx      = opts->idx;
    result.debias_n_kept   = 0;
    result.debias_declined = gate_declined ? gate_declined : "not_converged";
    return;
  }
  SubspaceDebiasOutcome db = compute_subspace_debias(
      n_x, mode, chol, sparse_solver, use_sparse, eval_log_joint, x_buf, *opts,
      hard, present_draw);
  result.debias_idx      = std::move(db.idx);
  result.debias_draws    = std::move(db.draws);
  result.debias_sigma_ss = std::move(db.sigma_ss);
  result.debias_n_kept   = db.n_kept;
  result.debias_accept   = db.accept;
  result.debias_scale    = db.scale;
  result.debias_declined = std::move(db.declined);
  result.debias_importance = std::move(db.importance);
  result.debias_importance_log_weights = std::move(db.importance_log_weights);
}

} // namespace tulpa

#endif // TULPA_SUBSPACE_DEBIAS_H
