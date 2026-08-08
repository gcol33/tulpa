// subspace_debias.h
//
// Subspace debias (gcol33/tulpa#304): exact Metropolis correction applied to
// ONLY the latent coordinates the inner-layer diagnostics flagged, with the rest
// carried at their Gaussian conditional.
//
// WHAT IS SAMPLED. Write H for the converged Newton Hessian (posterior
// precision), Sigma = H^{-1}, and S for a set of q latent indices. Solving
// H V = E_S gives the n_x x q matrix V whose column b is Sigma e_{S_b}, and
// Sigma_SS = V[S, :] is the inner Laplace's own marginal covariance of x_S. The
// map
//
//   x(u) = mode + V Sigma_SS^{-1} L u,      Sigma_SS = L L'
//
// satisfies x_S(u) = mode_S + L u and x_{-S}(u) = E_piG(x_{-S} | x_S = x_S(u)),
// so it is the q-dimensional Gaussian-conditional-mean SURFACE through the
// joint mode -- the exact generalization of the one-dimensional curve both
// inner-layer diagnostics already walk (inner_laplace_probe.h). The Gaussian
// approximation restricted to it is exactly N(0, I) in u, because
// (V Sigma_SS^{-1} L)' H (V Sigma_SS^{-1} L) = L' Sigma_SS^{-1} L = I.
//
// The target is the joint log density along that surface. Up to a constant, it
// is the Gaussian-conditioned marginal of x_S: the exact conditional of x_{-S}
// is replaced by the Gaussian one, whose normalizer does not move with u, while
// the non-Gaussianity of the likelihood in the x_S directions is kept exactly.
// That is the whole approximation, and it is the reason S may have to be closed
// under strong posterior coupling -- a coordinate strongly coupled to a member
// of S and left OUT of S is being carried linearly, which is precisely the
// error the correction is trying to remove. Adding it to S moves it under the
// sampler instead. Which side of that line a given fit falls on is a measured
// question, not an assumed one (see R/subspace_debias.R and
// tests/testthat/test-subspace-debias-recovery.R).
//
// WHY RANDOM-WALK METROPOLIS. Every evaluation is one call of the Newton loop's
// own penalized objective -- O(N), no factorization, no derivative -- so a
// gradient-based sampler would buy nothing here and would need a derivative the
// loop does not expose along the surface. The walk is spherical in u, where the
// Gaussian is exactly N(0, I), so the Laplace shaping is built into the
// coordinates rather than into a proposal covariance. Scale schedule, target
// acceptance and accept test are the shared primitives in rwmh.h, the same ones
// the covariance Gibbs sweep runs on.
//
// EMPTY S IS NOT A SPECIAL CASE OF ANYTHING. With no flagged coordinate the
// routine is never entered, nothing is drawn, and no random number is consumed,
// so a fit whose selector returns nothing is the plain Laplace fit unchanged.

#ifndef TULPA_SUBSPACE_DEBIAS_H
#define TULPA_SUBSPACE_DEBIAS_H

#include "inner_laplace_probe.h"
#include "laplace_cholesky.h"
#include "rwmh.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

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
};

struct SubspaceDebiasOutcome {
  std::vector<int> idx;         // S, echoed back (0-based)
  std::vector<double> draws;    // n_kept * q, column-major [n_kept x q];
                                // entry (s, b) is x_{S_b} - mode_{S_b}
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
};

// Build the Gaussian-conditional-mean surface through the mode over the index
// set S, then run the random-walk Metropolis correction on it.
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
    const SubspaceDebiasOptions& opts
) {
  SubspaceDebiasOutcome out;
  out.idx = opts.idx;
  const int q = static_cast<int>(opts.idx.size());
  if (q == 0) { out.declined = "no_probe_indices"; return out; }

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

  // --- random-walk Metropolis in u -----------------------------------------
  std::vector<double> u(q, 0.0), u_prop(q, 0.0);
  auto target_at = [&](const std::vector<double>& uu) -> double {
    for (int k = 0; k < n_x; k++) {
      double acc = mode[k];
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

  double scale = rw_init_scale(q);
  const double target = rw_target_accept(q);
  const int thin = std::max(1, opts.thin);
  const int n_warm = std::max(0, opts.n_warmup);
  const int n_rec = std::max(0, opts.n_iter);
  const int n_sweep = n_warm + n_rec;

  int n_kept = 0;
  for (int s = n_warm; s < n_sweep; s += thin) n_kept++;
  out.n_kept = n_kept;
  out.draws.assign(static_cast<std::size_t>(n_kept) * q, 0.0);

  long acc_rec = 0;
  int kept = 0;
  for (int sweep = 1; sweep <= n_sweep; sweep++) {
    const bool adapting = sweep <= n_warm;
    for (int a = 0; a < q; a++) u_prop[a] = u[a] + scale * R::rnorm(0.0, 1.0);
    const double lp_prop = target_at(u_prop);
    const bool acc = rw_accept(lp_prop - lp_cur);
    if (acc) { u = u_prop; lp_cur = lp_prop; }
    if (adapting) {
      scale = rw_adapt_scale(scale, rw_adapt_gain(sweep), acc ? 1.0 : 0.0, target);
    } else if ((sweep - n_warm - 1) % thin == 0 && kept < n_kept) {
      // d = L u, the draw in the natural x_S - mode_S coordinates.
      for (int a = 0; a < q; a++) {
        double d = 0.0;
        for (int c = 0; c <= a; c++) d += L[a + c * q] * u[c];
        out.draws[static_cast<std::size_t>(a) * n_kept + kept] = d;
      }
      acc_rec += acc ? 1 : 0;
      kept++;
    }
  }

  for (int k = 0; k < n_x; k++) x_buf[k] = mode[k];   // restore
  out.n_kept = kept;
  out.accept = (kept > 0) ? static_cast<double>(acc_rec) / kept
                          : std::numeric_limits<double>::quiet_NaN();
  out.scale = scale;
  return out;
}

// Guard, run, and record the correction on a solver result, in one call.
//
// Every Newton loop that can carry the correction -- the single-arm spec loop
// (laplace_newton.h), the dense joint loop and the sparse joint loop
// (laplace_newton_joint*.h) -- reaches the sampler through here, so the
// "nullptr or empty index set is a no-op" contract and the mapping from
// SubspaceDebiasOutcome onto the result fields are written once. Templated on
// the result type only to avoid including laplace_core.h from here; the single
// instantiation is LaplaceResult.
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
    const SubspaceDebiasOptions* opts
) {
  if (!opts || opts->idx.empty()) return;
  SubspaceDebiasOutcome db = compute_subspace_debias(
      n_x, mode, chol, sparse_solver, use_sparse, eval_log_joint, x_buf, *opts);
  result.debias_idx      = std::move(db.idx);
  result.debias_draws    = std::move(db.draws);
  result.debias_sigma_ss = std::move(db.sigma_ss);
  result.debias_n_kept   = db.n_kept;
  result.debias_accept   = db.accept;
  result.debias_scale    = db.scale;
  result.debias_declined = std::move(db.declined);
}

} // namespace tulpa

#endif // TULPA_SUBSPACE_DEBIAS_H
