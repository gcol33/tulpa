// inner_cila.h
//
// Corrected integrated Laplace: an importance-sampling
// DEBIAS applied to the inner layer, after Lai, Margossian & Sheldon,
// arXiv:2605.20345 eq (4)-(7) and the Sec 6 / Proposition 5 weighted-particle
// recovery of the latent field.
//
// WHICH LAYER THIS IS. The inner Gaussian stays exactly as the Newton loop
// built it and the outer grid stays exactly as the integrator laid it out. What
// changes is that the cell's marginal is evaluated at DRAWS from the inner
// Gaussian rather than at its mode alone: with z ~ hatpi(z | theta, y),
//
//   hatpi_z(theta, y) = pi(theta) pi(z | theta) pi(y | theta, z)
//                       / hatpi(z | theta, y)
//
// is an unbiased estimator of pi(theta, y) (the paper's Proposition 1), and the
// same M ratios reweight the draws themselves into a corrected latent posterior.
// Increasing M drives both toward the exact posterior, which is what makes this
// a debias rather than a second approximation.
//
// THE RATIO IS NOT THE ONE inner_laplace_is.h FORMS, AND THE DIFFERENCE IS
// SCOPE. Both objects are "joint density over inner Gaussian" at the same mode
// and the same live factor, and both read the density through the Newton loop's
// own penalized-objective closure -- which is why the two share
// inner_laplace_probe.h's solve and this file needs no likelihood knowledge
// either. But the inner importance k-hat is deliberately restricted to ONE
// dimension per probed index: it walks the Gaussian-conditional-mean curve
// x(t) = mode + (t / sigma_i^2) v_i with every other coordinate pinned at its
// conditional mean, because an importance k-hat over n_x coordinates would
// report n_x rather than the approximation. A DEBIAS cannot use that slice: the
// conditional mean is not a draw, so reweighting along the curve corrects the
// shape of one marginal under the Gaussian's own conditional structure and
// leaves the joint untouched. The correction therefore samples the FULL latent
// block, z = mode + L^-T eps with H = L L', which is the regime the k-hat's
// header warns about. That the correction nonetheless works here is a measured
// fact about the proposal rather than an assumption: the inner Gaussian is a
// near-exact proposal for these models, and dimension costs efficiency only in
// proportion to how far it is from exact. The same PSIS instrument grades it
// (tulpa_psis on these weights), so the cost of the wider scope is reported by
// the fit rather than assumed away.
//
// WHY THE RANDOMNESS IS ENGINE-OWNED. Every auxiliary point set here comes from
// a deterministic engine stream keyed by the cell index (splitmix64, the same
// generator inner_laplace_is.h draws from) or from the Sobol net, never from R's
// RNG. Three things follow: requesting the correction leaves a fit's other
// posterior draws bit-for-bit unchanged, the outer grid can still be integrated
// in parallel (the subspace debias forces a serial grid precisely because it
// draws from R), and the reported correction does not flap with the seed. The
// shift realization of the randomized-QMC variant is fixed by `seed`; a
// different seed is an independent realization of the same estimator.
//
// THE REDUCTION THAT PINS IT. The paper notes that the uncorrected integrated
// Laplace is the QMC variant evaluated at u_i = 1/2 for every i, which puts
// z = mode. At that single auxiliary point the estimator below returns
// log p_joint(mode) + (d/2) log 2pi - (1/2) log|H|, which is the Newton loop's
// own log_marginal term for term. That is the arbiter the tests hold it to.

#ifndef TULPA_INNER_CILA_H
#define TULPA_INNER_CILA_H

#include "laplace_cholesky.h"
#include "sobol.h"
#include "sparse_cholesky.h"
#include <Rcpp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace tulpa {

enum class CilaVariant : int {
  QMC  = 0,   // eq (6): the fixed Sobol net, no randomization
  IS   = 1,   // eq (5): iid draws from the inner Gaussian
  RQMC = 2    // eq (7): the net under n_shift independent random shifts
};

// Unit-cube points are pushed through the normal quantile, so a point at an
// endpoint would map to an infinite deviate. Both the net and the shifted net
// can land exactly on 0, so the clamp is not optional.
inline constexpr double CILA_U_CLAMP = 9.094947017729282e-13;  // 2^-40

// Auxiliary draws are turned into latent vectors a chunk at a time. The sparse
// path solves a whole chunk in one CHOLMOD call, so the chunk is what makes the
// per-draw cost the triangular solve rather than a call into the solver; the
// bound on it is the transient, n_x * CILA_DRAW_CHUNK doubles, which stays
// under a megabyte for latent dimensions in the low thousands.
inline constexpr int CILA_DRAW_CHUNK = 64;

struct CilaOptions {
  int n_points = 1024;                 // total auxiliary points M per cell
  CilaVariant variant = CilaVariant::QMC;
  int n_shift = 8;                     // RQMC only; M splits across the shifts
  int n_fixed = 0;                     // retained leading latent prefix
  // Stream key, supplied by the caller (R/cila.R owns the default). Zero leaves
  // each cell's stream keyed by its grid index alone.
  std::uint64_t seed = 0;

  bool active() const { return n_points > 0; }
};

struct CilaOutcome {
  std::vector<double> log_w;   // M unnormalized log importance ratios
  std::vector<double> fixed;   // M * n_fixed, column-major [M x n_fixed]
  double log_marginal = std::numeric_limits<double>::quiet_NaN();
  int n_points = 0;
  int n_fixed = 0;
  CilaVariant variant_used = CilaVariant::QMC;
  // Why the correction produced nothing, when it produced nothing. Empty on
  // success. Vocabulary: "not_converged" (no mode to propose from),
  // "sparse_factor_unavailable" (the cell routed to the sparse solver and left
  // no live factor to draw through), "sparse_factor_not_ll" (the live factor is
  // a simplicial LDL', which carries no square root the draw can apply --
  // CHOLMOD_DLt gives D^-1 where a draw needs D^-1/2), "degenerate_proposal" (a
  // non-positive pivot or a non-finite log-determinant, so the inner Gaussian
  // has no density), "objective_not_finite" (the joint density is not finite at
  // the mode).
  std::string declined;
  // Set when the REQUESTED point set could not be built and a different one was
  // used instead, so a fit never silently reports a variant it did not run.
  // Vocabulary: "sobol_dim_exceeded".
  std::string fallback;
};

// splitmix64, keyed so that each cell of an outer grid draws an independent
// stream at a fixed seed. Same generator as inner_laplace_is.h; kept as a small
// stateful helper here because this file needs uniforms as well as normals.
struct CilaStream {
  std::uint64_t state;
  explicit CilaStream(std::uint64_t key) : state(key) {}
  double u01() {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z =  z ^ (z >> 31);
    return (static_cast<double>(z >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  }
  double normal() {
    const double u1 = u01();
    const double u2 = u01();
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(6.283185307179586 * u2);
  }
};

inline double cila_u_to_z(double u) {
  const double c = std::min(std::max(u, CILA_U_CLAMP), 1.0 - CILA_U_CLAMP);
  return R::qnorm(c, 0.0, 1.0, 1, 0);
}

// Build the M x d standard-normal auxiliary matrix, column-major.
// Returns false only when nothing could be built at all.
inline bool cila_build_aux(const CilaOptions& opts, int d, std::uint64_t key,
                           std::vector<double>& eps, CilaVariant& used,
                           std::string& fallback) {
  const int M = opts.n_points;
  if (M <= 0 || d <= 0) return false;
  used = opts.variant;
  if ((used == CilaVariant::QMC || used == CilaVariant::RQMC) &&
      d > SOBOL_MAX_DIM) {
    fallback = "sobol_dim_exceeded";
    used = CilaVariant::IS;
  }
  eps.assign(static_cast<std::size_t>(M) * d, 0.0);

  if (used == CilaVariant::IS) {
    CilaStream rng(key);
    for (std::size_t e = 0; e < eps.size(); e++) eps[e] = rng.normal();
    return true;
  }

  if (used == CilaVariant::QMC) {
    std::vector<double> u(static_cast<std::size_t>(M) * d);
    if (!sobol_points(M, d, u.data())) return false;
    for (std::size_t e = 0; e < eps.size(); e++) eps[e] = cila_u_to_z(u[e]);
    return true;
  }

  // RQMC: n = M / n_shift points of the net, stacked over n_shift shifts.
  const int n_shift = std::max(1, std::min(opts.n_shift, M));
  const int n = M / n_shift;
  if (n <= 0) return false;
  std::vector<double> v(static_cast<std::size_t>(n) * d);
  if (!sobol_points(n, d, v.data())) return false;
  CilaStream rng(key);
  std::vector<double> shift(d);
  const int M_used = n * n_shift;
  eps.assign(static_cast<std::size_t>(M_used) * d, 0.0);
  for (int s = 0; s < n_shift; s++) {
    for (int j = 0; j < d; j++) shift[j] = rng.u01();
    for (int j = 0; j < d; j++) {
      for (int i = 0; i < n; i++) {
        double x = v[static_cast<std::size_t>(j) * n + i] + shift[j];
        if (x >= 1.0) x -= 1.0;
        eps[static_cast<std::size_t>(j) * M_used + s * n + i] = cila_u_to_z(x);
      }
    }
  }
  return true;
}

// Run the correction at one outer cell.
//
// `eval_log_joint(x) -> double` is the Newton loop's own penalized-objective
// closure (log-likelihood + log-prior at latent x), so this routine carries no
// likelihood knowledge and serves the single-arm and joint loops alike.
// `present_draw(x)` is applied to each drawn latent vector before its leading
// prefix is retained, so a loop that presents its mode under a post-Newton
// centering fold presents its draws the same way; pass a no-op where the
// reported mode is the Newton iterate itself.
//
// `x_buf` is caller-supplied scratch sized n_x; it must hold `mode` on entry and
// is restored to it on return. It carries the draw into `eval_log_joint` and
// then into `present_draw`, which rewrites it in place: the next draw overwrites
// every entry and the mode is written back at the end, so the two uses never
// meet. Holding a second Rcpp vector for the presented copy would allocate from
// R's heap, and this routine runs on the outer grid's worker threads, where
// Rf_allocVector is not thread-safe.
//
// The factor is reused as-is, with no refactorization: a draw is one
// back-substitution against it.
template <typename EvalLogJoint, typename PresentDraw>
inline CilaOutcome compute_inner_cila(
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    EvalLogJoint eval_log_joint,
    PresentDraw present_draw,
    Rcpp::NumericVector& x_buf,
    const CilaOptions& opts,
    std::uint64_t cell_key
) {
  CilaOutcome out;
  out.n_fixed = std::max(0, std::min(opts.n_fixed, n_x));

  // 0.5 log|H| from the live factor, which is also the check that the inner
  // Gaussian has a density at all. The sparse factor reports it directly; the
  // dense one is the sum of the log-diagonal of its own L.
  double half_log_det = 0.0;
  if (use_sparse) {
    // A draw is P' L^-T eps against the SAME factor the solve left resident, so
    // the sparse cell proposes from its own inner Gaussian with no
    // refactorization. Only an LL' factor has a square root
    // on that route; a simplicial LDL' fallback says so instead of drawing from
    // the wrong covariance.
    if (!sparse_solver.factored()) {
      out.declined = "sparse_factor_unavailable";
      return out;
    }
    const double ld = sparse_solver.log_determinant();
    if (!std::isfinite(ld)) {
      out.declined = "degenerate_proposal";
      return out;
    }
    half_log_det = 0.5 * ld;
  } else {
    for (int j = 0; j < n_x; j++) {
      const double d = chol.L[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * n_x];
      if (!(d > 0.0) || !std::isfinite(d)) {
        out.declined = "degenerate_proposal";
        return out;
      }
      half_log_det += std::log(d);
    }
  }

  std::vector<double> eps;
  CilaVariant used = opts.variant;
  if (!cila_build_aux(opts, n_x, cell_key ^ opts.seed, eps, used,
                      out.fallback)) {
    out.declined = "degenerate_proposal";
    return out;
  }
  out.variant_used = used;
  const int M = static_cast<int>(eps.size() / static_cast<std::size_t>(n_x));
  out.n_points = M;
  out.log_w.assign(M, std::numeric_limits<double>::quiet_NaN());
  out.fixed.assign(static_cast<std::size_t>(M) * out.n_fixed, 0.0);

  // The proposal's log density is constant except for the standardized radius,
  // because z = mode + L^-T eps carries (z - mode)' H (z - mode) = eps' eps.
  const double log_q_const =
      -0.5 * n_x * 1.8378770664093453 + half_log_det;

  // Draws are produced a CHUNK at a time rather than one at a time. On the
  // sparse path that is what makes the correction affordable at all: CHOLMOD
  // solves a whole [n_x x chunk] block in one call, so the per-draw cost is the
  // triangular solve itself instead of a call into the solver. On the dense
  // path the chunk is a bounded scratch buffer and the back-substitution is
  // unchanged. The chunk bounds the transient at n_x * CILA_DRAW_CHUNK doubles,
  // which is what keeps a large sparse latent from materialising an [M x n_x]
  // block.
  std::vector<double> block(static_cast<std::size_t>(n_x) * CILA_DRAW_CHUNK, 0.0);
  std::vector<double> eps_block;
  if (use_sparse) {
    eps_block.assign(static_cast<std::size_t>(n_x) * CILA_DRAW_CHUNK, 0.0);
  }
  for (int i0 = 0; i0 < M; i0 += CILA_DRAW_CHUNK) {
    const int nc = std::min(CILA_DRAW_CHUNK, M - i0);
    if (use_sparse) {
      for (int c = 0; c < nc; c++) {
        for (int j = 0; j < n_x; j++) {
          eps_block[static_cast<std::size_t>(c) * n_x + j] =
              eps[static_cast<std::size_t>(j) * M + (i0 + c)];
        }
      }
      if (!sparse_solver.apply_inv_chol_factor(eps_block.data(), block.data(),
                                               n_x, nc)) {
        out.declined = "sparse_factor_not_ll";
        out.log_w.clear();
        out.fixed.clear();
        out.n_points = 0;
        return out;
      }
    } else {
      for (int c = 0; c < nc; c++) {
        // Back-substitution L' draw = eps_i, the only linear algebra per draw.
        double* draw = block.data() + static_cast<std::size_t>(c) * n_x;
        for (int j = n_x - 1; j >= 0; j--) {
          double sum = eps[static_cast<std::size_t>(j) * M + (i0 + c)];
          for (int k = j + 1; k < n_x; k++) {
            sum -= chol.L[static_cast<std::size_t>(k) +
                          static_cast<std::size_t>(j) * n_x] * draw[k];
          }
          draw[j] = sum /
              chol.L[static_cast<std::size_t>(j) + static_cast<std::size_t>(j) * n_x];
        }
      }
    }
    for (int c = 0; c < nc; c++) {
      const int i = i0 + c;
      const double* draw = block.data() + static_cast<std::size_t>(c) * n_x;
      double quad = 0.0;
      for (int j = n_x - 1; j >= 0; j--) {
        const double e = eps[static_cast<std::size_t>(j) * M + i];
        quad += e * e;
      }
      for (int j = 0; j < n_x; j++) x_buf[j] = mode[j] + draw[j];
      const double lp = eval_log_joint(x_buf);
      out.log_w[i] = lp - (log_q_const - 0.5 * quad);
      if (out.n_fixed > 0) {
        // The presented draw replaces the buffer's contents; the next draw
        // rewrites every entry and the mode is restored below.
        present_draw(x_buf);
        for (int b = 0; b < out.n_fixed; b++) {
          out.fixed[static_cast<std::size_t>(b) * M + i] = x_buf[b];
        }
      }
    }
  }

  for (int j = 0; j < n_x; j++) x_buf[j] = mode[j];   // restore

  double mx = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < M; i++) {
    if (std::isfinite(out.log_w[i]) && out.log_w[i] > mx) mx = out.log_w[i];
  }
  if (!std::isfinite(mx)) {
    out.declined = "objective_not_finite";
    out.log_w.clear();
    out.fixed.clear();
    out.n_points = 0;
    return out;
  }
  double acc = 0.0;
  for (int i = 0; i < M; i++) {
    acc += std::isfinite(out.log_w[i]) ? std::exp(out.log_w[i] - mx) : 0.0;
  }
  out.log_marginal = mx + std::log(acc) - std::log(static_cast<double>(M));
  return out;
}

// Guard, run, and record the correction on a solver result, in one call. Every
// Newton loop that can carry it reaches the estimator through here, so the
// "nullptr or inactive request is a no-op" contract and the mapping onto the
// result fields are written once. Templated on the result type only to avoid
// including laplace_core.h from here.
template <typename Result, typename EvalLogJoint, typename PresentDraw>
inline void run_inner_cila(
    Result& result,
    int n_x,
    const std::vector<double>& mode,
    DenseCholeskyScratch& chol,
    SparseCholeskySolver& sparse_solver,
    bool use_sparse,
    EvalLogJoint eval_log_joint,
    PresentDraw present_draw,
    Rcpp::NumericVector& x_buf,
    const CilaOptions* opts,
    std::uint64_t cell_key
) {
  if (!opts || !opts->active()) return;
  if (!result.converged) {
    result.cila_declined = "not_converged";
    result.cila_requested = true;
    return;
  }
  CilaOutcome cl = compute_inner_cila(n_x, mode, chol, sparse_solver, use_sparse,
                                      eval_log_joint, present_draw, x_buf, *opts,
                                      cell_key);
  result.cila_requested   = true;
  result.cila_log_w       = std::move(cl.log_w);
  result.cila_fixed       = std::move(cl.fixed);
  result.cila_log_marginal = cl.log_marginal;
  result.cila_n_points    = cl.n_points;
  result.cila_n_fixed     = cl.n_fixed;
  result.cila_variant     = static_cast<int>(cl.variant_used);
  result.cila_declined    = std::move(cl.declined);
  result.cila_fallback    = std::move(cl.fallback);
}

} // namespace tulpa

#endif // TULPA_INNER_CILA_H
