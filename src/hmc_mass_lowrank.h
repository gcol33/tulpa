// hmc_mass_lowrank.h
// A mass matrix that is DIAGONAL PLUS LOW RANK over one contiguous parameter
// block, with the low-rank directions given as GROUP SUMS.
//
//   M_block = D + U Lambda U',   D = diag(1 / var),
//
// where column g of U is the indicator of a group of block coordinates, so
// (U' x)_g = sum_{i in g} x_i. That is the shape every soft sum-to-zero
// penalty in the engine contributes to a precision: `s2z_precision(n) *
// (sum_i phi_i)^2` is one group over the whole field, and a Knorr-Held
// Type-IV interaction pinned on both margins is S row groups plus T column
// groups. Nothing here is Type-IV-specific -- the caller supplies the groups.
//
// WHY a diagonal cannot do this job. A diagonal metric only rescales
// coordinates, so what it can reach is bounded by cond(Q) after the best
// diagonal rescaling. Measured on four Type-IV configurations
// (dev_notes/issue585/cond585.R): cond(Q) 1.3e5 to 2.1e5, after the marginal
// rescaling diag(Q^-1) 1.1e5 to 2.0e5, after Jacobi unchanged to four figures
// -- and 11.5 to 51 on the subspace with the margin directions deleted. The
// margin eigendirections are 1_S (x) a and b (x) 1_T, linear combinations
// rather than coordinates, which is why no diagonal reaches them and why the
// remedy has to carry those directions explicitly (gcol33/tulpa#597).
//
// COST. The inverse is Woodbury on a k x k inner matrix, k = the number of
// groups:
//
//   M^-1 = D^-1 - D^-1 U (Lambda^-1 + U' D^-1 U)^-1 U' D^-1,
//
// so one k x k factorization per metric install and O(n + nnz(U) + k^2) per
// leapfrog step, against the O(n^2) a dense metric pays. The inner matrix is
// PD for any positive `lambda` and `var` even where U' D^-1 U is singular --
// the Type-IV margins are, since the grand total appears in every group --
// because Lambda^-1 is added to it.
#ifndef TULPA_HMC_MASS_LOWRANK_H
#define TULPA_HMC_MASS_LOWRANK_H

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <Eigen/Dense>

namespace tulpa_hmc {

struct LowRankMassTerm {
  // Placement in the full parameter vector.
  int start = 0;
  int n = 0;

  // Groups as CSR over block-local coordinates: group g owns
  // group_idx[group_ptr[g] .. group_ptr[g+1]).
  std::vector<int> group_ptr;
  std::vector<int> group_idx;

  // One positive weight per group: the precision the penalty puts on that
  // group's sum, in the coordinate the sampler holds.
  std::vector<double> lambda;

  // D^-1 over the block -- the inverse-mass (variance) diagonal the metric
  // would carry with no low-rank term at all.
  std::vector<double> var;

  // Filled by factorize().
  std::vector<int> coord_ptr;   // transpose of the CSR: groups per coordinate
  std::vector<int> coord_grp;
  Eigen::LLT<Eigen::MatrixXd> K_llt;   // Lambda^-1 + U' D^-1 U
  bool ready = false;

  // Per-step scratch of length k. The metric object is per chain, so a mutable
  // buffer here is not shared across the parallel-chain region.
  mutable Eigen::VectorXd work_a, work_b;

  int rank() const { return static_cast<int>(lambda.size()); }
  bool covers(int j) const { return ready && j >= start && j < start + n; }

  // Build the transpose, assemble the inner matrix and factorize it. Returns
  // false -- leaving ready = false -- on a malformed layout, a non-positive
  // weight or variance, or a refused factorization; the caller then keeps the
  // plain diagonal rather than sampling under a metric it could not invert.
  bool factorize() {
    ready = false;
    const int k = rank();
    if (n <= 0 || k <= 0) return false;
    if (static_cast<int>(var.size()) != n) return false;
    if (static_cast<int>(group_ptr.size()) != k + 1) return false;
    if (group_ptr.front() != 0 ||
        group_ptr.back() != static_cast<int>(group_idx.size())) return false;
    for (int i = 0; i < n; i++) {
      if (!(var[i] > 0.0) || !std::isfinite(var[i])) return false;
    }
    for (int g = 0; g < k; g++) {
      if (!(lambda[g] > 0.0) || !std::isfinite(lambda[g])) return false;
      if (group_ptr[g] > group_ptr[g + 1]) return false;
    }
    for (int e = 0; e < static_cast<int>(group_idx.size()); e++) {
      if (group_idx[e] < 0 || group_idx[e] >= n) return false;
    }

    // Transpose: which groups each coordinate belongs to.
    coord_ptr.assign(n + 1, 0);
    for (int e = 0; e < static_cast<int>(group_idx.size()); e++) {
      coord_ptr[group_idx[e] + 1]++;
    }
    for (int i = 0; i < n; i++) coord_ptr[i + 1] += coord_ptr[i];
    coord_grp.assign(group_idx.size(), 0);
    {
      std::vector<int> fill(coord_ptr.begin(), coord_ptr.end() - 1);
      for (int g = 0; g < k; g++) {
        for (int e = group_ptr[g]; e < group_ptr[g + 1]; e++) {
          coord_grp[fill[group_idx[e]]++] = g;
        }
      }
    }

    // K = Lambda^-1 + U' D^-1 U. Entry (a, b) is the variance summed over the
    // coordinates both groups contain, so one pass over the transpose fills
    // every pair without forming U.
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(k, k);
    for (int g = 0; g < k; g++) K(g, g) = 1.0 / lambda[g];
    for (int i = 0; i < n; i++) {
      const double v = var[i];
      for (int e1 = coord_ptr[i]; e1 < coord_ptr[i + 1]; e1++) {
        const int a = coord_grp[e1];
        for (int e2 = coord_ptr[i]; e2 < coord_ptr[i + 1]; e2++) {
          K(a, coord_grp[e2]) += v;
        }
      }
    }

    K_llt.compute(K);
    if (K_llt.info() != Eigen::Success) return false;
    work_a.resize(k);
    work_b.resize(k);
    ready = true;
    return true;
  }

  // out[start .. start+n) = (M^-1 p)[start .. start+n). Reads p over the block
  // only; the caller owns every coordinate outside it.
  void apply_inv(const double* p_full, double* out_full) const {
    const double* p = p_full + start;
    double* out = out_full + start;
    const int k = rank();
    for (int g = 0; g < k; g++) {
      double s = 0.0;
      for (int e = group_ptr[g]; e < group_ptr[g + 1]; e++) {
        const int i = group_idx[e];
        s += var[i] * p[i];
      }
      work_a[g] = s;
    }
    work_b = K_llt.solve(work_a);
    for (int i = 0; i < n; i++) {
      double corr = 0.0;
      for (int e = coord_ptr[i]; e < coord_ptr[i + 1]; e++) {
        corr += work_b[coord_grp[e]];
      }
      out[i] = var[i] * (p[i] - corr);
    }
  }

  // p_block' M^-1 p_block. The Woodbury correction is a' K^-1 a on the same
  // group sums apply_inv forms, so this needs no n-length buffer.
  double quadform(const double* p_full) const {
    const double* p = p_full + start;
    const int k = rank();
    double q = 0.0;
    for (int i = 0; i < n; i++) q += var[i] * p[i] * p[i];
    for (int g = 0; g < k; g++) {
      double s = 0.0;
      for (int e = group_ptr[g]; e < group_ptr[g + 1]; e++) {
        const int i = group_idx[e];
        s += var[i] * p[i];
      }
      work_a[g] = s;
    }
    work_b = K_llt.solve(work_a);
    return q - work_a.dot(work_b);
  }

  // Draw the block's momentum, p_block ~ N(0, M_block). OVERWRITES the block.
  //
  // p = D^(1/2) z1 + U Lambda^(1/2) z2 with z1, z2 independent standard
  // normals has covariance D + U Lambda U' = M EXACTLY: it is a sum of two
  // independent Gaussians, so no square root of the sum is needed and none is
  // formed. (gcol33/tulpa#597 records this construction as a trap. It is not
  // one -- test-lowrank-mass.R scores the realized covariance against M and
  // pins the algebra -- and the reading it was written against, taking the two
  // terms as a claimed factorization of M^(1/2), is what the sum is not.)
  void sample_momentum(double* p_full, std::mt19937& rng) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    double* p = p_full + start;
    for (int i = 0; i < n; i++) p[i] = normal(rng) / std::sqrt(var[i]);
    const int k = rank();
    for (int g = 0; g < k; g++) {
      const double u = normal(rng) * std::sqrt(lambda[g]);
      for (int e = group_ptr[g]; e < group_ptr[g + 1]; e++) {
        p[group_idx[e]] += u;
      }
    }
  }
};

// The two-margin instance: block coordinates laid out as s * T + t (temporal
// varying fastest, the convention SpatiotemporalData::st_flat carries), pinned
// along the S row sums and the T column sums. `var` is the block's
// inverse-mass diagonal, copied in.
inline LowRankMassTerm make_margin_mass_term(
    int start, int S, int T,
    double lambda_row, double lambda_col,
    const double* var, int n_var
) {
  LowRankMassTerm t;
  if (S <= 0 || T <= 0 || n_var != S * T) return t;   // rank 0: never ready
  t.start = start;
  t.n = S * T;
  t.var.assign(var, var + t.n);
  t.lambda.reserve(S + T);
  t.group_ptr.reserve(S + T + 1);
  t.group_idx.reserve(2 * t.n);
  t.group_ptr.push_back(0);
  for (int s = 0; s < S; s++) {                 // row sums: I_S (x) J_T
    for (int tt = 0; tt < T; tt++) t.group_idx.push_back(s * T + tt);
    t.group_ptr.push_back(static_cast<int>(t.group_idx.size()));
    t.lambda.push_back(lambda_row);
  }
  for (int tt = 0; tt < T; tt++) {              // column sums: J_S (x) I_T
    for (int s = 0; s < S; s++) t.group_idx.push_back(s * T + tt);
    t.group_ptr.push_back(static_cast<int>(t.group_idx.size()));
    t.lambda.push_back(lambda_col);
  }
  return t;
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_MASS_LOWRANK_H
