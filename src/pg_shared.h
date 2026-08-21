// pg_shared.h
// Shared helpers for Pólya-Gamma Gibbs samplers

#ifndef TULPA_PG_SHARED_H
#define TULPA_PG_SHARED_H

#include <Rcpp.h>
#include <cmath>
#include <algorithm>
#include <vector>

#include "pg_binomial.h"   // update_beta, update_re
#include "pg_rng.h"        // rpg_int, rpg_real
#include "linalg_fast.h"   // shared small-dense Cholesky / NNGP solve core
#include "omp_threads.h"   // tulpa_omp_team_size_req, tulpa_parallel_for

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa {

// ============================================================================
// Input validation
//
// Rcpp's vector and matrix subscripts are unchecked, so every R-supplied index
// is validated once at kernel entry and used raw in the sweeps.
// ============================================================================

// A 1-based R index vector used as an array subscript: `idx` must have length
// `expected_len` and every entry must be a non-NA integer in [1, upper].
inline void pg_check_index(const Rcpp::IntegerVector& idx, int expected_len,
                           int upper, const char* name) {
  if (idx.size() != expected_len) {
    Rcpp::stop("`%s` has length %d but must have length %d.",
               name, static_cast<int>(idx.size()), expected_len);
  }
  if (upper < 1) {
    Rcpp::stop("`%s` indexes an empty range (upper bound %d).", name, upper);
  }
  for (int i = 0; i < expected_len; i++) {
    const int v = idx[i];
    if (Rcpp::IntegerVector::is_na(v) || v < 1 || v > upper) {
      Rcpp::stop("`%s[%d]` is %d; must be a 1-based index in [1, %d].",
                 name, i + 1, v, upper);
    }
  }
}

// Number of saved draws the sweep loop actually produces: it saves whenever
// (iter - n_warmup) %% thin == 0 for iter in [n_warmup, n_iter), which is
// ceil((n_iter - n_warmup) / thin) iterations.
inline int pg_n_save(int n_iter, int n_warmup, int thin) {
  if (thin < 1) {
    Rcpp::stop("`thin` must be >= 1; got %d.", thin);
  }
  if (n_warmup < 0) {
    Rcpp::stop("`n_warmup` must be >= 0; got %d.", n_warmup);
  }
  if (n_iter < n_warmup) {
    Rcpp::stop("`n_iter` (%d) must be >= `n_warmup` (%d).", n_iter, n_warmup);
  }
  return (n_iter - n_warmup + thin - 1) / thin;
}

// ============================================================================
// Adjacency
//
// Flat CSR form of the 1-based R neighbour list the areal kernels take, built
// and validated once per fit: entry ranges, the `n_neighbors` / `adj_list`
// agreement, and the connected-component labelling the ICAR rank correction
// and the component-level block update both read.
// ============================================================================
struct PgAdjacency {
  int n = 0;
  std::vector<int> row_ptr;    // n + 1
  std::vector<int> col_idx;    // 0-based neighbour ids
  std::vector<int> component;  // 0-based component label per unit
  int n_components = 0;

  int degree(int j) const { return row_ptr[j + 1] - row_ptr[j]; }
};

// The neighbour list a kernel walks raw, checked once at entry: `n_neighbors`
// bounds the loop while `adj_list` supplies the entries, so a disagreement
// between them reads past the end of the neighbour vector, and a 1-based entry
// outside [1, n_units] subscripts the field vector out of bounds.
inline void pg_check_adjacency(const Rcpp::List& adj_list,
                               const Rcpp::IntegerVector& n_neighbors,
                               int n_units) {
  if (n_units <= 0) {
    Rcpp::stop("`n_spatial_units` must be >= 1; got %d.", n_units);
  }
  if (adj_list.size() != n_units) {
    Rcpp::stop("`adj_list` has %d element(s) but `n_spatial_units` is %d.",
               static_cast<int>(adj_list.size()), n_units);
  }
  if (n_neighbors.size() != n_units) {
    Rcpp::stop("`n_neighbors` has length %d but `n_spatial_units` is %d.",
               static_cast<int>(n_neighbors.size()), n_units);
  }
  for (int j = 0; j < n_units; j++) {
    Rcpp::IntegerVector nb = adj_list[j];
    const int declared = n_neighbors[j];
    if (Rcpp::IntegerVector::is_na(declared) || declared < 0) {
      Rcpp::stop("`n_neighbors[%d]` is %d; must be >= 0.", j + 1, declared);
    }
    if (nb.size() != declared) {
      Rcpp::stop("`adj_list[[%d]]` holds %d neighbour(s) but `n_neighbors[%d]` "
                 "is %d.", j + 1, static_cast<int>(nb.size()), j + 1, declared);
    }
    for (int k = 0; k < declared; k++) {
      const int v = nb[k];
      if (Rcpp::IntegerVector::is_na(v) || v < 1 || v > n_units) {
        Rcpp::stop("`adj_list[[%d]][%d]` is %d; must be a 1-based unit index "
                   "in [1, %d].", j + 1, k + 1, v, n_units);
      }
    }
  }
}

inline PgAdjacency pg_build_adjacency(const Rcpp::List& adj_list,
                                      const Rcpp::IntegerVector& n_neighbors,
                                      int n_units) {
  pg_check_adjacency(adj_list, n_neighbors, n_units);

  PgAdjacency adj;
  adj.n = n_units;
  adj.row_ptr.assign(n_units + 1, 0);
  for (int j = 0; j < n_units; j++) {
    adj.row_ptr[j + 1] = adj.row_ptr[j] + n_neighbors[j];
  }

  adj.col_idx.resize(adj.row_ptr[n_units]);
  for (int j = 0; j < n_units; j++) {
    Rcpp::IntegerVector nb = adj_list[j];
    const int base = adj.row_ptr[j];
    for (int k = 0; k < nb.size(); k++) {
      adj.col_idx[base + k] = nb[k] - 1;
    }
  }

  // Connected components by depth-first traversal on the CSR form.
  adj.component.assign(n_units, -1);
  std::vector<int> stack;
  for (int s0 = 0; s0 < n_units; s0++) {
    if (adj.component[s0] >= 0) continue;
    adj.component[s0] = adj.n_components;
    stack.clear();
    stack.push_back(s0);
    while (!stack.empty()) {
      const int s = stack.back();
      stack.pop_back();
      for (int e = adj.row_ptr[s]; e < adj.row_ptr[s + 1]; e++) {
        const int t = adj.col_idx[e];
        if (adj.component[t] < 0) {
          adj.component[t] = adj.n_components;
          stack.push_back(t);
        }
      }
    }
    adj.n_components++;
  }
  return adj;
}

// An ICAR field is improper along one constant direction per graph component.
// The overall level is confounded with the intercept and absorbed there; the
// remaining component contrasts are identified by the data alone, so a
// component carrying no observation leaves the posterior improper.
inline void pg_check_components_observed(const PgAdjacency& adj,
                                         const Rcpp::IntegerVector& group) {
  if (adj.n_components <= 1) return;
  std::vector<int> seen(adj.n_components, 0);
  for (int i = 0; i < group.size(); i++) {
    seen[adj.component[group[i] - 1]] = 1;
  }
  for (int c = 0; c < adj.n_components; c++) {
    if (!seen[c]) {
      Rcpp::stop("The adjacency graph has %d connected components and "
                 "component %d carries no observation, so its field level is "
                 "unidentified and the posterior is improper. Drop the "
                 "unobserved units or connect them.", adj.n_components, c + 1);
    }
  }
}

// ============================================================================
// Polya-Gamma sufficient statistics
//
// The data precision and precision-weighted residual of a group-level effect's
// Gaussian full conditional. `group` maps observation i to a 1-based group; a
// null pointer maps observation i to group i (the continuous-field kernels,
// which carry one observation per location).
// ============================================================================
inline void pg_accumulate_stats(int N, const int* group, int n_groups,
                                const double* omega, const double* kappa,
                                const double* offset,
                                double* sum_omega, double* sum_resid) {
  std::fill(sum_omega, sum_omega + n_groups, 0.0);
  std::fill(sum_resid, sum_resid + n_groups, 0.0);
  for (int i = 0; i < N; i++) {
    const int g = group ? (group[i] - 1) : i;
    if (g < 0 || g >= n_groups) continue;
    sum_omega[g] += omega[i];
    sum_resid[g] += kappa[i] - omega[i] * offset[i];
  }
}

// ============================================================================
// Intercept detection
//
// Centring a latent effect and adding the removed level to beta[0] shifts eta
// by level * X(i, 0), so the move leaves eta unchanged -- and the sampler's
// target unchanged -- only when the first column of X is an all-ones intercept.
// A design with no columns at all makes the beta[0] write itself out of bounds.
// ============================================================================
inline bool pg_has_intercept(const Rcpp::NumericMatrix& X) {
  if (X.ncol() < 1) return false;
  for (int i = 0; i < X.nrow(); i++) {
    if (X(i, 0) != 1.0) return false;
  }
  return true;
}

inline void pg_require_intercept(bool has_intercept, const char* what) {
  if (!has_intercept) {
    Rcpp::stop("The %s Gibbs sampler centres its latent effects and absorbs "
               "the removed level into the intercept, which leaves eta "
               "unchanged only when the first column of `X` is an all-ones "
               "intercept. Supply a design with an intercept column.", what);
  }
}

inline void pg_require_intercept(const Rcpp::NumericMatrix& X,
                                 const char* what) {
  pg_require_intercept(pg_has_intercept(X), what);
}

// ============================================================================
// Sequential NNGP topology and regression weights
// ============================================================================

// Geometry-only NNGP bookkeeping, built once per fit: the ordered-position to
// original-location map, each position's parent set in original-location ids,
// and the transpose of that map -- for each position i, the positions j that
// carry i in their parent set, with the slot i occupies there. The full
// conditional of w_i picks up its own factor AND every factor in which it is a
// parent, so the child lists are what make the sweep the full conditional.
struct PgNngpTopology {
  int n = 0, nn = 0;
  std::vector<int> orig;         // n
  std::vector<int> cnt;          // n: parent count
  std::vector<int> parent_pos;   // n * nn: parent ordered positions
  std::vector<int> parent_orig;  // n * nn: parent original-location ids
  std::vector<int> child_ptr;    // n + 1
  std::vector<int> child_pos;    // ordered position of the child
  std::vector<int> child_slot;   // slot of the parent inside that child
};

inline PgNngpTopology pg_nngp_topology(const Rcpp::IntegerMatrix& nn_idx,
                                       const Rcpp::NumericMatrix& nn_dist,
                                       const Rcpp::IntegerVector& nn_order,
                                       int n_spatial, int nn) {
  if (n_spatial <= 0) Rcpp::stop("`n_spatial` must be >= 1; got %d.", n_spatial);
  if (nn < 0) Rcpp::stop("`nn` must be >= 0; got %d.", nn);
  if (nn_idx.nrow() != n_spatial || nn_idx.ncol() < nn) {
    Rcpp::stop("`nn_idx` is %dx%d but must be %dx%d.",
               static_cast<int>(nn_idx.nrow()), static_cast<int>(nn_idx.ncol()),
               n_spatial, nn);
  }
  if (nn_dist.nrow() != n_spatial || nn_dist.ncol() < nn) {
    Rcpp::stop("`nn_dist` is %dx%d but must be %dx%d.",
               static_cast<int>(nn_dist.nrow()), static_cast<int>(nn_dist.ncol()),
               n_spatial, nn);
  }
  if (nn_order.size() != n_spatial) {
    Rcpp::stop("`nn_order` has length %d but must have length %d.",
               static_cast<int>(nn_order.size()), n_spatial);
  }

  PgNngpTopology top;
  top.n = n_spatial;
  top.nn = nn;
  top.orig.resize(n_spatial);
  top.cnt.assign(n_spatial, 0);
  top.parent_pos.assign(static_cast<size_t>(n_spatial) * nn, -1);
  top.parent_orig.assign(static_cast<size_t>(n_spatial) * nn, -1);

  for (int i = 0; i < n_spatial; i++) {
    const int o = nn_order[i];
    if (Rcpp::IntegerVector::is_na(o) || o < 0 || o >= n_spatial) {
      Rcpp::stop("`nn_order[%d]` is %d; must be a 0-based location index in "
                 "[0, %d].", i + 1, o, n_spatial - 1);
    }
    top.orig[i] = o;
  }

  int n_child = 0;
  for (int i = 0; i < n_spatial; i++) {
    for (int t = 0; t < nn; t++) {
      const int pos1 = nn_idx(i, t);
      if (Rcpp::IntegerVector::is_na(pos1) || pos1 <= 0) break;
      if (pos1 > n_spatial) {
        Rcpp::stop("`nn_idx[%d, %d]` is %d; must be a 1-based ordered position "
                   "in [1, %d].", i + 1, t + 1, pos1, n_spatial);
      }
      const int pos = pos1 - 1;
      top.parent_pos[static_cast<size_t>(i) * nn + t] = pos;
      top.parent_orig[static_cast<size_t>(i) * nn + t] = top.orig[pos];
      top.cnt[i]++;
      n_child++;
    }
  }

  top.child_ptr.assign(n_spatial + 1, 0);
  for (int i = 0; i < n_spatial; i++) {
    for (int t = 0; t < top.cnt[i]; t++) {
      top.child_ptr[top.parent_pos[static_cast<size_t>(i) * nn + t] + 1]++;
    }
  }
  for (int i = 0; i < n_spatial; i++) top.child_ptr[i + 1] += top.child_ptr[i];
  top.child_pos.resize(n_child);
  top.child_slot.resize(n_child);
  std::vector<int> fill(top.child_ptr.begin(), top.child_ptr.end() - 1);
  for (int i = 0; i < n_spatial; i++) {
    for (int t = 0; t < top.cnt[i]; t++) {
      const int par = top.parent_pos[static_cast<size_t>(i) * nn + t];
      const int slot = fill[par]++;
      top.child_pos[slot] = i;
      top.child_slot[slot] = t;
    }
  }
  return top;
}

// Kriging weights B_i (on the parent set) and conditional variance F_i of the
// sequential NNGP factor, at unit marginal variance. B is scale-free and F
// scales linearly in sigma2, so one pass serves the field sweep, the sigma2
// conditional and the phi proposal.
struct PgNngpFactors {
  std::vector<double> B;  // n * nn
  std::vector<double> F;  // n
};

inline void pg_nngp_factors(double phi_gp, int cov_type,
                            const Rcpp::NumericMatrix& coords,
                            const Rcpp::NumericMatrix& nn_dist,
                            const PgNngpTopology& top,
                            PgNngpFactors& out) {
  const int n = top.n, nn = top.nn;
  out.B.assign(static_cast<size_t>(n) * nn, 0.0);
  out.F.assign(n, 1.0);

  auto compute_cov = [phi_gp, cov_type](double d) {
    if (d < 1e-10) return 1.0;
    if (cov_type == 0) {
      return std::exp(-d / phi_gp);
    } else if (cov_type == 1) {
      const double x = std::sqrt(3.0) * d / phi_gp;
      return (1.0 + x) * std::exp(-x);
    } else {
      const double x = std::sqrt(5.0) * d / phi_gp;
      return (1.0 + x + x * x / 3.0) * std::exp(-x);
    }
  };

  std::vector<double> c_vec, C_mat, L, zeros;
  for (int i = 0; i < n; i++) {
    const int m = top.cnt[i];
    if (m == 0) {
      out.F[i] = 1.0;
      continue;
    }
    c_vec.assign(m, 0.0);
    C_mat.assign(static_cast<size_t>(m) * m, 0.0);
    for (int j = 0; j < m; j++) c_vec[j] = compute_cov(nn_dist(i, j));
    for (int j1 = 0; j1 < m; j1++) {
      const int o1 = top.parent_orig[static_cast<size_t>(i) * nn + j1];
      C_mat[static_cast<size_t>(j1) * m + j1] = 1.0;
      for (int j2 = j1 + 1; j2 < m; j2++) {
        const int o2 = top.parent_orig[static_cast<size_t>(i) * nn + j2];
        const double v = compute_cov(tulpa_linalg::coords_dist(coords, o1, o2));
        C_mat[static_cast<size_t>(j1) * m + j2] = v;
        C_mat[static_cast<size_t>(j2) * m + j1] = v;
      }
    }
    L.assign(static_cast<size_t>(m) * m, 0.0);
    if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
            C_mat.data(), L.data(), m, m, tulpa_linalg::kNngpNugget)) {
      // Neighbour correlation not positive definite: condition this location
      // on nothing rather than krige it against an unusable factor. Its B row
      // stays zero and its conditional variance is the marginal 1.0.
      out.F[i] = 1.0;
      continue;
    }
    zeros.assign(m, 0.0);
    double cm = 0.0, cv = 1.0;
    tulpa_linalg::nngp_moments_from_chol<tulpa_linalg::TriLayout::RowMajor>(
        L.data(), m, m, c_vec.data(), zeros.data(), 1.0,
        tulpa_linalg::kNngpVarFloor, cm, cv,
        out.B.data() + static_cast<size_t>(i) * nn);
    out.F[i] = cv;
  }
}

// Conditional mean of the sequential NNGP factor of ordered position i.
inline double pg_nngp_parent_mean(const PgNngpTopology& top,
                                  const PgNngpFactors& fac,
                                  const std::vector<double>& w, int i) {
  double m = 0.0;
  const size_t base = static_cast<size_t>(i) * top.nn;
  for (int t = 0; t < top.cnt[i]; t++) {
    m += fac.B[base + t] * w[top.parent_orig[base + t]];
  }
  return m;
}

// Standardized quadratic form sum_i (w_i - m_i)^2 / F_i of the NNGP joint
// density at unit marginal variance.
inline double pg_nngp_quadform(const PgNngpTopology& top,
                               const PgNngpFactors& fac,
                               const std::vector<double>& w) {
  double q = 0.0;
  for (int i = 0; i < top.n; i++) {
    const double r = w[top.orig[i]] - pg_nngp_parent_mean(top, fac, w, i);
    q += r * r / fac.F[i];
  }
  return q;
}

// One NNGP scale: the field, its marginal variance and its range, together
// with the topology and the per-sweep factor scratch.
struct PgNngpScale {
  std::vector<double> w;
  double sigma2 = 1.0;
  double phi = 1.0;
  PgNngpTopology top;
  PgNngpFactors fac, fac_prop;
};

// ============================================================================
// Scale priors
//
// Half-Cauchy(0, scale) on a standard deviation, sampled through Gelman
// (2006)'s auxiliary-variable scale mixture:
//   sigma^2 | a  ~ InvGamma((rank + 1)/2, ss/2 + 1/a)
//   a | sigma^2  ~ InvGamma(1, 1/scale^2 + 1/sigma^2)
// Taking the mixture is what makes the prior a half-Cauchy; holding `a` fixed
// at scale^2 collapses it to InvGamma(1/2, scale^2/2), which has no Cauchy
// tail. `rank` is the number of independent quadratic terms in `ss` -- the
// number of effects for an iid block, the GMRF rank for a structured arm.
// ============================================================================
struct PgScaleState {
  double sigma = 1.0;
  double aux = 1.0;
};

inline PgScaleState pg_scale_state_init(double scale) {
  PgScaleState st;
  st.sigma = 1.0;
  st.aux = 2.0 * scale * scale;  // prior mean of a ~ InvGamma(1/2, 1/scale^2)
  return st;
}

inline void pg_update_scale_halfcauchy(double ss, int rank, double scale,
                                       PgScaleState& st) {
  if (rank <= 0) return;
  const double shape = (rank + 1.0) / 2.0;
  const double rate = 0.5 * ss + 1.0 / st.aux;
  const double sigma_sq = 1.0 / R::rgamma(shape, 1.0 / rate);
  if (!std::isfinite(sigma_sq) || sigma_sq <= 0.0) return;
  st.sigma = std::sqrt(sigma_sq);
  const double rate_aux = 1.0 / (scale * scale) + 1.0 / sigma_sq;
  st.aux = 1.0 / R::rgamma(1.0, 1.0 / rate_aux);
}

inline void pg_update_scale_halfcauchy(const Rcpp::NumericVector& effects,
                                       double scale, PgScaleState& st) {
  double ss = 0.0;
  for (int j = 0; j < effects.size(); j++) ss += effects[j] * effects[j];
  pg_update_scale_halfcauchy(ss, static_cast<int>(effects.size()), scale, st);
}

// Penalized-complexity prior on a marginal standard deviation (Simpson et al.
// 2017): sigma ~ Exponential(lambda) with lambda = -log(alpha) / U, so
// P(sigma > U) = alpha. Returns the log density of sigma2 = sigma^2 including
// the change-of-variables Jacobian.
inline void pg_check_pc_prior(double U, double alpha, const char* name) {
  if (!(U > 0.0)) {
    Rcpp::stop("`prior_sigma_%s_U` must be > 0; got %g.", name, U);
  }
  if (!(alpha > 0.0 && alpha < 1.0)) {
    Rcpp::stop("`prior_sigma_%s_alpha` is the tail probability P(sigma > U) "
               "and must lie in (0, 1); got %g.", name, alpha);
  }
}

inline double pg_log_prior_sigma2_pc(double sigma2, double U, double alpha) {
  if (!(sigma2 > 0.0)) return R_NegInf;
  const double lambda = -std::log(alpha) / U;
  const double sigma = std::sqrt(sigma2);
  return std::log(lambda) - lambda * sigma - std::log(2.0 * sigma);
}

// Draw from N(mean, sd^2) truncated to (0, inf) (Robert 1995). Used for the
// Gaussian full conditional of a positive scale parameter.
inline double rtruncnorm_pos(double mean, double sd) {
  double alpha = -mean / sd;   // lower truncation on the standardized scale
  double z;
  if (alpha <= 0.0) {
    // Naive rejection from the standard normal.
    do { z = R::norm_rand(); } while (z < alpha);
  } else {
    // One-sided tail: shifted-exponential proposal with acceptance.
    double lam = 0.5 * (alpha + std::sqrt(alpha * alpha + 4.0));
    while (true) {
      double zp = alpha - std::log(R::unif_rand()) / lam;
      double acc = std::exp(-0.5 * (zp - lam) * (zp - lam));
      if (R::unif_rand() <= acc) { z = zp; break; }
    }
  }
  return mean + sd * z;
}

// Draw from N(M^{-1} b, M^{-1}) for a symmetric positive-definite M held
// row-major in `M_flat`, which is overwritten by its Cholesky factor. Solving
// L' z_star = z for z ~ N(0, I) gives Cov(z_star) = (L L')^{-1} = M^{-1}.
inline void pg_draw_gaussian_precision(double* M_flat, int n, const double* b,
                                       double* out, const char* what) {
  std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
  if (!tulpa_linalg::chol_factor_lower<tulpa_linalg::TriLayout::RowMajor>(
          M_flat, L.data(), n, n, /*nugget=*/0.0)) {
    Rcpp::stop("The %s full conditional has a %d x %d precision matrix that is "
               "not positive definite.", what, n, n);
  }
  std::vector<double> y(n), mean(n), z(n), z_star(n);
  tulpa_linalg::tri_solve_lower<tulpa_linalg::TriLayout::RowMajor>(
      L.data(), n, n, b, y.data());
  tulpa_linalg::tri_solve_lower_transpose<tulpa_linalg::TriLayout::RowMajor>(
      L.data(), n, n, y.data(), mean.data());
  for (int i = 0; i < n; i++) z[i] = R::rnorm(0.0, 1.0);
  tulpa_linalg::tri_solve_lower_transpose<tulpa_linalg::TriLayout::RowMajor>(
      L.data(), n, n, z.data(), z_star.data());
  for (int i = 0; i < n; i++) out[i] = mean[i] + z_star[i];
}

// ============================================================================
// Shared per-iteration core for PG Gibbs spatial samplers
// Steps 1-5 are identical across ICAR, BYM2, and RSR variants:
//   1. Compute linear predictor (eta = X*beta + re + spatial)
//   2. Sample omega ~ PG(n, eta)
//   3. Update beta
//   4. Recompute X_beta
//   5. Update RE + sigma_re
//
// The caller must set spatial_contrib BEFORE calling this function.
// After return, X_beta, re_contrib, and offset are updated.
// ============================================================================
inline void pg_gibbs_core_step(
    int N, int p,
    Rcpp::NumericVector& beta,
    Rcpp::NumericVector& re,
    PgScaleState& sigma_re,
    Rcpp::NumericVector& omega,
    Rcpp::NumericVector& eta,
    Rcpp::NumericVector& X_beta,
    Rcpp::NumericVector& re_contrib,
    const Rcpp::NumericVector& spatial_contrib,
    Rcpp::NumericVector& offset,
    const Rcpp::NumericVector& kappa,
    const Rcpp::IntegerVector& n_trials,
    const Rcpp::NumericMatrix& X,
    const Rcpp::IntegerVector& re_group,
    int n_re_groups,
    double prior_beta_sd,
    double prior_sigma_re_scale,
    int n_threads = 1
) {
    // Every per-observation region below goes through tulpa_parallel_for: at a
    // team of one it runs a plain loop rather than entering libgomp, and these
    // sit inside the Gibbs sweep, so the entry would be paid five times per
    // iteration. Each row writes its own slots, so the routes are bit-identical.
    const int team = tulpa_omp_team_size_req(n_threads, N);
    // 1. Compute linear predictor
    tulpa_parallel_for(team, N, [&](int i) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
            X_beta[i] += X(i, j) * beta[j];
        }
        re_contrib[i] = (n_re_groups > 0) ? re[re_group[i] - 1] : 0.0;
        eta[i] = X_beta[i] + re_contrib[i] + spatial_contrib[i];
    });

    // 2. Sample omega ~ PG(n, eta) — NOT parallelized (R's RNG not thread-safe)
    for (int i = 0; i < N; i++) {
        omega[i] = rpg_int(n_trials[i], eta[i]);
    }

    // 3. Update beta
    tulpa_parallel_for(team, N, [&](int i) {
        offset[i] = re_contrib[i] + spatial_contrib[i];
    });
    beta = update_beta(kappa, omega, X, offset, prior_beta_sd);

    // 4. Recompute X_beta
    tulpa_parallel_for(team, N, [&](int i) {
        X_beta[i] = 0.0;
        for (int j = 0; j < p; j++) {
            X_beta[i] += X(i, j) * beta[j];
        }
    });

    // 5. Update random effects
    if (n_re_groups > 0) {
        tulpa_parallel_for(team, N, [&](int i) {
            offset[i] = X_beta[i] + spatial_contrib[i];
        });
        re = update_re(kappa, omega, offset, re_group, n_re_groups, sigma_re.sigma);
        pg_update_scale_halfcauchy(re, prior_sigma_re_scale, sigma_re);

        tulpa_parallel_for(team, N, [&](int i) {
            re_contrib[i] = re[re_group[i] - 1];
        });
    }
}

// ============================================================================
// Common scaffolding for PG binomial Gibbs samplers
//
// Bundles state shared by every variant (no-spatial, ICAR, BYM2, RSR, GP,
// multiscale GP, temporal): working vectors, draw matrices for beta/re/sigma_re
// (+ optional eta), and the per-iteration save of those common fields. Per-
// variant storage (spatial, GP hypers, temporal components, etc.) stays in the
// caller.
// ============================================================================
struct PgGibbsCommon {
  int N, p, n_re_groups, n_save;
  bool store_eta;
  // Whether X's first column is an all-ones intercept. Absorbing a removed
  // field level into beta[0] shifts eta by level * X(i, 0), so it leaves eta
  // unchanged only under an intercept column.
  bool has_intercept;
  // Clamped team size for the per-obs regions: the caller's n_threads bounded
  // by OMP_THREAD_LIMIT / max threads / N. Passed as a num_threads(...)
  // clause instead of mutating the process-global OpenMP default.
  int n_threads_team = 1;

  // Current chain state
  Rcpp::NumericVector beta, re;
  PgScaleState sigma_re;

  // Per-observation working vectors (passed to pg_gibbs_core_step)
  Rcpp::NumericVector omega, kappa, eta, X_beta, re_contrib, offset;

  // Draws for the always-present fields
  Rcpp::NumericMatrix beta_draws, re_draws;
  Rcpp::NumericVector sigma_re_draws;
  Rcpp::NumericMatrix eta_draws;

  PgGibbsCommon(const Rcpp::IntegerVector& y,
                const Rcpp::IntegerVector& n_trials,
                const Rcpp::NumericMatrix& X,
                const Rcpp::IntegerVector& re_group,
                int n_re_groups_,
                int n_save_,
                double prior_sigma_re_scale,
                int n_threads,
                bool store_eta_)
    : N(y.size()), p(X.ncol()), n_re_groups(n_re_groups_), n_save(n_save_),
      store_eta(store_eta_),
      has_intercept(false),
      beta(X.ncol(), 0.0),
      re(n_re_groups_, 0.0),
      sigma_re(pg_scale_state_init(prior_sigma_re_scale)),
      omega(y.size(), 0.5),
      kappa(y.size()),
      eta(y.size()),
      X_beta(y.size()),
      re_contrib(y.size()),
      offset(y.size()),
      beta_draws(n_save_, X.ncol()),
      re_draws(n_save_, n_re_groups_),
      sigma_re_draws(n_save_),
      eta_draws(store_eta_ ? n_save_ : 0, store_eta_ ? y.size() : 0)
  {
    if (N < 1) Rcpp::stop("`y` is empty.");
    if (n_trials.size() != N) {
      Rcpp::stop("`n` has length %d but `y` has length %d.",
                 static_cast<int>(n_trials.size()), N);
    }
    if (X.nrow() != N) {
      Rcpp::stop("`X` has %d row(s) but `y` has length %d.",
                 static_cast<int>(X.nrow()), N);
    }
    if (n_re_groups < 0) {
      Rcpp::stop("`n_re_groups` must be >= 0; got %d.", n_re_groups);
    }
    if (n_re_groups > 0) pg_check_index(re_group, N, n_re_groups, "re_group");

    has_intercept = pg_has_intercept(X);

    n_threads_team = tulpa_omp_team_size_req(n_threads, N);
    for (int i = 0; i < N; i++) {
      const int ni = n_trials[i];
      if (Rcpp::IntegerVector::is_na(ni) || ni < 0) {
        Rcpp::stop("`n[%d]` is %d; must be a non-negative trial count.",
                   i + 1, ni);
      }
      kappa[i] = static_cast<double>(y[i]) - 0.5 * static_cast<double>(ni);
    }
  }

  void require_intercept(const char* what) const {
    pg_require_intercept(has_intercept, what);
  }

  // Absorb a level removed from a latent field into the intercept, keeping eta
  // unchanged, and refresh the cached X*beta the rest of the sweep reads.
  void absorb_level(double level) {
    if (level == 0.0) return;
    beta[0] += level;
    for (int i = 0; i < N; i++) X_beta[i] += level;
  }

  // Save common per-iteration draws. Caller is responsible for invoking this
  // (and per-variant save logic) only when (iter >= n_warmup && (iter - n_warmup) % thin == 0).
  void save(int save_idx) {
    if (save_idx < 0 || save_idx >= n_save) {
      Rcpp::stop("Draw index %d is outside the %d allocated row(s).",
                 save_idx, n_save);
    }
    for (int j = 0; j < p; j++) beta_draws(save_idx, j) = beta[j];
    for (int g = 0; g < n_re_groups; g++) re_draws(save_idx, g) = re[g];
    sigma_re_draws[save_idx] = sigma_re.sigma;
    if (store_eta) {
      for (int i = 0; i < N; i++) eta_draws(save_idx, i) = eta[i];
    }
  }
};

// ============================================================================
// NNGP scale update: the field by its full conditional under the NNGP joint
// density, then (sigma2, phi).
//
// The NNGP density factorizes over the ordering as
//   p(w) = prod_i N(w_i | B_i' w_{N(i)}, F_i),
// so the full conditional of w_i carries its own factor AND every factor j for
// which i is a parent: each contributes B_j[t]^2 / F_j to the precision and
// B_j[t] (w_j - sum_{t' != t} B_j[t'] w_{N(j), t'}) / F_j to the
// precision-weighted mean. `sum_omega` / `sum_resid` are the Polya-Gamma
// likelihood aggregates for this scale.
//
// sigma2 carries a PC prior (P(sigma > U) = alpha) and is drawn by a
// random-walk Metropolis step on log sigma2; phi carries a uniform prior on
// [lower, upper] and a log-random-walk Metropolis step on the NNGP density.
// Both read the same factor pass, since B is scale-free and F scales linearly.
// ============================================================================
inline void pg_nngp_scale_update(
    PgNngpScale& sc, int cov_type,
    const Rcpp::NumericMatrix& coords, const Rcpp::NumericMatrix& nn_dist,
    const std::vector<double>& sum_omega, const std::vector<double>& sum_resid,
    double prior_sigma_U, double prior_sigma_alpha,
    double prior_phi_lower, double prior_phi_upper
) {
  const PgNngpTopology& top = sc.top;
  const int n = top.n, nn = top.nn;
  pg_nngp_factors(sc.phi, cov_type, coords, nn_dist, top, sc.fac);

  for (int i = 0; i < n; i++) {
    const int obs_i = top.orig[i];

    // Own factor.
    double tau_post = 1.0 / (sc.sigma2 * sc.fac.F[i]);
    double mean_num = tau_post * pg_nngp_parent_mean(top, sc.fac, sc.w, i);

    // Factors in which this location is a parent.
    for (int e = top.child_ptr[i]; e < top.child_ptr[i + 1]; e++) {
      const int j = top.child_pos[e];
      const int t = top.child_slot[e];
      const size_t jb = static_cast<size_t>(j) * nn;
      const double b = sc.fac.B[jb + t];
      const double prec_j = 1.0 / (sc.sigma2 * sc.fac.F[j]);
      double resid = sc.w[top.orig[j]];
      for (int t2 = 0; t2 < top.cnt[j]; t2++) {
        if (t2 == t) continue;
        resid -= sc.fac.B[jb + t2] * sc.w[top.parent_orig[jb + t2]];
      }
      tau_post += b * b * prec_j;
      mean_num += b * resid * prec_j;
    }

    // Polya-Gamma data term.
    tau_post += sum_omega[obs_i];
    mean_num += sum_resid[obs_i];

    sc.w[obs_i] = R::rnorm(mean_num / tau_post, 1.0 / std::sqrt(tau_post));
  }

  // sigma2 | w, phi: the NNGP density contributes
  // -0.5 n log sigma2 - 0.5 Q0 / sigma2, the PC prior -lambda sqrt(sigma2),
  // and the log-scale proposal its Jacobian.
  const double Q0 = pg_nngp_quadform(top, sc.fac, sc.w);
  {
    const double u_curr = std::log(sc.sigma2);
    const double u_prop = R::rnorm(u_curr, 0.15);
    const double s2_prop = tulpa_linalg::safe_exp(u_prop);
    if (std::isfinite(s2_prop) && s2_prop > 0.0) {
      auto log_target = [&](double u, double s2) {
        return -0.5 * n * u - 0.5 * Q0 / s2
             + pg_log_prior_sigma2_pc(s2, prior_sigma_U, prior_sigma_alpha) + u;
      };
      const double lr = log_target(u_prop, s2_prop) - log_target(u_curr, sc.sigma2);
      if (std::isfinite(lr) && std::log(R::runif(0, 1)) < lr) sc.sigma2 = s2_prop;
    }
  }

  // phi | w, sigma2: log-random-walk Metropolis on the NNGP density.
  const double phi_prop = sc.phi * tulpa_linalg::safe_exp(R::rnorm(0, 0.1));
  if (std::isfinite(phi_prop) &&
      phi_prop >= prior_phi_lower && phi_prop <= prior_phi_upper) {
    pg_nngp_factors(phi_prop, cov_type, coords, nn_dist, top, sc.fac_prop);
    double ll_curr = 0.0, ll_prop = 0.0;
    for (int i = 0; i < n; i++) {
      const double rc = sc.w[top.orig[i]] - pg_nngp_parent_mean(top, sc.fac, sc.w, i);
      const double rp = sc.w[top.orig[i]] - pg_nngp_parent_mean(top, sc.fac_prop, sc.w, i);
      const double vc = sc.sigma2 * sc.fac.F[i];
      const double vp = sc.sigma2 * sc.fac_prop.F[i];
      ll_curr += -0.5 * std::log(vc) - 0.5 * rc * rc / vc;
      ll_prop += -0.5 * std::log(vp) - 0.5 * rp * rp / vp;
    }
    const double log_ratio = ll_prop - ll_curr + std::log(phi_prop / sc.phi);
    if (std::isfinite(log_ratio) && std::log(R::runif(0, 1)) < log_ratio) {
      sc.phi = phi_prop;
    }
  }
}

} // namespace tulpa

#endif // TULPA_PG_SHARED_H
