// linalg_fast.h
// Fast linear algebra operations for tulpa HMC
// Uses cache-friendly algorithms and SIMD-friendly patterns

#ifndef TULPA_LINALG_FAST_H
#define TULPA_LINALG_FAST_H

#include "omp_threads.h"
#include "tulpa/portable_math.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa_linalg {

// ============================================================================
// Coordinate geometry (single source for the neighbour-to-neighbour distance
// every NNGP kernel needs)
// ============================================================================

// Squared Euclidean distance between rows `i` and `j` of a coordinate matrix,
// over EVERY column the matrix carries -- so the coordinate DIMENSION is
// whatever the caller supplied, and a 1-column matrix is a 1-D domain rather
// than a 2-D one with a column missing.
//
// The three NNGP neighbour-covariance loops -- the Laplace kernel, the batched
// builder and the PG-Gibbs sweep -- each formed this by hand over columns 0 and
// 1. Column 1 was read unconditionally, so an `n x 1` coordinate matrix (a
// transect, a depth profile, a time axis, any 1-D domain) indexed `1 * n + i`,
// which is `n` doubles PAST the end of its own allocation: the neighbour
// covariance was built from whatever the heap held behind the matrix. Nothing
// crashed, because the read lands inside the R heap and returns a finite
// double; the fit simply stopped being a function of its data, and moved with
// the process's allocation history.
//
// Templated on the matrix type rather than taking `Rcpp::NumericMatrix` so this
// header stays free of the Rcpp dependency its other callers do not have; the
// requirement is `operator()(int, int)` and `ncol()`.
template <typename Mat>
inline double coords_dist2(const Mat& coords, int i, int j) {
    const int d = coords.ncol();
    double s = 0.0;
    for (int k = 0; k < d; ++k) {
        const double dk = coords(i, k) - coords(j, k);
        s += dk * dk;
    }
    return s;
}

// Euclidean distance between two rows of a coordinate matrix. The kernels want
// the distance itself; `coords_dist2` is exposed because a squared distance is
// what a covariance argument sometimes needs directly.
template <typename Mat>
inline double coords_dist(const Mat& coords, int i, int j) {
    return std::sqrt(coords_dist2(coords, i, j));
}

// The cross-matrix form: row `i` of `a` against row `j` of `b`. Prediction at
// new locations is the same metric the fit was built on, so it reads the
// coordinate dimension the same way rather than assuming 2 -- a predictor
// hard-coded to two columns against a dimension-general fit would disagree with
// it on any other width, which is a silent metric mismatch rather than an
// error. The two matrices must agree on their column count; they describe the
// same domain.
template <typename MatA, typename MatB>
inline double coords_dist2(const MatA& a, int i, const MatB& b, int j) {
    const int d = a.ncol();
    if (b.ncol() != d) {
        throw std::invalid_argument(
            "coordinate matrices describe the same domain and must have the "
            "same number of columns; got " + std::to_string(d) + " and " +
            std::to_string(b.ncol()) + ".");
    }
    double s = 0.0;
    for (int k = 0; k < d; ++k) {
        const double dk = a(i, k) - b(j, k);
        s += dk * dk;
    }
    return s;
}

template <typename MatA, typename MatB>
inline double coords_dist(const MatA& a, int i, const MatB& b, int j) {
    return std::sqrt(coords_dist2(a, i, b, j));
}

// The arity guard for the OTHER kind of consumer: a site that copies
// coordinates into a FLAT buffer at stride 2 (`GPData::coords` and its
// siblings) is 2-D by its own storage layout, so it cannot read a 1-D or 3-D
// matrix at all. Those sites also indexed column 1 unconditionally, which is
// the same out-of-bounds read `coords_dist()` above removes -- but making them
// dimension-general would mean changing a struct layout the samplers and the
// ABI share, so they REFUSE the input instead of misreading it.
//
// Throws `std::invalid_argument` rather than calling `Rcpp::stop` so this
// header keeps its Rcpp-free include list; Rcpp turns a `std::exception` into
// an R error at the `.Call` boundary, so the caller sees an ordinary condition.
template <typename Mat>
inline void require_coords_2col(const Mat& coords, const char* who) {
    const int d = coords.ncol();
    if (d != 2) {
        throw std::invalid_argument(
            std::string(who) + " requires a coordinate matrix with exactly 2 "
            "columns (x, y); got " + std::to_string(d) +
            ". This path stores coordinates at a fixed 2-D stride. The "
            "nested-Laplace NNGP/GP kernels accept any number of coordinate "
            "columns.");
    }
}

// ============================================================================
// Small dense Cholesky (single source for the per-location NNGP /
// CAR / GP kernels; all hand-rolled copies route through these)
// ============================================================================

// Smallest pivot a Cholesky factorization accepts. At or below it the
// factorization reports failure rather than flooring the pivot: flooring
// substitutes a different matrix for the one the caller handed in, and the
// caller cannot tell that it happened.
constexpr double kCholMinPivot = 1e-12;

// Conditioning constants for an NNGP neighbour covariance.
//
// `kNngpNugget` is a diagonal NUGGET: it is added to every diagonal entry
// BEFORE the factorization, so it changes the matrix being factorized and is
// part of the density the kernel evaluates. `kNngpVarFloor` is a bound on the
// conditional variance that comes out of that factorization. The two are
// different objects and are named separately.
//
// Every path that conditions a GP field on its neighbours reads these -- the
// sampler density and its autodiff twin (hmc_gp_*, via tulpa_gp::kGpJitter /
// kGpVarFloor), the Laplace kernel (gpu_nngp_laplace.h, laplace_core.cpp) and
// the Polya-Gamma sweep (pg_shared.h) -- so the same field is conditioned
// identically whichever path fits it. The SVC kernel runs deliberately looser
// values of its own (tulpa_svc::kSvcJitter / kSvcVarFloor) and passes them
// explicitly.
constexpr double kNngpNugget = 1e-8;
constexpr double kNngpVarFloor = 1e-10;

// Storage convention for a dense triangular factor. Both appear in this
// package: the per-neighbourhood NNGP / CAR / GP kernels build their factors
// row-major, the adapted HMC mass matrix comes out of Eigen column-major.
//
// The two are related by transposition -- a column-major lower-triangular
// factor is the same bytes as a row-major upper-triangular one -- so a factor
// read under the wrong convention does not crash, does not produce NaN, and
// trips no dimension check. It solves against the transpose and returns a
// plausible vector. Consumed that way, a cuSOLVER factor corrupted every
// NNGP fit with 51+ locations while staying finite and ordinary-looking.
//
// The layout is therefore a required template argument on every routine below
// that indexes off the diagonal: a call site has to state which convention it
// is asserting, and cannot inherit one by argument order.
enum class TriLayout { RowMajor, ColMajor };

// Offset of element (i, j) in an ld-strided n x n buffer.
template <TriLayout LO>
inline int tri_index(int i, int j, int ld) {
  if constexpr (LO == TriLayout::RowMajor) {
    return i * ld + j;
  } else {
    return j * ld + i;
  }
}

// Lower-Cholesky factorization of the leading n x n block of an ld-strided
// dense SPD matrix, with `nugget` added to every diagonal entry first. That is
// the same object `tulpa_nngp::chol_decomp` applies, so the double core here
// and the templated core the autodiff copies run factorize the same matrix;
// `nugget = 0.0` factorizes A itself.
//
// A nugget is NOT a pivot floor. Flooring a pivot at the nugget leaves a
// well-conditioned input untouched and silently replaces an ill-conditioned
// one, so two paths that should evaluate the same density diverge on exactly
// the inputs where it matters, with no way to tell from the result. Returns
// false on a pivot at or below kCholMinPivot, leaving L unusable, so a caller
// hears about a non-PD matrix instead of receiving a plausible factor of a
// different one.
//
// Supports in-place use (L == A); the opposite triangle of L is left
// untouched, so callers wanting zeros there must pass a zero-initialized L.
template <TriLayout LO>
inline bool chol_factor_lower(const double* A, double* L, int n, int ld,
                              double nugget) {
  for (int j = 0; j < n; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = A[tri_index<LO>(j, k, ld)];
      if (j == k) sum += nugget;
      for (int m = 0; m < k; m++) {
        sum -= L[tri_index<LO>(j, m, ld)] * L[tri_index<LO>(k, m, ld)];
      }
      if (j == k) {
        if (sum <= kCholMinPivot) return false;  // Not positive definite
        L[tri_index<LO>(j, j, ld)] = std::sqrt(sum);
      } else {
        L[tri_index<LO>(j, k, ld)] = sum / L[tri_index<LO>(k, k, ld)];
      }
    }
  }
  return true;
}

// Forward substitution: solve L y = b for lower-triangular L.
template <TriLayout LO>
inline void tri_solve_lower(const double* L, int n, int ld,
                            const double* b, double* y) {
  for (int i = 0; i < n; i++) {
    double sum = b[i];
    for (int j = 0; j < i; j++) {
      sum -= L[tri_index<LO>(i, j, ld)] * y[j];
    }
    y[i] = sum / L[tri_index<LO>(i, i, ld)];
  }
}

// Back substitution: solve L' x = y for lower-triangular L.
template <TriLayout LO>
inline void tri_solve_lower_transpose(const double* L, int n, int ld,
                                      const double* y, double* x) {
  for (int i = n - 1; i >= 0; i--) {
    double sum = y[i];
    for (int j = i + 1; j < n; j++) {
      sum -= L[tri_index<LO>(j, i, ld)] * x[j];
    }
    x[i] = sum / L[tri_index<LO>(i, i, ld)];
  }
}

// log|A| = 2 * sum(log(L_ii)) from a Cholesky factor. The diagonal sits at
// i * ld + i under both conventions, so this one takes no layout.
inline double chol_log_det(const double* L, int n, int ld) {
  double log_det = 0.0;
  for (int i = 0; i < n; i++) {
    log_det += std::log(L[i * ld + i]);
  }
  return 2.0 * log_det;
}

// NNGP kriging moments from an already-factored neighbor covariance:
// cond_mean = c' C^{-1} w_nb, cond_var = max(var_floor, sigma2 - c' C^{-1} c).
// `w_nb` holds the neighbor values gathered in the same order as `c_vec`.
// If `alpha_out` is non-null the kriging weights C^{-1} c are written there
// (length n).
template <TriLayout LO>
inline void nngp_moments_from_chol(const double* L, int n, int ld,
                                   const double* c_vec, const double* w_nb,
                                   double sigma2, double var_floor,
                                   double& cond_mean, double& cond_var,
                                   double* alpha_out = nullptr) {
  std::vector<double> y(n), alpha_local;
  double* alpha = alpha_out;
  if (!alpha) {
    alpha_local.resize(n);
    alpha = alpha_local.data();
  }
  tri_solve_lower<LO>(L, n, ld, c_vec, y.data());
  tri_solve_lower_transpose<LO>(L, n, ld, y.data(), alpha);

  double cm = 0.0;
  double c_Cinv_c = 0.0;
  for (int j = 0; j < n; j++) {
    cm += alpha[j] * w_nb[j];
    c_Cinv_c += c_vec[j] * alpha[j];
  }
  cond_mean = cm;
  // sigma2 - c'C^-1 c is a Schur complement of a PSD matrix, so it is >= 0
  // whenever L really is a factor of C. The floor keeps 1/cond_var finite when
  // it is not -- and a bound floor is a strong signal, because 1/var_floor =
  // 1e10 lands directly on the NNGP precision's diagonal and asserts the node
  // is known to 1e-5 sd given its neighbours. It reads as near-determinism
  // while usually meaning the factor is wrong: a column-major cuSOLVER factor
  // consumed row-major floored 47 of 150
  // nodes on a fixture whose true minimum conditional variance is 2.5e-02.
  cond_var = std::max(var_floor, sigma2 - c_Cinv_c);
}

// Full NNGP conditional-moments core: factorize the neighbor covariance
// C (n x n) with `nugget` on its diagonal, then compute the kriging moments
// against c_vec / w_nb. C is symmetric, so the factor's layout is an internal
// detail here.
//
// Returns false when C + nugget*I is not positive definite. The location is
// then reported as conditioned on NOTHING -- cond_mean 0, cond_var sigma2, the
// same moments a location with no neighbours gets -- rather than kriged
// against an unusable factor.
inline bool nngp_conditional_moments(const double* C, const double* c_vec,
                                     const double* w_nb, int n, double sigma2,
                                     double nugget, double var_floor,
                                     double& cond_mean, double& cond_var) {
  std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
  if (!chol_factor_lower<TriLayout::RowMajor>(C, L.data(), n, n, nugget)) {
    cond_mean = 0.0;
    cond_var = sigma2;
    return false;
  }
  nngp_moments_from_chol<TriLayout::RowMajor>(L.data(), n, n, c_vec, w_nb,
                                              sigma2, var_floor, cond_mean,
                                              cond_var);
  return true;
}

// ============================================================================
// Vector operations (SIMD-friendly)
// ============================================================================

// Dot product with loop unrolling
inline double dot_product(const double* x, const double* y, int n) {
  double sum = 0.0;
  int i = 0;

  // Process 4 elements at a time (SIMD-friendly)
  for (; i + 3 < n; i += 4) {
    sum += x[i] * y[i] + x[i+1] * y[i+1] +
           x[i+2] * y[i+2] + x[i+3] * y[i+3];
  }

  // Handle remaining elements
  for (; i < n; i++) {
    sum += x[i] * y[i];
  }

  return sum;
}

// Dot product with stride
inline double dot_product_strided(const double* x, int stride_x,
                                   const double* y, int stride_y, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += x[i * stride_x] * y[i * stride_y];
  }
  return sum;
}

// Vector sum
inline double vector_sum(const double* x, int n) {
  double sum = 0.0;
  int i = 0;

  for (; i + 3 < n; i += 4) {
    sum += x[i] + x[i+1] + x[i+2] + x[i+3];
  }
  for (; i < n; i++) {
    sum += x[i];
  }

  return sum;
}

// Vector L2 norm squared
inline double norm_squared(const double* x, int n) {
  return dot_product(x, x, n);
}

// axpy: y = a*x + y
inline void axpy(double a, const double* x, double* y, int n) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    y[i] += a * x[i];
    y[i+1] += a * x[i+1];
    y[i+2] += a * x[i+2];
    y[i+3] += a * x[i+3];
  }
  for (; i < n; i++) {
    y[i] += a * x[i];
  }
}

// Scale vector: x = a*x
inline void scale(double a, double* x, int n) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    x[i] *= a;
    x[i+1] *= a;
    x[i+2] *= a;
    x[i+3] *= a;
  }
  for (; i < n; i++) {
    x[i] *= a;
  }
}

// Weighted axpy: y[i] += a * w[i] * x[i]  (mass-scaled momentum update)
inline void axpy_weighted(double a, const double* w, const double* x, double* y, int n) {
  int i = 0;
  for (; i + 3 < n; i += 4) {
    y[i]   += a * w[i]   * x[i];
    y[i+1] += a * w[i+1] * x[i+1];
    y[i+2] += a * w[i+2] * x[i+2];
    y[i+3] += a * w[i+3] * x[i+3];
  }
  for (; i < n; i++) {
    y[i] += a * w[i] * x[i];
  }
}

// Weighted norm squared: sum(x[i]^2 * w[i])  (for kinetic energy)
inline double weighted_norm_squared(const double* x, const double* w, int n) {
  double sum = 0.0;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    sum += x[i]   * x[i]   * w[i]
         + x[i+1] * x[i+1] * w[i+1]
         + x[i+2] * x[i+2] * w[i+2]
         + x[i+3] * x[i+3] * w[i+3];
  }
  for (; i < n; i++) {
    sum += x[i] * x[i] * w[i];
  }
  return sum;
}

// Copy n doubles from src to dst (thin wrapper over memcpy for clarity)
inline void vec_copy(const double* src, double* dst, int n) {
  std::memcpy(dst, src, n * sizeof(double));
}

// ============================================================================
// Matrix-vector operations (row-major storage)
// ============================================================================

// Matrix-vector multiply: y = X * beta
// X is N x p stored row-major (X_flat[i*p + j] = X[i,j])
inline void matvec(const double* X_flat, const double* beta,
                   double* y, int N, int p) {
  // Reached per leapfrog step through precompute_generic_fixed_eta, so the
  // region entry is on the sampler's hot path: at a team of one
  // tulpa_parallel_for takes a plain loop instead of entering libgomp. Rows
  // write disjoint y slots, so the routes are bit-identical.
  const int team = tulpa_omp_team_size(N);
  tulpa_parallel_for(team, N, [&](int i) {
    y[i] = dot_product(&X_flat[i * p], beta, p);
  });
}

// Matrix-vector multiply with accumulation: y += X * beta
inline void matvec_add(const double* X_flat, const double* beta,
                       double* y, int N, int p) {

  // Same one-thread route as matvec above: rows write disjoint y slots, so
  // the plain loop and the region are bit-identical, and a team of one skips
  // libgomp instead of entering it to run the body serially anyway.
  tulpa_parallel_for(tulpa_omp_team_size(N), N, [&](int i) {
    y[i] += dot_product(&X_flat[i * p], beta, p);
  });
}

// Transposed matrix-vector multiply: y = X' * x
// Returns p-dimensional vector
inline void matvec_transpose(const double* X_flat, const double* x,
                             double* y, int N, int p) {

  // Initialize output to zero
  std::fill(y, y + p, 0.0);

  // Sequential accumulation (thread-safe without reduction)
  for (int i = 0; i < N; i++) {
    double xi = x[i];
    const double* row = &X_flat[i * p];
    for (int j = 0; j < p; j++) {
      y[j] += row[j] * xi;
    }
  }
}

// ============================================================================
// Batch linear predictor computation
// ============================================================================

// Compute linear predictors for all observations
// eta_num[i] = X_num[i,:] * beta_num
// eta_denom[i] = X_denom[i,:] * beta_denom
inline void compute_linear_predictors(
    const double* X_num_flat, const double* beta_num, int p_num,
    const double* X_denom_flat, const double* beta_denom, int p_denom,
    double* eta_num, double* eta_denom, int N, int n_threads = 1) {

  tulpa_parallel_for(tulpa_omp_team_size_req(n_threads, N), N, [&](int i) {
    eta_num[i] = dot_product(&X_num_flat[i * p_num], beta_num, p_num);
    eta_denom[i] = dot_product(&X_denom_flat[i * p_denom], beta_denom, p_denom);
  });
}

// ============================================================================
// Sparse operations for adjacency matrices
// ============================================================================

// Sparse matrix-vector multiply (CSR format)
// For ICAR: y = A * x where A is adjacency
inline void sparse_matvec_csr(
    const int* row_ptr, const int* col_idx, const double* values,
    const double* x, double* y, int n_rows) {

  tulpa_parallel_for(tulpa_omp_team_size(n_rows), n_rows, [&](int i) {
    double sum = 0.0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      sum += values[k] * x[col_idx[k]];
    }
    y[i] = sum;
  });
}

// Sparse quadratic form: x' * L * x for Laplacian L
// L = D - A where D is diagonal of degrees, A is adjacency
// Uses: x'Lx = sum_edges (x_i - x_j)^2 for unweighted graph
inline double sparse_laplacian_quadform(
    const int* row_ptr, const int* col_idx,
    const double* x, int n_rows) {

  double quad_form = 0.0;

  // Sum over all edges (count each once)
  for (int i = 0; i < n_rows; i++) {
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      int j = col_idx[k];
      if (j > i) {  // Count each edge once
        double diff = x[i] - x[j];
        quad_form += diff * diff;
      }
    }
  }

  return quad_form;
}

// ============================================================================
// Memory-efficient operations
// ============================================================================

// Block processing for large datasets
// Processes data in chunks to improve cache efficiency
template<typename Func>
inline double block_reduce(int N, int block_size, Func f) {
  double total = 0.0;
  int n_blocks = (N + block_size - 1) / block_size;

  for (int b = 0; b < n_blocks; b++) {
    int start = b * block_size;
    int end = std::min(start + block_size, N);
    total += f(start, end);
  }

  return total;
}

// Parallel block reduce. The block totals are summed through
// tulpa_parallel_sum, which cuts the block range into contiguous chunks by
// index arithmetic and adds the chunk totals in chunk order. A raw
// `reduction(+:)` clause here would leave the combination order to the runtime,
// which is the non-reproducibility tulpa_parallel_sum exists to remove -- and
// this was the last one in the tree (gcol33/tulpa#618).
template<typename Func>
inline double parallel_block_reduce(int N, int block_size, int n_threads, Func f) {
  const int n_blocks = (N + block_size - 1) / block_size;
  const int team = tulpa_omp_team_size_req(n_threads, n_blocks);

  return tulpa_parallel_sum(team, n_blocks, [&](int b) {
    const int start = b * block_size;
    const int end = std::min(start + block_size, N);
    return f(start, end);
  });
}

// ============================================================================
// Numerical utilities
// ============================================================================

// Safe exponential that prevents overflow/underflow. The bounds are
// tulpa::math::EXP_ARG_MAX / EXP_ARG_MIN in tulpa/portable_math.h, the one
// place the value path and the three autodiff paths take them from, so the
// double kernels here clamp at exactly the same argument the templated log
// posterior does.
inline double safe_exp(double x) {
  return std::exp(tulpa::math::clamp_exp_arg(x));
}

// Clamp value to range [lo, hi]
inline double clamp(double x, double lo, double hi) {
  return std::max(lo, std::min(x, hi));
}

// Check if value is finite (not NaN or Inf)
inline bool is_finite(double x) {
  return std::isfinite(x);
}

// Safe inverse logit (logistic function) that avoids overflow
// Returns 1 / (1 + exp(-x))
inline double safe_inv_logit(double x) {
  if (x > 0) {
    double exp_neg_x = safe_exp(-x);
    return 1.0 / (1.0 + exp_neg_x);
  } else {
    double exp_x = safe_exp(x);
    return exp_x / (1.0 + exp_x);
  }
}

// Log-sum-exp for numerical stability
inline double log_sum_exp(double a, double b) {
  double max_val = std::max(a, b);
  if (!std::isfinite(max_val)) return max_val;
  return max_val + std::log(std::exp(a - max_val) + std::exp(b - max_val));
}

// Vectorized log-sum-exp
inline double log_sum_exp_vec(const double* x, int n) {
  if (n == 0) return -std::numeric_limits<double>::infinity();

  double max_val = *std::max_element(x, x + n);
  if (!std::isfinite(max_val)) return max_val;

  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    sum += std::exp(x[i] - max_val);
  }

  return max_val + std::log(sum);
}

// Softmax (in-place)
inline void softmax_inplace(double* x, int n) {
  double max_val = *std::max_element(x, x + n);
  double sum = 0.0;

  for (int i = 0; i < n; i++) {
    x[i] = std::exp(x[i] - max_val);
    sum += x[i];
  }

  for (int i = 0; i < n; i++) {
    x[i] /= sum;
  }
}

// ============================================================================
// Conjugate Gradient (CG) for Symmetric Positive Definite Systems
// The CORRECT solver for GP covariance matrices K + σ²I
// Reference: Hestenes & Stiefel (1952), "Methods of Conjugate Gradients"
// ============================================================================

// Result struct for iterative solvers
struct IterativeSolverResult {
  std::vector<double> x;       // Solution vector
  int iterations;              // Number of iterations used
  double residual_norm;        // Final residual norm ||b - Ax||
  bool converged;              // Whether convergence was achieved
};

// Standard Conjugate Gradient for SPD systems
// A_func: function that computes A*v for given v (matrix-free interface)
// b: right-hand side vector
// x0: initial guess (if empty, uses zero)
// tol: convergence tolerance for relative residual ||r||/||b||
// max_iter: maximum number of iterations
//
// For GP regression: solves (K + σ²I) α = y
// Complexity: O(N × k × iter) where k is kernel evaluation cost
// Typically converges in O(√κ) iterations where κ is condition number
template<typename MatVecFunc>
IterativeSolverResult cg_solve(
    MatVecFunc A_func,
    const std::vector<double>& b,
    const std::vector<double>& x0 = {},
    double tol = 1e-8,
    int max_iter = 1000
) {
  const int n = static_cast<int>(b.size());

  IterativeSolverResult result;
  result.iterations = 0;
  result.converged = false;
  result.x.resize(n);

  // Initialize solution
  if (x0.empty()) {
    std::fill(result.x.begin(), result.x.end(), 0.0);
  } else {
    std::copy(x0.begin(), x0.end(), result.x.begin());
  }

  // Compute initial residual r = b - A*x
  std::vector<double> r(n), p(n), Ap(n);
  A_func(result.x.data(), Ap.data());

  double b_norm = std::sqrt(dot_product(b.data(), b.data(), n));
  if (b_norm < 1e-14) b_norm = 1.0;  // Avoid division by zero

  for (int i = 0; i < n; i++) {
    r[i] = b[i] - Ap[i];
    p[i] = r[i];  // Initial search direction = residual
  }

  double r_dot_r = dot_product(r.data(), r.data(), n);
  double r_norm = std::sqrt(r_dot_r);

  // Check if already converged
  if (r_norm / b_norm < tol) {
    result.residual_norm = r_norm;
    result.converged = true;
    return result;
  }

  // CG iteration
  for (int iter = 0; iter < max_iter; iter++) {
    result.iterations = iter + 1;

    // Compute A*p
    A_func(p.data(), Ap.data());

    // α = (r·r) / (p·Ap)
    double p_dot_Ap = dot_product(p.data(), Ap.data(), n);

    if (std::abs(p_dot_Ap) < 1e-30) {
      // Breakdown: p is in null space of A (shouldn't happen for SPD)
      break;
    }

    double alpha = r_dot_r / p_dot_Ap;

    // Update solution: x = x + α*p
    // Update residual: r = r - α*Ap
    for (int i = 0; i < n; i++) {
      result.x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    // Compute new residual norm
    double r_dot_r_new = dot_product(r.data(), r.data(), n);
    r_norm = std::sqrt(r_dot_r_new);

    // Check convergence
    if (r_norm / b_norm < tol) {
      result.residual_norm = r_norm;
      result.converged = true;
      return result;
    }

    // β = (r_new·r_new) / (r·r)
    double beta = r_dot_r_new / r_dot_r;
    r_dot_r = r_dot_r_new;

    // Update search direction: p = r + β*p
    for (int i = 0; i < n; i++) {
      p[i] = r[i] + beta * p[i];
    }
  }

  // Did not converge within max_iter
  result.residual_norm = r_norm;
  return result;
}

// ============================================================================
// Preconditioned Conjugate Gradient (PCG)
// M_solve: applies preconditioner M^{-1} to a vector
// For GP: use incomplete Cholesky, diagonal, or HSGP-based preconditioner
// Converges in O(√(κ(M^{-1}A))) iterations instead of O(√κ(A))
// ============================================================================
template<typename MatVecFunc, typename PrecondFunc>
IterativeSolverResult pcg_solve(
    MatVecFunc A_func,
    PrecondFunc M_solve,
    const std::vector<double>& b,
    const std::vector<double>& x0 = {},
    double tol = 1e-8,
    int max_iter = 1000
) {
  const int n = static_cast<int>(b.size());

  IterativeSolverResult result;
  result.iterations = 0;
  result.converged = false;
  result.x.resize(n);

  // Initialize solution
  if (x0.empty()) {
    std::fill(result.x.begin(), result.x.end(), 0.0);
  } else {
    std::copy(x0.begin(), x0.end(), result.x.begin());
  }

  // Compute initial residual r = b - A*x
  std::vector<double> r(n), z(n), p(n), Ap(n);
  A_func(result.x.data(), Ap.data());

  double b_norm = std::sqrt(dot_product(b.data(), b.data(), n));
  if (b_norm < 1e-14) b_norm = 1.0;

  for (int i = 0; i < n; i++) {
    r[i] = b[i] - Ap[i];
  }

  double r_norm = std::sqrt(dot_product(r.data(), r.data(), n));

  // Check if already converged
  if (r_norm / b_norm < tol) {
    result.residual_norm = r_norm;
    result.converged = true;
    return result;
  }

  // Apply preconditioner: z = M^{-1} * r
  M_solve(r.data(), z.data());

  // Initial search direction
  std::copy(z.begin(), z.end(), p.begin());

  double r_dot_z = dot_product(r.data(), z.data(), n);

  // PCG iteration
  for (int iter = 0; iter < max_iter; iter++) {
    result.iterations = iter + 1;

    // Compute A*p
    A_func(p.data(), Ap.data());

    // α = (r·z) / (p·Ap)
    double p_dot_Ap = dot_product(p.data(), Ap.data(), n);

    if (std::abs(p_dot_Ap) < 1e-30) {
      break;
    }

    double alpha = r_dot_z / p_dot_Ap;

    // Update solution and residual
    for (int i = 0; i < n; i++) {
      result.x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    r_norm = std::sqrt(dot_product(r.data(), r.data(), n));

    // Check convergence
    if (r_norm / b_norm < tol) {
      result.residual_norm = r_norm;
      result.converged = true;
      return result;
    }

    // Apply preconditioner: z = M^{-1} * r
    M_solve(r.data(), z.data());

    double r_dot_z_new = dot_product(r.data(), z.data(), n);

    // β = (r_new·z_new) / (r·z)
    double beta = r_dot_z_new / r_dot_z;
    r_dot_z = r_dot_z_new;

    // Update search direction
    for (int i = 0; i < n; i++) {
      p[i] = z[i] + beta * p[i];
    }
  }

  result.residual_norm = r_norm;
  return result;
}

// ============================================================================
// GP-specific helpers
// ============================================================================

// Create a kernel-vector product function for squared exponential kernel
// coords: N x 2 matrix (row-major) of coordinates
// sigma_sq: variance parameter
// lengthscale: lengthscale parameter
// Returns lambda that computes K(theta) * v for any vector v
inline auto make_se_kernel_matvec(
    const double* coords, int N,
    double sigma_sq, double lengthscale
) {
  return [=](const double* v, double* result) {
    const double inv_l2 = 1.0 / (lengthscale * lengthscale);

    tulpa_parallel_for(tulpa_omp_team_size(N), N, [&](int i) {
      double sum = 0.0;
      double xi = coords[2*i];
      double yi = coords[2*i + 1];

      for (int j = 0; j < N; j++) {
        double dx = xi - coords[2*j];
        double dy = yi - coords[2*j + 1];
        double dist_sq = dx*dx + dy*dy;
        double kij = sigma_sq * std::exp(-0.5 * dist_sq * inv_l2);
        if (i == j) kij += 1e-6;  // Jitter for numerical stability
        sum += kij * v[j];
      }
      result[i] = sum;
    });
  };
}

// Diagonal preconditioner (simple but effective for GP)
// Uses the diagonal of K as preconditioner
inline auto make_diagonal_precond(
    const double* diag, int N
) {
  return [=](const double* v, double* result) {
    for (int i = 0; i < N; i++) {
      result[i] = v[i] / diag[i];
    }
  };
}

// ============================================================================
// Dense matrix operations (column-major storage, for mass matrix support)
// ============================================================================

// Row i of a symmetric column-major n×n matrix, dotted with x.
static inline double sym_row_dot(const double* A, const double* x, int n, int i) {
  double sum = 0.0;
  for (int j = 0; j < n; j++) {
    sum += A[j * n + i] * x[j];  // column-major: A[i,j] = A[j*n + i]
  }
  return sum;
}

// Symmetric matrix-vector product: y = A * x
// A is n×n symmetric, stored column-major (only uses lower triangle logic
// but reads full matrix for simplicity since it's symmetric)
inline void symmatvec(const double* A, const double* x, double* y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] = sym_row_dot(A, x, n, i);
  }
}

// Quadratic form: x^T * A * x (A symmetric n×n, column-major)
inline double quadratic_form(const double* x, const double* A, int n) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    result += x[i] * sym_row_dot(A, x, n, i);
  }
  return result;
}

// Fused scale + matvec + add: y += alpha * A * x (A symmetric n×n, column-major)
inline void axpy_matvec(double alpha, const double* A, const double* x,
                        double* y, int n) {
  for (int i = 0; i < n; i++) {
    y[i] += alpha * sym_row_dot(A, x, n, i);
  }
}

}  // namespace tulpa_linalg

#endif  // TULPA_LINALG_FAST_H
