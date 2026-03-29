// linalg_fast.h
// Fast linear algebra operations for tulpa HMC
// Uses cache-friendly algorithms and SIMD-friendly patterns

#ifndef TULPA_LINALG_FAST_H
#define TULPA_LINALG_FAST_H

#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tulpa_linalg {

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

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    y[i] = dot_product(&X_flat[i * p], beta, p);
  }
}

// Matrix-vector multiply with accumulation: y += X * beta
inline void matvec_add(const double* X_flat, const double* beta,
                       double* y, int N, int p) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < N; i++) {
    y[i] += dot_product(&X_flat[i * p], beta, p);
  }
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

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
  #endif
  for (int i = 0; i < N; i++) {
    eta_num[i] = dot_product(&X_num_flat[i * p_num], beta_num, p_num);
    eta_denom[i] = dot_product(&X_denom_flat[i * p_denom], beta_denom, p_denom);
  }
}

// ============================================================================
// Sparse operations for adjacency matrices
// ============================================================================

// Sparse matrix-vector multiply (CSR format)
// For ICAR: y = A * x where A is adjacency
inline void sparse_matvec_csr(
    const int* row_ptr, const int* col_idx, const double* values,
    const double* x, double* y, int n_rows) {

  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int i = 0; i < n_rows; i++) {
    double sum = 0.0;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      sum += values[k] * x[col_idx[k]];
    }
    y[i] = sum;
  }
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

// Parallel block reduce
template<typename Func>
inline double parallel_block_reduce(int N, int block_size, int n_threads, Func f) {
  double total = 0.0;
  int n_blocks = (N + block_size - 1) / block_size;

  #ifdef _OPENMP
  #pragma omp parallel for reduction(+:total) num_threads(n_threads)
  #endif
  for (int b = 0; b < n_blocks; b++) {
    int start = b * block_size;
    int end = std::min(start + block_size, N);
    total += f(start, end);
  }

  return total;
}

// ============================================================================
// Numerical utilities
// ============================================================================

// Maximum safe argument for exp() to avoid overflow
// exp(709) ≈ 8.2e307, exp(710) = inf
constexpr double EXP_MAX_ARG = 700.0;

// Minimum safe argument for exp() to avoid underflow to exact zero
// exp(-745) ≈ 5e-324 (smallest subnormal), exp(-746) = 0
constexpr double EXP_MIN_ARG = -700.0;

// Safe exponential that prevents overflow/underflow
// Returns exp(x) clamped to finite range
inline double safe_exp(double x) {
  if (x > EXP_MAX_ARG) return std::exp(EXP_MAX_ARG);  // ~1e304
  if (x < EXP_MIN_ARG) return std::exp(EXP_MIN_ARG);  // ~1e-304
  return std::exp(x);
}

// Safe log that handles zero and negative inputs
// Returns log(x) or -infinity for x <= 0
inline double safe_log(double x) {
  if (x <= 0.0) return -std::numeric_limits<double>::infinity();
  return std::log(x);
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

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
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
    }
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

// Symmetric matrix-vector product: y = A * x
// A is n×n symmetric, stored column-major (only uses lower triangle logic
// but reads full matrix for simplicity since it's symmetric)
inline void symmatvec(const double* A, const double* x, double* y, int n) {
  for (int i = 0; i < n; i++) {
    double sum = 0.0;
    for (int j = 0; j < n; j++) {
      sum += A[j * n + i] * x[j];  // column-major: A[i,j] = A[j*n + i]
    }
    y[i] = sum;
  }
}

// Quadratic form: x^T * A * x (A symmetric n×n, column-major)
inline double quadratic_form(const double* x, const double* A, int n) {
  double result = 0.0;
  for (int i = 0; i < n; i++) {
    double Ax_i = 0.0;
    for (int j = 0; j < n; j++) {
      Ax_i += A[j * n + i] * x[j];
    }
    result += x[i] * Ax_i;
  }
  return result;
}

// Forward substitution: solve L*y = b (L lower triangular, column-major)
inline void tri_solve_lower(const double* L, const double* b, double* y, int n) {
  for (int i = 0; i < n; i++) {
    double sum = b[i];
    for (int j = 0; j < i; j++) {
      sum -= L[j * n + i] * y[j];  // L[i,j] = L[j*n + i]
    }
    y[i] = sum / L[i * n + i];  // L[i,i] = L[i*n + i]
  }
}

// Back substitution: solve L^T*y = b (L lower triangular, column-major)
inline void tri_solve_upper_transpose(const double* L, const double* b, double* y, int n) {
  for (int i = n - 1; i >= 0; i--) {
    double sum = b[i];
    for (int j = i + 1; j < n; j++) {
      sum -= L[i * n + j] * y[j];  // L^T[i,j] = L[j,i] = L[i*n + j]
    }
    y[i] = sum / L[i * n + i];  // L^T[i,i] = L[i,i] = L[i*n + i]
  }
}

// Fused scale + matvec + add: y += alpha * A * x (A symmetric n×n, column-major)
inline void axpy_matvec(double alpha, const double* A, const double* x,
                        double* y, int n) {
  for (int i = 0; i < n; i++) {
    double Ax_i = 0.0;
    for (int j = 0; j < n; j++) {
      Ax_i += A[j * n + i] * x[j];
    }
    y[i] += alpha * Ax_i;
  }
}

}  // namespace tulpa_linalg

#endif  // QUOTR_LINALG_FAST_H
