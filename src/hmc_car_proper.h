// hmc_car_proper.h
// Proper CAR (Conditional Autoregressive) spatial effects
// Unlike ICAR (rho=1 fixed), proper CAR estimates rho from data.
//
// Wired into hmc_sampler.cpp: SpatialType::CAR_PROPER allocates a logit-rho
// parameter alongside log_tau and phi_spatial. The log-prior term and
// gradient are computed via the helpers below.
//
// Performance note: car_log_det() runs a dense O(n^3) Cholesky each call
// because rho (and therefore Q) changes every gradient evaluation. For
// large adjacency graphs swap in tulpa::SparseCholeskySolver — Q has the
// same sparsity as the adjacency, so analyze() is one-shot and only the
// numeric factorization runs per gradient call.

#ifndef TULPA_HMC_CAR_PROPER_H
#define TULPA_HMC_CAR_PROPER_H

#include <vector>
#include <cmath>

namespace tulpa_car_proper {

// CSR matvec for proper-CAR precision: result = (D - rho*W) * v
// Reduces to icar_precision_matvec() when rho == 1.
inline void car_precision_matvec(
    const double* v,
    double* result,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double rho
) {
  for (int i = 0; i < n; i++) {
    double r = static_cast<double>(n_neighbors[i]) * v[i];
    int start = adj_row_ptr[i];
    int end = adj_row_ptr[i + 1];
    for (int k = start; k < end; k++) {
      r -= rho * v[adj_col_idx[k]];
    }
    result[i] = r;
  }
}

// Proper CAR data structure
struct ProperCARData {
  int n_spatial;                          // Number of spatial units

  // CSR format for adjacency matrix
  std::vector<int> adj_row_ptr;           // Row pointers (length n_spatial + 1)
  std::vector<int> adj_col_idx;           // Column indices
  std::vector<int> n_neighbors;           // Number of neighbors per unit

  // Rho bounds (from eigenvalue analysis)
  double rho_lower;                       // Lower bound for rho (typically 0)
  double rho_upper;                       // Upper bound for rho (typically 1)

  bool shared;                            // Whether effect is shared between num/denom
};

// -----------------------------------------------------------------------------
// Proper CAR precision matrix and log-determinant
// -----------------------------------------------------------------------------

// Compute Q = D - rho * W (precision matrix for proper CAR)
// Returns as flat vector (row-major)
inline std::vector<double> compute_car_precision(
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double rho
) {
  std::vector<double> Q(n * n, 0.0);

  for (int i = 0; i < n; i++) {
    // Diagonal: D_ii = n_neighbors[i]
    Q[i * n + i] = static_cast<double>(n_neighbors[i]);

    // Off-diagonal: -rho * W_ij
    int start = adj_row_ptr[i];
    int end = adj_row_ptr[i + 1];
    for (int k = start; k < end; k++) {
      int j = adj_col_idx[k];
      Q[i * n + j] = -rho;
    }
  }

  return Q;
}

// Compute log-determinant of Q = D - rho * W using Cholesky
// Returns -INFINITY if Q is not positive definite
inline double car_log_det(
    int n,
    const std::vector<double>& Q
) {
  // Cholesky decomposition: Q = L * L^T
  std::vector<double> L(n * n, 0.0);

  for (int j = 0; j < n; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = Q[j * n + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * n + m] * L[k * n + m];
      }
      if (j == k) {
        if (sum <= 0) return -INFINITY;  // Not positive definite
        L[j * n + j] = std::sqrt(sum);
      } else {
        L[j * n + k] = sum / L[k * n + k];
      }
    }
  }

  // log|Q| = 2 * sum(log(L_ii))
  double log_det = 0.0;
  for (int i = 0; i < n; i++) {
    log_det += std::log(L[i * n + i]);
  }
  return 2.0 * log_det;
}

// Compute quadratic form phi' Q phi
inline double car_quadratic_form(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double rho
) {
  double quad = 0.0;

  for (int i = 0; i < n; i++) {
    // Contribution from diagonal: n_neighbors[i] * phi[i]^2
    quad += n_neighbors[i] * phi[i] * phi[i];

    // Contribution from off-diagonal: -2 * rho * phi[i] * sum(phi[j] for j~i)
    // (factor of 2 because we only count upper triangle and it's symmetric)
    int start = adj_row_ptr[i];
    int end = adj_row_ptr[i + 1];
    for (int k = start; k < end; k++) {
      int j = adj_col_idx[k];
      if (j > i) {  // Only count each edge once
        quad -= 2.0 * rho * phi[i] * phi[j];
      }
    }
  }

  return quad;
}

// -----------------------------------------------------------------------------
// Proper CAR log-prior
// -----------------------------------------------------------------------------

// Compute log-prior for proper CAR: p(phi | tau, rho)
// p(phi | tau, rho) propto |Q|^{1/2} exp(-0.5 * tau * phi' Q phi)
// where Q = D - rho * W
inline double car_proper_log_prior(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double tau,
    double rho
) {
  // Compute quadratic form
  double quad = car_quadratic_form(phi, n, adj_row_ptr, adj_col_idx,
                                    n_neighbors, rho);

  // Compute log-determinant (expensive - consider caching)
  std::vector<double> Q = compute_car_precision(n, adj_row_ptr, adj_col_idx,
                                                 n_neighbors, rho);
  double log_det = car_log_det(n, Q);

  if (std::isinf(log_det) && log_det < 0) {
    return -INFINITY;  // Q not positive definite
  }

  // Log-prior: 0.5 * log|Q| + 0.5 * n * log(tau) - 0.5 * tau * phi' Q phi
  return 0.5 * log_det + 0.5 * n * std::log(tau) - 0.5 * tau * quad;
}

// Simplified version that assumes Q is positive definite and pre-computes
// the log-determinant for efficiency
inline double car_proper_log_prior_fast(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double tau,
    double rho,
    double log_det_Q  // Pre-computed log|Q|
) {
  double quad = car_quadratic_form(phi, n, adj_row_ptr, adj_col_idx,
                                    n_neighbors, rho);

  return 0.5 * log_det_Q + 0.5 * n * std::log(tau) - 0.5 * tau * quad;
}

// -----------------------------------------------------------------------------
// Priors for hyperparameters
// -----------------------------------------------------------------------------

// Log prior for rho: Beta(a, b) scaled to (rho_lower, rho_upper)
// Default: Uniform on (0, 1) = Beta(1, 1)
inline double log_prior_rho(double rho, double lower, double upper,
                            double a = 1.0, double b = 1.0) {
  if (rho <= lower || rho >= upper) return -INFINITY;

  // Transform to (0, 1)
  double u = (rho - lower) / (upper - lower);

  // Beta(a, b) log-density (up to normalizing constant)
  // We include the Jacobian from the scaling
  return (a - 1.0) * std::log(u) + (b - 1.0) * std::log(1.0 - u) -
         std::log(upper - lower);
}

// Log prior for tau: Gamma(shape, rate)
inline double log_prior_tau(double tau, double shape, double rate) {
  if (tau <= 0) return -INFINITY;
  return (shape - 1.0) * std::log(tau) - rate * tau +
         shape * std::log(rate) - std::lgamma(shape);
}

// -----------------------------------------------------------------------------
// Gradient computation (for HMC)
// -----------------------------------------------------------------------------

// Gradient of log-prior w.r.t. phi
inline void car_proper_gradient_phi(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double tau,
    double rho,
    double* grad_phi  // Output: length n
) {
  // d/d(phi_i) [-0.5 * tau * phi' Q phi]
  // = -tau * (Q phi)_i
  // = -tau * (n_neighbors[i] * phi[i] - rho * sum(phi[j] for j~i))

  for (int i = 0; i < n; i++) {
    double Qphi_i = n_neighbors[i] * phi[i];

    int start = adj_row_ptr[i];
    int end = adj_row_ptr[i + 1];
    for (int k = start; k < end; k++) {
      int j = adj_col_idx[k];
      Qphi_i -= rho * phi[j];
    }

    grad_phi[i] = -tau * Qphi_i;
  }
}

// Numerical gradient of log-prior w.r.t. rho (for HMC).
// Kept for reference / sanity-check tests; the production HMC path uses
// car_proper_log_det_and_grad_rho() below for an analytical derivative.
inline double car_proper_gradient_rho(
    const double* phi,
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double tau,
    double rho,
    double lower,
    double upper,
    double a = 1.0,
    double b = 1.0,
    double epsilon = 1e-6
) {
  // Finite difference for gradient
  double rho_plus = std::min(rho + epsilon, upper - epsilon);
  double rho_minus = std::max(rho - epsilon, lower + epsilon);

  double ll_plus = car_proper_log_prior(phi, n, adj_row_ptr, adj_col_idx,
                                         n_neighbors, tau, rho_plus) +
                   log_prior_rho(rho_plus, lower, upper, a, b);

  double ll_minus = car_proper_log_prior(phi, n, adj_row_ptr, adj_col_idx,
                                          n_neighbors, tau, rho_minus) +
                    log_prior_rho(rho_minus, lower, upper, a, b);

  return (ll_plus - ll_minus) / (rho_plus - rho_minus);
}

// Compute log|Q(rho)| AND tr(Q^{-1} W) in a single dense factorization.
//
// Both quantities are needed by the HMC gradient w.r.t. rho:
//   d/dρ [0.5 log|Q(ρ)| - 0.5 τ φ'Q(ρ)φ]
//     = -0.5 tr(Q^{-1} W) + 0.5 τ φ' W φ
// (since dQ/dρ = -W, so d log|Q|/dρ = tr(Q^{-1} dQ/dρ) = -tr(Q^{-1} W).)
//
// Sets log_det_out = log|Q|, trace_QinvW_out = tr(Q^{-1} W).
// Returns true on success, false if Q is not positive definite.
//
// Complexity: O(n^3) dense Cholesky + O(n * nnz(W)) trace accumulation.
// For large graphs prefer a sparse Cholesky + selected inversion.
inline bool car_proper_log_det_and_grad_rho(
    int n,
    const std::vector<int>& adj_row_ptr,
    const std::vector<int>& adj_col_idx,
    const std::vector<int>& n_neighbors,
    double rho,
    double* log_det_out,
    double* trace_QinvW_out
) {
  // Build Q = D - rho*W (column-major dense for Cholesky)
  std::vector<double> Q = compute_car_precision(n, adj_row_ptr, adj_col_idx,
                                                 n_neighbors, rho);

  // In-place Cholesky: Q = L * L^T, lower triangle stored in Q (row-major).
  // Use a fresh L matrix to avoid clobbering Q (we still need W structure
  // but W lives in adj_*, not Q).
  std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
  for (int j = 0; j < n; j++) {
    for (int k = 0; k <= j; k++) {
      double sum = Q[j * n + k];
      for (int m = 0; m < k; m++) {
        sum -= L[j * n + m] * L[k * n + m];
      }
      if (j == k) {
        if (sum <= 0.0) return false;  // not PD
        L[j * n + j] = std::sqrt(sum);
      } else {
        L[j * n + k] = sum / L[k * n + k];
      }
    }
  }

  // log|Q| = 2 * sum(log(L_ii))
  double log_det = 0.0;
  for (int i = 0; i < n; i++) {
    log_det += std::log(L[i * n + i]);
  }
  log_det *= 2.0;

  // tr(Q^{-1} W) = sum_{i~j} (Q^{-1})_{ij}.
  // Compute the full inverse via L * L^T = Q  =>  Q^{-1} = L^{-T} L^{-1}.
  // For each column k: solve L y = e_k (forward), then L^T x = y (back).
  // Accumulate the W-weighted trace using the adjacency CSR.
  //
  // Memory: O(n) per column, total O(n^2) work for the inverse, then
  // O(nnz(W)) accumulation. For n in the hundreds this is still cheap.
  std::vector<double> col(n);
  std::vector<double> Qinv(static_cast<size_t>(n) * n, 0.0);
  for (int k = 0; k < n; k++) {
    // forward solve L y = e_k
    for (int i = 0; i < n; i++) col[i] = (i == k) ? 1.0 : 0.0;
    for (int i = 0; i < n; i++) {
      double s = col[i];
      for (int j = 0; j < i; j++) s -= L[i * n + j] * col[j];
      col[i] = s / L[i * n + i];
    }
    // back solve L^T x = y (in place)
    for (int i = n - 1; i >= 0; i--) {
      double s = col[i];
      for (int j = i + 1; j < n; j++) s -= L[j * n + i] * col[j];
      col[i] = s / L[i * n + i];
    }
    // Store column k of Q^{-1}
    for (int i = 0; i < n; i++) Qinv[i * n + k] = col[i];
  }

  // Trace of Q^{-1} W: W has 1 at (i,j) for i~j (off-diagonal only since
  // adjacency excludes self-loops). tr(Q^{-1} W) = sum_i sum_{j~i} (Q^{-1})_{ij}.
  double trace = 0.0;
  for (int i = 0; i < n; i++) {
    int start = adj_row_ptr[i];
    int end = adj_row_ptr[i + 1];
    for (int idx = start; idx < end; idx++) {
      int j = adj_col_idx[idx];
      trace += Qinv[i * n + j];
    }
  }

  *log_det_out = log_det;
  *trace_QinvW_out = trace;
  return true;
}

} // namespace tulpa_car_proper

#endif // TULPA_HMC_CAR_PROPER_H
