#ifndef TULPA_VI_MATH_H
#define TULPA_VI_MATH_H
#include <RcppEigen.h>
#include <cmath>
#include <deque>
#include <limits>
#include <string>
#include <vector>
namespace tulpa { namespace vi {
// ---------------------------------------------------------------------
// Adam Optimizer State
// ---------------------------------------------------------------------

struct AdamState {
  Eigen::VectorXd m;  // First moment estimate
  Eigen::VectorXd v;  // Second moment estimate
  int t;              // Iteration counter

  AdamState() : t(0) {}

  AdamState(int n_params) : m(Eigen::VectorXd::Zero(n_params)),
                            v(Eigen::VectorXd::Zero(n_params)),
                            t(0) {}

  void reset() {
    m.setZero();
    v.setZero();
    t = 0;
  }
};

// ---------------------------------------------------------------------
// Adam Optimizer
// ---------------------------------------------------------------------

class AdamOptimizer {
public:
  double alpha;   // Learning rate
  double beta1;   // First moment decay
  double beta2;   // Second moment decay
  double eps;     // Numerical stability

  AdamOptimizer(double alpha_ = 0.01,
                double beta1_ = 0.9,
                double beta2_ = 0.999,
                double eps_ = 1e-8)
    : alpha(alpha_), beta1(beta1_), beta2(beta2_), eps(eps_) {}

  // Initialize state for given parameter dimension
  AdamState init_state(int n_params) const {
    return AdamState(n_params);
  }

  // Perform one optimization step
  // Returns updated parameters
  Eigen::VectorXd step(const Eigen::VectorXd& params,
                       const Eigen::VectorXd& grads,
                       AdamState& state) const {
    state.t++;

    // Update biased first moment estimate
    state.m = beta1 * state.m + (1.0 - beta1) * grads;

    // Update biased second moment estimate
    state.v = beta2 * state.v + (1.0 - beta2) * grads.array().square().matrix();

    // Bias-corrected estimates
    double bias_correction1 = 1.0 - std::pow(beta1, state.t);
    double bias_correction2 = 1.0 - std::pow(beta2, state.t);

    Eigen::VectorXd m_hat = state.m / bias_correction1;
    Eigen::VectorXd v_hat = state.v / bias_correction2;

    // Update parameters (gradient ascent for ELBO maximization)
    Eigen::VectorXd update = alpha * m_hat.array() / (v_hat.array().sqrt() + eps);

    return params + update;
  }

  // Step with gradient clipping
  Eigen::VectorXd step_clipped(const Eigen::VectorXd& params,
                               const Eigen::VectorXd& grads,
                               AdamState& state,
                               double max_grad_norm = 10.0) const {
    // Clip gradients by norm
    double grad_norm = grads.norm();
    Eigen::VectorXd clipped_grads = grads;
    if (grad_norm > max_grad_norm) {
      clipped_grads = grads * (max_grad_norm / grad_norm);
    }

    return step(params, clipped_grads, state);
  }
};

// ---------------------------------------------------------------------
// Convergence Checker
// ---------------------------------------------------------------------

// An ELBO offset cancels from the gain and span. Convergence also requires
// every gradient coordinate to be stationary over a complete window, with
// Bonferroni correction for the number of coordinates.
const double kVIStationaryAlpha = 0.05;
class ConvergenceChecker {
public:
  double tol_grad;         // Gradient norm tolerance
  double tol_rel_elbo;     // Window ELBO gain, as a fraction of the run's span
  int patience;            // Width of the window the gain is measured over

  // Internal state
  std::deque<double> window;   // the last `patience` ELBO values
  double best_elbo;
  double worst_elbo;
  bool has_best;           // false until the first ELBO has been recorded

  // Gradient moments over the current block of `patience` iterations, and the
  // verdict of the last complete block.
  Eigen::VectorXd grad_sum, grad_sumsq;
  int block_count = 0;
  bool grad_stationary = false;

  ConvergenceChecker(double tol_grad_ = 1e-4,
                     double tol_rel_elbo_ = 0.01,
                     int patience_ = 50)
    : tol_grad(tol_grad_),
      tol_rel_elbo(tol_rel_elbo_),
      patience(patience_ < 2 ? 2 : patience_),
      best_elbo(-std::numeric_limits<double>::infinity()),
      worst_elbo(std::numeric_limits<double>::infinity()),
      has_best(false) {}

  void reset() {
    window.clear();
    best_elbo = -std::numeric_limits<double>::infinity();
    worst_elbo = std::numeric_limits<double>::infinity();
    has_best = false;
    grad_sum.resize(0);
    grad_sumsq.resize(0);
    block_count = 0;
    grad_stationary = false;
  }

  // Fold one stochastic gradient into the block; at the block's end, test
  // every coordinate's mean against zero and start the next block.
  void observe_gradient(const Eigen::VectorXd& g) {
    if (grad_sum.size() != g.size()) {
      grad_sum = Eigen::VectorXd::Zero(g.size());
      grad_sumsq = Eigen::VectorXd::Zero(g.size());
      block_count = 0;
    }
    if (!g.allFinite()) return;
    grad_sum += g;
    grad_sumsq += g.array().square().matrix();
    if (++block_count < patience) return;

    const double n = block_count;
    const double z = R::qnorm(1.0 - 0.5 * kVIStationaryAlpha / g.size(),
                              0.0, 1.0, 1, 0);
    bool stationary = true;
    for (int j = 0; j < g.size() && stationary; ++j) {
      const double mean = grad_sum(j) / n;
      const double var = std::max(0.0, (grad_sumsq(j) - n * mean * mean) / (n - 1.0));
      stationary = std::abs(mean) <= z * std::sqrt(var / n);
    }
    grad_stationary = stationary;
    grad_sum.setZero();
    grad_sumsq.setZero();
    block_count = 0;
  }

  // Check if converged, returns convergence reason or empty string
  std::string check(double elbo, const Eigen::VectorXd& grad) {
    // Check gradient norm
    if (grad.norm() < tol_grad) {
      return "gradient_norm";
    }
    observe_gradient(grad);

    // A non-finite ELBO carries no information about the plateau; it neither
    // extends the run's span nor enters the window.
    if (!std::isfinite(elbo)) return "";

    if (!has_best || elbo > best_elbo)  best_elbo = elbo;
    if (!has_best || elbo < worst_elbo) worst_elbo = elbo;
    has_best = true;

    window.push_back(elbo);
    if (static_cast<int>(window.size()) > patience) window.pop_front();
    if (static_cast<int>(window.size()) < patience) return "";

    const int h = patience / 2;
    double first = 0.0, second = 0.0;
    for (int i = 0; i < h; ++i) first += window[i];
    for (int i = patience - h; i < patience; ++i) second += window[i];
    const double gain = (second - first) / h;
    const double span = best_elbo - worst_elbo;

    if (gain <= tol_rel_elbo * span && grad_stationary) {
      return "patience";
    }

    return "";  // Not converged
  }
};

// The path derivative holds q's parameters fixed in its density:
// (grad log p(theta) - grad log q(theta)) d theta / d phi.
// Roeder, Wu and Duvenaud (2017), equation 8, arXiv:1703.09194.
inline void path_gradient(Eigen::VectorXd& gradient,
                          const Eigen::VectorXd& precision_delta) {
  gradient += precision_delta;
}

// L and LQ describe the same covariance for an orthogonal Q. Procrustes
// alignment puts factor gradients in one frame before averaging them.
inline Eigen::MatrixXd align_factor_gradient(const Eigen::MatrixXd& L,
                                              const Eigen::MatrixXd& gradient,
                                              const Eigen::MatrixXd& reference) {
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(L.transpose()*reference,
      Eigen::ComputeFullU | Eigen::ComputeFullV);
  return gradient*(svd.matrixU()*svd.matrixV().transpose());
}

class LowRankPrecision {
  Eigen::VectorXd inverse_diagonal;
  Eigen::MatrixXd inverse_diagonal_factor;
  Eigen::LLT<Eigen::MatrixXd> factor;
public:
  LowRankPrecision(const Eigen::VectorXd& d, const Eigen::MatrixXd& L)
    : inverse_diagonal(d.array().square().inverse()),
      inverse_diagonal_factor(inverse_diagonal.asDiagonal()*L),
      factor(Eigen::MatrixXd::Identity(L.cols(),L.cols()) +
             L.transpose()*inverse_diagonal_factor) {}

  Eigen::VectorXd apply(const Eigen::VectorXd& delta) const {
    const Eigen::VectorXd scaled = inverse_diagonal.array()*delta.array();
    return scaled - inverse_diagonal_factor *
      factor.solve(inverse_diagonal_factor.transpose()*delta);
  }
};


} }
#endif
