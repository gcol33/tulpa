// vi_types.h
// Variational Inference types, configs, and parameter structures
// Supports mean-field, low-rank, and full-rank Gaussian approximations

#ifndef TULPA_VI_TYPES_H
#define TULPA_VI_TYPES_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <memory>
#include <random>

// Use Eigen for matrix operations
#include <RcppEigen.h>

namespace tulpa {
namespace vi {

// ---------------------------------------------------------------------
// VI Variant Enumeration
// ---------------------------------------------------------------------

enum class VIVariant {
  MEANFIELD,   // Diagonal covariance: q(θ) = ∏ N(θᵢ; μᵢ, σᵢ²)
  LOWRANK,     // Low-rank + diagonal: q(θ) = N(θ; μ, LL' + D)
  FULLRANK,    // Full covariance: q(θ) = N(θ; μ, LL')
  AUTO         // Auto-select based on dimension
};

// Convert string to VIVariant
inline VIVariant parse_variant(const std::string& s) {
  if (s == "meanfield") return VIVariant::MEANFIELD;
  if (s == "lowrank") return VIVariant::LOWRANK;
  if (s == "fullrank") return VIVariant::FULLRANK;
  if (s == "auto") return VIVariant::AUTO;
  Rcpp::stop("Unknown VI variant: " + s);
  return VIVariant::AUTO;  // Never reached
}

// Convert VIVariant to string
inline std::string variant_to_string(VIVariant v) {
  switch (v) {
    case VIVariant::MEANFIELD: return "meanfield";
    case VIVariant::LOWRANK: return "lowrank";
    case VIVariant::FULLRANK: return "fullrank";
    case VIVariant::AUTO: return "auto";
  }
  return "unknown";
}

// ---------------------------------------------------------------------
// VI Configuration
// ---------------------------------------------------------------------

struct VIConfig {
  // Variant selection
  VIVariant variant = VIVariant::AUTO;

  // Optimization parameters
  int max_iter = 10000;
  int mc_samples = 10;        // Monte Carlo samples for gradient
  double tol_grad = 1e-4;     // Gradient norm tolerance
  double tol_rel_elbo = 0.01; // Relative ELBO change tolerance
  int patience = 50;          // Early stopping patience

  // Adam optimizer parameters
  double adam_alpha = 0.01;
  double adam_beta1 = 0.9;
  double adam_beta2 = 0.999;
  double adam_eps = 1e-8;

  // Low-rank specific
  int rank = -1;  // -1 = auto (D/10 clamped to [10, 50])

  // Initialization
  bool use_laplace_init = true;

  // Auto-selection thresholds
  int fullrank_threshold = 200;
  int lowrank_threshold = 2000;

  // Output control
  bool verbose = true;
  int print_every = 100;

  // Random seed
  unsigned int seed = 0;
};

// Parse config from R list
inline VIConfig parse_vi_config(Rcpp::List options) {
  VIConfig config;

  if (options.containsElementNamed("variant")) {
    config.variant = parse_variant(Rcpp::as<std::string>(options["variant"]));
  }
  if (options.containsElementNamed("max_iter")) {
    config.max_iter = Rcpp::as<int>(options["max_iter"]);
  }
  if (options.containsElementNamed("mc_samples")) {
    config.mc_samples = Rcpp::as<int>(options["mc_samples"]);
  }
  if (options.containsElementNamed("tol_grad")) {
    config.tol_grad = Rcpp::as<double>(options["tol_grad"]);
  }
  if (options.containsElementNamed("tol_rel_elbo")) {
    config.tol_rel_elbo = Rcpp::as<double>(options["tol_rel_elbo"]);
  }
  if (options.containsElementNamed("patience")) {
    config.patience = Rcpp::as<int>(options["patience"]);
  }
  if (options.containsElementNamed("adam_alpha")) {
    config.adam_alpha = Rcpp::as<double>(options["adam_alpha"]);
  }
  if (options.containsElementNamed("adam_beta1")) {
    config.adam_beta1 = Rcpp::as<double>(options["adam_beta1"]);
  }
  if (options.containsElementNamed("adam_beta2")) {
    config.adam_beta2 = Rcpp::as<double>(options["adam_beta2"]);
  }
  if (options.containsElementNamed("adam_eps")) {
    config.adam_eps = Rcpp::as<double>(options["adam_eps"]);
  }
  if (options.containsElementNamed("rank")) {
    config.rank = Rcpp::as<int>(options["rank"]);
  }
  if (options.containsElementNamed("use_laplace_init")) {
    config.use_laplace_init = Rcpp::as<bool>(options["use_laplace_init"]);
  }
  if (options.containsElementNamed("verbose")) {
    config.verbose = Rcpp::as<bool>(options["verbose"]);
  }
  if (options.containsElementNamed("print_every")) {
    config.print_every = Rcpp::as<int>(options["print_every"]);
  }
  if (options.containsElementNamed("seed")) {
    config.seed = Rcpp::as<unsigned int>(options["seed"]);
  }

  return config;
}

// ---------------------------------------------------------------------
// Variant Selection Logic
// ---------------------------------------------------------------------

inline VIVariant select_variant(int D, const VIConfig& config) {
  if (config.variant != VIVariant::AUTO) {
    return config.variant;
  }

  if (D < config.fullrank_threshold) {
    return VIVariant::FULLRANK;
  } else if (D < config.lowrank_threshold) {
    return VIVariant::LOWRANK;
  } else {
    return VIVariant::MEANFIELD;
  }
}

inline int select_rank(int D, const VIConfig& config) {
  if (config.rank > 0) {
    return config.rank;
  }
  // Auto: D/10 clamped to [10, 50]
  return std::max(10, std::min(D / 10, 50));
}

// ---------------------------------------------------------------------
// Variational Parameter Base Class
// ---------------------------------------------------------------------

struct VIParamsBase {
  virtual ~VIParamsBase() = default;

  // Dimension of the target parameter space
  virtual int dim() const = 0;

  // Number of variational parameters
  virtual int n_variational_params() const = 0;

  // Sample from q(θ)
  virtual Eigen::VectorXd sample(std::mt19937& rng) const = 0;

  // Entropy H[q]
  virtual double entropy() const = 0;

  // Get mean
  virtual Eigen::VectorXd mean() const = 0;

  // Get covariance matrix (may be expensive for full-rank)
  virtual Eigen::MatrixXd covariance() const = 0;

  // Flatten to vector (for optimization)
  virtual Eigen::VectorXd flatten() const = 0;

  // Unflatten from vector
  virtual void unflatten(const Eigen::VectorXd& x) = 0;
};

// ---------------------------------------------------------------------
// Mean-Field Parameters
// ---------------------------------------------------------------------

struct MeanFieldParams : VIParamsBase {
  Eigen::VectorXd mu;        // Mean vector (D)
  Eigen::VectorXd log_sigma; // Log standard deviations (D)

  MeanFieldParams() = default;

  MeanFieldParams(int D) : mu(Eigen::VectorXd::Zero(D)),
                           log_sigma(Eigen::VectorXd::Constant(D, -1.0)) {}

  int dim() const override {
    return static_cast<int>(mu.size());
  }

  int n_variational_params() const override {
    return 2 * dim();
  }

  Eigen::VectorXd sample(std::mt19937& rng) const override {
    std::normal_distribution<double> N01(0.0, 1.0);
    int D = dim();
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }
    // θ = μ + σ ⊙ ε
    return mu.array() + exp(log_sigma.array()) * eps.array();
  }

  double entropy() const override {
    // H[q] = Σᵢ log σᵢ + D/2 (1 + log 2π)
    return log_sigma.sum() + 0.5 * dim() * (1.0 + std::log(2.0 * M_PI));
  }

  Eigen::VectorXd mean() const override {
    return mu;
  }

  Eigen::MatrixXd covariance() const override {
    Eigen::VectorXd var = exp(2.0 * log_sigma.array());
    return var.asDiagonal();
  }

  Eigen::VectorXd flatten() const override {
    Eigen::VectorXd x(n_variational_params());
    x.head(dim()) = mu;
    x.tail(dim()) = log_sigma;
    return x;
  }

  void unflatten(const Eigen::VectorXd& x) override {
    int D = dim();
    mu = x.head(D);
    log_sigma = x.tail(D);
  }
};

// ---------------------------------------------------------------------
// Full-Rank Parameters
// ---------------------------------------------------------------------

struct FullRankParams : VIParamsBase {
  Eigen::VectorXd mu;  // Mean vector (D)
  Eigen::MatrixXd L;   // Lower triangular Cholesky factor (D x D)

  FullRankParams() = default;

  FullRankParams(int D) : mu(Eigen::VectorXd::Zero(D)),
                          L(Eigen::MatrixXd::Identity(D, D) * 0.5) {}

  int dim() const override {
    return static_cast<int>(mu.size());
  }

  int n_variational_params() const override {
    int D = dim();
    return D + D * (D + 1) / 2;  // mu + lower triangle of L
  }

  Eigen::VectorXd sample(std::mt19937& rng) const override {
    std::normal_distribution<double> N01(0.0, 1.0);
    int D = dim();
    Eigen::VectorXd eps(D);
    for (int i = 0; i < D; ++i) {
      eps(i) = N01(rng);
    }
    // θ = μ + L ε
    return mu + L * eps;
  }

  double entropy() const override {
    // H[q] = Σᵢ log |Lᵢᵢ| + D/2 (1 + log 2π)
    double log_det = 0.0;
    int D = dim();
    for (int i = 0; i < D; ++i) {
      log_det += std::log(std::abs(L(i, i)) + 1e-10);
    }
    return log_det + 0.5 * D * (1.0 + std::log(2.0 * M_PI));
  }

  Eigen::VectorXd mean() const override {
    return mu;
  }

  Eigen::MatrixXd covariance() const override {
    return L * L.transpose();
  }

  Eigen::VectorXd flatten() const override {
    int D = dim();
    int n_L = D * (D + 1) / 2;
    Eigen::VectorXd x(D + n_L);

    x.head(D) = mu;

    // Flatten lower triangle row by row
    int idx = D;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j <= i; ++j) {
        x(idx++) = L(i, j);
      }
    }

    return x;
  }

  void unflatten(const Eigen::VectorXd& x) override {
    int D = dim();
    mu = x.head(D);

    // Unflatten lower triangle
    L.setZero();
    int idx = D;
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j <= i; ++j) {
        L(i, j) = x(idx++);
      }
      // Ensure positive diagonal
      if (L(i, i) < 0.01) L(i, i) = 0.01;
    }
  }
};

// ---------------------------------------------------------------------
// Low-Rank Parameters
// ---------------------------------------------------------------------

struct LowRankParams : VIParamsBase {
  Eigen::VectorXd mu;     // Mean vector (D)
  Eigen::MatrixXd L;      // Low-rank factor (D x r)
  Eigen::VectorXd log_d;  // Log diagonal component (D)

  LowRankParams() = default;

  LowRankParams(int D, int r) : mu(Eigen::VectorXd::Zero(D)),
                                 L(Eigen::MatrixXd::Zero(D, r) * 0.1),
                                 log_d(Eigen::VectorXd::Constant(D, -1.0)) {
    // Initialize L with small random values
    std::mt19937 rng(42);
    std::normal_distribution<double> N01(0.0, 0.1);
    for (int i = 0; i < D; ++i) {
      for (int j = 0; j < r; ++j) {
        L(i, j) = N01(rng);
      }
    }
  }

  int dim() const override {
    return static_cast<int>(mu.size());
  }

  int rank() const {
    return static_cast<int>(L.cols());
  }

  int n_variational_params() const override {
    return dim() + dim() * rank() + dim();  // mu + L + log_d
  }

  Eigen::VectorXd sample(std::mt19937& rng) const override {
    std::normal_distribution<double> N01(0.0, 1.0);
    int D = dim();
    int r = rank();

    Eigen::VectorXd eta(r), eps(D);
    for (int i = 0; i < r; ++i) eta(i) = N01(rng);
    for (int i = 0; i < D; ++i) eps(i) = N01(rng);

    // θ = μ + L η + d ⊙ ε
    Eigen::VectorXd d = exp(log_d.array());
    return mu + L * eta + d.asDiagonal() * eps;
  }

  double entropy() const override {
    // Σ = LL' + D where D = diag(d²)
    // log|Σ| via matrix determinant lemma:
    // log|LL' + D| = log|D| + log|I + L'D⁻¹L|
    int D_dim = dim();
    int r = rank();
    Eigen::VectorXd d2 = exp(2.0 * log_d.array());

    // log|D| = sum(2 * log_d)
    double log_det_D = 2.0 * log_d.sum();

    // Compute I + L'D⁻¹L (r × r matrix)
    Eigen::MatrixXd LtDinvL = Eigen::MatrixXd::Identity(r, r);
    for (int i = 0; i < D_dim; ++i) {
      LtDinvL.noalias() += L.row(i).transpose() * L.row(i) / d2(i);
    }

    // log|I + L'D⁻¹L| via Cholesky
    Eigen::LLT<Eigen::MatrixXd> llt(LtDinvL);
    double log_det_inner = 0.0;
    Eigen::MatrixXd L_chol = llt.matrixL();
    for (int i = 0; i < r; ++i) {
      log_det_inner += 2.0 * std::log(L_chol(i, i));
    }

    double log_det_Sigma = log_det_D + log_det_inner;
    return 0.5 * log_det_Sigma + 0.5 * D_dim * (1.0 + std::log(2.0 * M_PI));
  }

  Eigen::VectorXd mean() const override {
    return mu;
  }

  Eigen::MatrixXd covariance() const override {
    Eigen::VectorXd d2 = exp(2.0 * log_d.array());
    Eigen::MatrixXd LLt = L * L.transpose();
    Eigen::MatrixXd D_mat = d2.asDiagonal();
    return LLt + D_mat;
  }

  Eigen::VectorXd flatten() const override {
    int D = dim();
    int r = rank();
    Eigen::VectorXd x(n_variational_params());

    x.head(D) = mu;

    // Flatten L column by column
    Eigen::Map<Eigen::VectorXd>(x.data() + D, D * r) =
        Eigen::Map<const Eigen::VectorXd>(L.data(), D * r);

    x.tail(D) = log_d;

    return x;
  }

  void unflatten(const Eigen::VectorXd& x) override {
    int D = dim();
    int r = rank();

    mu = x.head(D);

    Eigen::Map<Eigen::MatrixXd>(L.data(), D, r) =
        Eigen::Map<const Eigen::MatrixXd>(x.data() + D, D, r);

    log_d = x.tail(D);
  }
};

// ---------------------------------------------------------------------
// VI Result Structure
// ---------------------------------------------------------------------

struct VIResult {
  VIVariant variant_used;

  // Final variational parameters
  Eigen::VectorXd mu;          // Posterior mean
  Eigen::MatrixXd Sigma;       // Posterior covariance

  // For low-rank: store components separately
  Eigen::MatrixXd L_factor;    // Low-rank factor or Cholesky
  Eigen::VectorXd d_diag;      // Diagonal component (low-rank only)
  int rank_used;               // Rank used (low-rank only)

  // Optimization results
  double final_elbo;
  int iterations;
  bool converged;
  std::vector<double> elbo_history;

  // Posterior samples for diagnostics
  Eigen::MatrixXd samples;     // (n_samples × D)

  // Diagnostics
  double psis_k;               // Pareto-k diagnostic (-1 if not computed)

  VIResult() : variant_used(VIVariant::AUTO), rank_used(0),
               final_elbo(-std::numeric_limits<double>::infinity()),
               iterations(0), converged(false), psis_k(-1.0) {}
};

// Convert VIResult to R list
inline Rcpp::List vi_result_to_list(const VIResult& result) {
  return Rcpp::List::create(
    Rcpp::Named("variant") = variant_to_string(result.variant_used),
    Rcpp::Named("mu") = result.mu,
    Rcpp::Named("Sigma") = result.Sigma,
    Rcpp::Named("L") = result.L_factor,
    Rcpp::Named("d") = result.d_diag,
    Rcpp::Named("rank") = result.rank_used,
    Rcpp::Named("elbo") = result.final_elbo,
    Rcpp::Named("iterations") = result.iterations,
    Rcpp::Named("converged") = result.converged,
    Rcpp::Named("elbo_history") = result.elbo_history,
    Rcpp::Named("samples") = result.samples,
    Rcpp::Named("psis_k") = result.psis_k
  );
}

} // namespace vi
} // namespace tulpa

#endif // TULPA_VI_TYPES_H
