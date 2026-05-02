// hmc_sampler_adapt.h
// Fragment of hmc_sampler.h. Included from the umbrella header inside
// namespace tulpa_hmc { ... }; do NOT add a namespace wrapper here.
// WelfordCovStats, WelfordStats, DualAveraging.
#ifndef TULPA_HMC_SAMPLER_ADAPT_H
#define TULPA_HMC_SAMPLER_ADAPT_H

// =====================================================================
// Online full covariance estimator (Welford's algorithm for outer products)
// =====================================================================

class WelfordCovStats {
public:
  int dim;
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // dim×dim: running sum of (x - mean_old)(x - mean_new)^T

  WelfordCovStats() : dim(0), n(0) {}

  explicit WelfordCovStats(int d) : dim(d), n(0),
    mean(d, 0.0), M2(static_cast<size_t>(d) * d, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    std::vector<double> delta(dim);
    for (int i = 0; i < dim; i++) {
      delta[i] = x[i] - mean[i];
      mean[i] += delta[i] / n;
    }
    // Outer product update: M2 += (x - mean_new) * delta^T
    for (int i = 0; i < dim; i++) {
      double dx_new = x[i] - mean[i];
      for (int j = 0; j <= i; j++) {
        double val = dx_new * delta[j];
        M2[static_cast<size_t>(j) * dim + i] += val;
        if (i != j) {
          M2[static_cast<size_t>(i) * dim + j] += val;  // Symmetric
        }
      }
    }
  }

  // Get covariance with Oracle Approximating Shrinkage (OAS)
  // (Chen, Wiesel, Eldar, Hero 2010)
  //
  // Σ_shrunk = (1 - ρ) * S + ρ * (tr(S)/p) * I
  //
  // OAS automatically adapts shrinkage intensity:
  //   - When n < p (rank-deficient): ρ → 1, shrinks heavily toward scaled identity
  //   - When n >> p: ρ → 0, recovers the unregularized sample covariance
  // Guarantees positive definiteness when ρ > 0.
  //
  // shrinkage_intensity is set as a side-effect for logging.
  mutable double shrinkage_intensity = 0.0;

  std::vector<double> covariance() const {
    std::vector<double> cov(static_cast<size_t>(dim) * dim, 0.0);
    if (n < 2) {
      // Return identity
      for (int i = 0; i < dim; i++) {
        cov[static_cast<size_t>(i) * dim + i] = 1.0;
      }
      shrinkage_intensity = 1.0;
      return cov;
    }

    // Step 1: Compute sample covariance S = M2 / (n - 1)
    double inv_nm1 = 1.0 / (n - 1);
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        cov[static_cast<size_t>(j) * dim + i] =
            M2[static_cast<size_t>(j) * dim + i] * inv_nm1;
      }
      // Ensure diagonal is positive
      double& diag = cov[static_cast<size_t>(i) * dim + i];
      diag = std::max(1e-6, diag);
    }

    // Step 2: Compute OAS shrinkage intensity
    // tr(S) and tr(S^2)
    double trS = 0.0;
    double trS2 = 0.0;
    for (int i = 0; i < dim; i++) {
      trS += cov[static_cast<size_t>(i) * dim + i];
    }
    // tr(S^2) = sum of all S_ij^2 (Frobenius norm squared)
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        double s_ij = cov[static_cast<size_t>(j) * dim + i];
        trS2 += s_ij * s_ij;
      }
    }

    // OAS formula (Eq. 23 from Chen et al. 2010):
    // ρ = clamp( ((1 - 2/p) * tr(S²) + tr(S)²) /
    //            ((n + 1 - 2/p) * (tr(S²) - tr(S)²/p)), 0, 1 )
    double p = static_cast<double>(dim);
    double nf = static_cast<double>(n);
    double trS_sq = trS * trS;
    double numer = (1.0 - 2.0 / p) * trS2 + trS_sq;
    double denom = (nf + 1.0 - 2.0 / p) * (trS2 - trS_sq / p);

    double rho;
    if (std::abs(denom) < 1e-12) {
      // Denominator ~0 means S ≈ c*I already, minimal shrinkage needed
      rho = 0.0;
    } else {
      rho = std::max(0.0, std::min(1.0, numer / denom));
    }

    // When n < p, sample covariance is rank-deficient (rank n-1 < p).
    // OAS can underestimate the needed shrinkage for singular matrices.
    // Floor ρ at (1 - n/p) to fill the rank gap: this ensures the
    // p-(n-1) zero eigenvalues get lifted to ρ * tr(S)/p > 0.
    if (n < dim) {
      double rho_floor = 1.0 - nf / p;
      rho = std::max(rho, rho_floor);
    }

    // Even when n > p, moderate n/p ratios (2-5) can produce poorly
    // conditioned matrices. Apply a floor that decays as n/p grows.
    // At n/p=2: floor=0.08, n/p=5: floor=0.05, n/p=10+: floor=0.0
    if (nf < 10.0 * p) {
      double rho_cond_floor = 0.1 * (1.0 - nf / (10.0 * p));
      rho = std::max(rho, rho_cond_floor);
    }
    shrinkage_intensity = rho;

    // Step 3: Apply shrinkage: Σ = (1 - ρ) * S + ρ * (tr(S)/p) * I
    double target_diag = trS / p;  // Scaled identity target
    double one_minus_rho = 1.0 - rho;
    for (int i = 0; i < dim; i++) {
      for (int j = 0; j < dim; j++) {
        size_t idx = static_cast<size_t>(j) * dim + i;
        if (i == j) {
          cov[idx] = one_minus_rho * cov[idx] + rho * target_diag;
        } else {
          cov[idx] = one_minus_rho * cov[idx];
        }
      }
    }

    return cov;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

// =====================================================================
// Welford's online algorithm for mean and variance
// Used for diagonal mass matrix estimation during warmup
// =====================================================================

class WelfordStats {
public:
  int n;
  std::vector<double> mean;
  std::vector<double> M2;  // Sum of squared differences from mean

  WelfordStats(int dim) : n(0), mean(dim, 0.0), M2(dim, 0.0) {}

  void update(const std::vector<double>& x) {
    n++;
    for (size_t i = 0; i < x.size(); i++) {
      double delta = x[i] - mean[i];
      mean[i] += delta / n;
      double delta2 = x[i] - mean[i];
      M2[i] += delta * delta2;
    }
  }

  std::vector<double> variance() const {
    std::vector<double> var(mean.size());
    if (n < 2) {
      // Return unit variance if not enough samples
      std::fill(var.begin(), var.end(), 1.0);
    } else {
      for (size_t i = 0; i < mean.size(); i++) {
        var[i] = M2[i] / (n - 1);
        // Ensure minimum variance to avoid numerical issues
        if (var[i] < 1e-6) var[i] = 1e-6;
      }
    }
    return var;
  }

  // Get inverse mass matrix (= variance, regularized for stability)
  // For HMC: M = diag(1/var), so M^{-1} = diag(var)
  // High variance parameters should move faster in position space
  // Uses Stan-style Bayesian shrinkage toward unit variance to prevent
  // extreme mass matrix entries from small sample sizes
  std::vector<double> inv_mass() const {
    auto var = variance();
    double shrink = (n < 2) ? 0.0 : (double)n / (n + 5.0);
    for (size_t i = 0; i < var.size(); i++) {
      var[i] = shrink * var[i] + 1e-3 * (5.0 / (n + 5.0));
      // Safety clamp for extreme values
      var[i] = std::max(1e-3, std::min(var[i], 1e3));
    }
    return var;
  }

  // Get sqrt of mass matrix for momentum sampling
  // Since M = diag(1/var), sqrt(M) = diag(1/sqrt(var))
  // p ~ N(0, M), so p_i = z_i / sqrt(var_i)
  // Uses the same regularized variance as inv_mass()
  std::vector<double> sqrt_mass() const {
    auto inv_m = inv_mass();
    std::vector<double> sqrt_m(inv_m.size());
    for (size_t i = 0; i < inv_m.size(); i++) {
      sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
    }
    return sqrt_m;
  }

  void reset() {
    n = 0;
    std::fill(mean.begin(), mean.end(), 0.0);
    std::fill(M2.begin(), M2.end(), 0.0);
  }
};

struct DualAveraging {
  double mu, log_epsilon_bar, H_bar;
  double gamma, t0, kappa;
  double target_accept;  // Target acceptance rate (dimension-adaptive)
  int m;

  // Default constructor with dimension-adaptive target
  // target_boost: additional boost to target acceptance for challenging models
  //               (e.g., +0.10 for MSGP+temporal combinations)
  DualAveraging(double epsilon_init = 1.0, int n_params = 1, double target_boost = 0.0);
  double update(double alpha);
  double final_epsilon() const;

  // Compute dimension-adaptive target acceptance rate
  // Higher targets (closer to 1) = smaller step sizes = fewer divergences
  // but slower exploration. Stan default is 0.80.
  // target_boost: additional boost for challenging model combinations
  static double compute_target(int n_params, double target_boost = 0.0) {
    // Use higher targets (0.75-0.85) to avoid divergences in
    // challenging models (ICAR, hierarchical negbin, etc.)
    double base_target;
    if (n_params <= 5) base_target = 0.85;
    else if (n_params <= 20) base_target = 0.82;
    else if (n_params <= 50) base_target = 0.80;
    else if (n_params <= 100) base_target = 0.78;
    else base_target = 0.75;

    // Apply boost, cap at 0.99
    return std::min(0.99, base_target + target_boost);
  }
};

#endif  // TULPA_HMC_SAMPLER_ADAPT_H
