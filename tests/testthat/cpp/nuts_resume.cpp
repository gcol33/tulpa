#include <Rcpp.h>
#include "hmc_sampler.h"
#include <tulpa/likelihood.h>
#include <tulpa/autodiff_arena.h>

namespace {
constexpr double variance[] = {0.002, 0.005, 0.007};

template <typename T>
T normal_density(int i, const T*, const T&, const T&,
                 const std::vector<T>& q, const tulpa::ModelData&,
                 const tulpa::ParamLayout& layout, const void*) {
  const T x = q[layout.extra_offset + i];
  return T(-0.5 / variance[i]) * x * x;
}

double potential(const std::vector<double>& q) {
  double out = 0;
  for (int j = 0; j < 3; ++j) out += 0.5 * q[j] * q[j] / variance[j];
  return out;
}

template <typename T>
T correlated_normal(int i, const T*, const T&, const T&,
                    const std::vector<T>& q, const tulpa::ModelData&,
                    const tulpa::ParamLayout& layout, const void* context) {
  const double rho = *static_cast<const double*>(context);
  const T x = q[layout.extra_offset];
  const T y = q[layout.extra_offset + 1];
  return T(-0.5 / (1 - rho * rho)) *
      (i == 0 ? x * x - T(2 * rho) * x * y : y * y);
}
}

// [[Rcpp::export]]
Rcpp::List cpp_test_nuts_stepsize_search(int seed, std::string metric, double initial) {
  using namespace tulpa_hmc;
  double rho = metric == "diag" ? 0 : 0.6;
  tulpa::LikelihoodSpec spec;
  spec.name = "correlated_normal";
  spec.n_extra_params = 2;
  spec.ll_double = correlated_normal<double>;
  spec.ll_arena = correlated_normal<tulpa::arena::Var>;
  ModelData data;
  data.N = 2;
  data.n_processes = 1;
  data.processes.resize(1);
  data.processes[0].p = 0;
  data.sharing.init(1);
  data.likelihood_spec = &spec;
  data.model_response_data = &rho;
  const auto layout = compute_param_layout(data);
  DenseMassMatrix mass;
  if (metric == "dense") {
    mass.init(2, MassMatrixType::DENSE);
    const double covariance[] = {1, rho, rho, 1};
    if (!mass.update_from_covariance(covariance, 100)) Rcpp::stop("Metric failed");
  } else if (metric == "block") {
    mass.init_block_diag(2, {{0, 2}});
    mass.set_diagonal({1, 1}, {1, 1});
    auto& block = mass.blocks[0];
    block.inv_mass[0] = block.inv_mass[5] = 1;
    block.inv_mass[1] = block.inv_mass[4] = rho;
    if (!MassBlock::cholesky_small(block.inv_mass, block.L_inv_mass, 2))
      Rcpp::stop("Block failed");
    block.adapted = true;
  } else {
    mass.init(2, MassMatrixType::DIAG);
    mass.set_diagonal({1, 1}, {1, 1});
  }
  GradientModeFitScope scope;
  set_gradient_mode(GradientMode::AUTODIFF_ARENA);
  std::mt19937 actual_rng(seed), expected_rng(seed);
  const std::vector<double> q(2, 0);
  const double actual = find_reasonable_epsilon(q, data, layout, actual_rng, mass, initial);

  // At q = 0 with inverse mass equal to the Gaussian covariance, the
  // velocity-Verlet energy error is epsilon^4 * ||z||^2 / 8.
  double expected = initial;
  auto error = [&]() {
    std::normal_distribution<double> normal(0, 1);
    if (metric == "block") { normal(expected_rng); normal(expected_rng); }
    const double z0 = normal(expected_rng), z1 = normal(expected_rng);
    return -std::pow(expected, 4) * (z0 * z0 + z1 * z1) / 8;
  };
  const int direction = error() > std::log(0.8) ? 1 : -1;
  for (;;) {
    const double delta = error();
    if ((direction == 1 && delta <= std::log(0.8)) ||
        (direction == -1 && delta >= std::log(0.8))) break;
    expected *= direction == 1 ? 2 : 0.5;
  }
  return Rcpp::List::create(Rcpp::Named("actual") = actual,
      Rcpp::Named("expected") = expected,
      Rcpp::Named("rng_equal") = (actual_rng == expected_rng));
}

// [[Rcpp::export]]
Rcpp::List cpp_test_nuts_resume_kernel(int seed, double epsilon = 0.0) {
  using namespace tulpa_hmc;
  tulpa::LikelihoodSpec spec;
  spec.name = "normal_resume";
  spec.n_processes = 1;
  spec.n_extra_params = 3;
  spec.ll_double = normal_density<double>;
  spec.ll_arena = normal_density<tulpa::arena::Var>;
  ModelData data;
  data.N = 3;
  data.n_processes = 1;
  data.processes.resize(1);
  data.processes[0].p = 0;
  data.sharing.init(1);
  data.likelihood_spec = &spec;
  const auto layout = compute_param_layout(data);
  const auto reference = run_hmc_chain_cpp({0, 0, 0}, data, layout,
      401, 400, 0, 1, 9, false, 7, MassMatrixType::DIAG, 0.85);
  const auto& q0 = reference.final_position;
  const auto& inv = reference.inv_metric_diag;
  const double step = epsilon > 0 ? epsilon : reference.epsilon;
  const auto resumed = run_hmc_chain_cpp(q0, data, layout,
      1, 0, 0, 1, seed, false, 1, MassMatrixType::DIAG, 0.85, nullptr, inv, step);

  // At depth one the only candidate is one velocity-Verlet step on the
  // quadratic potential. Its closed-form state and energy define the kernel.
  std::mt19937 rng(seed + 12345);
  std::uniform_real_distribution<double> uniform(0, 1);
  const double jittered = step * (1 + 0.2 * (2 * uniform(rng) - 1));
  std::normal_distribution<double> normal(0, 1);
  std::vector<double> p(3), q1(3), p1(3);
  double H0 = potential(q0);
  for (int j = 0; j < 3; ++j) {
    p[j] = normal(rng) / std::sqrt(inv[j]);
    H0 += 0.5 * inv[j] * p[j] * p[j];
  }
  std::uniform_int_distribution<int> direction(0, 1);
  const double h = jittered * (2 * direction(rng) - 1);
  double H1 = 0;
  for (int j = 0; j < 3; ++j) {
    const double half = p[j] - 0.5 * h * q0[j] / variance[j];
    q1[j] = q0[j] + h * inv[j] * half;
    p1[j] = half - 0.5 * h * q1[j] / variance[j];
    H1 += 0.5 * inv[j] * p1[j] * p1[j];
  }
  H1 += potential(q1);
  const double candidate_prob = 1 / (1 + std::exp(H1 - H0));
  const auto expected = uniform(rng) < candidate_prob ? q1 : q0;
  return Rcpp::List::create(
      Rcpp::Named("draw") = resumed.final_position,
      Rcpp::Named("expected") = expected,
      Rcpp::Named("energy") = resumed.energy[0],
      Rcpp::Named("expected_energy") = H0,
      Rcpp::Named("accept") = resumed.accept_prob[0],
      Rcpp::Named("expected_accept") = std::min(1.0, std::exp(H0 - H1)),
      Rcpp::Named("epsilon") = resumed.epsilon,
      Rcpp::Named("expected_epsilon") = step,
      Rcpp::Named("inv_metric") = resumed.inv_metric_diag,
      Rcpp::Named("expected_metric") = inv,
      Rcpp::Named("divergences") = resumed.divergent[0]);
}
