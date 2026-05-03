// test_helpers.cpp
// Rcpp wrappers to expose internal C++ functions for unit testing

#include <Rcpp.h>
#include <cmath>
#include <vector>
#include <random>
#include "autodiff.h"
#include "laplace_core.h"
#include "pg_binomial.h"
#include "hmc_gp.h"

using namespace Rcpp;

// ---------------------------------------------------------------------------
// Leapfrog integrator (standalone version for testing)
// ---------------------------------------------------------------------------

// Simple quadratic potential for testing: U(q) = 0.5 * sum(q^2)
// Gradient: dU/dq = q
// [[Rcpp::export]]
List cpp_test_leapfrog(
    NumericVector q_init,
    NumericVector p_init,
    double epsilon,
    int L
) {
  int d = q_init.size();
  std::vector<double> q(q_init.begin(), q_init.end());
  std::vector<double> p(p_init.begin(), p_init.end());

  // Leapfrog integration with quadratic potential U(q) = 0.5 * sum(q^2)
  // Gradient dU/dq = q

  // Half step for momentum
  for (int i = 0; i < d; i++) {
    p[i] = p[i] - 0.5 * epsilon * q[i];
  }

  // Full steps
  for (int step = 0; step < L - 1; step++) {
    // Full step for position
    for (int i = 0; i < d; i++) {
      q[i] = q[i] + epsilon * p[i];
    }
    // Full step for momentum
    for (int i = 0; i < d; i++) {
      p[i] = p[i] - epsilon * q[i];
    }
  }

  // Final full step for position
  for (int i = 0; i < d; i++) {
    q[i] = q[i] + epsilon * p[i];
  }

  // Half step for momentum
  for (int i = 0; i < d; i++) {
    p[i] = p[i] - 0.5 * epsilon * q[i];
  }

  return List::create(
    Named("q") = NumericVector(q.begin(), q.end()),
    Named("p") = NumericVector(p.begin(), p.end())
  );
}

// ---------------------------------------------------------------------------
// Hamiltonian computation for testing
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_hamiltonian(NumericVector q, NumericVector p) {
  // H(q, p) = U(q) + K(p) = 0.5 * sum(q^2) + 0.5 * sum(p^2)
  double H = 0.0;
  for (int i = 0; i < q.size(); i++) {
    H += 0.5 * q[i] * q[i] + 0.5 * p[i] * p[i];
  }
  return H;
}

// ---------------------------------------------------------------------------
// Log-sum-exp (numerically stable)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_log_sum_exp(NumericVector log_vals) {
  if (log_vals.size() == 0) return R_NegInf;

  double max_val = *std::max_element(log_vals.begin(), log_vals.end());
  if (max_val == R_NegInf) return R_NegInf;

  double sum_exp = 0.0;
  for (int i = 0; i < log_vals.size(); i++) {
    sum_exp += std::exp(log_vals[i] - max_val);
  }

  return max_val + std::log(sum_exp);
}

// ---------------------------------------------------------------------------
// Softmax (numerically stable)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector cpp_test_softmax(NumericVector x) {
  double max_val = *std::max_element(x.begin(), x.end());
  NumericVector result(x.size());
  double sum_exp = 0.0;

  for (int i = 0; i < x.size(); i++) {
    result[i] = std::exp(x[i] - max_val);
    sum_exp += result[i];
  }

  for (int i = 0; i < x.size(); i++) {
    result[i] /= sum_exp;
  }

  return result;
}

// ---------------------------------------------------------------------------
// Inverse logit (sigmoid)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector cpp_test_inv_logit(NumericVector x) {
  NumericVector result(x.size());
  for (int i = 0; i < x.size(); i++) {
    result[i] = 1.0 / (1.0 + std::exp(-x[i]));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Log-gamma function wrapper
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_lgamma(double x) {
  return std::lgamma(x);
}

// ---------------------------------------------------------------------------
// Poisson log-likelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_poisson_loglik(IntegerVector y, NumericVector lambda) {
  double ll = 0.0;
  for (int i = 0; i < y.size(); i++) {
    if (lambda[i] > 0) {
      ll += y[i] * std::log(lambda[i]) - lambda[i] - std::lgamma(y[i] + 1);
    } else if (y[i] > 0) {
      return R_NegInf;
    }
  }
  return ll;
}

// ---------------------------------------------------------------------------
// Binomial log-likelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_binomial_loglik(IntegerVector y, IntegerVector n, NumericVector p) {
  double ll = 0.0;
  for (int i = 0; i < y.size(); i++) {
    if (p[i] > 0 && p[i] < 1) {
      ll += y[i] * std::log(p[i]) + (n[i] - y[i]) * std::log(1 - p[i]);
      ll += std::lgamma(n[i] + 1) - std::lgamma(y[i] + 1) - std::lgamma(n[i] - y[i] + 1);
    } else if ((p[i] <= 0 && y[i] > 0) || (p[i] >= 1 && y[i] < n[i])) {
      return R_NegInf;
    }
  }
  return ll;
}

// ---------------------------------------------------------------------------
// Negative binomial log-likelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_negbin_loglik(IntegerVector y, NumericVector mu, double phi) {
  double ll = 0.0;
  for (int i = 0; i < y.size(); i++) {
    double r = phi;  // size parameter
    double p = r / (r + mu[i]);  // success probability
    ll += std::lgamma(y[i] + r) - std::lgamma(r) - std::lgamma(y[i] + 1);
    ll += r * std::log(p) + y[i] * std::log(1 - p);
  }
  return ll;
}

// ---------------------------------------------------------------------------
// Normal log-likelihood
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double cpp_test_normal_loglik(NumericVector y, NumericVector mu, double sigma) {
  double ll = 0.0;
  double tau = 1.0 / (sigma * sigma);
  for (int i = 0; i < y.size(); i++) {
    double resid = y[i] - mu[i];
    ll += -0.5 * std::log(2 * M_PI) - std::log(sigma) - 0.5 * tau * resid * resid;
  }
  return ll;
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (for testing covariance matrices)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix cpp_test_cholesky(NumericMatrix A) {
  int n = A.nrow();
  NumericMatrix L(n, n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0.0;
      for (int k = 0; k < j; k++) {
        sum += L(i, k) * L(j, k);
      }
      if (i == j) {
        L(i, j) = std::sqrt(A(i, i) - sum);
      } else {
        L(i, j) = (A(i, j) - sum) / L(j, j);
      }
    }
  }

  return L;
}

// ---------------------------------------------------------------------------
// Matrix-vector multiplication
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector cpp_test_matvec(NumericMatrix A, NumericVector x) {
  int n = A.nrow();
  int m = A.ncol();
  NumericVector result(n);

  for (int i = 0; i < n; i++) {
    result[i] = 0.0;
    for (int j = 0; j < m; j++) {
      result[i] += A(i, j) * x[j];
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Autodiff tests - test the actual autodiff.h implementation
// ---------------------------------------------------------------------------

using namespace tulpa::ad;

namespace {
// Shared scaffolding for unary autodiff tests:
// build a single-input Var, apply f, run backward, collect val/adj.
template <class F>
inline List unary_autodiff_test(double x_val, F&& f, double expected_grad) {
  init_tape();
  Var x(x_val);
  Var result = f(x);
  result.backward();
  double value = result.val();
  double grad = x.adj();
  clear_tape();
  return List::create(
    Named("value") = value,
    Named("gradient") = grad,
    Named("expected_gradient") = expected_grad
  );
}
}  // namespace

// [[Rcpp::export]]
List cpp_test_autodiff_gradient(NumericVector x_vals) {
  // Test gradient computation for f(x) = sum(x^2)
  // Gradient should be 2*x
  init_tape();

  int n = x_vals.size();
  std::vector<Var> x(n);

  // Initialize with values
  for (int i = 0; i < n; i++) {
    x[i] = Var(x_vals[i]);
  }

  // Compute f(x) = sum(x^2)
  Var result = Var(0.0);
  for (int i = 0; i < n; i++) {
    result = result + x[i] * x[i];
  }

  // Backward pass
  result.backward();

  // Extract gradients
  NumericVector grads(n);
  for (int i = 0; i < n; i++) {
    grads[i] = x[i].adj();
  }

  double value = result.val();
  clear_tape();

  return List::create(
    Named("value") = value,
    Named("gradient") = grads
  );
}

// [[Rcpp::export]]
List cpp_test_autodiff_exp_chain(double x_val) {
  // Test chain rule: f(x) = exp(x^2), f'(x) = 2x * exp(x^2)
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::exp(x * x); },
    2.0 * x_val * std::exp(x_val * x_val));
}

// [[Rcpp::export]]
List cpp_test_autodiff_log_likelihood(IntegerVector y, NumericVector eta) {
  // Test Poisson log-likelihood gradient w.r.t. linear predictor
  // ll = sum(y * eta - exp(eta))
  // dll/deta = y - exp(eta)
  init_tape();

  int n = y.size();
  std::vector<Var> eta_ad(n);

  for (int i = 0; i < n; i++) {
    eta_ad[i] = Var(eta[i]);
  }

  Var ll = Var(0.0);
  for (int i = 0; i < n; i++) {
    ll = ll + Var((double)y[i]) * eta_ad[i] - tulpa::ad::exp(eta_ad[i]);
  }

  ll.backward();

  NumericVector grads(n);
  NumericVector expected_grads(n);
  for (int i = 0; i < n; i++) {
    grads[i] = eta_ad[i].adj();
    expected_grads[i] = y[i] - std::exp(eta[i]);
  }

  double value = ll.val();
  clear_tape();

  return List::create(
    Named("value") = value,
    Named("gradient") = grads,
    Named("expected_gradient") = expected_grads
  );
}

// [[Rcpp::export]]
List cpp_test_autodiff_division(double a_val, double b_val) {
  // Test division: f(a, b) = a / b
  // df/da = 1/b, df/db = -a/b^2
  init_tape();

  Var a(a_val);
  Var b(b_val);

  Var result = a / b;

  result.backward();

  double value = result.val();
  double grad_a = a.adj();
  double grad_b = b.adj();

  clear_tape();

  return List::create(
    Named("value") = value,
    Named("grad_a") = grad_a,
    Named("grad_b") = grad_b,
    Named("expected_grad_a") = 1.0 / b_val,
    Named("expected_grad_b") = -a_val / (b_val * b_val)
  );
}

// [[Rcpp::export]]
List cpp_test_autodiff_lgamma(double x_val) {
  // d/dx lgamma(x) = digamma(x)
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::lgamma(x); },
    R::digamma(x_val));
}

// [[Rcpp::export]]
List cpp_test_autodiff_softplus(double x_val) {
  // f(x) = log(1+exp(x)), f'(x) = sigmoid(x)
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::softplus(x); },
    1.0 / (1.0 + std::exp(-x_val)));
}

// [[Rcpp::export]]
List cpp_test_autodiff_inv_logit(double x_val) {
  // f(x) = 1/(1+exp(-x)), f'(x) = f(x)*(1-f(x))
  double s = 1.0 / (1.0 + std::exp(-x_val));
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::inv_logit(x); },
    s * (1.0 - s));
}

// ---------------------------------------------------------------------------
// Autodiff: Additional math functions
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
List cpp_test_autodiff_log(double x_val) {
  // f(x) = log(x), f'(x) = 1/x
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::log(x); },
    1.0 / x_val);
}

// [[Rcpp::export]]
List cpp_test_autodiff_sqrt(double x_val) {
  // f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::sqrt(x); },
    1.0 / (2.0 * std::sqrt(x_val)));
}

// [[Rcpp::export]]
List cpp_test_autodiff_pow(double x_val, double p) {
  // f(x) = x^p, f'(x) = p * x^(p-1)
  return unary_autodiff_test(x_val,
    [p](Var x) { return tulpa::ad::pow(x, p); },
    p * std::pow(x_val, p - 1.0));
}

// [[Rcpp::export]]
List cpp_test_autodiff_log1p(double x_val) {
  // f(x) = log(1+x), f'(x) = 1/(1+x)
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::log1p(x); },
    1.0 / (1.0 + x_val));
}

// [[Rcpp::export]]
List cpp_test_autodiff_log_sum_exp(double a_val, double b_val) {
  // Test log_sum_exp: f(a,b) = log(exp(a) + exp(b))
  // df/da = exp(a) / (exp(a) + exp(b)) = softmax(a)
  // df/db = exp(b) / (exp(a) + exp(b)) = softmax(b)
  init_tape();

  Var a(a_val);
  Var b(b_val);
  Var result = tulpa::ad::log_sum_exp(a, b);

  result.backward();

  double value = result.val();
  double grad_a = a.adj();
  double grad_b = b.adj();

  // Expected gradients (softmax)
  double max_val = std::max(a_val, b_val);
  double sum_exp = std::exp(a_val - max_val) + std::exp(b_val - max_val);
  double expected_a = std::exp(a_val - max_val) / sum_exp;
  double expected_b = std::exp(b_val - max_val) / sum_exp;

  clear_tape();

  return List::create(
    Named("value") = value,
    Named("grad_a") = grad_a,
    Named("grad_b") = grad_b,
    Named("expected_grad_a") = expected_a,
    Named("expected_grad_b") = expected_b
  );
}

// [[Rcpp::export]]
List cpp_test_autodiff_logit(double x_val) {
  // f(x) = log(x/(1-x)), f'(x) = 1/(x*(1-x))
  return unary_autodiff_test(x_val,
    [](Var x) { return tulpa::ad::logit(x); },
    1.0 / (x_val * (1.0 - x_val)));
}

// ---------------------------------------------------------------------------
// Autodiff: Compound expressions (test gradient accumulation)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
List cpp_test_autodiff_negbin_loglik(IntegerVector y, NumericVector mu, double phi) {
  // Test negative binomial log-likelihood gradient
  // ll = sum( lgamma(y + phi) - lgamma(phi) - lgamma(y + 1)
  //           + phi * log(phi/(phi+mu)) + y * log(mu/(phi+mu)) )
  init_tape();

  int n = y.size();
  std::vector<Var> mu_ad(n);
  for (int i = 0; i < n; i++) {
    mu_ad[i] = Var(mu[i]);
  }
  Var phi_ad(phi);

  Var ll = Var(0.0);
  for (int i = 0; i < n; i++) {
    // lgamma terms (y and phi are treated as constants for gradient purposes)
    ll = ll + tulpa::ad::lgamma(Var((double)y[i]) + phi_ad);
    ll = ll - tulpa::ad::lgamma(phi_ad);

    // phi * log(phi/(phi+mu)) + y * log(mu/(phi+mu))
    Var rate = phi_ad + mu_ad[i];
    ll = ll + phi_ad * tulpa::ad::log(phi_ad / rate);
    ll = ll + Var((double)y[i]) * tulpa::ad::log(mu_ad[i] / rate);
  }

  ll.backward();

  NumericVector grads_mu(n);
  for (int i = 0; i < n; i++) {
    grads_mu[i] = mu_ad[i].adj();
  }
  double grad_phi = phi_ad.adj();
  double value = ll.val();

  clear_tape();

  return List::create(
    Named("value") = value,
    Named("gradient_mu") = grads_mu,
    Named("gradient_phi") = grad_phi
  );
}

// ---------------------------------------------------------------------------
// Laplace core likelihood functions (from laplace_core.cpp)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
List cpp_test_laplace_binomial(int y, int n, double eta) {
  double ll = tulpa::log_lik_binomial(y, n, eta);
  double grad = tulpa::grad_log_lik_binomial(y, n, eta);
  double neg_hess = tulpa::neg_hess_log_lik_binomial(y, n, eta);

  return List::create(
    Named("log_lik") = ll,
    Named("gradient") = grad,
    Named("neg_hessian") = neg_hess
  );
}

// [[Rcpp::export]]
List cpp_test_laplace_negbin(int y, double eta, double phi) {
  double ll = tulpa::log_lik_negbin(y, eta, phi);
  double grad = tulpa::grad_log_lik_negbin(y, eta, phi);
  double neg_hess = tulpa::neg_hess_log_lik_negbin(y, eta, phi);

  return List::create(
    Named("log_lik") = ll,
    Named("gradient") = grad,
    Named("neg_hessian") = neg_hess
  );
}

// [[Rcpp::export]]
List cpp_test_laplace_poisson(int y, double eta) {
  double ll = tulpa::log_lik_poisson(y, eta);
  double grad = tulpa::grad_log_lik_poisson(y, eta);
  double neg_hess = tulpa::neg_hess_log_lik_poisson(y, eta);

  return List::create(
    Named("log_lik") = ll,
    Named("gradient") = grad,
    Named("neg_hessian") = neg_hess
  );
}

// ---------------------------------------------------------------------------
// PG Binomial helper functions (from pg_binomial.cpp)
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector cpp_test_pg_update_beta(
    NumericVector kappa,
    NumericVector omega,
    NumericMatrix X,
    NumericVector re_contrib,
    double prior_sd
) {
  return tulpa::update_beta(kappa, omega, X, re_contrib, prior_sd);
}

// [[Rcpp::export]]
NumericVector cpp_test_pg_update_re(
    NumericVector kappa,
    NumericVector omega,
    NumericVector X_beta,
    IntegerVector group,
    int n_groups,
    double sigma_re
) {
  return tulpa::update_re(kappa, omega, X_beta, group, n_groups, sigma_re);
}

// [[Rcpp::export]]
double cpp_test_pg_update_sigma_re(NumericVector re, double scale) {
  return tulpa::update_sigma_re(re, scale);
}

// ---------------------------------------------------------------------------
// linalg_fast.h test wrappers
// ---------------------------------------------------------------------------

#include "linalg_fast.h"

// [[Rcpp::export]]
double cpp_test_dot_product(NumericVector x, NumericVector y) {
  return tulpa_linalg::dot_product(x.begin(), y.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_norm_squared(NumericVector x) {
  return tulpa_linalg::norm_squared(x.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_vector_sum(NumericVector x) {
  return tulpa_linalg::vector_sum(x.begin(), x.size());
}

// [[Rcpp::export]]
NumericVector cpp_test_axpy(double a, NumericVector x, NumericVector y) {
  NumericVector result = clone(y);
  tulpa_linalg::axpy(a, x.begin(), result.begin(), x.size());
  return result;
}

// [[Rcpp::export]]
NumericVector cpp_test_scale(double a, NumericVector x) {
  NumericVector result = clone(x);
  tulpa_linalg::scale(a, result.begin(), x.size());
  return result;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec(NumericMatrix X, NumericVector beta) {
  int N = X.nrow();
  int p = X.ncol();
  std::vector<double> X_flat(N * p);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p; j++) {
      X_flat[i * p + j] = X(i, j);
    }
  }
  NumericVector y(N);
  tulpa_linalg::matvec(X_flat.data(), beta.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec_add(NumericMatrix X, NumericVector beta, NumericVector y_init) {
  int N = X.nrow();
  int p = X.ncol();
  std::vector<double> X_flat(N * p);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p; j++) {
      X_flat[i * p + j] = X(i, j);
    }
  }
  NumericVector y = clone(y_init);
  tulpa_linalg::matvec_add(X_flat.data(), beta.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
NumericVector cpp_test_linalg_matvec_transpose(NumericMatrix X, NumericVector x) {
  int N = X.nrow();
  int p = X.ncol();
  std::vector<double> X_flat(N * p);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p; j++) {
      X_flat[i * p + j] = X(i, j);
    }
  }
  NumericVector y(p);
  tulpa_linalg::matvec_transpose(X_flat.data(), x.begin(), y.begin(), N, p);
  return y;
}

// [[Rcpp::export]]
double cpp_test_sparse_laplacian_quadform(IntegerVector row_ptr, IntegerVector col_idx, NumericVector x) {
  return tulpa_linalg::sparse_laplacian_quadform(row_ptr.begin(), col_idx.begin(), x.begin(), x.size());
}

// [[Rcpp::export]]
double cpp_test_linalg_log_sum_exp(double a, double b) {
  return tulpa_linalg::log_sum_exp(a, b);
}

// [[Rcpp::export]]
double cpp_test_linalg_log_sum_exp_vec(NumericVector x) {
  return tulpa_linalg::log_sum_exp_vec(x.begin(), x.size());
}

// [[Rcpp::export]]
NumericVector cpp_test_softmax_inplace(NumericVector x) {
  NumericVector result = clone(x);
  tulpa_linalg::softmax_inplace(result.begin(), result.size());
  return result;
}

// [[Rcpp::export]]
List cpp_test_compute_linear_predictors(
    NumericMatrix X_num, NumericVector beta_num,
    NumericMatrix X_denom, NumericVector beta_denom,
    int n_threads
) {
  int N = X_num.nrow();
  int p_num = X_num.ncol();
  int p_denom = X_denom.ncol();

  std::vector<double> X_num_flat(N * p_num);
  std::vector<double> X_denom_flat(N * p_denom);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p_num; j++) {
      X_num_flat[i * p_num + j] = X_num(i, j);
    }
    for (int j = 0; j < p_denom; j++) {
      X_denom_flat[i * p_denom + j] = X_denom(i, j);
    }
  }

  NumericVector eta_num(N);
  NumericVector eta_denom(N);

  tulpa_linalg::compute_linear_predictors(
    X_num_flat.data(), beta_num.begin(), p_num,
    X_denom_flat.data(), beta_denom.begin(), p_denom,
    eta_num.begin(), eta_denom.begin(), N, n_threads
  );

  return List::create(
    Named("eta_num") = eta_num,
    Named("eta_denom") = eta_denom
  );
}

// ---------------------------------------------------------------------------
// hmc_temporal.h test wrappers
// ---------------------------------------------------------------------------

#include "hmc_temporal.h"

// [[Rcpp::export]]
double cpp_test_rw1_quadratic_form(NumericVector phi, bool cyclic) {
  return tulpa_temporal::rw1_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_rw2_quadratic_form(NumericVector phi, bool cyclic) {
  return tulpa_temporal::rw2_quadratic_form(phi.begin(), phi.size(), cyclic);
}

// [[Rcpp::export]]
double cpp_test_ar1_log_density(NumericVector phi, double rho, double tau) {
  return tulpa_temporal::ar1_log_density(phi.begin(), phi.size(), rho, tau);
}

// [[Rcpp::export]]
double cpp_test_temporal_log_prior(
    NumericVector phi,
    std::string type_str,
    double tau,
    double rho,
    bool cyclic
) {
  tulpa_temporal::TemporalType type;
  if (type_str == "rw1") {
    type = tulpa_temporal::TemporalType::RW1;
  } else if (type_str == "rw2") {
    type = tulpa_temporal::TemporalType::RW2;
  } else if (type_str == "ar1") {
    type = tulpa_temporal::TemporalType::AR1;
  } else {
    type = tulpa_temporal::TemporalType::NONE;
  }

  return tulpa_temporal::temporal_log_prior(
    phi.begin(), phi.size(), type, tau, rho, cyclic
  );
}

// [[Rcpp::export]]
double cpp_test_sum_to_zero_penalty(NumericVector phi, double lambda) {
  return tulpa_temporal::sum_to_zero_penalty(phi.begin(), phi.size(), lambda);
}

// ---------------------------------------------------------------------------
// hmc_zi.h test wrappers
// ---------------------------------------------------------------------------

#include "hmc_zi.h"

// [[Rcpp::export]]
double cpp_test_log1pexp(double x) {
  return tulpa_zi::log1pexp(x);
}

// [[Rcpp::export]]
double cpp_test_zi_logistic(double x) {
  return tulpa_zi::logistic(x);
}

// [[Rcpp::export]]
double cpp_test_log_logistic(double x) {
  return tulpa_zi::log_logistic(x);
}

// [[Rcpp::export]]
double cpp_test_log1m_logistic(double x) {
  return tulpa_zi::log1m_logistic(x);
}

// [[Rcpp::export]]
double cpp_test_zi_poisson_lpmf(int y, double mu) {
  return tulpa_zi::poisson_lpmf(y, mu);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_lpmf(int y, double mu, double phi) {
  return tulpa_zi::negbin_lpmf(y, mu, phi);
}

// [[Rcpp::export]]
double cpp_test_zi_poisson_lpmf_logit(int y, double mu, double logit_zi) {
  return tulpa_zi::zi_poisson_lpmf_logit(y, mu, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_lpmf_logit(int y, double mu, double phi, double logit_zi) {
  return tulpa_zi::zi_negbin_lpmf_logit(y, mu, phi, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_truncated_poisson_lpmf(int y, double mu) {
  return tulpa_zi::truncated_poisson_lpmf(y, mu);
}

// [[Rcpp::export]]
double cpp_test_truncated_negbin_lpmf(int y, double mu, double phi) {
  return tulpa_zi::truncated_negbin_lpmf(y, mu, phi);
}

// [[Rcpp::export]]
double cpp_test_hurdle_poisson_lpmf_logit(int y, double mu, double logit_theta) {
  return tulpa_zi::hurdle_poisson_lpmf_logit(y, mu, logit_theta);
}

// [[Rcpp::export]]
double cpp_test_hurdle_negbin_lpmf_logit(int y, double mu, double phi, double logit_theta) {
  return tulpa_zi::hurdle_negbin_lpmf_logit(y, mu, phi, logit_theta);
}

// [[Rcpp::export]]
double cpp_test_zi_log_likelihood(int y, double mu, double phi, double logit_zi, std::string zi_type_str) {
  tulpa_zi::ZIType zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  return tulpa_zi::zi_log_likelihood(y, mu, phi, logit_zi, zi_type);
}

// [[Rcpp::export]]
double cpp_test_zi_poisson_grad_logit_zi(int y, double mu, double logit_zi) {
  return tulpa_zi::zi_poisson_grad_logit_zi(y, mu, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_zi_negbin_grad_logit_zi(int y, double mu, double phi, double logit_zi) {
  return tulpa_zi::zi_negbin_grad_logit_zi(y, mu, phi, logit_zi);
}

// [[Rcpp::export]]
double cpp_test_hurdle_grad_logit_theta(int y, double logit_theta) {
  return tulpa_zi::hurdle_grad_logit_theta(y, logit_theta);
}

// ---------------------------------------------------------------------------
// OpenMP parallel execution tests
// ---------------------------------------------------------------------------

#ifdef _OPENMP
#include <omp.h>
#endif

// cpp_test_get_max_threads removed — use cpp_get_max_threads (hmc_sampler.cpp)

// [[Rcpp::export]]
List cpp_test_parallel_dot_products(NumericMatrix X, NumericVector y, int n_threads) {
  // Test OpenMP parallel reduction with multiple dot products
  int N = X.nrow();
  int p = X.ncol();

  std::vector<double> X_flat(N * p);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < p; j++) {
      X_flat[i * p + j] = X(i, j);
    }
  }

  NumericVector results(N);
  double total_sum = 0.0;

#ifdef _OPENMP
  #pragma omp parallel for reduction(+:total_sum) schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < N; i++) {
    double dot = tulpa_linalg::dot_product(&X_flat[i * p], y.begin(), p);
    results[i] = dot;
    total_sum += dot;
  }

  return List::create(
    Named("results") = results,
    Named("total_sum") = total_sum
  );
}

// [[Rcpp::export]]
List cpp_test_parallel_likelihood(
    IntegerVector y,
    NumericVector mu,
    int n_threads
) {
  // Test OpenMP parallel reduction for likelihood computation
  int N = y.size();
  double log_lik = 0.0;

#ifdef _OPENMP
  #pragma omp parallel for reduction(+:log_lik) schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < N; i++) {
    // Poisson log-likelihood
    if (mu[i] > 0) {
      log_lik += y[i] * std::log(mu[i]) - mu[i] - std::lgamma(y[i] + 1);
    }
  }

  return List::create(
    Named("log_lik") = log_lik,
    Named("n_threads_requested") = n_threads
  );
}

// Test for thread-local storage correctness
// [[Rcpp::export]]
NumericVector cpp_test_parallel_independent(int n, int n_threads) {
  // Each thread computes independent values, then we verify results
  NumericVector results(n);

#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads)
#endif
  for (int i = 0; i < n; i++) {
    // Compute a deterministic function that doesn't depend on shared state
    results[i] = std::sin(i * 0.1) * std::cos(i * 0.2) + i;
  }

  return results;
}

// ---------------------------------------------------------------------------
// NNGP solver dispatch test wrapper
// ---------------------------------------------------------------------------
//
// Computes the centered NNGP log-likelihood and full hyperparameter gradients
// at a given (w, sigma2, phi) using a chosen solver ("cholesky", "cg", "pcg").
// Used by tests/testthat/test-gp-cg.R to verify the CG path agrees with the
// Cholesky reference within numerical tolerance.
//
// Inputs use 1-based nn_idx / nn_order matching the rest of the package.
//
// [[Rcpp::export]]
List cpp_test_gp_solver_dispatch(
    NumericVector w,
    double sigma2,
    double phi,
    NumericMatrix coords,
    IntegerMatrix nn_idx,
    NumericMatrix nn_dist,
    NumericVector nn_neighbor_dist,  // length N*nn*nn, row-major
    IntegerVector nn_order,
    IntegerVector nn_order_inv,
    int cov_type,
    std::string solver,
    double cg_tol,
    int cg_maxiter
) {
  int N = coords.nrow();
  int nn = nn_idx.ncol();

  tulpa::GPData gp;
  gp.n_obs = N;
  gp.nn = nn;
  gp.coords.resize(N * 2);
  for (int i = 0; i < N; i++) {
    gp.coords[i * 2 + 0] = coords(i, 0);
    gp.coords[i * 2 + 1] = coords(i, 1);
  }
  gp.nn_idx.resize(N * nn);
  gp.nn_dist.resize(N * nn);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < nn; j++) {
      gp.nn_idx[i * nn + j] = nn_idx(i, j);
      gp.nn_dist[i * nn + j] = nn_dist(i, j);
    }
  }
  gp.nn_neighbor_dist.assign(nn_neighbor_dist.begin(), nn_neighbor_dist.end());
  gp.nn_order.assign(nn_order.begin(), nn_order.end());
  gp.nn_order_inv.assign(nn_order_inv.begin(), nn_order_inv.end());
  gp.cov_type = static_cast<tulpa::CovType>(cov_type);

  gp.solver_config.solver = tulpa_gp::parse_gp_solver(solver);
  gp.solver_config.cg_tol = cg_tol;
  gp.solver_config.cg_maxiter = cg_maxiter;
  gp.solver_config.n_obs = N;

  std::vector<double> w_vec(w.begin(), w.end());
  double ll = tulpa_gp::gp_nngp_log_lik(w_vec, sigma2, phi, gp);

  tulpa_gp::NNGPGradients grads;
  tulpa_gp::gp_nngp_gradients(w_vec, sigma2, phi, gp, grads);

  NumericVector grad_w_out(grads.grad_w.begin(), grads.grad_w.end());

  return List::create(
    _["log_lik"] = ll,
    _["grad_w"] = grad_w_out,
    _["grad_log_sigma2"] = grads.grad_log_sigma2,
    _["grad_log_phi"] = grads.grad_log_phi,
    _["solver"] = solver
  );
}
