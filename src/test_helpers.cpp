// test_helpers.cpp
// Rcpp wrappers to expose internal C++ functions for unit testing

#include <Rcpp.h>
#include <RcppEigen.h>
#include <cmath>
#include <vector>
#include <random>
#include "autodiff.h"
#include "tulpa/autodiff_arena.h"
#include "tulpa/autodiff_fwd.h"
#include "autodiff_utils.h"
#include "spde_nc_transform.h"
#include "spde_qbuilder.h"
#include "laplace_core.h"
#include "pg_binomial.h"
#include "pg_shared.h"
#include "pg_spatial.h"
#include "hmc_gp.h"
#include "pc_prior.h"
#include "sparse_hessian.h"
#include "sparse_cholesky.h"
#include "mcar_block_factory.h"
#include "gpu_nngp_laplace.h"
#include "laplace_spatial_priors.h"
#include "mem_budget.h"
#include "hmc_sampler.h"           // require_spatial_partition

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
  if (x.size() == 0) return NumericVector(0);

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
  if (lambda.size() < y.size()) {
    Rcpp::stop("length(lambda) (%d) must be at least length(y) (%d).",
               (int)lambda.size(), (int)y.size());
  }
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
  if (n.size() < y.size() || p.size() < y.size()) {
    Rcpp::stop("length(n) (%d) and length(p) (%d) must each be at least "
               "length(y) (%d).",
               (int)n.size(), (int)p.size(), (int)y.size());
  }
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
  if (mu.size() < y.size()) {
    Rcpp::stop("length(mu) (%d) must be at least length(y) (%d).",
               (int)mu.size(), (int)y.size());
  }
  double ll = 0.0;
  for (int i = 0; i < y.size(); i++) {
    double r = phi;  // size parameter
    // A negative binomial with mean 0 puts all its mass on y = 0, so the
    // boundary is answered before the log: p is exactly 1 there, and
    // y * log(1 - p) is 0 * -Inf.
    if (mu[i] <= 0.0) {
      if (y[i] > 0) return R_NegInf;
      continue;
    }
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
  if (mu.size() < y.size()) {
    Rcpp::stop("length(mu) (%d) must be at least length(y) (%d).",
               (int)mu.size(), (int)y.size());
  }
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
  if (A.ncol() != n) {
    Rcpp::stop("A must be square (got %d x %d).", n, (int)A.ncol());
  }
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
  if ((int)x.size() != m) {
    Rcpp::stop("length(x) (%d) must equal ncol(A) (%d).", (int)x.size(), m);
  }
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
  tulpa::PgScaleState st = tulpa::pg_scale_state_init(scale);
  tulpa::pg_update_scale_halfcauchy(re, scale, st);
  return st.sigma;
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
#include "hmc_temporal_multiscale.h"

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

  return tulpa_temporal::log_prior_temporal(
    phi.begin(), phi.size(), type, tau, rho, cyclic
  );
}

// Single-source intrinsic-GMRF rank normalizer (rw1_rank / rw2_rank), consumed
// by both the production templated log-posterior (compute_temporal_prior) and
// the double twin. Exposed so a test can pin the cyclic rank (T-1 for RW1 and
// RW2) that both paths depend on.
// [[Rcpp::export]]
int cpp_test_temporal_rank(std::string type_str, int T_len, bool cyclic) {
  if (type_str == "rw1") return tulpa_temporal::rw1_rank(T_len, cyclic);
  if (type_str == "rw2") return tulpa_temporal::rw2_rank(T_len, cyclic);
  return 0;
}

// [[Rcpp::export]]
double cpp_test_sum_to_zero_penalty(NumericVector phi) {
  return tulpa_temporal::sum_to_zero_penalty(phi.begin(), phi.size());
}

// [[Rcpp::export]]
double cpp_test_s2z_precision(int n, double kappa) {
  return tulpa::s2z_precision(n, kappa);
}

// Multiscale temporal log-prior probe. The multiscale block is populated only
// by consumer packages (has_multiscale_temporal is never set on tulpa's own
// front door), so this is the only place tulpa can assert that its intrinsic
// trend and seasonal arms are pinned to sum-to-zero (gcol33/tulpa#241).
// [[Rcpp::export]]
double cpp_test_multiscale_temporal_log_lik(
    NumericVector trend, NumericVector seasonal, NumericVector short_term,
    double sigma2_trend, double sigma2_seasonal, double sigma2_short,
    double rho_short, std::string trend_type, int seasonal_period,
    std::string short_term_type
) {
  // On the sampler path rho is 2 * inv_logit(logit_rho) - 1, so |rho| < 1 by
  // construction. This entry point takes it raw, and the AR1 density is the
  // stationary one: outside the unit interval its 1 - rho^2 factor is at its
  // floor and the number returned is not the density of anything.
  if (!R_finite(rho_short) || std::fabs(rho_short) >= 1.0) {
    Rcpp::stop("cpp_test_multiscale_temporal_log_lik: rho_short (%g) must be "
               "finite and satisfy |rho_short| < 1.", rho_short);
  }

  auto as_type = [](const std::string& s) {
    if (s == "rw1") return tulpa::TemporalType::RW1;
    if (s == "rw2") return tulpa::TemporalType::RW2;
    if (s == "ar1") return tulpa::TemporalType::AR1;
    if (s == "iid") return tulpa::TemporalType::IID;
    return tulpa::TemporalType::NONE;
  };

  tulpa::MultiscaleTemporalData md;
  md.n_times = trend.size();
  md.trend_type = as_type(trend_type);
  md.seasonal_period = seasonal_period;
  md.short_term_type = as_type(short_term_type);

  std::vector<double> tr(trend.begin(), trend.end());
  std::vector<double> se(seasonal.begin(), seasonal.end());
  std::vector<double> st(short_term.begin(), short_term.end());

  return tulpa_temporal::multiscale_temporal_log_lik(
      tr, se, st, sigma2_trend, sigma2_seasonal, sigma2_short, rho_short, md);
}

// ---------------------------------------------------------------------------
// OpenMP parallel execution tests
// ---------------------------------------------------------------------------

#ifdef _OPENMP
#include <omp.h>
#endif

#include "omp_threads.h"

// These three probes run in R CMD check, where _R_CHECK_LIMIT_CORES_ caps the
// permitted team at two, so each takes its requested count through
// tulpa_omp_team_size_req rather than as a num_threads clause of its own. The
// requested count is still reported, so a test can see request and team apart.

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
  #pragma omp parallel for reduction(+:total_sum) schedule(static) num_threads(tulpa_omp_team_size_req(n_threads, N))
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
  #pragma omp parallel for reduction(+:log_lik) schedule(static) num_threads(tulpa_omp_team_size_req(n_threads, N))
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
  #pragma omp parallel for schedule(static) num_threads(tulpa_omp_team_size_req(n_threads, n))
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
  tulpa_linalg::require_coords_2col(coords, "the GPData test helper");
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

// ---------------------------------------------------------------------------
// Arena custom_backward smoke test
// ---------------------------------------------------------------------------
//
// Computes the same loss two ways on the same (x, y):
//
//   a = x * y
//   b = x + y
//   L = a^2 + b^2
//
// Path A ("standard"): everything via the per-op SoA nodes (multiply, add,
// square). Path B ("custom"): wraps (x, y) -> (a, b) in a single
// add_custom_backward block whose adjoint callback does
//   dx += y * da + 1 * db
//   dy += x * da + 1 * db
// then reuses the standard path for L = a^2 + b^2 on the CB outputs.
//
// The two paths must agree on L and on dL/dx, dL/dy. This exercises every
// branch of the modified Arena::backward(): standard binary, custom
// stash-only, and custom trigger.
// [[Rcpp::export]]
List cpp_test_arena_custom_backward(double x_val, double y_val) {
  using tulpa::arena::Arena;
  using tulpa::arena::ArenaScope;
  using tulpa::arena::Var;
  using tulpa::arena::CustomBackwardFn;

  // ----- Path A: standard arena ops only -----
  double L_std = 0.0, dx_std = 0.0, dy_std = 0.0;
  {
    ArenaScope scope;
    Arena* ar = scope.arena();
    Var x(ar, x_val);
    Var y(ar, y_val);
    Var a = x * y;
    Var b = x + y;
    Var L = a * a + b * b;
    L.backward();
    L_std  = L.val();
    dx_std = x.adj();
    dy_std = y.adj();
  }

  // ----- Path B: (x, y) -> (a, b) via add_custom_backward -----
  double L_cb = 0.0, dx_cb = 0.0, dy_cb = 0.0;
  {
    ArenaScope scope;
    Arena* ar = scope.arena();
    Var x(ar, x_val);
    Var y(ar, y_val);

    std::vector<int32_t> in_idx = {x.idx_, y.idx_};
    std::vector<double>  out_vals = {x_val * y_val, x_val + y_val};

    CustomBackwardFn fn = [](
      const double* input_vals,  int /*n_in*/,
      const double* /*output_vals*/, int /*n_out*/,
      const double* output_adjs,
      double* input_adjs
    ) {
      const double xv = input_vals[0];
      const double yv = input_vals[1];
      const double da = output_adjs[0];   // dL/da
      const double db = output_adjs[1];   // dL/db
      // a = x * y  -> da/dx = y, da/dy = x
      // b = x + y  -> db/dx = 1, db/dy = 1
      input_adjs[0] = yv * da + 1.0 * db;
      input_adjs[1] = xv * da + 1.0 * db;
    };

    std::vector<int32_t> out_idx;
    ar->add_custom_backward(in_idx, out_vals, fn, out_idx);

    Var a; a.arena_ = ar; a.idx_ = out_idx[0];
    Var b; b.arena_ = ar; b.idx_ = out_idx[1];
    Var L = a * a + b * b;
    L.backward();
    L_cb  = L.val();
    dx_cb = x.adj();
    dy_cb = y.adj();
  }

  // Analytical reference: L(x,y) = (xy)^2 + (x+y)^2
  // dL/dx = 2*x*y^2 + 2(x+y)
  // dL/dy = 2*x^2*y + 2(x+y)
  const double dx_an = 2.0 * x_val * y_val * y_val + 2.0 * (x_val + y_val);
  const double dy_an = 2.0 * x_val * x_val * y_val + 2.0 * (x_val + y_val);
  const double L_an  = (x_val * y_val) * (x_val * y_val) +
                       (x_val + y_val) * (x_val + y_val);

  return List::create(
    _["L_std"]   = L_std,    _["L_cb"]   = L_cb,    _["L_analytical"]   = L_an,
    _["dx_std"]  = dx_std,   _["dx_cb"]  = dx_cb,   _["dx_analytical"]  = dx_an,
    _["dy_std"]  = dy_std,   _["dy_cb"]  = dy_cb,   _["dy_analytical"]  = dy_an
  );
}

// ---------------------------------------------------------------------------
// SPDE non-centered transform: gradient vs central differences
// ---------------------------------------------------------------------------
//
// Builds an SPDE field on the supplied (C0_diag, G1) FEM matrices, computes
// w = L^{-T} z under Q(kappa, tau) via the new arena custom_backward block,
// and a scalar loss L(z, log_kappa, log_tau) = sum(w^2). Reverse-mode arena
// gradients are then compared against central differences in (z[i], log_kappa,
// log_tau). Returns both vectors so the testthat layer can assert agreement.
// [[Rcpp::export]]
List cpp_test_spde_nc_transform_grad(
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x,
    Rcpp::IntegerVector G1_i,
    Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector z_init,
    double log_kappa_val,
    double log_tau_val,
    double fd_eps = 1e-5,
    Rcpp::Nullable<Rcpp::NumericVector> poles_nullable   = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
  using tulpa::arena::Arena;
  using tulpa::arena::ArenaScope;
  using tulpa::arena::Var;
  using tulpa::SpdeNcTransform;

  const int n = z_init.size();

  std::vector<double> C0_d(C0_diag.begin(), C0_diag.end());
  std::vector<double> G1_xv(G1_x.begin(), G1_x.end());
  std::vector<int>    G1_iv(G1_i.begin(), G1_i.end());
  std::vector<int>    G1_pv(G1_p.begin(), G1_p.end());

  // Rational coefficients (optional). Empty -> integer alpha=2.
  std::vector<double> poles_v, weights_v;
  if (poles_nullable.isNotNull() && weights_nullable.isNotNull()) {
    poles_v   = Rcpp::as<std::vector<double>>(poles_nullable);
    weights_v = Rcpp::as<std::vector<double>>(weights_nullable);
  }

  // Reverse-mode arena gradient.
  std::vector<double> grad_z_arena(n, 0.0);
  double grad_log_kappa_arena = 0.0;
  double grad_log_tau_arena   = 0.0;
  double L_arena = 0.0;
  {
    SpdeNcTransform transform;
    transform.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);

    ArenaScope scope;
    Arena* ar = scope.arena();
    std::vector<Var> z_v(n);
    for (int i = 0; i < n; i++) z_v[i] = Var(ar, z_init[i]);
    Var log_kappa_v(ar, log_kappa_val);
    Var log_tau_v  (ar, log_tau_val);

    std::vector<Var> w_v = tulpa::spde_nc_transform_arena(
        ar, z_v, log_kappa_v, log_tau_v, transform);

    // Loss: sum(w^2).
    Var L(ar, 0.0);
    for (int i = 0; i < n; i++) L = L + w_v[i] * w_v[i];

    L.backward();
    L_arena = L.val();
    for (int i = 0; i < n; i++) grad_z_arena[i] = z_v[i].adj();
    grad_log_kappa_arena = log_kappa_v.adj();
    grad_log_tau_arena   = log_tau_v.adj();
  }

  // Central-difference reference. Each evaluation rebuilds the transform on
  // a fresh ArenaScope, runs the forward, and reads sum(w^2) directly.
  auto eval_loss = [&](const std::vector<double>& z_pt,
                       double log_k, double log_t) -> double {
    SpdeNcTransform transform;
    transform.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    Eigen::VectorXd z_e(n);
    for (int i = 0; i < n; i++) z_e[i] = z_pt[i];
    Eigen::VectorXd w = transform.forward(z_e, std::exp(log_k), std::exp(log_t));
    return w.squaredNorm();
  };

  std::vector<double> z_pt(z_init.begin(), z_init.end());

  std::vector<double> grad_z_fd(n, 0.0);
  for (int i = 0; i < n; i++) {
    z_pt[i] += fd_eps;
    double fp = eval_loss(z_pt, log_kappa_val, log_tau_val);
    z_pt[i] -= 2 * fd_eps;
    double fm = eval_loss(z_pt, log_kappa_val, log_tau_val);
    z_pt[i] += fd_eps;
    grad_z_fd[i] = (fp - fm) / (2.0 * fd_eps);
  }

  double fp = eval_loss(z_pt, log_kappa_val + fd_eps, log_tau_val);
  double fm = eval_loss(z_pt, log_kappa_val - fd_eps, log_tau_val);
  double grad_log_kappa_fd = (fp - fm) / (2.0 * fd_eps);

  fp = eval_loss(z_pt, log_kappa_val, log_tau_val + fd_eps);
  fm = eval_loss(z_pt, log_kappa_val, log_tau_val - fd_eps);
  double grad_log_tau_fd = (fp - fm) / (2.0 * fd_eps);

  return List::create(
    _["L"]                    = L_arena,
    _["grad_z_arena"]         = NumericVector(grad_z_arena.begin(),  grad_z_arena.end()),
    _["grad_z_fd"]            = NumericVector(grad_z_fd.begin(),     grad_z_fd.end()),
    _["grad_log_kappa_arena"] = grad_log_kappa_arena,
    _["grad_log_kappa_fd"]    = grad_log_kappa_fd,
    _["grad_log_tau_arena"]   = grad_log_tau_arena,
    _["grad_log_tau_fd"]      = grad_log_tau_fd
  );
}

// ---------------------------------------------------------------------------
// SPDE non-centered transform: forward-mode tangent vs central differences
// ---------------------------------------------------------------------------
//
// Validates SpdeNcTransform::forward_with_tangent, the kernel powering the
// fwd::Dual instantiation of compute_log_post_generic for joint-NUTS SPDE.
// For each scalar input axis (z[0]..z[n-1], log_kappa, log_tau) we seed a
// unit tangent, run forward_with_tangent, and compare the resulting dw
// against central differences of forward() along the same axis.
// [[Rcpp::export]]
List cpp_test_spde_nc_transform_fwd(
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x,
    Rcpp::IntegerVector G1_i,
    Rcpp::IntegerVector G1_p,
    Rcpp::NumericVector z_init,
    double log_kappa_val,
    double log_tau_val,
    double fd_eps = 1e-5,
    Rcpp::Nullable<Rcpp::NumericVector> poles_nullable   = R_NilValue,
    Rcpp::Nullable<Rcpp::NumericVector> weights_nullable = R_NilValue
) {
  using tulpa::SpdeNcTransform;
  const int n = z_init.size();

  std::vector<double> C0_d(C0_diag.begin(), C0_diag.end());
  std::vector<double> G1_xv(G1_x.begin(), G1_x.end());
  std::vector<int>    G1_iv(G1_i.begin(), G1_i.end());
  std::vector<int>    G1_pv(G1_p.begin(), G1_p.end());

  std::vector<double> poles_v, weights_v;
  if (poles_nullable.isNotNull() && weights_nullable.isNotNull()) {
    poles_v   = Rcpp::as<std::vector<double>>(poles_nullable);
    weights_v = Rcpp::as<std::vector<double>>(weights_nullable);
  }

  const double kappa = std::exp(log_kappa_val);
  const double tau   = std::exp(log_tau_val);

  Eigen::VectorXd z(n);
  for (int i = 0; i < n; i++) z[i] = z_init[i];

  Eigen::VectorXd w_baseline;
  {
    SpdeNcTransform tx;
    tx.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    w_baseline = tx.forward(z, kappa, tau);
  }

  auto forward_at = [&](const Eigen::VectorXd& z_pt,
                        double log_k, double log_t) -> Eigen::VectorXd {
    SpdeNcTransform tx;
    tx.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    return tx.forward(z_pt, std::exp(log_k), std::exp(log_t));
  };

  Rcpp::NumericMatrix dw_fwd(n, n + 2);
  Rcpp::NumericMatrix dw_fd (n, n + 2);

  for (int i = 0; i < n; i++) {
    SpdeNcTransform tx;
    tx.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    Eigen::VectorXd dz = Eigen::VectorXd::Zero(n);
    dz[i] = 1.0;
    Eigen::VectorXd w_v, dw_v;
    tx.forward_with_tangent(z, dz, kappa, 0.0, tau, 0.0, w_v, dw_v);
    for (int k = 0; k < n; k++) dw_fwd(k, i) = dw_v[k];

    Eigen::VectorXd z_pt = z;
    z_pt[i] += fd_eps;
    Eigen::VectorXd wp = forward_at(z_pt, log_kappa_val, log_tau_val);
    z_pt[i] -= 2 * fd_eps;
    Eigen::VectorXd wm = forward_at(z_pt, log_kappa_val, log_tau_val);
    for (int k = 0; k < n; k++) dw_fd(k, i) = (wp[k] - wm[k]) / (2 * fd_eps);
  }

  {
    SpdeNcTransform tx;
    tx.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    Eigen::VectorXd dz0 = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd w_v, dw_v;
    tx.forward_with_tangent(z, dz0, kappa, 1.0, tau, 0.0, w_v, dw_v);
    for (int k = 0; k < n; k++) dw_fwd(k, n) = dw_v[k];

    Eigen::VectorXd wp = forward_at(z, log_kappa_val + fd_eps, log_tau_val);
    Eigen::VectorXd wm = forward_at(z, log_kappa_val - fd_eps, log_tau_val);
    for (int k = 0; k < n; k++) dw_fd(k, n) = (wp[k] - wm[k]) / (2 * fd_eps);
  }

  {
    SpdeNcTransform tx;
    tx.init(n, C0_d, G1_xv, G1_iv, G1_pv, poles_v, weights_v);
    Eigen::VectorXd dz0 = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd w_v, dw_v;
    tx.forward_with_tangent(z, dz0, kappa, 0.0, tau, 1.0, w_v, dw_v);
    for (int k = 0; k < n; k++) dw_fwd(k, n + 1) = dw_v[k];

    Eigen::VectorXd wp = forward_at(z, log_kappa_val, log_tau_val + fd_eps);
    Eigen::VectorXd wm = forward_at(z, log_kappa_val, log_tau_val - fd_eps);
    for (int k = 0; k < n; k++) dw_fd(k, n + 1) = (wp[k] - wm[k]) / (2 * fd_eps);
  }

  return List::create(
    _["w"]      = NumericVector(w_baseline.data(), w_baseline.data() + n),
    _["dw_fwd"] = dw_fwd,
    _["dw_fd"]  = dw_fd
  );
}

// ---------------------------------------------------------------------------
// Sum-to-zero block-Schur log-determinant + Newton step
// ---------------------------------------------------------------------------
//
// Drives the exact block-Schur path (s2z_block_schur / s2z_log_det_block_schur)
// for B = A + sum_k coef_k 1_k 1_k' on a caller-supplied symmetric PD `A` and a
// set of rank-1 sum-to-zero pins, and returns three independent computations of
// log|B| plus the step error against a dense reference:
//   * ld_block_schur  -- s2z_log_det_block_schur (the path under test);
//   * ld_direct       -- s2z_log_det_direct (the CHOLMOD full-(A+11') reference,
//                        itself validated dense-11'-vs-rank1 in ef208f0);
//   * ld_dense        -- an independent Eigen LLT factorization of dense B;
//   * max_dstep       -- max|delta_block_schur - B^{-1} grad| (Eigen LLT solve).
// `A` is read as a full symmetric matrix; its lower-triangle nonzeros seed the
// SparseHessianBuilder pattern. Field membership is the union of the pin ranges.
//
// When `pin_idx` is supplied it is a length-K list of 0-based ABSOLUTE node
// vectors, one per pin (pin_start / pin_n are then ignored). This drives the
// arbitrary node-index rank-1 path -- a genuine disconnected map's
// non-contiguous component -- through the SAME Woodbury / block-Schur code as a
// contiguous pin, so the dense Eigen reference over the same node sets is the
// oracle for it.
//
// [[Rcpp::export]]
List cpp_test_s2z_block_schur(
    NumericMatrix A,
    IntegerVector pin_start,   // 0-based start of each rank-1 pin (contiguous path)
    IntegerVector pin_n,       // length of each pin (contiguous path)
    NumericVector pin_coef,    // coef of each pin
    NumericVector grad,
    Rcpp::Nullable<NumericMatrix> pin_coupling = R_NilValue,  // dense K x K D
    Rcpp::Nullable<Rcpp::List> pin_idx = R_NilValue           // per-pin node list
) {
  const int n = A.nrow();
  const int K = pin_coef.size();
  const bool use_idx = pin_idx.isNotNull();

  // Stable storage for the per-pin absolute node lists (S2ZRank1 holds a
  // pointer into these; they outlive the solve below).
  std::vector<std::vector<int>> node_store(K);
  if (use_idx) {
    Rcpp::List li(pin_idx);
    for (int k = 0; k < K; ++k) {
      Rcpp::IntegerVector v = li[k];
      node_store[k].assign(v.begin(), v.end());
    }
  } else {
    for (int k = 0; k < K; ++k) {
      node_store[k].resize(pin_n[k]);
      for (int i = 0; i < pin_n[k]; ++i) node_store[k][i] = pin_start[k] + i;
    }
  }

  // Pattern + scatter from the lower-triangle nonzeros of A.
  std::vector<std::pair<int,int>> pattern;
  for (int c = 0; c < n; ++c)
    for (int r = c; r < n; ++r)
      if (A(r, c) != 0.0) pattern.push_back({r, c});

  tulpa::SparseHessianBuilder H;
  H.init(n, pattern);
  H.zero();
  for (int c = 0; c < n; ++c)
    for (int r = c; r < n; ++r)
      if (A(r, c) != 0.0) H.add(r, c, A(r, c));
  for (int k = 0; k < K; ++k) {
    if (use_idx)
      H.add_s2z_rank1(0, (int) node_store[k].size(), node_store[k].data(),
                      pin_coef[k]);
    else
      H.add_s2z_rank1(pin_start[k], pin_n[k], pin_coef[k]);
  }

  // Dense coupling D over the K pins (row-major), or diag(coef) when absent.
  std::vector<double> Dfull;
  if (pin_coupling.isNotNull()) {
    NumericMatrix Dm(pin_coupling);
    if (Dm.nrow() != K || Dm.ncol() != K)
      Rcpp::stop("pin_coupling must be K x K");
    Dfull.resize((std::size_t) K * K);
    for (int a = 0; a < K; ++a)
      for (int b = 0; b < K; ++b) Dfull[(std::size_t) a * K + b] = Dm(a, b);
    H.set_s2z_coupling(Dfull);
  }

  // Path under test: log-det (determinant-only wrapper) and the step.
  const double ld_block_schur =
      tulpa::s2z_log_det_block_schur(H, H.s2z_rank1, NA_REAL);
  const double ld_direct =
      tulpa::s2z_log_det_direct(H, H.s2z_rank1, NA_REAL, nullptr);

  std::vector<double> g(grad.begin(), grad.end());
  std::vector<double> delta_bs(n, 0.0);
  double ld_bs_step = NA_REAL;
  const bool ok =
      tulpa::s2z_block_schur(H, H.s2z_rank1, g.data(), delta_bs.data(), &ld_bs_step);

  // Independent dense reference: B = A + sum_k coef_k 1_k 1_k', Eigen LLT.
  Eigen::MatrixXd B(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) B(i, j) = A(i, j);
  // (U D U')_{pq} = D[a,b] for p in block a, q in block b, over the same node
  // sets the pins carry (contiguous or arbitrary).
  for (int a = 0; a < K; ++a)
    for (int b = 0; b < K; ++b) {
      const double d = Dfull.empty()
          ? (a == b ? pin_coef[a] : 0.0)
          : Dfull[(std::size_t) a * K + b];
      if (d == 0.0) continue;
      for (int i : node_store[a])
        for (int j : node_store[b])
          B(i, j) += d;
    }
  Eigen::LLT<Eigen::MatrixXd> llt(B);
  double ld_dense = NA_REAL;
  double max_dstep = NA_REAL;
  if (llt.info() == Eigen::Success) {
    const Eigen::MatrixXd& L = llt.matrixL();
    double ld = 0.0;
    for (int i = 0; i < n; ++i) ld += 2.0 * std::log(L(i, i));
    ld_dense = ld;
    Eigen::VectorXd gv(n);
    for (int i = 0; i < n; ++i) gv[i] = grad[i];
    const Eigen::VectorXd dd = llt.solve(gv);
    double md = 0.0;
    for (int i = 0; i < n; ++i)
      md = std::max(md, std::fabs(delta_bs[i] - dd[i]));
    max_dstep = md;
  }

  return List::create(
    _["ld_block_schur"]      = ld_block_schur,
    _["ld_direct"]           = ld_direct,
    _["ld_dense"]            = ld_dense,
    _["ld_block_schur_step"] = ld_bs_step,
    _["ok"]                  = ok,
    _["max_dstep"]           = max_dstep
  );
}

// ---------------------------------------------------------------------------
// MCAR prior assembly: Sigma^-1 (x) Q + gradient + log-prior
// ---------------------------------------------------------------------------
//
// Drives the MCAR block's add_prior_sparse + log_prior at one log-Cholesky
// coordinate so an R test can check them against the direct algebra:
//   * the assembled prior Hessian (densified) should equal kronecker(Sigma^-1, Q);
//   * grad should equal -kronecker(Sigma^-1, Q) x  minus the per-field sum-to-
//     zero pin gradient;
//   * log_prior should equal the closed form, incl. the Sigma-dependent
//     normalizer 0.5 (n-1) log|Sigma^-1|.
// `theta_logchol` is the p(p+1)/2 log-Cholesky vector; the adjacency is CSR.
//
// Direct driver of the intrinsic-ICAR log-prior (laplace_spatial_priors.cpp).
// The connected-component partition is derived from the passed adjacency (the
// single source), so a test can assert the per-component rank-deficiency
// invariant on ANY graph: for a block-diagonal I_L (x) Q graph the log-prior
// equals the sum of the L independent single-component log-priors (the (n - L)
// normalizer and the per-component sum-to-zero penalty), and for a genuine
// disconnected map the pins land on the real (unequal, possibly
// non-contiguous) component node sets. Spatial start is 0.
// [[Rcpp::export]]
double cpp_test_log_prior_icar(
    NumericVector x, int n, double tau,
    IntegerVector adj_rp, IntegerVector adj_ci, IntegerVector nnbr
) {
  const tulpa::GraphPartition sp =
      tulpa::graph_partition(n, adj_rp.begin(), adj_ci.begin());
  return tulpa::log_prior_icar(x, /*spatial_start=*/0, n, tau,
                                adj_rp, adj_ci, nnbr, sp);
}

// The Hessian densification, gradient read-back and Sigma^-1 unpacking shared
// by the MCAR and MIID probes below: both drive the same LatentBlock prior
// interface at one log-Cholesky coordinate, and only the block construction
// differs.
static List mcar_family_prior_probe(tulpa::LatentBlock& blk,
                                    const NumericVector& theta_logchol,
                                    int p, int n, const NumericVector& x) {
  const int n_x = p * n;
  std::vector<std::pair<int,int>> pat;
  blk.add_prior_pattern(pat);
  tulpa::SparseHessianBuilder H;
  H.init(n_x, pat);
  H.zero();
  tulpa::DenseVec grad(n_x, 0.0);
  blk.add_prior_sparse(H, grad, x, 0);
  const double lp = blk.log_prior(x, 0);

  NumericMatrix Hd(n_x, n_x);
  for (int c = 0; c < n_x; ++c)
    for (int pp = H.col_ptr[c]; pp < H.col_ptr[c + 1]; ++pp) {
      const int r = H.row_idx[pp];
      const double v = H.values[pp];
      Hd(r, c) += v;
      if (r != c) Hd(c, r) += v;
    }

  std::vector<double> Sinv; double log_det_Sigma;
  tulpa::mcar_sigma_inv_from_logchol(theta_logchol.begin(), p, Sinv, log_det_Sigma);
  NumericMatrix Sinv_m(p, p);
  for (int a = 0; a < p; ++a)
    for (int b = 0; b < p; ++b) Sinv_m(a, b) = Sinv[(std::size_t) a * p + b];

  return List::create(
    _["H"]             = Hd,
    _["grad"]          = NumericVector(grad.begin(), grad.end()),
    _["log_prior"]     = lp,
    _["Sinv"]          = Sinv_m,
    _["log_det_Sigma"] = log_det_Sigma
  );
}

// [[Rcpp::export]]
List cpp_test_mcar_prior(
    NumericVector theta_logchol, int p, int n,
    IntegerVector adj_rp, IntegerVector adj_ci, IntegerVector nnbr,
    NumericVector x
) {
  const int m = p * (p + 1) / 2;
  NumericMatrix tg(1, m);
  for (int t = 0; t < m; ++t) tg(0, t) = theta_logchol[t];

  std::vector<Rcpp::IntegerVector> cell_idx;                 // unused by the prior
  std::vector<std::vector<Rcpp::NumericVector>> field_weight;
  const tulpa::GraphPartition sp =
      tulpa::graph_partition(n, adj_rp.begin(), adj_ci.begin());
  tulpa::LatentBlock blk = tulpa::make_mcar_block(
      /*start=*/0, n, p, /*axis0=*/0, tg, cell_idx, field_weight,
      adj_rp, adj_ci, nnbr, /*copy_arm=*/-1, /*axis_alpha=*/-1, sp);

  return mcar_family_prior_probe(blk, theta_logchol, p, n, x);
}

// Expose the outer-grid memory-budget arithmetic (mem_budget.h) so the
// "budget against free RAM, not installed" decision and the per-thread cap are
// unit-testable without a running fit. Byte counts cross R as doubles.
// [[Rcpp::export]]
double cpp_test_outer_thread_mem_budget(double avail_bytes, double total_bytes) {
  return static_cast<double>(tulpa::outer_thread_mem_budget(
      static_cast<std::size_t>(avail_bytes),
      static_cast<std::size_t>(total_bytes)));
}

// [[Rcpp::export]]
int cpp_test_outer_thread_cap(double budget_bytes, double per_thread_bytes) {
  return tulpa::outer_thread_cap(static_cast<std::size_t>(budget_bytes),
                                 static_cast<std::size_t>(per_thread_bytes));
}

// Measure the CHOLMOD factor bytes for a k-by-k 2D lattice GMRF precision
// pattern (diagonal + 4-neighbour edges, lower triangle stype = -1) via the
// same supernodal symbolic analyze the outer-grid memory clamp uses. Returns
// the pattern nnz, the measured factor bytes, and the old flat 2x-nnz guess so
// a test can confirm the measurement captures the superlinear 2D fill-in the
// guess misses.
// [[Rcpp::export]]
List cpp_test_grid_factor_bytes(int k) {
  const int n = k * k;
  auto id = [k](int r, int c) { return r * k + c; };
  std::vector<std::pair<int,int>> pat;
  pat.reserve(static_cast<std::size_t>(n) * 3);
  for (int r = 0; r < k; ++r) {
    for (int c = 0; c < k; ++c) {
      const int u = id(r, c);
      pat.emplace_back(u, u);                          // diagonal
      if (c + 1 < k) pat.emplace_back(u, id(r, c + 1));  // east edge
      if (r + 1 < k) pat.emplace_back(u, id(r + 1, c));  // south edge
    }
  }
  tulpa::SparseHessianBuilder H;
  H.init(n, pat);   // dedups + normalises to lower triangle
  H.zero();
  const std::size_t nnz_Q = H.values.size();

  tulpa::SparseCholeskySolver solver;
  cholmod_sparse Hv = H.as_cholmod(&solver.common());
  solver.analyze(&Hv);
  const std::size_t factor_bytes = solver.analyzed_factor_bytes();

  return List::create(
    _["n"]            = n,
    _["nnz_Q"]        = static_cast<double>(nnz_Q),
    _["factor_bytes"] = static_cast<double>(factor_bytes),
    _["old_guess"]    = static_cast<double>(nnz_Q) * 2.0 * sizeof(double)
  );
}

// Direct algebra check for the MIID block (mcar with Q = I): the precision must
// be Sigma^-1 (x) I_n exactly, with the matching gradient (no sum-to-zero pin,
// P is full rank) and log-prior (Sigma-dependent normalizer 0.5 n log|Sigma^-1|,
// full n not n - 1).
// [[Rcpp::export]]
List cpp_test_miid_prior(
    NumericVector theta_logchol, int p, int n, NumericVector x
) {
  const int m = p * (p + 1) / 2;
  NumericMatrix tg(1, m);
  for (int t = 0; t < m; ++t) tg(0, t) = theta_logchol[t];

  std::vector<Rcpp::IntegerVector> group_idx;               // unused by the prior
  std::vector<std::vector<Rcpp::NumericVector>> field_weight;
  tulpa::LatentBlock blk = tulpa::make_miid_block(
      /*start=*/0, n, p, /*axis0=*/0, tg, group_idx, field_weight,
      /*copy_arm=*/-1, /*axis_alpha=*/-1);

  return mcar_family_prior_probe(blk, theta_logchol, p, n, x);
}

// ---------------------------------------------------------------------------
// PC-prior scales (A4). Exposes every parameterization of the
// shared PC prior so a test can verify each against the base sigma-density plus
// a numerical change-of-variables Jacobian.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_pc_prior_scales(double sigma, double U, double alpha) {
  using namespace tulpa;
  const double sigma2    = sigma * sigma;
  const double tau       = 1.0 / sigma2;
  return Rcpp::NumericVector::create(
    Rcpp::_["sigma"]      = log_prior_sigma_pc<double>(sigma, U, alpha),
    Rcpp::_["log_sigma"]  = log_prior_log_sigma_pc<double>(std::log(sigma), U, alpha),
    Rcpp::_["sigma2"]     = log_prior_sigma2_pc<double>(sigma2, U, alpha),
    Rcpp::_["log_sigma2"] = log_prior_log_sigma2_pc<double>(std::log(sigma2), U, alpha),
    Rcpp::_["tau"]        = log_prior_tau_pc<double>(tau, U, alpha),
    Rcpp::_["log_tau"]    = log_prior_log_tau_pc<double>(std::log(tau), U, alpha)
  );
}

// ---------------------------------------------------------------------------
// Out-of-pattern Hessian writes. Scatters every lower-triangle nonzero of A
// through SparseHessianBuilder::add() against a pattern that optionally omits
// one of them, so the scatter reaches an entry the pattern cannot hold -- the
// invariant that broke for unequal / non-contiguous areal components (#241) and
// weighted-entry blocks (#242). `slot_val` exercises the cached-slot write path
// (scatter_slot) through an unresolved index. With raise = TRUE the guard's
// check() surfaces the error the solve drivers raise.
// [[Rcpp::export]]
Rcpp::List cpp_test_hessian_pattern_guard(
    Rcpp::NumericMatrix A,
    int    omit_entry = -1,
    bool   raise      = false,
    double slot_val   = 0.0
) {
  const int n = A.nrow();

  std::vector<std::pair<int,int>> nz;
  for (int c = 0; c < n; ++c)
    for (int r = c; r < n; ++r)
      if (A(r, c) != 0.0) nz.push_back({r, c});

  std::vector<std::pair<int,int>> pattern;
  for (int e = 0; e < (int) nz.size(); ++e)
    if (e != omit_entry) pattern.push_back(nz[e]);

  tulpa::SparseHessianBuilder H;
  H.init(n, pattern);
  H.zero();

  const tulpa::HessianPatternGuard guard;
  for (const auto& rc : nz) H.add(rc.first, rc.second, A(rc.first, rc.second));

  // Cached-slot path: an unresolved slot discards slot_val. A structural zero
  // there is not a drop -- index caches resolve whole cross products and hold
  // -1 for pairs no observation ever touches.
  double sink = 0.0;
  tulpa::scatter_slot(&sink, -1, slot_val);

  const double dropped = (double) guard.dropped();
  double stored = 0.0;
  for (double v : H.values) stored += v;

  if (raise) guard.check("cpp_test_hessian_pattern_guard");

  return Rcpp::List::create(
    Rcpp::_["dropped"]    = dropped,
    Rcpp::_["nnz"]        = H.nnz,
    Rcpp::_["stored_sum"] = stored
  );
}

// Reports how an intrinsic areal field's component partition came out, and
// whether compute_param_layout's guard accepts it. `use_setter = FALSE`
// reproduces a consumer that assigns the CSR adjacency and nothing else: the
// partition stays default-constructed, so it describes 0 nodes and reports 0
// components even though n_spatial_components still reads its own default of 1.
// The augmentation iterates over components, so 0 of them pins nothing.
// [[Rcpp::export]]
Rcpp::List cpp_spatial_partition_probe(int n_units,
                                       Rcpp::IntegerVector row_ptr,
                                       Rcpp::IntegerVector col_idx,
                                       Rcpp::IntegerVector n_neighbors,
                                       bool use_setter) {
  tulpa::ModelData d;
  if (use_setter) {
    d.set_spatial_adjacency(n_units,
                            std::vector<int>(row_ptr.begin(), row_ptr.end()),
                            std::vector<int>(col_idx.begin(), col_idx.end()),
                            std::vector<int>(n_neighbors.begin(), n_neighbors.end()));
  } else {
    d.n_spatial_units = n_units;
    d.adj_row_ptr.assign(row_ptr.begin(), row_ptr.end());
    d.adj_col_idx.assign(col_idx.begin(), col_idx.end());
    d.n_neighbors.assign(n_neighbors.begin(), n_neighbors.end());
  }

  bool accepted = true;
  std::string error;
  try {
    tulpa_hmc::require_spatial_partition(d);
  } catch (std::exception& e) {
    accepted = false;
    error = e.what();
  }

  return Rcpp::List::create(
    Rcpp::_["partition_n"]     = d.spatial_partition.n,
    Rcpp::_["components_seen"] = d.spatial_partition.n_components(),
    Rcpp::_["n_components"]    = d.n_spatial_components,
    Rcpp::_["accepted"]        = accepted,
    Rcpp::_["error"]           = error
  );
}

// =====================================================================
// SPDE FEM assembly probe (gcol33/tulpa#280)
// =====================================================================
// Assembles Q at the requested operator order and returns it as CSC, so the
// testthat layer can compare the compiled binomial expansion against an
// independently built K (C^-1 K)^(alpha-1) at each integer alpha. Also returns
// the (kappa, tau) the shared Matern conversion produces for the given nu
// (gcol33/tulpa#279), so both formulas are checked at one call site.
// [[Rcpp::export]]
List cpp_test_spde_assemble(
    Rcpp::NumericVector C0_diag,
    Rcpp::NumericVector G1_x,
    Rcpp::IntegerVector G1_i,
    Rcpp::IntegerVector G1_p,
    double range, double sigma, double nu, int alpha
) {
  const int n = C0_diag.size();
  double kappa, tau;
  std::tie(kappa, tau) = tulpa::spde_range_sigma_to_kappa_tau(range, sigma, nu);

  tulpa::SpdeQBuilder qb;
  qb.init(n, C0_diag, G1_x, G1_i, G1_p, alpha);
  qb.rebuild(kappa, tau);

  return Rcpp::List::create(
    Rcpp::_["Q_p"]   = Rcpp::IntegerVector(qb.Q_p.begin(), qb.Q_p.end()),
    Rcpp::_["Q_i"]   = Rcpp::IntegerVector(qb.Q_i.begin(), qb.Q_i.end()),
    Rcpp::_["Q_x"]   = Rcpp::NumericVector(qb.Q_x.begin(), qb.Q_x.end()),
    Rcpp::_["kappa"] = kappa,
    Rcpp::_["tau"]   = tau,
    Rcpp::_["alpha"] = qb.alpha_order
  );
}

// ---------------------------------------------------------------------------
// NNGP prior scatter, next to the pieces needed to rebuild Lambda
// independently (gcol33/tulpa#278).
//
// Runs batch_nngp_scatter, then applies the prior scatter to the (alpha, cv)
// it returned. Handing back the filled Hessian alongside those coefficients
// lets the caller assemble Lambda = (I - A)' D^-1 (I - A) from scratch and
// score the scatter against the matrix it claims to build. `cv` also reports
// how many locations hit the conditional-variance floor, which sets Lambda's
// magnitude (see nngp_moments_from_chol).
//
// gp_start = 0 (the block is the whole latent) so the returned matrix is the
// prior alone, with no data scatter or ridge folded in.
// [[Rcpp::export]]
Rcpp::List cpp_test_nngp_prior_scatter(
    Rcpp::NumericVector w,
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial, int nn,
    double sigma2, double phi_gp, int cov_type
) {
  std::vector<double> w_block(w.begin(), w.end());
  std::vector<double> cond_mean, cv, alpha;
  bool gpu_used = false;
  // (alpha, cv) do not depend on w: the conditional regression weights and
  // variances are functions of the covariance alone. Pass w through anyway so
  // the call matches how the block factory builds them.
  tulpa::batch_nngp_scatter(w_block, n_spatial, nn, sigma2, phi_gp, cov_type,
                            coords, nn_idx, nn_dist, nn_order,
                            cond_mean, cv, gpu_used, &alpha);

  std::vector<std::pair<int,int>> pattern;
  tulpa::make_nngp_prior_sparsity_pattern(pattern, nn_idx, nn_order,
                                          n_spatial, nn, /*gp_start=*/0);
  tulpa::SparseHessianBuilder H_builder;
  H_builder.init(n_spatial, pattern);
  H_builder.zero();
  tulpa::DenseVec grad_sparse(n_spatial, 0.0);
  const tulpa::HessianPatternGuard guard;
  tulpa::apply_nngp_full_prior_sparse(grad_sparse, H_builder, w_block, alpha,
                                      cv, nn_idx, nn_order, n_spatial, nn,
                                      /*gp_start=*/0);
  const double dropped = (double) guard.dropped();

  // Sparse lower triangle -> dense symmetric, so the caller compares like
  // with like.
  Rcpp::NumericMatrix H_sp(n_spatial, n_spatial);
  for (int c = 0; c < n_spatial; c++) {
    for (int e = H_builder.col_ptr[c]; e < H_builder.col_ptr[c + 1]; e++) {
      const int r = H_builder.row_idx[e];
      H_sp(r, c) = H_builder.values[e];
      if (r != c) H_sp(c, r) = H_builder.values[e];
    }
  }

  Rcpp::NumericMatrix alpha_mat(n_spatial, nn);
  for (int i = 0; i < n_spatial; i++)
    for (int k = 0; k < nn; k++)
      alpha_mat(i, k) = alpha[(std::size_t) i * nn + k];

  return Rcpp::List::create(
    Rcpp::_["H"]       = H_sp,
    Rcpp::_["grad"]    = Rcpp::NumericVector(grad_sparse.begin(), grad_sparse.end()),
    Rcpp::_["alpha"]   = alpha_mat,
    Rcpp::_["cv"]      = Rcpp::NumericVector(cv.begin(), cv.end()),
    Rcpp::_["dropped"]  = dropped,
    Rcpp::_["nnz"]      = H_builder.nnz,
    Rcpp::_["gpu_used"] = gpu_used,
    // The neighbour covariance is factorized as C + nugget * I, so a reference
    // scores the returned moments against that matrix rather than against C.
    Rcpp::_["nugget"]     = tulpa_linalg::kNngpNugget,
    Rcpp::_["cond_var_floor"] = tulpa_linalg::kNngpVarFloor
  );
}


// ---------------------------------------------------------------------------
// log_sum_exp on every scalar type, including the both-arguments--inf corner
// the mixture likelihoods reach when two components both underflow. The double
// core short-circuits there; an AD overload without the same guard computes
// inf - inf and stores NaN as the partial.
// [[Rcpp::export]]
List cpp_test_lse_guard(double a, double b) {
  const double v_double = tulpa_linalg::log_sum_exp(a, b);

  fwd::Dual ad_(a, 1.0), bd(b, 0.0);
  fwd::Dual rd = fwd::log_sum_exp(ad_, bd);

  double v_tape = 0.0, ga_tape = 0.0, gb_tape = 0.0;
  {
    tulpa::ad::TapeScope scope;
    tulpa::ad::Var av(scope.tape, a), bv(scope.tape, b);
    tulpa::ad::Var rv = tulpa::ad::log_sum_exp(av, bv);
    rv.backward();
    v_tape = rv.val(); ga_tape = av.adj(); gb_tape = bv.adj();
  }

  double v_arena = 0.0, ga_arena = 0.0, gb_arena = 0.0;
  {
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Arena* ar = scope.arena();
    tulpa::arena::Var av(ar, a), bv(ar, b);
    tulpa::arena::Var rv = tulpa::arena::log_sum_exp(av, bv);
    rv.backward();
    v_arena = rv.val(); ga_arena = av.adj(); gb_arena = bv.adj();
  }

  return List::create(
    Named("value_double") = v_double,
    Named("value_dual")   = rd.val,
    Named("grad_a_dual")  = rd.grad,
    Named("value_tape")   = v_tape,
    Named("grad_a_tape")  = ga_tape,
    Named("grad_b_tape")  = gb_tape,
    Named("value_arena")  = v_arena,
    Named("grad_a_arena") = ga_arena,
    Named("grad_b_arena") = gb_arena
  );
}

// One scalar primitive on every scalar type the templated log posterior is
// instantiated for
// ---------------------------------------------------------------------------
//
// compute_log_post is instantiated for double (the value path, which the
// numerical gradient finite-differences) and for the two reverse-mode Var types
// (the live gradient). The runtime check compares those two evaluations as
// though they were one function, so each primitive has to agree in value and in
// derivative across all four types -- including at the guard boundaries, where
// an unguarded overload returns Inf or NaN and its guarded twin returns a
// finite number.
//
// `fn` is one of exp / log / sqrt / pow / logit; `p` is the exponent read by
// pow. The forward-mode dual is seeded at 1, so its `grad` is f'(x).
// [[Rcpp::export]]
List cpp_test_scalar_guard(std::string fn, double x, double p = 2.0) {
  const bool is_exp   = (fn == "exp");
  const bool is_log   = (fn == "log");
  const bool is_sqrt  = (fn == "sqrt");
  const bool is_pow   = (fn == "pow");
  const bool is_logit = (fn == "logit");
  const bool is_expm1 = (fn == "expm1");
  const bool is_l1me  = (fn == "log1m_exp");
  if (!(is_exp || is_log || is_sqrt || is_pow || is_logit || is_expm1 ||
        is_l1me)) {
    Rcpp::stop("Unknown primitive '%s'; expected exp, log, sqrt, pow, logit, "
               "expm1 or log1m_exp.", fn.c_str());
  }

  // Value path.
  double v_double;
  if (is_exp)        v_double = tulpa::math::safe_exp(x);
  else if (is_log)   v_double = tulpa::math::safe_log(x);
  else if (is_sqrt)  v_double = tulpa::math::safe_sqrt(x);
  else if (is_pow)   v_double = std::pow(x, p);
  else if (is_expm1) v_double = tulpa::math::expm1_fn(x);
  else if (is_l1me)  v_double = tulpa::math::log1m_exp_fn(x);
  else {
    const double c = tulpa::math::clamp_prob(x);
    v_double = std::log(c / (1.0 - c));
  }

  // Forward mode. pow and logit have no dual overload in the shared set, so
  // they are reported as the value path with no derivative.
  double v_dual = v_double, g_dual = NA_REAL;
  if (is_exp || is_log || is_sqrt || is_expm1 || is_l1me) {
    fwd::Dual xd(x, 1.0);
    fwd::Dual rd = is_exp   ? tulpa::math::safe_exp(xd)
                 : is_log   ? tulpa::math::safe_log(xd)
                 : is_sqrt  ? tulpa::math::safe_sqrt(xd)
                 : is_expm1 ? tulpa::math::expm1_fn(xd)
                            : tulpa::math::log1m_exp_fn(xd);
    v_dual = rd.val;
    g_dual = rd.grad;
  }

  // Tape reverse mode.
  double v_tape = 0.0, g_tape = 0.0;
  {
    tulpa::ad::TapeScope scope;
    tulpa::ad::Var xv(scope.tape, x);
    tulpa::ad::Var rv = is_exp   ? tulpa::ad::exp(xv)
                      : is_log   ? tulpa::ad::log(xv)
                      : is_sqrt  ? tulpa::ad::sqrt(xv)
                      : is_pow   ? tulpa::ad::pow(xv, p)
                      : is_expm1 ? tulpa::math::expm1_fn(xv)
                      : is_l1me  ? tulpa::math::log1m_exp_fn(xv)
                                 : tulpa::ad::logit(xv);
    rv.backward();
    v_tape = rv.val();
    g_tape = xv.adj();
  }

  // Arena reverse mode.
  double v_arena = 0.0, g_arena = 0.0;
  {
    tulpa::arena::ArenaScope scope;
    tulpa::arena::Arena* ar = scope.arena();
    tulpa::arena::Var xv(ar, x);
    tulpa::arena::Var rv = is_exp   ? tulpa::arena::exp(xv)
                         : is_log   ? tulpa::arena::log(xv)
                         : is_sqrt  ? tulpa::arena::sqrt(xv)
                         : is_pow   ? tulpa::arena::pow(xv, p)
                         : is_expm1 ? tulpa::math::expm1_fn(xv)
                         : is_l1me  ? tulpa::math::log1m_exp_fn(xv)
                                    : tulpa::arena::logit(xv);
    rv.backward();
    v_arena = rv.val();
    g_arena = xv.adj();
  }

  return List::create(
    Named("value_double") = v_double,
    Named("value_dual")   = v_dual,
    Named("grad_dual")    = g_dual,
    Named("value_tape")   = v_tape,
    Named("grad_tape")    = g_tape,
    Named("value_arena")  = v_arena,
    Named("grad_arena")   = g_arena
  );
}


// ---------------------------------------------------------------------------
// One ICAR field sweep, exposed so the full conditional can be scored against
// its analytic form (gcol33/tulpa#423).
//
// The sweep is Gauss-Seidel: unit j's conditional mean reads the neighbours the
// sweep has already reached at their new values and every other neighbour at
// the value `phi` arrives with. A replica drawing from the same R stream
// reproduces the result only if it reads the incoming field the same way, so
// the comparison pins the conditional rather than a summary of it.
//
// `phi` is copied, not modified, so a caller can sweep the same starting field
// under several taus.
// [[Rcpp::export]]
Rcpp::List cpp_test_update_spatial_icar(
    Rcpp::NumericVector kappa,
    Rcpp::NumericVector omega,
    Rcpp::NumericVector offset,
    Rcpp::IntegerVector group,
    Rcpp::List adj_list,
    Rcpp::IntegerVector n_neighbors,
    int n_units,
    double tau,
    Rcpp::NumericVector phi
) {
  const tulpa::PgAdjacency adj =
      tulpa::pg_build_adjacency(adj_list, n_neighbors, n_units);
  if (phi.size() != n_units) {
    Rcpp::stop("`phi` has length %d; must be n_units (%d).",
               static_cast<int>(phi.size()), n_units);
  }
  Rcpp::NumericVector phi_out = Rcpp::clone(phi);
  double removed_mean = 0.0;
  tulpa::update_spatial_icar(kappa, omega, offset, group, adj, tau,
                             phi_out, removed_mean);
  return Rcpp::List::create(
    Rcpp::_["phi"]          = phi_out,
    Rcpp::_["removed_mean"] = removed_mean,
    Rcpp::_["n_components"] = adj.n_components,
    Rcpp::_["component"]    = Rcpp::IntegerVector(adj.component.begin(),
                                                  adj.component.end()),
    Rcpp::_["isolated_prec"] = tulpa::PG_ICAR_ISOLATED_PREC
  );
}
