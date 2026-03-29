// autodiff_fwd.h
// Forward-mode automatic differentiation using dual numbers
// Fast alternative to tape-based autodiff (A_t) - no heap allocation
//
// Usage:
//   fwd::Dual x(3.0, 1.0);  // x = 3.0, dx/dx = 1.0
//   fwd::Dual y(2.0, 0.0);  // y = 2.0, dy/dx = 0.0 (constant w.r.t. x)
//   fwd::Dual z = x * y + fwd::exp(x);
//   // z.val = 3*2 + exp(3) = 26.09
//   // z.grad = y + exp(x) = 2 + exp(3) = 22.09 (derivative w.r.t. x)

#ifndef TULPA_AUTODIFF_FWD_H
#define TULPA_AUTODIFF_FWD_H

#include <cmath>
#include <vector>
#include <algorithm>
#include <Rcpp.h>
#include "portable_math.h"

namespace fwd {

// ============================================================================
// Dual number class for forward-mode autodiff
// ============================================================================

class Dual {
public:
    double val;   // Function value
    double grad;  // Derivative w.r.t. the seeded parameter

    // Constructors
    Dual() : val(0.0), grad(0.0) {}
    Dual(double v) : val(v), grad(0.0) {}
    Dual(double v, double g) : val(v), grad(g) {}

    // Compound assignment operators
    Dual& operator+=(const Dual& rhs) {
        val += rhs.val;
        grad += rhs.grad;
        return *this;
    }

    Dual& operator-=(const Dual& rhs) {
        val -= rhs.val;
        grad -= rhs.grad;
        return *this;
    }

    Dual& operator*=(const Dual& rhs) {
        grad = val * rhs.grad + grad * rhs.val;
        val *= rhs.val;
        return *this;
    }

    Dual& operator/=(const Dual& rhs) {
        grad = (grad * rhs.val - val * rhs.grad) / (rhs.val * rhs.val);
        val /= rhs.val;
        return *this;
    }

    // Compound with double
    Dual& operator+=(double rhs) {
        val += rhs;
        return *this;
    }

    Dual& operator-=(double rhs) {
        val -= rhs;
        return *this;
    }

    Dual& operator*=(double rhs) {
        val *= rhs;
        grad *= rhs;
        return *this;
    }

    Dual& operator/=(double rhs) {
        val /= rhs;
        grad /= rhs;
        return *this;
    }

    // Unary minus
    Dual operator-() const {
        return Dual(-val, -grad);
    }
};

// ============================================================================
// Arithmetic operators: Dual op Dual
// ============================================================================

inline Dual operator+(const Dual& a, const Dual& b) {
    return Dual(a.val + b.val, a.grad + b.grad);
}

inline Dual operator-(const Dual& a, const Dual& b) {
    return Dual(a.val - b.val, a.grad - b.grad);
}

inline Dual operator*(const Dual& a, const Dual& b) {
    // d(a*b) = a*db + b*da
    return Dual(a.val * b.val, a.val * b.grad + a.grad * b.val);
}

inline Dual operator/(const Dual& a, const Dual& b) {
    // d(a/b) = (da*b - a*db) / b^2
    double b2 = b.val * b.val;
    return Dual(a.val / b.val, (a.grad * b.val - a.val * b.grad) / b2);
}

// ============================================================================
// Arithmetic operators: Dual op double
// ============================================================================

inline Dual operator+(const Dual& a, double b) {
    return Dual(a.val + b, a.grad);
}

inline Dual operator-(const Dual& a, double b) {
    return Dual(a.val - b, a.grad);
}

inline Dual operator*(const Dual& a, double b) {
    return Dual(a.val * b, a.grad * b);
}

inline Dual operator/(const Dual& a, double b) {
    return Dual(a.val / b, a.grad / b);
}

// ============================================================================
// Arithmetic operators: double op Dual
// ============================================================================

inline Dual operator+(double a, const Dual& b) {
    return Dual(a + b.val, b.grad);
}

inline Dual operator-(double a, const Dual& b) {
    return Dual(a - b.val, -b.grad);
}

inline Dual operator*(double a, const Dual& b) {
    return Dual(a * b.val, a * b.grad);
}

inline Dual operator/(double a, const Dual& b) {
    // d(a/b) = -a * db / b^2
    double b2 = b.val * b.val;
    return Dual(a / b.val, -a * b.grad / b2);
}

// ============================================================================
// Arithmetic operators: Dual op int
// ============================================================================

inline Dual operator+(const Dual& a, int b) {
    return Dual(a.val + b, a.grad);
}

inline Dual operator-(const Dual& a, int b) {
    return Dual(a.val - b, a.grad);
}

inline Dual operator*(const Dual& a, int b) {
    return Dual(a.val * b, a.grad * b);
}

inline Dual operator/(const Dual& a, int b) {
    return Dual(a.val / b, a.grad / b);
}

// ============================================================================
// Arithmetic operators: int op Dual
// ============================================================================

inline Dual operator+(int a, const Dual& b) {
    return Dual(a + b.val, b.grad);
}

inline Dual operator-(int a, const Dual& b) {
    return Dual(a - b.val, -b.grad);
}

inline Dual operator*(int a, const Dual& b) {
    return Dual(a * b.val, a * b.grad);
}

inline Dual operator/(int a, const Dual& b) {
    double b2 = b.val * b.val;
    return Dual(static_cast<double>(a) / b.val, -a * b.grad / b2);
}

// ============================================================================
// Comparison operators
// ============================================================================

inline bool operator<(const Dual& a, const Dual& b) { return a.val < b.val; }
inline bool operator>(const Dual& a, const Dual& b) { return a.val > b.val; }
inline bool operator<=(const Dual& a, const Dual& b) { return a.val <= b.val; }
inline bool operator>=(const Dual& a, const Dual& b) { return a.val >= b.val; }
inline bool operator==(const Dual& a, const Dual& b) { return a.val == b.val; }
inline bool operator!=(const Dual& a, const Dual& b) { return a.val != b.val; }

inline bool operator<(const Dual& a, double b) { return a.val < b; }
inline bool operator>(const Dual& a, double b) { return a.val > b; }
inline bool operator<=(const Dual& a, double b) { return a.val <= b; }
inline bool operator>=(const Dual& a, double b) { return a.val >= b; }
inline bool operator==(const Dual& a, double b) { return a.val == b; }
inline bool operator!=(const Dual& a, double b) { return a.val != b; }

inline bool operator<(double a, const Dual& b) { return a < b.val; }
inline bool operator>(double a, const Dual& b) { return a > b.val; }
inline bool operator<=(double a, const Dual& b) { return a <= b.val; }
inline bool operator>=(double a, const Dual& b) { return a >= b.val; }
inline bool operator==(double a, const Dual& b) { return a == b.val; }
inline bool operator!=(double a, const Dual& b) { return a != b.val; }

// ============================================================================
// Math functions
// ============================================================================

inline Dual exp(const Dual& x) {
    double e = std::exp(x.val);
    return Dual(e, e * x.grad);
}

inline Dual log(const Dual& x) {
    return Dual(std::log(x.val), x.grad / x.val);
}

inline Dual safe_log(const Dual& x) {
    if (x.val <= 0.0) {
        return Dual(-1e10, 0.0);
    }
    return Dual(std::log(x.val), x.grad / x.val);
}

inline Dual sqrt(const Dual& x) {
    double s = std::sqrt(x.val);
    return Dual(s, 0.5 * x.grad / s);
}

inline Dual pow(const Dual& x, double n) {
    // d(x^n) = n * x^(n-1) * dx
    double p = std::pow(x.val, n);
    double dp = n * std::pow(x.val, n - 1);
    return Dual(p, dp * x.grad);
}

inline Dual pow(const Dual& x, const Dual& y) {
    // x^y = exp(y * log(x))
    // d(x^y) = x^y * (y' * log(x) + y * x'/x)
    double p = std::pow(x.val, y.val);
    double dp = p * (y.grad * std::log(x.val) + y.val * x.grad / x.val);
    return Dual(p, dp);
}

inline Dual log1p(const Dual& x) {
    // d(log(1+x)) = dx / (1+x)
    return Dual(std::log1p(x.val), x.grad / (1.0 + x.val));
}

inline Dual expm1(const Dual& x) {
    // d(exp(x)-1) = exp(x) * dx
    double e = std::exp(x.val);
    return Dual(e - 1.0, e * x.grad);
}

inline Dual softplus(const Dual& x) {
    // softplus(x) = log(1 + exp(x))
    // For numerical stability: if x > 20, softplus(x) ≈ x
    if (x.val > 20.0) {
        return x;
    } else if (x.val < -20.0) {
        return exp(x);
    }
    // d(softplus) = exp(x) / (1 + exp(x)) = inv_logit(x)
    double e = std::exp(x.val);
    double sp = std::log1p(e);
    double dsp = e / (1.0 + e);
    return Dual(sp, dsp * x.grad);
}

inline Dual inv_logit(const Dual& x) {
    // inv_logit(x) = 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x))
    // d(inv_logit) = inv_logit(x) * (1 - inv_logit(x))
    double p;
    if (x.val > 0) {
        double exp_neg_x = std::exp(-x.val);
        p = 1.0 / (1.0 + exp_neg_x);
    } else {
        double exp_x = std::exp(x.val);
        p = exp_x / (1.0 + exp_x);
    }
    double dp = p * (1.0 - p);
    return Dual(p, dp * x.grad);
}

inline Dual logit(const Dual& x) {
    // logit(x) = log(x / (1-x))
    // d(logit) = 1/(x*(1-x))
    double l = std::log(x.val / (1.0 - x.val));
    double dl = 1.0 / (x.val * (1.0 - x.val));
    return Dual(l, dl * x.grad);
}

inline Dual lgamma(const Dual& x) {
    // d(lgamma(x)) = digamma(x)
    return Dual(std::lgamma(x.val), tulpa::math::portable_digamma(x.val) * x.grad);
}

inline Dual digamma(const Dual& x) {
    // d(digamma(x)) = trigamma(x)
    return Dual(tulpa::math::portable_digamma(x.val), tulpa::math::portable_trigamma(x.val) * x.grad);
}

inline Dual abs(const Dual& x) {
    if (x.val >= 0) {
        return x;
    } else {
        return -x;
    }
}

inline Dual fmin(const Dual& a, const Dual& b) {
    return (a.val <= b.val) ? a : b;
}

inline Dual fmax(const Dual& a, const Dual& b) {
    return (a.val >= b.val) ? a : b;
}

inline Dual fmin(const Dual& a, double b) {
    return (a.val <= b) ? a : Dual(b, 0.0);
}

inline Dual fmax(const Dual& a, double b) {
    return (a.val >= b) ? a : Dual(b, 0.0);
}

// Log-sum-exp for numerical stability
inline Dual log_sum_exp(const Dual& a, const Dual& b) {
    // log(exp(a) + exp(b))
    // = max(a,b) + log(1 + exp(min(a,b) - max(a,b)))
    double max_val = std::max(a.val, b.val);
    double exp_a = std::exp(a.val - max_val);
    double exp_b = std::exp(b.val - max_val);
    double sum_exp = exp_a + exp_b;
    double val = max_val + std::log(sum_exp);
    // d(lse) = (exp(a)*da + exp(b)*db) / (exp(a) + exp(b))
    //        = (exp(a-max)*da + exp(b-max)*db) / (exp(a-max) + exp(b-max))
    double grad = (exp_a * a.grad + exp_b * b.grad) / sum_exp;
    return Dual(val, grad);
}

// Sin/cos/tan if needed
inline Dual sin(const Dual& x) {
    return Dual(std::sin(x.val), std::cos(x.val) * x.grad);
}

inline Dual cos(const Dual& x) {
    return Dual(std::cos(x.val), -std::sin(x.val) * x.grad);
}

inline Dual tan(const Dual& x) {
    double c = std::cos(x.val);
    return Dual(std::tan(x.val), x.grad / (c * c));
}

// ============================================================================
// Type trait for detecting Dual
// ============================================================================

template<typename T>
struct is_dual : std::false_type {};

template<>
struct is_dual<Dual> : std::true_type {};

// ============================================================================
// Utility functions for gradient computation
// ============================================================================

// Seed a vector of duals for computing gradient w.r.t. parameter i
inline void seed_duals(std::vector<Dual>& duals, const std::vector<double>& values, int seed_idx) {
    int n = static_cast<int>(values.size());
    duals.resize(n);
    for (int i = 0; i < n; i++) {
        duals[i].val = values[i];
        duals[i].grad = (i == seed_idx) ? 1.0 : 0.0;
    }
}

// Re-seed an existing vector for a different parameter
inline void reseed_duals(std::vector<Dual>& duals, int seed_idx) {
    int n = static_cast<int>(duals.size());
    for (int i = 0; i < n; i++) {
        duals[i].grad = (i == seed_idx) ? 1.0 : 0.0;
    }
}

// Update values in existing duals (keeping structure)
inline void update_values(std::vector<Dual>& duals, const std::vector<double>& values) {
    for (size_t i = 0; i < values.size(); i++) {
        duals[i].val = values[i];
    }
}

// Extract values from duals
inline std::vector<double> get_values(const std::vector<Dual>& duals) {
    std::vector<double> vals(duals.size());
    for (size_t i = 0; i < duals.size(); i++) {
        vals[i] = duals[i].val;
    }
    return vals;
}

// Value extraction (for template compatibility)
inline double get_value(const Dual& x) {
    return x.val;
}

// Dot product with double array and Dual coefficients
inline Dual dot_product_mixed(const double* x, const Dual* y, int n) {
    Dual sum(0.0, 0.0);
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

inline Dual dot_product_mixed(const double* x, const std::vector<Dual>& y, int n) {
    Dual sum(0.0, 0.0);
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// Sum of Dual vector
inline Dual sum(const std::vector<Dual>& x) {
    Dual s(0.0, 0.0);
    for (const auto& xi : x) {
        s += xi;
    }
    return s;
}

}  // namespace fwd

#endif  // TULPA_AUTODIFF_FWD_H
