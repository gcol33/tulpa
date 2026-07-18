// autodiff.h
// Reverse-mode automatic differentiation for ratiod
// Thread-safe implementation: Tape passed as parameter, no global state

#ifndef TULPA_AUTODIFF_H
#define TULPA_AUTODIFF_H

#include <Rcpp.h>
#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include "tulpa/portable_math.h"

namespace tulpa {
namespace ad {

// Forward declarations
class Tape;
class Var;

// ---------------------------------------------------------------------
// Tape: Records computation graph for reverse-mode AD
// ---------------------------------------------------------------------
class Tape {
public:
  struct Node {
    double value;
    double adjoint;
    std::function<void(Tape*)> backward;

    Node(double v = 0.0) : value(v), adjoint(0.0), backward([](Tape*){}) {}
  };

  std::vector<Node> nodes;

  size_t add_node(double value) {
    nodes.emplace_back(value);
    return nodes.size() - 1;
  }

  void clear() {
    nodes.clear();
  }

  void zero_adjoints() {
    for (auto& node : nodes) {
      node.adjoint = 0.0;
    }
  }

  void backward(size_t root_idx) {
    if (root_idx >= nodes.size()) return;

    nodes[root_idx].adjoint = 1.0;

    // Reverse pass
    for (int i = static_cast<int>(nodes.size()) - 1; i >= 0; --i) {
      nodes[i].backward(this);
    }
  }
};

// Create a new tape (caller owns the pointer)
inline Tape* create_tape() {
  return new Tape();
}

// Delete a tape
inline void delete_tape(Tape* tape) {
  if (tape != nullptr) {
    delete tape;
  }
}

// Thread-local current tape pointer for Var(double) constructor
// Set by TapeScope, used when creating Var from scalar
inline Tape*& current_tape() {
  static thread_local Tape* tape = nullptr;
  return tape;
}

// RAII wrapper for tape management
// Also sets thread-local current_tape for Var(double) constructor
class TapeScope {
public:
  Tape* tape;
  Tape* prev_tape;  // Save previous tape for nested scopes

  TapeScope() : tape(new Tape()), prev_tape(current_tape()) {
    current_tape() = tape;
  }
  ~TapeScope() {
    current_tape() = prev_tape;  // Restore previous tape
    delete tape;
  }

  // Non-copyable
  TapeScope(const TapeScope&) = delete;
  TapeScope& operator=(const TapeScope&) = delete;

  // Movable
  TapeScope(TapeScope&& other) noexcept : tape(other.tape), prev_tape(other.prev_tape) {
    other.tape = nullptr;
  }
  TapeScope& operator=(TapeScope&& other) noexcept {
    if (this != &other) {
      current_tape() = prev_tape;
      delete tape;
      tape = other.tape;
      prev_tape = other.prev_tape;
      other.tape = nullptr;
      current_tape() = tape;   // adopt the moved-from scope's active tape
    }
    return *this;
  }
};

// ---------------------------------------------------------------------
// Var: Automatic differentiation variable
// ---------------------------------------------------------------------
class Var {
public:
  Tape* tape;
  size_t idx;

  // Default constructor (creates invalid Var)
  Var() : tape(nullptr), idx(0) {}

  // Construct from tape and value (preferred for thread-safe code)
  Var(Tape* t, double value) : tape(t) {
    if (tape != nullptr) {
      idx = tape->add_node(value);
    } else {
      idx = 0;
    }
  }

  // Construct from value using global tape (for backward compatibility)
  // WARNING: Not thread-safe! Use Var(tape, value) in parallel code.
  explicit Var(double value);

  // Get value
  double val() const {
    if (tape != nullptr && idx < tape->nodes.size()) {
      return tape->nodes[idx].value;
    }
    return 0.0;
  }

  // Get adjoint (gradient)
  double adj() const {
    if (tape != nullptr && idx < tape->nodes.size()) {
      return tape->nodes[idx].adjoint;
    }
    return 0.0;
  }

  // Set value
  void set_val(double v) {
    if (tape != nullptr && idx < tape->nodes.size()) {
      tape->nodes[idx].value = v;
    }
  }

  // Compute gradients via backward pass
  void backward() {
    if (tape != nullptr) {
      tape->backward(idx);
    }
  }
};

// ---------------------------------------------------------------------
// Arithmetic operations with gradient tracking
// ---------------------------------------------------------------------

// Addition
inline Var operator+(const Var& a, const Var& b) {
  Tape* tape = a.tape;
  Var result(tape, a.val() + b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint;
    t->nodes[b_idx].adjoint += t->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator+(const Var& a, double b) {
  Tape* tape = a.tape;
  Var result(tape, a.val() + b);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator+(double a, const Var& b) {
  return b + a;
}

// Subtraction
inline Var operator-(const Var& a, const Var& b) {
  Tape* tape = a.tape;
  Var result(tape, a.val() - b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint;
    t->nodes[b_idx].adjoint -= t->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator-(const Var& a, double b) {
  return a + (-b);
}

inline Var operator-(double a, const Var& b) {
  Tape* tape = b.tape;
  Var result(tape, a - b.val());

  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [b_idx, r_idx](Tape* t) {
    t->nodes[b_idx].adjoint -= t->nodes[r_idx].adjoint;
  };

  return result;
}

inline Var operator-(const Var& a) {
  return 0.0 - a;
}

// Multiplication
inline Var operator*(const Var& a, const Var& b) {
  Tape* tape = a.tape;
  Var result(tape, a.val() * b.val());

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;
  double a_val = a.val();
  double b_val = b.val();

  tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, a_val, b_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * b_val;
    t->nodes[b_idx].adjoint += t->nodes[r_idx].adjoint * a_val;
  };

  return result;
}

inline Var operator*(const Var& a, double b) {
  Tape* tape = a.tape;
  Var result(tape, a.val() * b);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, b](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * b;
  };

  return result;
}

inline Var operator*(double a, const Var& b) {
  return b * a;
}

// Division
inline Var operator/(const Var& a, const Var& b) {
  Tape* tape = a.tape;
  double b_val = b.val();
  Var result(tape, a.val() / b_val);

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;
  double a_val = a.val();

  tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, a_val, b_val](Tape* t) {
    double adj = t->nodes[r_idx].adjoint;
    t->nodes[a_idx].adjoint += adj / b_val;
    t->nodes[b_idx].adjoint -= adj * a_val / (b_val * b_val);
  };

  return result;
}

inline Var operator/(const Var& a, double b) {
  return a * (1.0 / b);
}

inline Var operator/(double a, const Var& b) {
  Tape* tape = b.tape;
  double b_val = b.val();
  Var result(tape, a / b_val);

  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [b_idx, r_idx, a, b_val](Tape* t) {
    t->nodes[b_idx].adjoint -= t->nodes[r_idx].adjoint * a / (b_val * b_val);
  };

  return result;
}

// ---------------------------------------------------------------------
// Mathematical functions
// ---------------------------------------------------------------------

inline Var exp(const Var& a) {
  Tape* tape = a.tape;
  double exp_val = std::exp(a.val());
  Var result(tape, exp_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, exp_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * exp_val;
  };

  return result;
}

inline Var log(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  // Protect against log of non-positive values
  // Gradient uses clamped value to avoid division by zero
  double safe_val = (a_val > 1e-15) ? a_val : 1e-15;
  double log_val = (a_val > 0.0) ? std::log(a_val) : -1e10;
  Var result(tape, log_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, safe_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint / safe_val;
  };

  return result;
}

inline Var sqrt(const Var& a) {
  Tape* tape = a.tape;
  double sqrt_val = std::sqrt(a.val());
  Var result(tape, sqrt_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, sqrt_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint / (2.0 * sqrt_val);
  };

  return result;
}

inline Var pow(const Var& a, double p) {
  Tape* tape = a.tape;
  double a_val = a.val();
  double pow_val = std::pow(a_val, p);
  Var result(tape, pow_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, a_val, p](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * p * std::pow(a_val, p - 1.0);
  };

  return result;
}

// Softplus: log(1 + exp(x)) - numerically stable
inline Var softplus(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  double result_val;
  double sigmoid_val;

  if (a_val > 20.0) {
    result_val = a_val;
    sigmoid_val = 1.0;
  } else if (a_val < -20.0) {
    result_val = std::exp(a_val);
    sigmoid_val = result_val;
  } else {
    result_val = std::log1p(std::exp(a_val));
    sigmoid_val = 1.0 / (1.0 + std::exp(-a_val));
  }

  Var result(tape, result_val);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, sigmoid_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * sigmoid_val;
  };

  return result;
}

// Log-sum-exp (numerically stable)
inline Var log_sum_exp(const Var& a, const Var& b) {
  Tape* tape = a.tape;
  double a_val = a.val();
  double b_val = b.val();
  double max_val = std::max(a_val, b_val);
  double result_val = max_val + std::log(std::exp(a_val - max_val) + std::exp(b_val - max_val));

  Var result(tape, result_val);

  size_t a_idx = a.idx;
  size_t b_idx = b.idx;
  size_t r_idx = result.idx;

  // Softmax weights
  double w_a = std::exp(a_val - result_val);
  double w_b = std::exp(b_val - result_val);

  tape->nodes[r_idx].backward = [a_idx, b_idx, r_idx, w_a, w_b](Tape* t) {
    double adj = t->nodes[r_idx].adjoint;
    t->nodes[a_idx].adjoint += adj * w_a;
    t->nodes[b_idx].adjoint += adj * w_b;
  };

  return result;
}

// Logit function: log(x / (1-x))
inline Var logit(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  Var result(tape, std::log(a_val / (1.0 - a_val)));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, a_val](Tape* t) {
    double deriv = 1.0 / (a_val * (1.0 - a_val));
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * deriv;
  };

  return result;
}

// Inverse logit (sigmoid): 1 / (1 + exp(-x))
inline Var inv_logit(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  double sigmoid;

  if (a_val >= 0) {
    sigmoid = 1.0 / (1.0 + std::exp(-a_val));
  } else {
    double exp_a = std::exp(a_val);
    sigmoid = exp_a / (1.0 + exp_a);
  }

  Var result(tape, sigmoid);

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, sigmoid](Tape* t) {
    double deriv = sigmoid * (1.0 - sigmoid);
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * deriv;
  };

  return result;
}

// Log-gamma function
inline Var lgamma(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  Var result(tape, std::lgamma(a_val));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  // Capture digamma value eagerly to avoid R:: call during backward pass
  double digamma_val = tulpa::math::portable_digamma(a_val);
  tape->nodes[r_idx].backward = [a_idx, r_idx, digamma_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint * digamma_val;
  };

  return result;
}

// Log1p: log(1 + x) - numerically stable for small x
inline Var log1p(const Var& a) {
  Tape* tape = a.tape;
  double a_val = a.val();
  Var result(tape, std::log1p(a_val));

  size_t a_idx = a.idx;
  size_t r_idx = result.idx;

  tape->nodes[r_idx].backward = [a_idx, r_idx, a_val](Tape* t) {
    t->nodes[a_idx].adjoint += t->nodes[r_idx].adjoint / (1.0 + a_val);
  };

  return result;
}

// ---------------------------------------------------------------------
// Utility: Vector of Var
// ---------------------------------------------------------------------

inline std::vector<Var> make_vars(Tape* tape, const std::vector<double>& values) {
  std::vector<Var> vars;
  vars.reserve(values.size());
  for (double v : values) {
    vars.emplace_back(tape, v);
  }
  return vars;
}

inline std::vector<double> get_values(const std::vector<Var>& vars) {
  std::vector<double> values;
  values.reserve(vars.size());
  for (const auto& v : vars) {
    values.push_back(v.val());
  }
  return values;
}

inline std::vector<double> get_adjoints(const std::vector<Var>& vars) {
  std::vector<double> adjoints;
  adjoints.reserve(vars.size());
  for (const auto& v : vars) {
    adjoints.push_back(v.adj());
  }
  return adjoints;
}

// ---------------------------------------------------------------------
// Backward compatibility: global tape functions (deprecated)
// These are provided for transition but should not be used in new code
// ---------------------------------------------------------------------

// Deprecated global tape pointer - DO NOT USE IN NEW CODE
// Kept only for backward compatibility during transition
extern Tape* global_tape;

// Deprecated: use TapeScope or create_tape() instead
inline void init_tape() {
  if (global_tape != nullptr) {
    delete global_tape;
  }
  global_tape = new Tape();
}

// Deprecated: use TapeScope or delete_tape() instead
inline void clear_tape() {
  if (global_tape != nullptr) {
    delete global_tape;
    global_tape = nullptr;
  }
}

// Deprecated: Var constructor using global tape
// Use Var(tape, value) instead
inline Var make_var_global(double value) {
  return Var(global_tape, value);
}

// Deprecated: make_vars using global tape
// Use make_vars(tape, values) instead
inline std::vector<Var> make_vars(const std::vector<double>& values) {
  return make_vars(global_tape, values);
}

} // namespace ad
} // namespace tulpa

#endif // TULPA_AUTODIFF_H
