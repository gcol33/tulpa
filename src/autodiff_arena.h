// autodiff_arena.h
// Arena-based reverse-mode automatic differentiation
//
// Performance: ~10-30x faster than tape-based autodiff (ad::Var)
// Key optimizations:
//   - Structure-of-Arrays (SoA) layout for cache locality
//   - Pre-computed partial derivatives (no std::function closures)
//   - Single contiguous memory allocation (no per-operation heap alloc)
//   - Tight backward pass loop (~5 memory accesses per node)
//
// Thread-safe via ArenaScope RAII (same pattern as TapeScope)
//
// Usage:
//   arena::ArenaScope scope;
//   Arena* ar = scope.arena();
//   auto params_ar = arena::make_vars(ar, param_values);
//   arena::Var lp = compute_log_post_impl(params_ar, data, layout);
//   lp.backward();
//   auto grads = arena::get_adjoints(params_ar);

#ifndef TULPA_AUTODIFF_ARENA_H
#define TULPA_AUTODIFF_ARENA_H

#include <Rcpp.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "portable_math.h"

namespace tulpa {
namespace arena {

// ============================================================================
// Arena: SoA memory pool for computation graph nodes
// ============================================================================
//
// Each node stores:
//   - value (double): forward-pass result
//   - adjoint (double): accumulated gradient (backward pass)
//   - operand_a, operand_b (int32): indices of input nodes (-1 = none)
//   - partial_a, partial_b (double): pre-computed partial derivatives
//
// Node types by operand pattern:
//   - Leaf: operand_a = -1, operand_b = -1 (input variable or constant)
//   - Unary: operand_a >= 0, operand_b = -1 (exp, log, sqrt, etc.)
//   - Binary: operand_a >= 0, operand_b >= 0 (+, -, *, /)

class Arena {
public:
    static constexpr int32_t DEFAULT_CAPACITY = 131072;  // 128K nodes, ~5MB

private:
    // SoA storage (single contiguous allocation for cache locality)
    double* values_;
    double* adjoints_;
    double* partials_a_;
    double* partials_b_;
    int32_t* operand_a_;
    int32_t* operand_b_;

    int32_t size_;
    int32_t capacity_;
    char* block_;  // Owning pointer for the single allocation

    void allocate(int32_t cap) {
        size_t d_bytes = static_cast<size_t>(cap) * 4 * sizeof(double);
        size_t i_bytes = static_cast<size_t>(cap) * 2 * sizeof(int32_t);
        block_ = new char[d_bytes + i_bytes];

        values_    = reinterpret_cast<double*>(block_);
        adjoints_  = values_ + cap;
        partials_a_ = adjoints_ + cap;
        partials_b_ = partials_a_ + cap;
        operand_a_ = reinterpret_cast<int32_t*>(partials_b_ + cap);
        operand_b_ = operand_a_ + cap;

        capacity_ = cap;
    }

public:
    Arena(int32_t initial_capacity = DEFAULT_CAPACITY)
        : size_(0), block_(nullptr) {
        allocate(initial_capacity);
    }

    ~Arena() { delete[] block_; }

    // Non-copyable, non-movable (Vars hold raw Arena* pointers)
    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;

    // Grow capacity (doubles it, copies existing data)
    void grow() {
        int32_t old_cap = capacity_;
        char* old_block = block_;

        // Compute old array pointers (into old_block)
        double* old_vals = reinterpret_cast<double*>(old_block);
        double* old_adjs = old_vals + old_cap;
        double* old_pa   = old_adjs + old_cap;
        double* old_pb   = old_pa + old_cap;
        int32_t* old_oa  = reinterpret_cast<int32_t*>(old_pb + old_cap);
        int32_t* old_ob  = old_oa + old_cap;

        // Allocate new block (2x capacity)
        allocate(old_cap * 2);

        // Copy existing data
        std::memcpy(values_,    old_vals, static_cast<size_t>(size_) * sizeof(double));
        std::memcpy(adjoints_,  old_adjs, static_cast<size_t>(size_) * sizeof(double));
        std::memcpy(partials_a_, old_pa,  static_cast<size_t>(size_) * sizeof(double));
        std::memcpy(partials_b_, old_pb,  static_cast<size_t>(size_) * sizeof(double));
        std::memcpy(operand_a_, old_oa,   static_cast<size_t>(size_) * sizeof(int32_t));
        std::memcpy(operand_b_, old_ob,   static_cast<size_t>(size_) * sizeof(int32_t));

        delete[] old_block;
    }

    // ----- Node creation (forward pass) -----

    // Leaf node: input variable or constant (~1ns)
    int32_t add_leaf(double value) {
        if (size_ >= capacity_) grow();
        int32_t idx = size_++;
        values_[idx] = value;
        operand_a_[idx] = -1;
        operand_b_[idx] = -1;
        return idx;
    }

    // Unary operation: single input with pre-computed partial (~2ns)
    int32_t add_unary(double value, int32_t operand, double partial) {
        if (size_ >= capacity_) grow();
        int32_t idx = size_++;
        values_[idx] = value;
        operand_a_[idx] = operand;
        operand_b_[idx] = -1;
        partials_a_[idx] = partial;
        return idx;
    }

    // Binary operation: two inputs with pre-computed partials (~3ns)
    int32_t add_binary(double value, int32_t op_a, int32_t op_b,
                       double pa, double pb) {
        if (size_ >= capacity_) grow();
        int32_t idx = size_++;
        values_[idx] = value;
        operand_a_[idx] = op_a;
        operand_b_[idx] = op_b;
        partials_a_[idx] = pa;
        partials_b_[idx] = pb;
        return idx;
    }

    // ----- Accessors -----

    double value(int32_t idx) const { return values_[idx]; }
    double adjoint(int32_t idx) const { return adjoints_[idx]; }

    // ----- Backward pass (reverse-mode gradient accumulation) -----

    void backward(int32_t root) {
        // Zero adjoints up to root, then seed root with 1.0
        std::memset(adjoints_, 0, static_cast<size_t>(root + 1) * sizeof(double));
        adjoints_[root] = 1.0;

        // Tight reverse loop: ~5 memory accesses per node
        for (int32_t i = root; i >= 0; --i) {
            double adj = adjoints_[i];
            if (adj == 0.0) continue;  // Skip dead nodes

            int32_t oa = operand_a_[i];
            if (oa >= 0) {
                adjoints_[oa] += adj * partials_a_[i];
                int32_t ob = operand_b_[i];
                if (ob >= 0) {
                    adjoints_[ob] += adj * partials_b_[i];
                }
            }
        }
    }

    // Reset for reuse (no deallocation, ~0ns)
    void reset() { size_ = 0; }

    int32_t size() const { return size_; }
    int32_t capacity() const { return capacity_; }
};

// ============================================================================
// Thread-local arena pointer (set by ArenaScope)
// ============================================================================

inline Arena*& current_arena() {
    static thread_local Arena* arena = nullptr;
    return arena;
}

// ============================================================================
// ArenaScope: RAII wrapper for arena lifecycle
// ============================================================================
//
// Creates an arena on construction, sets it as the current thread-local arena.
// Restores the previous arena on destruction (supports nesting).

class ArenaScope {
    Arena arena_;
    Arena* prev_;

public:
    ArenaScope(int32_t capacity = Arena::DEFAULT_CAPACITY)
        : arena_(capacity), prev_(current_arena()) {
        current_arena() = &arena_;
    }

    ~ArenaScope() {
        current_arena() = prev_;
    }

    // Non-copyable, non-movable
    ArenaScope(const ArenaScope&) = delete;
    ArenaScope& operator=(const ArenaScope&) = delete;

    Arena* arena() { return &arena_; }
};

// ============================================================================
// Var: lightweight autodiff variable (12 bytes: Arena* + int32_t)
// ============================================================================
//
// Same interface as ad::Var for template compatibility:
//   - val() returns double value
//   - adj() returns double adjoint (gradient)
//   - backward() triggers reverse-mode gradient computation
//   - All arithmetic operators and math functions supported
//
// Plugs directly into compute_log_post_impl<T>() with zero changes.

class Var {
public:
    Arena* arena_;
    int32_t idx_;

    // Default constructor (invalid var — for vector initialization)
    Var() : arena_(nullptr), idx_(0) {}

    // Construct from value using thread-local current arena
    // Used by template code: T(0.0), T(1.0), T(y + 1.0), etc.
    explicit Var(double value) : arena_(current_arena()), idx_(0) {
        if (arena_) idx_ = arena_->add_leaf(value);
    }

    // Construct from arena and value (preferred for explicit arena usage)
    Var(Arena* a, double value) : arena_(a), idx_(a->add_leaf(value)) {}

    double val() const { return arena_ ? arena_->value(idx_) : 0.0; }
    double adj() const { return arena_ ? arena_->adjoint(idx_) : 0.0; }

    void backward() { if (arena_) arena_->backward(idx_); }

    // Compound assignment operators (create new nodes, update this to point to them)
    Var& operator+=(const Var& rhs);
    Var& operator-=(const Var& rhs);
    Var& operator*=(const Var& rhs);
    Var& operator/=(const Var& rhs);
    Var& operator+=(double rhs);
    Var& operator-=(double rhs);
    Var& operator*=(double rhs);
    Var& operator/=(double rhs);
};

// ============================================================================
// Arithmetic operators: Var op Var
// ============================================================================

inline Var operator+(const Var& a, const Var& b) {
    Arena* ar = a.arena_;
    // d(a+b)/da = 1, d(a+b)/db = 1
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_binary(a.val() + b.val(), a.idx_, b.idx_, 1.0, 1.0);
    return r;
}

inline Var operator-(const Var& a, const Var& b) {
    Arena* ar = a.arena_;
    // d(a-b)/da = 1, d(a-b)/db = -1
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_binary(a.val() - b.val(), a.idx_, b.idx_, 1.0, -1.0);
    return r;
}

inline Var operator*(const Var& a, const Var& b) {
    Arena* ar = a.arena_;
    // d(a*b)/da = b, d(a*b)/db = a
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_binary(a.val() * b.val(), a.idx_, b.idx_, b.val(), a.val());
    return r;
}

inline Var operator/(const Var& a, const Var& b) {
    Arena* ar = a.arena_;
    double bv = b.val();
    double av = a.val();
    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_binary(av / bv, a.idx_, b.idx_, 1.0 / bv, -av / (bv * bv));
    return r;
}

// ============================================================================
// Arithmetic operators: Var op double
// ============================================================================

inline Var operator+(const Var& a, double b) {
    Arena* ar = a.arena_;
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a.val() + b, a.idx_, 1.0);
    return r;
}

inline Var operator-(const Var& a, double b) {
    Arena* ar = a.arena_;
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a.val() - b, a.idx_, 1.0);
    return r;
}

inline Var operator*(const Var& a, double b) {
    Arena* ar = a.arena_;
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a.val() * b, a.idx_, b);
    return r;
}

inline Var operator/(const Var& a, double b) {
    Arena* ar = a.arena_;
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a.val() / b, a.idx_, 1.0 / b);
    return r;
}

// ============================================================================
// Arithmetic operators: double op Var
// ============================================================================

inline Var operator+(double a, const Var& b) {
    return b + a;
}

inline Var operator-(double a, const Var& b) {
    Arena* ar = b.arena_;
    // d(a-b)/db = -1
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a - b.val(), b.idx_, -1.0);
    return r;
}

inline Var operator*(double a, const Var& b) {
    return b * a;
}

inline Var operator/(double a, const Var& b) {
    Arena* ar = b.arena_;
    double bv = b.val();
    // d(a/b)/db = -a/b^2
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(a / bv, b.idx_, -a / (bv * bv));
    return r;
}

// ============================================================================
// Arithmetic operators: Var op int, int op Var
// ============================================================================

inline Var operator+(const Var& a, int b) { return a + static_cast<double>(b); }
inline Var operator-(const Var& a, int b) { return a - static_cast<double>(b); }
inline Var operator*(const Var& a, int b) { return a * static_cast<double>(b); }
inline Var operator/(const Var& a, int b) { return a / static_cast<double>(b); }

inline Var operator+(int a, const Var& b) { return static_cast<double>(a) + b; }
inline Var operator-(int a, const Var& b) { return static_cast<double>(a) - b; }
inline Var operator*(int a, const Var& b) { return static_cast<double>(a) * b; }
inline Var operator/(int a, const Var& b) { return static_cast<double>(a) / b; }

// ============================================================================
// Unary minus
// ============================================================================

inline Var operator-(const Var& a) {
    Arena* ar = a.arena_;
    // d(-a)/da = -1
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(-a.val(), a.idx_, -1.0);
    return r;
}

// ============================================================================
// Compound assignment implementations
// ============================================================================

inline Var& Var::operator+=(const Var& rhs) { *this = *this + rhs; return *this; }
inline Var& Var::operator-=(const Var& rhs) { *this = *this - rhs; return *this; }
inline Var& Var::operator*=(const Var& rhs) { *this = *this * rhs; return *this; }
inline Var& Var::operator/=(const Var& rhs) { *this = *this / rhs; return *this; }
inline Var& Var::operator+=(double rhs) { *this = *this + rhs; return *this; }
inline Var& Var::operator-=(double rhs) { *this = *this - rhs; return *this; }
inline Var& Var::operator*=(double rhs) { *this = *this * rhs; return *this; }
inline Var& Var::operator/=(double rhs) { *this = *this / rhs; return *this; }

// ============================================================================
// Comparison operators (value-based, no gradient tracking)
// ============================================================================

inline bool operator<(const Var& a, const Var& b)  { return a.val() < b.val(); }
inline bool operator>(const Var& a, const Var& b)  { return a.val() > b.val(); }
inline bool operator<=(const Var& a, const Var& b) { return a.val() <= b.val(); }
inline bool operator>=(const Var& a, const Var& b) { return a.val() >= b.val(); }
inline bool operator==(const Var& a, const Var& b) { return a.val() == b.val(); }
inline bool operator!=(const Var& a, const Var& b) { return a.val() != b.val(); }

inline bool operator<(const Var& a, double b)  { return a.val() < b; }
inline bool operator>(const Var& a, double b)  { return a.val() > b; }
inline bool operator<=(const Var& a, double b) { return a.val() <= b; }
inline bool operator>=(const Var& a, double b) { return a.val() >= b; }
inline bool operator==(const Var& a, double b) { return a.val() == b; }
inline bool operator!=(const Var& a, double b) { return a.val() != b; }

inline bool operator<(double a, const Var& b)  { return a < b.val(); }
inline bool operator>(double a, const Var& b)  { return a > b.val(); }
inline bool operator<=(double a, const Var& b) { return a <= b.val(); }
inline bool operator>=(double a, const Var& b) { return a >= b.val(); }
inline bool operator==(double a, const Var& b) { return a == b.val(); }
inline bool operator!=(double a, const Var& b) { return a != b.val(); }

// ============================================================================
// Math functions
// ============================================================================

inline Var exp(const Var& a) {
    Arena* ar = a.arena_;
    double e = std::exp(a.val());
    // d(exp(x))/dx = exp(x)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(e, a.idx_, e);
    return r;
}

inline Var log(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    double safe_val = (av > 1e-15) ? av : 1e-15;
    double log_val = (av > 0.0) ? std::log(av) : -1e10;
    // d(log(x))/dx = 1/x
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(log_val, a.idx_, 1.0 / safe_val);
    return r;
}

inline Var sqrt(const Var& a) {
    Arena* ar = a.arena_;
    double s = std::sqrt(a.val());
    // d(sqrt(x))/dx = 1 / (2*sqrt(x))
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(s, a.idx_, 0.5 / s);
    return r;
}

inline Var pow(const Var& a, double p) {
    Arena* ar = a.arena_;
    double av = a.val();
    double pv = std::pow(av, p);
    // d(x^p)/dx = p * x^(p-1)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(pv, a.idx_, p * std::pow(av, p - 1.0));
    return r;
}

inline Var softplus(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    double result_val, sigmoid_val;

    if (av > 20.0) {
        result_val = av;
        sigmoid_val = 1.0;
    } else if (av < -20.0) {
        result_val = std::exp(av);
        sigmoid_val = result_val;
    } else {
        result_val = std::log1p(std::exp(av));
        sigmoid_val = 1.0 / (1.0 + std::exp(-av));
    }

    // d(softplus(x))/dx = sigmoid(x)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(result_val, a.idx_, sigmoid_val);
    return r;
}

inline Var log_sum_exp(const Var& a, const Var& b) {
    Arena* ar = a.arena_;
    double av = a.val();
    double bv = b.val();
    double max_val = std::max(av, bv);
    double result_val = max_val + std::log(std::exp(av - max_val) + std::exp(bv - max_val));

    // Softmax weights: d(lse)/da = exp(a) / (exp(a)+exp(b))
    double w_a = std::exp(av - result_val);
    double w_b = std::exp(bv - result_val);

    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_binary(result_val, a.idx_, b.idx_, w_a, w_b);
    return r;
}

inline Var logit(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    double result_val = std::log(av / (1.0 - av));
    // d(logit(x))/dx = 1 / (x * (1-x))
    double deriv = 1.0 / (av * (1.0 - av));

    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(result_val, a.idx_, deriv);
    return r;
}

inline Var inv_logit(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    double sigmoid;

    if (av >= 0) {
        sigmoid = 1.0 / (1.0 + std::exp(-av));
    } else {
        double exp_a = std::exp(av);
        sigmoid = exp_a / (1.0 + exp_a);
    }

    // d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(sigmoid, a.idx_, sigmoid * (1.0 - sigmoid));
    return r;
}

inline Var lgamma(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    // d(lgamma(x))/dx = digamma(x)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(std::lgamma(av), a.idx_, tulpa::math::portable_digamma(av));
    return r;
}

inline Var log1p(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    // d(log(1+x))/dx = 1 / (1+x)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(std::log1p(av), a.idx_, 1.0 / (1.0 + av));
    return r;
}

inline Var digamma(const Var& a) {
    Arena* ar = a.arena_;
    double av = a.val();
    // d(digamma(x))/dx = trigamma(x)
    Var r;
    r.arena_ = ar;
    r.idx_ = ar->add_unary(tulpa::math::portable_digamma(av), a.idx_, tulpa::math::portable_trigamma(av));
    return r;
}

inline Var abs(const Var& a) {
    return (a.val() >= 0) ? a : -a;
}

// ============================================================================
// Utility functions
// ============================================================================

// Create Var vector from double values (using explicit arena)
inline std::vector<Var> make_vars(Arena* arena, const std::vector<double>& values) {
    std::vector<Var> vars;
    vars.reserve(values.size());
    for (double v : values) {
        vars.emplace_back(arena, v);
    }
    return vars;
}

// Extract values from Var vector
inline std::vector<double> get_values(const std::vector<Var>& vars) {
    std::vector<double> values;
    values.reserve(vars.size());
    for (const auto& v : vars) {
        values.push_back(v.val());
    }
    return values;
}

// Extract adjoints (gradients) from Var vector
inline std::vector<double> get_adjoints(const std::vector<Var>& vars) {
    std::vector<double> adjoints;
    adjoints.reserve(vars.size());
    for (const auto& v : vars) {
        adjoints.push_back(v.adj());
    }
    return adjoints;
}

// Value extraction (for template compatibility)
inline double get_value(const Var& x) {
    return x.val();
}

// Dot product with double array and Var coefficients
inline Var dot_product_mixed(const double* x, const Var* y, int n) {
    Var sum(0.0);
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

inline Var dot_product_mixed(const double* x, const std::vector<Var>& y, int n) {
    Var sum(0.0);
    for (int i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

}  // namespace arena
}  // namespace tulpa

#endif  // TULPA_AUTODIFF_ARENA_H
