#ifndef TULPA_DEVICE_BACKEND_H
#define TULPA_DEVICE_BACKEND_H

// The engine's device interface. The engine owns the dispatch policy and every
// CPU route; an accelerator package (tulpaGPU) implements Backend and registers
// it through the C-callable "tulpa_device_register". The engine reaches the
// backend only through these virtual interfaces and the backend reaches the
// engine only through Host, so neither DLL calls a non-virtual function of the
// other. No exception crosses the boundary: every route reports a Status.

#include <cstddef>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "tulpa/device_likelihood.h"

// Bump when any type below changes layout or a virtual interface changes.
#define TULPA_DEVICE_BACKEND_ABI 1

namespace tulpa { namespace device {

struct Status {
    int code = 0;
    std::string message;
    explicit operator bool() const { return code == 0; }
    static Status read(const char* library, int code, const char* operation,
                       const std::string& detail = "") {
        if (!code) return {};
        return {code, std::string(library) + "::" + operation + " failed (status " +
            std::to_string(code) + ")" + (detail.empty() ? "" : ": " + detail)};
    }
    void absorb(const Status& other) { if (*this && !other) *this = other; }
};

enum class Capability { Driver, Solver, Blas, Compiler, Sparse };
enum class Choice { Auto, Cpu, Cuda };
enum class DenseSystem { Symmetric, Lower, LowerTranspose };

// The registered device operations and the capabilities each needs. A
// threshold, a stage record and a selection all name one of these.
struct Operation {
    std::string name;
    std::vector<Capability> capabilities;
};
inline const std::vector<Operation>& operations() {
    static const std::vector<Operation> table = {
        {"nngp_cholesky", {Capability::Solver}}, {"axpy", {Capability::Compiler}},
        {"sparse_cholesky", {Capability::Sparse}},
        {"likelihood", {Capability::Compiler, Capability::Blas}},
        {"likelihood_weights", {Capability::Compiler, Capability::Blas}},
        {"dense_gp", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"temporal_gp", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"rsr_gibbs", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"gp_rsr_gibbs", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"dense_spec", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"dense_precision", {Capability::Solver, Capability::Blas, Capability::Compiler}},
        {"hsgp_product", {Capability::Blas}},
        {"hsgp_transpose", {Capability::Blas}},
        {"hsgp_crossprod", {Capability::Blas, Capability::Compiler}},
        {"hsgp_cross", {Capability::Blas}},
        {"dense_adjoint", {Capability::Blas, Capability::Compiler}},
        {"sparse_selected_inverse", {Capability::Sparse, Capability::Compiler}},
        {"laplace_dense", {Capability::Solver, Capability::Compiler}},
        {"laplace_full_csc", {Capability::Solver, Capability::Compiler}},
        {"laplace_dense_inverse", {Capability::Solver, Capability::Compiler}},
        {"laplace_scatter", {Capability::Compiler}},
        {"sparse_uniform", {Capability::Sparse}}
    };
    return table;
}

struct Threshold {
    std::string operation;
    int variant;
    double size;
    double batch_size = 0.0;
};

struct Info {
    bool compiled = true, available = false;
    int count = 0, driver_version = 0, init_count = 0;
    std::string reason, name, compute_capability;
    double memory_mb = 0;
    std::vector<std::string> libraries, library_reasons;
};

// An affine map from reconstructed coefficients to one observation's linear
// predictor, built by evaluating the observation model on these symbols.
struct Projection {
    double constant = 0;
    std::vector<std::pair<int,double>> terms;
    explicit Projection(double value = 0) : constant(value) {}
    static Projection leaf(int index) {
        Projection out; out.terms.emplace_back(index,1); return out;
    }
    Projection& operator+=(const Projection& rhs) {
        constant+=rhs.constant;
        terms.insert(terms.end(),rhs.terms.begin(),rhs.terms.end()); return *this;
    }
    Projection& operator+=(double rhs) { constant+=rhs; return *this; }
};
inline Projection operator+(const Projection& a, const Projection& b) {
    Projection out(a.constant+b.constant);
    out.terms=a.terms; out.terms.insert(out.terms.end(),b.terms.begin(),b.terms.end());
    return out;
}
inline Projection operator-(const Projection& a, const Projection& b) {
    Projection out(a.constant-b.constant); out.terms=a.terms;
    for (const auto& term : b.terms) out.terms.emplace_back(term.first,-term.second);
    return out;
}
inline Projection operator*(const Projection& a, const Projection& b) {
    if (!a.terms.empty() && !b.terms.empty())
        throw std::logic_error("observation projection must be affine in reconstructed fields");
    Projection out(a.constant*b.constant);
    for (const auto& term : a.terms) out.terms.emplace_back(term.first,term.second*b.constant);
    for (const auto& term : b.terms) out.terms.emplace_back(term.first,term.second*a.constant);
    return out;
}
inline Projection operator*(double a, const Projection& b) { return Projection(a)*b; }
inline Projection operator*(const Projection& a, double b) { return a*Projection(b); }

// The dense columns and sparse rows share one coefficient vector. Sparse
// transpose segments reduce each field score without unordered atomic adds.
struct ObservationPlan {
    int n = 0, np = 1, width = 0;
    std::vector<int> beta_start, beta_count, rowptr, column;
    std::vector<double> x, offset, weight;
    void add_design(const double* matrix, int p, int start, bool column_major=false) {
        beta_start.push_back(start); beta_count.push_back(p);
        x.reserve(x.size()+static_cast<std::size_t>(n)*p);
        for (int j=0; j<p; ++j) for (int i=0; i<n; ++i)
            x.push_back(matrix[column_major ? static_cast<std::size_t>(j)*n+i : static_cast<std::size_t>(i)*p+j]);
    }
    void append_row(const Projection& row, double additional_offset=0) {
        if (rowptr.empty()) {
            offset.reserve(static_cast<std::size_t>(n)*np);
            rowptr.reserve(static_cast<std::size_t>(n)*np+1);
            rowptr.push_back(0);
        }
        offset.push_back(row.constant+additional_offset);
        for (const auto& term : row.terms) if (term.second!=0) {
            column.push_back(term.first); weight.push_back(term.second);
        }
        rowptr.push_back(static_cast<int>(column.size()));
    }
};

struct LikelihoodResult {
    std::vector<double> value, gradient, score, curvature, eta;
};

// ---- What the engine offers the backend -------------------------------------

// The fit's dispatch policy. A failure noted here is recorded against the fit
// without unwinding.
class PolicyView {
public:
    virtual bool select(const std::string& operation, std::size_t size, int variant,
                        Status& status) const = 0;
    virtual bool batch_selected(const std::string& operation, std::size_t size,
                                int variant) const = 0;
    virtual void record(const std::string& operation, bool cuda) = 0;
    virtual void note(const Status& status) = 0;
    virtual std::shared_ptr<void> workspace(const void* owner, const void* context,
        const std::string& kind, const std::function<std::shared_ptr<void>()>& create) = 0;
protected:
    ~PolicyView() = default;
};

// The outer-grid cells solved together. submit() groups requests carrying one
// key and runs `process` once per group on one thread; claim() returns shared
// state registered under a key, creating it when no entry accepts the caller.
class BatchView {
public:
    virtual int width() const = 0;
    virtual Status submit(const void* key, void* request,
        const std::function<void(const std::vector<void*>&)>& process) = 0;
    virtual std::shared_ptr<void> claim(const std::string& key,
        const std::function<std::shared_ptr<void>()>& create,
        const std::function<bool(const std::shared_ptr<void>&)>& acquire) = 0;
protected:
    ~BatchView() = default;
};

class Host {
public:
    virtual std::shared_ptr<PolicyView> policy() = 0;
    // The calling thread's batch, or null outside a batched outer grid.
    virtual BatchView* batch() = 0;
protected:
    ~Host() = default;
};

// ---- What the backend offers the engine -------------------------------------
// Matrices are column-major unless a name says otherwise. A route owns its
// device workspace across evaluations; selection and stage records stay with
// the engine.

class DenseBasisRoute {
public:
    virtual ~DenseBasisRoute() = default;
    virtual Status upload(const double* basis, int rows, int columns) = 0;
    virtual Status apply(const double* coefficients, double* out, bool transpose) = 0;
    // A' diag(weights) A with A = basis * diag(scale) (scale may be null). With
    // `out` null the product stays on the device for factor_projected().
    virtual Status gram(const double* weights, const double* scale, double* out) = 0;
};

class DenseFactorRoute {
public:
    virtual ~DenseFactorRoute() = default;
    // `matrix` is the full symmetric matrix. The Laplace route keeps its own
    // stream and joins the batched factorization of the calling thread's batch.
    virtual Status factorize(const double* matrix, int n, const std::string& operation,
                             bool laplace, bool& positive) = 0;
    // Stationary covariance of the time points, `kind` as tulpa_dense_covariance.
    virtual Status covariance(const double* times, int n, double sigma2, double phi,
        int kind, double period, double nugget, bool& positive) = 0;
    // Factors gram + prior_scale * prior + constant_augmentation, the gram
    // product being the one `gram` left on the device.
    virtual Status factor_projected(DenseBasisRoute& gram, const double* prior, int n,
        double prior_scale, double constant_augmentation, bool& positive) = 0;
    virtual Status solve(const double* rhs, double* out, int columns, DenseSystem system) = 0;
    // The lower factor with the strict upper triangle zeroed.
    virtual Status lower(double* out) = 0;
    virtual double log_determinant() const = 0;
};

class DenseScatterRoute {
public:
    virtual ~DenseScatterRoute() = default;
    virtual void bind(const double* row_major_design, int rows, int columns) = 0;
    // X' g and the lower triangle of X' diag(w) X (row-major). `batch` is the
    // cells' scope when batched launches are selected, else null.
    virtual Status apply(const double* scores, const double* weights, double* gradient,
                         double* lower_row_major, BatchView* batch) = 0;
};

// One factor belongs to one worker and one stream. Numeric updates preserve
// the analysis; a caller keeps ownership of a stream it supplies.
class SparseFactorRoute {
public:
    virtual ~SparseFactorRoute() = default;
    virtual Status analyze(int n, const int* p, const int* i, const double* x, void* stream) = 0;
    virtual Status refactor(const double* x, bool& positive_definite) = 0;
    virtual Status solve(const double* b, double* x, int columns) = 0;
    // p/i describe lower CSC entries in the ordering perm; only these entries
    // leave the device.
    virtual Status inverse_entries(const std::vector<int>& p, const std::vector<int>& i,
                                   const std::vector<int>& perm, std::vector<double>& z) = 0;
    virtual Status inverse_diagonal(const std::vector<int>& indices, std::vector<double>& z) = 0;
    virtual double log_determinant() const = 0;
    virtual bool has_log_determinant() const = 0;
    virtual std::size_t factor_bytes() const = 0;
    virtual std::size_t analyses() const = 0;
    virtual const std::vector<int>& permutation() const = 0;
    virtual void* stream() const = 0;
};

// The plan has been validated by the engine; a refreshed plan keeps its layout.
class LikelihoodRoute {
public:
    virtual ~LikelihoodRoute() = default;
    virtual Status refresh_projection(const ObservationPlan& plan) = 0;
    virtual Status evaluate(const DeviceLikelihood& kernel, const std::vector<double>& parameters,
        int chains, bool batched, bool weights, LikelihoodResult& result) = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual Status initialize() = 0;
    virtual Info info() = 0;
    virtual Status supports(Capability capability) = 0;
    virtual std::vector<Threshold> thresholds() = 0;
    virtual Status memory(std::size_t& free, std::size_t& total) = 0;
    virtual std::unique_ptr<DenseBasisRoute> dense_basis() = 0;
    virtual std::unique_ptr<DenseFactorRoute> dense_factor() = 0;
    virtual std::unique_ptr<DenseScatterRoute> dense_scatter() = 0;
    virtual std::unique_ptr<SparseFactorRoute> sparse_factor() = 0;
    virtual std::unique_ptr<LikelihoodRoute> likelihood(const ObservationPlan& plan) = 0;
    virtual Status cross_product(const double* left, int rows, int columns,
        const double* right, int right_columns, double* out) = 0;
    // The adjoint of a lower Cholesky factor, given the factor and the adjoint
    // of the factor (its strict upper triangle zero).
    virtual Status cholesky_adjoint(const double* lower, const double* adjoint,
                                    double* out, int n) = 0;
    // Lower factors of `count` stacked n x n matrices, in place.
    virtual Status cholesky_batch(double* matrices, int n, int count) = 0;
    virtual Status axpy(const double* x, double* y, double a, int n) = 0;
};

// Both DLLs pass standard-library objects by value and by reference, so they
// must agree on these layouts as well as on the interface version.
struct BackendLayout {
    int abi, string_bytes, vector_bytes, function_bytes, shared_bytes;
};
inline BackendLayout backend_layout() {
    return {TULPA_DEVICE_BACKEND_ABI, static_cast<int>(sizeof(std::string)),
        static_cast<int>(sizeof(std::vector<double>)),
        static_cast<int>(sizeof(std::function<void()>)),
        static_cast<int>(sizeof(std::shared_ptr<void>))};
}

// "tulpa_device_register": returns the engine's Host, or null with `reason`
// filled when the layouts differ. "tulpa_device_unregister" detaches a backend;
// routes it already returned stay valid, so the backend keeps its DLL loaded.
using RegisterBackendFn = Host* (*)(Backend* backend, const BackendLayout* layout,
                                    char* reason, int reason_size);
using UnregisterBackendFn = void (*)(Backend* backend);

} }
#endif
