#include "device_dense.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <thread>
#include "tulpa/portable_math.h"
#include "../inst/device/dense_covariance.cu"

#ifdef _WIN32
#define TULPA_DENSE_CALL __stdcall
#else
#define TULPA_DENSE_CALL
#endif

namespace tulpa { namespace device {
namespace dense_detail {
using Matrix = Eigen::MatrixXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Address = unsigned long long;

struct Memory {
    Context& context = Context::instance();
    Status status;
    std::map<int, std::unique_ptr<Buffer>> buffers;
    ~Memory() {
        if (buffers.empty()) return;
        Status cleanup;
        Lease lease(context, cleanup);
        buffers.clear();
    }
    Buffer& reserve(int slot, std::size_t bytes) {
        auto& buffer = buffers[slot];
        if (!buffer || buffer->size() < bytes) {
            buffer.reset();
            buffer.reset(new Buffer(context, bytes, status));
        }
        return *buffer;
    }
    template<class Fn> Fn symbol(Capability capability, const char* name) {
        return reinterpret_cast<Fn>(context.symbol(capability, name, status));
    }
    bool finish(const std::shared_ptr<Policy>& policy) {
        policy->fail(status);
        return static_cast<bool>(status);
    }
};

struct ThreadWork {
    std::map<std::string, std::unique_ptr<DenseFactor>> factors;
    std::map<std::pair<const double*, bool>, std::unique_ptr<DenseBasis>> bases;
    std::weak_ptr<Policy> basis_policy;
    Memory cross, adjoint;
};

ThreadWork& thread_work() {
    static thread_local ThreadWork* current = nullptr;
    if (current) return *current;
    Context::instance();
    struct OwnedWork {
        std::mutex mutex;
        std::map<std::thread::id, std::unique_ptr<ThreadWork>> threads;
    };
    static OwnedWork owned;
    std::lock_guard<std::mutex> lock(owned.mutex);
    auto& work = owned.threads[std::this_thread::get_id()];
    if (!work) work.reset(new ThreadWork);
    current = work.get();
    return *current;
}

Matrix read_matrix(const double* data, int rows, int columns, bool row_major) {
    if (row_major) return Eigen::Map<const RowMatrix>(data, rows, columns);
    return Eigen::Map<const Matrix>(data, rows, columns);
}

bool select(const std::string& operation, std::size_t size, int variant,
            const std::shared_ptr<Policy>& policy, bool& cuda) {
    Status status;
    cuda = policy->select(operation, size, variant, status);
    policy->fail(status);
    return static_cast<bool>(status);
}

Status gemm(Memory& memory, bool ta, bool tb, int rows, int columns, int inner,
            Address a, int lda, Address b, int ldb, Address out) {
    using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, int,
        const double*, const double*, int, const double*, int, const double*, double*, int);
    auto fn = memory.symbol<Fn>(Capability::Blas, "cublasDgemm_v2");
    void* handle = memory.context.handle(Capability::Blas, memory.status);
    const double one = 1, zero = 0;
    if (!memory.status) return memory.status;
    return Status::read("cuBLAS", fn(handle, ta ? 1 : 0, tb ? 1 : 0,
        rows, columns, inner, &one, reinterpret_cast<double*>(a), lda,
        reinterpret_cast<double*>(b), ldb, &zero, reinterpret_cast<double*>(out), rows), "cublasDgemm_v2");
}

Status triangular_solve(Memory& memory, bool right, bool transpose, int rows, int columns,
                        Address lower, int leading_dimension, Address rhs) {
    using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, int, int,
        const double*, const double*, int, double*, int);
    auto fn = memory.symbol<Fn>(Capability::Blas, "cublasDtrsm_v2");
    auto handle = memory.context.handle(Capability::Blas, memory.status);
    const double one = 1;
    if (!memory.status) return memory.status;
    return Status::read("cuBLAS", fn(handle, right ? 1 : 0, 0, transpose ? 1 : 0, 0,
        rows, columns, &one, reinterpret_cast<double*>(lower), leading_dimension,
        reinterpret_cast<double*>(rhs), rows), "cublasDtrsm_v2");
}
}

struct DenseFactor::Impl {
    dense_detail::Memory memory;
    std::shared_ptr<Policy> policy;
    Eigen::LLT<Eigen::MatrixXd> cpu;
    Eigen::MatrixXd lower_cache;
    std::vector<double> times_cache;
    Eigen::MatrixXd prior_cache;
    std::unique_ptr<DenseBasis> projection;
    dense_detail::Address factor_address = 0;
    int n = 0;
    bool cuda = false, valid = false;
    double determinant = std::numeric_limits<double>::quiet_NaN();

    bool begin(int order, const std::string& operation, int variant = 0) {
        valid = false;
        determinant = std::numeric_limits<double>::quiet_NaN();
        lower_cache.resize(0, 0);
        factor_address = 0;
        n = order;
        policy = current_policy();
        return n > 0 && dense_detail::select(operation, n, variant, policy, cuda);
    }
    bool factor_device() {
        using SizeFn = int (TULPA_DENSE_CALL*)(void*, int, int, double*, int, int*);
        using FactorFn = int (TULPA_DENSE_CALL*)(void*, int, int, double*, int, double*, int, int*);
        auto size = memory.symbol<SizeFn>(Capability::Solver, "cusolverDnDpotrf_bufferSize");
        auto factor = memory.symbol<FactorFn>(Capability::Solver, "cusolverDnDpotrf");
        void* handle = memory.context.handle(Capability::Solver, memory.status);
        if (!factor_address) factor_address = memory.reserve(0, sizeof(double) * static_cast<std::size_t>(n) * n).address();
        auto& info = memory.reserve(1, sizeof(int));
        int workspace = 0, code = 0;
        if (memory.status) memory.status = Status::read("cuSOLVER", size(handle, 0, n,
            reinterpret_cast<double*>(factor_address), n, &workspace), "cusolverDnDpotrf_bufferSize");
        auto& work = memory.reserve(2, sizeof(double) * std::max(1, workspace));
        if (memory.status) memory.status = Status::read("cuSOLVER", factor(handle, 0, n,
            reinterpret_cast<double*>(factor_address), n, reinterpret_cast<double*>(work.address()),
            workspace, reinterpret_cast<int*>(info.address())), "cusolverDnDpotrf");
        if (memory.status) memory.status = info.download(&code, sizeof(code));
        if (code < 0) memory.status = Status::read("cuSOLVER", code, "potrf info");
        if (!memory.status || code > 0) return false;
        auto& diagonal = memory.reserve(3, sizeof(double) * n);
        auto a = factor_address, d = diagonal.address();
        void* args[] = {&a, &d, &n};
        Launch shape; shape.grid_x = (n + 255) / 256;
        memory.status = memory.context.launch(kernel_source("dense_ops"), "tulpa_dense_diagonal", shape, args);
        std::vector<double> host(n);
        if (memory.status) memory.status = diagonal.download(host.data(), sizeof(double) * n);
        determinant = 0;
        for (double v : host) {
            if (!(v > 0) || !std::isfinite(v)) return false;
            determinant += 2 * std::log(v);
        }
        return static_cast<bool>(memory.status) && std::isfinite(determinant);
    }
    bool factor_cpu(const Eigen::MatrixXd& matrix) {
        cpu.compute(matrix);
        if (cpu.info() != Eigen::Success) return false;
        const auto diagonal = cpu.matrixLLT().diagonal();
        if (!diagonal.allFinite() || (diagonal.array() <= 0).any()) return false;
        determinant = 2 * diagonal.array().log().sum();
        return std::isfinite(determinant);
    }
};

DenseFactor::DenseFactor() : impl_(new Impl) {}
DenseFactor::~DenseFactor() = default;
bool DenseFactor::factorize(const double* data, int n, bool row_major,
                            const std::string& operation, double nugget) {
    auto& p = *impl_;
    if (!p.begin(n, operation)) return false;
    Eigen::MatrixXd matrix = dense_detail::read_matrix(data, n, n, row_major);
    matrix = matrix.selfadjointView<Eigen::Lower>();
    matrix.diagonal().array() += nugget;
    if (!matrix.allFinite()) return false;
    if (!p.cuda) p.valid = p.factor_cpu(matrix);
    else {
        auto& m = p.memory;
        {
            Lease lease(m.context, m.status);
            auto& input = m.reserve(0, sizeof(double) * matrix.size());
            if (m.status) m.status = input.upload(matrix.data(), sizeof(double) * matrix.size());
            if (m.status) p.valid = p.factor_device();
        }
        if (!m.finish(p.policy)) return false;
    }
    p.policy->record(operation, p.cuda);
    return p.valid;
}

bool DenseFactor::covariance(const std::vector<double>& times, double sigma2, double phi,
    int kind, double period, double nugget, const std::string& operation, int variant) {
    auto& p = *impl_;
    const int n = static_cast<int>(times.size());
    if (!p.begin(n, operation, variant)) return false;
    if (!(sigma2 > 0) || !(phi > 0) || !std::isfinite(sigma2) || !std::isfinite(phi) ||
        !std::isfinite(nugget) || nugget < 0 || kind < 0 || kind > 4 ||
        (kind == 4 && (!(period > 0) || !std::isfinite(period))) ||
        !std::all_of(times.begin(), times.end(), [](double t) { return std::isfinite(t); })) return false;
    if (!p.cuda) {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(n, n);
        for (int j = 0; j < n; ++j) for (int i = j; i < n; ++i) {
            const double d = std::abs(times[i] - times[j]);
            const double s = kind == 4 ? std::sin(3.14159265358979323846 * d / period) : 0;
            matrix(i, j) = tulpa_dense_covariance(d, sigma2, phi, kind, s * s,
                [](double x) { return std::exp(tulpa::math::clamp_exp_arg(x)); });
        }
        matrix.diagonal().array() += nugget;
        p.valid = matrix.allFinite() && p.factor_cpu(matrix);
    } else {
        auto& m = p.memory;
        {
            Lease lease(m.context, m.status);
            auto& input = m.reserve(4, sizeof(double) * n);
            if (m.status && p.times_cache != times) {
                m.status = input.upload(times.data(), sizeof(double) * n);
                if (m.status) p.times_cache = times;
            }
            auto& output = m.reserve(0, sizeof(double) * static_cast<std::size_t>(n) * n);
            auto t = input.address(), a = output.address();
            int order = n;
            double exp_lower = tulpa::math::EXP_ARG_MIN, exp_upper = tulpa::math::EXP_ARG_MAX;
            void* args[] = {&t, &a, &order, &sigma2, &phi, &kind, &period, &nugget, &exp_lower, &exp_upper};
            Launch shape; shape.grid_x = (static_cast<std::size_t>(n) * n + 255) / 256;
            if (m.status) m.status = m.context.launch(kernel_source("dense_covariance"),
                "tulpa_dense_covariance_kernel", shape, args);
            if (m.status) p.valid = p.factor_device();
        }
        if (!m.finish(p.policy)) return false;
    }
    p.policy->record(operation, p.cuda);
    return p.valid;
}

bool DenseFactor::solve(const double* rhs, double* out, int columns, DenseSystem system) {
    auto& p = *impl_;
    if (!p.valid || columns <= 0) return false;
    if (!p.cuda) {
        Eigen::Map<const Eigen::MatrixXd> b(rhs, p.n, columns);
        Eigen::Map<Eigen::MatrixXd> x(out, p.n, columns);
        if (system == DenseSystem::Lower) x = p.cpu.matrixL().solve(b).eval();
        else if (system == DenseSystem::LowerTranspose) x = p.cpu.matrixU().solve(b).eval();
        else x = p.cpu.solve(b).eval();
        return x.allFinite();
    }
    auto& m = p.memory;
    {
        Lease lease(m.context, m.status);
        auto& b = m.reserve(5, sizeof(double) * p.n * columns);
        if (m.status) m.status = b.upload(rhs, sizeof(double) * p.n * columns);
        const auto a = p.factor_address;
        if (system != DenseSystem::Symmetric) {
            if (m.status) m.status = dense_detail::triangular_solve(m, false,
                system == DenseSystem::LowerTranspose, p.n, columns, a, p.n, b.address());
        } else {
            using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, const double*, int, double*, int, int*);
            auto fn = m.symbol<Fn>(Capability::Solver, "cusolverDnDpotrs");
            auto handle = m.context.handle(Capability::Solver, m.status);
            auto& info = *m.buffers.at(1);
            if (m.status) m.status = Status::read("cuSOLVER", fn(handle, 0, p.n, columns,
                reinterpret_cast<double*>(a), p.n, reinterpret_cast<double*>(b.address()), p.n,
                reinterpret_cast<int*>(info.address())), "cusolverDnDpotrs");
            int code = 0;
            if (m.status) m.status = info.download(&code, sizeof(code));
            if (code) m.status = Status::read("cuSOLVER", code, "potrs info");
        }
        if (m.status) m.status = b.download(out, sizeof(double) * p.n * columns);
    }
    return m.finish(p.policy) && Eigen::Map<Eigen::MatrixXd>(out, p.n, columns).allFinite();
}

bool DenseFactor::lower(double* out, bool row_major) {
    auto& p = *impl_;
    if (!p.valid) return false;
    if (!p.lower_cache.size()) {
        if (!p.cuda) p.lower_cache = p.cpu.matrixL();
        else {
            auto& m = p.memory;
            p.lower_cache.resize(p.n, p.n);
            {
                Lease lease(m.context, m.status);
                if (m.status) m.status = m.context.download(p.lower_cache.data(), p.factor_address, sizeof(double) * p.n * p.n);
            }
            if (!m.finish(p.policy)) return false;
            p.lower_cache.triangularView<Eigen::StrictlyUpper>().setZero();
        }
    }
    if (row_major) Eigen::Map<dense_detail::RowMatrix>(out, p.n, p.n) = p.lower_cache;
    else std::copy(p.lower_cache.data(), p.lower_cache.data() + p.lower_cache.size(), out);
    return true;
}
double DenseFactor::log_determinant() const { return impl_->determinant; }
bool DenseFactor::used_cuda() const { return impl_->cuda; }

struct DenseBasis::Impl {
    dense_detail::Memory memory;
    Eigen::MatrixXd basis;
    std::shared_ptr<Policy> policy;
    bool cuda = false, uploaded = false;
    bool immutable;
    const double* source = nullptr;
    bool source_row_major = false;
    std::weak_ptr<Policy> source_policy;
    explicit Impl(bool immutable_) : immutable(immutable_) {}
    bool begin(const double* data, int rows, int columns, bool row_major, const std::string& operation,
               std::size_t dispatch_size, int dispatch_variant) {
        policy = current_policy();
        if (rows <= 0 || columns <= 0 || !dense_detail::select(operation,
            dispatch_size ? dispatch_size : static_cast<std::size_t>(rows) * columns,
            dispatch_variant < 0 ? columns : dispatch_variant, policy, cuda)) return false;
        if (immutable && source == data && basis.rows() == rows && basis.cols() == columns &&
            source_row_major == row_major && source_policy.lock() == policy) return true;
        const auto bind = [&](const auto& input) {
            if (basis.rows() != rows || basis.cols() != columns || !(basis.array() == input.array()).all()) {
                basis = input; uploaded = false;
            }
        };
        if (row_major) bind(Eigen::Map<const dense_detail::RowMatrix>(data, rows, columns));
        else bind(Eigen::Map<const Eigen::MatrixXd>(data, rows, columns));
        source = data;
        source_row_major = row_major;
        source_policy = policy;
        return basis.allFinite();
    }
    void upload() {
        auto& target = memory.reserve(0, sizeof(double) * basis.size());
        if (!uploaded && memory.status) {
            memory.status = target.upload(basis.data(), sizeof(double) * basis.size());
            uploaded = static_cast<bool>(memory.status);
        }
    }
};
DenseBasis::DenseBasis(bool immutable) : impl_(new Impl(immutable)) {}
DenseBasis::~DenseBasis() = default;
bool DenseBasis::apply(const double* basis, int rows, int columns, bool row_major,
                       const double* coefficients, double* out, bool transpose,
                       const std::string& requested_operation, std::size_t dispatch_size, int dispatch_variant) {
    auto& p = *impl_;
    const std::string operation = requested_operation.empty()
        ? (transpose ? "hsgp_transpose" : "hsgp_product") : requested_operation;
    if (!p.begin(basis, rows, columns, row_major, operation, dispatch_size, dispatch_variant)) return false;
    const int n_in = transpose ? rows : columns, n_out = transpose ? columns : rows;
    if (!p.cuda) {
        Eigen::Map<const Eigen::VectorXd> x(coefficients, n_in);
        Eigen::Map<Eigen::VectorXd> y(out, n_out);
        if (transpose) y.noalias() = p.basis.transpose() * x;
        else y.noalias() = p.basis * x;
    } else {
        auto& m = p.memory;
        {
            Lease lease(m.context, m.status);
            p.upload();
            auto& x = m.reserve(1, sizeof(double) * n_in);
            auto& y = m.reserve(2, sizeof(double) * n_out);
            if (m.status) m.status = x.upload(coefficients, sizeof(double) * n_in);
            using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, const double*, const double*, int,
                const double*, int, const double*, double*, int);
            auto fn = m.symbol<Fn>(Capability::Blas, "cublasDgemv_v2");
            auto handle = m.context.handle(Capability::Blas, m.status);
            const double one = 1, zero = 0;
            if (m.status) m.status = Status::read("cuBLAS", fn(handle, transpose ? 1 : 0,
                rows, columns, &one, reinterpret_cast<double*>(m.buffers.at(0)->address()), rows,
                reinterpret_cast<double*>(x.address()), 1, &zero, reinterpret_cast<double*>(y.address()), 1), "cublasDgemv_v2");
            if (m.status) m.status = y.download(out, sizeof(double) * n_out);
        }
        if (!m.finish(p.policy)) return false;
    }
    p.policy->record(operation, p.cuda);
    return Eigen::Map<Eigen::VectorXd>(out, n_out).allFinite();
}

bool DenseBasis::gram(const double* basis, int rows, int columns, bool row_major,
    const double* weights, const double* scale, double* out, const std::string& operation,
    std::size_t dispatch_size, int dispatch_variant) {
    auto& p = *impl_;
    if (!p.begin(basis, rows, columns, row_major, operation, dispatch_size, dispatch_variant)) return false;
    if (!p.cuda) {
        Eigen::MatrixXd a = p.basis;
        if (scale) a *= Eigen::Map<const Eigen::VectorXd>(scale, columns).asDiagonal();
        Eigen::Map<Eigen::MatrixXd>(out, columns, columns).noalias() =
            a.transpose() * Eigen::Map<const Eigen::VectorXd>(weights, rows).asDiagonal() * a;
    } else {
        auto& m = p.memory;
        {
            Lease lease(m.context, m.status);
            p.upload();
            auto& w = m.reserve(3, sizeof(double) * rows);
            auto& s = m.reserve(4, sizeof(double) * columns);
            auto& a = m.reserve(5, sizeof(double) * p.basis.size());
            auto& wa = m.reserve(6, sizeof(double) * p.basis.size());
            auto& result = m.reserve(7, sizeof(double) * columns * columns);
            if (m.status) m.status = w.upload(weights, sizeof(double) * rows);
            if (m.status && scale) m.status = s.upload(scale, sizeof(double) * columns);
            auto bp = m.buffers.at(0)->address(), wp = w.address(), sp = scale ? s.address() : 0ULL;
            auto ap = a.address(), wap = wa.address();
            int symmetric = std::all_of(weights, weights + rows, [](double w) { return w >= 0; });
            void* args[] = {&bp, &wp, &sp, &ap, &wap, &rows, &columns, &symmetric};
            Launch shape; shape.grid_x = (static_cast<std::size_t>(rows) * columns + 255) / 256;
            if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_weight", shape, args);
            if (symmetric) {
                using Fn = int (TULPA_DENSE_CALL*)(void*, int, int, int, int, const double*,
                    const double*, int, const double*, double*, int);
                auto fn = m.symbol<Fn>(Capability::Blas, "cublasDsyrk_v2");
                auto handle = m.context.handle(Capability::Blas, m.status);
                const double one = 1, zero = 0;
                if (m.status) m.status = Status::read("cuBLAS", fn(handle, 0, 1, columns, rows,
                    &one, reinterpret_cast<double*>(ap), rows, &zero,
                    reinterpret_cast<double*>(result.address()), columns), "cublasDsyrk_v2");
            } else if (m.status) m.status = dense_detail::gemm(m, true, false, columns, columns, rows,
                ap, rows, wap, rows, result.address());
            if (m.status && out) m.status = result.download(out, sizeof(double) * columns * columns);
            if (symmetric && m.status && out) for (int j = 0; j < columns; ++j) for (int i = 0; i < j; ++i)
                out[i + static_cast<std::size_t>(j) * columns] = out[j + static_cast<std::size_t>(i) * columns];
        }
        if (!m.finish(p.policy)) return false;
    }
    p.policy->record(operation, p.cuda);
    return !out || Eigen::Map<Eigen::MatrixXd>(out, columns, columns).allFinite();
}

bool DenseFactor::projected(const double* projector, int n, const double* weights,
    const double* residual, const double* prior, bool prior_row_major, double prior_scale,
    double constant_augmentation, double* linear, const std::string& operation) {
    auto& p = *impl_;
    if (!p.begin(n, operation)) return false;
    if (!p.projection) p.projection.reset(new DenseBasis(true));
    auto& projection = *p.projection;
    if (!projection.apply(projector, n, n, true, residual, linear, true, operation, n, 0)) return false;
    Eigen::MatrixXd matrix;
    if (!p.cuda) matrix.resize(n, n);
    if (!projection.gram(projector, n, n, true, weights, nullptr,
                          p.cuda ? nullptr : matrix.data(), operation, n, 0)) return false;
    const Eigen::MatrixXd prior_matrix = dense_detail::read_matrix(prior, n, n, prior_row_major);
    if (!p.cuda) {
        matrix += prior_scale * prior_matrix;
        matrix.array() += constant_augmentation;
        p.valid = matrix.allFinite() && p.factor_cpu(matrix);
    } else {
        auto& m = p.memory;
        {
            Lease lease(m.context, m.status);
            auto& prior_buffer = m.reserve(6, sizeof(double) * static_cast<std::size_t>(n) * n);
            if (p.prior_cache.rows() != n || !(p.prior_cache.array() == prior_matrix.array()).all()) {
                if (m.status) m.status = prior_buffer.upload(prior_matrix.data(), sizeof(double) * prior_matrix.size());
                if (m.status) p.prior_cache = prior_matrix;
            }
            p.factor_address = projection.impl_->memory.buffers.at(7)->address();
            auto a = p.factor_address, b = prior_buffer.address();
            void* args[] = {&a, &b, &n, &prior_scale, &constant_augmentation};
            Launch shape; shape.grid_x = (static_cast<std::size_t>(n) * n + 255) / 256;
            if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_add_prior", shape, args);
            if (m.status) p.valid = p.factor_device();
        }
        if (!m.finish(p.policy)) return false;
    }
    p.policy->record(operation, p.cuda);
    return p.valid;
}

DenseFactor& dense_factor(const std::string& operation) {
    auto& factor = dense_detail::thread_work().factors[operation];
    if (!factor) factor.reset(new DenseFactor);
    return *factor;
}
DenseBasis& dense_basis(const double* identity, bool immutable) {
    auto& work = dense_detail::thread_work();
    const auto policy = current_policy();
    if (work.basis_policy.lock() != policy) { work.bases.clear(); work.basis_policy = policy; }
    auto& basis = work.bases[{identity, immutable}];
    if (!basis) basis.reset(new DenseBasis(immutable));
    return *basis;
}

bool projected_precision(const double* projector, int n, const double* weights,
                          const double* residual, double* matrix, double* linear,
                          const std::string& operation) {
    auto& workspace = dense_basis(projector);
    return workspace.gram(projector, n, n, true, weights, nullptr, matrix, operation, n, 0) &&
        workspace.apply(projector, n, n, true, residual, linear, true, operation, n, 0);
}

bool cross_product(const double* left, int rows, int columns, const double* right,
                   int right_columns, double* out) {
    const auto policy = current_policy();
    bool cuda = false;
    if (!dense_detail::select("hsgp_cross", static_cast<std::size_t>(rows) * columns,
                              cross_variant(columns, right_columns), policy, cuda)) return false;
    if (!cuda) {
        Eigen::Map<Eigen::MatrixXd>(out, columns, right_columns).noalias() =
            Eigen::Map<const Eigen::MatrixXd>(left, rows, columns).transpose() *
            Eigen::Map<const Eigen::MatrixXd>(right, rows, right_columns);
    } else {
        auto& m = dense_detail::thread_work().cross;
        {
            Lease lease(m.context, m.status);
            const std::size_t a_bytes = sizeof(double) * static_cast<std::size_t>(rows) * columns;
            const std::size_t b_bytes = sizeof(double) * static_cast<std::size_t>(rows) * right_columns;
            const std::size_t c_bytes = sizeof(double) * static_cast<std::size_t>(columns) * right_columns;
            auto& a = m.reserve(0, a_bytes); auto& b = m.reserve(1, b_bytes); auto& c = m.reserve(2, c_bytes);
            if (m.status) m.status = a.upload(left, a_bytes);
            if (m.status) m.status = b.upload(right, b_bytes);
            if (m.status) m.status = dense_detail::gemm(m, true, false, columns, right_columns, rows,
                a.address(), rows, b.address(), rows, c.address());
            if (m.status) m.status = c.download(out, c_bytes);
        }
        if (!m.finish(policy)) return false;
    }
    policy->record("hsgp_cross", cuda);
    return Eigen::Map<Eigen::MatrixXd>(out, columns, right_columns).allFinite();
}

int cross_variant(int left_columns, int right_columns) {
    if (left_columns < 0 || right_columns < 0) return -1;
    const long long sum = static_cast<long long>(left_columns) + right_columns;
    if (sum > 65534) return -1;
    const long long code = sum * (sum + 1) / 2 + right_columns;
    return code <= std::numeric_limits<int>::max() ? static_cast<int>(code) : -1;
}

bool cholesky_adjoint(const double* lower_row_major, const double* adjoint_row_major,
                      double* out_row_major, int n) {
    const auto policy = current_policy();
    bool cuda = false;
    if (!dense_detail::select("dense_adjoint", n, 0, policy, cuda)) return false;
    const Eigen::MatrixXd lower = dense_detail::read_matrix(lower_row_major, n, n, true);
    Eigen::MatrixXd gradient = dense_detail::read_matrix(adjoint_row_major, n, n, true);
    gradient.triangularView<Eigen::StrictlyUpper>().setZero();
    Eigen::MatrixXd result(n, n);
    if (!cuda) {
        const Eigen::MatrixXd product = lower.transpose() * gradient;
        Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(n, n);
        phi.triangularView<Eigen::Lower>() = product.triangularView<Eigen::Lower>();
        phi.diagonal() *= 0.5;
        const Eigen::MatrixXd symmetric = 0.5 * (phi + phi.transpose());
        const Eigen::MatrixXd left = lower.transpose().triangularView<Eigen::Upper>().solve(symmetric);
        result = lower.transpose().triangularView<Eigen::Upper>().solve(left.transpose()).transpose();
    } else {
        auto& m = dense_detail::thread_work().adjoint;
        {
            Lease lease(m.context, m.status);
            const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n) * n;
            auto& l = m.reserve(0, bytes);
            auto& g = m.reserve(1, bytes);
            auto& product = m.reserve(2, bytes);
            auto& symmetric = m.reserve(3, bytes);
            if (m.status) m.status = l.upload(lower.data(), bytes);
            if (m.status) m.status = g.upload(gradient.data(), bytes);
            if (m.status) m.status = dense_detail::gemm(m, true, false, n, n, n,
                l.address(), n, g.address(), n, product.address());
            auto p = product.address(), s = symmetric.address();
            void* args[] = {&p, &s, &n};
            Launch shape; shape.grid_x = (static_cast<std::size_t>(n) * n + 255) / 256;
            if (m.status) m.status = m.context.launch(kernel_source("dense_ops"), "tulpa_dense_chol_project", shape, args);
            if (m.status) m.status = dense_detail::triangular_solve(m, false, true, n, n, l.address(), n, s);
            if (m.status) m.status = dense_detail::triangular_solve(m, true, false, n, n, l.address(), n, s);
            if (m.status) m.status = symmetric.download(result.data(), bytes);
        }
        if (!m.finish(policy)) return false;
    }
    Eigen::Map<dense_detail::RowMatrix>(out_row_major, n, n) = result;
    policy->record("dense_adjoint", cuda);
    return result.allFinite();
}

} }
#undef TULPA_DENSE_CALL
