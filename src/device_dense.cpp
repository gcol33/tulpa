#include "device_dense.h"
#include "device_batch.h"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <thread>
#include "tulpa/portable_math.h"
#include "tulpa/dense_covariance.h"

namespace tulpa { namespace device {
namespace dense_detail {
using Matrix = Eigen::MatrixXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct ThreadWork {
    std::map<std::string, std::unique_ptr<DenseFactor>> factors;
    std::map<std::pair<const double*, bool>, std::unique_ptr<DenseBasis>> bases;
    std::weak_ptr<Policy> basis_policy;
};

ThreadWork& thread_work() {
    static thread_local ThreadWork* current = nullptr;
    if (current) return *current;
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

// Records a device failure against the fit; inside an outer-grid cell the
// failure raises CellFailure and the cell is solved again on the CPU.
bool finish(const std::shared_ptr<Policy>& policy, const Status& status) {
    policy->fail(status);
    return static_cast<bool>(status);
}

template<class Route, class Make>
Status ensure(std::unique_ptr<Route>& route, Make make) {
    if (route) return {};
    Backend* b = backend();
    if (!b) return backend_status();
    route = make(*b);
    return route ? Status{} : Status::read("device", -1, "route", "the backend returned no route");
}
}

struct DenseBasis::Impl {
    Eigen::MatrixXd basis;
    std::shared_ptr<Policy> policy;
    std::unique_ptr<DenseBasisRoute> route;
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
    Status upload() {
        Status status = dense_detail::ensure(route, [](Backend& b) { return b.dense_basis(); });
        if (status && !uploaded) {
            status = route->upload(basis.data(), static_cast<int>(basis.rows()), static_cast<int>(basis.cols()));
            uploaded = static_cast<bool>(status);
        }
        return status;
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
        Status status = p.upload();
        if (status) status = p.route->apply(coefficients, out, transpose);
        if (!dense_detail::finish(p.policy, status)) return false;
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
        Status status = p.upload();
        if (status) status = p.route->gram(weights, scale, out);
        if (!dense_detail::finish(p.policy, status)) return false;
    }
    p.policy->record(operation, p.cuda);
    return !out || Eigen::Map<Eigen::MatrixXd>(out, columns, columns).allFinite();
}

struct DenseFactor::Impl {
    std::shared_ptr<Policy> policy;
    std::unique_ptr<DenseFactorRoute> route;
    Eigen::LLT<Eigen::MatrixXd> cpu;
    Eigen::MatrixXd lower_cache;
    std::unique_ptr<DenseBasis> projection;
    int n = 0;
    bool cuda = false, valid = false, laplace = false;
    std::string operation;
    double determinant = std::numeric_limits<double>::quiet_NaN();

    bool begin(int order, const std::string& operation, int variant = 0) {
        valid = false;
        determinant = std::numeric_limits<double>::quiet_NaN();
        lower_cache.resize(0, 0);
        n = order;
        this->operation = operation;
        laplace = operation.compare(0, 8, "laplace_") == 0;
        policy = current_policy();
        return n > 0 && dense_detail::select(operation, n, variant, policy, cuda);
    }
    Status device() { return dense_detail::ensure(route, [](Backend& b) { return b.dense_factor(); }); }
    // A device factor is usable when it succeeded and is positive definite.
    bool commit(Status status, bool positive) {
        if (!dense_detail::finish(policy, status)) return false;
        determinant = positive ? route->log_determinant() : std::numeric_limits<double>::quiet_NaN();
        valid = positive && std::isfinite(determinant);
        policy->record(operation, true);
        return true;
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
                            const std::string& operation, double nugget, int variant) {
    auto& p = *impl_;
    if (!p.begin(n, operation, variant)) return false;
    Eigen::MatrixXd matrix = dense_detail::read_matrix(data, n, n, row_major);
    matrix = matrix.selfadjointView<Eigen::Lower>();
    matrix.diagonal().array() += nugget;
    if (!matrix.allFinite()) return false;
    if (!p.cuda) {
        p.valid = p.factor_cpu(matrix);
        p.policy->record(operation, false);
        return p.valid;
    }
    bool positive = false;
    Status status = p.device();
    if (status) status = p.route->factorize(matrix.data(), n, operation, p.laplace, positive);
    return p.commit(status, positive) && p.valid;
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
        p.policy->record(operation, false);
        return p.valid;
    }
    bool positive = false;
    Status status = p.device();
    if (status) status = p.route->covariance(times.data(), n, sigma2, phi, kind, period, nugget, positive);
    return p.commit(status, positive) && p.valid;
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
    Status status = p.route->solve(rhs, out, columns, system);
    if (status && !Eigen::Map<Eigen::MatrixXd>(out, p.n, columns).allFinite())
        status = Status::read("CUDA", -1, "dense solve", "nonfinite solution");
    return dense_detail::finish(p.policy, status);
}

bool DenseFactor::lower(double* out, bool row_major) {
    auto& p = *impl_;
    if (!p.valid) return false;
    if (!p.lower_cache.size()) {
        if (!p.cuda) p.lower_cache = p.cpu.matrixL();
        else {
            p.lower_cache.resize(p.n, p.n);
            if (!dense_detail::finish(p.policy, p.route->lower(p.lower_cache.data()))) {
                p.lower_cache.resize(0, 0);
                return false;
            }
        }
    }
    if (row_major) Eigen::Map<dense_detail::RowMatrix>(out, p.n, p.n) = p.lower_cache;
    else std::copy(p.lower_cache.data(), p.lower_cache.data() + p.lower_cache.size(), out);
    return true;
}
double DenseFactor::log_determinant() const { return impl_->determinant; }
bool DenseFactor::used_cuda() const { return impl_->cuda; }

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
        p.policy->record(operation, false);
        return p.valid;
    }
    bool positive = false;
    Status status = projection.impl_->cuda && projection.impl_->route ? p.device() :
        Status::read("device", -1, "projected factor", "the projection product is not on the device");
    if (status) status = p.route->factor_projected(*projection.impl_->route, prior_matrix.data(), n,
        prior_scale, constant_augmentation, positive);
    return p.commit(status, positive) && p.valid;
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
        Backend* b = backend();
        const Status status = b ? b->cross_product(left, rows, columns, right, right_columns, out)
                                : backend_status();
        if (!dense_detail::finish(policy, status)) return false;
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
        Backend* b = backend();
        const Status status = b ? b->cholesky_adjoint(lower.data(), gradient.data(), result.data(), n)
                                : backend_status();
        if (!dense_detail::finish(policy, status)) return false;
    }
    Eigen::Map<dense_detail::RowMatrix>(out_row_major, n, n) = result;
    policy->record("dense_adjoint", cuda);
    return result.allFinite();
}

} }
