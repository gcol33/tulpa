#include <RcppEigen.h>
#include "device_dense.h"
#include "dense_basis_apply.h"
#include "temporal_gp_kernel.h"
#include "hsgp_field.h"
#include "dense_basis_products.h"
#include <chrono>
#include <thread>

namespace tulpa_dense_fixture {
struct Scope {
    std::shared_ptr<tulpa::device::Policy> policy, previous;
    explicit Scope(const std::string& choice) {
        policy = std::make_shared<tulpa::device::Policy>(tulpa::device::parse_choice(choice), tulpa::device::thresholds());
        previous = tulpa::device::exchange_policy(policy);
    }
    ~Scope() { tulpa::device::exchange_policy(previous); }
    void check() const { const auto s = policy->failure(); if (!s) Rcpp::stop(s.message); }
};

template<class T>
T temporal(const std::vector<T>& q, const std::vector<double>& times, int kind,
           bool centered, double relative_nugget, bool tvc) {
    using tulpa::math::safe_exp;
    const int n = times.size();
    const auto cov = kind == 3 ? tulpa::TemporalCovType::GAUSSIAN
        : kind == 4 ? tulpa::TemporalCovType::PERIODIC : tulpa::TemporalCovType::MATERN;
    const double nu = kind == 1 ? 1.5 : 2.5;
    tulpa_temporal_gp::FieldKernel<T> kernel(times, n, safe_exp(q[n]), safe_exp(q[n + 1]),
        cov, nu, 1.7, relative_nugget, tvc, !centered);
    if (!kernel.valid()) return T(-INFINITY);
    std::vector<T> field(q.begin(), q.begin() + n);
    T value(0.0);
    kernel.accumulate_log_density(field, 0, !centered, value);
    if (tvc) {
        T mean(0.0);
        for (const auto& v : field) mean = mean + v / T(n);
        for (auto& v : field) v = v - mean;
    }
    for (int i = 0; i < n; ++i) {
        const T residual = field[i] - T(std::cos(i * 0.13));
        value = value - T(0.5) * residual * residual;
    }
    return value;
}

template<class T>
T hsgp(const tulpa::HSGPData& basis, const std::vector<T>& params) {
    std::vector<T> field;
    tulpa::priors::evaluate_hsgp_field(params.data(), tulpa::math::safe_exp(params[basis.m_total]),
        tulpa::math::safe_exp(params[basis.m_total + 1]), basis, field);
    T value(0.0);
    for (int j = 0; j < basis.m_total; ++j) value = value - T(0.5) * params[j] * params[j];
    for (int i = 0; i < basis.n_obs; ++i) {
        const T residual = field[i] - T(std::sin(i * 0.13));
        value = value - T(0.5) * residual * residual;
    }
    return value;
}
}

// [[Rcpp::export]]
Rcpp::List cpp_device_dense(const Eigen::MatrixXd& matrix, const Eigen::MatrixXd& rhs,
    std::string operation, std::string choice, int repetitions = 1) {
    if (!matrix.rows() || matrix.rows() != matrix.cols() || rhs.rows() != matrix.rows() ||
        rhs.cols() < 1 || repetitions < 1) Rcpp::stop("invalid dense fixture dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    tulpa::device::DenseFactor factor;
    Eigen::MatrixXd solution(rhs.rows(), rhs.cols());
    const tulpa_coordinate_linalg::Matrix input = matrix;
    tulpa_coordinate_linalg::Matrix lower(matrix.rows(), matrix.cols());
    const auto evaluate = [&]() {
        const bool ok = factor.factorize(input.data(), matrix.rows(), true, operation);
        scope.check();
        if (!ok) Rcpp::stop("dense fixture matrix is not positive definite");
        if ((operation == "dense_gp" || operation == "dense_spec") &&
                !factor.lower(lower.data(), operation != "dense_spec")) {
            scope.check(); Rcpp::stop("dense fixture lower download failed");
        }
        if (!factor.solve(rhs.data(), solution.data(), rhs.cols())) {
            scope.check(); Rcpp::stop("dense fixture solve failed");
        }
    };
    const auto first_start = std::chrono::steady_clock::now();
    evaluate();
    const double first_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - first_start).count();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) evaluate();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    if (!factor.lower(lower.data(), true)) { scope.check(); Rcpp::stop("dense fixture lower download failed"); }
    Eigen::VectorXd z(matrix.rows()), perturbation(matrix.rows());
    for (int i = 0; i < z.size(); ++i) z[i] = std::sin(0.31 * i);
    if (!factor.solve(z.data(), perturbation.data(), 1, tulpa::device::DenseSystem::LowerTranspose)) {
        scope.check(); Rcpp::stop("dense fixture draw failed");
    }
    scope.check();
    return Rcpp::List::create(Rcpp::_["lower"] = lower, Rcpp::_["solution"] = solution,
        Rcpp::_["logdet"] = factor.log_determinant(), Rcpp::_["draw"] = (solution.col(0) + perturbation).eval(),
        Rcpp::_["log_density"] = -0.5 * (factor.log_determinant() + (rhs.col(0).dot(solution.col(0)))),
        Rcpp::_["gradient"] = (-solution.col(0)).eval(),
        Rcpp::_["used_cuda"] = factor.used_cuda(), Rcpp::_["seconds"] = seconds,
        Rcpp::_["first_seconds"] = first_seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_temporal_dense(Rcpp::NumericVector times, Rcpp::NumericVector q,
    int kind, bool centered, double relative_nugget, bool tvc, std::string choice,
    int repetitions = 1) {
    if (times.size() < 1 || q.size() != times.size() + 2 || repetitions < 1 || kind < 1 || kind > 4)
        Rcpp::stop("invalid temporal dense fixture dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    const std::vector<double> t(times.begin(), times.end()), x(q.begin(), q.end());
    std::vector<double> gradient(q.size());
    double value = 0, arena_value = 0;
    const auto evaluate = [&]() {
        tulpa::arena::ArenaScope arena;
        std::vector<tulpa::arena::Var> variables;
        for (double v : x) variables.emplace_back(v);
        auto result = tulpa_dense_fixture::temporal(variables, t, kind, centered, relative_nugget, tvc);
        result.backward();
        arena_value = result.val();
        for (std::size_t i = 0; i < variables.size(); ++i) gradient[i] = variables[i].adj();
    };
    const auto first_start = std::chrono::steady_clock::now();
    evaluate();
    const double first_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - first_start).count();
    const auto start = std::chrono::steady_clock::now();
    for (int trial = 0; trial < repetitions; ++trial) evaluate();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    value = tulpa_dense_fixture::temporal(x, t, kind, centered, relative_nugget, tvc);
    scope.check();
    const bool used = tulpa::device::dense_factor("temporal_gp").used_cuda();
    return Rcpp::List::create(Rcpp::_["log_density"] = value, Rcpp::_["arena_value"] = arena_value,
        Rcpp::_["gradient"] = gradient, Rcpp::_["seconds"] = seconds, Rcpp::_["used_cuda"] = used,
        Rcpp::_["first_seconds"] = first_seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_hsgp_dense(const Eigen::MatrixXd& basis, const Eigen::VectorXd& weights,
    const Eigen::VectorXd& scale, const Eigen::VectorXd& beta, std::string choice, int repetitions = 1,
    std::string action = "all") {
    if (!basis.rows() || !basis.cols() || weights.size() != basis.rows() ||
        scale.size() != basis.cols() || beta.size() != basis.cols() || repetitions < 1)
        Rcpp::stop("invalid HSGP fixture dimensions");
    if (action != "all" && action != "product" && action != "transpose" && action != "crossprod") Rcpp::stop("invalid HSGP fixture action");
    tulpa_dense_fixture::Scope scope(choice);
    tulpa::device::DenseBasis workspace;
    const tulpa_coordinate_linalg::Matrix stored_basis = basis;
    Eigen::MatrixXd gram(basis.cols(), basis.cols());
    Eigen::VectorXd field(basis.rows()), gradient(basis.cols());
    const Eigen::VectorXd scaled_beta = scale.array() * beta.array();
    Eigen::VectorXd transpose_score;
    if (action == "transpose") {
        field.noalias() = basis * scaled_beta;
        transpose_score = -weights.array() * field.array();
    }
    const auto evaluate = [&]() {
        bool ok = true;
        if (action == "all" || action == "crossprod") ok = workspace.gram(stored_basis.data(), basis.rows(), basis.cols(), true,
            weights.data(), scale.data(), gram.data());
        if (action == "all" || action == "product") ok = ok && workspace.apply(stored_basis.data(), basis.rows(), basis.cols(), true, scaled_beta.data(), field.data());
        if (!ok) { scope.check(); Rcpp::stop("HSGP fixture product failed"); }
        if (action == "all" || action == "transpose") {
            const Eigen::VectorXd score = action == "transpose" ? transpose_score :
                Eigen::VectorXd((-weights.array() * field.array()).matrix());
            if (!workspace.apply(stored_basis.data(), basis.rows(), basis.cols(), true, score.data(), gradient.data(), true)) {
                scope.check(); Rcpp::stop("HSGP fixture gradient failed");
            }
            gradient.array() *= scale.array();
        }
    };
    const auto first_start = std::chrono::steady_clock::now();
    evaluate();
    const double first_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - first_start).count();
    const auto start = std::chrono::steady_clock::now();
    for (int trial = 0; trial < repetitions; ++trial) evaluate();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    scope.check();
    Rcpp::List result = Rcpp::List::create(Rcpp::_["seconds"] = seconds, Rcpp::_["first_seconds"] = first_seconds);
    if (action == "all" || action == "crossprod") result["gram"] = gram;
    if (action == "all" || action == "product") {
        result["field"] = field;
        result["log_density"] = -0.5 * field.dot((weights.array() * field.array()).matrix());
    }
    if (action == "all" || action == "transpose") result["gradient"] = gradient;
    return result;
}

// [[Rcpp::export]]
Rcpp::List cpp_device_dense_adjoint(const Eigen::MatrixXd& lower, const Eigen::MatrixXd& adjoint,
    std::string choice, int repetitions = 1) {
    if (!lower.rows() || lower.rows() != lower.cols() || adjoint.rows() != lower.rows() ||
        adjoint.cols() != lower.cols() || repetitions < 1) Rcpp::stop("invalid dense adjoint dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    using Matrix = tulpa_coordinate_linalg::Matrix;
    const Matrix l = lower, g = adjoint;
    Matrix out(l.rows(), l.cols());
    const auto start = std::chrono::steady_clock::now();
    for (int trial = 0; trial < repetitions; ++trial) {
        if (!tulpa::device::cholesky_adjoint(l.data(), g.data(), out.data(), l.rows())) {
            scope.check(); Rcpp::stop("dense adjoint failed");
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    scope.check();
    return Rcpp::List::create(Rcpp::_["gradient"] = out, Rcpp::_["seconds"] = seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_projected_dense(const Eigen::MatrixXd& projector, const Eigen::MatrixXd& prior,
    const Eigen::VectorXd& weights, const Eigen::VectorXd& residual, std::string operation,
    std::string choice, int repetitions = 1, bool changing_prior = false) {
    const int n = projector.rows();
    if (n < 1 || projector.cols() != n || prior.rows() != n || prior.cols() != n ||
        weights.size() != n || residual.size() != n || repetitions < 1) Rcpp::stop("invalid projected fixture dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    const tulpa_coordinate_linalg::Matrix p = projector;
    Eigen::MatrixXd precision(n, n);
    Eigen::VectorXd linear(n), mean(n), perturbation(n), z(n);
    for (int i = 0; i < n; ++i) z[i] = std::sin(0.31 * i);
    tulpa::device::DenseFactor factor;
    Eigen::MatrixXd current_prior = prior;
    int evaluation = 0;
    const auto evaluate = [&]() {
        if (changing_prior) current_prior = (1.0 + 0.01 * std::sin(evaluation++)) * prior;
        if (!factor.projected(p.data(), n, weights.data(), residual.data(), current_prior.data(), false,
                1.0, 0.0, linear.data(), operation) ||
            !factor.solve(linear.data(), mean.data()) ||
            !factor.solve(z.data(), perturbation.data(), 1, tulpa::device::DenseSystem::LowerTranspose)) {
            scope.check(); Rcpp::stop("projected factor or solve failed");
        }
    };
    const auto first_start = std::chrono::steady_clock::now();
    evaluate();
    const double first_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - first_start).count();
    const auto start = std::chrono::steady_clock::now();
    for (int trial = 0; trial < repetitions; ++trial) evaluate();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    scope.check();
    if (!tulpa::device::projected_precision(p.data(), n, weights.data(), residual.data(),
            precision.data(), linear.data(), operation)) { scope.check(); Rcpp::stop("projected precision read failed"); }
    precision += current_prior;
    const Eigen::VectorXd draw = mean + perturbation;
    const Eigen::VectorXd gradient = linear - precision * draw;
    return Rcpp::List::create(Rcpp::_["precision"] = precision, Rcpp::_["linear"] = linear,
        Rcpp::_["mean"] = mean, Rcpp::_["draw"] = draw, Rcpp::_["gradient"] = gradient,
        Rcpp::_["logdet"] = factor.log_determinant(), Rcpp::_["log_density"] =
            0.5 * factor.log_determinant() - 0.5 * perturbation.dot(precision * perturbation),
        Rcpp::_["used_cuda"] = factor.used_cuda(), Rcpp::_["seconds"] = seconds,
        Rcpp::_["first_seconds"] = first_seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_hsgp_sampler(const Eigen::MatrixXd& basis, Rcpp::NumericVector eigenvalues,
    Rcpp::NumericVector parameters, int kernel, std::string choice) {
    if (!basis.rows() || !basis.cols() || eigenvalues.size() != basis.cols() ||
        parameters.size() != basis.cols() + 2 || kernel < 0 || kernel > 2) Rcpp::stop("invalid HSGP sampler dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    tulpa::HSGPData data;
    data.n_obs = basis.rows(); data.m_total = basis.cols(); data.kernel = kernel;
    const tulpa_coordinate_linalg::Matrix row_basis = basis;
    data.phi_flat.assign(row_basis.data(), row_basis.data() + row_basis.size());
    data.eigenvalues.assign(eigenvalues.begin(), eigenvalues.end());
    const std::vector<double> q(parameters.begin(), parameters.end());
    const double value = tulpa_dense_fixture::hsgp(data, q);
    tulpa::arena::ArenaScope arena;
    std::vector<tulpa::arena::Var> variables;
    for (double v : q) variables.emplace_back(v);
    auto result = tulpa_dense_fixture::hsgp(data, variables);
    result.backward();
    std::vector<double> gradient(q.size());
    for (std::size_t j = 0; j < q.size(); ++j) gradient[j] = variables[j].adj();
    scope.check();
    return Rcpp::List::create(Rcpp::_["log_density"] = value,
        Rcpp::_["arena_value"] = result.val(), Rcpp::_["gradient"] = gradient);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_hsgp_cross(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right,
    std::string choice, int repetitions = 1) {
    if (!left.rows() || !left.cols() || right.rows() != left.rows() || !right.cols() || repetitions < 1)
        Rcpp::stop("invalid HSGP cross-product dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    Eigen::MatrixXd out(left.cols(), right.cols());
    const auto start = std::chrono::steady_clock::now();
    for (int trial = 0; trial < repetitions; ++trial) {
        if (!tulpa::device::cross_product(left.data(), left.rows(), left.cols(), right.data(),
                                         right.cols(), out.data())) {
            scope.check(); Rcpp::stop("HSGP cross-product failed");
        }
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    scope.check();
    return Rcpp::List::create(Rcpp::_["value"] = out, Rcpp::_["seconds"] = seconds,
        Rcpp::_["variant"] = tulpa::device::cross_variant(left.cols(), right.cols()));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_basis_moments(const Eigen::MatrixXd& basis, const Eigen::VectorXd& weights,
    const Eigen::VectorXd& score, const Eigen::VectorXd& scale, const Eigen::VectorXd& amplitude,
    std::string choice) {
    const int n = basis.rows(), m = basis.cols();
    if (n < 1 || m < 1 || weights.size() != n || score.size() != n || scale.size() != m || amplitude.size() != n)
        Rcpp::stop("invalid basis moment dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    const tulpa_coordinate_linalg::Matrix rows = basis;
    tulpa::LatentBlock::DenseBasisBatch batch;
    batch.data = rows.data(); batch.N_k = n; batch.m_per_arm = m; batch.sqrt_S = scale.data();
    tulpa::DenseBasisProducts product;
    product.g_obs = score; product.w_obs = weights;
    tulpa::dense_basis_products(batch, product, amplitude.data());
    scope.check();
    return Rcpp::List::create(Rcpp::_["gram"] = product.H_block,
        Rcpp::_["gradient"] = product.grad_block, Rcpp::_["weighted_basis"] = product.WQ);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_device_dense_threads(int n, int threads, std::string choice) {
    if (n < 1 || threads < 1 || threads > 32) Rcpp::stop("invalid threaded dense dimensions");
    tulpa_dense_fixture::Scope scope(choice);
    std::vector<double> output(threads * 3);
    std::vector<int> ok(threads, 0);
    std::vector<std::thread> workers;
    for (int t = 0; t < threads; ++t) workers.emplace_back([&, t] {
        Eigen::MatrixXd matrix(n, n);
        Eigen::VectorXd rhs(n), solved(n), field(n);
        for (int j = 0; j < n; ++j) {
            rhs[j] = std::sin(j * 0.31 + t);
            for (int i = 0; i < n; ++i) matrix(i, j) = (1 + t * 0.02) * std::exp(-std::abs(i - j) / 4.0) + (i == j ? 0.1 : 0);
        }
        auto& factor = tulpa::device::dense_factor("dense_gp");
        if (!factor.factorize(matrix.data(), n, false, "dense_gp") || !factor.solve(rhs.data(), solved.data())) return;
        if (!tulpa::device::dense_basis(matrix.data(), false).apply(matrix.data(), n, n, false, rhs.data(), field.data())) return;
        tulpa_coordinate_linalg::Matrix lower(n, n), gradient = tulpa_coordinate_linalg::Matrix::Ones(n, n), result(n, n);
        if (!factor.lower(lower.data(), true) || !tulpa::device::cholesky_adjoint(lower.data(), gradient.data(), result.data(), n)) return;
        output[t] = -0.5 * (factor.log_determinant() + rhs.dot(solved));
        output[threads + t] = field.squaredNorm();
        output[2 * threads + t] = result.squaredNorm();
        ok[t] = 1;
    });
    for (auto& worker : workers) worker.join();
    scope.check();
    if (std::find(ok.begin(), ok.end(), 0) != ok.end()) Rcpp::stop("threaded dense operation failed");
    return Rcpp::NumericMatrix(threads, 3, output.begin());
}
