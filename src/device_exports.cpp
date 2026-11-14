#include <RcppEigen.h>
#include "device.h"
#include "gpu_nngp_laplace.h"
#include "sparse_cholesky.h"
#include "cholmod_view.h"
#include <chrono>
#include <thread>

namespace {
using tulpa::device::Status;
void device_check(const Status& status) { if (!status) Rcpp::stop(status.message); }

struct DeviceSession {
    std::shared_ptr<tulpa::device::Policy> policy, previous;
    bool closed = false;
    explicit DeviceSession(tulpa::device::Choice choice) {
        policy = std::make_shared<tulpa::device::Policy>(choice, tulpa::device::thresholds());
        previous = tulpa::device::exchange_policy(policy);
        if (previous && (choice == tulpa::device::Choice::Auto || choice == previous->choice())) {
            policy = previous;
            tulpa::device::exchange_policy(policy);
        } else policy->record("inference", false);
    }
    void close() { if (!closed) { tulpa::device::exchange_policy(previous); closed = true; } }
    ~DeviceSession() { close(); }
};

Rcpp::DataFrame device_stages(const tulpa::device::Policy& policy) {
    Rcpp::CharacterVector name, device;
    Rcpp::NumericVector cpu, cuda;
    for (const auto& stage : policy.stages()) {
        name.push_back(stage.operation);
        device.push_back(stage.cuda ? (stage.cpu ? "cpu,cuda" : "cuda") : "cpu");
        cpu.push_back(stage.cpu); cuda.push_back(stage.cuda);
    }
    return Rcpp::DataFrame::create(Rcpp::Named("stage") = name,
        Rcpp::Named("device") = device, Rcpp::Named("cpu_calls") = cpu,
        Rcpp::Named("cuda_calls") = cuda);
}
}

// [[Rcpp::export]]
Rcpp::List cpp_device_sparse_state(std::string choice, int threads = 4) {
    if (threads < 1 || threads > 32) Rcpp::stop("invalid sparse worker count");
    DeviceSession session(tulpa::device::parse_choice(choice));
    std::vector<int> success(threads), analyses(threads);
    std::vector<std::string> errors(threads);
    auto worker = [&](int id) {
        try {
            const int p[] = {0, 1, 2, 3}, i[] = {0, 1, 2};
            double values[] = {1.0 + id, 2.0 + id, 3.0 + id};
            double rhs[] = {1, 1, 1}, x[] = {0, 0, 0};
            auto A = tulpa::cholmod_lower_view(3, p, i, values, 3);
            tulpa::SparseCholeskySolver solver;
            solver.analyze(&A);
            bool ok = solver.factorize(&A);
            values[0] = -1;
            ok = ok && !solver.factorize(&A) && !solver.solve(rhs, x, 3);
            for (double v : x) ok = ok && std::isnan(v);
            values[0] = 1.0 + id;
            ok = ok && solver.factorize(&A) && solver.solve(rhs, x, 3);
            for (int j = 0; j < 3; ++j) ok = ok && std::abs(x[j] - 1/values[j]) < 1e-12;
            auto diagonal = solver.selected_inversion_diagonal();
            ok = ok && diagonal.size() == 3;
            for (int j = 0; ok && j < 3; ++j) ok = std::abs(diagonal[j] - 1/values[j]) < 1e-12;
            ok = ok && solver.apply_inv_chol_factor(rhs, x, 3, 1);
            for (int j = 0; j < 3; ++j) ok = ok && std::abs(x[j]*x[j] - 1/values[j]) < 1e-12;
            if (solver.uses_cuda()) {
                tulpa::device::SparseFactor subset;
                auto status = subset.analyze(3, p, i, values, solver.stream());
                bool positive = false;
                if (status) status = subset.refactor(values, positive);
                std::vector<double> selected;
                ok = ok && bool(status) && positive;
                const auto invalid = subset.inverse_entries({0, 4, 2, 3}, {0, 1, 2}, {0, 1, 2}, selected);
                ok = ok && !invalid;
                if (status) status = subset.inverse_diagonal({2, 0, 2}, selected);
                ok = ok && bool(status) && selected.size() == 3;
                if (ok) ok = std::abs(selected[0] - 1/values[2]) < 1e-12 &&
                    std::abs(selected[1] - 1/values[0]) < 1e-12 && selected[0] == selected[2];
            }
            success[id] = ok;
            analyses[id] = static_cast<int>(solver.analysis_count());
        } catch (const std::exception& error) { errors[id] = error.what(); }
    };
    std::vector<std::thread> workers;
    for (int id = 0; id < threads; ++id) workers.emplace_back(worker, id);
    for (auto& thread : workers) thread.join();
    device_check(session.policy->failure());
    return Rcpp::List::create(Rcpp::Named("success") = success,
        Rcpp::Named("analyses") = analyses, Rcpp::Named("errors") = errors);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_sparse(Rcpp::IntegerVector p, Rcpp::IntegerVector i,
                             Rcpp::NumericMatrix values, Rcpp::NumericMatrix b,
                             std::string choice, bool full_inverse = false,
                             int repetitions = 1) {
    const int n = p.size() - 1;
    if (n <= 0 || b.nrow() != n || values.nrow() != i.size() ||
        values.ncol() == 0 || p[0] != 0 || p[n] != i.size() || repetitions < 1)
        Rcpp::stop("invalid sparse fixture dimensions");
    for (int j = 0; j < n; ++j) {
        if (p[j] < 0 || p[j + 1] <= p[j] || p[j + 1] > i.size() || i[p[j]] != j)
            Rcpp::stop("invalid sparse fixture pointers or diagonal");
        for (int k = p[j]; k < p[j + 1]; ++k)
            if (i[k] < j || i[k] >= n || (k > p[j] && i[k] <= i[k - 1]))
                Rcpp::stop("invalid sparse fixture indices");
    }
    DeviceSession session(tulpa::device::parse_choice(choice));
    tulpa::SparseCholeskySolver solver;
    Rcpp::NumericMatrix x(n, b.ncol());
    std::vector<double> diagonal;
    tulpa::SparseCholeskySolver::SelectedInverse inverse;
    using Clock = std::chrono::steady_clock;
    double analyze_seconds = 0, factor_seconds = 0, solve_seconds = 0, inverse_seconds = 0;
    auto A = tulpa::cholmod_lower_view(n, p.begin(), i.begin(), values.begin(), i.size());
    auto start = Clock::now();
    solver.analyze(&A);
    analyze_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    device_check(session.policy->failure());
    bool ok = solver.analyzed();
    Rcpp::NumericVector logdet(values.ncol());
    for (int rep = 0; rep < repetitions && ok; ++rep) {
        for (int v = 0; v < values.ncol() && ok; ++v) {
            A.x = values.begin() + static_cast<std::size_t>(v) * i.size();
            start = Clock::now();
            ok = solver.factorize(&A);
            if (ok) logdet[v] = solver.log_determinant();
            factor_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            device_check(session.policy->failure());
            if (!ok) break;
            start = Clock::now();
            ok = solver.solve(b.begin(), x.begin(), n, b.ncol());
            solve_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            start = Clock::now();
            if (full_inverse) {
                inverse = solver.selected_inversion_full();
                ok = ok && inverse.n == n;
                diagonal.resize(n);
                for (int j = 0; j < n && ok; ++j) diagonal[j] = inverse.at(j, j);
            } else {
                diagonal = solver.selected_inversion_diagonal();
                ok = ok && diagonal.size() == static_cast<std::size_t>(n);
            }
            inverse_seconds += std::chrono::duration<double>(Clock::now() - start).count();
            device_check(session.policy->failure());
        }
    }
    session.close();
    const double calls = repetitions * values.ncol();
    return Rcpp::List::create(Rcpp::Named("ok") = ok,
        Rcpp::Named("used_cuda") = solver.uses_cuda(),
        Rcpp::Named("solution") = x, Rcpp::Named("logdet") = logdet,
        Rcpp::Named("diagonal") = diagonal,
        Rcpp::Named("inverse_p") = inverse.Lp, Rcpp::Named("inverse_i") = inverse.Li,
        Rcpp::Named("inverse_x") = inverse.Zx, Rcpp::Named("permutation_inverse") = inverse.perm_inv,
        Rcpp::Named("analyses") = static_cast<double>(solver.analysis_count()),
        Rcpp::Named("factor_bytes") = static_cast<double>(solver.analyzed_factor_bytes()),
        Rcpp::Named("analysis_seconds") = analyze_seconds,
        Rcpp::Named("factor_seconds") = factor_seconds/calls,
        Rcpp::Named("solve_seconds") = solve_seconds/calls,
        Rcpp::Named("inverse_seconds") = inverse_seconds/calls,
        Rcpp::Named("seconds") = (factor_seconds + solve_seconds + inverse_seconds)/calls);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_info() {
    auto info = tulpa::device::Context::instance().info();
    Rcpp::CharacterVector libraries = Rcpp::wrap(info.libraries);
    Rcpp::CharacterVector reasons = Rcpp::wrap(info.library_reasons);
    Rcpp::CharacterVector names = Rcpp::CharacterVector::create("driver", "cusolver", "cublas", "nvrtc", "cudss");
    libraries.names() = names; reasons.names() = names;
    return Rcpp::List::create(Rcpp::Named("available") = info.available,
        Rcpp::Named("compiled") = info.compiled,
        Rcpp::Named("backend") = info.available ? "cuda" : "cpu",
        Rcpp::Named("reason") = info.reason,
        Rcpp::Named("device_count") = info.count,
        Rcpp::Named("name") = info.name,
        Rcpp::Named("memory_mb") = info.memory_mb,
        Rcpp::Named("compute_capability") = info.compute_capability,
        Rcpp::Named("driver_version") = info.driver_version,
        Rcpp::Named("initializations") = info.init_count,
        Rcpp::Named("libraries") = libraries,
        Rcpp::Named("library_reasons") = reasons);
}

// [[Rcpp::export]]
Rcpp::CharacterVector cpp_device_operations() {
    Rcpp::CharacterVector result;
    for (const auto& operation : tulpa::device::operations()) result.push_back(operation.name);
    return result;
}

// [[Rcpp::export]]
std::string cpp_device_validate(SEXP value) {
    if (TYPEOF(value) != STRSXP || Rf_xlength(value) != 1 || STRING_ELT(value, 0) == NA_STRING)
        Rcpp::stop("control$device must be one of 'auto', 'cpu', 'cuda'");
    std::string name = Rcpp::as<std::string>(value);
    auto choice = tulpa::device::parse_choice(name);
    if (choice == tulpa::device::Choice::Cuda)
        device_check(tulpa::device::Context::instance().initialize());
    return name;
}

// [[Rcpp::export]]
void cpp_device_configure(Rcpp::DataFrame table, Rcpp::List sources) {
    Rcpp::CharacterVector operation = table["operation"];
    Rcpp::IntegerVector variant = table["variant"];
    Rcpp::NumericVector size = table["threshold"];
    std::vector<tulpa::device::Threshold> thresholds;
    for (R_xlen_t i = 0; i < operation.size(); ++i) {
        if (Rcpp::IntegerVector::is_na(variant[i]) || std::isnan(size[i]) || size[i] < 0)
            Rcpp::stop("invalid device threshold");
        thresholds.push_back({Rcpp::as<std::string>(operation[i]), variant[i], size[i]});
    }
    Rcpp::CharacterVector names = sources.names();
    std::map<std::string, std::string> kernels;
    for (R_xlen_t i = 0; i < sources.size(); ++i)
        kernels.emplace(Rcpp::as<std::string>(names[i]), Rcpp::as<std::string>(sources[i]));
    tulpa::device::configure(thresholds, kernels);
}

// [[Rcpp::export]]
SEXP cpp_device_begin(std::string choice) {
    return Rcpp::XPtr<DeviceSession>(new DeviceSession(tulpa::device::parse_choice(choice)), true);
}

// [[Rcpp::export]]
void cpp_device_end(SEXP session) {
    Rcpp::XPtr<DeviceSession> scope(session);
    scope->close();
    device_check(scope->policy->failure());
}

// [[Rcpp::export]]
Rcpp::List cpp_device_snapshot() {
    auto policy = tulpa::device::current_policy();
    device_check(policy->failure());
    auto choice = policy->choice();
    return Rcpp::List::create(Rcpp::Named("requested") =
        (choice == tulpa::device::Choice::Auto ? "auto" :
         choice == tulpa::device::Choice::Cpu ? "cpu" : "cuda"),
        Rcpp::Named("stages") = device_stages(*policy));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_probe(std::string operation, double size, int variant,
                            std::string choice) {
    if (!std::isfinite(size) || size < 0) Rcpp::stop("size must be finite and non-negative");
    tulpa::device::Policy policy(tulpa::device::parse_choice(choice), tulpa::device::thresholds());
    Status status;
    bool cuda = policy.select(operation, static_cast<std::size_t>(size), variant, status);
    device_check(status);
    return Rcpp::List::create(Rcpp::Named("device") = cuda ? "cuda" : "cpu");
}

// [[Rcpp::export]]
Rcpp::List cpp_device_test(std::string action, Rcpp::NumericVector input,
                           std::string source = "", int threads = 8) {
    auto& context = tulpa::device::Context::instance();
    Status status;
    std::vector<double> output(input.begin(), input.end());
    if (action == "status") device_check(Status::read("CUDA", 1, "test_status", "invalid value"));
    if (action == "init") {
        if (threads < 1 || threads > 64) Rcpp::stop("threads must be in [1, 64]");
        std::vector<Status> statuses(threads);
        std::vector<std::thread> workers;
        for (int i = 0; i < threads; ++i) workers.emplace_back([&, i] { statuses[i] = context.initialize(); });
        for (auto& worker : workers) worker.join();
        for (const auto& s : statuses) status.absorb(s);
    } else if (action == "buffer" || action == "bounds" || action == "compile") {
        status = context.initialize();
        {
            tulpa::device::Lease lease(context, status);
            if (action == "compile") {
                void* module = nullptr; void* kernel = nullptr;
                if (status) status = context.compile(source, "tulpa_axpy", module, kernel);
                if (module) status.absorb(context.unload(module));
            } else {
                tulpa::device::Buffer buffer(context, output.size() * sizeof(double), status);
                if (status) status = buffer.upload(output.data(), buffer.size() + (action == "bounds" ? 1 : 0));
                if (status) status = buffer.download(output.data(), buffer.size());
            }
        }
    } else if (action != "init") Rcpp::stop("unknown device test action");
    device_check(status);
    return Rcpp::List::create(Rcpp::Named("output") = output,
        Rcpp::Named("initializations") = context.info().init_count);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_axpy(Rcpp::NumericVector x, Rcpp::NumericVector y,
                           double a, std::string choice, int repetitions = 1) {
    if (repetitions <= 0) Rcpp::stop("repetitions must be positive");
    std::vector<double> xv(x.begin(), x.end()), yv(y.begin(), y.end());
    tulpa::device::Policy policy(tulpa::device::parse_choice(choice), tulpa::device::thresholds());
    bool used = false;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        std::copy(y.begin(), y.end(), yv.begin());
        device_check(tulpa::device::axpy(xv, yv, a, policy, used));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    return Rcpp::List::create(Rcpp::Named("value") = yv, Rcpp::Named("used_cuda") = used,
        Rcpp::Named("seconds") = seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_cholesky(Rcpp::NumericMatrix matrices, int n,
                               std::string choice, int repetitions = 1) {
    if (n <= 0 || matrices.ncol() != static_cast<double>(n) * n || repetitions <= 0)
        Rcpp::stop("invalid batch dimensions or repetition count");
    std::vector<std::vector<double>> original(matrices.nrow(), std::vector<double>(static_cast<std::size_t>(n) * n));
    for (int b = 0; b < matrices.nrow(); ++b) for (int j = 0; j < matrices.ncol(); ++j)
        original[b][j] = matrices(b, j);
    tulpa::device::Policy policy(tulpa::device::parse_choice(choice), tulpa::device::thresholds());
    bool used = false;
    auto factors = original;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        factors = original;
        device_check(tulpa::device::cholesky_batch(factors, n, policy, used));
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    Rcpp::NumericMatrix result(matrices.nrow(), matrices.ncol());
    for (int b = 0; b < result.nrow(); ++b) for (int j = 0; j < result.ncol(); ++j) result(b, j) = factors[b][j];
    return Rcpp::List::create(Rcpp::Named("factors") = result, Rcpp::Named("used_cuda") = used,
        Rcpp::Named("seconds") = seconds);
}

// [[Rcpp::export]]
Rcpp::List cpp_device_nngp(Rcpp::NumericVector w, Rcpp::NumericMatrix coords,
                          Rcpp::IntegerMatrix nn_idx, Rcpp::NumericMatrix nn_dist,
                          Rcpp::IntegerVector nn_order, double sigma2, double phi,
                          int cov_type, std::string choice, int repetitions = 1) {
    if (repetitions <= 0 || w.size() != coords.nrow()) Rcpp::stop("invalid NNGP input dimensions");
    DeviceSession session(tulpa::device::parse_choice(choice));
    std::vector<double> field(w.begin(), w.end()), mean, variance, alpha;
    bool used = false;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        tulpa::batch_nngp_scatter(field, coords.nrow(), nn_idx.ncol(), sigma2, phi,
            cov_type, coords, nn_idx, nn_dist, nn_order, mean, variance, used, &alpha);
        device_check(session.policy->failure());
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions;
    return Rcpp::List::create(Rcpp::Named("mean") = mean,
        Rcpp::Named("variance") = variance, Rcpp::Named("alpha") = alpha,
        Rcpp::Named("used_cuda") = used, Rcpp::Named("seconds") = seconds);
}
