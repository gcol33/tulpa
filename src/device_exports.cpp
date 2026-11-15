#include <RcppEigen.h>
#include "device.h"
#include "gpu_nngp_laplace.h"
#include "sparse_cholesky.h"
#include "cholmod_view.h"
#include "laplace_cholesky.h"
#include <chrono>
#include <atomic>
#include <thread>
#include <limits>

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

void device_sparse_inputs(const Rcpp::IntegerVector& p, const Rcpp::IntegerVector& i,
                          const Rcpp::NumericMatrix& values, const Rcpp::NumericMatrix& b) {
    const int n = p.size()-1;
    if (n <= 0 || b.nrow() != n || values.nrow() != i.size() || values.ncol() == 0 ||
        p[0] != 0 || p[n] != i.size()) Rcpp::stop("invalid sparse fixture dimensions");
    for (int j = 0; j < n; ++j) {
        if (p[j] < 0 || p[j+1] <= p[j] || p[j+1] > i.size() || i[p[j]] != j)
            Rcpp::stop("invalid sparse fixture pointers or diagonal");
        for (int k = p[j]; k < p[j+1]; ++k)
            if (i[k] < j || i[k] >= n || (k > p[j] && i[k] <= i[k-1]))
                Rcpp::stop("invalid sparse fixture indices");
    }
}

template<class Scratch, class Result, class Solve>
void device_cell_jobs(int count, int workers, tulpa::device::BatchScope& batch,
                      std::vector<Result>& results, std::vector<std::string>& errors, Solve solve) {
    std::atomic<int> next{0};
    auto worker = [&] {
        Scratch scratch;
        int k;
        while ((k = next.fetch_add(1)) < count) {
            try {
                results[k] = tulpa::device::run_cell(batch, [&] { return solve(k,scratch); },
                                                    "outer_cell_cpu_recovery");
            } catch (const std::exception& e) { errors[k] = e.what(); }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < workers; ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
}
}

// [[Rcpp::export]]
Rcpp::List cpp_device_scatter_batch(Rcpp::NumericMatrix design, Rcpp::NumericMatrix scores,
    Rcpp::NumericMatrix weights, Rcpp::NumericMatrix scales, std::string choice,
    int workers = 4, bool batched = true) {
    const int rows = design.nrow(), columns = design.ncol(), count = scores.ncol();
    if (rows < 1 || columns < 1 || static_cast<double>(columns)*columns > std::numeric_limits<int>::max() ||
        count < 1 || workers < 1 || workers > 32 ||
        scores.nrow() != rows || weights.nrow() != rows || weights.ncol() != count ||
        scales.nrow() != columns || scales.ncol() != count)
        Rcpp::stop("invalid dense scatter dimensions");
    DeviceSession session(tulpa::device::parse_choice(choice));
    tulpa::device::BatchScope batch(batched ? workers : 1);
    struct Scratch { tulpa::device::DenseScatter scatter; std::vector<double> design; };
    struct Result {
        std::vector<double> gradient, hessian;
        bool device_cpu_recovered = false;
    };
    std::vector<Result> results(count);
    std::vector<std::string> errors(count);
    std::vector<double> row_design(static_cast<std::size_t>(rows)*columns);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < columns; ++j)
        row_design[static_cast<std::size_t>(i)*columns+j] = design(i,j);
    const auto start = std::chrono::steady_clock::now();
    device_cell_jobs<Scratch>(count, workers, batch, results, errors, [&](int k, Scratch& scratch) {
        scratch.design.resize(row_design.size());
        for (int i = 0; i < rows; ++i) for (int j = 0; j < columns; ++j)
            scratch.design[static_cast<std::size_t>(i)*columns+j] =
                row_design[static_cast<std::size_t>(i)*columns+j]*scales(j,k);
        scratch.scatter.bind(scratch.design.data(), rows, columns);
        Result result; result.gradient.resize(columns);
        result.hessian.resize(static_cast<std::size_t>(columns)*columns);
        auto status = scratch.scatter.apply(scores.begin()+static_cast<std::size_t>(k)*rows,
            weights.begin()+static_cast<std::size_t>(k)*rows, result.gradient.data(), result.hessian.data());
        tulpa::device::cell_status(status);
        if (!status) session.policy->fail(status);
        return result;
    });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    Rcpp::NumericMatrix gradient(columns,count), hessian(count,columns*columns);
    Rcpp::LogicalVector recovered(count);
    for (int k = 0; k < count; ++k) {
        recovered[k] = results[k].device_cpu_recovered;
        for (int j = 0; j < static_cast<int>(results[k].gradient.size()); ++j) gradient(j,k) = results[k].gradient[j];
        for (int j = 0; j < static_cast<int>(results[k].hessian.size()); ++j) hessian(k,j) = results[k].hessian[j];
    }
    device_check(session.policy->failure());
    return Rcpp::List::create(Rcpp::Named("gradient")=gradient,Rcpp::Named("hessian")=hessian,
        Rcpp::Named("recovered")=recovered,Rcpp::Named("errors")=errors,
        Rcpp::Named("seconds")=seconds,Rcpp::Named("stages")=device_stages(*session.policy));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_laplace_batch(Rcpp::NumericMatrix matrices, Rcpp::NumericMatrix rhs,
    int n, std::string choice, int workers = 4, bool batched = true, int fail_cell = -1) {
    const int count = matrices.nrow();
    if (n < 1 || workers < 1 || workers > 32 || count < 1 ||
        matrices.ncol() != static_cast<double>(n)*n || rhs.nrow() != count || rhs.ncol() != n)
        Rcpp::stop("invalid Laplace batch dimensions");
    DeviceSession session(tulpa::device::parse_choice(choice));
    tulpa::device::BatchScope batch(batched ? workers : 1);
    std::vector<tulpa::DenseMat> h(count, tulpa::DenseMat(n, tulpa::DenseVec(n, 0.0)));
    std::vector<tulpa::DenseVec> b(count, tulpa::DenseVec(n));
    for (int k = 0; k < count; ++k) {
        for (int i = 0; i < n; ++i) b[k][i] = rhs(k, i);
        for (int i = 0; i < n*n; ++i) h[k].data()[i] = matrices(k, i);
    }
    std::vector<tulpa::LaplaceResult> results(count);
    std::vector<std::string> errors(count);
    const auto start = std::chrono::steady_clock::now();
    device_cell_jobs<tulpa::DenseCholeskyScratch>(count,workers,batch,results,errors,
                [&](int k, tulpa::DenseCholeskyScratch& scratch) {
                    if (k == fail_cell && !tulpa::device::cell_cpu())
                        tulpa::device::cell_status(Status::read("CUDA", -1, "injected cell failure"));
                    tulpa::LaplaceResult r;
                    r.mode.resize(n);
                    r.converged = tulpa::dense_cholesky_solve_raw(h[k], b[k], n, scratch, r.mode, r.log_det_Q);
                    return r;
                });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    device_check(session.policy->failure());
    Rcpp::NumericMatrix solution(count, n);
    Rcpp::NumericVector logdet(count);
    Rcpp::LogicalVector recovered(count), positive(count);
    for (int k = 0; k < count; ++k) {
        logdet[k] = results[k].log_det_Q; recovered[k] = results[k].device_cpu_recovered;
        positive[k] = results[k].converged;
        for (int i = 0; i < n; ++i) solution(k, i) = results[k].mode.size() == static_cast<std::size_t>(n) ? results[k].mode[i] : NA_REAL;
    }
    return Rcpp::List::create(Rcpp::Named("solution")=solution, Rcpp::Named("logdet")=logdet,
        Rcpp::Named("recovered")=recovered, Rcpp::Named("positive")=positive,
        Rcpp::Named("errors")=errors, Rcpp::Named("seconds")=seconds,
        Rcpp::Named("stages")=device_stages(*session.policy));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_sparse_state(std::string choice, int threads = 4, bool batched = false) {
    if (threads < 1 || threads > 32) Rcpp::stop("invalid sparse worker count");
    DeviceSession session(tulpa::device::parse_choice(choice));
    tulpa::device::BatchScope batch(batched ? threads : 1);
    std::vector<int> success(threads), analyses(threads);
    std::vector<std::string> errors(threads);
    auto worker = [&](int id) {
        tulpa::device::CellScope cell(&batch);
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
            double expected_logdet = 0.0;
            for (double v : values) expected_logdet += std::log(v);
            if (std::abs(solver.log_determinant() - expected_logdet) > 1e-12)
                throw std::runtime_error("uniform logdet actual=" + std::to_string(solver.log_determinant()) + " expected=" + std::to_string(expected_logdet));
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
        Rcpp::Named("analyses") = analyses, Rcpp::Named("errors") = errors,
        Rcpp::Named("stages") = device_stages(*session.policy));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_sparse_batch(Rcpp::IntegerVector p, Rcpp::IntegerVector i,
    Rcpp::NumericMatrix values, Rcpp::NumericMatrix b, std::string choice,
    int workers = 4, bool batched = true, bool selected_inverse = false,
    bool per_cell_rhs = false) {
    device_sparse_inputs(p, i, values, b);
    if (workers < 1 || workers > 32) Rcpp::stop("invalid sparse worker count");
    const int n = p.size()-1, count = values.ncol();
    if (per_cell_rhs && b.ncol()%count != 0)
        Rcpp::stop("invalid sparse RHS batches");
    const int columns = b.ncol()/(per_cell_rhs ? count : 1);
    DeviceSession session(tulpa::device::parse_choice(choice));
    tulpa::device::BatchScope batch(batched ? workers : 1);
    std::vector<tulpa::LaplaceResult> result(count);
    std::vector<std::string> errors(count);
    const auto start = std::chrono::steady_clock::now();
    device_cell_jobs<std::unique_ptr<tulpa::SparseCholeskySolver>>(count,workers,batch,result,errors,
                [&](int k, std::unique_ptr<tulpa::SparseCholeskySolver>& owner) {
                    if (!owner) owner.reset(new tulpa::SparseCholeskySolver(selected_inverse && columns == 0 ?
                        tulpa::SparseCholeskySolver::Purpose::SelectedInverse :
                        tulpa::SparseCholeskySolver::Purpose::Solve));
                    auto& solver = *owner;
                    tulpa::LaplaceResult r;
                    auto A = tulpa::cholmod_lower_view(n, p.begin(), i.begin(),
                        values.begin() + static_cast<std::size_t>(k)*i.size(), i.size());
                    if (!solver.analyzed()) solver.analyze(&A);
                    r.converged = solver.factorize(&A);
                    r.log_det_Q = solver.log_determinant();
                    r.mode.resize(static_cast<std::size_t>(n)*columns);
                    const auto offset = per_cell_rhs ? static_cast<std::size_t>(k)*n*columns : 0;
                    if (r.converged) r.converged = solver.solve(b.begin()+offset, r.mode.data(), n, columns);
                    if (r.converged && selected_inverse) r.eta_var = solver.selected_inversion_diagonal();
                    return r;
                });
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    Rcpp::NumericVector logdet(count), solution(static_cast<std::size_t>(n)*columns*count),
        diagonal(selected_inverse ? static_cast<std::size_t>(n)*count : 0);
    Rcpp::LogicalVector positive(count), recovered(count);
    for (int k = 0; k < count; ++k) {
        logdet[k] = result[k].log_det_Q; positive[k] = result[k].converged;
        recovered[k] = result[k].device_cpu_recovered;
        std::copy(result[k].mode.begin(), result[k].mode.end(), solution.begin() + static_cast<std::size_t>(k)*n*columns);
        if (selected_inverse) std::copy(result[k].eta_var.begin(), result[k].eta_var.end(), diagonal.begin() + static_cast<std::size_t>(k)*n);
    }
    device_check(session.policy->failure());
    return Rcpp::List::create(Rcpp::Named("solution")=solution, Rcpp::Named("logdet")=logdet,
        Rcpp::Named("diagonal")=diagonal, Rcpp::Named("positive")=positive,
        Rcpp::Named("recovered")=recovered, Rcpp::Named("errors")=errors,
        Rcpp::Named("seconds")=seconds, Rcpp::Named("stages")=device_stages(*session.policy));
}

// [[Rcpp::export]]
Rcpp::List cpp_device_sparse(Rcpp::IntegerVector p, Rcpp::IntegerVector i,
                             Rcpp::NumericMatrix values, Rcpp::NumericMatrix b,
                             std::string choice, bool full_inverse = false,
                             int repetitions = 1) {
    const int n = p.size() - 1;
    device_sparse_inputs(p, i, values, b);
    if (repetitions < 1) Rcpp::stop("invalid sparse repetition count");
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
    Rcpp::NumericVector batch_size = table.containsElementNamed("batch_threshold")
        ? Rcpp::NumericVector(table["batch_threshold"]) : Rcpp::NumericVector(table.nrows(), 0.0);
    std::vector<tulpa::device::Threshold> thresholds;
    for (R_xlen_t i = 0; i < operation.size(); ++i) {
        if (Rcpp::IntegerVector::is_na(variant[i]) || std::isnan(size[i]) || size[i] < 0 ||
            std::isnan(batch_size[i]) || batch_size[i] < 0)
            Rcpp::stop("invalid device threshold");
        thresholds.push_back({Rcpp::as<std::string>(operation[i]), variant[i], size[i], batch_size[i]});
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
