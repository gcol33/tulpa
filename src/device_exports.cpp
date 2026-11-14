#include <RcppEigen.h>
#include "device.h"
#include "gpu_nngp_laplace.h"
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
