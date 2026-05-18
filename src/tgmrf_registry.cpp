// tgmrf_registry.cpp
// Process-global registry of user-compiled tgmrf specs + the
// `tulpa_register_tgmrf` registered C callable that user DLLs use to
// insert specs at load time.
//
// Registration is wired from src/tulpa_init.cpp::tulpa_register_callables()
// so the existing R_init_tulpa hook picks it up at DLL load.

#include "tgmrf_registry.h"

#include <Rcpp.h>
#include <R_ext/Rdynload.h>
#include <mutex>
#include <unordered_map>
#include <string>

namespace tulpa {
namespace tgmrf_backend {

namespace {

// Singleton container. Lazy-initialised on first access; the static-init
// order across translation units does not matter because user DLLs only
// touch the registry via the registered C callable, which goes through
// `register_spec` below, which constructs the container on demand.
struct RegistryImpl {
    std::mutex mtx;
    std::unordered_map<std::string, TgmrfSpec> specs;
};

inline RegistryImpl& registry() {
    static RegistryImpl r;
    return r;
}

} // namespace

void register_spec(const std::string& id, const TgmrfSpec& spec) {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    r.specs[id] = spec;
}

const TgmrfSpec* lookup_spec(const std::string& id) {
    // Read after registration is single-writer-by-time-of-use, so no lock
    // needed. (Inserts only happen at user-DLL load; lookups happen later
    // from the inference layers on the R-driven main thread.)
    auto& r = registry();
    auto it = r.specs.find(id);
    if (it == r.specs.end()) return nullptr;
    return &it->second;
}

int registry_size() {
    auto& r = registry();
    std::lock_guard<std::mutex> lock(r.mtx);
    return static_cast<int>(r.specs.size());
}

} // namespace tgmrf_backend
} // namespace tulpa

// ============================================================================
// Registered C callable wrappers. The signatures match
// tulpa::tgmrf_backend::RegisterFn from inst/include/tulpa/tgmrf.h.
// ============================================================================

static void tulpa_register_tgmrf_impl(const char* id,
                                       const tulpa::tgmrf_backend::TgmrfSpec* spec) {
    if (id == nullptr || spec == nullptr) return;
    tulpa::tgmrf_backend::register_spec(std::string(id), *spec);
}

void tulpa_register_tgmrf_callables(DllInfo* dll) {
    R_RegisterCCallable("tulpa", "tulpa_register_tgmrf",
                        (DL_FUNC)&tulpa_register_tgmrf_impl);
}

// ============================================================================
// Rcpp-export helpers for the R-side constructor / dispatch.
// ============================================================================

// [[Rcpp::export]]
bool cpp_tgmrf_registry_has(std::string id) {
    return tulpa::tgmrf_backend::lookup_spec(id) != nullptr;
}

// [[Rcpp::export]]
int cpp_tgmrf_registry_size() {
    return tulpa::tgmrf_backend::registry_size();
}

// Evaluate the registered spec's Q(theta) at the given numeric theta and
// return the CSC triple + logdet + log_prior_theta -- the same wire format
// the R-closure path produces in R/nested_laplace.R::.nl_block_spec_for_cpp.
//
// Used by R-side dispatch to precompute Q at every joint-grid row when the
// block was registered via tgmrf_cpp().
//
// [[Rcpp::export]]
Rcpp::List cpp_tgmrf_eval(std::string id, Rcpp::NumericVector theta) {
    const tulpa::tgmrf_backend::TgmrfSpec* spec =
        tulpa::tgmrf_backend::lookup_spec(id);
    if (spec == nullptr) {
        Rcpp::stop("tgmrf_cpp: no spec registered under id '%s'.", id.c_str());
    }
    if (spec->Q_double == nullptr) {
        Rcpp::stop("tgmrf_cpp: spec '%s' has no double-precision Q kernel.",
                   id.c_str());
    }

    Eigen::Map<const Eigen::VectorXd> th(theta.begin(), theta.size());
    Eigen::VectorXd th_vec = th;  // copy to a non-mapped vector so the user
                                  // kernel can store it / take its address
                                  // without ABI surprises.
    Eigen::SparseMatrix<double> Q = spec->Q_double(th_vec);
    Q.makeCompressed();

    int n = Q.rows();
    if (Q.cols() != n) {
        Rcpp::stop("tgmrf_cpp: spec '%s' returned a non-square Q (%dx%d).",
                   id.c_str(), n, (int)Q.cols());
    }

    int nnz = Q.nonZeros();
    Rcpp::IntegerVector p_out(n + 1);
    Rcpp::IntegerVector i_out(nnz);
    Rcpp::NumericVector x_out(nnz);
    const int*    p_in = Q.outerIndexPtr();
    const int*    i_in = Q.innerIndexPtr();
    const double* x_in = Q.valuePtr();
    for (int j = 0; j <= n; j++) p_out[j] = p_in[j];
    for (int k = 0; k < nnz; k++) {
        i_out[k] = i_in[k];
        x_out[k] = x_in[k];
    }

    double lp = 0.0;
    if (spec->log_prior_double != nullptr) {
        lp = spec->log_prior_double(th_vec);
    }

    return Rcpp::List::create(
        Rcpp::Named("n")        = n,
        Rcpp::Named("p")        = p_out,
        Rcpp::Named("i")        = i_out,
        Rcpp::Named("x")        = x_out,
        Rcpp::Named("log_prior") = lp
    );
}

// Evaluate just mu(theta). Returns NULL if the spec has no mu kernel
// (matches the R-side `mu = NULL` zero-mean default).
//
// [[Rcpp::export]]
SEXP cpp_tgmrf_eval_mu(std::string id, Rcpp::NumericVector theta) {
    const tulpa::tgmrf_backend::TgmrfSpec* spec =
        tulpa::tgmrf_backend::lookup_spec(id);
    if (spec == nullptr) {
        Rcpp::stop("tgmrf_cpp: no spec registered under id '%s'.", id.c_str());
    }
    if (spec->mu_double == nullptr) {
        return R_NilValue;
    }
    Eigen::Map<const Eigen::VectorXd> th(theta.begin(), theta.size());
    Eigen::VectorXd th_vec = th;
    Eigen::Matrix<double, Eigen::Dynamic, 1> mu = spec->mu_double(th_vec);
    if (mu.size() == 0) {
        return R_NilValue;
    }
    Rcpp::NumericVector out(mu.size());
    for (int i = 0; i < mu.size(); i++) out[i] = mu[i];
    return out;
}
