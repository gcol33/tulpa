// tgmrf_periodic_ar1.cpp
// C++ fast backend for the periodic-AR1 latent block worked example
// (see inst/examples/tgmrf_periodic_ar1.R for the R-closure analogue).
//
// Build a periodic-AR1 precision matrix in templated arithmetic, register
// it under the id "tgmrf_periodic_ar1" via TULPA_REGISTER_TGMRF. The R
// constructor `tgmrf_cpp()` compiles this file with Rcpp::sourceCpp and
// then looks up the spec by id.
//
// Compile from R via:
//   blk <- tulpa::tgmrf_cpp(
//     cpp_file = "tgmrf_periodic_ar1.cpp",
//     id       = "tgmrf_periodic_ar1",
//     init     = c(log_sigma = 0, atanh_rho = atanh(0.3)),
//     bounds   = list(lower = c(log(0.3), atanh(0.0)),
//                     upper = c(log(3.0), atanh(0.95)))
//   )
//
// Precision matrix (n_latent = N, tridiagonal with wrap-around):
//   Q_ii        = (1 + rho^2) / sigma^2
//   Q_{i, i+1}  = -rho / sigma^2  (and the wrap-around Q_{0, N-1})
//
// The block uses N = 80 by default; change kN below to adjust. The R
// constructor reads back n_latent via one Q(init) evaluation, so the size
// is captured automatically -- this constant only matters here.

// [[Rcpp::depends(RcppEigen, tulpa)]]

#include <tulpa/tgmrf.h>

// Latent dimension. Kept fixed so the example .cpp does not need a
// runtime-configurable size; for arbitrary N a user would parameterise
// this via a registry-stored argument or precompile multiple sizes.
static constexpr int kN = 80;

template <typename T>
Eigen::SparseMatrix<T> periodic_ar1_Q(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    // theta = (log_sigma, atanh_rho)
    const T sigma = exp(theta(0));
    const T rho   = tanh(theta(1));
    const T tau   = T(1) / (sigma * sigma);
    const T d     = tau * (T(1) + rho * rho);
    const T o     = -tau * rho;

    Eigen::SparseMatrix<T> Q(kN, kN);
    typedef Eigen::Triplet<T> Triplet;
    std::vector<Triplet> trips;
    trips.reserve(3 * kN);

    for (int i = 0; i < kN; ++i) {
        trips.emplace_back(i, i, d);
    }
    for (int i = 0; i < kN - 1; ++i) {
        trips.emplace_back(i,     i + 1, o);
        trips.emplace_back(i + 1, i,     o);
    }
    // Wrap-around entries.
    trips.emplace_back(0,      kN - 1, o);
    trips.emplace_back(kN - 1, 0,      o);

    Q.setFromTriplets(trips.begin(), trips.end());
    Q.makeCompressed();
    return Q;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> periodic_ar1_mu(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& /*theta*/) {
    // Zero mean. Returning an empty vector tells the R constructor to set
    // `mu = NULL` on the resulting block (matching the tgmrf() default).
    return Eigen::Matrix<T, Eigen::Dynamic, 1>();
}

template <typename T>
T periodic_ar1_log_prior(
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    // Same prior as the R example: N(0, 1) on each component.
    const T LOG2PI = T(1.8378770664093453);
    T lp = T(0);
    for (int j = 0; j < theta.size(); ++j) {
        lp += T(-0.5) * theta(j) * theta(j) - T(0.5) * LOG2PI;
    }
    return lp;
}

TULPA_REGISTER_TGMRF(
    "tgmrf_periodic_ar1",
    periodic_ar1_Q,
    periodic_ar1_mu,
    periodic_ar1_log_prior
)

// Rcpp::sourceCpp will not dyn.load() the compiled DLL unless at least one
// `// [[Rcpp::export]]` marker is present in the source. The macro above
// defines `tulpa_tgmrf_cpp_id()` for us; this prototype + comment is what
// Rcpp's attribute scanner sees pre-preprocessing.

// [[Rcpp::export]]
std::string tulpa_tgmrf_cpp_id();
