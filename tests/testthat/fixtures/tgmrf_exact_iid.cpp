// [[Rcpp::depends(RcppEigen, tulpa)]]
#include <tulpa/tgmrf.h>

template <typename T>
Eigen::SparseMatrix<T> exact_iid_Q(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    Eigen::SparseMatrix<T> Q(8, 8);
    for (int j = 0; j < 8; ++j) Q.insert(j, j) = exp(T(-2) * theta[0]);
    Q.makeCompressed();
    return Q;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> exact_iid_mu(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    return Eigen::Matrix<T, Eigen::Dynamic, 1>();
}

template <typename T>
T exact_iid_prior(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    const T u = (theta[0] - T(.5)) / T(.5);
    return T(-.5) * u * u;
}

TULPA_REGISTER_TGMRF("tgmrf_exact_iid", exact_iid_Q, exact_iid_mu, exact_iid_prior)

// [[Rcpp::export]]
std::string tulpa_tgmrf_cpp_id();
