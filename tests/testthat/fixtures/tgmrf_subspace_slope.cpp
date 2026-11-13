// [[Rcpp::depends(RcppEigen, tulpa)]]
#include <tulpa/tgmrf.h>

// At x = (-1, 1), b0 + x b1 has covariance sigma^2 C. The group
// coefficient covariance is sigma^2 [[1, .425], [.425, .25]].
template <typename T>
Eigen::SparseMatrix<T> subspace_slope_Q(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    Eigen::SparseMatrix<T> Q(8, 8);
    const T precision = exp(T(-2) * theta[0]) / T(.2775);
    for (int g = 0; g < 4; ++g) {
        Q.insert(2*g, 2*g) = T(2.1) * precision;
        Q.insert(2*g+1, 2*g+1) = T(.4) * precision;
        Q.insert(2*g, 2*g+1) = T(-.75) * precision;
        Q.insert(2*g+1, 2*g) = T(-.75) * precision;
    }
    Q.makeCompressed();
    return Q;
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> subspace_slope_mu(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    return Eigen::Matrix<T, Eigen::Dynamic, 1>();
}

template <typename T>
T subspace_slope_prior(const Eigen::Matrix<T, Eigen::Dynamic, 1>& theta) {
    const T u = (theta[0] - T(.5)) / T(.5);
    return T(-.5) * u * u;
}

TULPA_REGISTER_TGMRF("tgmrf_subspace_slope", subspace_slope_Q,
                     subspace_slope_mu, subspace_slope_prior)

// [[Rcpp::export]]
std::string tulpa_tgmrf_cpp_id();
