#ifndef TULPA_SCALAR_PRIORS_H
#define TULPA_SCALAR_PRIORS_H
#include <cmath>
namespace tulpa { namespace math {
// Normal prior: -0.5 * tau * x^2, the N(0, 1 / tau) log-density up to the
// constant log_prior_normal_log_normalizer(tau).
template<typename T>
inline T log_prior_normal(const T& x, double tau) {
    return T(-0.5) * tau * x * x;
}

// 0.5 log(tau / (2 pi)): what log_prior_normal leaves out of the N(0, 1 / tau)
// log-density.
inline double log_prior_normal_log_normalizer(double tau) {
    return 0.5 * std::log(tau / (2.0 * 3.14159265358979323846));
}


inline double log_prior_normal_gradient(double x,double tau) { return -tau*x; }
template<class T>
inline T log_prior_gamma_value(const T& log_phi,const T& phi,double shape,double rate) {
    return (shape-1.0)*log_phi-rate*phi+log_phi;
}
inline double log_prior_gamma_gradient(double phi,double shape,double rate) {
    return shape-rate*phi;
}
} }
#endif
