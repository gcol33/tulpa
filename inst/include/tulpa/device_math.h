#ifndef TULPA_DEVICE_MATH_H
#define TULPA_DEVICE_MATH_H

#ifdef __CUDACC_RTC__
#define TULPA_HD __host__ __device__
#define TULPA_CONSTANT __device__ __constant__
namespace std {
using ::log; using ::log1p; using ::exp; using ::expm1; using ::pow;
using ::sqrt; using ::lgamma; using ::fma; using ::frexp;
using ::isfinite; using ::isinf; using ::isnan;
TULPA_HD inline double abs(double x) { return x < 0 ? -x : x; }
template<class T> struct numeric_limits;
template<> struct numeric_limits<double> {
    TULPA_HD static double quiet_NaN() { return ::nan(""); }
    TULPA_HD static double infinity() { return 1.0/0.0; }
};
}
#else
#define TULPA_HD
#define TULPA_CONSTANT
#include <cmath>
#include <limits>
#endif

namespace tulpa {
constexpr double kMuFloor=1e-15;
namespace math {
constexpr double EXP_ARG_MAX=700.0, EXP_ARG_MIN=-700.0;
constexpr double PROB_EPS=1e-15;

TULPA_HD inline double clamp_exp_arg(double x) {
    if (x>EXP_ARG_MAX) return EXP_ARG_MAX;
    if (x<EXP_ARG_MIN) return EXP_ARG_MIN;
    return x;
}
TULPA_HD inline double clamp_prob(double p) {
    if (p<PROB_EPS) return PROB_EPS;
    if (p>1-PROB_EPS) return 1-PROB_EPS;
    return p;
}
} }

#endif
