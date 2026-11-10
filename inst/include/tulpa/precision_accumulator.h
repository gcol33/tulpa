#ifndef TULPA_PRECISION_ACCUMULATOR_H
#define TULPA_PRECISION_ACCUMULATOR_H

#include <cmath>
#include <array>

namespace tulpa { namespace math { namespace detail {

// Unevaluated sum with an FMA product residual. The low component preserves
// rounding errors using binary64 arithmetic on every platform.
struct PrecisionAccumulator {
    double hi, lo;
    PrecisionAccumulator(double value = 0) : hi(value), lo(0) {}
    PrecisionAccumulator(double high, double low) : hi(high), lo(low) {}
    explicit operator double() const { return static_cast<double>(hi+lo); }
};

inline PrecisionAccumulator operator+(PrecisionAccumulator a, PrecisionAccumulator b) {
    const double s = a.hi+b.hi;
    if (!std::isfinite(s)) return s;
    const double v = s-a.hi;
    const double e = (a.hi-(s-v))+(b.hi-v)+a.lo+b.lo;
    const double h = s+e;
    if (!std::isfinite(h)) return h;
    return {h, e-(h-s)};
}
inline PrecisionAccumulator operator-(PrecisionAccumulator a, PrecisionAccumulator b) {
    return a+PrecisionAccumulator(-b.hi,-b.lo);
}
inline PrecisionAccumulator operator*(PrecisionAccumulator a, PrecisionAccumulator b) {
    const double p = a.hi*b.hi;
    if (!std::isfinite(p)) return p;
    const double e = std::fma(a.hi,b.hi,-p)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;
    const double h = p+e;
    if (!std::isfinite(h)) return h;
    return {h,e-(h-p)};
}
inline PrecisionAccumulator operator/(PrecisionAccumulator a, PrecisionAccumulator b) {
    const double q = a.hi/b.hi;
    if (!std::isfinite(q)) return q;
    const PrecisionAccumulator r = a-b*PrecisionAccumulator(q);
    return PrecisionAccumulator(q)+PrecisionAccumulator((r.hi+r.lo)/b.hi);
}
inline PrecisionAccumulator& operator+=(PrecisionAccumulator& a, PrecisionAccumulator b) {
    a = a+b;
    return a;
}
inline PrecisionAccumulator& operator*=(PrecisionAccumulator& a, PrecisionAccumulator b) {
    a = a*b;
    return a;
}
inline PrecisionAccumulator accumulator_power(PrecisionAccumulator x, int n) {
    PrecisionAccumulator out(1);
    for (int k=0; k<n; ++k) out *= x;
    return out;
}

// sum_{n>=1} z^(n-1)/(2n+1). At the branch bounds the omitted tail
// is below 1e-25; all callers have 0 <= z <= 4/9.
inline PrecisionAccumulator accumulator_log_tail(PrecisionAccumulator z) {
    static const std::array<PrecisionAccumulator,80> c = [] {
        std::array<PrecisionAccumulator,80> out;
        for (int j=0; j<80; ++j)
            out[j] = PrecisionAccumulator(1)/PrecisionAccumulator(2*j+3);
        return out;
    }();
    const int n = z.hi <= .01 ? 14 : z.hi <= .125 ? 28 : 80;
    PrecisionAccumulator sum;
    for (int j=n-1; j>=0; --j) sum = sum*z+c[j];
    return sum;
}

// log(x) = e log(2) + 2 atanh((m-1)/(m+1)), x = 2^e m,
// 1 <= m < 2.
inline PrecisionAccumulator accumulator_log(PrecisionAccumulator x) {
    int e;
    std::frexp(x.hi,&e);
    --e;
    const PrecisionAccumulator m(std::ldexp(x.hi,-e),std::ldexp(x.lo,-e));
    const PrecisionAccumulator r = (m-PrecisionAccumulator(1))/(m+PrecisionAccumulator(1));
    const PrecisionAccumulator r2 = r*r;
    const PrecisionAccumulator sum = PrecisionAccumulator(1)+r2*accumulator_log_tail(r2);
    const PrecisionAccumulator ln2(0x1.62e42fefa39efp-1L,
                                  2.31904681384629961549485546387547865e-17L);
    return PrecisionAccumulator(e)*ln2+PrecisionAccumulator(2)*r*sum;
}

} } }
#endif
