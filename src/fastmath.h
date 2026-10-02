// fastmath.h
// Double-precision exp and log for the per-observation likelihood kernels.
//
// The Laplace kernels evaluate one exp per observation per Newton step and an
// exp and a log per line-search trial, and on the UCRT toolchain std::exp costs
// ~80 cycles and std::log ~51, most of a binomial solve. These are the
// table-driven routines of Arm Optimized Routines (math/exp.c, math/log.c), the
// ones glibc and musl ship: 128-entry tables, worst-case error 0.52 ulp for exp
// and 0.52 ulp for log on the no-FMA paths ported here, special values (0,
// subnormals, infinities, NaN, overflow, underflow) handled as the C library
// handles them, without setting errno or raising floating-point exceptions.
// The same code runs on every platform, so the kernels that use it agree
// across operating systems where the system libm does not.
//
// Ported from Arm Optimized Routines.
// Copyright (c) 2018-2025, Arm Limited.
// SPDX-License-Identifier: MIT OR Apache-2.0 WITH LLVM-exception
// Used here under the MIT license; see inst/COPYRIGHTS.

#ifndef TULPA_FASTMATH_H
#define TULPA_FASTMATH_H

#include <cstdint>
#include <cstring>
#include <limits>

namespace tulpa {
namespace fastmath {

namespace detail {

constexpr int kExpTableBits = 7;
constexpr int kLogTableBits = 7;
struct LogEntry { double a, b; };   // {invc, logc} or {chi, clo}

extern const std::uint64_t exp_tab[2 * (1 << kExpTableBits)];
extern const LogEntry log_tab[1 << kLogTableBits];
extern const LogEntry log_tab2[1 << kLogTableBits];

inline std::uint64_t asuint64(double x) {
    std::uint64_t u;
    std::memcpy(&u, &x, sizeof u);
    return u;
}
inline double asdouble(std::uint64_t u) {
    double x;
    std::memcpy(&x, &u, sizeof x);
    return x;
}
inline std::uint32_t top12(double x) {
    return static_cast<std::uint32_t>(asuint64(x) >> 52);
}

// Arm's exp specialcase(): scale * (1 + tmp) where the exponent of scale may
// have overflowed (k > 0) or must land in the subnormal range (k < 0).
inline double exp_specialcase(double tmp, std::uint64_t sbits,
                              std::uint64_t ki) {
    double scale, y;
    if ((ki & 0x80000000) == 0) {
        // k > 0: the exponent of scale might have overflowed by <= 460.
        sbits -= 1009ull << 52;
        scale = asdouble(sbits);
        y = 0x1p1009 * (scale + scale * tmp);
        return y;
    }
    // k < 0: round in the normal range first, then scale into the subnormal
    // range, so the result is not rounded twice.
    sbits += 1022ull << 52;
    scale = asdouble(sbits);
    y = scale + scale * tmp;
    if (y < 1.0) {
        double hi, lo;
        lo = scale - y + scale * tmp;
        hi = 1.0 + y;
        lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if (y == 0.0) y = 0.0;   // no -0.0
    }
    return 0x1p-1022 * y;
}

} // namespace detail

inline double exp(double x) {
    using namespace detail;
    constexpr int N = 1 << kExpTableBits;
    constexpr double InvLn2N = 0x1.71547652b82fep0 * N;
    constexpr double NegLn2hiN = -0x1.62e42fefa0000p-8;
    constexpr double NegLn2loN = -0x1.cf79abc9e3b3ap-47;
    constexpr double Shift = 0x1.8p52;
    constexpr double C2 = 0x1.ffffffffffdbdp-2;
    constexpr double C3 = 0x1.555555555543cp-3;
    constexpr double C4 = 0x1.55555cf172b91p-5;
    constexpr double C5 = 0x1.1111167a4d017p-7;

    std::uint32_t abstop = top12(x) & 0x7ff;
    if (abstop - top12(0x1p-54) >= top12(512.0) - top12(0x1p-54)) {
        if (abstop - top12(0x1p-54) >= 0x80000000) {
            // |x| < 2^-54 (0 included): exp(x) rounds to 1 + x.
            return 1.0 + x;
        }
        if (abstop >= top12(1024.0)) {
            if (asuint64(x) == asuint64(-std::numeric_limits<double>::infinity()))
                return 0.0;
            if (abstop >= top12(std::numeric_limits<double>::infinity()))
                return 1.0 + x;   // +inf or NaN
            return (asuint64(x) >> 63) ? 0.0
                                       : std::numeric_limits<double>::infinity();
        }
        // Large |x| is handled by the special case below.
        abstop = 0;
    }

    // exp(x) = 2^(k/N) * exp(r), x = ln2/N * k + r, |r| <= ln2/2N.
    const double z = InvLn2N * x;
    double kd = z + Shift;
    const std::uint64_t ki = asuint64(kd);
    kd -= Shift;
    const double r = x + kd * NegLn2hiN + kd * NegLn2loN;
    // 2^(k/N) ~= scale * (1 + tail).
    const std::uint64_t idx = 2 * (ki % N);
    const std::uint64_t top = ki << (52 - kExpTableBits);
    const double tail = asdouble(exp_tab[idx]);
    const std::uint64_t sbits = exp_tab[idx + 1] + top;
    const double r2 = r * r;
    const double tmp = tail + r + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5);
    if (abstop == 0) return exp_specialcase(tmp, sbits, ki);
    const double scale = asdouble(sbits);
    return scale + scale * tmp;
}

inline double log(double x) {
    using namespace detail;
    constexpr int N = 1 << kLogTableBits;
    constexpr std::uint64_t OFF = 0x3fe6000000000000ull;
    constexpr double Ln2hi = 0x1.62e42fefa3800p-1;
    constexpr double Ln2lo = 0x1.ef35793c76730p-45;
    constexpr double A0 = -0x1.0000000000001p-1;
    constexpr double A1 = 0x1.555555551305bp-2;
    constexpr double A2 = -0x1.fffffffeb459p-3;
    constexpr double A3 = 0x1.999b324f10111p-3;
    constexpr double A4 = -0x1.55575e506c89fp-3;
    constexpr double B0 = -0x1p-1;
    constexpr double B1 = 0x1.5555555555577p-2;
    constexpr double B2 = -0x1.ffffffffffdcbp-3;
    constexpr double B3 = 0x1.999999995dd0cp-3;
    constexpr double B4 = -0x1.55555556745a7p-3;
    constexpr double B5 = 0x1.24924a344de3p-3;
    constexpr double B6 = -0x1.fffffa4423d65p-4;
    constexpr double B7 = 0x1.c7184282ad6cap-4;
    constexpr double B8 = -0x1.999eb43b068ffp-4;
    constexpr double B9 = 0x1.78182f7afd085p-4;
    constexpr double B10 = -0x1.5521375d145cdp-4;

    std::uint64_t ix = asuint64(x);
    const std::uint32_t top = static_cast<std::uint32_t>(ix >> 48);

    const std::uint64_t LO = asuint64(1.0 - 0x1p-4);
    const std::uint64_t HI = asuint64(1.0 + 0x1.09p-4);
    if (ix - LO < HI - LO) {
        // x close to 1: a direct polynomial in r = x - 1.
        if (ix == asuint64(1.0)) return 0.0;
        const double r = x - 1.0;
        const double r2 = r * r;
        const double r3 = r * r2;
        double y = r3 * (B1 + r * B2 + r2 * B3
                         + r3 * (B4 + r * B5 + r2 * B6
                                 + r3 * (B7 + r * B8 + r2 * B9 + r3 * B10)));
        double w = r * 0x1p27;
        const double rhi = r + w - w;
        const double rlo = r - rhi;
        w = rhi * rhi * B0;
        const double hi = r + w;
        double lo = r - hi + w;
        lo += B0 * rlo * (rhi + r);
        y += lo;
        y += hi;
        return y;
    }
    if (top - 0x0010 >= 0x7ff0 - 0x0010) {
        // x < 2^-1022, or inf, or NaN.
        if (ix * 2 == 0) return -std::numeric_limits<double>::infinity();
        if (ix == asuint64(std::numeric_limits<double>::infinity())) return x;
        if ((top & 0x8000) || (top & 0x7ff0) == 0x7ff0)
            return std::numeric_limits<double>::quiet_NaN();
        // Subnormal: normalize.
        ix = asuint64(x * 0x1p52);
        ix -= 52ull << 52;
    }

    // x = 2^k z, z in [OFF, 2 OFF) exact; c is near the centre of z's
    // subinterval. log(x) = log1p(z/c - 1) + log(c) + k Ln2.
    const std::uint64_t tmp = ix - OFF;
    const int i = static_cast<int>((tmp >> (52 - kLogTableBits)) % N);
    const int k = static_cast<int>(static_cast<std::int64_t>(tmp) >> 52);
    const std::uint64_t iz = ix - (tmp & (0xfffull << 52));
    const double invc = log_tab[i].a;
    const double logc = log_tab[i].b;
    const double z = asdouble(iz);
    const double r = (z - log_tab2[i].a - log_tab2[i].b) * invc;
    const double kd = static_cast<double>(k);
    const double w = kd * Ln2hi + logc;
    const double hi = w + r;
    const double lo = w - hi + r + kd * Ln2lo;
    const double r2 = r * r;
    return lo + r2 * A0 + r * r2 * (A1 + r * A2 + r2 * (A3 + r * A4)) + hi;
}

} // namespace fastmath
} // namespace tulpa

#endif // TULPA_FASTMATH_H
