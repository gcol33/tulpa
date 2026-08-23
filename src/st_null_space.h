// st_null_space.h
// The second direction of an intrinsic TEMPORAL null space, and which
// spatiotemporal interactions need it pinned.
//
// soft_sum_to_zero.h pins a field's SUM, which is the whole kernel of an
// intrinsic operator whose null space is the constants. An RW2 marginal's
// kernel is the constants AND the linear ramp, so an interaction built on one
// carries a second family of unpenalized directions:
//
//   null(Q_s (x) Q_t) = null(Q_s) (x) R^T + R^S (x) null(Q_t),
//
// and with null(Q_t) = span{1_T, v} the row sums and the column sums span
// S + T - 1 of the T + 2S - 2 dimensions. The S - 1 left over are the
// site-specific linear time trends summing to zero across sites, and they
// carry no prior curvature at all: at the fixture's own position their
// posterior is held by the likelihood alone while every other direction
// stiffens with tau, so cond(M^-1 Q) under any metric built from the two
// margins grows linearly in tau without bound.
//
// The engine already reported the right dimension on the other side of the
// same prior: tulpa_priors_st.h's normalizer reads rank_space * rank_time,
// which is 8 * 2 = 16 against ST = 36 on the 3x3 / T = 4 fixture. It was
// written for a 20-dimensional kernel while the penalty beside it pinned 12.
//
// TYPE_II is the same shortfall by the same derivation -- its kernel is
// R^S (x) null(Q_t), so RW2 leaves the same S - 1 per-site ramps -- and takes
// the same term. TYPE_III's kernel is null(Q_s) (x) R^T, already spanned by
// the column sums, and TYPE_I is proper.
//
// CYCLIC RW2 is NOT affected: a linear ramp is not periodic, so rw2_rank
// reports T - 1 there and the kernel is the constants alone. Nothing is added
// for it.
#ifndef TULPA_ST_NULL_SPACE_H
#define TULPA_ST_NULL_SPACE_H

#include "tulpa/soft_sum_to_zero.h"
#include "tulpa/types.h"

namespace tulpa_st {

using tulpa::STType;
using tulpa::TemporalType;

// Does this interaction's kernel reach past what the row and column sums pin?
inline bool st_needs_trend_pin(STType type, TemporalType temporal, bool cyclic) {
    if (temporal != TemporalType::RW2 || cyclic) return false;
    return type == STType::TYPE_IV || type == STType::TYPE_II;
}

// The ramp direction over T time points, centred so it is orthogonal to the
// constant the row sums already pin.
inline double st_trend_weight(int t, int T) {
    return static_cast<double>(t) - 0.5 * static_cast<double>(T - 1);
}

// v' v = sum_t (t - (T-1)/2)^2 = T (T^2 - 1) / 12.
inline double st_trend_norm2(int T) {
    if (T < 2) return 0.0;
    const double n = static_cast<double>(T);
    return n * (n * n - 1.0) / 12.0;
}

// Precision on each site's trend COEFFICIENT v' delta_s / v' v, at the sd the
// row and column margins hold their own coefficients at.
inline double st_trend_precision(int T) {
    return tulpa::s2z_precision_weighted(st_trend_norm2(T));
}

}  // namespace tulpa_st

#endif  // TULPA_ST_NULL_SPACE_H
