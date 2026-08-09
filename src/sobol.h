#ifndef TULPA_SOBOL_H
#define TULPA_SOBOL_H

// Sobol' low-discrepancy sequence over the unit cube, Gray-code recurrence,
// 32-bit direction integers, Joe & Kuo direction numbers.
//
// Construction. Dimension j holds direction integers v_1 .. v_32, where
// v_k is the k-th direction number m_k placed at bit 32 - k, so v_k read as a
// binary fraction is m_k / 2^k. For k <= s the m_k come from the table; beyond
// the degree they follow the primitive polynomial's recurrence, which acts on
// the shifted integers directly:
//
//   v_k = v_{k-s} XOR (v_{k-s} >> s) XOR (sum over the polynomial's set bits
//                                          a_t of v_{k-t}),  t = 1 .. s-1
//
// Dimension 1 carries no polynomial: every m_k is 1, so v_k = 1 << (32 - k).
//
// Point i is the running Gray-code XOR X_i = X_{i-1} XOR v_c, where c is the
// 1-based position of the rightmost zero bit of i - 1, starting from X_0 = 0.
//
// SKIPPING THE ORIGIN. X_0 is the zero vector. Callers map these points
// through qnorm, and qnorm(0) is -Inf, so emitting it would poison the whole
// point set. sobol_points() therefore returns points i = 1 .. n: the origin is
// never emitted and the returned set is the sequence's second through
// (n+1)-th points. Two consequences worth knowing at the call site:
//   - the returned points are not the first 2^m points of the sequence, so the
//     net property holds over the first 2^m - 1 of them together with the
//     dropped origin: each elementary interval [j/2^m, (j+1)/2^m) with
//     j = 1 .. 2^m - 1 holds exactly one point in every one-dimensional
//     projection, and interval 0 holds none;
//   - X_i for 1 <= i < 2^32 is never zero in any coordinate (the v_k are
//     linearly independent over GF(2)), and X_i / 2^32 <= 1 - 2^-32 is never
//     one, so the clamp below is a guard rather than a working part.
//
// The table is compiled in; nothing here allocates and nothing throws.

#include <cstddef>

#include "sobol_direction_numbers.h"

namespace tulpa {

// Largest dimension the compiled direction-number table covers.
static const int SOBOL_MAX_DIM = static_cast<int>(sobol_table::kMaxDim);

// Bits carried by a direction integer, and the count of direction integers
// held per dimension.
static const unsigned int SOBOL_BITS = 32u;

namespace sobol_detail {

static const double kTwoPow32Inv = 1.0 / 4294967296.0;

// Half a unit in the last place of the 32-bit fraction: the guard that keeps a
// returned coordinate off the open interval's endpoints.
static const double kUnitEps = kTwoPow32Inv * 0.5;

inline double to_unit(unsigned int x) {
  double u = static_cast<double>(x) * kTwoPow32Inv;
  if (u < kUnitEps) return kUnitEps;
  if (u > 1.0 - kUnitEps) return 1.0 - kUnitEps;
  return u;
}

// Direction integers of dimension `dim` (1-based) into v[1 .. SOBOL_BITS].
// v[0] is unused so the indexing matches the recurrence above.
inline void direction_integers(int dim, unsigned int* v) {
  if (dim == 1) {
    for (unsigned int k = 1u; k <= SOBOL_BITS; ++k) {
      v[k] = 1u << (SOBOL_BITS - k);
    }
    return;
  }

  const unsigned int idx = static_cast<unsigned int>(dim) - 2u;
  const unsigned int s = sobol_table::kDegree[idx];
  const unsigned int a = sobol_table::kPoly[idx];
  const unsigned int* m = sobol_table::kInit + sobol_table::kOffset[idx];

  const unsigned int n_seed = (s < SOBOL_BITS) ? s : SOBOL_BITS;
  for (unsigned int k = 1u; k <= n_seed; ++k) {
    v[k] = m[k - 1u] << (SOBOL_BITS - k);
  }
  for (unsigned int k = s + 1u; k <= SOBOL_BITS; ++k) {
    unsigned int val = v[k - s] ^ (v[k - s] >> s);
    for (unsigned int t = 1u; t + 1u <= s; ++t) {
      if ((a >> (s - 1u - t)) & 1u) val ^= v[k - t];
    }
    v[k] = val;
  }
}

// 1-based position of the rightmost zero bit of `i`.
inline unsigned int rightmost_zero_bit(unsigned int i) {
  unsigned int c = 1u;
  while (i & 1u) {
    i >>= 1;
    ++c;
  }
  return c;
}

}  // namespace sobol_detail

// Write the Sobol' points i = 1 .. n in dimension d into `out`, column-major
// [n x d] so column j holds coordinate j of every point. Returns false without
// touching `out` when d is outside [1, SOBOL_MAX_DIM] or n is not positive.
inline bool sobol_points(int n, int d, double* out) {
  if (n <= 0 || d <= 0 || d > SOBOL_MAX_DIM || out == 0) return false;

  unsigned int v[SOBOL_BITS + 1u];
  for (int j = 0; j < d; ++j) {
    sobol_detail::direction_integers(j + 1, v);
    unsigned int x = 0u;
    double* col = out + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    for (int i = 1; i <= n; ++i) {
      const unsigned int c =
          sobol_detail::rightmost_zero_bit(static_cast<unsigned int>(i - 1));
      x ^= v[c];
      col[i - 1] = sobol_detail::to_unit(x);
    }
  }
  return true;
}

}  // namespace tulpa

#endif  // TULPA_SOBOL_H
