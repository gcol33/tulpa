// soft_sum_to_zero.h
// Soft sum-to-zero identification constant for intrinsic (rank-deficient)
// latent fields.
//
// An intrinsic precision (ICAR, RW1, RW2, and the interaction fields built
// from them) has a constant null direction per connected component. That
// direction is jointly unidentified with the intercept: the level is free to
// live in the unpenalised field constant, where no prior acts on it. The
// standard treatment adds a Gaussian penalty on the field's sum,
//
//     -0.5 * lambda * (sum_i phi_i)^2,
//
// which pins the constant without touching any deviation from it (its Hessian
// is the rank-1 lambda * 11', supported entirely on the constant eigenspace).
//
// The reference idiom (Morris et al. 2019, "Bayesian hierarchical spatial
// models"; the same constant brms and the Stan ICAR case study use) fixes the
// STANDARD DEVIATION of that sum rather than its precision:
//
//     sum(phi) ~ normal(0, kappa * n),   kappa = 0.001
//
// so the pinned quantity scales with the field size: the field MEAN is held at
// sd = kappa regardless of n. `s2z_precision(n)` converts that to the
// precision the penalty bodies below actually take.
//
// Callers pass a PRECISION. Passing `kappa` itself -- the shape of the bug this
// header exists to prevent -- is weaker by a factor of (kappa*n)^-2 * kappa^-1,
// i.e. ~2.5e6 at n = 20, which leaves the constant direction free at sd 31.6
// and the constraint effectively absent.

#ifndef TULPA_SOFT_SUM_TO_ZERO_H
#define TULPA_SOFT_SUM_TO_ZERO_H

namespace tulpa {

// sd(sum_i phi_i) = S2Z_KAPPA * n.
constexpr double S2Z_KAPPA = 0.001;

// Precision of the soft sum-to-zero penalty over a sum of `n` field values.
// `n` is the number of terms IN THE SUM, not the size of the whole field: a
// field pinned per connected component passes the component size, and an
// interaction pinned along both margins passes each margin's own length.
inline double s2z_precision(int n, double kappa = S2Z_KAPPA) {
    const double sd = kappa * static_cast<double>(n > 0 ? n : 1);
    return 1.0 / (sd * sd);
}

}  // namespace tulpa

#endif  // TULPA_SOFT_SUM_TO_ZERO_H
