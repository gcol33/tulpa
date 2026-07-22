// sum_to_zero.h
// Identification of the constant null direction of an intrinsic
// (rank-deficient) latent field: ICAR, RW1, RW2, cyclic RW, and the
// interaction fields built from them.
//
// An intrinsic precision Q has one constant null direction per connected
// component. That direction is jointly unidentified with the intercept, so it
// has to be removed before the field reaches the linear predictor.
//
// The field is identified by AUGMENTING the precision rather than penalising
// the sum. Per component c of size J_c,
//
//     Q_aug = Q + sum_c (1_c 1_c') / J_c,
//
// so the constant direction carries the field's own precision tau, and the
// field is CENTRED per component on its way into eta. The direction is then
// absent from the likelihood entirely: it is a free N(0, 1/tau) draw that
// integrates out, rather than a stiff direction the sampler still has to
// traverse. This is what INLA's `constr=TRUE` and Stan's `sum_to_zero_vector`
// do (gcol33/tulpa#241).
//
// Two consequences, both load-bearing:
//
//   1. RANK. 1_c 1_c'/J_c has eigenvalue 1 on component c's constant direction,
//      where Q had 0, so Q_aug is FULL RANK. The normalizer takes
//      J * log tau, not (J - n_components) * log tau. Keeping the deficient
//      rank while adding the augmentation term makes the tau-marginal wrong and
//      biases the variance component low.
//
//   2. CENTRING IS NOT OPTIONAL. The augmented constant direction carries
//      precision tau (order 1), where the soft penalty this replaced carried
//      1/(kappa*J)^2 (400 at J = 50). Augmenting a path that does NOT centre
//      leaves the level ~400x freer than before, which is the aliasing of #241
//      made worse. A path either does both or neither.
//
// Equivalence: integrating the n_components freed constants contributes
// -0.5 * n_components * log tau, cancelling the +0.5 * n_components * log tau
// the full rank adds, so this agrees with the hard-constrained density on the
// sum-to-zero subspace for every identified quantity.

#ifndef TULPA_SUM_TO_ZERO_H
#define TULPA_SUM_TO_ZERO_H

namespace tulpa {

// Coefficient of (sum_{i in c} phi_i)^2 in the augmented quadratic form:
// phi' Q_aug phi = phi' Q phi + sum_c (sum_{i in c} phi_i)^2 / J_c, scaled by
// the field precision tau. `n` is the size of the component being summed, not
// the size of the whole field.
template <typename T>
inline T s2z_aug_coef(const T& tau, int n) {
    return tau / T(static_cast<double>(n > 0 ? n : 1));
}

// Rank of Q_aug = Q + sum_c 1_c 1_c'/J_c.
//
// The augmentation fills exactly ONE direction per pinned component -- that
// component's constant -- so the rank is rank(Q) plus the number of pins. It is
// NOT the field length in general, and the difference is not cosmetic:
//
//   ICAR, L components   null space = the L constants      (J - L) + L = J
//   RW1                  null space = the constant         (n - 1) + 1 = n
//   cyclic RW1/RW2       null space = the constant         (n - 1) + 1 = n
//   RW2, non-cyclic      null space = constant AND LINEAR  (n - 2) + 1 = n - 1
//
// An RW2 field stays rank-deficient by one after a sum-to-zero augmentation,
// because nothing here touches its linear null direction. Passing the field
// length there would overstate the rank and bias the variance component.
inline int s2z_aug_rank(int rank_Q, int n_pins) {
    const int r = rank_Q + n_pins;
    return r > 0 ? r : 0;
}

// Absolute index of the k-th member of a component whose base offset is `start`
// and whose field-local member list is `idx` (nullptr means the contiguous run
// start..start+size-1). One accessor so the contiguous and the general
// (disconnected / non-contiguous) node layouts share every primitive below.
inline int s2z_node(int start, const int* idx, int k) {
    return start + (idx ? idx[k] : k);
}

// Sum of one component. `idx` is the field-local node list (nullptr for the
// contiguous run [start, start + size)).
template <typename T>
inline T s2z_component_sum(const T* phi, int start, const int* idx, int size) {
    T s = T(0.0);
    for (int i = 0; i < size; ++i) s = s + phi[s2z_node(start, idx, i)];
    return s;
}
template <typename T>
inline T s2z_component_sum(const T* phi, int start, int size) {
    return s2z_component_sum(phi, start, static_cast<const int*>(nullptr), size);
}

// Mean of one component. The quantity subtracted on the way into eta.
template <typename T>
inline T s2z_component_mean(const T* phi, int start, const int* idx, int size) {
    if (size <= 0) return T(0.0);
    return s2z_component_sum(phi, start, idx, size)
         / T(static_cast<double>(size));
}
template <typename T>
inline T s2z_component_mean(const T* phi, int start, int size) {
    return s2z_component_mean(phi, start, static_cast<const int*>(nullptr), size);
}

// Augmented quadratic contribution of one component: tau * (sum)^2 / size.
// Enters the log-prior as -0.5 * this.
template <typename T>
inline T s2z_aug_quad(const T* phi, int start, const int* idx, int size,
                      const T& tau) {
    const T s = s2z_component_sum(phi, start, idx, size);
    return s2z_aug_coef(tau, size) * s * s;
}
template <typename T>
inline T s2z_aug_quad(const T* phi, int start, int size, const T& tau) {
    return s2z_aug_quad(phi, start, static_cast<const int*>(nullptr), size, tau);
}

// Centre one component in place, returning the removed mean so the caller can
// fold it into the intercept where the path keeps eta invariant.
template <typename T>
inline T s2z_centre_component(T* phi, int start, int size) {
    const T m = s2z_component_mean(phi, start, size);
    for (int i = 0; i < size; ++i) phi[start + i] = phi[start + i] - m;
    return m;
}

}  // namespace tulpa

#endif  // TULPA_SUM_TO_ZERO_H
