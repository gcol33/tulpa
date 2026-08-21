// spde_zero_mass.h
// The FEM lumped-mass floor and the orphan diagonal ridge, shared by every
// SPDE precision assembly.
//
// Upstream mesh refiners emit Steiner points without retriangulating, so a
// vertex can reach the assembly with zero (or near-zero) FEM mass. Two flavors
// occur: truly disconnected vertices (zero mass AND no G1 connectivity), and
// zero-mass vertices that DO carry G1 connectivity (a Steiner point on a
// constraint edge with degenerate incident-triangle area). Every operator level
// above the stiffness matrix carries a C^-1, undefined where C_diag = 0, so the
// contribution for that row zeros out and Q is rank-deficient at the node;
// CHOLMOD then reports "not positive definite" and no solver on the model makes
// progress.
//
// The treatment in both flavors: floor the inverse mass at zero and place a
// unit precision ridge on the orphan diagonal so Q stays PD. The FEM projector
// A is built only from triangles with valid area, so an orphan latent carries no
// likelihood weight and is effectively pinned at zero.
//
// The ridge is theta-INDEPENDENT: it is added to Q after the (kappa, tau)
// assembly, so d Q / d theta reads the assembled part alone. A consumer that
// differentiates Q must subtract the ridge before applying a closed form that
// assumes Q is homogeneous in tau (dQ/dlog_tau = 2 Q holds for the assembled
// part only).

#ifndef TULPA_SPDE_ZERO_MASS_H
#define TULPA_SPDE_ZERO_MASS_H

namespace tulpa {

// FEM lumped-mass entries at or below this are treated as zero.
constexpr double SPDE_C0_EPS = 1e-15;

// Precision placed on an orphan node's diagonal.
constexpr double SPDE_ORPHAN_RIDGE = 1.0;

inline bool spde_is_orphan_mass(double c0) { return c0 <= SPDE_C0_EPS; }

// 1 / c0 with the floor applied: an orphan node drops every path that runs
// through it out of the operator chain.
inline double spde_c0_inv(double c0) {
    return (c0 > SPDE_C0_EPS) ? 1.0 / c0 : 0.0;
}

} // namespace tulpa

#endif // TULPA_SPDE_ZERO_MASS_H
