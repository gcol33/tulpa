// spde_nc_transform.h
// Non-centered SPDE transform with sparse-Cholesky forward + reverse-mode
// adjoint. Hosts the second leg of gcol33/tulpa#a (joint NUTS over hypers).
//
// Setup. Q(kappa, tau) is the SPDE precision built from FEM matrices
//   C0 (lumped mass, diagonal) and G1 (stiffness). Integer-alpha case
//   (alpha = 2, nu = 1, d = 2):
//     K = kappa^2 C0 + G1
//     Q = tau^2 K diag(1/C0) K
// Cholesky Q = L L^T. The non-centered transform takes a unit-Gaussian
// auxiliary z and produces w = L^{-T} z. Used by joint NUTS over
// (log_kappa, log_tau, z, ...): NUTS samples z ~ N(0, I) directly while
// w marginally has N(0, Q^{-1}).
//
// Forward and adjoint.
//   forward:   factor Q at the current (kappa, tau), back-solve L^T w = z.
//   backward:  given dw = dL_loss/dw, compute (dz, dlog_kappa, dlog_tau)
//              via the Murray (2016) implicit Cholesky derivative.
//
// Phase 1 scope: alpha = 2 only. Rational fractional-alpha and the
// (kappa, tau) <-> (range, sigma) reparameterisation land in (a.iii)'s
// wrapper, not here.
//
// Implementation note. All Eigen-heavy code lives in spde_nc_transform.cpp
// so the auto-generated MinGW export list does not have to enumerate
// thousands of header-inline template instantiations. The arena hook
// `spde_nc_transform_arena` is also out-of-line for the same reason.

#ifndef TULPA_SPDE_NC_TRANSFORM_H
#define TULPA_SPDE_NC_TRANSFORM_H

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <vector>

#include "tulpa/autodiff_arena.h"

namespace tulpa {

class SpdeNcTransform {
public:
    using SpMat   = Eigen::SparseMatrix<double>;
    using SolverT = Eigen::SimplicialLLT<SpMat, Eigen::Lower, Eigen::NaturalOrdering<int>>;

    int             n_mesh = 0;
    Eigen::VectorXd C0_diag;
    Eigen::VectorXd C0_inv_diag;
    SpMat           G1;

    // Cached forward state (filled by forward(); read by backward()).
    double  last_kappa = 0.0;
    double  last_tau   = 0.0;
    SpMat   last_K;       // kappa^2 C0 + G1
    SpMat   last_Q;       // tau^2 K D K
    SolverT llt;
    bool    factored = false;

    void init(int n,
              const std::vector<double>& C0_d,
              const std::vector<double>& G1_x,
              const std::vector<int>&    G1_i,
              const std::vector<int>&    G1_p);

    // K(kappa) = kappa^2 C0 + G1.
    SpMat build_K(double kappa) const;

    // Q(kappa, tau) = tau^2 K diag(1/C0) K.
    SpMat build_Q(const SpMat& K, double tau) const;

    // Forward: w = L^{-T} z. Caches K, Q, L for the matching backward call.
    Eigen::VectorXd forward(const Eigen::VectorXd& z, double kappa, double tau);

    // Backward: given dw = dL_loss/dw, fill (dz, dlog_kappa, dlog_tau).
    void backward(
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& w,
        const Eigen::VectorXd& dw,
        Eigen::VectorXd&       dz_out,
        double&                dlog_kappa_out,
        double&                dlog_tau_out
    ) const;
};

// Arena hook. Registers a custom_backward block with inputs
// [z[0..n-1], log_kappa, log_tau] and outputs [w[0..n-1]] = L^{-T} z.
// `transform` is captured by reference inside the backward callback and
// must outlive the arena's backward sweep.
std::vector<arena::Var> spde_nc_transform_arena(
    arena::Arena*                  ar,
    const std::vector<arena::Var>& z,
    const arena::Var&              log_kappa,
    const arena::Var&              log_tau,
    SpdeNcTransform&               transform
);

} // namespace tulpa

#endif // TULPA_SPDE_NC_TRANSFORM_H
