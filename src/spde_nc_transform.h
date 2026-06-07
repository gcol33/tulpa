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
// Two operator-order modes, dispatched on whether init() was called with
// rational poles+weights:
//
//   Integer alpha = 2 (default; poles empty):
//       K = kappa^2 C0 + G1
//       Q = tau^2 K diag(1/C0) K
//
//   Rational alpha (fractional nu; poles/weights from Bolin et al. 2023):
//       K_k = (kappa^2 + r_k) C0 + G1
//       Q   = tau^2 sum_k w_k K_k diag(1/C0) K_k
//
//   In both cases the non-centered transform is w = L^{-T} z with
//   L L^T = Q(kappa, tau).
//
// The adjoint preserves the same trace-collapsing structure (Murray 2016).
// Only the closed forms for dQ/dlog_kappa change:
//
//   Integer:  dQ/dlog_kappa = 4 kappa^2 tau^2 K
//             dQ/dlog_tau   = 2 Q
//
//   Rational: dQ/dlog_kappa = 4 kappa^2 tau^2 K_sum
//             dQ/dlog_tau   = 2 Q
//             where K_sum := sum_k w_k K_k
//                          = (kappa^2 W + R_sum) C0 + W G1
//             with W = sum_k w_k, R_sum = sum_k w_k r_k.
//
// The (kappa, tau) <-> (range, sigma) reparameterisation lives in (a.iii)'s
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

    // Rational approximation coefficients (Bolin et al. 2023). Empty means
    // integer alpha = 2 (the default fast path). Both vectors are
    // populated by init() and immutable afterwards.
    std::vector<double> poles_;
    std::vector<double> weights_;
    double              W_     = 0.0;  // sum_k w_k
    double              R_sum_ = 0.0;  // sum_k w_k r_k

    // Cached forward state (filled by forward(); read by backward()).
    double  last_kappa = 0.0;
    double  last_tau   = 0.0;
    SpMat   last_K;       // Integer: kappa^2 C0 + G1. Rational: unused.
    SpMat   last_K_sum;   // Rational: (kappa^2 W + R_sum) C0 + W G1.
                          // Integer:  unused (degenerates to last_K).
    SpMat   last_Q;       // Integer:  tau^2 K D K.
                          // Rational: tau^2 sum_k w_k K_k D K_k.
    SolverT llt;
    bool    factored = false;
    bool    fixed_mode = false;  // set by init_fixed(); see gcol33/tulpa#87.

    void init(int n,
              const std::vector<double>& C0_d,
              const std::vector<double>& G1_x,
              const std::vector<int>&    G1_i,
              const std::vector<int>&    G1_p,
              const std::vector<double>& poles   = {},
              const std::vector<double>& weights = {});

    // Fixed-hyper non-centered setup (gcol33/tulpa#87). Factors a directly
    // supplied precision Q (CSC, full-symmetric storage — the lower triangle
    // is read) once and caches L. No FEM rebuild, no hyper adjoint: the
    // transform is the constant linear map v = L^{-T} z. Works uniformly for
    // the integer SpdeQBuilder Q and the fractional precomputed (rational)
    // Q, since both arrive on SpdeModelData::Q_{p,i,x}.
    void init_fixed(const std::vector<int>&    Q_p,
                    const std::vector<int>&    Q_i,
                    const std::vector<double>& Q_x,
                    int                        n);

    // Fixed-mode forward / adjoint / tangent (no kappa/tau, no hyper grads).
    //   forward_fixed:               v   = L^{-T} z
    //   backward_fixed:              dz  = L^{-1} dv
    //   forward_fixed_with_tangent:  v   = L^{-T} z, dv = L^{-T} dz
    Eigen::VectorXd forward_fixed(const Eigen::VectorXd& z) const;
    void backward_fixed(const Eigen::VectorXd& dv,
                        Eigen::VectorXd&       dz_out) const;
    void forward_fixed_with_tangent(const Eigen::VectorXd& z,
                                    const Eigen::VectorXd& dz,
                                    Eigen::VectorXd&       v_out,
                                    Eigen::VectorXd&       dv_out) const;

    bool is_rational() const { return !poles_.empty(); }

    // K(kappa) = kappa^2 C0 + G1. Equivalent to build_K_shifted(kappa^2, 0).
    SpMat build_K(double kappa) const;

    // K_shifted(kappa^2 + r) = (kappa^2 + r) C0 + G1. Used per-pole in
    // the rational forward to build K_k.
    SpMat build_K_shifted(double k2_eff) const;

    // Integer-alpha Q: tau^2 K diag(1/C0) K.
    SpMat build_Q(const SpMat& K, double tau) const;

    // Rational-alpha Q: tau^2 sum_k w_k K_k diag(1/C0) K_k.
    // Side effect: caches the (kappa^2 W + R_sum) C0 + W G1 matrix in
    // last_K_sum so backward() can read it without rebuilding.
    SpMat build_Q_rational(double kappa, double tau);

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

    // Forward-mode tangent. Given a perturbation direction (dz, dlog_kappa,
    // dlog_tau), fills w_out = L^{-T}(kappa, tau) z and
    //   dw_out = L^{-T} (dz - Phi(M)^T z)
    // where M = L^{-1} dQ L^{-T} and
    //   dQ = dlog_kappa * (4 kappa^2 tau^2 K_for_logkappa)
    //      + dlog_tau   * (2 Q).
    // Caches K, Q, L just like forward(); a subsequent backward() call on
    // the same transform sees the same factorization.
    void forward_with_tangent(
        const Eigen::VectorXd& z,
        const Eigen::VectorXd& dz,
        double                 kappa,
        double                 dlog_kappa,
        double                 tau,
        double                 dlog_tau,
        Eigen::VectorXd&       w_out,
        Eigen::VectorXd&       dw_out);
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

// Fixed-hyper arena hook (gcol33/tulpa#87). Registers a custom_backward
// block with inputs [z[0..n-1]] and outputs [v[0..n-1]] = L^{-T} z. The
// backward callback computes dz = L^{-1} dv only — there are no hyper slots.
// `transform` is captured by reference and must outlive the backward sweep.
std::vector<arena::Var> spde_nc_transform_fixed_arena(
    arena::Arena*                  ar,
    const std::vector<arena::Var>& z,
    SpdeNcTransform&               transform
);

} // namespace tulpa

#endif // TULPA_SPDE_NC_TRANSFORM_H
