// joint_nested_laplace_api.h
// Cross-package nested-Laplace API for *joint multi-likelihood* models.
//
// Mirrors nested_laplace_api.h, but each shim takes a list of arms (per-arm
// y, n_trials, X, spatial_idx, family, phi, optional RE) sharing one latent
// vector x. The first backend is BYM2; the inner Newton primitive
// (laplace_newton_solve_joint) and shim interface generalise to any single
// shared prior block.
//
// Latent layout for a K-arm joint fit on backend B with shared spatial block
// of total size n_lat_struct:
//   x = [ beta_1 (p_1) | ... | beta_K (p_K)
//       | re_1 (n_re_1) | ... | re_K (n_re_K)
//       | structured (n_lat_struct) ]
// Each arm carries its own fixed-effects design matrix (X_k, size N_k x p_k)
// and its own RE block. The structured spatial block is shared across all
// arms (with unit precision) — the per-arm field amplitude enters as a
// multiplier on eta: eta_arm = X beta + sigma_arm * z_s. One arm may be
// flagged as the "copy" arm — it sees sigma_pos_k from a dedicated outer-
// grid axis; all other arms see sigma_occ_k from the donor sigma axis.
//
// gcol33/tulpa#18: replaces the (sigma, alpha) parameterization where alpha
// scaled the copy arm's contribution to a shared sigma * z field. The old
// parameterization produced a posterior ridge along constant alpha * sigma
// at small n_pos because both arms' likelihoods anchored only the *product*
// alpha * sigma. The new (sigma_occ, sigma_pos) axes are anchored
// independently by their own arm's likelihood. alpha = sigma_pos / sigma_occ
// is recovered post-hoc on the R side.
//
// See dev_notes/joint_nested_laplace.md for the full math derivation.

#ifndef TULPA_JOINT_NESTED_LAPLACE_API_H
#define TULPA_JOINT_NESTED_LAPLACE_API_H

#include <R_ext/Rdynload.h>
#include "model_data.h"             // brings in TULPA_ABI_VERSION
#include "nuts_api.h"               // for check_abi_version()
#include "nested_laplace_api.h"     // re-uses NestedLaplaceShimResult

namespace tulpa {

// ----------------------------------------------------------------------------
// JointArmCxx — one arm of a joint nested-Laplace fit (POD, ABI-stable).
//
// Caller owns all pointers. The shim copies into Rcpp containers internally.
// Append-only: new fields go at the end; older shims ignore them.
// ----------------------------------------------------------------------------
struct JointArmCxx {
    const double* y;             // [N]
    const int*    n_trials;      // [N], use 1's for non-binomial
    const double* X_flat;        // [N * p], column-major (Eigen / Rcpp default)
    const double* re_idx;        // [N], 1-based RE group index per obs (0/NA -> no RE)
    const int*    spatial_idx;   // [N], 1-based map obs -> spatial unit
    int           N;
    int           p;
    int           n_re_groups;   // 0 disables RE block for this arm
    double        sigma_re;      // ignored when n_re_groups == 0
    const char*   family;        // "binomial", "gaussian", "poisson", "neg_binomial_2", "beta", ...
    double        phi;           // dispersion (gaussian SD, negbin size, beta precision)
};

// ----------------------------------------------------------------------------
// BYM2 joint shim. Outer grid over (sigma_occ, rho [, sigma_pos]).
//
// arms           : [n_arms] arm specs, see JointArmCxx.
// copy_arm       : 0-based index of the arm that uses sigma_pos. Pass -1 for
//                  "no copy" (all arms use sigma_occ, sigma_pos_grid ignored).
// sigma_occ_grid : [n_grid] donor-arm field amplitude (sigma on all
//                  non-copy arms).
// sigma_pos_grid : [n_grid] copy-arm field amplitude (paired pointwise with
//                  sigma_occ_grid / rho_grid). Ignored when copy_arm == -1
//                  (caller may pass nullptr).
//
// The outer grid is *paired*, not Cartesian: caller builds the Cartesian
// product before calling. Length n_grid for all three vectors.
//
// store_modes is always 1 for this shim. store_Q is honoured per the
// existing single-arm convention.
// ----------------------------------------------------------------------------
typedef void (*NestedLaplaceJointBym2Fn)(
    const JointArmCxx* arms, int n_arms,
    int copy_arm,
    int n_spatial_units,
    const int* adj_row_ptr, const int* adj_col_idx, const int* n_neighbors,
    double scale_factor,
    const double* sigma_occ_grid,
    const double* rho_grid,
    const double* sigma_pos_grid,
    int n_grid,
    int max_iter, double tol, int n_threads,
    const double* x_init, int n_x_init,
    int store_Q,
    NestedLaplaceShimResult* result_out
);

inline NestedLaplaceJointBym2Fn get_nested_laplace_joint_bym2_fn() {
    static NestedLaplaceJointBym2Fn fn = nullptr;
    if (!fn) {
        check_abi_version();
        fn = (NestedLaplaceJointBym2Fn)R_GetCCallable(
            "tulpa", "tulpa_nested_laplace_joint_bym2");
    }
    return fn;
}

} // namespace tulpa

#endif // TULPA_JOINT_NESTED_LAPLACE_API_H
