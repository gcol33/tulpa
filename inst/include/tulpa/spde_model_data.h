// spde_model_data.h
// SpdeModelData — the SPDE FEM + projection block that ModelData carries
// when SpatialType::SPDE is active.
//
// Kept distinct from spde_api.h (which exposes the nested-Laplace shim
// signature for downstream packages). This header lives inside the model_data
// stable section so callers can build a SpatialType::SPDE ModelData without
// pulling in the Laplace shim machinery.
//
// Lifecycle:
//   1. Caller builds the FEM matrices (C0_diag, G1 CSC) + projection A (CSC)
//      from a fmesher/tulpaMesh mesh, plus the Matern smoothness nu.
//   2. Caller supplies the (kappa, tau_spde) at which Q is built. For Phase 1
//      these are fixed across the chain — joint NUTS over (log_kappa,
//      log_tau_spde) is a follow-on arc (see TODO.md).
//   3. tulpa builds Q once at ModelData setup, caches Q_x + log|Q|, and
//      reuses them across every gradient evaluation.

#ifndef TULPA_SPDE_MODEL_DATA_H
#define TULPA_SPDE_MODEL_DATA_H

#include <memory>
#include <vector>

namespace tulpa {

// Forward-decl of the non-centered transform owner. Definition lives in
// src/spde_nc_transform.h and pulls in Eigen, so we only expose the type
// name here. The full type is needed wherever the transform is built or
// invoked (spde_nc_apply.cpp); other translation units that just hold a
// SpdeModelData by value only need this declaration because shared_ptr's
// destructor uses a type-erased deleter.
class SpdeNcTransform;

// Per-observation A-row entry (column index + interpolation weight). The
// FEM projection A is sparse: each obs is a convex combination of ~3
// triangle-vertex weights, so this is a tiny vector per row.
struct SpdeARowEntry {
    int    mesh_idx;
    double weight;
};

struct SpdeModelData {
    // ----- FEM topology (fixed across the chain) -----
    int n_mesh = 0;
    double nu  = 1.0;          // Matern smoothness; alpha = nu + d/2, d = 2.

    // C0 lumped-mass diagonal.
    std::vector<double> C0_diag;

    // G1 stiffness in CSC: G1_x / G1_i / G1_p of length nnz / nnz / n_mesh+1.
    std::vector<double> G1_x;
    std::vector<int>    G1_i;
    std::vector<int>    G1_p;

    // ----- Projection A (n_obs x n_mesh), per-row dense storage -----
    // a_rows[i] is the (mesh_idx, weight) list for observation i.
    std::vector<std::vector<SpdeARowEntry>> a_rows;

    // ----- Hyperparameters -----
    // kappa / tau_spde act as init values for the joint-NUTS hyper slots when
    // joint_hypers == true; they are the fixed values used to build Q (and
    // reused every log-post call) when joint_hypers == false.
    double kappa    = 1.0;     // Matern range proxy: kappa = sqrt(8 nu)/range.
    double tau_spde = 1.0;     // SPDE scale: tau = 1 / (sqrt(4 pi) kappa sigma).
    int    alpha    = 2;       // Integer SPDE operator order; 2 = standard nu=1.

    // Joint-NUTS over (log_kappa, log_tau_spde): when true, the param layout
    // reserves two extra slots after the z-block and the structured HMC path
    // (i) treats spde_w_start..spde_w_end as z (non-centered draws), (ii)
    // computes w = L^{-T}(theta) z per evaluation, (iii) places a PC prior
    // on (range, sigma) per Fuglstad et al. 2019. Default false keeps
    // backward-compat with the legacy fixed-hyper inner-Laplace callers.
    bool joint_hypers = false;

    // ----- Cached Q at (kappa, tau_spde) -----
    // CSC sparsity pattern + numeric values. Built once during ModelData
    // setup and held constant for every log-post call this chain.
    std::vector<int>    Q_p;   // Length n_mesh + 1
    std::vector<int>    Q_i;   // Length nnz
    std::vector<double> Q_x;   // Length nnz
    double log_det_Q = 0.0;    // Cached log|Q| (constant under fixed hypers).

    // ----- Rational SPDE coefficients (optional, fractional alpha) -----
    // Empty vectors disable the rational path and fall back to the integer
    // alpha rebuild.
    std::vector<double> rational_poles;
    std::vector<double> rational_weights;

    // ----- Joint-NUTS non-centered transform cache -----
    // Lazily built on the first log-post evaluation when joint_hypers ==
    // true; remains null in legacy fixed-hyper mode (which uses the
    // Q_p/Q_i/Q_x cache above instead). Holds the Eigen-format FEM
    // matrices, the current (kappa, tau) Cholesky factor, and the adjoint
    // hook from (a.i). Mutable so const ModelData references can populate
    // it during a log-post evaluation.
    //
    // shared_ptr (not unique_ptr) so SpdeModelData stays trivially
    // copy-constructible — laplace_spec.cpp makes a defensive value copy
    // of ModelData to override RE state, and that path never sets
    // joint_hypers, so the shared cache is never exercised across copies
    // in practice. Forward declaration above keeps Eigen out of this
    // public header.
    mutable std::shared_ptr<SpdeNcTransform> nc_transform;
};

} // namespace tulpa

#endif // TULPA_SPDE_MODEL_DATA_H
