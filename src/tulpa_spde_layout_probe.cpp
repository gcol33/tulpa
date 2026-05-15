// tulpa_spde_layout_probe.cpp
//
// Test-only Rcpp probe that builds a minimal SpatialType::SPDE ModelData,
// calls compute_param_layout, and returns the resulting slot indices so
// tests can assert the SPDE-NUTS layout invariants without invoking the
// full sampler. Used by tests/testthat/test-spatial-spde-api.R.
//
// Intentionally minimal: no likelihood spec, no projection A, no FEM
// matrices. The allocator only consults data.spde_data.n_mesh and
// data.spde_data.joint_hypers when laying out the SPDE block, so this is
// enough to exercise the slot arithmetic.

#include <Rcpp.h>
#include "hmc_sampler.h"
#include "tulpa/likelihood.h"
#include "tulpa_priors_spde.h"
#include "spde_qbuilder.h"
#include "spde_nc_apply.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// [[Rcpp::export]]
Rcpp::List cpp_spde_layout_probe(int n_mesh,
                                  int p,
                                  bool joint_hypers,
                                  int n_extra_params = 0) {
    if (n_mesh <= 0)  Rcpp::stop("n_mesh must be positive");
    if (p     <= 0)   Rcpp::stop("p must be positive");

    ModelData data;
    data.N           = 1;             // unused by the allocator
    data.n_processes = 1;
    data.sigma_beta  = 10.0;

    tulpa::ProcessData proc;
    proc.p = p;
    proc.X_flat.assign(static_cast<size_t>(p), 0.0);
    data.processes.push_back(proc);
    data.sharing.init(1);

    // SPDE block. joint_hypers gates the two hyper slots.
    data.spatial_type        = tulpa::SpatialType::SPDE;
    data.has_spde            = true;
    data.spde_data.n_mesh    = n_mesh;
    data.spde_data.joint_hypers = joint_hypers;

    // Minimal likelihood spec so the extra-param block participates in the
    // allocator's idx accumulation.
    tulpa::LikelihoodSpec spec;
    spec.name           = "probe";
    spec.n_processes    = 1;
    spec.n_extra_params = n_extra_params;
    data.likelihood_spec     = &spec;
    data.model_response_data = nullptr;

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);

    return Rcpp::List::create(
        Rcpp::Named("total_params")        = layout.total_params,
        Rcpp::Named("is_spde")             = layout.is_spde,
        Rcpp::Named("spde_w_start")        = layout.spde_w_start,
        Rcpp::Named("spde_w_end")          = layout.spde_w_end,
        Rcpp::Named("log_kappa_spde_idx")  = layout.log_kappa_spde_idx,
        Rcpp::Named("log_tau_spde_idx")    = layout.log_tau_spde_idx,
        Rcpp::Named("extra_offset")        = layout.extra_offset,
        Rcpp::Named("n_extra_params")      = layout.n_extra_params
    );
}

// Evaluate just the SPDE latent-block prior for a given (z or w) vector.
// Joint-mode (joint_hypers = true): returns -0.5 * sum z^2, ignoring Q.
// Legacy mode (joint_hypers = false): builds Q at (kappa, tau_spde) and
// returns -0.5 * w' Q w. Used by tests to verify the two-branch dispatch.
//
// `vals` corresponds to z (joint) or w (legacy) for the n_mesh-long block.
// FEM matrices C0_diag / G1 are required for the legacy path to build Q;
// they are unused in joint mode and may be empty.
// [[Rcpp::export]]
Rcpp::List cpp_spde_prior_probe(Rcpp::NumericVector vals,
                                  bool joint_hypers,
                                  Rcpp::NumericVector C0_diag,
                                  Rcpp::NumericVector G1_x,
                                  Rcpp::IntegerVector G1_i,
                                  Rcpp::IntegerVector G1_p,
                                  double kappa    = 1.0,
                                  double tau_spde = 1.0,
                                  int    alpha    = 2) {
    const int n_mesh = vals.size();
    if (n_mesh <= 0) Rcpp::stop("vals must be non-empty");

    ModelData data;
    data.N           = 1;
    data.n_processes = 1;
    data.sigma_beta  = 10.0;

    tulpa::ProcessData proc;
    proc.p = 1;
    proc.X_flat.assign(1, 0.0);
    data.processes.push_back(proc);
    data.sharing.init(1);

    data.spatial_type           = tulpa::SpatialType::SPDE;
    data.has_spde               = true;
    auto& sm = data.spde_data;
    sm.n_mesh    = n_mesh;
    sm.joint_hypers = joint_hypers;
    sm.kappa     = kappa;
    sm.tau_spde  = tau_spde;
    sm.alpha     = alpha;

    if (!joint_hypers) {
        if (C0_diag.size() != n_mesh) {
            Rcpp::stop("legacy mode requires C0_diag of length n_mesh");
        }
        sm.C0_diag.assign(C0_diag.begin(), C0_diag.end());
        sm.G1_x.assign(G1_x.begin(), G1_x.end());
        sm.G1_i.assign(G1_i.begin(), G1_i.end());
        sm.G1_p.assign(G1_p.begin(), G1_p.end());

        tulpa::SpdeQBuilder qb;
        qb.init(n_mesh, C0_diag, G1_x, G1_i, G1_p);
        qb.rebuild(kappa, tau_spde, alpha);
        sm.Q_p.assign(qb.Q_p.begin(), qb.Q_p.end());
        sm.Q_i.assign(qb.Q_i.begin(), qb.Q_i.end());
        sm.Q_x.assign(qb.Q_x.begin(), qb.Q_x.end());
    }

    tulpa::LikelihoodSpec spec;
    spec.name        = "probe";
    spec.n_processes = 1;
    data.likelihood_spec     = &spec;
    data.model_response_data = nullptr;

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);

    std::vector<double> params(layout.total_params, 0.0);
    for (int j = 0; j < n_mesh; j++) {
        params[layout.spde_w_start + j] = vals[j];
    }

    std::vector<double> spde_w_out;
    double prior_val = tulpa::priors::compute_spde_prior<double>(
        params, data, layout, spde_w_out);

    return Rcpp::List::create(
        Rcpp::Named("prior_val")     = prior_val,
        Rcpp::Named("spde_w_filled") = !spde_w_out.empty()
    );
}

// Apply the joint-NUTS NC transform z -> w = L^{-T}(theta) z and return w.
// Builds a minimal SPDE ModelData with joint_hypers = true, fills the FEM
// matrices, places z in the param vector at spde_w_start..end, places
// (log_kappa, log_tau) in the hyper slots, and calls
// apply_spde_nc_transform_double. Used by tests to verify the transform
// (i) is consistent with the unit-Gaussian-on-z prior — i.e.
// z^T z == w^T Q(theta) w — and (ii) is linear in z.
// [[Rcpp::export]]
Rcpp::List cpp_spde_nc_apply_probe(Rcpp::NumericVector z,
                                    double log_kappa,
                                    double log_tau,
                                    Rcpp::NumericVector C0_diag,
                                    Rcpp::NumericVector G1_x,
                                    Rcpp::IntegerVector G1_i,
                                    Rcpp::IntegerVector G1_p) {
    const int n_mesh = z.size();
    if (n_mesh <= 0) Rcpp::stop("z must be non-empty");
    if (C0_diag.size() != n_mesh) {
        Rcpp::stop("C0_diag length (%d) must equal n_mesh (%d)",
                   (int)C0_diag.size(), n_mesh);
    }

    ModelData data;
    data.N           = 1;
    data.n_processes = 1;
    data.sigma_beta  = 10.0;

    tulpa::ProcessData proc;
    proc.p = 1;
    proc.X_flat.assign(1, 0.0);
    data.processes.push_back(proc);
    data.sharing.init(1);

    data.spatial_type           = tulpa::SpatialType::SPDE;
    data.has_spde               = true;
    auto& sm = data.spde_data;
    sm.n_mesh       = n_mesh;
    sm.joint_hypers = true;
    sm.C0_diag.assign(C0_diag.begin(), C0_diag.end());
    sm.G1_x.assign(G1_x.begin(), G1_x.end());
    sm.G1_i.assign(G1_i.begin(), G1_i.end());
    sm.G1_p.assign(G1_p.begin(), G1_p.end());

    tulpa::LikelihoodSpec spec;
    spec.name        = "probe";
    spec.n_processes = 1;
    data.likelihood_spec     = &spec;
    data.model_response_data = nullptr;

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);

    std::vector<double> params(layout.total_params, 0.0);
    for (int j = 0; j < n_mesh; j++) {
        params[layout.spde_w_start + j] = z[j];
    }
    params[layout.log_kappa_spde_idx] = log_kappa;
    params[layout.log_tau_spde_idx]   = log_tau;

    std::vector<double> w_out;
    tulpa::apply_spde_nc_transform_double(params, data, layout, w_out);

    return Rcpp::List::create(
        Rcpp::Named("w")                  = Rcpp::wrap(w_out),
        Rcpp::Named("n_mesh")             = n_mesh,
        Rcpp::Named("log_kappa_idx")      = layout.log_kappa_spde_idx,
        Rcpp::Named("log_tau_idx")        = layout.log_tau_spde_idx,
        Rcpp::Named("nc_transform_built") = (bool) sm.nc_transform
    );
}
