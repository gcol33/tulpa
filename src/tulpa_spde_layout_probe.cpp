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
