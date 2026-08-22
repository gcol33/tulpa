// test_hsgp_warm_start.cpp
// Test-only entry point: warm_start_mass_matrix() on a ModelData whose HSGP
// flag and HSGP indices disagree.
//
// layout.is_hsgp follows data.spatial_type alone, while the three HSGP index
// fields are assigned only when data.has_hsgp is also set. A model declaring an
// HSGP field with no basis block therefore reaches the warm start with the flag
// true and the indices at their -1 sentinel, which is a state no fitted path
// can be asked for from R -- the front door builds the two together. This
// assembles it directly, as the layout code allows, and reports both the flag /
// index pair and the diagonal the warm start produced.
//
// The two scalar writes are what the state exposes: the basis-coefficient loop
// above them runs from -1 to -1 and is empty, so a sweep of the other index
// fields is not what is being checked here -- one specific pair is.

#include <Rcpp.h>
#include <vector>

#include "hmc_sampler.h"
#include "hmc_sampler_config.h"
#include "log_post_impl.h"
#include "tulpa/likelihood.h"

using tulpa_hmc::ModelData;
using tulpa_hmc::ParamLayout;

// [[Rcpp::export]]
Rcpp::List cpp_test_hsgp_warm_start(bool has_hsgp, int n_basis = 4,
                                     int n_spatial = 6) {
    ModelData data;
    data.N = n_spatial;
    data.n_processes = 1;
    data.sigma_beta = 10.0;

    tulpa_hmc::ProcessData proc;
    proc.p = 1;
    proc.X_flat.assign(static_cast<std::size_t>(n_spatial), 1.0);
    data.processes.push_back(proc);
    data.sharing.init(1);

    tulpa::LikelihoodSpec spec;
    spec.name = "probe";
    spec.n_processes = 1;
    data.likelihood_spec = &spec;
    data.model_response_data = nullptr;

    data.spatial_type = tulpa::SpatialType::HSGP;
    data.has_hsgp = has_hsgp;
    if (has_hsgp) {
        auto& h = data.hsgp_data;
        h.n_obs = n_spatial;
        h.n_dim = 1;
        h.m_per_dim = n_basis;
        h.m_total = n_basis;
        h.L1 = 1.5;
        h.phi_flat.assign(static_cast<std::size_t>(n_spatial) * n_basis, 0.0);
        h.eigenvalues.assign(static_cast<std::size_t>(n_basis), 1.0);
        h.coords_scaled.assign(static_cast<std::size_t>(n_spatial), 0.0);
    }

    ParamLayout layout = tulpa_hmc::compute_param_layout(data);

    tulpa_hmc::DenseMassMatrix mass;
    mass.init(layout.total_params, tulpa_hmc::MassMatrixType::DIAG);
    tulpa_hmc::warm_start_mass_matrix(mass, data, layout, layout.total_params,
                                      /*verbose=*/false);

    return Rcpp::List::create(
        Rcpp::_["is_hsgp"]      = layout.is_hsgp,
        Rcpp::_["sigma2_idx"]   = layout.log_sigma2_hsgp_idx,
        Rcpp::_["ls_idx"]       = layout.log_lengthscale_hsgp_idx,
        Rcpp::_["beta_start"]   = layout.hsgp_beta_start,
        Rcpp::_["beta_end"]     = layout.hsgp_beta_end,
        Rcpp::_["total_params"] = layout.total_params,
        Rcpp::_["inv_mass"]     = Rcpp::NumericVector(
            mass.inv_mass_diag.begin(), mass.inv_mass_diag.end()),
        Rcpp::_["sqrt_mass"]    = Rcpp::NumericVector(
            mass.sqrt_mass_diag.begin(), mass.sqrt_mass_diag.end())
    );
}
