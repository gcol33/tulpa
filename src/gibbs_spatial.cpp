// gibbs_spatial.cpp
// Rcpp interface for the Gibbs spatial sampler

#include "gibbs_spatial.h"
#include <Rcpp.h>

// [[Rcpp::export]]
Rcpp::List cpp_gibbs_spatial(Rcpp::List data_list) {
    using namespace gibbs;

    // Extract data from R list
    Rcpp::IntegerVector y_num_r = data_list["y_num"];
    Rcpp::NumericVector X_num_r = data_list["X_num"];
    Rcpp::IntegerVector spatial_group_r = data_list["spatial_group"];
    Rcpp::IntegerVector adj_row_ptr_r = data_list["adj_row_ptr"];
    Rcpp::IntegerVector adj_col_idx_r = data_list["adj_col_idx"];
    Rcpp::IntegerVector n_neighbors_r = data_list["n_neighbors"];

    int N = Rcpp::as<int>(data_list["N"]);
    int S = Rcpp::as<int>(data_list["S"]);
    int p_num = Rcpp::as<int>(data_list["p_num"]);
    int p_denom = Rcpp::as<int>(data_list["p_denom"]);
    int n_iter = Rcpp::as<int>(data_list["n_iter"]);
    int n_warmup = Rcpp::as<int>(data_list["n_warmup"]);
    int thin = Rcpp::as<int>(data_list["thin"]);
    unsigned int seed = Rcpp::as<unsigned int>(data_list["seed"]);
    bool verbose = Rcpp::as<bool>(data_list["verbose"]);
    std::string family_str = Rcpp::as<std::string>(data_list["family"]);

    // Map family string
    GibbsFamily family;
    if (family_str == "poisson_gamma") family = GibbsFamily::POISSON_GAMMA;
    else if (family_str == "negbin_negbin") family = GibbsFamily::NEGBIN_NEGBIN;
    else if (family_str == "binomial") family = GibbsFamily::BINOMIAL;
    else if (family_str == "negbin_gamma") family = GibbsFamily::NEGBIN_GAMMA;
    else if (family_str == "gamma_gamma") family = GibbsFamily::GAMMA_GAMMA;
    else if (family_str == "lognormal") family = GibbsFamily::LOGNORMAL;
    else if (family_str == "beta_binomial") family = GibbsFamily::BETA_BINOMIAL;
    else Rcpp::stop("Gibbs sampler: unsupported family '%s'", family_str.c_str());

    bool is_binomial = (family == GibbsFamily::BINOMIAL || family == GibbsFamily::BETA_BINOMIAL);

    // Build GibbsData struct
    GibbsData d;
    d.N = N;
    d.S = S;
    d.p_num = p_num;
    d.p_denom = is_binomial ? 0 : p_denom;
    d.family = family;
    // BYM2 settings
    bool is_bym2 = data_list.containsElementNamed("is_bym2") &&
                    Rcpp::as<bool>(data_list["is_bym2"]);
    d.is_bym2 = is_bym2;
    d.bym2_scale = data_list.containsElementNamed("bym2_scale") ?
                   Rcpp::as<double>(data_list["bym2_scale"]) : 1.0;

    // Convert spatial_group from 1-based to 0-based
    std::vector<int> sg(N);
    for (int i = 0; i < N; i++) sg[i] = spatial_group_r[i] - 1;
    d.spatial_group = sg.data();

    d.y_num = y_num_r.begin();
    d.y_num_cont = nullptr;
    d.X_num = REAL(X_num_r);

    // Continuous numerator (gamma_gamma, lognormal)
    Rcpp::NumericVector y_num_cont_r;
    bool needs_cont_num = (family == GibbsFamily::GAMMA_GAMMA || family == GibbsFamily::LOGNORMAL);
    if (needs_cont_num && data_list.containsElementNamed("y_num_cont")) {
        y_num_cont_r = Rcpp::as<Rcpp::NumericVector>(data_list["y_num_cont"]);
        d.y_num_cont = REAL(y_num_cont_r);
    }

    // Denominator data (depends on family)
    Rcpp::NumericVector X_denom_r;
    Rcpp::IntegerVector y_denom_int_r;
    Rcpp::NumericVector y_denom_cont_r;

    if (!is_binomial) {
        X_denom_r = Rcpp::as<Rcpp::NumericVector>(data_list["X_denom"]);
        d.X_denom = REAL(X_denom_r);
    } else {
        d.X_denom = nullptr;
    }

    if (family == GibbsFamily::NEGBIN_NEGBIN) {
        y_denom_int_r = Rcpp::as<Rcpp::IntegerVector>(data_list["y_denom"]);
        d.y_denom = y_denom_int_r.begin();
        d.y_denom_cont = nullptr;
    } else if (is_binomial) {
        // Binomial, Beta-Binomial: integer trials
        y_denom_int_r = Rcpp::as<Rcpp::IntegerVector>(data_list["y_denom"]);
        d.y_denom = y_denom_int_r.begin();
        d.y_denom_cont = nullptr;
    } else {
        // Poisson-Gamma, NegBin-Gamma, Gamma-Gamma, Lognormal: continuous denominator
        y_denom_cont_r = Rcpp::as<Rcpp::NumericVector>(data_list["y_denom_cont"]);
        d.y_denom_cont = REAL(y_denom_cont_r);
        d.y_denom = nullptr;
    }

    // Adjacency (convert from 1-based to 0-based)
    std::vector<int> adj_rp(adj_row_ptr_r.begin(), adj_row_ptr_r.end());
    std::vector<int> adj_ci(adj_col_idx_r.size());
    for (int k = 0; k < (int)adj_col_idx_r.size(); k++)
        adj_ci[k] = adj_col_idx_r[k] - 1;  // 0-based
    std::vector<int> nn(n_neighbors_r.begin(), n_neighbors_r.end());
    d.adj_row_ptr = adj_rp.data();
    d.adj_col_idx = adj_ci.data();
    d.n_neighbors = nn.data();

    // Build site->obs mapping (ICAR/BYM2 only, not HSGP)
    if (S > 0 && !d.is_hsgp) {
        build_site_obs_map(d);
    }

    // TVC data
    std::vector<int> tvc_ti, tvc_gi;
    std::vector<double> tvc_X;
    if (data_list.containsElementNamed("has_tvc") && Rcpp::as<bool>(data_list["has_tvc"])) {
        d.has_tvc = true;
        d.tvc_n_times = Rcpp::as<int>(data_list["tvc_n_times"]);
        d.tvc_n_groups = Rcpp::as<int>(data_list["tvc_n_groups"]);
        d.tvc_n_terms = Rcpp::as<int>(data_list["tvc_n_terms"]);
        d.tvc_shared = Rcpp::as<bool>(data_list["tvc_shared"]);
        tvc_ti = Rcpp::as<std::vector<int>>(data_list["tvc_time_index"]);
        tvc_gi = Rcpp::as<std::vector<int>>(data_list["tvc_group_index"]);
        tvc_X = Rcpp::as<std::vector<double>>(data_list["tvc_X"]);
        d.tvc_time_index = tvc_ti;
        d.tvc_group_index = tvc_gi;
        d.tvc_X = tvc_X;
        // Build time->obs mapping
        build_time_obs_map(d);
        if (verbose) {
            Rcpp::Rcout << "  TVC: " << d.tvc_n_terms << " term(s), "
                        << d.tvc_n_times << " time points" << std::endl;
        }
    }

    // Temporal GMRF data
    std::vector<int> temp_ti, temp_gi;
    if (data_list.containsElementNamed("has_temporal") && Rcpp::as<bool>(data_list["has_temporal"])) {
        d.has_temporal = true;
        d.temporal_n_times = Rcpp::as<int>(data_list["temporal_n_times"]);
        d.temporal_n_groups = Rcpp::as<int>(data_list["temporal_n_groups"]);
        d.temporal_shared = Rcpp::as<bool>(data_list["temporal_shared"]);
        std::string temp_type = Rcpp::as<std::string>(data_list["temporal_type"]);
        d.temporal_type = (temp_type == "ar1") ? 1 : 0;  // 0=RW1, 1=AR1
        temp_ti = Rcpp::as<std::vector<int>>(data_list["temporal_time_index"]);
        temp_gi = Rcpp::as<std::vector<int>>(data_list["temporal_group_index"]);
        d.temporal_time_idx = temp_ti;
        d.temporal_group_idx = temp_gi;
        if (verbose) {
            Rcpp::Rcout << "  Temporal: " << temp_type << " (" << d.temporal_n_times
                        << " time points, " << d.temporal_n_groups << " groups)" << std::endl;
        }
    }

    // HSGP data
    bool is_hsgp = data_list.containsElementNamed("is_hsgp") &&
                   Rcpp::as<bool>(data_list["is_hsgp"]);
    d.is_hsgp = is_hsgp;
    if (is_hsgp) {
        d.hsgp_m_total = Rcpp::as<int>(data_list["hsgp_m_total"]);
        d.hsgp_Phi_flat = Rcpp::as<std::vector<double>>(data_list["hsgp_Phi"]);
        d.hsgp_eigenvalues = Rcpp::as<std::vector<double>>(data_list["hsgp_eigenvalues"]);
        d.hsgp_shared = data_list.containsElementNamed("hsgp_shared") ?
                        Rcpp::as<bool>(data_list["hsgp_shared"]) : true;
    }

    // Run Gibbs sampler (ICAR, BYM2, or HSGP)
    GibbsResult res;
    if (is_hsgp) {
        if (verbose) {
            Rcpp::Rcout << "Running Gibbs sampler (HSGP)..." << std::endl;
            Rcpp::Rcout << "  Basis functions: " << d.hsgp_m_total
                        << ", Observations: " << N << std::endl;
            Rcpp::Rcout << "  Iterations: " << n_iter << " (warmup: " << n_warmup << ")" << std::endl;
        }
        res = run_gibbs_hsgp(d, n_iter, n_warmup, thin, seed, verbose);
        if (verbose) {
            Rcpp::Rcout << "Gibbs complete." << std::endl;
            Rcpp::Rcout << "  HSGP beta acceptance: " << res.accept_hsgp_beta << std::endl;
            Rcpp::Rcout << "  HSGP hyper acceptance: " << res.accept_hsgp_hyper << std::endl;
            Rcpp::Rcout << "  Beta acceptance: " << res.accept_beta << std::endl;
            Rcpp::Rcout << "  Dispersion acceptance: " << res.accept_disp << std::endl;
            if (d.has_tvc)
                Rcpp::Rcout << "  TVC acceptance: " << res.accept_tvc << std::endl;
            if (d.has_temporal)
                Rcpp::Rcout << "  Temporal acceptance: " << res.accept_temporal << std::endl;
        }
    } else {
        if (verbose) {
            Rcpp::Rcout << "Running Gibbs sampler (" << (is_bym2 ? "BYM2" : "ICAR") << ")..." << std::endl;
            Rcpp::Rcout << "  Sites: " << S << ", Observations: " << N << std::endl;
            Rcpp::Rcout << "  Iterations: " << n_iter << " (warmup: " << n_warmup << ")" << std::endl;
            if (is_bym2) Rcpp::Rcout << "  BYM2 scale factor: " << d.bym2_scale << std::endl;
        }
        res = is_bym2 ?
            run_gibbs_bym2(d, n_iter, n_warmup, thin, seed, verbose) :
            run_gibbs_icar(d, n_iter, n_warmup, thin, seed, verbose);
        if (verbose) {
            Rcpp::Rcout << "Gibbs complete." << std::endl;
            double avg_phi_accept = 0.0;
            for (int s = 0; s < S; s++) avg_phi_accept += res.accept_phi[s];
            avg_phi_accept /= S;
            Rcpp::Rcout << "  Avg phi acceptance: " << avg_phi_accept << std::endl;
            Rcpp::Rcout << "  Beta acceptance: " << res.accept_beta << std::endl;
            Rcpp::Rcout << "  Dispersion acceptance: " << res.accept_disp << std::endl;
            if (d.has_tvc)
                Rcpp::Rcout << "  TVC acceptance: " << res.accept_tvc << std::endl;
            if (d.has_temporal)
                Rcpp::Rcout << "  Temporal acceptance: " << res.accept_temporal << std::endl;
        }
    }

    // Convert param names
    Rcpp::CharacterVector param_names(res.param_names.size());
    for (size_t i = 0; i < res.param_names.size(); i++)
        param_names[i] = res.param_names[i];

    // Return results
    Rcpp::List result = Rcpp::List::create(
        Rcpp::Named("draws") = Rcpp::NumericMatrix(res.n_save, res.n_params,
                                                     res.draws_flat.begin()),
        Rcpp::Named("param_names") = param_names,
        Rcpp::Named("n_save") = res.n_save,
        Rcpp::Named("n_params") = res.n_params,
        Rcpp::Named("S") = S,
        Rcpp::Named("accept_beta") = res.accept_beta,
        Rcpp::Named("accept_disp") = res.accept_disp,
        Rcpp::Named("is_hsgp") = is_hsgp
    );

    // Phi draws (ICAR/BYM2 only)
    if (!is_hsgp && S > 0) {
        result["phi_draws"] = Rcpp::NumericMatrix(res.n_save, S, res.phi_draws_flat.begin());
        result["accept_phi"] = Rcpp::wrap(res.accept_phi);
    }

    // HSGP draws
    if (is_hsgp && res.hsgp_m_total > 0) {
        result["hsgp_beta_draws"] = Rcpp::NumericMatrix(res.n_save, res.hsgp_m_total,
                                                          res.hsgp_beta_draws_flat.begin());
        result["hsgp_f_draws"] = Rcpp::NumericMatrix(res.n_save, N,
                                                       res.hsgp_f_draws_flat.begin());
        result["accept_hsgp_beta"] = res.accept_hsgp_beta;
        result["accept_hsgp_hyper"] = res.accept_hsgp_hyper;
    }

    // TVC results
    if (d.has_tvc && res.tvc_n_w > 0) {
        result["tvc_w_draws"] = Rcpp::NumericMatrix(res.n_save, res.tvc_n_w,
                                                     res.tvc_w_draws_flat.begin());
        result["tvc_tau_draws"] = Rcpp::NumericMatrix(res.n_save, d.tvc_n_terms,
                                                       res.tvc_tau_draws_flat.begin());
        result["accept_tvc"] = res.accept_tvc;
    }

    // Temporal GMRF results
    if (d.has_temporal && res.temporal_n_params > 0) {
        result["temporal_draws"] = Rcpp::NumericMatrix(res.n_save, res.temporal_n_params,
                                                        res.temporal_draws_flat.begin());
        result["accept_temporal"] = res.accept_temporal;
    }

    return result;
}
