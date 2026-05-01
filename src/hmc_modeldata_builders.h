// hmc_modeldata_builders.h
// Shared Rcpp helpers for constructing legacy HMC ModelData blocks.
//
// These helpers keep the two Rcpp HMC entry points aligned. They preserve the
// historical row-major layout used by the hot gradient paths and centralize
// string-to-enum parsing for the legacy ratio-model fields.

#ifndef TULPA_HMC_MODELDATA_BUILDERS_H
#define TULPA_HMC_MODELDATA_BUILDERS_H

#include <Rcpp.h>
#include <string>
#include <vector>
#include "hmc_zi.h"
#include "tulpa/model_data.h"

namespace tulpa_hmc {

inline void flatten_numeric_matrix(const Rcpp::NumericMatrix& matrix,
                                   int n_rows,
                                   int n_cols,
                                   std::vector<double>& out) {
  out.resize(n_rows * n_cols);
  for (int i = 0; i < n_rows; i++) {
    for (int j = 0; j < n_cols; j++) {
      out[i * n_cols + j] = matrix(i, j);
    }
  }
}

// R callers sometimes pass random-effect group matrices as numeric matrices.
// Normalize to integer storage before copying into ModelData.
inline Rcpp::IntegerMatrix integer_matrix_or_dummy(SEXP matrix_sexp) {
  if (!Rf_isMatrix(matrix_sexp)) {
    Rcpp::IntegerMatrix dummy(1, 1);
    dummy(0, 0) = 0;
    return dummy;
  }

  if (TYPEOF(matrix_sexp) == INTSXP) {
    return Rcpp::as<Rcpp::IntegerMatrix>(matrix_sexp);
  }

  Rcpp::NumericMatrix numeric_matrix(matrix_sexp);
  Rcpp::IntegerMatrix integer_matrix(numeric_matrix.nrow(), numeric_matrix.ncol());
  for (int i = 0; i < numeric_matrix.nrow(); i++) {
    for (int j = 0; j < numeric_matrix.ncol(); j++) {
      integer_matrix(i, j) = static_cast<int>(numeric_matrix(i, j));
    }
  }
  return integer_matrix;
}

// Optional design matrices are represented as a 1x1 placeholder when the
// corresponding parameter count is zero. Callers must pass the explicit p_* so
// placeholders are never read as N-row design matrices.
inline Rcpp::NumericMatrix numeric_matrix_or_dummy(SEXP matrix_sexp) {
  if (!Rf_isNull(matrix_sexp) && Rf_isMatrix(matrix_sexp)) {
    return Rcpp::as<Rcpp::NumericMatrix>(matrix_sexp);
  }

  Rcpp::NumericMatrix dummy(1, 1);
  dummy(0, 0) = 1.0;
  return dummy;
}

// Legacy fit paths historically fell back to poisson-gamma for unknown strings.
// Keep that behavior centralized so cpp_hmc_fit() and cpp_hmc_fit_gp() agree.
inline tulpa::ModelType parse_legacy_model_type(const std::string& model_type_str) {
  if (model_type_str == "binomial") {
    return tulpa::ModelType::BINOMIAL;
  }
  if (model_type_str == "negbin_negbin") {
    return tulpa::ModelType::NEGBIN_NEGBIN;
  }
  if (model_type_str == "poisson_gamma") {
    return tulpa::ModelType::POISSON_GAMMA;
  }
  if (model_type_str == "negbin_gamma") {
    return tulpa::ModelType::NEGBIN_GAMMA;
  }
  if (model_type_str == "gamma_gamma") {
    return tulpa::ModelType::GAMMA_GAMMA;
  }
  if (model_type_str == "lognormal") {
    return tulpa::ModelType::LOGNORMAL;
  }
  if (model_type_str == "beta_binomial") {
    return tulpa::ModelType::BETA_BINOMIAL;
  }
  return tulpa::ModelType::POISSON_GAMMA;
}

// Populate response vectors and flattened numerator/denominator design
// matrices shared by all legacy ratio-model HMC paths.
inline void populate_legacy_ratio_data(tulpa::ModelData& data,
                                       const Rcpp::IntegerVector& y_num,
                                       const Rcpp::IntegerVector& y_denom,
                                       const Rcpp::NumericVector& y_num_cont,
                                       const Rcpp::NumericVector& y_denom_cont,
                                       const Rcpp::NumericMatrix& X_num,
                                       const Rcpp::NumericMatrix& X_denom,
                                       const std::string& model_type_str) {
  data.legacy.y_num = std::vector<int>(y_num.begin(), y_num.end());
  data.legacy.y_denom = std::vector<int>(y_denom.begin(), y_denom.end());
  data.legacy.y_num_cont = std::vector<double>(y_num_cont.begin(), y_num_cont.end());
  data.legacy.y_denom_cont = std::vector<double>(y_denom_cont.begin(), y_denom_cont.end());

  data.N = y_num.size();
  data.legacy.p_num = X_num.ncol();
  data.legacy.p_denom = X_denom.ncol();
  data.legacy.model_type = parse_legacy_model_type(model_type_str);

  flatten_numeric_matrix(X_num, data.N, data.legacy.p_num, data.legacy.X_num_flat);
  flatten_numeric_matrix(X_denom, data.N, data.legacy.p_denom, data.legacy.X_denom_flat);
}

// Parse lattice spatial types used by the non-GP HMC entry point.
inline tulpa::SpatialType parse_lattice_spatial_type(const std::string& spatial_type_str) {
  if (spatial_type_str == "icar") {
    return tulpa::SpatialType::ICAR;
  }
  if (spatial_type_str == "bym2") {
    return tulpa::SpatialType::BYM2;
  }
  if (spatial_type_str == "car_proper") {
    return tulpa::SpatialType::CAR_PROPER;
  }
  return tulpa::SpatialType::NONE;
}

// Parse temporal structures owned directly by the HMC sampler. TVC uses the
// same enum but has a different default, so it has a separate parser below.
inline tulpa::TemporalType parse_hmc_temporal_type(const std::string& temporal_type_str) {
  if (temporal_type_str == "rw1") {
    return tulpa::TemporalType::RW1;
  }
  if (temporal_type_str == "rw2") {
    return tulpa::TemporalType::RW2;
  }
  if (temporal_type_str == "ar1") {
    return tulpa::TemporalType::AR1;
  }
  if (temporal_type_str == "gp") {
    return tulpa::TemporalType::GP;
  }
  return tulpa::TemporalType::NONE;
}

// TVC defaults to RW1 for compatibility with the pre-split Rcpp setup.
inline tulpa::TemporalType parse_tvc_temporal_type(const std::string& structure_str) {
  if (structure_str == "rw1") {
    return tulpa::TemporalType::RW1;
  }
  if (structure_str == "rw2") {
    return tulpa::TemporalType::RW2;
  }
  if (structure_str == "ar1") {
    return tulpa::TemporalType::AR1;
  }
  if (structure_str == "iid") {
    return tulpa::TemporalType::IID;
  }
  return tulpa::TemporalType::RW1;
}

// Zero-inflation design population uses the explicit p_zi supplied by R. This
// matters for OI-only models, which pass a placeholder X_zi with p_zi == 0.
inline void populate_zi_data(tulpa::ModelData& data,
                             const std::string& zi_type_str,
                             const Rcpp::NumericMatrix& X_zi,
                             double zi_prior_sd,
                             int p_zi) {
  data.zi_type = tulpa_zi::parse_zi_type(zi_type_str);
  data.p_zi = p_zi;
  data.zi_prior_sd = zi_prior_sd;
  flatten_numeric_matrix(X_zi, data.N, data.p_zi, data.X_zi_flat);
}

// One-inflation is optional and has no design block when p_oi == 0.
inline void populate_oi_data(tulpa::ModelData& data,
                             const Rcpp::NumericMatrix& X_oi,
                             int p_oi,
                             double oi_prior_sd) {
  data.p_oi = p_oi;
  data.oi_prior_sd = oi_prior_sd;
  if (p_oi > 0) {
    flatten_numeric_matrix(X_oi, data.N, data.p_oi, data.X_oi_flat);
  }
}

// Reset the TVC fields that downstream layout code reads when TVC is absent.
inline void clear_tvc_data(tulpa::ModelData& data) {
  data.tvc_data.n_tvc = 0;
  data.tvc_data.n_times = 0;
  data.tvc_data.n_groups = 1;
}

}  // namespace tulpa_hmc

#endif  // TULPA_HMC_MODELDATA_BUILDERS_H
