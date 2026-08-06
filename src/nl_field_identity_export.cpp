// nl_field_identity_export.cpp
// Equivalence probe for NlFieldIdentity (gcol33/tulpa#286).
//
// The eleven cpp_nested_laplace_* entry points used to fold their structural
// fingerprint by hand, which put the same byte-fold loops in eleven places.
// NlFieldIdentity (nested_laplace_checkpoint.h) names each structural group
// once and the entry points chain the groups they carry.
//
// That fingerprint keys the grid checkpoint, so its VALUE is a contract: a
// resumed run loads a cell only when the seed matches, and a seed that shifts
// silently invalidates every checkpoint on disk while a seed that stops
// distinguishing two structures makes a resume reuse cells it should not. The
// reference sequences below are the folds the entry points wrote by hand,
// transcribed verbatim, and test-nl-field-identity.R checks the builder
// reproduces each one bit for bit. The duplication is deliberate here: in a
// test the second copy is the oracle.

#include <Rcpp.h>

#include <cstdint>
#include <string>

#include "nested_laplace_checkpoint.h"

namespace {

// The pre-extraction fold for an areal field, with the BYM2 mixing scale in the
// slot it occupied between the unit count and the adjacency.
void ref_areal(tulpa::Fingerprint& sfp, int n_spatial_units,
               const Rcpp::IntegerVector& adj_row_ptr,
               const Rcpp::IntegerVector& adj_col_idx,
               bool with_scale, double scale_factor) {
    sfp.fold_pod(n_spatial_units);
    if (with_scale) sfp.fold_pod(scale_factor);
    if (adj_row_ptr.size()) sfp.fold(adj_row_ptr.begin(),
                                     (std::size_t)adj_row_ptr.size() * sizeof(int));
    if (adj_col_idx.size()) sfp.fold(adj_col_idx.begin(),
                                     (std::size_t)adj_col_idx.size() * sizeof(int));
}

void ref_nngp(tulpa::Fingerprint& sfp, int n_spatial, int nn, int cov_type,
              const Rcpp::NumericMatrix& coords,
              const Rcpp::IntegerMatrix& nn_idx,
              const Rcpp::IntegerVector& spatial_idx) {
    sfp.fold_pod(n_spatial);
    sfp.fold_pod(nn);
    sfp.fold_pod(cov_type);
    if (coords.size())      sfp.fold(coords.begin(),
                                     (std::size_t)coords.size() * sizeof(double));
    if (nn_idx.size())      sfp.fold(nn_idx.begin(),
                                     (std::size_t)nn_idx.size() * sizeof(int));
    if (spatial_idx.size()) sfp.fold(spatial_idx.begin(),
                                     (std::size_t)spatial_idx.size() * sizeof(int));
}

void ref_hsgp(tulpa::Fingerprint& sfp, int M,
              const Rcpp::NumericMatrix& phi_basis,
              const Rcpp::NumericVector& lambda_eig) {
    sfp.fold_pod(M);
    if (phi_basis.size())  sfp.fold(phi_basis.begin(),
                                    (std::size_t)phi_basis.size() * sizeof(double));
    if (lambda_eig.size()) sfp.fold(lambda_eig.begin(),
                                    (std::size_t)lambda_eig.size() * sizeof(double));
}

void ref_temporal(tulpa::Fingerprint& sfp, const std::string& temporal_type,
                  int n_times, bool cyclic,
                  const Rcpp::IntegerVector& temporal_idx,
                  bool with_groups, int n_groups) {
    sfp.fold_str(temporal_type);
    sfp.fold_pod(n_times);
    if (with_groups) sfp.fold_pod(n_groups);
    sfp.fold_pod(cyclic);
    if (temporal_idx.size()) sfp.fold(temporal_idx.begin(),
                                      (std::size_t)temporal_idx.size() * sizeof(int));
}

}  // namespace

// Seed for one field model under both routes. `kind` selects the group
// composition, matching the eleven entry points:
//   "areal", "areal_scaled", "nngp", "hsgp", "temporal",
//   "areal+temporal", "areal_scaled+temporal", "nngp+temporal", "hsgp+temporal"
// Returns c(reference = <hand-folded>, builder = <NlFieldIdentity>) as strings,
// since a uint64 does not survive a double.
// [[Rcpp::export]]
Rcpp::CharacterVector cpp_test_nl_field_seed(
    std::string kind, std::string tag,
    int n_spatial_units, double scale_factor,
    Rcpp::IntegerVector adj_row_ptr, Rcpp::IntegerVector adj_col_idx,
    int n_spatial, int nn, int cov_type,
    Rcpp::NumericMatrix coords, Rcpp::IntegerMatrix nn_idx,
    Rcpp::IntegerVector spatial_idx,
    int M, Rcpp::NumericMatrix phi_basis, Rcpp::NumericVector lambda_eig,
    std::string temporal_type, int n_times, bool cyclic,
    Rcpp::IntegerVector temporal_idx, int n_groups, bool with_groups
) {
    const bool has_areal    = kind.rfind("areal", 0) == 0;
    const bool areal_scaled = kind.rfind("areal_scaled", 0) == 0;
    const bool has_nngp     = kind.rfind("nngp", 0) == 0;
    const bool has_hsgp     = kind.rfind("hsgp", 0) == 0;
    const bool has_temporal = kind == "temporal" ||
                              kind.find("+temporal") != std::string::npos;
    const bool temporal_only = (kind == "temporal");

    tulpa::Fingerprint ref;
    ref.fold_str(tag);
    if (has_areal) {
        ref_areal(ref, n_spatial_units, adj_row_ptr, adj_col_idx,
                  areal_scaled, scale_factor);
    } else if (has_nngp) {
        ref_nngp(ref, n_spatial, nn, cov_type, coords, nn_idx, spatial_idx);
    } else if (has_hsgp) {
        ref_hsgp(ref, M, phi_basis, lambda_eig);
    }
    if (has_temporal) {
        ref_temporal(ref, temporal_type, n_times, cyclic, temporal_idx,
                     temporal_only && with_groups, n_groups);
    }

    tulpa::NlFieldIdentity id(tag.c_str());
    if (has_areal) {
        if (areal_scaled) {
            id.areal(n_spatial_units, adj_row_ptr, adj_col_idx, scale_factor);
        } else {
            id.areal(n_spatial_units, adj_row_ptr, adj_col_idx);
        }
    } else if (has_nngp) {
        id.nngp(n_spatial, nn, cov_type, coords, nn_idx, spatial_idx);
    } else if (has_hsgp) {
        id.hsgp(M, phi_basis, lambda_eig);
    }
    if (has_temporal) {
        if (temporal_only && with_groups) {
            id.temporal(temporal_type, n_times, cyclic, temporal_idx, n_groups);
        } else {
            id.temporal(temporal_type, n_times, cyclic, temporal_idx);
        }
    }

    return Rcpp::CharacterVector::create(
        Rcpp::Named("reference") = std::to_string(ref.value()),
        Rcpp::Named("builder")   = std::to_string(id.seed()));
}
