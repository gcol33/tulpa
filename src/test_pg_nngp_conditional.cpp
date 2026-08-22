// test_pg_nngp_conditional.cpp
// Test-only entry point: the NNGP prior's full conditional of one field value,
// alongside the geometry needed to rebuild the joint precision outside the
// engine.
//
// pg_nngp_field_conditional() is the moment pair the Polya-Gamma GP Gibbs sweep
// draws from, before the likelihood aggregates are added. Its correctness claim
// is a matrix identity -- the pair is row i of
// Lambda = (I - A)' D^-1 (I - A) -- and the sweep itself draws, so nothing in
// the fitted path can state that identity. This returns the pair for every
// location together with (B, F) and the ordering maps, so R can assemble Lambda
// densely and compare.

#include <Rcpp.h>
#include <vector>

#include "pg_shared.h"

// [[Rcpp::export]]
Rcpp::List cpp_test_pg_nngp_conditional(
    Rcpp::NumericMatrix coords,
    Rcpp::IntegerMatrix nn_idx,
    Rcpp::NumericMatrix nn_dist,
    Rcpp::IntegerVector nn_order,
    int n_spatial,
    int nn,
    Rcpp::NumericVector w,
    double sigma2,
    double phi,
    int cov_type = 0
) {
    if (static_cast<int>(w.size()) != n_spatial) {
        Rcpp::stop("length(w) (%d) must equal n_spatial (%d).",
                   static_cast<int>(w.size()), n_spatial);
    }

    tulpa::PgNngpTopology top =
        tulpa::pg_nngp_topology(nn_idx, nn_dist, nn_order, n_spatial, nn);

    tulpa::PgNngpFactors fac;
    tulpa::pg_nngp_factors(phi, cov_type, coords, nn_dist, top, fac);

    std::vector<double> wv(w.begin(), w.end());

    // Indexed by ORDERED position, matching B / F / the topology maps. The
    // sweep reads position i and writes original location top.orig[i].
    Rcpp::NumericVector prec(n_spatial), mean_num(n_spatial);
    for (int i = 0; i < n_spatial; i++) {
        const tulpa::PgNngpCond c =
            tulpa::pg_nngp_field_conditional(top, fac, wv, sigma2, i);
        prec[i]     = c.prec;
        mean_num[i] = c.mean_num;
    }

    Rcpp::IntegerVector orig(n_spatial), cnt(n_spatial);
    for (int i = 0; i < n_spatial; i++) {
        orig[i] = top.orig[i];
        cnt[i]  = top.cnt[i];
    }
    Rcpp::IntegerVector parent_pos(top.parent_pos.begin(), top.parent_pos.end());
    Rcpp::IntegerVector parent_orig(top.parent_orig.begin(),
                                    top.parent_orig.end());
    Rcpp::NumericVector B(fac.B.begin(), fac.B.end());
    Rcpp::NumericVector F(fac.F.begin(), fac.F.end());

    return Rcpp::List::create(
        Rcpp::_["prec"]        = prec,
        Rcpp::_["mean_num"]    = mean_num,
        Rcpp::_["orig"]        = orig,
        Rcpp::_["cnt"]         = cnt,
        Rcpp::_["parent_pos"]  = parent_pos,
        Rcpp::_["parent_orig"] = parent_orig,
        Rcpp::_["B"]           = B,
        Rcpp::_["F"]           = F,
        Rcpp::_["nn"]          = nn
    );
}
