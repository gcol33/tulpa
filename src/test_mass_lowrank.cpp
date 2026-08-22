// test_mass_lowrank.cpp
// R-reachable probe for the diagonal-plus-low-rank metric (gcol33/tulpa#597).
//
// The metric is installed on a real DenseMassMatrix and driven through the
// same four entry points the sampler uses -- inv_mass_times_p, kinetic_energy,
// sample_momentum and the fused apply_drift -- so what the tests score is the
// object the chain runs under rather than a re-implementation of the Woodbury
// identity. The arbiter on the R side is a DENSE M = D + U Lambda U' built
// from the same groups, inverted with solve(): outside the assembly, not a
// second copy of it.

#include <Rcpp.h>

#include <cmath>
#include <random>
#include <vector>

#include "hmc_mass_drift.h"
#include "hmc_sampler_mass_blocks.h"

using tulpa_hmc::DenseMassMatrix;
using tulpa_hmc::LowRankMassTerm;

namespace {

// Build the metric: dimension n_full, inverse-mass diagonal `inv_mass_diag`,
// and one low-rank term over [start, start + n) with the supplied groups.
bool build_metric(
    DenseMassMatrix& mass,
    const Rcpp::NumericVector& inv_mass_diag,
    int start,
    const Rcpp::IntegerVector& group_ptr,
    const Rcpp::IntegerVector& group_idx,
    const Rcpp::NumericVector& lambda,
    const Rcpp::NumericVector& group_w
) {
  const int n_full = inv_mass_diag.size();
  mass.init(n_full, tulpa::MassMatrixType::DIAG);
  std::vector<double> inv_m(inv_mass_diag.begin(), inv_mass_diag.end());
  std::vector<double> sqrt_m(n_full);
  for (int i = 0; i < n_full; i++) sqrt_m[i] = 1.0 / std::sqrt(inv_m[i]);
  mass.set_diagonal(inv_m, sqrt_m);

  LowRankMassTerm term;
  term.start = start;
  // The block length is fixed by the diagonal the caller handed over, not
  // inferred from the groups: a group set that misses a coordinate still
  // describes a term over the whole block, with that coordinate carrying the
  // diagonal alone.
  term.n = n_full - start;
  term.group_ptr.assign(group_ptr.begin(), group_ptr.end());
  term.group_idx.assign(group_idx.begin(), group_idx.end());
  // Empty means unit weights, the indicator groups the term shipped with.
  term.group_w.assign(group_w.begin(), group_w.end());
  term.lambda.assign(lambda.begin(), lambda.end());
  term.var.assign(inv_m.begin() + start, inv_m.end());
  return mass.install_lowrank(std::move(term));
}

}  // namespace

// The metric applied to one momentum, four ways.
// [[Rcpp::export]]
Rcpp::List cpp_test_lowrank_mass_apply(
    Rcpp::NumericVector inv_mass_diag,
    int start,
    Rcpp::IntegerVector group_ptr,
    Rcpp::IntegerVector group_idx,
    Rcpp::NumericVector lambda,
    Rcpp::NumericVector p,
    double coeff = 0.37,
    Rcpp::NumericVector group_w = Rcpp::NumericVector::create()
) {
  DenseMassMatrix mass;
  const bool ok = build_metric(mass, inv_mass_diag, start, group_ptr,
                               group_idx, lambda, group_w);
  const int n = inv_mass_diag.size();
  if (p.size() != n) Rcpp::stop("p must have length(inv_mass_diag) entries");

  Rcpp::NumericVector inv_p(n, 0.0);
  Rcpp::NumericVector drift(n, 0.0);
  double ke = NA_REAL;
  if (ok) {
    std::vector<double> pv(p.begin(), p.end());
    std::vector<double> out(n, 0.0);
    mass.inv_mass_times_p(pv.data(), out.data());
    for (int i = 0; i < n; i++) inv_p[i] = out[i];
    ke = mass.kinetic_energy(pv.data());

    // apply_drift accumulates into q, so it is scored from a zero start.
    std::vector<double> q(n, 0.0), scratch(n, 0.0);
    tulpa_hmc::apply_drift(coeff, q.data(), pv.data(), mass, scratch.data(), n);
    for (int i = 0; i < n; i++) drift[i] = q[i];
  }

  return Rcpp::List::create(
      Rcpp::Named("ok") = ok,
      Rcpp::Named("rank") = ok ? mass.lowrank.front().rank() : 0,
      Rcpp::Named("inv_mass_times_p") = inv_p,
      Rcpp::Named("kinetic_energy") = ke,
      Rcpp::Named("drift") = drift,
      Rcpp::Named("coeff") = coeff);
}

// n_draws momentum draws from the installed metric, one per row.
// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_test_lowrank_mass_momentum(
    Rcpp::NumericVector inv_mass_diag,
    int start,
    Rcpp::IntegerVector group_ptr,
    Rcpp::IntegerVector group_idx,
    Rcpp::NumericVector lambda,
    int n_draws,
    int seed = 1,
    Rcpp::NumericVector group_w = Rcpp::NumericVector::create()
) {
  DenseMassMatrix mass;
  if (!build_metric(mass, inv_mass_diag, start, group_ptr, group_idx, lambda,
                    group_w)) {
    Rcpp::stop("the low-rank term was refused");
  }
  const int n = inv_mass_diag.size();
  std::mt19937 rng(static_cast<unsigned int>(seed));
  Rcpp::NumericMatrix out(n_draws, n);
  std::vector<double> p(n, 0.0);
  for (int s = 0; s < n_draws; s++) {
    mass.sample_momentum(p.data(), rng);
    for (int j = 0; j < n; j++) out(s, j) = p[j];
  }
  return out;
}

// The two-margin builder's own output: the groups make_margin_mass_term lays
// down for an S x T block, so a test can say the row and column sets are the
// ones the s * T + t layout implies rather than inferring them from a matvec.
// [[Rcpp::export]]
Rcpp::List cpp_test_margin_mass_term(
    int S, int T, double lambda_row, double lambda_col,
    Rcpp::NumericVector var, int start = 0, double lambda_trend = 0.0
) {
  std::vector<double> v(var.begin(), var.end());
  LowRankMassTerm term = tulpa_hmc::make_margin_mass_term(
      start, S, T, lambda_row, lambda_col, v.data(), (int)v.size(),
      lambda_trend);
  const bool ok = term.factorize();
  return Rcpp::List::create(
      Rcpp::Named("ok") = ok,
      Rcpp::Named("rank") = term.rank(),
      Rcpp::Named("start") = term.start,
      Rcpp::Named("n") = term.n,
      Rcpp::Named("group_ptr") = Rcpp::wrap(term.group_ptr),
      Rcpp::Named("group_idx") = Rcpp::wrap(term.group_idx),
      Rcpp::Named("group_w") = Rcpp::wrap(term.group_w),
      Rcpp::Named("lambda") = Rcpp::wrap(term.lambda));
}
