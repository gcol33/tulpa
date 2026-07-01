// psis_kernel.cpp
// Deterministic core of tulpa_psis(): the generalized-Pareto tail fit
// (Zhang & Stephens 2009, empirical-Bayes profile with the Vehtari et al. 2024
// weakly-informative shape prior) plus the Pareto smoothing of the upper-tail
// log weights. The R reference is tulpa_psis() / .tulpa_gpd_fit() / .tulpa_qgpd()
// in R/psis.R (retained as the oracle in test-psis.R); this port mirrors them so
// the two agree to libm rounding. The bootstrap k-uncertainty stays in R (it
// resamples with R's RNG, so the resamples remain reproducible); each refit just
// calls this kernel.

#include <Rcpp.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace Rcpp;

namespace {

double logsumexp(const std::vector<double>& x) {
  double m = -std::numeric_limits<double>::infinity();
  for (double v : x) if (v > m) m = v;
  if (!std::isfinite(m)) return m;
  double s = 0.0;
  for (double v : x) s += std::exp(v - m);
  return m + std::log(s);
}

// Quantile function of the GPD (location 0), matching .tulpa_qgpd.
double qgpd_one(double p, double k, double sigma) {
  if (std::isnan(sigma) || sigma <= 0.0)
    return std::numeric_limits<double>::quiet_NaN();
  if (std::fabs(k) < 1e-30) return sigma * (-std::log1p(-p));
  return sigma * std::expm1(-k * std::log1p(-p)) / k;
}

// Zhang-Stephens GPD fit on positive exceedances, matching .tulpa_gpd_fit
// (wip = TRUE, min_grid_pts = 30). `x` is copied and sorted internally.
void gpd_fit(std::vector<double> x, double& k_out, double& sigma_out) {
  std::sort(x.begin(), x.end());
  const int N = (int) x.size();
  const double prior = 3.0;
  const int M = 30 + (int) std::floor(std::sqrt((double) N));
  const double xstar = x[(int) std::floor(N / 4.0 + 0.5) - 1];  // 25% quantile
  const double xN = x[N - 1];

  std::vector<double> l_theta(M), theta(M);
  for (int j = 1; j <= M; ++j) {
    double th = 1.0 / xN +
                (1.0 - std::sqrt((double) M / ((double) j - 0.5))) / (prior * xstar);
    theta[j - 1] = th;
    double kt = 0.0;
    for (double xi : x) kt += std::log1p(-th * xi);
    kt /= N;
    l_theta[j - 1] = N * (std::log(-th / kt) - kt - 1.0);
  }
  double lse = logsumexp(l_theta);
  double theta_hat = 0.0;
  for (int j = 0; j < M; ++j) theta_hat += theta[j] * std::exp(l_theta[j] - lse);

  double k = 0.0;
  for (double xi : x) k += std::log1p(-theta_hat * xi);
  k /= N;
  sigma_out = -k / theta_hat;
  k = (k * N + 0.5 * 10.0) / (N + 10.0);              // shrink toward 1/2
  if (std::isnan(k)) k = std::numeric_limits<double>::infinity();
  k_out = k;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_psis(Rcpp::NumericVector log_ratios, int tail_len) {
  const int S = log_ratios.size();               // R filtered finite, S >= 5
  std::vector<double> lw(S);
  double mx = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < S; ++i) if (log_ratios[i] > mx) mx = log_ratios[i];
  for (int i = 0; i < S; ++i) lw[i] = log_ratios[i] - mx;

  double k_hat = NA_REAL;
  if (tail_len >= 5 && S >= 25) {
    std::vector<int> ord(S);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b) { return lw[a] < lw[b]; });  // ascending
    const int cut0 = S - tail_len;                 // first tail position
    const double cutoff = lw[ord[cut0 - 1]];
    const double exp_cut = std::exp(cutoff);

    std::vector<double> exceed(tail_len);
    int npos = 0;
    for (int j = 0; j < tail_len; ++j) {
      exceed[j] = std::exp(lw[ord[cut0 + j]]) - exp_cut;
      if (exceed[j] > 0.0) ++npos;
    }
    if (npos >= 5) {
      double kf, sf;
      gpd_fit(exceed, kf, sf);
      k_hat = kf;
      double lwmax = *std::max_element(lw.begin(), lw.end());  // = 0 (shifted)
      for (int j = 0; j < tail_len; ++j) {
        double pp = ((double) (j + 1) - 0.5) / tail_len;
        double sm = std::log(exp_cut + qgpd_one(pp, kf, sf));
        if (sm > lwmax) sm = lwmax;
        lw[ord[cut0 + j]] = sm;
      }
    }
  }

  double ls = logsumexp(lw);
  double sw2 = 0.0;
  Rcpp::NumericVector log_weights(S);
  for (int i = 0; i < S; ++i) {
    lw[i] -= ls;
    log_weights[i] = lw[i];
    double w = std::exp(lw[i]);
    sw2 += w * w;
  }
  return Rcpp::List::create(
    Rcpp::Named("pareto_k")    = k_hat,
    Rcpp::Named("is_ess")      = 1.0 / sw2,
    Rcpp::Named("log_weights") = log_weights,
    Rcpp::Named("tail_len")    = tail_len);
}
