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
#ifdef _OPENMP
#include <omp.h>
#endif

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

// Normalized smoothed PSIS log-weights for a finite log-ratio vector (S >= 5).
// Fills `lw` (length S) and `k_hat`. Deterministic -- no RNG. Shared by
// cpp_tulpa_psis and the LOO-PIT weighting.
inline void psis_logweights(const double* lr, int S, int tail_len,
                            std::vector<double>& lw, double& k_hat) {
  lw.assign(S, 0.0);
  double mx = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < S; ++i) if (lr[i] > mx) mx = lr[i];
  for (int i = 0; i < S; ++i) lw[i] = lr[i] - mx;
  k_hat = NA_REAL;
  if (tail_len >= 5 && S >= 25) {
    std::vector<int> ord(S); std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b) { return lw[a] < lw[b]; });
    const int cut0 = S - tail_len;
    const double cutoff = lw[ord[cut0 - 1]]; const double exp_cut = std::exp(cutoff);
    std::vector<double> exceed(tail_len); int npos = 0;
    for (int j = 0; j < tail_len; ++j) {
      exceed[j] = std::exp(lw[ord[cut0 + j]]) - exp_cut;
      if (exceed[j] > 0.0) ++npos;
    }
    if (npos >= 5) {
      double kf, sf; gpd_fit(exceed, kf, sf); k_hat = kf;
      double lwmax = *std::max_element(lw.begin(), lw.end());
      for (int j = 0; j < tail_len; ++j) {
        double pp = ((double) (j + 1) - 0.5) / tail_len;
        double sm = std::log(exp_cut + qgpd_one(pp, kf, sf));
        if (sm > lwmax) sm = lwmax;
        lw[ord[cut0 + j]] = sm;
      }
    }
  }
  double ls = logsumexp(lw);
  for (int i = 0; i < S; ++i) lw[i] -= ls;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_tulpa_psis(Rcpp::NumericVector log_ratios, int tail_len) {
  const int S = log_ratios.size();               // R filtered finite, S >= 5
  std::vector<double> lw; double k_hat;
  psis_logweights(log_ratios.begin(), S, tail_len, lw, k_hat);
  double sw2 = 0.0;
  Rcpp::NumericVector log_weights(S);
  for (int i = 0; i < S; ++i) {
    log_weights[i] = lw[i];
    double w = std::exp(lw[i]); sw2 += w * w;
  }
  return Rcpp::List::create(
    Rcpp::Named("pareto_k")    = k_hat,
    Rcpp::Named("is_ess")      = 1.0 / sw2,
    Rcpp::Named("log_weights") = log_weights,
    Rcpp::Named("tail_len")    = tail_len);
}

// Leave-one-out randomized PIT from a pointwise log-likelihood `ll` [S x N] and
// per-draw predictive-CDF limits `Fl` / `Fu` [S x N]. Per observation the PSIS
// leave-one-out weights (from -ll[, i]) reweight the CDF limits; a column with
// any non-finite ratio falls back to equal weights (matching the R driver). The
// PSIS is deterministic, so the columns parallelise; the single uniform jitter
// per observation is the only RNG and is drawn serially at the end in index
// order (matching runif(N) in .tobs_loo_pit_from_limits). Byte-identical.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_psis_loo_pit(Rcpp::NumericMatrix ll, Rcpp::NumericMatrix Fl,
                                     Rcpp::NumericMatrix Fu, int tail_len,
                                     int n_threads) {
  const int S = ll.nrow(), N = ll.ncol();
  std::vector<double> fl_loo(N), fu_loo(N);
  const double* pll = ll.begin(); const double* pfl = Fl.begin();
  const double* pfu = Fu.begin();
#ifdef _OPENMP
  #pragma omp parallel for schedule(static) num_threads(n_threads > 0 ? n_threads : 1)
#endif
  for (int i = 0; i < N; ++i) {
    std::vector<double> lr(S); int nfin = 0;
    for (int s = 0; s < S; ++s) {
      lr[s] = -pll[(std::size_t) i * S + s];
      if (std::isfinite(lr[s])) ++nfin;
    }
    double a = 0.0, b = 0.0;
    if (nfin == S) {
      std::vector<double> lw; double kh;
      psis_logweights(lr.data(), S, tail_len, lw, kh);
      for (int s = 0; s < S; ++s) {
        double w = std::exp(lw[s]);
        a += w * pfl[(std::size_t) i * S + s];
        b += w * pfu[(std::size_t) i * S + s];
      }
    } else {
      double w = 1.0 / S;
      for (int s = 0; s < S; ++s) {
        a += w * pfl[(std::size_t) i * S + s];
        b += w * pfu[(std::size_t) i * S + s];
      }
    }
    fl_loo[i] = a; fu_loo[i] = b;
  }
  Rcpp::RNGScope scope;
  Rcpp::NumericVector pit(N);
  for (int i = 0; i < N; ++i) {
    double u = R::unif_rand();
    double v = fl_loo[i] + u * (fu_loo[i] - fl_loo[i]);
    pit[i] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
  }
  return pit;
}

// Randomized PIT from posterior predictive-CDF limits (tulpa_pit). `cdf_upper` /
// `cdf_lower` are [S x N] draw matrices column-averaged to the per-observation
// limits; with a lower limit the PIT is jittered uniformly across [Fl, Fu],
// otherwise a tiny jitter breaks ties. The uniforms are drawn in index order,
// matching runif(N) in the R body. Byte-identical.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_tulpa_pit(Rcpp::NumericMatrix cdf_upper,
                                  Rcpp::NumericMatrix cdf_lower,
                                  bool has_lower, bool jitter) {
  const int S = cdf_upper.nrow(), N = cdf_upper.ncol();
  std::vector<double> Fu(N);
  for (int j = 0; j < N; ++j) {
    double a = 0.0; for (int s = 0; s < S; ++s) a += cdf_upper(s, j);
    Fu[j] = a / S;
  }
  Rcpp::RNGScope scope;
  Rcpp::NumericVector pit(N);
  if (has_lower) {
    std::vector<double> Fl(N);
    for (int j = 0; j < N; ++j) {
      double a = 0.0; for (int s = 0; s < S; ++s) a += cdf_lower(s, j);
      Fl[j] = a / S;
    }
    for (int j = 0; j < N; ++j) {
      double u = R::unif_rand();
      double v = Fl[j] + u * (Fu[j] - Fl[j]);
      pit[j] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
  } else {
    for (int j = 0; j < N; ++j) {
      double v = Fu[j];
      if (jitter) v += R::unif_rand() * 1e-6;
      pit[j] = v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
    }
  }
  return pit;
}
