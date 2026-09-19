// brasil.h
// BRASIL: Best Rational Approximation by Successive Interval Length adjustment
// (Hofreither 2021, doi:10.1007/s11075-020-01042-0). Degree-(m, m) best uniform
// (minimax) rational approximation of a scalar function on an interval, in
// barycentric form, with its zeros and poles extracted.
//
// This is the compiled counterpart of the reference implementation in
// R/brasil.R, which stays as the oracle the port must reproduce
// (test-brasil-cpp-oracle.R pins them together). The fractional-SPDE fitter
// calls this one: every (range, sigma) cell the outer integrator visits needs
// its own rational approximation on that cell's spectrum interval, so the
// search runs once per cell -- 1102 times for a single n = 40 fit -- and at
// R speed that is ~0.55 s apiece, which is the whole cost of a fractional fit
// (gcol33/tulpa#818).
//
// Differences from the R oracle are confined to the two linear-algebra
// primitives R outsources to LAPACK: the Loewner nullspace comes from a Jacobi
// SVD rather than dgesdd, and the secular-polynomial roots from a companion
// matrix rather than polyroot's Jenkins-Traub. Both are defined up to the same
// invariances the consumers already have (the barycentric weights up to scale,
// the root set up to order), so the approximation they define is the same one.

#ifndef TULPA_BRASIL_H
#define TULPA_BRASIL_H

#include <RcppEigen.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tulpa {
namespace brasil {

using Eigen::MatrixXd;
using Eigen::VectorXd;

// A barycentric rational: r(x) = sum_j w_j f_j / (x - z_j) over the same sum
// without f, interpolating (nodes, values) exactly at the nodes.
struct BaryRat {
  VectorXd nodes;
  VectorXd values;
  VectorXd weights;
};

// Chebyshev nodes of the first kind, n points in [a, b].
inline VectorXd cheb_nodes(int n, double a, double b) {
  VectorXd out(n);
  for (int i = 0; i < n; ++i) {
    const double t = 1.0 - std::cos((2.0 * (i + 1) - 1.0) / (2.0 * n) * M_PI);
    out[i] = t * ((b - a) / 2.0) + a;
  }
  return out;
}

// Coefficients (ascending powers) of prod(x - roots).
inline VectorXd poly_from_roots(const VectorXd& roots) {
  VectorXd coef(1);
  coef[0] = 1.0;
  for (int r = 0; r < roots.size(); ++r) {
    const double root = roots[r];
    VectorXd next(coef.size() + 1);
    next.setZero();
    // c(0, coef) - c(root * coef, 0)
    for (int i = 0; i < coef.size(); ++i) next[i + 1] += coef[i];
    for (int i = 0; i < coef.size(); ++i) next[i] -= root * coef[i];
    coef = next;
  }
  return coef;
}

// Barycentric rational interpolant of (nodes, values) with odd length 2m+1;
// numerator and denominator both have degree <= m. Knockaert (2008) Loewner
// construction: the weights are a nullspace vector of the Loewner matrix.
inline BaryRat bary_interp_rat(const VectorXd& nodes, const VectorXd& values) {
  const int N = static_cast<int>(values.size());
  const int na = (N + 1) / 2;   // 1st, 3rd, ... (the "a" set)
  const int nb = N / 2;         // 2nd, 4th, ... (the "b" set)
  VectorXd xa(na), va(na), xb(nb), vb(nb);
  for (int i = 0; i < na; ++i) { xa[i] = nodes[2 * i];     va[i] = values[2 * i]; }
  for (int i = 0; i < nb; ++i) { xb[i] = nodes[2 * i + 1]; vb[i] = values[2 * i + 1]; }

  MatrixXd B(nb, na);
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < na; ++j) {
      B(i, j) = (vb[i] - va[j]) / (xb[i] - xa[j]);
    }
  }

  // Nullspace vector = right singular vector of the smallest singular value.
  Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeFullV);
  BaryRat br;
  br.nodes = xa;
  br.values = va;
  br.weights = svd.matrixV().col(na - 1);
  return br;
}

// Evaluate a barycentric rational at one point, exact at the nodes.
inline double bary_eval1(const BaryRat& br, double x) {
  const int n = static_cast<int>(br.nodes.size());
  double num = 0.0, den = 0.0;
  for (int j = 0; j < n; ++j) {
    const double d = x - br.nodes[j];
    if (d == 0.0) return br.values[j];
    const double c = br.weights[j] / d;
    num += c * br.values[j];
    den += c;
  }
  return num / den;
}

// Real roots of the cleared secular polynomial
// q(x) = sum_j w_j prod_{k != j} (x - z_k), via a companion-matrix eigenproblem.
inline std::vector<double> bary_secular_roots(const VectorXd& z, const VectorXd& w) {
  const int N = static_cast<int>(z.size());
  VectorXd acc = VectorXd::Zero(N);   // degree N-1 polynomial, ascending
  for (int j = 0; j < N; ++j) {
    VectorXd zj(N - 1);
    int k = 0;
    for (int i = 0; i < N; ++i) if (i != j) zj[k++] = z[i];
    const VectorXd pj = poly_from_roots(zj);
    acc.head(pj.size()) += w[j] * pj;
  }

  // Drop leading (highest-power) zeros so the companion sees the true degree.
  int len = N;
  while (len > 1 && acc[len - 1] == 0.0) --len;
  std::vector<double> out;
  if (len <= 1) return out;

  const int deg = len - 1;
  MatrixXd C = MatrixXd::Zero(deg, deg);
  const double lead = acc[deg];
  for (int i = 0; i < deg; ++i) C(i, deg - 1) = -acc[i] / lead;
  for (int i = 1; i < deg; ++i) C(i, i - 1) = 1.0;

  Eigen::EigenSolver<MatrixXd> es(C, /*computeEigenvectors=*/false);
  const Eigen::VectorXcd ev = es.eigenvalues();
  for (int i = 0; i < ev.size(); ++i) {
    const double re = ev[i].real(), im = ev[i].imag();
    if (std::abs(im) < 1e-6 * (1.0 + std::abs(re))) out.push_back(re);
  }
  return out;
}

inline std::vector<double> bary_poles(const BaryRat& br) {
  return bary_secular_roots(br.nodes, br.weights);
}

inline std::vector<double> bary_zeros(const BaryRat& br) {
  return bary_secular_roots(br.nodes, br.weights.cwiseProduct(br.values));
}

// Per-subinterval local maxima of g via golden-section search on each
// subinterval of `nodes` (which carries the two interval endpoints).
template <typename Fn>
inline void local_maxima_golden(Fn&& g, const VectorXd& nodes, int num_iter,
                                VectorXd& out_x, VectorXd& out_v) {
  const double gm = (3.0 - std::sqrt(5.0)) / 2.0;
  const int n_int = static_cast<int>(nodes.size()) - 1;
  out_x.resize(n_int);
  out_v.resize(n_int);
  for (int i = 0; i < n_int; ++i) {
    double z0 = nodes[i], z1 = nodes[i] + (nodes[i + 1] - nodes[i]) * gm,
           z2 = nodes[i + 1];
    double gb = g(z1);
    for (int k = 0; k < num_iter; ++k) {
      const double mid = (z0 + z2) / 2.0;
      const double far = (z1 <= mid) ? z2 : z0;
      const double xx = z1 + gm * (far - z1);
      const double gx = g(xx);
      if (gx > gb) {
        if (xx > z1) z0 = z1; else z2 = z1;
        z1 = xx; gb = gx;
      } else {
        if (xx < z1) z0 = xx; else z2 = xx;
      }
    }
    out_x[i] = z1;
    out_v[i] = gb;
  }
}

struct BrasilResult {
  BaryRat br;
  bool converged = false;
  double error = std::numeric_limits<double>::quiet_NaN();
  double deviation = std::numeric_limits<double>::quiet_NaN();
  int iterations = 0;
};

// Best degree-(m, m) rational approximation of f on [a, b].
template <typename Fn>
inline BrasilResult brasil(Fn&& f, double a, double b, int m,
                           double tol = 1e-4, int maxiter = 1000,
                           double max_step_size = 0.1, double step_factor = 0.1,
                           int num_iter = 30, int init_steps = 100) {
  BrasilResult res;
  const int nn = 2 * m + 1;
  VectorXd nodes = cheb_nodes(nn, a, b);
  VectorXd ext(nn + 2), vals(nn), lmx, lmv;

  for (int k = 1; k <= init_steps + maxiter; ++k) {
    for (int i = 0; i < nn; ++i) vals[i] = f(nodes[i]);
    res.br = bary_interp_rat(nodes, vals);
    const BaryRat& br = res.br;
    auto errfun = [&](double x) { return std::abs(f(x) - bary_eval1(br, x)); };

    ext[0] = a;
    ext.segment(1, nn) = nodes;
    ext[nn + 1] = b;
    local_maxima_golden(errfun, ext, num_iter, lmx, lmv);

    const double max_err = lmv.maxCoeff(), min_err = lmv.minCoeff();
    res.error = max_err;
    res.deviation = max_err / min_err - 1.0;
    res.converged = (res.deviation <= tol);
    res.iterations = k;
    if (res.converged || k == init_steps + maxiter) break;

    if (k <= init_steps) {
      // Phase 1: move the node nearest the lowest-error interval onto the
      // highest-error point.
      int max_i = 0;
      for (int i = 1; i < lmv.size(); ++i) if (lmv[i] > lmv[max_i]) max_i = i;
      double max_x = lmx[max_i];
      if (max_x == a)      max_x = (3.0 * a + nodes[0]) / 4.0;
      else if (max_x == b) max_x = (nodes[nn - 1] + 3.0 * b) / 4.0;

      int min_k = 0;
      for (int i = 1; i < lmv.size(); ++i) if (lmv[i] < lmv[min_k]) min_k = i;
      int min_j;
      if (min_k == 0) {
        min_j = 0;
      } else if (min_k == static_cast<int>(lmv.size()) - 1) {
        min_j = nn - 1;
      } else {
        min_j = (std::abs(max_x - nodes[min_k - 1]) < std::abs(max_x - nodes[min_k]))
                  ? min_k : min_k - 1;
      }
      nodes[min_j] = max_x;
      std::sort(nodes.data(), nodes.data() + nn);
    } else {
      // Phase 2: scale subinterval lengths by normalized local-error deviation.
      const int nl = static_cast<int>(lmv.size());   // == nn + 1
      VectorXd lens(nl);
      for (int i = 0; i < nl; ++i) lens[i] = ext[i + 1] - ext[i];
      const double mean_err = lmv.mean();
      double max_dev = 0.0;
      for (int i = 0; i < nl; ++i) max_dev = std::max(max_dev, std::abs(lmv[i] - mean_err));
      const double stepsize = std::min(max_step_size, step_factor * max_dev / mean_err);
      for (int i = 0; i < nl; ++i) {
        const double ndev = (lmv[i] - mean_err) / max_dev;
        lens[i] *= std::pow(1.0 - stepsize, ndev);
      }
      lens *= (b - a) / lens.sum();
      double cum = a;
      for (int i = 0; i < nn; ++i) { cum += lens[i]; nodes[i] = cum; }
    }
  }
  return res;
}

// Rational-SPDE roots: the zeros / poles of the best (order, order) rational
// approximation of x^{-beta_rem} on [spectrum_ratio, 1], mapped to the rSPDE
// operator factors (rb from the poles, rc from the zeros) with the scale
// constant that matches the approximation at the interval midpoint. The R
// oracle is .spde_rational_roots().
struct RationalRoots {
  std::vector<double> rb;
  std::vector<double> rc;
  double scale = 1.0;
  int m_beta = 1;
  double beta_rem = 1.0;
  double error = std::numeric_limits<double>::quiet_NaN();
  double deviation = std::numeric_limits<double>::quiet_NaN();
  bool converged = false;
  int iterations = 0;
};

inline RationalRoots rational_roots(int order, double beta, double spectrum_ratio,
                                    double tol = 1e-7) {
  RationalRoots out;
  out.m_beta = std::max(1, static_cast<int>(std::floor(beta)));
  out.beta_rem = beta - (out.m_beta - 1);
  const double br_exp = out.beta_rem;

  BrasilResult res = brasil([br_exp](double x) { return std::pow(x, -br_exp); },
                            spectrum_ratio, 1.0, order, tol);

  const std::vector<double> zr = bary_zeros(res.br);
  const std::vector<double> pr = bary_poles(res.br);
  const double xm = (spectrum_ratio + 1.0) / 2.0;

  double num = 1.0, den = 1.0;
  for (double p : pr) num *= (xm - p);
  for (double z : zr) den *= (xm - z);
  out.scale = bary_eval1(res.br, xm) * num / den;

  out.rb.reserve(pr.size());
  for (double p : pr) out.rb.push_back(1.0 / p);
  out.rc.reserve(zr.size());
  for (double z : zr) out.rc.push_back(1.0 / z);

  out.error = res.error;
  out.deviation = res.deviation;
  out.converged = res.converged;
  out.iterations = res.iterations;
  return out;
}

}  // namespace brasil
}  // namespace tulpa

#endif  // TULPA_BRASIL_H
