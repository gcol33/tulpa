// The within-cell log-quadratic read's per-row table: a row's density on the
// common grid `x`, integrated by the trapezoid rule into its normalized CDF.
// `.nl_lq_group()` (R/within_cell_log_quadratic.R) builds the row -- its nodes,
// the Newton-form quadratics through consecutive node triples, the box slabs and
// the extent its tails reach -- and this tabulates it, once per row.

#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>

namespace {

// findInterval(x, v): the number of v[i] <= x, with v ascending. Under
// `rightmost_closed`, x == v[n - 1] counts n - 1, so the last node closes the
// last interval rather than opening one past it.
inline int find_interval(const double* v, int n, double x, bool rightmost_closed) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (v[mid] <= x) lo = mid + 1; else hi = mid;
    }
    if (rightmost_closed && n > 0 && lo == n && x == v[n - 1]) return n - 1;
    return lo;
}

// The quadratic through nodes (i, i + 1, i + 2), 0-based i, in Newton form:
// lg_i + d1_i (x - u_i) + d2_i (x - u_i) (x - u_{i+1}).
inline double lq_quad(const double* lg, const double* d1, const double* d2,
                      const double* u, int i, double x) {
    return lg[i] + d1[i] * (x - u[i]) + d2[i] * (x - u[i]) * (x - u[i + 1]);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::NumericVector cpp_nl_lq_group_cdf(bool is_lq,
                                        Rcpp::NumericVector u,
                                        Rcpp::NumericVector lg,
                                        Rcpp::NumericVector d1,
                                        Rcpp::NumericVector d2,
                                        Rcpp::NumericVector lt,
                                        Rcpp::NumericVector lo_b,
                                        Rcpp::NumericVector hi_b,
                                        double lo, double hi,
                                        Rcpp::NumericVector x) {
    const int m = x.size();
    const int n = u.size();
    if (lg.size() != n || lo_b.size() != n || hi_b.size() != n) {
        Rcpp::stop("cpp_nl_lq_group_cdf: u, lg, lo_b and hi_b must share a length.");
    }
    if (is_lq && (n < 3 || d1.size() != n - 2 || d2.size() != n - 2 ||
                  lt.size() != n)) {
        Rcpp::stop("cpp_nl_lq_group_cdf: a log-quadratic row needs n >= 3 nodes, "
                   "n - 2 quadratics and n slabs.");
    }
    if (m < 2) Rcpp::stop("cpp_nl_lq_group_cdf: the grid needs two points.");

    const double NEG_INF = -std::numeric_limits<double>::infinity();
    std::vector<double> f(m, 0.0);

    if (!is_lq) {
        // A box row: each cell's box holds its own mass, flat across the box.
        double top = NEG_INF;
        for (int k = 0; k < n; k++) if (lg[k] > top) top = lg[k];
        for (int k = 0; k < n; k++) {
            const double h = std::exp(lg[k] - top);
            for (int j = 0; j < m; j++) {
                if (x[j] >= lo_b[k] && x[j] < hi_b[k]) f[j] += h;
            }
        }
    } else {
        // The interpolated log density plus the slab of the box x falls in:
        // between nodes the mean of the two quadratics covering the interval,
        // past the outer nodes (out to the extent) the outer quadratic where it
        // is concave and the outer node's level where it is not.
        std::vector<double> L(m, NEG_INF);
        double top = NEG_INF;
        for (int j = 0; j < m; j++) {
            const double xj = x[j];
            double lgx = NEG_INF;
            const int seg = find_interval(u.begin(), n, xj, true);
            if (seg >= 1 && seg <= n - 1) {
                const int left = (seg >= 2) ? seg - 2 : seg - 1;
                const int right = (seg <= n - 2) ? seg - 1 : seg - 2;
                lgx = 0.5 * (lq_quad(lg.begin(), d1.begin(), d2.begin(), u.begin(), left, xj) +
                             lq_quad(lg.begin(), d1.begin(), d2.begin(), u.begin(), right, xj));
            }
            if (xj < u[0] && xj >= lo) {
                lgx = (d2[0] < 0)
                    ? lq_quad(lg.begin(), d1.begin(), d2.begin(), u.begin(), 0, xj)
                    : lg[0];
            }
            if (xj > u[n - 1] && xj <= hi) {
                lgx = (d2[n - 3] < 0)
                    ? lq_quad(lg.begin(), d1.begin(), d2.begin(), u.begin(), n - 3, xj)
                    : lg[n - 1];
            }
            int kb = find_interval(lo_b.begin(), n, xj, false);
            if (kb < 1) kb = 1;
            const bool inside = xj < hi_b[kb - 1] ||
                                (kb == n && xj >= hi_b[n - 1]) || xj < lo_b[0];
            const double Lj = lgx + (inside ? lt[kb - 1] : NEG_INF);
            L[j] = Lj;
            if (std::isfinite(Lj) && Lj > top) top = Lj;
        }
        if (std::isfinite(top)) {
            for (int j = 0; j < m; j++) {
                if (std::isfinite(L[j])) f[j] = std::exp(L[j] - top);
            }
        }
    }

    // Trapezoid CDF on the uniform grid, accumulated in long double as R's
    // cumsum() does, then normalized; NA throughout when the row carries no
    // finite positive mass on the grid.
    const double dx = x[1] - x[0];
    Rcpp::NumericVector cf(m);
    cf[0] = 0.0;
    long double acc = 0.0L;
    for (int j = 1; j < m; j++) {
        acc += (f[j] + f[j - 1]) / 2;
        cf[j] = static_cast<double>(acc) * dx;
    }
    const double total = cf[m - 1];
    if (!std::isfinite(total) || total <= 0) {
        return Rcpp::NumericVector(m, NA_REAL);
    }
    for (int j = 0; j < m; j++) cf[j] = cf[j] / total;
    return cf;
}
