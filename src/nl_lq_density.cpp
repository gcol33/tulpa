// The within-cell log-quadratic read's per-row table: a row's density on the
// common grid `x`, integrated by the trapezoid rule into its normalized CDF.
// `.nl_lq_group()` (R/within_cell_log_quadratic.R) builds the row -- its nodes,
// the Newton-form quadratics through consecutive node triples, the box slabs and
// the extent its tails reach -- and this tabulates it, once per row.

#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>

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

inline Rcpp::List empty_cdf_curve() {
    return Rcpp::List::create(Rcpp::Named("index") = Rcpp::IntegerVector(),
                              Rcpp::Named("F") = Rcpp::NumericVector());
}

inline Rcpp::List compact_cdf_curve(const Rcpp::NumericVector& cf) {
    const int m = cf.size();
    if (!m || !std::isfinite(cf[m - 1])) return empty_cdf_curve();
    std::vector<int> keep;
    keep.push_back(0);
    for (int j = 1; j < m; ++j) if (cf[j] != cf[j - 1]) {
        if (keep.back() != j - 1) keep.push_back(j - 1);
        keep.push_back(j);
    }
    if (keep.back() != m - 1) keep.push_back(m - 1);
    Rcpp::IntegerVector index(keep.size());
    Rcpp::NumericVector F(keep.size());
    for (std::size_t k = 0; k < keep.size(); ++k) {
        index[k] = keep[k] + 1;
        F[k] = cf[keep[k]];
    }
    return Rcpp::List::create(Rcpp::Named("index") = index, Rcpp::Named("F") = F);
}

// A box density changes only where a slab starts or ends. Its trapezoid CDF
// is linear between those grid positions and their immediate neighbours.
inline Rcpp::List box_cdf_curve(const Rcpp::NumericVector& lg,
                                const Rcpp::NumericVector& lo_b,
                                const Rcpp::NumericVector& hi_b,
                                const Rcpp::NumericVector& x) {
    struct Event { int pos; long double height; int count; };
    const int m = x.size(), n = lg.size();
    double top = -std::numeric_limits<double>::infinity();
    for (double v : lg) if (v > top) top = v;
    if (!std::isfinite(top)) return empty_cdf_curve();
    std::vector<Event> events;
    events.push_back({0, 0.0L, 0});
    events.push_back({m, 0.0L, 0});
    std::vector<int> knots = {0, m - 1};
    for (int k = 0; k < n; ++k) {
        const int a = std::lower_bound(x.begin(), x.end(), lo_b[k]) - x.begin();
        const int b = std::lower_bound(x.begin(), x.end(), hi_b[k]) - x.begin();
        const double h = std::exp(lg[k] - top);
        if (a >= b || h == 0.0) continue;
        events.push_back({a, h, 1});
        events.push_back({b, -h, -1});
        for (int j : {a - 1, a, b - 1, b}) if (j >= 0 && j < m) knots.push_back(j);
    }
    std::sort(events.begin(), events.end(), [](const Event& a, const Event& b) {
        return a.pos < b.pos;
    });
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    std::size_t e = 0;
    int last = 0, active = 0;
    long double height = 0, before = 0, f0 = 0;
    Rcpp::NumericVector F(knots.size());
    for (std::size_t k = 0; k < knots.size(); ++k) {
        const int j = knots[k];
        while (e < events.size() && events[e].pos <= j) {
            const int pos = events[e].pos;
            before += height * (pos - last);
            last = pos;
            while (e < events.size() && events[e].pos == pos) {
                height += events[e].height;
                active += events[e].count;
                ++e;
            }
            if (active == 0) height = 0;
        }
        if (j == 0) f0 = height;
        const long double sum = before + height * (j - last + 1);
        F[k] = static_cast<double>(sum - (f0 + height) / 2);
    }
    const double total = F[F.size() - 1];
    if (!std::isfinite(total) || total <= 0) return empty_cdf_curve();
    Rcpp::IntegerVector index(knots.size());
    for (std::size_t k = 0; k < knots.size(); ++k) {
        index[k] = knots[k] + 1;
        F[k] /= total;
    }
    return Rcpp::List::create(Rcpp::Named("index") = index, Rcpp::Named("F") = F);
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List cpp_nl_lq_group_cdf(Rcpp::List piece, Rcpp::NumericVector x) {
    const bool is_lq = Rcpp::as<std::string>(piece["kind"]) == "lq";
    Rcpp::NumericVector u = piece["u"], lg = piece["lg"];
    Rcpp::NumericVector lo_b = piece["lo_b"], hi_b = piece["hi_b"];
    Rcpp::NumericVector d1 = is_lq ? Rcpp::as<Rcpp::NumericVector>(piece["d1"]) : Rcpp::NumericVector();
    Rcpp::NumericVector d2 = is_lq ? Rcpp::as<Rcpp::NumericVector>(piece["d2"]) : Rcpp::NumericVector();
    Rcpp::NumericVector lt = is_lq ? Rcpp::as<Rcpp::NumericVector>(piece["lt"]) : Rcpp::NumericVector();
    const double lo = Rcpp::as<double>(piece["lo"]), hi = Rcpp::as<double>(piece["hi"]);
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
    if (!is_lq) {
        return box_cdf_curve(lg, lo_b, hi_b, x);
    }
    std::vector<double> f(m, 0.0);
    {
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
        return empty_cdf_curve();
    }
    for (int j = 0; j < m; j++) cf[j] = cf[j] / total;
    return compact_cdf_curve(cf);
}

// [[Rcpp::export]]
Rcpp::List cpp_nl_lq_mixture_cdf(Rcpp::List pieces, Rcpp::NumericVector mass,
                                Rcpp::NumericVector x) {
    const int m = x.size(), n = pieces.size();
    if (m < 2 || mass.size() != n) {
        Rcpp::stop("A CDF mixture needs two grid points and one mass per row.");
    }
    std::vector<long double> delta(m, 0.0L), integral(m, 0.0L);
    Rcpp::LogicalVector usable(n, false);
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(mass[i]) || mass[i] < 0) {
            Rcpp::stop("CDF row masses must be finite and nonnegative.");
        }
        if (mass[i] == 0) continue;
        Rcpp::List curve = cpp_nl_lq_group_cdf(Rcpp::as<Rcpp::List>(pieces[i]), x);
        Rcpp::IntegerVector index = curve["index"];
        Rcpp::NumericVector F = curve["F"];
        if (F.size() < 2) continue;
        usable[i] = true;
        for (int j = 0; j + 1 < F.size(); ++j) {
            const int a = index[j] - 1, b = index[j + 1] - 1;
            if (a < 0 || b >= m || b <= a) {
                Rcpp::stop("CDF knots must increase within the interpolation grid.");
            }
            const long double slope = static_cast<long double>(mass[i]) *
                (F[j + 1] - F[j]) / (b - a);
            delta[a] += slope;
            delta[b] -= slope;
        }
    }
    long double slope = 0.0L;
    for (int j = 1; j < m; ++j) {
        slope += delta[j - 1];
        integral[j] = integral[j - 1] + std::max(0.0L, slope);
    }
    Rcpp::NumericVector F(m, NA_REAL);
    const long double total = integral[m - 1];
    if (std::isfinite(total) && total > 0) {
        for (int j = 0; j < m; ++j) F[j] = static_cast<double>(integral[j] / total);
    }
    return Rcpp::List::create(Rcpp::Named("F") = F, Rcpp::Named("usable") = usable);
}
