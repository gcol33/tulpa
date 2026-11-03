// linear_constraint_rows.h
// A linear constraint on the latent vector as a sparse row: the constraint is
// sum_j w[j] * x[cols[j]] = 0, the indices 0-based. A sum-to-zero group is the
// row whose weights are all 1; a weighted level (a BYM2 field scaled per
// connected component) or a Knorr-Held null-space direction is any other row.
// One representation for every reader -- the inner Newton's constrained step,
// the kriging correction of a covariance read, the Woodbury fold of a soft
// pin -- so a constraint is stated once, where the block that carries it is
// built.

#ifndef TULPA_LINEAR_CONSTRAINT_ROWS_H
#define TULPA_LINEAR_CONSTRAINT_ROWS_H

#include <Rcpp.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tulpa {

struct LinearConstraintRow {
    std::vector<int>    cols;
    std::vector<double> w;    // empty: every weight is 1

    double weight(std::size_t e) const { return w.empty() ? 1.0 : w[e]; }
    std::size_t size() const { return cols.size(); }
};

// The indicator row of a group of latent indices. An index repeated within the
// group contributes once (the row is the indicator of the SET), kept at its
// first position so the summation order is the caller's.
inline LinearConstraintRow indicator_row(const std::vector<int>& group, int n_x) {
    LinearConstraintRow r;
    std::vector<char> seen(n_x > 0 ? n_x : 1, 0);
    r.cols.reserve(group.size());
    for (int latent : group) {
        if (latent < 0 || latent >= n_x || seen[latent]) continue;
        seen[latent] = 1;
        r.cols.push_back(latent);
    }
    return r;
}

inline std::vector<LinearConstraintRow> indicator_rows(
    const std::vector<std::vector<int>>& groups, int n_x
) {
    std::vector<LinearConstraintRow> out;
    out.reserve(groups.size());
    for (const auto& g : groups) out.push_back(indicator_row(g, n_x));
    return out;
}

// a' v for the row a.
inline double constraint_row_dot(const LinearConstraintRow& a, const double* v) {
    double s = 0.0;
    for (std::size_t e = 0; e < a.cols.size(); e++) s += a.weight(e) * v[a.cols[e]];
    return s;
}

// Read a list of constraint rows handed over from R. Element g is either an
// integer vector of 1-based latent indices (a sum-to-zero group, every weight
// 1) or a list(cols = <1-based indices>, w = <weights>). `n_x` > 0 bounds the
// indices; an index outside [1, n_x] is refused by name, since every reader
// writes it into an n_x-long solve buffer.
inline std::vector<LinearConstraintRow> read_constraint_rows(
    const Rcpp::List& lst, int n_x, const char* what
) {
    std::vector<LinearConstraintRow> rows(lst.size());
    for (int g = 0; g < lst.size(); g++) {
        SEXP el = lst[g];
        Rcpp::IntegerVector cols;
        Rcpp::NumericVector w;
        bool weighted = false;
        if (Rf_isNewList(el)) {
            Rcpp::List e(el);
            cols = Rcpp::as<Rcpp::IntegerVector>(e["cols"]);
            w = Rcpp::as<Rcpp::NumericVector>(e["w"]);
            if (w.size() != cols.size())
                Rcpp::stop("%s[[%d]]: `w` has length %d but `cols` has %d.",
                           what, g + 1, static_cast<int>(w.size()),
                           static_cast<int>(cols.size()));
            weighted = true;
        } else {
            cols = Rcpp::as<Rcpp::IntegerVector>(el);
        }
        LinearConstraintRow& r = rows[g];
        r.cols.reserve(cols.size());
        int max_v = 0;
        for (int e = 0; e < cols.size(); e++) {
            const int v = cols[e];
            if (v == NA_INTEGER || v < 1 || (n_x > 0 && v > n_x))
                Rcpp::stop("%s[[%d]][%d] is not a latent index in [1, %d].",
                           what, g + 1, e + 1, n_x);
            if (v > max_v) max_v = v;
        }
        if (weighted) {
            for (int e = 0; e < cols.size(); e++) r.cols.push_back(cols[e] - 1);
            r.w.assign(w.begin(), w.end());
        } else {
            // A group is the indicator of a SET: a repeated index counts once.
            std::vector<char> seen(static_cast<std::size_t>(max_v) + 1, 0);
            for (int e = 0; e < cols.size(); e++) {
                if (seen[cols[e]]) continue;
                seen[cols[e]] = 1;
                r.cols.push_back(cols[e] - 1);
            }
        }
    }
    return rows;
}

// The rows as R takes them back: an all-ones row is the integer vector of its
// 1-based indices, any other row list(cols = <1-based indices>, w = <weights>),
// the form read_constraint_rows() reads.
inline Rcpp::List write_constraint_rows(const std::vector<LinearConstraintRow>& rows) {
    Rcpp::List out(rows.size());
    for (std::size_t g = 0; g < rows.size(); g++) {
        const LinearConstraintRow& r = rows[g];
        Rcpp::IntegerVector cols(r.cols.size());
        for (std::size_t e = 0; e < r.cols.size(); e++) cols[e] = r.cols[e] + 1;
        bool ones = true;
        for (std::size_t e = 0; e < r.cols.size() && ones; e++)
            ones = r.weight(e) == 1.0;
        if (ones) {
            out[g] = cols;
        } else {
            Rcpp::NumericVector w(r.cols.size());
            for (std::size_t e = 0; e < r.cols.size(); e++) w[e] = r.weight(e);
            out[g] = Rcpp::List::create(Rcpp::Named("cols") = cols,
                                        Rcpp::Named("w") = w);
        }
    }
    return out;
}

// The constraints a cell's inner Gaussian carries, as a read of its stored
// precision Q needs them: the Gaussian has precision M = Q + S' D S, S the
// SOFT rows (the sum-to-zero pins a sparse solve folds at solve time and
// leaves off the stored Q, D their k x k precision, row-major), and is
// conditioned on the HARD rows A x = 0 (every block's level rows and
// structural rows). A pin the solve wrote into Q itself is not a soft row
// here: Q already holds it.
struct CellConstraintRows {
    std::vector<LinearConstraintRow> hard;
    std::vector<LinearConstraintRow> soft;
    std::vector<double> soft_D;
    bool empty() const { return hard.empty() && soft.empty(); }
};

}  // namespace tulpa

#endif  // TULPA_LINEAR_CONSTRAINT_ROWS_H
