#include <RcppEigen.h>
#include "coordinate_linalg.h"

// [[Rcpp::depends(RcppEigen)]]

using tulpa_coordinate_linalg::Var;

static std::vector<double> row_values(const Rcpp::NumericMatrix& x) {
    std::vector<double> out(x.nrow() * x.ncol());
    for (int i = 0; i < x.nrow(); ++i) for (int j = 0; j < x.ncol(); ++j)
        out[i * x.ncol() + j] = x(i, j);
    return out;
}

static Rcpp::NumericMatrix column_matrix(const std::vector<double>& x, int rows, int columns) {
    Rcpp::NumericMatrix out(rows, columns);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < columns; ++j)
        out(i, j) = x[i * columns + j];
    return out;
}

// [[Rcpp::export]]
Rcpp::List cpp_coordinate_product(Rcpp::NumericMatrix a, Rcpp::NumericMatrix b,
    Rcpp::NumericMatrix weight, bool transpose_left) {
    tulpa::arena::ArenaScope scope;
    const auto x = tulpa::arena::make_vars(scope.arena(), row_values(a));
    const auto y = tulpa::arena::make_vars(scope.arena(), row_values(b));
    const int rows = transpose_left ? a.ncol() : a.nrow();
    const int inner = transpose_left ? a.nrow() : a.ncol();
    const auto z = tulpa_coordinate_linalg::product(x, y, rows, inner, b.ncol(), transpose_left);
    Var objective(0);
    std::vector<double> result(z.size());
    for (int i = 0; i < rows; ++i) for (int j = 0; j < b.ncol(); ++j) {
        const int k = i * b.ncol() + j;
        objective = objective + z[k] * weight(i, j);
        result[k] = z[k].val();
    }
    objective.backward();
    return Rcpp::List::create(
        Rcpp::Named("value") = column_matrix(result, rows, b.ncol()),
        Rcpp::Named("left_gradient") = column_matrix(tulpa::arena::get_adjoints(x), a.nrow(), a.ncol()),
        Rcpp::Named("right_gradient") = column_matrix(tulpa::arena::get_adjoints(y), b.nrow(), b.ncol()));
}

// [[Rcpp::export]]
Rcpp::List cpp_coordinate_cholesky(Rcpp::NumericMatrix a, Rcpp::NumericMatrix weight) {
    tulpa::arena::ArenaScope scope;
    const auto x = tulpa::arena::make_vars(scope.arena(), row_values(a));
    std::vector<Var> lower;
    if (!tulpa_coordinate_linalg::cholesky(x, a.nrow(), lower))
        return Rcpp::List::create(Rcpp::Named("valid") = false);
    Var objective(0);
    std::vector<double> result(lower.size());
    for (int i = 0; i < a.nrow(); ++i) for (int j = 0; j < a.ncol(); ++j) {
        const int k = i * a.ncol() + j;
        objective = objective + lower[k] * weight(i, j);
        result[k] = lower[k].val();
    }
    objective.backward();
    return Rcpp::List::create(Rcpp::Named("valid") = true,
        Rcpp::Named("value") = column_matrix(result, a.nrow(), a.ncol()),
        Rcpp::Named("gradient") = column_matrix(tulpa::arena::get_adjoints(x), a.nrow(), a.ncol()));
}
