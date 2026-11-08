#include <RcppEigen.h>
#include <map>
#include <limits>
#include "outer_spline.h"

namespace tulpa_outer_compile {

using tulpa_outer_spline::Axis;
using tulpa_outer_spline::collocation;
using tulpa_outer_spline::transform_axis;

struct Term {
    std::vector<int> index;
    std::vector<Axis> axes;
    Rcpp::IntegerVector rows;
    double coefficient;

    explicit Term(Rcpp::List entry) : index(Rcpp::as<std::vector<int>>(entry["index"])),
        rows(entry["rows"]), coefficient(Rcpp::as<double>(entry["coefficient"])) {
        Rcpp::List nodes = entry["nodes"];
        for (int j = 0; j < nodes.size(); ++j) axes.emplace_back(Rcpp::NumericVector(nodes[j]), true);
        if (index.size() != axes.size() || !std::isfinite(coefficient))
            Rcpp::stop("Sparse spline terms have inconsistent dimensions.");
        std::size_t size = 1;
        for (const auto& axis : axes) size *= axis.x.size();
        if (size != static_cast<std::size_t>(rows.size()))
            Rcpp::stop("Sparse spline rows must cover their tensor.");
    }
};

inline Axis common_axis(const std::vector<Term>& terms, const std::vector<int>& members, int j) {
    int degree = 0;
    double lower = std::numeric_limits<double>::infinity(), upper = -lower;
    for (int member : members) {
        const auto& axis = terms[member].axes[j];
        degree = std::max(degree, axis.degree);
        if (axis.degree) {
            lower = std::min(lower, axis.x.front());
            upper = std::max(upper, axis.x.back());
        }
    }
    if (!degree) return terms[members.front()].axes[j];
    std::map<double, int> multiplicity;
    auto insert = [&](double value, int count) {
        if (value > lower && value < upper)
            multiplicity[value] = std::max(multiplicity[value], count);
    };
    for (int member : members) {
        const auto& axis = terms[member].axes[j];
        if (!axis.degree) continue;
        insert(axis.x.front(), degree);
        insert(axis.x.back(), degree);
        for (std::size_t k = axis.degree + 1; k + axis.degree + 1 < axis.knots.size();) {
            const double value = axis.knots[k];
            std::size_t next = k + 1;
            while (next < axis.knots.size() && axis.knots[next] == value) ++next;
            insert(value, next - k + degree - axis.degree);
            k = next;
        }
    }
    std::vector<double> knots(degree + 1, lower);
    for (const auto& item : multiplicity) knots.insert(knots.end(), item.second, item.first);
    knots.insert(knots.end(), degree + 1, upper);
    return Axis(std::move(knots), degree);
}

struct PlanGroup { std::vector<int> members; std::vector<Axis> axes; };
struct Plan { int compiled = 0; double cells = 0, corners = 0; std::vector<PlanGroup> groups; };

inline Plan plan(const std::vector<Term>& terms, int compiled) {
    const int d = terms.front().axes.size();
    std::map<std::vector<int>, std::vector<int>> membership;
    for (std::size_t i = 0; i < terms.size(); ++i)
        membership[std::vector<int>(terms[i].index.begin() + compiled, terms[i].index.end())].push_back(i);
    Plan result;
    result.compiled = compiled;
    for (const auto& item : membership) {
        PlanGroup group;
        group.members = item.second;
        double cells = 1, corners = 1;
        for (int j = 0; j < d; ++j) {
            group.axes.push_back(j < compiled ? common_axis(terms, group.members, j) :
                terms[group.members.front()].axes[j]);
            cells *= group.axes.back().x.size();
            corners *= group.axes.back().degree + 1;
        }
        result.cells += cells;
        result.corners += corners;
        result.groups.push_back(std::move(group));
    }
    return result;
}

struct Chunk { std::vector<int> index; std::vector<std::size_t> sizes; Eigen::MatrixXd values; };
struct Group { std::vector<Axis> axes; Eigen::MatrixXd coefficients; };
struct Spline { int d = 0, fields = 0, compiled = 0; double cells = 0; Eigen::VectorXd offset; std::vector<Group> groups; };
struct SerialEigen {
    int previous = Eigen::nbThreads();
    SerialEigen() { Eigen::setNbThreads(1); }
    ~SerialEigen() { Eigen::setNbThreads(previous); }
};

inline Group compile_group(const PlanGroup& group, const std::vector<Term>& terms,
    const Rcpp::NumericMatrix& values, const Eigen::VectorXd& center, int compiled) {
    const int d = group.axes.size(), fields = values.ncol();
    std::vector<Chunk> chunks;
    for (int member : group.members) {
        const auto& term = terms[member];
        Chunk chunk;
        chunk.index = term.index;
        for (const auto& axis : term.axes) chunk.sizes.push_back(axis.x.size());
        chunk.values.resize(term.rows.size(), fields);
        for (int f = 0; f < fields; ++f) for (int i = 0; i < term.rows.size(); ++i) {
            const int row = term.rows[i] - 1;
            if (row < 0 || row >= values.nrow() || !std::isfinite(values(row, f)))
                Rcpp::stop("Sparse spline rows and values must be finite and valid.");
            chunk.values(i, f) = (values(row, f) - center[f]) * term.coefficient;
        }
        chunks.push_back(std::move(chunk));
    }
    for (int j = 0; j < compiled; ++j) {
        const auto& common = group.axes[j];
        const Eigen::MatrixXd B = collocation(common, common.x);
        std::map<int, Eigen::MatrixXd> operators;
        std::map<std::vector<int>, Chunk> merged;
        for (auto& chunk : chunks) {
            Rcpp::checkUserInterrupt();
            const int level = chunk.index[j];
            if (!operators.count(level)) {
                const Axis* original = nullptr;
                for (int member : group.members) if (terms[member].index[j] == level) {
                    original = &terms[member].axes[j]; break;
                }
                Eigen::MatrixXd A;
                if (!original->degree) A = Eigen::MatrixXd::Ones(common.x.size(), 1);
                else {
                    Eigen::MatrixXd inverse = Eigen::MatrixXd::Identity(original->x.size(), original->x.size());
                    if (original->degree > 1) inverse = collocation(*original, original->x).partialPivLu().solve(inverse).eval();
                    A = B.partialPivLu().solve(collocation(*original, common.x) * inverse);
                }
                operators.emplace(level, std::move(A));
            }
            const Eigen::MatrixXd& A = operators.at(level);
            Eigen::MatrixXd projected = transform_axis(chunk.values, chunk.sizes, j,
                [&](const Eigen::MatrixXd& rhs) { return (A * rhs).eval(); });
            chunk.sizes[j] = common.x.size();
            std::vector<int> key(chunk.index.begin() + j + 1, chunk.index.begin() + compiled);
            auto found = merged.find(key);
            if (found == merged.end()) {
                chunk.values = std::move(projected);
                merged.emplace(std::move(key), std::move(chunk));
            } else found->second.values += projected;
        }
        chunks.clear();
        for (auto& item : merged) chunks.push_back(std::move(item.second));
    }
    Chunk chunk = std::move(chunks.front());
    for (std::size_t i = 1; i < chunks.size(); ++i) chunk.values += chunks[i].values;
    for (int j = compiled; j < d; ++j) {
        const auto& axis = group.axes[j];
        if (axis.degree <= 1) continue;
        const Eigen::MatrixXd B = collocation(axis, axis.x);
        chunk.values = transform_axis(chunk.values, chunk.sizes, j,
            [&](const Eigen::MatrixXd& rhs) { return B.partialPivLu().solve(rhs).eval(); });
    }
    return Group{group.axes, std::move(chunk.values)};
}

}

// [[Rcpp::export]]
Rcpp::List cpp_outer_sparse_compile(Rcpp::List input, Rcpp::NumericMatrix values,
                                      int n_query, SEXP previous = R_NilValue) {
    using namespace tulpa_outer_compile;
    if (!input.size() || !values.nrow() || n_query < 1)
        Rcpp::stop("Sparse spline compilation requires terms, values and queries.");
    SerialEigen serial;
    std::vector<Term> terms;
    for (int i = 0; i < input.size(); ++i) terms.emplace_back(Rcpp::List(input[i]));
    const int d = terms.front().axes.size();
    if (!d) Rcpp::stop("Sparse splines require at least one axis.");
    for (const auto& term : terms) if (term.axes.size() != static_cast<std::size_t>(d))
        Rcpp::stop("Sparse spline terms have inconsistent dimensions.");
    Plan selected = plan(terms, 0);
    const double budget = std::max(selected.cells, static_cast<double>(n_query) * terms.size());
    for (int k = 1; k <= d; ++k) {
        Plan candidate = plan(terms, k);
        if (candidate.cells <= budget && candidate.corners < selected.corners)
            selected = std::move(candidate);
    }
    Rcpp::XPtr<Spline> spline = previous == R_NilValue ? Rcpp::XPtr<Spline>(new Spline(), true) : Rcpp::XPtr<Spline>(previous);
    spline->groups.clear();
    spline->d = d; spline->fields = values.ncol(); spline->compiled = selected.compiled; spline->cells = selected.cells;
    Eigen::VectorXd center(values.ncol());
    const int center_row = terms.front().rows[0] - 1;
    if (center_row < 0 || center_row >= values.nrow()) Rcpp::stop("Sparse spline rows must be valid.");
    for (int f = 0; f < values.ncol(); ++f) center[f] = values(center_row, f);
    double coefficient_sum = 0;
    for (const auto& term : terms) coefficient_sum += term.coefficient;
    spline->offset = coefficient_sum * center;
    for (const auto& group : selected.groups)
        spline->groups.push_back(compile_group(group, terms, values, center, selected.compiled));
    return Rcpp::List::create(Rcpp::Named("pointer") = spline,
        Rcpp::Named("groups") = spline->groups.size(), Rcpp::Named("coefficients") = selected.cells,
        Rcpp::Named("compiled_axes") = selected.compiled);
}

// [[Rcpp::export]]
Rcpp::NumericMatrix cpp_outer_sparse_evaluate(SEXP pointer, Rcpp::NumericMatrix query) {
    Rcpp::XPtr<tulpa_outer_compile::Spline> spline(pointer);
    if (query.ncol() != spline->d) Rcpp::stop("Sparse spline query dimensions differ from its axes.");
    Rcpp::NumericMatrix output(query.nrow(), spline->fields);
    for (const auto& group : spline->groups)
        tulpa_outer_spline::accumulate(group.axes, group.coefficients, query, output);
    for (int f = 0; f < spline->fields; ++f) for (int i = 0; i < query.nrow(); ++i)
        output(i, f) += spline->offset[f];
    return output;
}

// [[Rcpp::export]]
void cpp_outer_sparse_clear(SEXP pointer) {
    Rcpp::XPtr<tulpa_outer_compile::Spline> spline(pointer);
    spline->groups.clear();
    spline->offset.resize(0);
    spline->d = spline->fields = 0;
}
