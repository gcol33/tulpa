// Test probes over src/laplace_profile.cpp, compiled against the engine by load_cpp_fixture().

#include <RcppEigen.h>
#include "laplace_profile.h"
#include "laplace_newton_loop.h"   // line_search_backtrack (probe)
#include <Rcpp.h>
#include <vector>

// One-dimensional quadratic line search from x = 0 along delta = 1, with an
// additive objective offset and trial evaluation error. Reports the accepted
// step, objective and number of evaluations.
// [[Rcpp::export]]
Rcpp::List cpp_line_search_probe(double slope, double c,
                                double obj_offset = 0.0,
                                double trial_error = 0.0) {
    Rcpp::NumericVector x(1, 0.0), x_try(1, 0.0);
    std::vector<double> delta(1, 1.0);
    double obj_old = obj_offset, obj_out = obj_offset;
    int n_evals = 0;
    auto eval_obj = [&](const Rcpp::NumericVector& xv) -> double {
        const double a = xv[0];
        return obj_offset + slope * a + c * a * a + trial_error;
    };
    double step = tulpa::line_search_backtrack(
        x, delta, 1, obj_old, slope, eval_obj, obj_out, x_try, &n_evals);
    return Rcpp::List::create(
        Rcpp::Named("step")    = step,
        Rcpp::Named("obj")     = obj_out,
        Rcpp::Named("n_evals") = n_evals
    );
}

// Convergence-test probe: one newton_converged verdict on a supplied Newton step
// and gradient, at a given accepted step scale, on a fresh solve state. Lets the
// test suite pin that the verdict reads the Newton PROPOSAL and not the damped
// step the line search settled on -- the property that keeps a proposal damped
// toward zero from clearing the tolerance.
// [[Rcpp::export]]
bool cpp_newton_converged_probe(Rcpp::NumericVector delta,
                                Rcpp::NumericVector grad,
                                double step_scale, double tol) {
    tulpa::NewtonConvState st;
    std::vector<double> d(delta.begin(), delta.end());
    std::vector<double> g(grad.begin(), grad.end());
    return tulpa::newton_converged(d, g, step_scale, (int)d.size(), tol, st);
}

// Trust-factor probe: replay newton_trust_scale over a supplied sequence of
// Newton decrements on one NewtonConvState and report the scale the line search
// would have opened each iteration with. The schedule is otherwise invisible from
// R -- a fit only shows where it landed, not the damping that took it there.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_newton_trust_probe(Rcpp::NumericVector decrements) {
    tulpa::NewtonConvState st;
    const int n = decrements.size();
    Rcpp::NumericVector out(n);
    for (int i = 0; i < n; i++) {
        out[i] = tulpa::newton_trust_scale(st, decrements[i]);
    }
    return out;
}
