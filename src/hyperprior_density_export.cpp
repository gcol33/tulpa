// hyperprior_density_export.cpp
// The outer grid's default hyperprior densities, read by R through one entry.
//
// Every density is the pc_prior.h one, so the grid integrates the same prior the
// samplers sample. `coordinate` names what the density is over: "natural" is the
// parameter itself, "log" is its logarithm (the coordinate a log-scale grid axis
// is spaced and integrated in), which adds log|d theta / d log theta| = log theta.

#include "pc_prior.h"
#include <Rcpp.h>
#include <Rmath.h>
#include <cmath>
#include <limits>
#include <string>

namespace {

// PC prior on a negative-binomial overdispersion a = 1 / size against the
// Poisson base model (a = 0), in the form R-INLA ships as `pc.mgamma`
// (`inla.pc.dgamma()`, rinla/R/pc-gamma.R):
//
//   d(a)     = sqrt(2 (log(1/a) - psi(1/a)))
//   log p(a) = log(lambda) - lambda d - log d - 2 log a
//              + log(psi'(1/a) - a)
//
// Both differences cancel as 1/a grows, so past z = 1/a = 1e4 they are read
// from the asymptotic series of the digamma and trigamma functions
// (Abramowitz & Stegun 6.3.18, 6.4.12), the switch R-INLA's C
// `priorfunc_pc_gamma` makes at the same point.
double log_prior_nb_overdispersion_pc(double a, double lambda) {
    const double z = 1.0 / a;
    double g, h;
    if (z > 1.0e4) {
        const double z2 = z * z;
        g = 0.5 / z + 1.0 / (12.0 * z2) - 1.0 / (120.0 * z2 * z2);
        h = 0.5 / z2 + 1.0 / (6.0 * z2 * z) - 1.0 / (30.0 * z2 * z2 * z);
    } else {
        g = std::log(z) - R::digamma(z);
        h = R::trigamma(z) - a;
    }
    const double d = std::sqrt(2.0 * g);
    return std::log(lambda) - lambda * d - std::log(d) - 2.0 * std::log(a) +
           std::log(h);
}

} // namespace

// `kind = "nb_size"` is the negative-binomial SIZE, the density above carried
// from a = 1 / size; it reads `lambda` and ignores the (U, alpha) anchors.
// [[Rcpp::export]]
Rcpp::NumericVector cpp_hyperprior_log_density(
    Rcpp::NumericVector x, std::string kind, std::string coordinate,
    double U, double alpha, double d = 2.0, double lambda = 7.0
) {
    const bool on_log = (coordinate == "log");
    if (!on_log && coordinate != "natural") {
        Rcpp::stop("`coordinate` must be \"natural\" or \"log\", not \"%s\".",
                   coordinate.c_str());
    }
    enum class Kind { SD, VARIANCE, PRECISION, RANGE, NB_SIZE };
    Kind k;
    if      (kind == "sd")        k = Kind::SD;
    else if (kind == "variance")  k = Kind::VARIANCE;
    else if (kind == "precision") k = Kind::PRECISION;
    else if (kind == "range")     k = Kind::RANGE;
    else if (kind == "nb_size")   k = Kind::NB_SIZE;
    else Rcpp::stop("Unknown hyperprior kind \"%s\".", kind.c_str());
    if (k == Kind::NB_SIZE) {
        if (!(lambda > 0.0) || !std::isfinite(lambda)) {
            Rcpp::stop("The negative-binomial PC prior needs lambda > 0; got %g.",
                       lambda);
        }
    } else if (!tulpa::pc_anchors_valid(U, alpha)) {
        Rcpp::stop("PC anchors need U > 0 and alpha in (0, 1); got U = %g, "
                   "alpha = %g.", U, alpha);
    }
    if (k == Kind::RANGE && !(d > 0.0)) {
        Rcpp::stop("The range PC prior needs a positive dimension; got d = %g.", d);
    }

    const double ninf = -std::numeric_limits<double>::infinity();
    const R_xlen_t n = x.size();
    Rcpp::NumericVector out(n);
    for (R_xlen_t i = 0; i < n; i++) {
        const double v = x[i];
        if (!std::isfinite(v) || (!on_log && v <= 0.0)) {
            out[i] = (std::isnan(v)) ? NA_REAL : ninf;
            continue;
        }
        double lp = 0.0;
        switch (k) {
        case Kind::SD:
            lp = on_log ? tulpa::log_prior_log_sigma_pc(v, U, alpha)
                        : tulpa::log_prior_sigma_pc(v, U, alpha);
            break;
        case Kind::VARIANCE:
            lp = on_log ? tulpa::log_prior_log_sigma2_pc(v, U, alpha)
                        : tulpa::log_prior_sigma2_pc(v, U, alpha);
            break;
        case Kind::PRECISION:
            lp = on_log ? tulpa::log_prior_log_tau_pc(v, U, alpha)
                        : tulpa::log_prior_tau_pc(v, U, alpha);
            break;
        case Kind::RANGE: {
            const double log_r = on_log ? v : std::log(v);
            lp = tulpa::log_prior_range_pc_d_at_log(log_r, U, alpha, d);
            if (on_log) lp += log_r;
            break;
        }
        case Kind::NB_SIZE: {
            // size s = 1 / a: |da/ds| = a^2, |da/dlog s| = a.
            const double log_s = on_log ? v : std::log(v);
            const double a = std::exp(-log_s);
            lp = log_prior_nb_overdispersion_pc(a, lambda) +
                 (on_log ? -log_s : -2.0 * log_s);
            break;
        }
        }
        out[i] = std::isfinite(lp) ? lp : ninf;
    }
    return out;
}
