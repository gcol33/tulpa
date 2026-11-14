#ifndef TULPA_LIKELIHOOD_MATH_H
#define TULPA_LIKELIHOOD_MATH_H

#include "device_math.h"
#include "count_math.h"

namespace tulpa { namespace math {

enum class FusedFamily { Gaussian, Binomial, Poisson, Negbin2, Gamma, Beta,
    BetaBinomial, TruncatedPoisson, TruncatedNegbin2, Negbin1 };

struct FusedRow {
    double value = 0, score = 0, weight = 0, observed = 0;
};

TULPA_HD inline double fused_logistic(double x) {
    const double e = std::exp(x < 0 ? x : -x);
    return x < 0 ? e/(1+e) : 1/(1+e);
}
TULPA_HD inline double fused_softplus(double x) {
    return CountScalarMath::softplus(x);
}
TULPA_HD inline double fused_exp(double x) {
    return std::exp(clamp_exp_arg(x));
}
TULPA_HD inline double fused_trigamma(double x) {
    return 1/x + (.5/x)/x + stirling_remainder_derivative(x, 2);
}

template<int Family> TULPA_HD inline FusedRow fused_base_row(
    double y, int n, double eta, double phi, double constant) {
    constexpr auto family = static_cast<FusedFamily>(Family);
    FusedRow r;
    if constexpr (family == FusedFamily::Gaussian) {
        const double d = y-eta, precision = 1/(phi*phi);
        r.value = -.5*(log_two_pi + 2*std::log(phi) + d*d*precision);
        r.score = d*precision; r.weight = r.observed = precision;
    } else if constexpr (family == FusedFamily::Binomial) {
        const double p = fused_logistic(eta);
        r.value = constant + y*eta-n*fused_softplus(eta);
        r.score = y-n*p; r.weight = r.observed = n*p*(1-p);
    } else if constexpr (family == FusedFamily::Beta || family == FusedFamily::BetaBinomial) {
        const double raw_p = fused_logistic(eta);
        const double p = clamp_prob(raw_p);
        const double q = 1-p, pq = p*q, a = p*phi, b = q*phi;
        if constexpr (family == FusedFamily::Beta) {
            r.value = beta_log_density(y, p, q, phi, std::log(y), std::log1p(-y));
            const double dmu=raw_p*(1-raw_p);
            r.score = dmu*beta_mu_score(y,p,phi);
            r.weight = (dmu*phi)*(dmu*phi)*(fused_trigamma(a)+fused_trigamma(b));
            r.observed = r.weight-(1-2*raw_p)*r.score;
            r.weight = r.observed;
        } else {
            const int yy = static_cast<int>(y);
            r.value = beta_binomial_log_pmf(yy,n,p,q,phi,constant);
            r.score = phi*pq*(polygamma_rising_difference(0,yy,a)
                                 - polygamma_rising_difference(0,n-yy,b));
            r.weight = n*pq/(1+(n-1)/(phi+1));
            r.observed = -(1-2*p)*r.score-(phi*pq)*(phi*pq)*
                (polygamma_rising_difference(1,yy,a)+polygamma_rising_difference(1,n-yy,b));
        }
    } else {
        const double mu = fused_exp(eta);
        if constexpr (family == FusedFamily::Gamma) {
            const double mean=mu<kMuFloor ? kMuFloor : mu;
            const double ratio=mu/mean;
            r.value = gamma_log_density(y,mean,phi,std::log(y));
            r.score = phi*(y-mean)/mean*ratio;
            r.weight = phi*ratio*ratio; r.observed = phi*y/mean*ratio*ratio;
        } else if constexpr (family == FusedFamily::Negbin1) {
            const double mean=mu<kMuFloor ? kMuFloor : mu;
            const double size = mean/phi;
            r.value = lgamma_ratio_count(static_cast<int>(y),size)+constant
                - (y+size)*std::log1p(phi)+y*std::log(phi);
            r.score = size*(polygamma_rising_difference(0,static_cast<int>(y),size)-std::log1p(phi));
            r.weight = mean/(1+phi);
            r.observed = -r.score-size*size*polygamma_rising_difference(1,static_cast<int>(y),size);
        } else {
            constexpr bool poisson = family == FusedFamily::Poisson || family == FusedFamily::TruncatedPoisson;
            if constexpr (poisson) {
                r.value = y*eta-mu+constant;
                r.score = y-mu; r.weight = r.observed = mu;
            } else {
                r.value = negbin_log_pmf_log(static_cast<int>(y),clamp_exp_arg(eta),std::log(phi),-constant);
                r.score = (y-mu)/(1+mu/phi);
                r.weight = mu/(1+mu/phi);
                r.observed = (phi+y)*mu/phi/((1+mu/phi)*(1+mu/phi));
                r.weight = r.observed;
            }
            if constexpr (family == FusedFamily::TruncatedPoisson || family == FusedFamily::TruncatedNegbin2) {
                const double mean=mu<kMuFloor ? kMuFloor : mu;
                const double a = poisson ? mean : phi*std::log1p(mean/phi);
                const double da = poisson ? mean : mean/(1+mean/phi);
                const double d2a = poisson ? mean : mean/((1+mean/phi)*(1+mean/phi));
                const double mass = -std::expm1(-a), q0 = std::exp(-a);
                const double correction = q0*da/mass;
                r.value -= std::log(mass);
                r.score -= correction;
                r.weight = da/mass-q0*(da/mass)*(da/mass);
                r.observed += q0*(d2a-da*da)/mass-correction*correction;
                if (y<1) r.value=-std::numeric_limits<double>::infinity();
            }
        }
    }
    return r;
}

template<int Family, bool Zi> TULPA_HD inline void fused_likelihood_row(
    const double* response, int n, int i, const double* eta, const double* constants,
    double& value, double* score, double* curvature) {
    const double y = response[i], weight = response[2*n+i];
    constexpr int np = Zi ? 2 : 1;
    value = 0;
    for (int k=0; k<np; ++k) score[k] = 0;
    for (int k=0; k<np*np; ++k) curvature[k] = 0;
    if (std::isnan(y) || weight == 0) return;
    const int trials = static_cast<int>(response[n+i]);
    const double phi = constants[0];
    FusedRow r = fused_base_row<Family>(y,trials,eta[0],phi,response[3*n+i]);
    value = r.value; score[0] = r.score; curvature[0] = r.weight;
    if constexpr (Zi) {
        const double pi = fused_logistic(eta[1]);
        const double log_pi = -fused_softplus(-eta[1]);
        const double log_q = -fused_softplus(eta[1]);
        if (y != 0) {
            value += log_q; score[1] = -pi; curvature[3] = pi*(1-pi);
        } else if constexpr (Family == static_cast<int>(FusedFamily::TruncatedPoisson) ||
                             Family == static_cast<int>(FusedFamily::TruncatedNegbin2)) {
            value = log_pi; score[0] = 0; score[1] = 1-pi;
            curvature[0] = 0; curvature[3] = pi*(1-pi);
        } else {
            const double a = log_pi, b = log_q+r.value;
            const double top = a > b ? a : b;
            value = top+std::log1p(std::exp(-std::abs(a-b)));
            const double post_base = std::exp(b-value), post_zero = std::exp(a-value);
            score[0] = post_base*r.score; score[1] = post_zero-pi;
            curvature[0] = post_base*r.observed-post_base*post_zero*r.score*r.score;
            curvature[1] = curvature[2] = post_base*post_zero*r.score;
            curvature[3] = pi*(1-pi)-post_base*post_zero;
        }
    }
    value *= weight;
    for (int k=0; k<np; ++k) score[k] *= weight;
    for (int k=0; k<np*np; ++k) curvature[k] *= weight;
}

} }
#endif
