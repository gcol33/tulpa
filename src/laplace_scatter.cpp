// laplace_scatter.cpp
// Observation likelihood gradient/Hessian scatter helpers for Laplace engines.

#include "laplace_scatter.h"
#include "laplace_family_link.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

namespace tulpa {

void scatter_obs_grad_hess_base(
    const NumericVector& y, const IntegerVector& n_trials,
    const NumericMatrix& X, const NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const NumericVector& eta, const std::string& family, double phi,
    DenseVec& grad, DenseMat& H, int n_threads
) {
    #ifdef _OPENMP
    #pragma omp parallel num_threads(n_threads > 0 ? n_threads : 1)
    {
        DenseVec grad_thread(grad.size(), 0.0);
        DenseMat H_thread(H.size(), DenseVec(H.size(), 0.0));

        #pragma omp for schedule(static)
        for (int i = 0; i < N; i++) {
            auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

            for (int j = 0; j < p; j++) {
                grad_thread[j] += gh.grad * X(i, j);
                for (int k = 0; k < p; k++) {
                    H_thread[j][k] += gh.neg_hess * X(i, j) * X(i, k);
                }
            }

            if (n_re_groups > 0) {
                int g = (int)re_idx[i] - 1;
                if (g >= 0 && g < n_re_groups) {
                    grad_thread[p + g] += gh.grad;
                    H_thread[p + g][p + g] += gh.neg_hess;
                    for (int j = 0; j < p; j++) {
                        H_thread[j][p + g] += gh.neg_hess * X(i, j);
                        H_thread[p + g][j] += gh.neg_hess * X(i, j);
                    }
                }
            }
        }

        #pragma omp critical
        {
            int n_x = (int)grad.size();
            for (int j = 0; j < n_x; j++) {
                grad[j] += grad_thread[j];
                for (int k = 0; k < n_x; k++) {
                    H[j][k] += H_thread[j][k];
                }
            }
        }
    }
    #else
    for (int i = 0; i < N; i++) {
        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);

        for (int j = 0; j < p; j++) {
            grad[j] += gh.grad * X(i, j);
            for (int k = 0; k < p; k++) {
                H[j][k] += gh.neg_hess * X(i, j) * X(i, k);
            }
        }

        if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
                grad[p + g] += gh.grad;
                H[p + g][p + g] += gh.neg_hess;
                for (int j = 0; j < p; j++) {
                    H[j][p + g] += gh.neg_hess * X(i, j);
                    H[p + g][j] += gh.neg_hess * X(i, j);
                }
            }
        }
    }
    #endif
}

void scatter_obs_with_latent(
    const NumericVector& y, const IntegerVector& n_trials,
    const NumericMatrix& X, const NumericVector& re_idx,
    int N, int p, int n_re_groups,
    const NumericVector& eta, const std::string& family, double phi,
    const std::vector<int>& effect_idx,
    const std::vector<double>& d_factors,
    DenseVec& grad, DenseMat& H, int n_threads
) {
    scatter_obs_grad_hess_base(y, n_trials, X, re_idx, N, p, n_re_groups,
                               eta, family, phi, grad, H, n_threads);

    for (int i = 0; i < N; i++) {
        int idx = effect_idx[i];
        if (idx < 0) continue;

        auto gh = grad_hess_for_family(y[i], n_trials[i], eta[i], family, phi);
        double d = d_factors[i];

        grad[idx] += gh.grad * d;
        H[idx][idx] += gh.neg_hess * d * d;

        for (int j = 0; j < p; j++) {
            H[j][idx] += gh.neg_hess * X(i, j) * d;
            H[idx][j] += gh.neg_hess * X(i, j) * d;
        }

        if (n_re_groups > 0) {
            int g = (int)re_idx[i] - 1;
            if (g >= 0 && g < n_re_groups) {
                H[p + g][idx] += gh.neg_hess * d;
                H[idx][p + g] += gh.neg_hess * d;
            }
        }
    }
}

} // namespace tulpa
