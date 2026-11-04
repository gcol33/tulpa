#include <RcppEigen.h>
#include "laplace_newton_joint.h"
#include "scatter_dense_basis.h"

namespace {

struct DenseBasisWorking {
    const std::vector<double>& weights;
    const std::vector<double>& scores;
};

Rcpp::List dense_basis_scatter(Rcpp::NumericMatrix A,
                               Rcpp::NumericMatrix B,
                               const std::vector<double>& weights,
                               const std::vector<double>& scores) {
    const int N = A.nrow(), ma = A.ncol(), mb = B.ncol();
    if (B.nrow() != N || weights.size() != N || N == 0)
        Rcpp::stop("Matching nonempty observation dimensions are required.");
    const DenseBasisWorking working{weights, scores};
    tulpa::LikelihoodSpec spec;
    spec.eta_weights_fn = [](
        int i, const double*, double, double, const std::vector<double>&,
        const tulpa::ModelData&, const tulpa::ParamLayout&, const void* response,
        double* gradient, double* curvature) {
        const auto& ww = *static_cast<const DenseBasisWorking*>(response);
        *gradient = ww.scores[i];
        *curvature = ww.weights[i];
    };
    tulpa::ArmSpecView view;
    view.spec = &spec;
    view.response_data = &working;
    view.data = &tulpa::joint_empty_model_data();
    view.layout = &tulpa::joint_empty_param_layout();
    view.params = &tulpa::joint_empty_params();
    tulpa::ParsedArm pa;
    pa.p = 2;
    pa.beta_start = 0;
    pa.n_re_groups = 2;
    pa.re_start = 2;
    pa.X = Rcpp::NumericMatrix(N, 2);
    pa.re_idx = Rcpp::NumericVector(N);
    for (int i = 0; i < N; ++i) {
        pa.X(i, 0) = 1.0;
        pa.X(i, 1) = (i + 1.0) / N;
        pa.re_idx[i] = 1 + i % 2;
    }
    tulpa::JointArm arm;
    arm.N = N;
    auto block = [N](Rcpp::NumericMatrix phi, int start) {
        const int M = phi.ncol();
        auto flat = std::make_shared<std::vector<double>>(N * M);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < M; ++j) (*flat)[i * M + j] = phi(i, j);
        tulpa::LatentBlock b;
        b.start = start;
        b.size = M;
        b.contrib_kind = tulpa::BlockContribKind::DENSE_BASIS;
        b.dense_basis_batch = [flat, N, M](int, int) {
            tulpa::LatentBlock::DenseBasisBatch batch;
            batch.data = flat->data();
            batch.sqrt_S = nullptr;
            batch.N_k = N;
            batch.m_per_arm = M;
            batch.m_offset_in_block = 0;
            return batch;
        };
        return b;
    };
    auto a = block(A, 4);
    auto b = block(B, 4 + ma);
    const int dim = 4 + ma + mb;
    std::vector<std::pair<int, int>> pattern;
    for (int j = 0; j < dim; ++j)
        for (int i = j; i < dim; ++i) pattern.emplace_back(i, j);
    tulpa::SparseHessianBuilder H;
    H.init(dim, pattern);
    tulpa::DenseVec gradient(dim, 0.0);
    tulpa::DenseBasisScratch sa, sb;
    Rcpp::NumericVector eta(N);
    tulpa::scatter_dense_basis_block(a, 0, 0, pa, arm, view, eta, -0.7, gradient, H, sa);
    tulpa::scatter_dense_basis_block(b, 0, 0, pa, arm, view, eta, 1.3, gradient, H, sb);
    tulpa::scatter_dense_basis_cross(a, b, 0, 0, -0.7, 1.3, sa, sb, H);
    Rcpp::NumericMatrix out(dim, dim);
    for (int j = 0; j < dim; ++j)
        for (int i = j; i < dim; ++i)
            out(i, j) = out(j, i) = H.values[H.lookup(i, j)];
    return Rcpp::List::create(Rcpp::Named("H") = out,
                              Rcpp::Named("gradient") = Rcpp::wrap(gradient));
}

} // namespace

// [[Rcpp::export]]
Rcpp::List cpp_test_dense_basis_scatter(Rcpp::NumericMatrix A,
                                       Rcpp::NumericMatrix B,
                                       Rcpp::NumericVector weights) {
    std::vector<double> w(weights.begin(), weights.end()), g(A.nrow());
    for (int i = 0; i < A.nrow(); ++i) g[i] = 1.0 + 0.1 * i;
    return dense_basis_scatter(A, B, w, g);
}

// [[Rcpp::export]]
Rcpp::List cpp_test_dense_basis_beta(Rcpp::NumericMatrix A,
                                    Rcpp::NumericMatrix B,
                                    Rcpp::NumericVector y,
                                    Rcpp::NumericVector eta, double phi) {
    const int N = A.nrow();
    if (y.size() != N || eta.size() != N) Rcpp::stop("Matching observation lengths are required.");
    std::vector<double> w(N), g(N);
    for (int i = 0; i < N; ++i) {
        const auto gh = tulpa::grad_hess_for_family(y[i], 1, eta[i], "beta", phi);
        w[i] = gh.neg_hess;
        g[i] = gh.grad;
    }
    return dense_basis_scatter(A, B, w, g);
}
