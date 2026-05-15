// spde_nc_transform.cpp
// Implementation of the non-centered SPDE transform.
//
// Derivation. F(w; theta) = L^T w - z = 0 with theta = (log_kappa, log_tau)
// and L L^T = Q(kappa, tau). Implicit-function theorem on F gives:
//   dz                = L^{-1} dw                       (forward solve)
//   L_bar (lower tri) = -w (dz)^T                       (rank-1)
//   M_theta           = L^{-1} (dQ/dtheta) L^{-T}       (symmetric)
//   Q_bar             = 0.5 L^{-T} S(L^T L_bar) L^{-1}
//   dlog_theta        = trace(Q_bar (dQ/dtheta))
// where S(M) keeps the lower triangle of M and halves the diagonal.
//
// We never need Q_bar explicitly. Murray (2016) eq. 2.2 plus the rank-1
// shape of L_bar collapse the trace to:
//   dlog_theta = -<z, Phi(M_theta) y>     y = L^{-1} dw
// where Phi(M) = lower(M) with the diagonal halved. Forming M_theta as a
// dense n_mesh-by-n_mesh matrix is O(n_mesh^3) but with n_mesh in the
// hundreds for typical SPDE meshes it stays cheap relative to one HMC
// trajectory. (a.iii) can replace this with a sparse partial-inverse path
// when the mesh sizes scale up.
//
// dQ/dtheta closed forms (alpha = 2):
//   dQ/dlog_kappa = 4 kappa^2 tau^2 K
//   dQ/dlog_tau   = 2 Q

#include "spde_nc_transform.h"

#include <RcppEigen.h>
#include <stdexcept>
#include <cmath>

namespace tulpa {

void SpdeNcTransform::init(int n,
                           const std::vector<double>& C0_d,
                           const std::vector<double>& G1_x,
                           const std::vector<int>&    G1_i,
                           const std::vector<int>&    G1_p)
{
    n_mesh  = n;
    C0_diag = Eigen::Map<const Eigen::VectorXd>(C0_d.data(), n);
    C0_inv_diag.resize(n);
    const double eps = 1e-15;
    for (int i = 0; i < n; i++) {
        C0_inv_diag[i] = (C0_diag[i] > eps) ? 1.0 / C0_diag[i] : 0.0;
    }
    G1 = SpMat(n, n);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(G1_x.size());
    for (int j = 0; j < n; j++) {
        for (int idx = G1_p[j]; idx < G1_p[j + 1]; idx++) {
            trips.emplace_back(G1_i[idx], j, G1_x[idx]);
        }
    }
    G1.setFromTriplets(trips.begin(), trips.end());
    G1.makeCompressed();
}

SpdeNcTransform::SpMat SpdeNcTransform::build_K(double kappa) const {
    SpMat K = G1;
    const double k2 = kappa * kappa;
    for (int i = 0; i < n_mesh; i++) {
        K.coeffRef(i, i) += k2 * C0_diag[i];
    }
    K.makeCompressed();
    return K;
}

SpdeNcTransform::SpMat SpdeNcTransform::build_Q(const SpMat& K, double tau) const {
    // K_DC = K * diag(1/C0): scale columns of K by 1/C0_diag[j].
    SpMat K_DC = K;
    for (int j = 0; j < n_mesh; j++) {
        for (SpMat::InnerIterator it(K_DC, j); it; ++it) {
            it.valueRef() *= C0_inv_diag[j];
        }
    }
    SpMat Q = K_DC * K;
    Q *= (tau * tau);
    Q.makeCompressed();
    return Q;
}

Eigen::VectorXd SpdeNcTransform::forward(const Eigen::VectorXd& z,
                                         double kappa, double tau)
{
    if (z.size() != n_mesh) throw std::runtime_error("z size mismatch");
    last_kappa = kappa;
    last_tau   = tau;
    last_K     = build_K(kappa);
    last_Q     = build_Q(last_K, tau);
    llt.compute(last_Q);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("SpdeNcTransform: Cholesky failed (Q not PD)");
    }
    factored = true;
    SpMat L = llt.matrixL();
    Eigen::VectorXd w = L.transpose().template triangularView<Eigen::Upper>().solve(z);
    return w;
}

void SpdeNcTransform::backward(
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& w,
    const Eigen::VectorXd& dw,
    Eigen::VectorXd&       dz_out,
    double&                dlog_kappa_out,
    double&                dlog_tau_out
) const
{
    if (!factored) throw std::runtime_error("SpdeNcTransform::backward without forward");
    const int n = n_mesh;
    (void) w;  // not needed in the rank-1 form; kept in the signature
               // because future variants (e.g. partial-inverse path) will
               // use it.

    SpMat L = llt.matrixL();
    Eigen::VectorXd y = L.template triangularView<Eigen::Lower>().solve(dw);
    dz_out = y;

    auto trace_term = [&](const SpMat& dQ_theta) -> double {
        // M = L^{-1} dQ_theta L^{-T} formed densely.
        Eigen::MatrixXd dQ_dense(dQ_theta);
        Eigen::MatrixXd tmp = L.template triangularView<Eigen::Lower>().solve(dQ_dense);
        Eigen::MatrixXd MT  = L.template triangularView<Eigen::Lower>().solve(tmp.transpose());
        Eigen::MatrixXd M   = MT.transpose();
        // phi_y = Phi(M) y (lower tri with diagonal halved, times y).
        Eigen::VectorXd phi_y = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; i++) {
            double acc = 0.5 * M(i, i) * y[i];
            for (int j = 0; j < i; j++) acc += M(i, j) * y[j];
            phi_y[i] = acc;
        }
        return -z.dot(phi_y);
    };

    SpMat dQ_logkappa = (4.0 * last_kappa * last_kappa * last_tau * last_tau) * last_K;
    dlog_kappa_out = trace_term(dQ_logkappa);

    SpMat dQ_logtau = 2.0 * last_Q;
    dlog_tau_out = trace_term(dQ_logtau);
}

std::vector<arena::Var> spde_nc_transform_arena(
    arena::Arena*                  ar,
    const std::vector<arena::Var>& z,
    const arena::Var&              log_kappa,
    const arena::Var&              log_tau,
    SpdeNcTransform&               transform
) {
    const int n = static_cast<int>(z.size());
    const double kappa = std::exp(log_kappa.val());
    const double tau   = std::exp(log_tau.val());

    Eigen::VectorXd z_eig(n);
    for (int i = 0; i < n; i++) z_eig[i] = z[i].val();

    Eigen::VectorXd w_eig = transform.forward(z_eig, kappa, tau);

    std::vector<int32_t> input_indices;
    input_indices.reserve(n + 2);
    for (int i = 0; i < n; i++) input_indices.push_back(z[i].idx_);
    input_indices.push_back(log_kappa.idx_);
    input_indices.push_back(log_tau.idx_);

    std::vector<double> output_values(w_eig.data(), w_eig.data() + n);

    auto cb = [&transform, n](
        const double* input_vals,  int /*n_in*/,
        const double* output_vals, int /*n_out*/,
        const double* output_adjs,
        double*       input_adjs
    ) {
        Eigen::VectorXd z_local(n), w_local(n), dw_local(n);
        for (int i = 0; i < n; i++) {
            z_local[i]  = input_vals[i];
            w_local[i]  = output_vals[i];
            dw_local[i] = output_adjs[i];
        }
        Eigen::VectorXd dz_local;
        double dlog_kappa = 0.0, dlog_tau = 0.0;
        transform.backward(z_local, w_local, dw_local,
                           dz_local, dlog_kappa, dlog_tau);
        for (int i = 0; i < n; i++) input_adjs[i] = dz_local[i];
        input_adjs[n]     = dlog_kappa;
        input_adjs[n + 1] = dlog_tau;
    };

    std::vector<int32_t> out_indices;
    ar->add_custom_backward(input_indices, output_values, cb, out_indices);

    std::vector<arena::Var> w(n);
    for (int i = 0; i < n; i++) {
        w[i].arena_ = ar;
        w[i].idx_   = out_indices[i];
    }
    return w;
}

} // namespace tulpa
