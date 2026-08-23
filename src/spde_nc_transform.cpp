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
// where Phi(M) = lower(M) with the diagonal halved.
//
// Streaming trace. Materializing M as a dense n_mesh-by-n_mesh matrix runs
// into O(n_mesh^2) memory (200 MB at n_mesh = 5000) and is painful above
// ~2000. We avoid it by decomposing
//   -<z, Phi(M) y> = -(T_lower + 0.5 D)
// with
//   T_lower = sum_{i>j} z[i] M[i,j] y[j]
//           = sum_j y[j] * u_j^T (dQ v_j),   u_j := sum_{i>j} z[i] v_i,
//   D       = sum_i z[i] M[i,i] y[i]
//           = sum_i (z[i] y[i]) v_i^T (dQ v_i),
//   v_j     := L^{-T} e_j.
// The recursion u_j = u_{j-1} - z[j] v_j (with u_{-1} = L^{-T} z) folds the
// n outer products into one dense-vector update per column. Peak memory is
// O(n_mesh); runtime stays dominated by the per-column back-solve.
//
// dQ/dtheta closed forms:
//   Integer alpha = 2:
//     dQ/dlog_kappa = 4 kappa^2 tau^2 K
//     dQ/dlog_tau   = 2 Q
//   Rational alpha (Bolin et al. 2023, poles {r_k}, weights {w_k}):
//     dQ/dlog_kappa = 4 kappa^2 tau^2 K_sum
//     dQ/dlog_tau   = 2 Q
//     where K_sum := sum_k w_k K_k = (kappa^2 W + R_sum) C0 + W G1,
//     with K_k = (kappa^2 + r_k) C0 + G1, W = sum_k w_k, R_sum = sum_k w_k r_k.
//   Both forms collapse the trace through the same code path; only the
//   matrix used for the log_kappa branch differs.

#include "spde_nc_transform.h"

#include <RcppEigen.h>
#include <stdexcept>
#include <cmath>

namespace tulpa {

void SpdeNcTransform::init(int n,
                           const std::vector<double>& C0_d,
                           const std::vector<double>& G1_x,
                           const std::vector<int>&    G1_i,
                           const std::vector<int>&    G1_p,
                           const std::vector<double>& poles,
                           const std::vector<double>& weights)
{
    n_mesh  = n;
    C0_diag = Eigen::Map<const Eigen::VectorXd>(C0_d.data(), n);
    C0_inv_diag.resize(n);
    orphan_diag = Eigen::VectorXd::Zero(n);
    has_orphans = false;
    for (int i = 0; i < n; i++) {
        C0_inv_diag[i] = spde_c0_inv(C0_diag[i]);
        if (spde_is_orphan_mass(C0_diag[i])) {
            orphan_diag[i] = SPDE_ORPHAN_RIDGE;
            has_orphans = true;
        }
    }
    orphan_ridge = SpMat(n, n);
    if (has_orphans) {
        std::vector<Eigen::Triplet<double>> ridge_trips;
        for (int i = 0; i < n; i++) {
            if (orphan_diag[i] != 0.0) {
                ridge_trips.emplace_back(i, i, orphan_diag[i]);
            }
        }
        orphan_ridge.setFromTriplets(ridge_trips.begin(), ridge_trips.end());
    }
    orphan_ridge.makeCompressed();
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

    // G1 diag(1/C0) G1 -- the top level of the alpha = 2 operator chain, and
    // the only part of Q that carries neither kappa nor tau. Built once here;
    // build_Q and build_Q_rational read it. The inverse mass is the FLOORED
    // one (spde_zero_mass.h), so an orphan drops every path that runs through
    // it out of this level, exactly as SpdeQBuilder's chain does.
    {
        SpMat GD = G1;
        for (int j = 0; j < n; j++) {
            for (SpMat::InnerIterator it(GD, j); it; ++it) {
                it.valueRef() *= C0_inv_diag[j];
            }
        }
        GDG = GD * G1;
        GDG.makeCompressed();
    }

    // Rational coefficients (Bolin et al. 2023). Empty -> integer alpha=2.
    if (poles.size() != weights.size()) {
        throw std::runtime_error("SpdeNcTransform::init: poles / weights "
                                 "length mismatch");
    }
    poles_   = poles;
    weights_ = weights;
    W_     = 0.0;
    R_sum_ = 0.0;
    for (size_t k = 0; k < poles_.size(); k++) {
        W_     += weights_[k];
        R_sum_ += weights_[k] * poles_[k];
    }
}

void SpdeNcTransform::init_fixed(const std::vector<int>&    Q_p,
                                 const std::vector<int>&    Q_i,
                                 const std::vector<double>& Q_x,
                                 int                        n)
{
    n_mesh = n;
    if (static_cast<int>(Q_p.size()) != n + 1) {
        throw std::runtime_error("SpdeNcTransform::init_fixed: Q_p length "
                                 "must be n_mesh + 1");
    }
    last_Q = SpMat(n, n);
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(Q_x.size());
    for (int j = 0; j < n; j++) {
        for (int idx = Q_p[j]; idx < Q_p[j + 1]; idx++) {
            trips.emplace_back(Q_i[idx], j, Q_x[idx]);
        }
    }
    last_Q.setFromTriplets(trips.begin(), trips.end());
    last_Q.makeCompressed();

    llt.compute(last_Q);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("SpdeNcTransform::init_fixed: Cholesky "
                                 "failed (Q not PD)");
    }
    factored   = true;
    fixed_mode = true;
}

Eigen::VectorXd SpdeNcTransform::forward_fixed(const Eigen::VectorXd& z) const
{
    if (!factored) {
        throw std::runtime_error("SpdeNcTransform::forward_fixed without "
                                 "init_fixed");
    }
    if (z.size() != n_mesh) throw std::runtime_error("z size mismatch");
    SpMat L = llt.matrixL();
    return L.transpose().template triangularView<Eigen::Upper>().solve(z);
}

void SpdeNcTransform::backward_fixed(const Eigen::VectorXd& dv,
                                     Eigen::VectorXd&       dz_out) const
{
    if (!factored) {
        throw std::runtime_error("SpdeNcTransform::backward_fixed without "
                                 "init_fixed");
    }
    SpMat L = llt.matrixL();
    dz_out = L.template triangularView<Eigen::Lower>().solve(dv);
}

void SpdeNcTransform::forward_fixed_with_tangent(
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& dz,
    Eigen::VectorXd&       v_out,
    Eigen::VectorXd&       dv_out) const
{
    if (!factored) {
        throw std::runtime_error("SpdeNcTransform::forward_fixed_with_tangent "
                                 "without init_fixed");
    }
    // Fixed (kappa, tau): the map is the constant linear v = L^{-T} z, so the
    // tangent is dv = L^{-T} dz (no M = L^{-1} dQ L^{-T} term).
    SpMat L = llt.matrixL();
    auto U = L.transpose().template triangularView<Eigen::Upper>();
    v_out  = U.solve(z);
    dv_out = U.solve(dz);
}

SpdeNcTransform::SpMat SpdeNcTransform::build_K(double kappa) const {
    return build_K_shifted(kappa * kappa);
}

SpdeNcTransform::SpMat SpdeNcTransform::build_K_shifted(double k2_eff) const {
    SpMat K = G1;
    for (int i = 0; i < n_mesh; i++) {
        K.coeffRef(i, i) += k2_eff * C0_diag[i];
    }
    K.makeCompressed();
    return K;
}

SpdeNcTransform::SpMat SpdeNcTransform::build_Q(double k2_eff, double tau) const {
    // Q = tau^2 (k2^2 C + 2 k2 G + G diag(1/C0) G), the alpha = 2 term of the
    // operator chain, which is what SpdeQBuilder assembles.
    //
    // NOT the product K diag(1/C0) K. The two are the same matrix only where
    // C diag(1/C0) = I, and the inverse mass is FLOORED to zero at a zero-mass
    // (orphan) mesh vertex (spde_zero_mass.h). At such a row the product drops
    // the cross terms k2 (C D G + G D C) entirely, so the non-centered
    // assembly and the Laplace one disagreed on exactly the orphan's row and
    // column, by tau^2 * 2 k2 * G[j, .] -- which is the disagreement the orphan
    // ridge exists to prevent.
    //
    // It also makes the hyper derivatives exact rather than approximately so:
    // d(k2^2)/dlog_kappa = 4 k2^2 and d(2 k2)/dlog_kappa = 4 k2 give
    // dQ/dlog_kappa = 4 kappa^2 tau^2 K with no appeal to C D = I, which is
    // the closed form backward() and forward_with_tangent() already apply and
    // the one implicit_diff.h uses on the Laplace side.
    //
    // G diag(1/C0) G carries no kappa, so init() builds it once.
    SpMat Q = (2.0 * k2_eff) * G1 + GDG;
    for (int i = 0; i < n_mesh; i++) {
        Q.coeffRef(i, i) += k2_eff * k2_eff * C0_diag[i];
    }
    Q *= (tau * tau);
    Q.makeCompressed();
    return Q;
}

SpdeNcTransform::SpMat SpdeNcTransform::build_Q_rational(double kappa, double tau) {
    // Q = tau^2 sum_k w_k Q_k, with Q_k the alpha = 2 expansion build_Q
    // applies at the shifted kappa^2 + r_k.
    const double k2   = kappa * kappa;
    const double tau2 = tau * tau;

    // Each per-pole term is the same expansion at a shifted k^2, and the sum
    // collapses termwise -- only two scalars depend on the poles:
    //   sum_k w_k [ (k2 + r_k)^2 C + 2 (k2 + r_k) G + G D G ]
    //     = (sum_k w_k (k2 + r_k)^2) C + 2 (sum_k w_k (k2 + r_k)) G + W G D G
    const int m = static_cast<int>(poles_.size());
    double c_diag = 0.0, c_g = 0.0;
    for (int k = 0; k < m; k++) {
        const double k2_eff = k2 + poles_[k];
        c_diag += weights_[k] * k2_eff * k2_eff;
        c_g    += weights_[k] * k2_eff;
    }
    SpMat Q = (2.0 * c_g) * G1 + W_ * GDG;
    for (int i = 0; i < n_mesh; i++) {
        Q.coeffRef(i, i) += c_diag * C0_diag[i];
    }
    Q *= tau2;
    Q.makeCompressed();

    // K_sum = sum_k w_k K_k = (k2 W + R_sum) C0 + W G1. Cache here so
    // backward() doesn't have to recompute.
    last_K_sum = W_ * G1;
    const double diag_coef = k2 * W_ + R_sum_;
    for (int i = 0; i < n_mesh; i++) {
        last_K_sum.coeffRef(i, i) += diag_coef * C0_diag[i];
    }
    last_K_sum.makeCompressed();

    return Q;
}

Eigen::VectorXd SpdeNcTransform::forward(const Eigen::VectorXd& z,
                                         double kappa, double tau)
{
    if (z.size() != n_mesh) throw std::runtime_error("z size mismatch");
    last_kappa = kappa;
    last_tau   = tau;
    if (is_rational()) {
        // Rational path also fills last_K_sum for the adjoint.
        last_Q = build_Q_rational(kappa, tau);
        // last_K is unused on this path; leave empty to surface bugs early
        // if a caller reads it.
        last_K = SpMat();
    } else {
        last_K = build_K(kappa);
        last_Q = build_Q(kappa * kappa, tau);
        last_K_sum = SpMat();
    }
    if (has_orphans) {
        last_Q += orphan_ridge;
        last_Q.makeCompressed();
    }
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

    SpMat L  = llt.matrixL();
    SpMat LT = L.transpose();
    Eigen::VectorXd y = L.template triangularView<Eigen::Lower>().solve(dw);
    dz_out = y;

    // Stream -<z, Phi(M) y> with M = L^{-1} dQ_theta L^{-T} by columns of M.
    // Trace splits as -(T_lower + 0.5 D) — see file-header derivation.
    // dQ_theta enters only as the sparse matvec dQ_theta * v_j; the per-
    // theta linear prefactor (e.g. 4 kappa^2 tau^2 for log_kappa) is
    // applied to the scalar trace afterwards rather than scaling K.
    auto trace_term = [&](const SpMat& K_theta) -> double {
        Eigen::VectorXd u =
            LT.template triangularView<Eigen::Upper>().solve(z);

        double T_lower = 0.0;
        double D       = 0.0;
        Eigen::VectorXd e_j = Eigen::VectorXd::Zero(n);

        for (int j = 0; j < n; ++j) {
            e_j[j] = 1.0;
            Eigen::VectorXd v_j =
                LT.template triangularView<Eigen::Upper>().solve(e_j);
            e_j[j] = 0.0;

            // u_j = u_{j-1} - z[j] v_j. Initial u_{-1} = L^{-T} z above.
            u.noalias() -= z[j] * v_j;

            const Eigen::VectorXd Kv = K_theta * v_j;
            T_lower += y[j]            * u.dot(Kv);
            D       += z[j] * y[j]     * v_j.dot(Kv);
        }
        return -(T_lower + 0.5 * D);
    };

    // Integer alpha=2: dQ/dlog_kappa = 4 kappa^2 tau^2 K.
    // Rational:        dQ/dlog_kappa = 4 kappa^2 tau^2 K_sum
    //                  where K_sum was cached on last_K_sum by build_Q_rational.
    // The trace is linear in dQ, so the (4 kappa^2 tau^2) factor multiplies
    // the scalar result instead of the (potentially large) sparse matrix.
    const SpMat& K_for_logkappa = is_rational() ? last_K_sum : last_K;
    const double c_logkappa =
        4.0 * last_kappa * last_kappa * last_tau * last_tau;
    dlog_kappa_out = c_logkappa * trace_term(K_for_logkappa);

    // dQ/dlog_tau = 2 (Q - R) for both paths: the assembled part carries the
    // tau^2, the orphan ridge R does not. Collapsing the assembled half:
    //   M_logtau = L^{-1} (2Q) L^{-T} = 2 L^{-1} L L^T L^{-T} = 2 I.
    // So Phi(M_logtau)[i,j] = M[i,j] for i>j (= 0), 0.5 × 2 = 1 for i=j.
    // Hence -<z, Phi(2I) y> = -<z, I y> = -z · y. No matrix solve needed —
    // a single dot product saves the entire dense O(n^2) trace work. The trace
    // is linear in dQ, so the ridge enters as its own streaming term, and only
    // on a mesh that carries one.
    dlog_tau_out = -z.dot(y);
    if (has_orphans) dlog_tau_out -= 2.0 * trace_term(orphan_ridge);
}

void SpdeNcTransform::forward_with_tangent(
    const Eigen::VectorXd& z,
    const Eigen::VectorXd& dz,
    double                 kappa,
    double                 dlog_kappa,
    double                 tau,
    double                 dlog_tau,
    Eigen::VectorXd&       w_out,
    Eigen::VectorXd&       dw_out)
{
    // Value pass. Populates llt, last_K[_sum], last_Q, last_kappa, last_tau.
    w_out = forward(z, kappa, tau);
    const int n = n_mesh;

    SpMat L  = llt.matrixL();
    SpMat LT = L.transpose();

    // dQ = dlog_kappa * (4 kappa^2 tau^2) K_for_logkappa + dlog_tau * 2 Q.
    // dQ is symmetric and sparse (same pattern as Q). Avoid forming it as
    // a single matrix when the prefactors are trivial — we'll apply dQ via
    // matvec, and matvec on (a*A + b*B) splits cleanly.
    const SpMat& K_for_logkappa = is_rational() ? last_K_sum : last_K;
    const double c_logkappa     = 4.0 * kappa * kappa * tau * tau * dlog_kappa;
    const double c_logtau       = 2.0 * dlog_tau;

    auto dQ_matvec = [&](const Eigen::VectorXd& v) -> Eigen::VectorXd {
        // (c_logkappa * K + c_logtau * (Q - R)) * v, computed without
        // materializing the linear combination. R is the theta-independent
        // orphan ridge, which does not scale with tau.
        Eigen::VectorXd out = c_logkappa * (K_for_logkappa * v);
        if (c_logtau != 0.0) {
            out.noalias() += c_logtau * (last_Q * v);
            if (has_orphans) {
                out -= c_logtau * orphan_diag.cwiseProduct(v);
            }
        }
        return out;
    };

    // Cheap: M z = L^{-1} (dQ (L^{-T} z)) = L^{-1} (dQ w).
    Eigen::VectorXd Mz = dQ_matvec(w_out);
    L.template triangularView<Eigen::Lower>().solveInPlace(Mz);

    // Expensive but matches the adjoint trace: Phi(M) z by streaming.
    //   (Phi(M) z)[i] = sum_{j<i} M[i,j] z[j] + 0.5 M[i,i] z[i]
    // Per column j, M[:, j] = L^{-1} (dQ (L^{-T} e_j)) contributes
    //   c_j[i] * z[j]   to PhiMz[i] for i > j,
    //   0.5 * c_j[j] * z[j]  to PhiMz[j].
    Eigen::VectorXd PhiMz = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd e_j   = Eigen::VectorXd::Zero(n);
    for (int j = 0; j < n; ++j) {
        e_j[j] = 1.0;
        Eigen::VectorXd v_j =
            LT.template triangularView<Eigen::Upper>().solve(e_j);
        e_j[j] = 0.0;

        Eigen::VectorXd c_j = dQ_matvec(v_j);
        L.template triangularView<Eigen::Lower>().solveInPlace(c_j);

        const double zj = z[j];
        if (zj != 0.0) {
            // Strict lower: i > j.
            PhiMz.tail(n - j - 1).noalias() += zj * c_j.tail(n - j - 1);
            // Halved diagonal: i = j.
            PhiMz[j] += 0.5 * zj * c_j[j];
        }
    }

    // Phi(M)^T z = M z - Phi(M) z (symmetric M).
    Eigen::VectorXd rhs = dz;
    rhs.noalias() -= (Mz - PhiMz);

    // dw = L^{-T} (dz - Phi(M)^T z).
    dw_out = LT.template triangularView<Eigen::Upper>().solve(rhs);
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

std::vector<arena::Var> spde_nc_transform_fixed_arena(
    arena::Arena*                  ar,
    const std::vector<arena::Var>& z,
    SpdeNcTransform&               transform
) {
    const int n = static_cast<int>(z.size());

    Eigen::VectorXd z_eig(n);
    for (int i = 0; i < n; i++) z_eig[i] = z[i].val();

    Eigen::VectorXd v_eig = transform.forward_fixed(z_eig);

    std::vector<int32_t> input_indices(n);
    for (int i = 0; i < n; i++) input_indices[i] = z[i].idx_;

    std::vector<double> output_values(v_eig.data(), v_eig.data() + n);

    auto cb = [&transform, n](
        const double* /*input_vals*/,  int /*n_in*/,
        const double* /*output_vals*/, int /*n_out*/,
        const double* output_adjs,
        double*       input_adjs
    ) {
        Eigen::VectorXd dv(n);
        for (int i = 0; i < n; i++) dv[i] = output_adjs[i];
        Eigen::VectorXd dz;
        transform.backward_fixed(dv, dz);
        for (int i = 0; i < n; i++) input_adjs[i] = dz[i];
    };

    std::vector<int32_t> out_indices;
    ar->add_custom_backward(input_indices, output_values, cb, out_indices);

    std::vector<arena::Var> v(n);
    for (int i = 0; i < n; i++) {
        v[i].arena_ = ar;
        v[i].idx_   = out_indices[i];
    }
    return v;
}

} // namespace tulpa
