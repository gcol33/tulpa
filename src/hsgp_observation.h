#ifndef TULPA_HSGP_OBSERVATION_H
#define TULPA_HSGP_OBSERVATION_H

#include <Eigen/QR>
#include <Eigen/Core>
#include <vector>
#include <utility>

namespace tulpa {

// A = Phi diag(sqrt(S)); A' = Q R. The likelihood depends on Q'z,
// whose prior is N(0,I). The orthogonal complement of z integrates to one.
// At prediction rows B, BQ gives the conditional loading and
// B(I-QQ')B' is the independent conditional covariance.
struct HsgpObservation {
    Eigen::MatrixXd rotation;
    Eigen::MatrixXd basis;

    void compute(const Eigen::MatrixXd& phi, const Eigen::VectorXd& spectral,
                 bool need_rotation = true) {
        Eigen::MatrixXd at = spectral.asDiagonal() * phi.transpose();
        compute_loading(at, need_rotation);
    }

    void compute_loading(const Eigen::MatrixXd& at, bool need_rotation = true) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(at);
        const int rank = std::min(at.rows(), at.cols());
        if (need_rotation)
            rotation = qr.householderQ() * Eigen::MatrixXd::Identity(at.rows(), rank);
        basis = qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>()
                    .toDenseMatrix().transpose();
    }

    void combine(const std::vector<HsgpObservation>& components,
                 const std::vector<double>& scales, bool need_rotation = true) {
        int reduced = 0, original = 0;
        for (const auto& p : components) {
            reduced += p.basis.cols();
            original += p.rotation.rows();
        }
        const int rows = components.front().basis.rows();
        Eigen::MatrixXd at(reduced, rows);
        int offset = 0;
        for (std::size_t b = 0; b < components.size(); ++b) {
            const auto& p = components[b];
            at.middleRows(offset, p.basis.cols()).noalias() =
                scales[b] * p.basis.transpose();
            offset += p.basis.cols();
        }
        compute_loading(at, need_rotation);
        if (need_rotation) {
            Eigen::MatrixXd full(original, rotation.cols());
            int row = 0;
            offset = 0;
            for (const auto& p : components) {
                full.middleRows(row, p.rotation.rows()).noalias() =
                    p.rotation * rotation.middleRows(offset, p.basis.cols());
                row += p.rotation.rows();
                offset += p.basis.cols();
            }
            rotation = std::move(full);
        }
    }

    Eigen::MatrixXd project(const Eigen::MatrixXd& phi,
                            const Eigen::VectorXd& spectral) const {
        return phi * spectral.asDiagonal() * rotation;
    }

    Eigen::MatrixXd residual_basis(const Eigen::MatrixXd& phi,
                                   const Eigen::VectorXd& spectral) const {
        Eigen::MatrixXd b = phi * spectral.asDiagonal();
        // The residual rows avoid cancellation in BB' - (BQ)(BQ)'.
        b -= (b * rotation).eval() * rotation.transpose();
        return b;
    }

    Eigen::VectorXd residual_variance(const Eigen::MatrixXd& phi,
                                      const Eigen::VectorXd& spectral) const {
        Eigen::MatrixXd b = residual_basis(phi, spectral);
        return b.rowwise().squaredNorm();
    }
};

} // namespace tulpa
#endif
