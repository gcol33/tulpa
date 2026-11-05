#ifndef TULPA_HSGP_OBSERVATION_H
#define TULPA_HSGP_OBSERVATION_H

#include <Eigen/QR>
#include <Eigen/Core>

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
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(at);
        const int rank = std::min(at.rows(), at.cols());
        if (need_rotation)
            rotation = qr.householderQ() * Eigen::MatrixXd::Identity(at.rows(), rank);
        basis = qr.matrixQR().topRows(rank).template triangularView<Eigen::Upper>()
                    .toDenseMatrix().transpose();
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
