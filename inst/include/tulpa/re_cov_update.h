// re_cov_update.h
// The closed-form covariance update of a grouped-random-effect EM. Given the
// group-averaged posterior second moment of one covariance block,
//   M = mean_g ( E[b_g | y] E[b_g | y]' + Cov(b_g | y) ),
// the update is M itself, symmetrized, projected to its diagonal for an
// uncorrelated block, with its eigenvalues floored at floor_sd^2 so it stays
// positive definite when a variance component collapses. Every EM over a
// grouped random effect takes its update from here (tulpa_re_em(), and the
// consumer packages' own drivers through LinkingTo), so the projection and the
// floor are one rule.

#ifndef TULPA_RE_COV_UPDATE_H
#define TULPA_RE_COV_UPDATE_H

#include <RcppEigen.h>

namespace tulpa {

inline Eigen::MatrixXd re_cov_update(const Eigen::MatrixXd& M, bool full,
                                     double floor_sd) {
    Eigen::MatrixXd S = 0.5 * (M + M.transpose());
    if (!full) S = Eigen::MatrixXd(S.diagonal().asDiagonal());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(S);
    const Eigen::VectorXd ev = es.eigenvalues().cwiseMax(floor_sd * floor_sd);
    return es.eigenvectors() * ev.asDiagonal() * es.eigenvectors().transpose();
}

} // namespace tulpa

#endif // TULPA_RE_COV_UPDATE_H
