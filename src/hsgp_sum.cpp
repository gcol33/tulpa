#include <RcppEigen.h>
#include "hsgp_sum.h"

namespace tulpa {

LatentBlock make_hsgp_sum_block(int start, const Rcpp::List& specs,
    const std::vector<int>& indices, const Rcpp::IntegerVector& axis_offsets,
    const Rcpp::NumericMatrix& theta_grid, const Rcpp::IntegerVector& n_obs) {
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    struct Component {
        Eigen::MatrixXd training;
        std::vector<Eigen::MatrixXd> phi;
        Eigen::VectorXd eigenvalues;
        priors::HsgpKernel kernel;
        int axis;
    };
    auto parts = std::make_shared<std::vector<Component>>();
    bool at_training = true;
    int width = 0, n_training = 0, n_arms = 0;
    for (int b : indices) {
        Rcpp::List p = specs[b];
        Rcpp::List phi = p.containsElementNamed("phi")
            ? Rcpp::as<Rcpp::List>(p["phi"])
            : Rcpp::List::create(p["phi_basis"]);
        if (phi.size() != n_obs.size())
            Rcpp::stop("HSGP phi must contain one basis matrix per response arm.");
        const bool separate = p.containsElementNamed("training_phi");
        Rcpp::List train = separate ? Rcpp::as<Rcpp::List>(p["training_phi"]) : phi;
        if (!parts->empty() && phi.size() != parts->front().phi.size())
            Rcpp::stop("HSGP components must have the same response arms.");
        at_training = at_training && !separate;
        Component part;
        part.axis = axis_offsets[b];
        if (axis_offsets[b + 1] - part.axis != 2)
            Rcpp::stop("HSGP components require sigma2 and lengthscale axes.");
        nl_grid_axis_positive("hsgp sigma2 axis", theta_grid, part.axis);
        nl_grid_axis_positive("hsgp lengthscale axis", theta_grid, part.axis + 1);
        part.kernel = priors::hsgp_kernel_from_spec(p, "hsgp block");
        Rcpp::NumericVector eig = p.containsElementNamed("eigenvalues")
            ? Rcpp::as<Rcpp::NumericVector>(p["eigenvalues"])
            : Rcpp::as<Rcpp::NumericVector>(p["lambda_eig"]);
        part.eigenvalues = Eigen::Map<const Eigen::VectorXd>(eig.begin(), eig.size());
        if (p.containsElementNamed("m_total") && Rcpp::as<int>(p["m_total"]) != eig.size())
            Rcpp::stop("HSGP m_total differs from its eigenvalue count.");
        n_arms = phi.size();
        n_training = 0;
        for (int k = 0; k < train.size(); ++k)
            n_training += Rcpp::as<Rcpp::NumericMatrix>(train[k]).nrow();
        part.training.resize(n_training, eig.size());
        int row = 0;
        for (int k = 0; k < train.size(); ++k) {
            Rcpp::NumericMatrix mat = train[k];
            if (mat.ncol() != eig.size()) Rcpp::stop("HSGP training basis width differs from eigenvalues.");
            part.training.middleRows(row, mat.nrow()) =
                Eigen::Map<const Eigen::MatrixXd>(mat.begin(), mat.nrow(), mat.ncol());
            row += mat.nrow();
        }
        for (int k = 0; k < n_arms; ++k) {
            Rcpp::NumericMatrix mat = phi[k];
            if (mat.nrow() != n_obs[k])
                Rcpp::stop("HSGP basis rows must match the response arm's observations.");
            if (mat.ncol() != eig.size()) Rcpp::stop("HSGP prediction basis width differs from eigenvalues.");
            if (!parts->empty() && mat.nrow() != parts->front().phi[k].rows())
                Rcpp::stop("HSGP components must have matching rows within each response arm.");
            if (p.containsElementNamed("n_obs_per_arm")) {
                Rcpp::IntegerVector n_obs = p["n_obs_per_arm"];
                if (n_obs.size() != n_arms || n_obs[k] != mat.nrow())
                    Rcpp::stop("HSGP n_obs_per_arm differs from its basis rows.");
            }
            part.phi.emplace_back(Eigen::Map<const Eigen::MatrixXd>(mat.begin(), mat.nrow(), mat.ncol()));
        }
        if (p.containsElementNamed("svc_weight") && !Rf_isNull(p["svc_weight"])) {
            Rcpp::NumericVector w = p["svc_weight"];
            if (n_arms != 1 || w.size() != part.phi[0].rows())
                Rcpp::stop("HSGP svc_weight must contain one weight per observation.");
            part.phi[0].array().colwise() *= Eigen::Map<const Eigen::VectorXd>(w.begin(), w.size()).array();
            if (!separate) part.training = part.phi[0];
        }
        width += eig.size();
        parts->push_back(std::move(part));
    }
    struct Cell {
        std::vector<double> theta;
        std::vector<RowMatrix> basis;
        std::vector<Eigen::VectorXd> residual;
    };
    auto cache = std::make_shared<NlCellCache<Cell>>();
    LatentBlock block;
    block.start = start;
    block.size = n_training;
    block.contrib_kind = BlockContribKind::DENSE_BASIS;
    block.prior_kind = PriorFillKind::DIAGONAL_LOWRANK;
    block.d_fac = [](int) { return 1.0; };
    block.prep = [parts, cache, theta_grid, n_training, width, n_arms, at_training](int cell) {
        auto& out = cache->claim();
        std::vector<double> theta;
        for (const auto& p : *parts) {
            theta.push_back(theta_grid(cell, p.axis));
            theta.push_back(theta_grid(cell, p.axis + 1));
        }
        if (theta == out.theta) {
            cache->publish(cell);
            return true;
        }
        Eigen::MatrixXd at(width, n_training);
        std::vector<Eigen::VectorXd> spectral;
        int col = 0;
        for (const auto& p : *parts) {
            const double sigma2 = theta_grid(cell, p.axis), ell = theta_grid(cell, p.axis + 1);
            if (!(sigma2 > 0) || !(ell > 0)) return false;
            spectral.emplace_back(p.eigenvalues.size());
            priors::hsgp_sqrt_spectral_2d(p.kernel, sigma2, ell, p.eigenvalues.data(),
                                        p.eigenvalues.size(), spectral.back().data());
            at.middleRows(col, p.eigenvalues.size()).noalias() =
                spectral.back().asDiagonal() * p.training.transpose();
            col += p.eigenvalues.size();
        }
        HsgpObservation representation;
        representation.compute_loading(at, !at_training);
        out.basis.resize(n_arms);
        out.residual.resize(n_arms);
        int row = 0;
        for (int k = 0; k < n_arms; ++k) {
            const int n = (*parts)[0].phi[k].rows();
            if (at_training) {
                out.basis[k] = representation.basis.middleRows(row, n);
                out.residual[k].setZero(n);
                row += n;
            } else {
                Eigen::MatrixXd loading(n, width);
                int offset = 0;
                for (std::size_t b = 0; b < parts->size(); ++b) {
                    const auto& p = (*parts)[b];
                    loading.middleCols(offset, p.eigenvalues.size()).noalias() =
                        p.phi[k] * spectral[b].asDiagonal();
                    offset += p.eigenvalues.size();
                }
                out.basis[k].noalias() = loading * representation.rotation;
                loading -= out.basis[k] * representation.rotation.transpose();
                out.residual[k] = loading.rowwise().squaredNorm();
            }
        }
        out.theta = std::move(theta);
        cache->publish(cell);
        return true;
    };
    block.basis_eval = [cache, n_training](int i, int arm, int cell, double* out) {
        const auto& basis = cache->find(cell).basis[arm];
        std::copy(basis.data() + i * n_training, basis.data() + (i + 1) * n_training, out);
    };
    block.dense_basis_batch = [cache, n_training](int arm, int cell) {
        const auto& basis = cache->find(cell).basis[arm];
        LatentBlock::DenseBasisBatch out;
        out.data = basis.data();
        out.sqrt_S = nullptr;
        out.N_k = basis.rows();
        out.m_per_arm = n_training;
        out.m_offset_in_block = 0;
        return out;
    };
    set_hsgp_standard_normal_prior(block);
    if (!at_training) block.prediction_residual = [cache](int arm, int cell, int i) {
        return cache->find(cell).residual[arm](i);
    };
    return block;
}

} // namespace tulpa
