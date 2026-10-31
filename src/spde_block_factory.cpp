// spde_block_factory.cpp
// Definitions of the SPDE latent-block builders declared in spde_block_factory.h.

#include <RcppEigen.h>
#include "spde_block_factory.h"

namespace tulpa {

LatentBlock spde_assemble_block(
    int                                          start,
    int                                          n_mesh,
    std::shared_ptr<std::vector<ARows>>          a_rows_per_arm,
    std::shared_ptr<SpdeQBuilder>                qb,
    std::function<bool(int)>                     cell_in_domain,
    std::function<bool(int, SpdeCellState&)>     refresh,
    bool                                         center_field
) {
    auto cell_cache = std::make_shared<NlCellCache<SpdeCellState>>(
        [qb](SpdeCellState& st) { st.qb = *qb; });

    LatentBlock block;
    block.start = start;
    block.size  = n_mesh;
    block.contrib_kind = BlockContribKind::INDEXED_MULTI;
    block.prior_kind   = PriorFillKind::SPDE_Q;
    block.d_fac        = [](int) -> double { return 1.0; };
    // arm_scale left empty — copy not supported.

    // Per-obs (mesh_node_1based, weight) list. Pattern builder + sparse
    // scatter consume this for INDEXED_MULTI fill.
    block.obs_indices = make_projector_obs_indices(a_rows_per_arm);

    // idx left empty — INDEXED_MULTI uses obs_indices, not idx.

    block.prep = [cell_cache, cell_in_domain, refresh](int k_grid) -> bool {
        if (cell_in_domain && !cell_in_domain(k_grid)) return false;
        auto& st = cell_cache->claim();
        if (!refresh(k_grid, st)) return false;
        cell_cache->publish(k_grid);
        return true;
    };

    // Pattern: every Q nonzero contributes a lower-triangle entry in the
    // joint H pattern.
    block.add_prior_pattern = [start, n_mesh, qb](
        std::vector<std::pair<int,int>>& out
    ) {
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = qb->Q_p[col]; idx < qb->Q_p[col + 1]; idx++) {
                int row = qb->Q_i[idx];
                if (row >= col) {
                    out.emplace_back(start + row, start + col);
                }
            }
        }
    };

    // Sparse Q scatter. Mirrors the inline prior-Q block in
    // cpp_nested_laplace_spde (spde_laplace.cpp lines 402-410): gradient
    // uses the FULL Q (both triangles); H uses LOWER triangle only.
    block.add_prior_sparse = [start, n_mesh, cell_cache](
        SparseHessianBuilder& H, DenseVec& grad,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        const SpdeQBuilder& q_cell = cell_cache->find(k_grid).qb;
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                int row = q_cell.Q_i[idx];
                double q = q_cell.Q_x[idx];
                grad[start + row] -= q * x[start + col];
                if (row >= col) {
                    H.add(start + row, start + col, q);
                }
            }
        }
    };

    // Dense add_prior: the same FEM precision Q scattered into a DenseMat for
    // the dense Newton path (n_x < SPARSE_THRESHOLD), used by the non-joint
    // multi-block driver. Mirrors add_prior_sparse exactly -- gradient walks
    // the FULL Q (both triangles via the CSC), Hessian fills both triangles so
    // the caller's lower->upper symmetrise does not clobber it. Reads the same
    // qb->Q_x rebuilt by prep(), so there is one source of FEM assembly.
    block.add_prior = [start, n_mesh, cell_cache](
        DenseVec& grad, DenseMat& H,
        const Rcpp::NumericVector& x, int k_grid
    ) {
        const SpdeQBuilder& q_cell = cell_cache->find(k_grid).qb;
        for (int col = 0; col < n_mesh; col++) {
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                int row = q_cell.Q_i[idx];
                double q = q_cell.Q_x[idx];
                grad[start + row] -= q * x[start + col];
                H[start + row][start + col] += q;
            }
        }
    };

    block.log_prior = [start, n_mesh, cell_cache](
        const Rcpp::NumericVector& x, int k_grid
    ) -> double {
        const auto& st = cell_cache->find(k_grid);
        const SpdeQBuilder& q_cell = st.qb;
        double qf = 0.0;
        for (int col = 0; col < n_mesh; col++) {
            double x_col = x[start + col];
            for (int idx = q_cell.Q_p[col]; idx < q_cell.Q_p[col + 1]; idx++) {
                qf += x[start + q_cell.Q_i[idx]] * q_cell.Q_x[idx] * x_col;
            }
        }
        // log p(x|theta) = 0.5 log|Q(theta)| - 0.5 x'Qx (the theta-independent
        // -(n/2) log(2 pi) is dropped). 0.5 log|Q| is the prior normalizer set
        // by prep() each cell; it is the Occam term that makes the (range,sigma)
        // marginal interior-peaked rather than monotone in sigma. It is NOT
        // absorbed by the Laplace Hessian log-determinant (H = Q + A'WA != Q).
        return st.half_ldQ - 0.5 * qf;
    };

    // Sum-to-zero centring for intercept identifiability. Left empty for the
    // precomputed rational latent, whose auxiliary weights x are not the field
    // (u = Pr x) and whose proper prior (kappa^2 > 0) already identifies the
    // constant mode.
    if (center_field) {
        block.center = [start, n_mesh](Rcpp::NumericVector& x) {
            return tulpa::center_intercept(x, start, n_mesh);
        };
    }

    return block;
}

LatentBlock make_spde_block(
    int                            start,
    int                            n_mesh,
    const Rcpp::List&              A_x_per_arm,
    const Rcpp::List&              A_i_per_arm,
    const Rcpp::List&              A_p_per_arm,
    const Rcpp::IntegerVector&     n_obs_per_arm,
    int                            n_arms,
    int                            block_index,
    const Rcpp::NumericVector&     C0_diag,
    const Rcpp::NumericVector&     G1_x,
    const Rcpp::IntegerVector&     G1_i,
    const Rcpp::IntegerVector&     G1_p,
    double                         nu,
    int                            axis_range,
    int                            axis_sigma,
    const Rcpp::NumericMatrix&     theta_grid,
    bool                           use_rational,
    const std::vector<double>&     rational_poles,
    const std::vector<double>&     rational_weights,



    bool                           direct_kappa_tau,


    int*                           q_nnz_out) {
    auto a_rows_per_arm = build_projector_rows_per_arm(
        A_x_per_arm, A_i_per_arm, A_p_per_arm, n_obs_per_arm,
        n_arms, n_mesh, block_index, "spde");

    spde_validate_fem(n_mesh, C0_diag, G1_x, G1_i, G1_p);

    // alpha = nu + d/2 with d = 2.
    const int alpha = static_cast<int>(std::round(nu)) + 1;

    // Template QBuilder — pattern built once at factory time. It seeds the
    // per-cell slot copies and serves the (immutable) pattern reads in
    // add_prior_pattern. The chain is built for the requested operator order,
    // so the pattern widens with alpha; the rational assembly shifts the
    // alpha = 2 stencil regardless of the (fractional) nu it approximates.
    auto qb = std::make_shared<SpdeQBuilder>();
    qb->init(n_mesh, C0_diag, G1_x, G1_i, G1_p, use_rational ? 2 : alpha);
    if (q_nnz_out) *q_nnz_out = qb->nnz();

    auto in_domain = [axis_range, axis_sigma, theta_grid](int k_grid) -> bool {
        return theta_grid(k_grid, axis_range) > 0.0 &&
               theta_grid(k_grid, axis_sigma) > 0.0;
    };

    // Per outer-grid cell: rebuild Q values for the cell's (range, sigma) and
    // recompute the prior normalizer 0.5 log|Q(theta)| consumed by log_prior.
    auto refresh = [axis_range, axis_sigma, theta_grid, nu, use_rational,
                    direct_kappa_tau, rational_poles, rational_weights](
        int k_grid, SpdeCellState& st
    ) -> bool {
        const double a0 = theta_grid(k_grid, axis_range);
        const double a1 = theta_grid(k_grid, axis_sigma);
        double kappa, tau;
        if (direct_kappa_tau) {
            kappa = a0;
            tau   = a1;
        } else {
            std::tie(kappa, tau) = spde_range_sigma_to_kappa_tau(a0, a1, nu);
        }
        if (use_rational) {
            st.qb.rebuild_rational(kappa, tau, rational_poles, rational_weights);
        } else {
            st.qb.rebuild(kappa, tau);
        }
        return st.qld.half_logdet(st.qb, st.half_ldQ);
    };

    return spde_assemble_block(start, n_mesh, a_rows_per_arm, qb,
                               in_domain, refresh, /*center_field=*/true);
}

} // namespace tulpa
