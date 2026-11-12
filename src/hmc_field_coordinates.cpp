#include <RcppEigen.h>
#include "hmc_field_coordinates.h"
#include "hmc_gp.h"
#include "log_post_impl.h"
#include "nngp_cond.h"
#include "hmc_spatial_marginal.h"
#include "laplace_newton_loop.h"
#include "joint_pd_step.h"
#include "coordinate_linalg.h"
#include <memory>
#include <random>
#include <type_traits>

namespace tulpa_hmc {
namespace spatial_coordinates {

struct InvalidPrecision {};

struct PriorRoot {
    Eigen::MatrixXd root, inverse, centered;
    double log_determinant;
};

static PriorRoot areal_root(const ModelData& data, bool bym2) {
    const int n = data.n_spatial_units;
    Eigen::MatrixXd Q(n, n), P(n, n);
    std::vector<double> basis(n, 0.0), centered;
    const double* weights = bym2 && !data.bym2_node_prec.empty() ? data.bym2_node_prec.data() : nullptr;
    const auto quadratic = [&]() {
        return tulpa::priors::icar_quad_form<double>(basis.data(), data, nullptr, weights)
            + tulpa::priors::icar_sum_to_zero_augment(basis.data(), data, weights);
    };
    for (int i = 0; i < n; ++i) {
        basis[i] = 1.0;
        Q(i, i) = quadratic();
        tulpa::priors::icar_center_field(basis.data(), data, centered);
        for (int j = 0; j < n; ++j) P(j, i) = centered[j];
        basis[i] = 0.0;
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) {
        basis[i] = basis[j] = 1.0;
        Q(i, j) = Q(j, i) = .5 * (quadratic() - Q(i, i) - Q(j, j));
        basis[i] = basis[j] = 0.0;
    }
    Eigen::LLT<Eigen::MatrixXd> factor(Q);
    if (factor.info() != Eigen::Success) Rcpp::stop("Augmented areal precision must be positive definite");
    const Eigen::MatrixXd L = factor.matrixL();
    PriorRoot out;
    out.root = factor.matrixU().solve(Eigen::MatrixXd::Identity(n, n));
    out.inverse = L.transpose();
    out.centered = P * out.root;
    out.log_determinant = -L.diagonal().array().log().sum();
    return out;
}

struct NngpRoot {
    std::shared_ptr<tulpa_gp::NNGPNCWorkspace> workspace;
    std::vector<double> value;
};

static NngpRoot nngp_root(double log_variance, double log_range, const ModelData& data) {
    const auto view = tulpa_gp::make_gp_nc_view(data.gp_data);
    const int n = view.n_obs;
    NngpRoot out{std::make_shared<tulpa_gp::NNGPNCWorkspace>(), std::vector<double>(n * n, 0.0)};
    std::vector<double> zero(n, 0.0);
    tulpa_gp::nngp_nc_forward(zero.data(), std::exp(log_variance), std::exp(log_range), view, *out.workspace);
    const auto& ws = *out.workspace;
    for (int row = 0; row < n; ++row) {
        const int node = view.nn_order[row];
        out.value[node * n + node] = ws.sqrt_d[row];
        for (int k = 0; k < ws.B_n_nb[row]; ++k) {
            const int neighbor = ws.nb_idx_flat[row * view.nn + k];
            for (int j = 0; j < n; ++j)
                out.value[node * n + j] += ws.B_flat[row * view.nn + k] * out.value[neighbor * n + j];
        }
    }
    return out;
}

static std::vector<tulpa::arena::Var> nngp_root(
    const tulpa::arena::Var& log_variance, const tulpa::arena::Var& log_range, const ModelData& data) {
    const double variance = std::exp(log_variance.val()), range = std::exp(log_range.val());
    const auto view = tulpa_gp::make_gp_nc_view(data.gp_data);
    const int n = view.n_obs;
    auto root = nngp_root(log_variance.val(), log_range.val(), data);
    std::vector<double> variance_tangent(n * n, 0.0), range_tangent(n * n, 0.0);
    tulpa_gp::NNGPRowDerivatives derivatives(view.nn);
    const auto& ws = *root.workspace;
    for (int row = 0; row < n; ++row) {
        tulpa_gp::nngp_row_derivatives(row, variance, range, view, ws, derivatives);
        const int node = view.nn_order[row];
        variance_tangent[node * n + node] =
            tulpa_gp::nngp_conditional_log_variance_derivative(row, view, ws) / (2.0 * ws.sqrt_d[row]);
        range_tangent[node * n + node] = derivatives.d_variance_phi * range / (2.0 * ws.sqrt_d[row]);
        for (int j = 0; j < ws.B_n_nb[row]; ++j) {
            const int neighbor = ws.nb_idx_flat[row * view.nn + j];
            const double weight = ws.B_flat[row * view.nn + j];
            for (int column = 0; column < n; ++column) {
                variance_tangent[node * n + column] += weight * variance_tangent[neighbor * n + column]
                    + derivatives.d_alpha_sigma(j) * root.value[neighbor * n + column];
                range_tangent[node * n + column] += weight * range_tangent[neighbor * n + column]
                    + range * derivatives.d_alpha_phi(j) * root.value[neighbor * n + column];
            }
        }
    }
    auto backward = [variance_tangent, range_tangent](const double*, int, const double*, int,
        const double* adjoints, double* result) {
        for (size_t i = 0; i < variance_tangent.size(); ++i) {
            result[0] += adjoints[i] * variance_tangent[i];
            result[1] += adjoints[i] * range_tangent[i];
        }
    };
    std::vector<int32_t> indices;
    auto* arena = log_variance.arena_;
    arena->add_custom_backward({log_variance.idx_, log_range.idx_}, root.value, backward, indices);
    std::vector<tulpa::arena::Var> out(n * n);
    for (int i = 0; i < n * n; ++i) { out[i].arena_ = arena; out[i].idx_ = indices[i]; }
    return out;
}

template<class T> struct SpatialBasis {
    std::vector<T> native, field;
    T jacobian;
};

template<class T> static SpatialBasis<T> spatial_basis(const std::vector<T>& q,
    const ModelData& data, const ParamLayout& layout, const PriorRoot& areal, int n, int d) {
    using tulpa::math::safe_exp;
    using tulpa::math::safe_sqrt;
    SpatialBasis<T> out{std::vector<T>(d * d, T(0)), std::vector<T>(n * d, T(0)),
        T(layout.is_gp ? 0 : areal.log_determinant)};
    if (layout.is_gp) {
        if constexpr (std::is_same_v<T, double>) out.field = nngp_root(q[layout.log_sigma2_gp_idx], q[layout.log_phi_gp_idx], data).value;
        else out.field = nngp_root(q[layout.log_sigma2_gp_idx], q[layout.log_phi_gp_idx], data);
        for (int i = 0; i < d; ++i) out.native[i * d + i] = T(1);
    } else {
        const T amplitude = layout.is_bym2 ? T(1) : safe_exp(T(-.5) * q[layout.log_tau_spatial_idx]);
        const T rho = layout.is_bym2 ? tulpa::math::inv_logit(q[layout.logit_rho_bym2_idx]) : T(1);
        const T sigma = layout.is_bym2 ? safe_exp(q[layout.log_sigma_bym2_idx]) : T(1);
        const T structured = layout.is_bym2 ? sigma * safe_sqrt(rho) * T(data.bym2_scale_factor) : T(1);
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            out.native[i * d + j] = amplitude * T(areal.root(i, j));
            out.field[i * d + j] = structured * amplitude * T(areal.centered(i, j));
        }
        if (layout.is_bym2) {
            const T unstructured = sigma * safe_sqrt(T(1) - rho);
            for (int i = 0; i < n; ++i) {
                out.native[(n + i) * d + n + i] = T(1);
                out.field[i * d + n + i] = unstructured;
            }
        } else out.jacobian = out.jacobian - T(.5 * n) * q[layout.log_tau_spatial_idx];
    }
    return out;
}

struct Bym2Rotation {
    int spatial_start, theta_start, rho_index, sigma_index, n;
    std::vector<double> phi_factor, phi_inverse, theta_factor, singular_values, information;
    double log_jacobian;

    struct Rotation { double u, v, log_scale, sigma_score, rho_score; };
    Rotation rotation(int j, const std::vector<double>& position) const {
        const double eta = position[rho_index];
        const double log_d = singular_values[j] > 0.0 ? std::log(singular_values[j]) : -INFINITY;
        const double r = log_d + 0.5 * eta;
        const double a = std::exp(-std::abs(r));
        const double norm = std::sqrt(1.0 + a * a);
        const double u = r >= 0.0 ? 1.0 / norm : a / norm;
        const double v = r >= 0.0 ? a / norm : 1.0 / norm;
        const double soft = std::log1p(std::exp(-std::abs(eta)));
        const double log_rho = -std::max(-eta, 0.0) - soft;
        const double log_complement = -std::max(eta, 0.0) - soft;
        const double structured = log_rho + 2.0 * log_d;
        const double log_variance = 2.0 * position[sigma_index] + std::max(structured, log_complement)
            + std::log1p(std::exp(-std::abs(structured - log_complement)));
        const double log_reference = std::max(2.0 * log_d, 0.0)
            + std::log1p(std::exp(-std::abs(2.0 * log_d))) - std::log(2.0);
        const double log_information = information[j] > 0.0
            ? std::log(information[j]) + log_variance - log_reference : -INFINITY;
        const double log_denominator = std::max(log_information, 0.0)
            + std::log1p(std::exp(-std::abs(log_information)));
        const double fraction = std::exp(log_information - log_denominator);
        return {u, v, 0.5 * (std::log1p(information[j]) - log_denominator),
            -fraction, -0.5 * fraction * (u * u - std::exp(log_rho))};
    }

    double forward(const std::vector<double>& z, std::vector<double>& x) const {
        std::vector<double> s(n), t(n);
        double jacobian = log_jacobian;
        for (int j = 0; j < n; ++j) {
            const auto r = rotation(j, z);
            const double effect = std::exp(r.log_scale) * z[spatial_start + j];
            s[j] = r.u * effect - r.v * z[theta_start + j];
            t[j] = r.v * effect + r.u * z[theta_start + j];
            jacobian += r.log_scale;
        }
        for (int i = 0; i < n; ++i) {
            x[spatial_start + i] = x[theta_start + i] = 0.0;
            for (int j = 0; j < n; ++j) {
                x[spatial_start + i] += phi_factor[i * n + j] * s[j];
                x[theta_start + i] += theta_factor[i * n + j] * t[j];
            }
        }
        return jacobian;
    }

    void inverse(std::vector<double>& x) const {
        std::vector<double> s(n, 0.0), t(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                s[j] += phi_inverse[j * n + i] * x[spatial_start + i];
                t[j] += theta_factor[i * n + j] * x[theta_start + i];
            }
        for (int j = 0; j < n; ++j) {
            const auto r = rotation(j, x);
            x[spatial_start + j] = std::exp(-r.log_scale) * (r.u * s[j] + r.v * t[j]);
            x[theta_start + j] = -r.v * s[j] + r.u * t[j];
        }
    }

    void pullback(const std::vector<double>& z, std::vector<double>& gradient) const {
        std::vector<double> s(n, 0.0), t(n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                s[j] += phi_factor[i * n + j] * gradient[spatial_start + i];
                t[j] += theta_factor[i * n + j] * gradient[theta_start + i];
            }
        for (int j = 0; j < n; ++j) {
            const auto r = rotation(j, z);
            const double scale = std::exp(r.log_scale);
            const double effect = scale * z[spatial_start + j];
            const double effect_gradient = r.u * s[j] + r.v * t[j];
            const double du = 0.5 * r.u * r.v * r.v;
            const double dv = -0.5 * r.v * r.u * r.u;
            gradient[rho_index] += s[j] * (du * effect - dv * z[theta_start + j])
                + t[j] * (dv * effect + du * z[theta_start + j])
                + r.rho_score * (effect * effect_gradient + 1.0);
            gradient[sigma_index] += r.sigma_score * (effect * effect_gradient + 1.0);
            gradient[spatial_start + j] = scale * effect_gradient;
            gradient[theta_start + j] = -r.v * s[j] + r.u * t[j];
        }
    }
};

struct SpatialWorkingLikelihood {
    int n;
    std::vector<int> beta;
    std::vector<double> information, score, cross;

    SpatialWorkingLikelihood(const ModelData& model, const ParamLayout& layout)
        : n(layout.is_gp ? model.gp_data.n_obs : model.n_spatial_units),
          information(n * n, 0.0), score(n, 0.0) {
        for (int k = 0; k < model.n_processes; ++k)
            for (int j = 0; j < layout.process_beta_count[k]; ++j)
                beta.push_back(layout.process_beta_start[k] + j);
        cross.assign(n * beta.size(), 0.0);
        const int rank = model.spatial_restriction.rank;
        Eigen::MatrixXd U = Eigen::MatrixXd::Zero(n, rank), V = U;
        Eigen::MatrixXd QWQ = Eigen::MatrixXd::Zero(rank, rank);
        Eigen::VectorXd Qr = Eigen::VectorXd::Zero(rank);
        Eigen::MatrixXd QB = Eigen::MatrixXd::Zero(rank, beta.size());
        const auto& spec = *static_cast<const tulpa::LikelihoodSpec*>(model.likelihood_spec);
        if (!spec.working_eta.empty()) {
            if (spec.working_eta.size() != static_cast<size_t>(model.n_processes))
                Rcpp::stop("The likelihood working point must have one eta per process");
            for (double value : spec.working_eta)
                if (!std::isfinite(value)) Rcpp::stop("The likelihood working point must be finite");
        }
        std::vector<double> reference(layout.total_params, 0.0);
        tulpa::GenericLogPostState<double> state;
        tulpa::initialize_generic_state(reference, model, layout, state);
        tulpa::precompute_generic_fixed_eta(model, state);
        for (int i = 0; i < model.N; ++i) {
            double base[tulpa::MAX_PROCESSES], anchor[tulpa::MAX_PROCESSES], zi, oi;
            tulpa::generic_eta_at(i, model, layout, state, base);
            double anchor_scale = 1;
            for (int k = 0; k < model.n_processes; ++k) {
                anchor[k] = spec.working_eta.empty() ? base[k] : spec.working_eta[k];
                anchor_scale = std::max(anchor_scale, 1 + std::abs(anchor[k]));
            }
            tulpa::generic_zi_oi_logits(i, model, layout, state, zi, oi);
            const auto evaluate = [&](double a, int process, double b) {
                double eta[tulpa::MAX_PROCESSES];
                for (int k = 0; k < model.n_processes; ++k)
                    eta[k] = anchor[k] + (model.sharing.spatial[k] ? a : 0.0) + (k == process ? b : 0.0);
                return spec.ll_double(i, eta, zi, oi, reference, model, layout, model.model_response_data);
            };
            const double value = evaluate(0, -1, 0);
            const double h = std::pow(std::numeric_limits<double>::epsilon(), .25) *
                (spec.working_eta.empty() ? std::sqrt(1 + std::abs(value)) : anchor_scale);
            const double upper = evaluate(h, -1, 0), lower = evaluate(-h, -1, 0);
            double r = (upper - lower) / (2 * h);
            const double w = std::max(0.0, -(upper - 2 * value + lower) / (h * h));
            int column = 0;
            std::vector<double> row_cross(beta.size(), 0.0);
            for (int k = 0; k < model.n_processes; ++k) {
                const double coupling = -(evaluate(h, k, h) - evaluate(h, k, -h) -
                    evaluate(-h, k, h) + evaluate(-h, k, -h)) / (4 * h * h);
                if (!spec.working_eta.empty()) r += coupling * (anchor[k] - base[k]);
                if (!std::isfinite(coupling)) Rcpp::stop("Gaussian field coordinates require finite reference likelihood curvature");
                for (int j = 0; j < layout.process_beta_count[k]; ++j)
                    row_cross[column++] = coupling * model.processes[k].X_flat[i * layout.process_beta_count[k] + j];
            }
            if (!std::isfinite(r) || !std::isfinite(w)) Rcpp::stop("Gaussian field coordinates require finite reference likelihood curvature");
            const int unit = layout.is_gp ? model.gp_data.obs_to_loc[i] : model.spatial_group[i] - 1;
            score[unit] += r;
            information[unit * n + unit] += w;
            for (size_t b = 0; b < beta.size(); ++b) cross[unit * beta.size() + b] += row_cross[b];
            for (int a = 0; a < rank; ++a) {
                const double q = model.spatial_restriction.basis[i * rank + a];
                U(unit, a) += q; V(unit, a) += w * q; Qr[a] += q * r;
                for (int b = 0; b < rank; ++b)
                    QWQ(a, b) += q * w * model.spatial_restriction.basis[i * rank + b];
                for (size_t b = 0; b < beta.size(); ++b) QB(a, b) += q * row_cross[b];
            }
        }
        if (rank > 0) {
            using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            Eigen::Map<RowMatrix> precision(information.data(), n, n);
            Eigen::Map<RowMatrix> coupling(cross.data(), n, beta.size());
            Eigen::Map<Eigen::VectorXd> rhs(score.data(), n);
            precision.noalias() += U * QWQ * U.transpose();
            precision.noalias() -= V * U.transpose();
            precision.noalias() -= U * V.transpose();
            rhs.noalias() -= U * Qr;
            coupling.noalias() -= U * QB;
        }
    }
};

struct GaussianConditional {
    const ModelData* data;
    ParamLayout layout;
    int n, d, begin;
    PriorRoot areal;
    SpatialWorkingLikelihood working;

    GaussianConditional(const ModelData& model, const ParamLayout& positions)
        : data(&model), layout(positions), n(layout.is_gp ? model.gp_data.n_obs : model.n_spatial_units),
          d(layout.is_bym2 ? 2 * n : n), begin(layout.is_gp ? layout.gp_w_start : layout.spatial_start),
          working(model, positions) {
        if (!layout.is_gp) areal = areal_root(model, layout.is_bym2);
    }

    template<class T> struct Geometry {
        std::vector<T> root, chol, mean;
        T jacobian;
    };

    template<class T> Geometry<T> geometry(const std::vector<T>& q) const {
        using tulpa::math::safe_exp;
        using tulpa::math::safe_log;
        using tulpa::math::safe_sqrt;
        Geometry<T> out;
        auto basis = spatial_basis(q, *data, layout, areal, n, d);
        out.root = std::move(basis.native);
        out.jacobian = basis.jacobian;
        const auto& F = basis.field;
        std::vector<T> information(working.information.begin(), working.information.end());
        const auto weighted = tulpa_coordinate_linalg::product(information, F, n, n, d);
        auto H = tulpa_coordinate_linalg::product(F, weighted, d, n, d, true);
        std::vector<T> rhs(d, T(0)), b(n);
        for (int i = 0; i < n; ++i) {
            b[i] = T(working.score[i]);
            for (size_t j = 0; j < working.beta.size(); ++j)
                b[i] = b[i] - T(working.cross[i * working.beta.size() + j]) * q[working.beta[j]];
        }
        for (int i = 0; i < d; ++i) {
            for (int k = 0; k < n; ++k) rhs[i] = rhs[i] + F[k * d + i] * b[k];
            H[i * d + i] = H[i * d + i] + T(1);
        }
        if (!tulpa_coordinate_linalg::cholesky(H, d, out.chol)) throw InvalidPrecision{};
        std::vector<T> intermediate;
        tulpa_nngp::solve_lower(out.chol, d, rhs, intermediate);
        tulpa_nngp::solve_upper(out.chol, d, intermediate, out.mean);
        for (int i = 0; i < d; ++i) out.jacobian = out.jacobian - safe_log(out.chol[i * d + i]);
        return out;
    }

    template<class T> std::pair<std::vector<T>, T> evaluate(const std::vector<T>& q) const {
        const auto g = geometry(q);
        std::vector<T> z(q.begin() + begin, q.begin() + begin + d), u, field(d, T(0));
        tulpa_nngp::solve_upper(g.chol, d, z, u);
        for (int i = 0; i < d; ++i) u[i] = u[i] + g.mean[i];
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j)
            field[i] = field[i] + g.root[i * d + j] * u[j];
        return {field, g.jacobian};
    }

    double forward(const std::vector<double>& q, std::vector<double>& x) const {
        try {
            const auto result = evaluate(q);
            std::copy(result.first.begin(), result.first.end(), x.begin() + begin);
            return result.second;
        } catch (const InvalidPrecision&) {
            return -INFINITY;
        }
    }

    void inverse(std::vector<double>& x) const {
        const auto g = geometry(x);
        std::vector<double> u(d, 0.0);
        if (layout.is_gp) std::copy(x.begin() + begin, x.begin() + begin + d, u.begin());
        else {
            const double amplitude = layout.is_bym2 ? 1.0 : std::exp(.5 * x[layout.log_tau_spatial_idx]);
            for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j)
                u[i] += amplitude * areal.inverse(i, j) * x[begin + j];
            if (layout.is_bym2) std::copy(x.begin() + begin + n, x.begin() + begin + d, u.begin() + n);
        }
        for (int i = 0; i < d; ++i) {
            x[begin + i] = 0.0;
            for (int j = i; j < d; ++j) x[begin + i] += g.chol[j * d + i] * (u[j] - g.mean[j]);
        }
    }

    void pullback(const std::vector<double>& position, std::vector<double>& gradient) const {
        tulpa::arena::ArenaScope scope;
        const auto q = tulpa::arena::make_vars(scope.arena(), position);
        const auto result = evaluate(q);
        auto objective = result.second;
        for (int i = 0; i < d; ++i) objective = objective + result.first[i] * gradient[begin + i];
        objective.backward();
        const auto derivatives = tulpa::arena::get_adjoints(q);
        for (size_t i = 0; i < gradient.size(); ++i)
            if (i >= static_cast<size_t>(begin) && i < static_cast<size_t>(begin + d)) gradient[i] = derivatives[i];
            else gradient[i] += derivatives[i];
    }
};

struct SpatialMarginal {
    const ModelData* owner;
    uint64_t model_id;
    ModelData data;
    ParamLayout layout;
    int n, d, begin, outer_size;
    PriorRoot areal;
    std::vector<double> last_position, mode, chol;
    double last_value = -INFINITY;

    SpatialMarginal(const ModelData& model, const ParamLayout& outer)
        : owner(&model), model_id(model.unique_id), data(model),
          n(outer.is_gp ? model.gp_data.n_obs : model.n_spatial_units),
          d(outer.is_bym2 ? 2 * n : n), outer_size(outer.total_params) {
        const auto& spec = *static_cast<const tulpa::LikelihoodSpec*>(model.likelihood_spec);
        if (!spec.ll_arena || !spec.eta_weights_fn || !spec.eta_weights_arena_fn ||
            (spec.extra_prior && !spec.extra_prior_arena))
            Rcpp::stop("Collapsed spatial inference requires double and arena likelihood weights and priors");
        data.icar_collapsed = data.bym2_collapsed = data.gp_collapsed = false;
        if (outer.is_gp) data.gp_parameterization = 1;
        layout = compute_param_layout(data);
        begin = layout.is_gp ? layout.gp_w_start : layout.spatial_start;
        if (layout.total_params != outer_size + d) Rcpp::stop("Collapsed spatial layout must remove exactly its Gaussian field");
        if (!layout.is_gp) areal = areal_root(data, layout.is_bym2);
        mode.assign(d, 0.0);
    }

    template<class T> struct Prepared {
        std::vector<T> params, design, eta, zi, oi;
        SpatialBasis<T> basis;
        T prior;
    };

    template<class T> Prepared<T> prepare(const std::vector<T>& outer) const {
        Prepared<T> out;
        out.params.assign(layout.total_params, T(0));
        for (int i = 0; i < outer_size; ++i) out.params[i < begin ? i : i + d] = outer[i];
        out.basis = spatial_basis(out.params, data, layout, areal, n, d);
        out.design.resize(data.N * d);
        for (int j = 0; j < d; ++j) {
            std::vector<T> column(data.N);
            for (int i = 0; i < data.N; ++i) {
                const int unit = layout.is_gp ? data.gp_data.obs_to_loc[i] : data.spatial_group[i] - 1;
                column[i] = out.basis.field[unit * d + j];
            }
            data.spatial_restriction.apply(column);
            for (int i = 0; i < data.N; ++i) out.design[i * d + j] = column[i];
        }
        tulpa::GenericLogPostState<T> state;
        out.prior = tulpa::initialize_generic_state(out.params, data, layout, state);
        const auto& spec = *static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
        if constexpr (std::is_same_v<T, double>) {
            if (spec.extra_prior) out.prior += spec.extra_prior(out.params, layout, data.model_response_data);
        } else if (spec.extra_prior_arena) out.prior = out.prior + spec.extra_prior_arena(out.params, layout, data.model_response_data);
        tulpa::precompute_generic_fixed_eta(data, state);
        out.eta.resize(data.N * data.n_processes);
        out.zi.resize(data.N); out.oi.resize(data.N);
        for (int i = 0; i < data.N; ++i) {
            tulpa::generic_eta_at(i, data, layout, state, out.eta.data() + i * data.n_processes);
            tulpa::generic_zi_oi_logits(i, data, layout, state, out.zi[i], out.oi[i]);
        }
        return out;
    }

    template<class T> struct Inner {
        T likelihood;
        std::vector<T> score, precision;
    };

    template<class T> std::vector<T> parameters(const Prepared<T>& p, const std::vector<T>& u) const {
        auto params = p.params;
        for (int i = 0; i < d; ++i) for (int j = 0; j < d; ++j)
            params[begin + i] = params[begin + i] + p.basis.native[i * d + j] * u[j];
        return params;
    }

    template<class T> struct Observation { T likelihood, score, curvature; };

    template<unsigned Fields, class T> Observation<T> observation(const Prepared<T>& p,
        const std::vector<T>& params, const std::vector<T>& u, int i) const {
        const auto& spec = *static_cast<const tulpa::LikelihoodSpec*>(data.likelihood_spec);
        T eta[tulpa::MAX_PROCESSES], score[tulpa::MAX_PROCESSES], weights[tulpa::MAX_PROCESSES * tulpa::MAX_PROCESSES];
        T field(0);
        for (int j = 0; j < d; ++j) field = field + p.design[i * d + j] * u[j];
        for (int k = 0; k < data.n_processes; ++k)
            eta[k] = p.eta[i * data.n_processes + k] + (data.sharing.spatial[k] ? field : T(0));
        Observation<T> out{T(0), T(0), T(0)};
        if constexpr (std::is_same_v<T, double>) {
            if constexpr (Fields & 1) out.likelihood = spec.ll_double(i, eta, p.zi[i], p.oi[i], params, data, layout, data.model_response_data);
            if constexpr (Fields & 6) spec.eta_weights_fn(i, eta, p.zi[i], p.oi[i], params, data, layout, data.model_response_data, score, weights);
        } else {
            if constexpr (Fields & 1) out.likelihood = spec.ll_arena(i, eta, p.zi[i], p.oi[i], params, data, layout, data.model_response_data);
            if constexpr (Fields & 6) spec.eta_weights_arena_fn(i, eta, p.zi[i], p.oi[i], params, data, layout, data.model_response_data, score, weights);
        }
        for (int k = 0; k < data.n_processes; ++k) if (data.sharing.spatial[k]) {
            if constexpr (Fields & 2) out.score = out.score + score[k];
            if constexpr (Fields & 4) for (int l = 0; l < data.n_processes; ++l) if (data.sharing.spatial[l])
                out.curvature = out.curvature + weights[k * data.n_processes + l];
        }
        return out;
    }

    template<unsigned Fields = 7, class T> Inner<T> inner(const Prepared<T>& p, const std::vector<T>& u) const {
        Inner<T> out{T(0), std::vector<T>((Fields & 2) ? d : 0), std::vector<T>((Fields & 4) ? d * d : 0, T(0))};
        for (int j = 0; j < d; ++j) {
            if constexpr (Fields & 2) out.score[j] = -u[j];
            if constexpr (Fields & 4) out.precision[j * d + j] = T(1);
        }
        const auto params = parameters(p, u);
        for (int i = 0; i < data.N; ++i) {
            const auto row = observation<Fields>(p, params, u, i);
            if constexpr (Fields & 1) out.likelihood = out.likelihood + row.likelihood;
            for (int j = 0; j < d; ++j) {
                if constexpr (Fields & 2) out.score[j] = out.score[j] + p.design[i * d + j] * row.score;
                if constexpr (Fields & 4) for (int k = 0; k < d; ++k) out.precision[j * d + k] = out.precision[j * d + k] +
                    p.design[i * d + j] * row.curvature * p.design[i * d + k];
            }
        }
        return out;
    }

    static double penalized(const Inner<double>& evaluated, const std::vector<double>& u) {
        double value = evaluated.likelihood;
        for (double x : u) value -= .5 * x * x;
        return value;
    }

    bool solve(const std::vector<double>& position) {
        if (position == last_position) return std::isfinite(last_value);
        last_position = position;
        last_value = -INFINITY;
        const auto p = prepare(position);
        mode.assign(d, 0.0);
        struct Scratch { std::vector<double> grad, delta, x_try; } scratch;
        scratch.grad.resize(d); scratch.delta.resize(d); scratch.x_try.resize(d);
        Inner<double> evaluated;
        std::vector<double> evaluated_position;
        const auto refresh = [&]() { evaluated = inner(p, mode); evaluated_position = mode; scratch.grad = evaluated.score; };
        const auto solve_step = [&]() {
            auto precision = evaluated.precision;
            return tulpa::pd_lm_escalate([&](double*) {
                if (!tulpa_nngp::chol_decomp(precision, d, chol, 0)) return false;
                std::vector<double> intermediate;
                tulpa_nngp::solve_lower(chol, d, scratch.grad, intermediate);
                tulpa_nngp::solve_upper(chol, d, intermediate, scratch.delta);
                return true;
            }, [&](double load) { for (int j = 0; j < d; ++j) precision[j * d + j] += load; });
        };
        const auto objective = [&](const std::vector<double>& u) {
            return penalized(u == evaluated_position ? evaluated : inner<1>(p, u), u);
        };
        double value = 0;
        bool valid = false;
        tulpa::NewtonConvState convergence;
        int iterations = 0;
        for (int iteration = 0; iteration < 100; ++iteration)
            if (tulpa::newton_step(mode, scratch, d, iteration, 1e-10, refresh,
                solve_step, objective, value, valid, convergence, iterations)) break;
        evaluated = inner(p, mode);
        for (double gradient : evaluated.score) if (!std::isfinite(gradient) || std::abs(gradient) > 1e-7) return false;
        if (!tulpa_nngp::chol_decomp(evaluated.precision, d, chol, 0)) return false;
        last_value = p.prior + p.basis.jacobian + penalized(evaluated, mode) + .5 * d * std::log(2 * M_PI);
        for (int j = 0; j < d; ++j) last_value -= std::log(chol[j * d + j]);
        return std::isfinite(last_value);
    }

    template<class T> T prior(const std::vector<T>& position) const {
        const auto p = prepare(position);
        return p.prior + p.basis.jacobian + T(.5 * d * std::log(2 * M_PI));
    }

    void gradient(const std::vector<double>& position, std::vector<double>& result) const {
        tulpa::arena::ArenaScope scope;
        const auto q = tulpa::arena::make_vars(scope.arena(), position);
        const auto u = tulpa::arena::make_vars(scope.arena(), mode);
        const auto p = prepare(q);
        const auto params = parameters(p, u);
        std::vector<tulpa::arena::Var> scores(data.N);
        auto objective = p.prior + p.basis.jacobian;
        // -0.5 trace(H^{-1} dH) contracts the Hessian differential before
        // reverse differentiation, using the factor at the conditional mode.
        for (int j = 0; j < d; ++j) objective = objective - .5 * u[j] * u[j];
        for (int i = 0; i < data.N; ++i) {
            const auto row = observation<7>(p, params, u, i);
            scores[i] = row.score;
            std::vector<double> design(d), intermediate, variance;
            for (int j = 0; j < d; ++j) design[j] = p.design[i * d + j].val();
            tulpa_nngp::solve_lower(chol, d, design, intermediate);
            tulpa_nngp::solve_upper(chol, d, intermediate, variance);
            double leverage = 0;
            for (int j = 0; j < d; ++j) {
                leverage += design[j] * variance[j];
                objective = objective - p.design[i * d + j] * (row.curvature.val() * variance[j]);
            }
            objective = objective + row.likelihood - .5 * leverage * row.curvature;
        }
        objective.backward();
        result = tulpa::arena::get_adjoints(q);
        const auto mode_gradient = tulpa::arena::get_adjoints(u);
        // H du*/dtheta = d(score_u)/dtheta propagates the mode contribution.
        std::vector<double> intermediate, direction;
        tulpa_nngp::solve_lower(chol, d, mode_gradient, intermediate);
        tulpa_nngp::solve_upper(chol, d, intermediate, direction);
        tulpa::arena::Var implicit(0);
        for (int i = 0; i < data.N; ++i) {
            tulpa::arena::Var projected(0);
            for (int j = 0; j < d; ++j) projected = projected + p.design[i * d + j] * direction[j];
            implicit = implicit + projected * scores[i];
        }
        for (int j = 0; j < d; ++j) implicit = implicit - u[j] * direction[j];
        implicit.backward();
        const auto correction = tulpa::arena::get_adjoints(q);
        for (size_t j = 0; j < result.size(); ++j) result[j] += correction[j];
    }
};

template<class Map> static tulpa::FieldCoordinateOperator coordinate_operator(std::shared_ptr<Map> map) {
    return {
        [map](const std::vector<double>& q, std::vector<double>& x) { return map->forward(q, x); },
        [map](std::vector<double>& x) { map->inverse(x); },
        [map](const std::vector<double>& q, std::vector<double>& gradient) { map->pullback(q, gradient); }
    };
}

static SpatialMarginal& marginal(const ModelData& data, const ParamLayout& layout) {
    static thread_local SpatialMarginal* cache = nullptr;
    if (!cache || cache->owner != &data || cache->model_id != data.unique_id) {
        delete cache;
        cache = new SpatialMarginal(data, layout);
    }
    return *cache;
}

} // namespace spatial_coordinates

double compute_collapsed_spatial(const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout, bool prior_only) {
    auto& marginal = spatial_coordinates::marginal(data, layout);
    if (prior_only) return marginal.prior(params);
    return marginal.solve(params) ? marginal.last_value : -INFINITY;
}

double compute_collapsed_spatial_normalizer(const ModelData& data, const ParamLayout& layout,
    std::vector<std::string>* uncarried) {
    const auto& marginal = spatial_coordinates::marginal(data, layout);
    return tulpa::compute_log_prior_normalizer_generic_spec(marginal.data, marginal.layout, uncarried);
}

static void collapsed_gradient(const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout, std::vector<double>& gradient, double* log_post, bool prior_only) {
    auto& marginal = spatial_coordinates::marginal(data, layout);
    if (!prior_only && !marginal.solve(params)) {
        gradient.assign(params.size(), 0.0);
        if (log_post) *log_post = -INFINITY;
        return;
    }
    if (prior_only) {
        tulpa::arena::ArenaScope scope;
        const auto q = tulpa::arena::make_vars(scope.arena(), params);
        auto value = marginal.prior(q);
        value.backward();
        gradient = tulpa::arena::get_adjoints(q);
        if (log_post) *log_post = value.val();
    } else {
        marginal.gradient(params, gradient);
        if (log_post) *log_post = marginal.last_value;
    }
}

void compute_gradient_collapsed_spatial(const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout, std::vector<double>& gradient, double* log_post) {
    collapsed_gradient(params, data, layout, gradient, log_post, false);
}

void compute_prior_gradient_collapsed_spatial(const std::vector<double>& params, const ModelData& data,
    const ParamLayout& layout, std::vector<double>& gradient, double* log_post) {
    collapsed_gradient(params, data, layout, gradient, log_post, true);
}

void reconstruct_spatial_point(const std::vector<double>& position, const ModelData& data,
    const ParamLayout& layout, std::mt19937& rng, bool draw,
    double* structured, double* unstructured) {
    auto& marginal = spatial_coordinates::marginal(data, layout);
    if (!marginal.solve(position)) Rcpp::stop("Collapsed field reconstruction requires a converged conditional mode");
    const auto prepared = marginal.prepare(position);
    auto u = marginal.mode;
    std::normal_distribution<double> normal;
    if (draw) {
        std::vector<double> z(marginal.d), increment;
        for (double& x : z) x = normal(rng);
        tulpa_nngp::solve_upper(marginal.chol, marginal.d, z, increment);
        for (int j = 0; j < marginal.d; ++j) u[j] += increment[j];
    }
    const auto& root = layout.is_gp ? prepared.basis.field : prepared.basis.native;
    for (int i = 0; i < marginal.d; ++i) {
        double value = 0;
        for (int j = 0; j < marginal.d; ++j) value += root[i * marginal.d + j] * u[j];
        if (i < marginal.n) structured[i] = value;
        else if (unstructured) unstructured[i - marginal.n] = value;
    }
}

void reconstruct_collapsed_spatial(const ModelData* data, const ParamLayout* layout,
    const double* samples, int n_sample, unsigned int seed, bool draw,
    double* structured, double* unstructured) {
    std::mt19937 rng(seed);
    const int units = layout->is_gp ? data->gp_data.n_obs : data->n_spatial_units;
    for (int row = 0; row < n_sample; ++row) {
        const std::vector<double> q(samples + row * layout->total_params, samples + (row + 1) * layout->total_params);
        reconstruct_spatial_point(q, *data, *layout, rng, draw,
            structured + row * units, unstructured ? unstructured + row * units : nullptr);
    }
}

tulpa::FieldCoordinateOperator prepare_spatial_gaussian_coordinate(const ModelData& data, const ParamLayout& layout) {
    return spatial_coordinates::coordinate_operator(std::make_shared<spatial_coordinates::GaussianConditional>(data, layout));
}

tulpa::FieldCoordinateOperator prepare_bym2_coordinate(const ModelData& data, const ParamLayout& layout) {
    using namespace spatial_coordinates;
    const int n = data.n_spatial_units;
    const auto prior = areal_root(data, true);
    const Eigen::MatrixXd A = data.bym2_scale_factor * prior.centered;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::MatrixXd phi = prior.root * svd.matrixV();
    const Eigen::MatrixXd inverse = svd.matrixV().transpose() * prior.inverse;
    auto out = std::make_shared<Bym2Rotation>();
    out->spatial_start = layout.spatial_start;
    out->theta_start = layout.theta_bym2_start;
    out->rho_index = layout.logit_rho_bym2_idx;
    out->sigma_index = layout.log_sigma_bym2_idx;
    out->n = n;
    out->log_jacobian = prior.log_determinant;
    out->singular_values.assign(svd.singularValues().data(), svd.singularValues().data() + n);
    out->information.assign(n, 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        out->phi_factor.push_back(phi(i, j));
        out->phi_inverse.push_back(inverse(i, j));
        out->theta_factor.push_back(svd.matrixU()(i, j));
    }
    const SpatialWorkingLikelihood working(data, layout);
    for (int j = 0; j < n; ++j) {
        double curvature = 0;
        for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b)
            curvature += svd.matrixU()(a, j) * working.information[a * n + b] * svd.matrixU()(b, j);
        const double singular = out->singular_values[j];
        out->information[j] = std::max(0.0, .5 * (1 + singular * singular) * curvature);
    }
    return coordinate_operator(out);
}

} // namespace tulpa_hmc
