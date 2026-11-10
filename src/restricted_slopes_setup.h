#ifndef TULPA_RESTRICTED_SLOPES_SETUP_H
#define TULPA_RESTRICTED_SLOPES_SETUP_H
#include <Eigen/Dense>
#include <limits>
#include <tulpa/model_data.h>

namespace tulpa {
inline RestrictedSlopes restricted_slopes_setup(const ModelData& data, int n,
    int groups, const std::vector<int>& time, const std::vector<int>& group,
    bool spectral) {
    RestrictedSlopes out;
    out.n_times = n;
    out.n_groups = groups;
    out.nc_index = spectral ? 1 : n - 1;
    if (n < 3 || groups < 1 || time.size() != static_cast<size_t>(data.N))
        throw std::invalid_argument("restricted RW2 requires complete time indices and at least three times");
    out.linear.resize(n);
    double norm = 0.0;
    for (int t = 0; t < n; ++t) {
        out.linear[t] = t - 0.5 * (n - 1);
        norm += out.linear[t] * out.linear[t];
    }
    norm = std::sqrt(norm);
    for (double& value : out.linear) value /= norm;
    out.nc_scale = spectral ? 1.0 : norm;
    const int rank = data.temporal_restriction_rank;
    Eigen::MatrixXd cross = Eigen::MatrixXd::Zero(rank, groups);
    Eigen::VectorXd counts = Eigen::VectorXd::Zero(groups);
    auto index = [&](int i) { return group.empty() ? 0 : group[i] - 1; };
    if (!group.empty() && group.size() != time.size())
        throw std::invalid_argument("restricted RW2 group indices must match observations");
    for (int i = 0; i < data.N; ++i) {
        const int g = index(i), t = time[i] - 1;
        if (g < 0 || g >= groups || t < 0 || t >= n)
            throw std::invalid_argument("restricted RW2 indices are out of range");
        const double value = out.linear[t];
        counts[g] += value * value;
        for (int j = 0; j < rank; ++j)
            cross(j, g) += data.temporal_restriction_basis[i * rank + j] * value;
    }
    for (int g = 0; g < groups; ++g) {
        if (counts[g] == 0.0) out.unobserved.push_back(g);
        else cross.col(g) /= std::sqrt(counts[g]);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(cross, Eigen::ComputeThinV);
    std::vector<Eigen::VectorXd> directions;
    for (int j = 0; j < svd.singularValues().size(); ++j) {
        if (std::abs(svd.singularValues()[j] - 1.0) > 1e-10) continue;
        Eigen::VectorXd v = svd.matrixV().col(j);
        for (int g = 0; g < groups; ++g)
            v[g] = counts[g] == 0.0 ? 0.0 : v[g] / std::sqrt(counts[g]);
        Eigen::VectorXd mapped(data.N);
        for (int i = 0; i < data.N; ++i) mapped[i] = v[index(i)] * out.linear[time[i] - 1];
        Eigen::VectorXd projected = mapped;
        for (int k = 0; k < rank; ++k) {
            double dot = 0.0;
            for (int i = 0; i < data.N; ++i) dot += data.temporal_restriction_basis[i * rank + k] * mapped[i];
            for (int i = 0; i < data.N; ++i) projected[i] -= data.temporal_restriction_basis[i * rank + k] * dot;
        }
        if (projected.norm() > 128.0 * std::numeric_limits<double>::epsilon() *
            std::sqrt(static_cast<double>(data.N)) * mapped.norm()) continue;
        for (const auto& previous : directions) v -= previous.dot(v) * previous;
        v.normalize();
        directions.push_back(v);
        out.directions.insert(out.directions.end(), v.data(), v.data() + groups);
    }
    return out;
}

inline void prepare_temporal_restriction(ModelData& data) {
    data.temporal_restricted_slopes = RestrictedSlopes{};
    data.ms_trend_restricted_slopes = RestrictedSlopes{};
    if (data.temporal_restriction_rank == 0) return;
    if (data.temporal_type == TemporalType::RW2 && !data.temporal_cyclic)
        data.temporal_restricted_slopes = restricted_slopes_setup(data, data.n_times,
            data.n_temporal_groups, data.temporal_time_idx, data.temporal_group_idx, false);
    if (data.has_multiscale_temporal && data.multiscale_temporal_data.trend_type == TemporalType::RW2) {
        const auto& ms = data.multiscale_temporal_data;
        data.ms_trend_restricted_slopes = restricted_slopes_setup(data, ms.n_times,
            1, ms.time_index, {}, true);
    }
}
}
#endif
