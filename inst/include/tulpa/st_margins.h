#ifndef TULPA_ST_MARGINS_H
#define TULPA_ST_MARGINS_H

#include <vector>
#include <stdexcept>
#include <tulpa/graph_components.h>
#include <tulpa/soft_sum_to_zero.h>

namespace tulpa {

template<class Data>
void setup_st_spatial_partition(Data& st) {
    st.spatial_partition = GraphPartition{};
    using Type = decltype(st.type);
    if (st.type != Type::TYPE_III && st.type != Type::TYPE_IV) return;
    const int n = st.n_spatial;
    if (n < 1 || st.adj_row_ptr.size() != static_cast<size_t>(n + 1) ||
        st.adj_row_ptr.front() != 0 ||
        st.adj_row_ptr.back() != static_cast<int>(st.adj_col_idx.size()))
        throw std::invalid_argument("intrinsic ST spatial partition requires complete CSR adjacency");
    for (int s = 0; s < n; ++s)
        if (st.adj_row_ptr[s] > st.adj_row_ptr[s + 1])
            throw std::invalid_argument("intrinsic ST adjacency row pointers must be ordered");
    for (int index : st.adj_col_idx)
        if (index < 1 || index > n)
            throw std::invalid_argument("intrinsic ST adjacency indices must be 1-based spatial units");
    st.n_neighbors.resize(n);
    for (int s = 0; s < n; ++s) st.n_neighbors[s] = st.adj_row_ptr[s + 1] - st.adj_row_ptr[s];
    st.spatial_partition = graph_partition(n, st.adj_row_ptr.data(), st.adj_col_idx.data(), 1);
}

inline int st_partition_count(const GraphPartition& partition) {
    return partition.n > 0 ? partition.n_components() : 1;
}

template<class T, class TimeQuadratic>
T st_kronecker_quadratic(const T* field, int spatial, int times,
    const std::vector<int>& row_ptr, const std::vector<int>& col_idx,
    TimeQuadratic time_quadratic) {
    T out(0.0);
    std::vector<T> difference(times);
    for (int s = 0; s < spatial; ++s) for (int k = row_ptr[s]; k < row_ptr[s + 1]; ++k) {
        const int other = col_idx[k] - 1;
        if (other <= s) continue;
        for (int t = 0; t < times; ++t)
            difference[t] = field[s * times + t] - field[other * times + t];
        out = out + time_quadratic(difference);
    }
    return out;
}

template<class T>
void center_st_margins(std::vector<T>& field, int spatial, int times,
                       const GraphPartition& partition = GraphPartition{}) {
    for (int s = 0; s < spatial; ++s) {
        T mean(0.0);
        for (int t = 0; t < times; ++t) mean = mean + field[s * times + t];
        mean = mean / T(times);
        for (int t = 0; t < times; ++t) field[s * times + t] = field[s * times + t] - mean;
    }
    for (int c = 0; c < st_partition_count(partition); ++c) for (int t = 0; t < times; ++t) {
        const int count = partition.n > 0 ? partition.size(c) : spatial;
        const int* nodes = partition.n > 0 ? partition.nodes(c) : nullptr;
        T mean(0.0);
        for (int j = 0; j < count; ++j) mean = mean + field[(nodes ? nodes[j] : j) * times + t];
        mean = mean / T(count);
        for (int j = 0; j < count; ++j) {
            const int s = nodes ? nodes[j] : j;
            field[s * times + t] = field[s * times + t] - mean;
        }
    }
}

template<class T>
T st_margin_penalty(const T* field, int spatial, int times, bool space, bool time,
                     const GraphPartition& partition = GraphPartition{}) {
    T out(0.0);
    if (space) for (int c = 0; c < st_partition_count(partition); ++c) {
        const int count = partition.n > 0 ? partition.size(c) : spatial;
        const int* nodes = partition.n > 0 ? partition.nodes(c) : nullptr;
        for (int t = 0; t < times; ++t) {
            T sum(0.0);
            for (int j = 0; j < count; ++j) sum = sum + field[(nodes ? nodes[j] : j) * times + t];
            out = out - T(0.5 * s2z_precision(count)) * sum * sum;
        }
    }
    if (time) for (int s = 0; s < spatial; ++s) {
        T sum(0.0);
        for (int t = 0; t < times; ++t) sum = sum + field[s * times + t];
        out = out - T(0.5 * s2z_precision(times)) * sum * sum;
    }
    return out;
}

template<class T>
T centered_ar1_precision_det(const T& rho, const T& initial_precision, int times) {
    T geometric(0.0), sum_squares(0.0);
    for (int t = 1; t <= times; ++t) {
        geometric = T(1.0) + rho * geometric;
        if (t < times) sum_squares = sum_squares + geometric * geometric;
    }
    return (geometric * geometric + initial_precision * sum_squares) / T(times);
}

}
#endif
