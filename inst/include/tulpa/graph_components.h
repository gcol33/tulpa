// graph_components.h
// Connected-component analysis of an adjacency graph, shared by every intrinsic
// (rank-deficient) field: ICAR, BYM2, MCAR.
//
// An intrinsic precision Q = D - W has one constant null direction per connected
// component of the graph. Identifying the field (sum_to_zero.h) needs both the
// NUMBER of components (the rank normalizer is n - n_components) and their
// MEMBERSHIP (each component's constant is pinned over exactly that component's
// nodes). A single connected graph, a spatial(by=) replicate over the
// block-diagonal I_L (x) Q, and a genuine map with unequal disconnected pieces
// (a mainland plus islands, whose nodes need not be contiguous in the node
// ordering) are all just "the connected components of the graph" -- computed
// here once, so no consumer re-derives membership from a contiguity assumption.

#ifndef TULPA_GRAPH_COMPONENTS_H
#define TULPA_GRAPH_COMPONENTS_H

#include <vector>

namespace tulpa {

// Fill `label[s]` with the 0-based component index of node s and return the
// component count. Iterative DFS (an explicit stack, no recursion-depth risk).
// `col_base` is 0 for a 0-based adj_col_idx (the sampler ModelData adjacency)
// and 1 for a 1-based one (the spatiotemporal adjacency).
inline int label_graph_components(
    int S, const int* adj_row_ptr, const int* adj_col_idx, int col_base,
    std::vector<int>& label
) {
    label.assign(S > 0 ? S : 0, -1);
    if (S <= 0) return 0;
    std::vector<int> stack;
    int k = 0;
    for (int s0 = 0; s0 < S; ++s0) {
        if (label[s0] >= 0) continue;
        label[s0] = k;
        stack.clear();
        stack.push_back(s0);
        while (!stack.empty()) {
            int s = stack.back();
            stack.pop_back();
            const int row_end = adj_row_ptr[s + 1];
            for (int e = adj_row_ptr[s]; e < row_end; ++e) {
                int t = adj_col_idx[e] - col_base;
                if (t >= 0 && t < S && label[t] < 0) {
                    label[t] = k;
                    stack.push_back(t);
                }
            }
        }
        ++k;
    }
    return k;
}

// Number of connected components. Kept as a count-only entry point because the
// spatiotemporal rank normalizer evaluates it per gradient (a partition's node
// buckets would be allocated and thrown away there); it shares the DFS with the
// full partition through label_graph_components.
inline int count_graph_components(
    int S, const int* adj_row_ptr, const int* adj_col_idx, int col_base = 0
) {
    std::vector<int> label;
    return label_graph_components(S, adj_row_ptr, adj_col_idx, col_base, label);
}

// The component partition: for a disconnected graph, comp_ptr (L + 1) and
// comp_nodes (the S node indices grouped by component, increasing within each
// group -- the grouping order is immaterial since every consumer sums a
// component symmetrically). A SINGLE connected graph is left "trivial"
// (comp_ptr / comp_nodes empty), so the overwhelmingly common case (a connected
// map, a replicate's caller that never disconnects, an RW field) carries no
// per-node buffer and every consumer takes the contiguous [start, start + n)
// fast path with no indirection.
struct GraphPartition {
    int n = 0;
    std::vector<int> comp_ptr;    // size L + 1, or empty when trivial (L == 1)
    std::vector<int> comp_nodes;  // size n, or empty when trivial

    int n_components() const {
        if (n <= 0) return 0;
        return comp_ptr.empty() ? 1 : static_cast<int>(comp_ptr.size()) - 1;
    }
    bool trivial() const { return comp_ptr.empty(); }

    // Field-local node list of component c (nullptr for the trivial single
    // component, meaning the contiguous run 0..n-1), and its size.
    const int* nodes(int c) const {
        return comp_ptr.empty() ? nullptr : &comp_nodes[comp_ptr[c]];
    }
    int size(int c) const {
        return comp_ptr.empty() ? n : (comp_ptr[c + 1] - comp_ptr[c]);
    }
};

inline GraphPartition graph_partition(
    int S, const int* adj_row_ptr, const int* adj_col_idx, int col_base = 0
) {
    GraphPartition P;
    P.n = S > 0 ? S : 0;
    if (S <= 0) return P;
    std::vector<int> label;
    const int k = label_graph_components(S, adj_row_ptr, adj_col_idx, col_base,
                                         label);
    if (k <= 1) return P;                     // trivial: single component
    P.comp_ptr.assign(k + 1, 0);
    for (int i = 0; i < S; ++i) P.comp_ptr[label[i] + 1]++;
    for (int c = 0; c < k; ++c) P.comp_ptr[c + 1] += P.comp_ptr[c];
    P.comp_nodes.assign(S, 0);
    std::vector<int> cur(P.comp_ptr.begin(), P.comp_ptr.begin() + k);
    for (int i = 0; i < S; ++i) P.comp_nodes[cur[label[i]]++] = i;
    return P;
}

}  // namespace tulpa

#endif  // TULPA_GRAPH_COMPONENTS_H
