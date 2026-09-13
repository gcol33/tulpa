// row_classes.h
// Rows sharing a loading vector, and therefore a predictive variance.
//
// The per-row predictive variance of a linear predictor is a' H^{-1} a for the
// row's loading vector a = d eta / d x. Two rows whose loading vectors are
// bit-identical share it exactly, so one back-solve per distinct vector serves
// every row carrying it. Keys are compared on exact IEEE bit patterns: a
// tolerance would merge rows whose loading vectors differ, and every member of a
// class reads its variance off the representative's solve.

#ifndef TULPA_ROW_CLASSES_H
#define TULPA_ROW_CLASSES_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace tulpa {

struct RowClasses {
    std::vector<int> row_class;  // row -> class id in [0, n_class)
    std::vector<int> class_rep;  // class id -> the row solved for
    std::size_t size() const { return class_rep.size(); }
};

// Packed sparse row loadings: row r's (latent index, weight) pairs occupy
// [off[r], off[r + 1]) of `idx` / `w`, in the order the row's walk emits them.
struct RowLoadings {
    std::vector<std::size_t> off{0};
    std::vector<int>         idx;
    std::vector<double>      w;

    int n_rows() const { return static_cast<int>(off.size()) - 1; }
    void clear() { off.assign(1, 0); idx.clear(); w.clear(); }
    void push(int latent, double weight) {
        idx.push_back(latent);
        w.push_back(weight);
    }
    void end_row() { off.push_back(idx.size()); }
};

// FNV-1a, folded a 64-bit word at a time. The hash only groups CANDIDATES:
// every merge is confirmed entry by entry against the representative's key, so
// a collision costs one comparison rather than fusing two distinct rows.
inline std::uint64_t row_key_mix(std::uint64_t h, std::uint64_t word) {
    for (int byte = 0; byte < 8; byte++) {
        h ^= (word >> (8 * byte)) & 0xFFULL;
        h *= 1099511628211ULL;
    }
    return h;
}

inline std::uint64_t row_weight_bits(double w) {
    std::uint64_t bits;
    std::memcpy(&bits, &w, sizeof(bits));
    return bits;
}

// Group the rows of `L` by their exact (index, weight-bits) sequence. The
// representative of each class is its first row.
inline RowClasses row_classes_from_loadings(const RowLoadings& L) {
    RowClasses out;
    const int N = L.n_rows();
    if (N <= 0) return out;
    out.row_class.assign(N, 0);

    std::vector<std::uint64_t> row_hash(N, 0);
    for (int i = 0; i < N; i++) {
        std::uint64_t h = 1469598103934665603ULL;
        for (std::size_t s = L.off[i]; s < L.off[i + 1]; s++) {
            h = row_key_mix(h, static_cast<std::uint64_t>(
                                   static_cast<std::uint32_t>(L.idx[s])));
            h = row_key_mix(h, row_weight_bits(L.w[s]));
        }
        row_hash[i] = h;
    }

    auto key_equal = [&](int i, int j) {
        const std::size_t oi = L.off[i], oj = L.off[j];
        const std::size_t ni = L.off[i + 1] - oi;
        if (ni != L.off[j + 1] - oj) return false;
        for (std::size_t s = 0; s < ni; s++) {
            if (L.idx[oi + s] != L.idx[oj + s]) return false;
            if (row_weight_bits(L.w[oi + s]) != row_weight_bits(L.w[oj + s]))
                return false;
        }
        return true;
    };

    std::unordered_map<std::uint64_t, std::vector<int>> buckets;
    buckets.reserve(static_cast<std::size_t>(N));
    for (int i = 0; i < N; i++) {
        std::vector<int>& candidates = buckets[row_hash[i]];
        int cls = -1;
        for (int c : candidates) {
            if (key_equal(i, out.class_rep[c])) { cls = c; break; }
        }
        if (cls < 0) {
            cls = static_cast<int>(out.class_rep.size());
            out.class_rep.push_back(i);
            candidates.push_back(cls);
        }
        out.row_class[i] = cls;
    }
    return out;
}

} // namespace tulpa

#endif // TULPA_ROW_CLASSES_H
