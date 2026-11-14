#ifndef TULPA_DEVICE_SPARSE_H
#define TULPA_DEVICE_SPARSE_H

#include "device.h"

namespace tulpa { namespace device {

// One factor belongs to one worker and one stream. Numeric updates preserve
// analysis; callers retain ownership of an explicitly supplied stream.
class SparseFactor {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    SparseFactor();
    ~SparseFactor();
    SparseFactor(const SparseFactor&) = delete;
    SparseFactor& operator=(const SparseFactor&) = delete;
    Status analyze(int n, const int* p, const int* i, const double* x,
                   void* stream = nullptr);
    Status refactor(const double* x, bool& positive_definite);
    Status solve(const double* b, double* x, int columns);
    // p/i describe lower CSC entries in the ordering perm; only these entries
    // leave the device. Identity columns are solved in bounded blocks.
    Status inverse_entries(const std::vector<int>& p, const std::vector<int>& i,
                           const std::vector<int>& perm, std::vector<double>& z);
    Status inverse_diagonal(const std::vector<int>& indices, std::vector<double>& z);
    double log_determinant() const;
    std::size_t factor_bytes() const;
    std::size_t analyses() const;
    const std::vector<int>& permutation() const;
    void* stream() const;
};

} }
#endif
