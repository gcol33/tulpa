#ifndef TULPA_DEVICE_DENSE_H
#define TULPA_DEVICE_DENSE_H

#include "device.h"
#include <memory>

namespace tulpa { namespace device {

enum class DenseSystem { Symmetric, Lower, LowerTranspose };

// Matrices crossing this interface are column-major unless row_major is set.
// A factor owns its device workspace across evaluations and multiple solves.
class DenseFactor {
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    DenseFactor();
    ~DenseFactor();
    DenseFactor(const DenseFactor&) = delete;
    DenseFactor& operator=(const DenseFactor&) = delete;
    bool factorize(const double* matrix, int n, bool row_major,
                   const std::string& operation, double nugget = 0.0);
    bool covariance(const std::vector<double>& times, double sigma2, double phi,
                    int kind, double period, double nugget,
                    const std::string& operation, int variant = 0);
    bool projected(const double* projector, int n, const double* weights,
                   const double* residual, const double* prior, bool prior_row_major,
                   double prior_scale, double constant_augmentation, double* linear,
                   const std::string& operation);
    bool solve(const double* rhs, double* out, int columns = 1,
               DenseSystem system = DenseSystem::Symmetric);
    bool lower(double* out, bool row_major = false);
    double log_determinant() const;
    bool used_cuda() const;
};

// Immutable bases are retained on the device. Changing weights and coefficients
// are uploaded separately from the basis.
class DenseBasis {
    friend class DenseFactor;
    struct Impl;
    std::unique_ptr<Impl> impl_;
public:
    explicit DenseBasis(bool immutable = false);
    ~DenseBasis();
    DenseBasis(const DenseBasis&) = delete;
    DenseBasis& operator=(const DenseBasis&) = delete;
    bool apply(const double* basis, int rows, int columns, bool row_major,
               const double* coefficients, double* out, bool transpose = false,
               const std::string& operation = "",
               std::size_t dispatch_size = 0, int dispatch_variant = -1);
    bool gram(const double* basis, int rows, int columns, bool row_major,
              const double* weights, const double* scale, double* out,
              const std::string& operation = "hsgp_crossprod",
              std::size_t dispatch_size = 0, int dispatch_variant = -1);
};

DenseFactor& dense_factor(const std::string& operation);
DenseBasis& dense_basis(const double* identity = nullptr, bool immutable = true);
bool cholesky_adjoint(const double* lower_row_major, const double* adjoint_row_major,
                      double* out_row_major, int n);
bool cross_product(const double* left, int rows, int columns, const double* right,
                   int right_columns, double* out);
int cross_variant(int left_columns, int right_columns);
bool projected_precision(const double* projector, int n, const double* weights,
                          const double* residual, double* matrix, double* linear,
                          const std::string& operation);

} }
#endif
