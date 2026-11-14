#ifndef TULPA_DENSE_BASIS_PRODUCTS_H
#define TULPA_DENSE_BASIS_PRODUCTS_H

#include "latent_block.h"
#include "device_dense.h"
#include <Eigen/Dense>
#include <limits>

namespace tulpa {

struct DenseBasisView {
    LatentBlock::DenseBasisBatch batch;
    std::vector<double> storage;
};

inline DenseBasisView dense_basis_view(const LatentBlock& block, int rows, int arm, int cell) {
    DenseBasisView out;
    if (block.dense_basis_batch) out.batch = block.dense_basis_batch(arm, cell);
    else {
        out.storage.assign(static_cast<std::size_t>(rows) * block.size, 0.0);
        for (int i = 0; i < rows; ++i)
            block.basis_eval(i, arm, cell, out.storage.data() + static_cast<std::size_t>(i) * block.size);
        out.batch.data = out.storage.data();
        out.batch.N_k = rows;
        out.batch.m_per_arm = block.size;
    }
    return out;
}

inline void dense_basis_field(const LatentBlock& block, int rows, int arm, int cell,
                               const double* coefficients, std::vector<double>& field) {
    auto view = dense_basis_view(block, rows, arm, cell);
    const auto& b = view.batch;
    field.resize(rows);
    std::vector<double> scaled(b.m_per_arm);
    for (int j = 0; j < b.m_per_arm; ++j)
        scaled[j] = coefficients[b.m_offset_in_block + j] * (b.sqrt_S ? b.sqrt_S[j] : 1.0);
    if (b.N_k != rows || !b.data || !device::dense_basis(b.data, false).apply(
            b.data, rows, b.m_per_arm, true, scaled.data(), field.data()))
        std::fill(field.begin(), field.end(), std::numeric_limits<double>::quiet_NaN());
}

struct DenseBasisProducts {
    std::unique_ptr<device::DenseBasis> products;
    Eigen::VectorXd g_obs, w_obs, w_sqrt, w_signed_sqrt, amplitude, product_g, product_w;
    Eigen::MatrixXd WQ, signed_WQ, H_block;
    Eigen::VectorXd grad_block;
    bool has_negative = false;
};

inline void dense_basis_products(const LatentBlock::DenseBasisBatch& b, DenseBasisProducts& s,
                                  const double* amplitude = nullptr) {
    const int n = b.N_k, m = b.m_per_arm;
    s.w_sqrt.resize(n); s.w_signed_sqrt.resize(n);
    s.product_g = s.g_obs; s.product_w = s.w_obs;
    s.has_negative = false;
    for (int i = 0; i < n; ++i) {
        s.w_sqrt[i] = std::sqrt(std::abs(s.w_obs[i]));
        s.w_signed_sqrt[i] = s.w_obs[i] < 0 ? -s.w_sqrt[i] : s.w_sqrt[i];
        s.has_negative = s.has_negative || s.w_obs[i] < 0;
        if (amplitude) {
            s.product_g[i] *= amplitude[i];
            s.product_w[i] *= amplitude[i] * amplitude[i];
        }
    }
    using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RowMatrix> phi(b.data, n, m);
    s.WQ = s.w_sqrt.asDiagonal() * phi;
    if (b.sqrt_S) s.WQ *= Eigen::Map<const Eigen::VectorXd>(b.sqrt_S, m).asDiagonal();
    if (amplitude) s.WQ = Eigen::Map<const Eigen::VectorXd>(amplitude, n).asDiagonal() * s.WQ;
    if (s.has_negative) {
        s.signed_WQ = s.WQ;
        for (int i = 0; i < n; ++i) if (s.w_obs[i] < 0) s.signed_WQ.row(i) *= -1;
    }
    if (!s.products) s.products.reset(new device::DenseBasis);
    s.H_block.resize(m, m);
    if (!s.products->gram(b.data, n, m, true, s.product_w.data(), b.sqrt_S, s.H_block.data()))
        s.H_block.setConstant(std::numeric_limits<double>::quiet_NaN());
    s.grad_block.resize(m);
    if (!s.products->apply(b.data, n, m, true, s.product_g.data(), s.grad_block.data(), true))
        s.grad_block.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (b.sqrt_S) s.grad_block.array() *= Eigen::Map<const Eigen::VectorXd>(b.sqrt_S, m).array();
}

inline Eigen::MatrixXd dense_basis_cross_products(const DenseBasisProducts& a, const DenseBasisProducts& b) {
    Eigen::MatrixXd out(a.WQ.cols(), b.WQ.cols());
    if (!device::cross_product(a.WQ.data(), a.WQ.rows(), a.WQ.cols(),
            (b.has_negative ? b.signed_WQ : b.WQ).data(), b.WQ.cols(), out.data()))
        out.setConstant(std::numeric_limits<double>::quiet_NaN());
    return out;
}

}
#endif
