#ifndef TULPA_ORTHOGONAL_PROJECTION_H
#define TULPA_ORTHOGONAL_PROJECTION_H

#include <cmath>
#include <stdexcept>
#include <vector>

namespace tulpa {

// P_perp = I - Q Q', with Q stored row-major with orthonormal columns.
// Projection before a field gather represents S P_perp; projection of
// observation values represents P_perp S.
struct OrthogonalProjection {
    std::vector<double> basis;
    int rank = 0;

    void validate(int n) const {
        if (rank < 0 || rank > n || basis.size() != static_cast<size_t>(n) * rank)
            throw std::invalid_argument("Restriction basis dimensions must be N x rank");
        for (double x : basis)
            if (!std::isfinite(x))
                throw std::invalid_argument("Restriction basis must be finite");
        for (int j = 0; j < rank; ++j)
            for (int k = 0; k <= j; ++k) {
                double dot = 0.0;
                for (int i = 0; i < n; ++i) dot += basis[i * rank + j] * basis[i * rank + k];
                if (std::abs(dot - (j == k ? 1.0 : 0.0)) > 1e-8)
                    throw std::invalid_argument("Restriction basis columns must be orthonormal");
            }
    }

    template<typename T>
    void apply(std::vector<T>& values) const {
        std::vector<T> coef(rank, T(0.0));
        for (size_t i = 0; i < values.size(); ++i)
            for (int k = 0; k < rank; ++k)
                coef[k] = coef[k] + T(basis[i * rank + k]) * values[i];
        for (size_t i = 0; i < values.size(); ++i)
            for (int k = 0; k < rank; ++k)
                values[i] = values[i] - T(basis[i * rank + k]) * coef[k];
    }
};

} // namespace tulpa
#endif
