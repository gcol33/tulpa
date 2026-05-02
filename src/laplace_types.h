// laplace_types.h
// Shared lightweight types for the Laplace approximation engine.

#ifndef TULPA_LAPLACE_TYPES_H
#define TULPA_LAPLACE_TYPES_H

#include "laplace_core.h"
#include <vector>

namespace tulpa {

using DenseVec = std::vector<double>;
using DenseMat = std::vector<std::vector<double>>;

} // namespace tulpa

#endif // TULPA_LAPLACE_TYPES_H
