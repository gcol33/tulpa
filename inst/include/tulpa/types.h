#ifndef TULPA_TYPES_H
#define TULPA_TYPES_H

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace tulpa {

// ============================================================================
// Spatial covariance kernel types
// ============================================================================
// One code per KERNEL, not per family: the smoothness of a Matern is part of
// the code rather than a separate `nu` beside it, because a `nu` the kernel
// dispatch does not read is a smoothness the fit silently ignores. New codes
// append -- 0..3 are read by consumer packages that map their own `cov` string
// onto them.
enum class CovType : int {
    EXPONENTIAL = 0,
    MATERN32 = 1,
    GAUSSIAN = 2,
    SPHERICAL = 3,
    MATERN52 = 4
};

// ============================================================================
// Temporal covariance kernel types
// ============================================================================
enum class TemporalCovType : int {
    EXPONENTIAL = 0,
    MATERN = 1,
    GAUSSIAN = 2,
    PERIODIC = 3
};

// ============================================================================
// Spatial field types
// ============================================================================
enum class SpatialType : int {
    NONE = 0,
    ICAR = 1,
    BYM2 = 2,
    GP = 3,            // NNGP
    MULTISCALE_GP = 4,
    HSGP = 5,          // Hilbert Space GP
    CAR_PROPER = 6,
    SPDE = 7           // Continuous Matern via FEM (Lindgren–Rue 2011)
};

// ============================================================================
// Temporal structure types
// ============================================================================
enum class TemporalType : int {
    NONE = 0,
    RW1 = 1,
    RW2 = 2,
    AR1 = 3,
    IID = 4,
    GP = 5,
    MULTISCALE = 6
};

// ============================================================================
// Spatiotemporal interaction types (Knorr-Held + GP-based)
// ============================================================================
enum class STType : int {
    NONE = 0,
    TYPE_I = 1,        // Unstructured
    TYPE_II = 2,       // Temporal per location
    TYPE_III = 3,      // Spatial per time
    TYPE_IV = 4,       // Full Kronecker
    SEPARABLE = 5,     // Separable GP
    NONSEP_GP = 6      // Nonseparable GP
};

// ============================================================================
// Nonseparable spatiotemporal covariance types
// ============================================================================
enum class NonsepType : int {
    PRODUCT = 0,
    SUM = 1,
    GNEITING = 2,
    CRESSIE_HUANG = 3
};

// ============================================================================
// Zero-inflation types
// ============================================================================
enum class ZIType : int {
    NONE = 0,
    // Distribution-specific mechanisms: the engine sizes the zi/oi coefficient
    // blocks (has_zi / has_oi); model packages compute the likelihood.
    ZI_POISSON = 1,
    ZI_NEGBIN = 2,
    HURDLE_POISSON = 3,
    HURDLE_NEGBIN = 4,
    ZI_BINOMIAL = 5,
    HURDLE_BINOMIAL = 6,
    OI_BINOMIAL = 7,
    ZOIB = 8,
    // Generic (used by multi-process interface — model packages set ZI mechanism,
    // distribution handled by LikelihoodFn)
    ZI = 20,
    HURDLE = 21,
    OI = 22
};

// ============================================================================
// Inference tiers (epistemic guarantees)
// ============================================================================
enum class InferenceTier : int {
    EXACT = 1,         // Tier 1: Asymptotically correct posterior (HMC, ESS, PG, Gibbs)
    STRUCTURED = 2,    // Tier 2: Correct conditional on structural assumptions (Laplace)
    OPTIMIZED = 3      // Tier 3: No general correctness guarantee (VI, SGHMC)
};

// ============================================================================
// Gradient computation modes
// ============================================================================
enum class GradientMode : int {
    AUTO = 0,              // Select best available: H > A_r > N
    NUMERICAL = 1,         // N: Finite differences O(p*N)
    AUTODIFF_ARENA = 3,    // A_r: Arena reverse-mode O(N)
    HANDCODED = 5          // H: Analytical hand-coded O(N)
};

// ============================================================================
// Mass matrix types
// ============================================================================
enum class MassMatrixType : int {
    DIAG = 0,
    DENSE = 1,
    BLOCK_DIAG = 2,
    AUTO = 3,
    // A diagonal metric whose structured-field block is read from that block's
    // own posterior precision instead of from the Welford accumulator. It
    // resolves to DIAG before any leapfrog runs (see MassMatrixConfig), so no
    // kinetic-energy, momentum or drift path branches on it.
    GMRF = 4,
    // GMRF plus the field's soft sum-to-zero margin directions, carried as an
    // explicit low-rank term on top of that diagonal (hmc_mass_lowrank.h).
    // Those directions are linear combinations rather than coordinates, so
    // they are the part of the block's stiffness no diagonal metric of any
    // kind can rescale. Also resolves to DIAG, with the low-rank term riding
    // as an overlay on the same DenseMassMatrix.
    GMRF_MARGIN = 5
};

// ============================================================================
// RE parameterization
// ============================================================================
enum class REParam : int {
    CENTERED = 0,
    NON_CENTERED = 1
};

// ============================================================================
// Multi-scale GP sampler strategies
// ============================================================================
enum class MSGPSampler : int {
    AUTO = 0,
    NONCENTERED = 1,
    CENTERED = 2,
    INTERWEAVED = 3,
    ADAPTIVE = 4,
    RIEMANNIAN = 5,
    LBFGS = 6
};

// ============================================================================
// Generic string-to-enum parser (eliminates N identical if/else chains)
// ============================================================================
template<typename EnumT>
struct EnumEntry {
    const char* name;
    EnumT value;
};

template<typename EnumT, int N>
inline EnumT parse_enum(const std::string& str,
                         const EnumEntry<EnumT> (&table)[N],
                         EnumT default_val) {
    for (int i = 0; i < N; i++) {
        if (str == table[i].name) return table[i].value;
    }
    return default_val;
}

inline constexpr EnumEntry<GradientMode> GRADIENT_MODE_NAMES[] = {
    {"auto", GradientMode::AUTO},
    {"N", GradientMode::NUMERICAL}, {"numerical", GradientMode::NUMERICAL},
    {"A_r", GradientMode::AUTODIFF_ARENA}, {"arena", GradientMode::AUTODIFF_ARENA},
    {"H", GradientMode::HANDCODED}, {"handcoded", GradientMode::HANDCODED}
};

inline std::vector<std::string> gradient_mode_names() {
    std::vector<std::string> names;
    for (const auto& entry : GRADIENT_MODE_NAMES) names.emplace_back(entry.name);
    return names;
}

inline GradientMode parse_gradient_mode(const std::string& name) {
    std::string valid;
    for (const auto& entry : GRADIENT_MODE_NAMES) {
        if (name == entry.name) return entry.value;
        if (!valid.empty()) valid += ", ";
        valid += "'" + std::string(entry.name) + "'";
    }
    throw std::invalid_argument("Unknown gradient mode '" + name +
        "'. Expected one of " + valid + ".");
}

} // namespace tulpa

#endif // TULPA_TYPES_H
