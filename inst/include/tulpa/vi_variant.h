#ifndef TULPA_VI_VARIANT_H
#define TULPA_VI_VARIANT_H

#include <stdexcept>
#include <string>

namespace tulpa { namespace vi {

enum class VIVariant : int { MEANFIELD, LOWRANK, FULLRANK, AUTO };

struct VIVariantEntry {
    VIVariant value;
    const char* name;
};

inline constexpr VIVariantEntry vi_variants[] = {
    {VIVariant::MEANFIELD, "meanfield"},
    {VIVariant::LOWRANK, "lowrank"},
    {VIVariant::FULLRANK, "fullrank"},
    {VIVariant::AUTO, "auto"}
};

inline VIVariant parse_variant(const std::string& name) {
    for (const auto& entry : vi_variants)
        if (name == entry.name) return entry.value;
    throw std::invalid_argument("Unknown VI variant: " + name);
}

inline std::string variant_to_string(VIVariant value) {
    for (const auto& entry : vi_variants)
        if (value == entry.value) return entry.name;
    return "unknown";
}

} }
#endif
