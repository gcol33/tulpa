#ifndef TULPA_RANGE_PRIOR_H
#define TULPA_RANGE_PRIOR_H

namespace tulpa {

enum class RangePriorType { PC, UNIFORM, LOGNORMAL };

struct RangePrior {
    RangePriorType type = RangePriorType::PC;
    double lower = 0.0;
    double upper = 1.0;
    double meanlog = 0.0;
    double sdlog = 1.0;
};

} // namespace tulpa
#endif
