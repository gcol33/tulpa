#ifndef TULPA_ST_MARGINS_H
#define TULPA_ST_MARGINS_H

#include <vector>

namespace tulpa {

template<class T>
void center_st_margins(std::vector<T>& field, int spatial, int times) {
    for (int s = 0; s < spatial; ++s) {
        T mean(0.0);
        for (int t = 0; t < times; ++t) mean = mean + field[s * times + t];
        mean = mean / T(times);
        for (int t = 0; t < times; ++t) field[s * times + t] = field[s * times + t] - mean;
    }
    for (int t = 0; t < times; ++t) {
        T mean(0.0);
        for (int s = 0; s < spatial; ++s) mean = mean + field[s * times + t];
        mean = mean / T(spatial);
        for (int s = 0; s < spatial; ++s) field[s * times + t] = field[s * times + t] - mean;
    }
}

template<class T>
T centered_ar1_precision_det(const T& rho, const T& initial_precision, int times) {
    T geometric(0.0), sum_squares(0.0);
    for (int t = 1; t <= times; ++t) {
        geometric = T(1.0) + rho * geometric;
        if (t < times) sum_squares = sum_squares + geometric * geometric;
    }
    return (geometric * geometric + initial_precision * sum_squares) / T(times);
}

}
#endif
