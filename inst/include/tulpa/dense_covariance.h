#ifndef TULPA_DENSE_COVARIANCE
#define TULPA_DENSE_COVARIANCE
#ifdef __CUDACC_RTC__
#define TULPA_DENSE_INLINE __device__ inline
#else
#define TULPA_DENSE_INLINE inline
#endif

// kind: exponential, Matern 3/2, Matern 5/2, Gaussian, periodic. The host
// routes and the device kernel evaluate this one definition.
template<class T, class Exp>
TULPA_DENSE_INLINE T tulpa_dense_covariance(double distance, const T& sigma2,
    const T& phi, int kind, double periodic_sine2, Exp exponential) {
    const T r = T(distance) / phi;
    if (kind == 1) {
        const T u = T(1.7320508075688772935) * r;
        return sigma2 * (T(1.0) + u) * exponential(-u);
    }
    if (kind == 2) {
        const T u = T(2.2360679774997896964) * r;
        return sigma2 * (T(1.0) + u + u * u / T(3.0)) * exponential(-u);
    }
    if (kind == 3) return sigma2 * exponential(-r * r);
    if (kind == 4) return sigma2 * exponential(T(-2.0 * periodic_sine2) / (phi * phi));
    return sigma2 * exponential(-r);
}
#undef TULPA_DENSE_INLINE
#endif
