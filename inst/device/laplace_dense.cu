extern "C" __global__ void laplace_dense_solve(const double* L, double* rhs, int n) {
    double* x = rhs + static_cast<unsigned long long>(blockIdx.x)*n;
    for (int j = 0; j < n; ++j) {
        if (threadIdx.x == 0) x[j] /= L[j + j*n];
        __syncthreads();
        const double z = x[j];
        for (int i = j + 1 + threadIdx.x; i < n; i += blockDim.x) x[i] -= L[i + j*n]*z;
        __syncthreads();
    }
    for (int j = n - 1; j >= 0; --j) {
        if (threadIdx.x == 0) x[j] /= L[j + j*n];
        __syncthreads();
        const double z = x[j];
        for (int i = threadIdx.x; i < j; i += blockDim.x) x[i] -= L[j + i*n]*z;
        __syncthreads();
    }
}
