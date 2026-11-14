extern "C" __global__ void tulpa_axpy(const double* x, double* y,
                                     double a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
