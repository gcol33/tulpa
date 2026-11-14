extern "C" __global__ void tulpa_dense_diagonal(const double* matrix, double* diagonal, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) diagonal[i] = matrix[(unsigned long long)i * (n + 1)];
}

extern "C" __global__ void tulpa_dense_chol_project(const double* product, double* symmetric, int n) {
    const unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (unsigned long long)n * n) return;
    const int i = k % n, j = k / n;
    symmetric[k] = 0.5 * (i >= j ? product[k] : product[(unsigned long long)j + (unsigned long long)i * n]);
}

extern "C" __global__ void tulpa_dense_add_prior(double* matrix, const double* prior,
    int n, double scale, double constant) {
    const unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (unsigned long long)n * n) return;
    const int i = k % n, j = k / n;
    matrix[k] = i >= j ? matrix[k] + scale * prior[k] + constant : 0.0;
}

extern "C" __global__ void tulpa_dense_weight(const double* basis, const double* weights,
    const double* scale, double* scaled, double* weighted, int rows, int columns, int symmetric) {
    const unsigned long long k = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (unsigned long long)rows * columns) return;
    const double x = basis[k] * (scale ? scale[k / rows] : 1.0);
    scaled[k] = symmetric ? x * sqrt(weights[k % rows]) : x;
    weighted[k] = x * weights[k % rows];
}
