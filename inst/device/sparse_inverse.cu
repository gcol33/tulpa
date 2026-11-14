extern "C" __global__ void sparse_identity(double* b, int n, int width,
                                          int first, const int* columns) {
    const unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)n * width;
    if (k < total) b[k] = (int)(k % n) == columns[first + k / n] ? 1.0 : 0.0;
}

extern "C" __global__ void sparse_gather(const double* x, int n, int first,
                                        int width, const int* p, const int* i,
                                        const int* perm, double* z) {
    const int col = first + blockIdx.x;
    if (col >= first + width) return;
    for (int k = p[col] + threadIdx.x; k < p[col + 1]; k += blockDim.x)
        z[k] = x[perm[i[k]] + (unsigned long long)n * (col - first)];
}

extern "C" __global__ void sparse_gather_diagonal(const double* x, int n, int first,
                                                 int width, const int* columns,
                                                 double* z) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) z[first + col] = x[columns[first + col] + (unsigned long long)n * col];
}
