extern "C" __global__ void laplace_scatter(const unsigned long long* tasks, int rows, int columns) {
    const unsigned long long index = static_cast<unsigned long long>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (index >= static_cast<unsigned long long>(columns)*columns) return;
    const auto* task = tasks + static_cast<unsigned long long>(blockIdx.y)*3;
    const double* design = reinterpret_cast<const double*>(task[0]);
    const double* vectors = reinterpret_cast<const double*>(task[1]);
    double* out = reinterpret_cast<double*>(task[2]);
    const int r = index/columns, c = index%columns;
    double h = 0.0, g = 0.0;
    if (r >= c) for (int i = 0; i < rows; ++i) {
        const double a = design[static_cast<unsigned long long>(i)*columns+r];
        h += a*vectors[rows+i]*design[static_cast<unsigned long long>(i)*columns+c];
        if (r == c) g += a*vectors[i];
    }
    out[index] = h;
    if (r == c) out[static_cast<unsigned long long>(columns)*columns+r] = g;
}
