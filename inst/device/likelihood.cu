extern "C" __global__ void tulpa_likelihood(
    int n, int np, int width, const int* beta_start, const int* beta_count,
    const double* x, const double* offset, const int* rowptr, const int* column,
    const double* projection, const double* response, const double* constants,
    const double* parameters, double* scores, double* curvature, double* partial,
    double* predictors) {
    __shared__ double sums[256];
    const int chain = blockIdx.y;
    const double* par = parameters + (long long)chain*width;
    double sum = 0;
    for (int i = blockIdx.x*blockDim.x+threadIdx.x; i<n; i+=gridDim.x*blockDim.x) {
        double eta[8], score[8], hessian[64], value;
        long long xoff = 0;
        for (int k=0; k<np; ++k) {
            const int row = k*n+i;
            double e = offset[row];
            for (int j=0; j<beta_count[k]; ++j) e += x[xoff+(long long)j*n+i]*par[beta_start[k]+j];
            xoff += (long long)n*beta_count[k];
            for (int j=rowptr[row]; j<rowptr[row+1]; ++j) e += projection[j]*par[column[j]];
            eta[k] = e;
        }
        tulpa_likelihood_row(response,n,i,eta,constants,value,score,hessian);
        if (predictors) for (int k=0; k<np; ++k) predictors[((long long)chain*n+i)*np+k]=eta[k];
        sum += value;
        for (int k=0; k<np; ++k) scores[((long long)chain*np+k)*n+i] = score[k];
        if (curvature) for (int k=0; k<np*np; ++k)
            curvature[((long long)chain*n+i)*np*np+k] = hessian[k];
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int step=128; step; step/=2) {
        if (threadIdx.x<step) sums[threadIdx.x] += sums[threadIdx.x+step];
        __syncthreads();
    }
    if (!threadIdx.x) partial[(long long)chain*gridDim.x+blockIdx.x] = sums[0];
}

extern "C" __global__ void tulpa_likelihood_scatter(
    int n, int np, int width, const int* colptr, const int* row,
    const double* weight, const double* score, double* gradient) {
    __shared__ double sums[256];
    const int chain = blockIdx.y;
    for (int col=blockIdx.x; col<width; col+=gridDim.x) {
        double sum = 0;
        for (int j=colptr[col]+threadIdx.x; j<colptr[col+1]; j+=blockDim.x)
            sum += weight[j]*score[(long long)chain*n*np+row[j]];
        sums[threadIdx.x] = sum;
        __syncthreads();
        for (int step=128; step; step/=2) {
            if (threadIdx.x<step) sums[threadIdx.x] += sums[threadIdx.x+step];
            __syncthreads();
        }
        if (!threadIdx.x) gradient[(long long)chain*width+col] = sums[0];
        __syncthreads();
    }
}
