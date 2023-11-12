#include <cuda.h>
#include <stdio.h>

// Matrix Vector Multiplication with GPUs
#define N 16
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NWARPS 32
#define ROWS_PER_BLOCK 4

__global__
void matVecMulMultiWarp(double *matrix, double *vector, double *result, int rows, int cols) {
    __shared__ double s_mem[1024/WARP_SIZE];

    size_t row = blockIdx.x;
    int nwarps = blockDim.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    
    for (int i = lane; i < N; i += WARP_SIZE) { // TODO: figure out indexing
        sum += matrix[(row * cols) + i] * vector[i];
    }

    __syncthreads();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (lane == 0) {
        s_mem[warp_id] = sum;
    }

    double f_sum = 0.0;
    if (threadIdx.x == 0) {
        for (int i = 0; i < nwarps; i++) {
            f_sum += s_mem[i];
        }
        result[row] = f_sum;
    }
}

__global__
void matVecMulMultiRowOneBlock(double *matrix, double *vector, double *result, int rows, int cols) {
    size_t row = blockIdx.x * (blockDim.x/WARP_SIZE) + threadIdx.x / WARP_SIZE; // TODO: GET ROW
    int lane = threadIdx.x % WARP_SIZE;

    // printf("row: %d, lane: %d, thread: %d\n", (int) row, lane, threadIdx.x);

    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    
    for (int i = lane; i < N; i += WARP_SIZE) {
        sum += matrix[(row * cols) + i] * vector[i];
    }
    // printf("temp sum: %g\n", sum);

    __syncthreads();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        // printf("warping\n");
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    // printf("end sum: %g\n", sum);

    if (threadIdx.x == 0) {
        result[row] = sum;
    }
}

__global__
void matVecMulOneWarp(double *matrix, double *vector, double *result, int rows, int cols) {
    size_t row = blockIdx.x;
    int lane = threadIdx.x % WARP_SIZE ;

    // printf("row: %d, lane: %d, thread: %d\n", (int) row, lane, threadIdx.x);

    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    
    for (int i = lane; i < N; i += WARP_SIZE) {
        sum += matrix[(row * cols) + i] * vector[i];
    }
    // printf("temp sum: %g\n", sum);

    __syncthreads();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        // printf("warping\n");
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }
    // printf("end sum: %g\n", sum);

    if (threadIdx.x == 0) {
        result[row] = sum;
    }
}

__global__
void matVecMulNoWarp(double *matrix, double *vector, double *result, int rows, int cols) {
    size_t row = threadIdx.x;
    
    if (row < rows) {
        // printf("Row: %d\n", (int) row);
        double sum = 0.0;
        for (int i = 0; i < cols; i++) {
            sum += matrix[(row * cols) + i] * vector[i];
            // printf("%d, %g, %g\n", (row * cols) + i, vector[i], sum);
        }
        result[row] = sum;
    }
}  

void mat_vec_mul() {
    // double *vx_h = new double[N];
    // double *vy_h = new double[N];
    // double *vx_d, *vy_d; 

    // double **mat = new double *[N];
    // for (int i = 0; i< n; i++) {
    //     mat[i] = new double[N];
    // }

    // double **mat_d;
    double vec_h[N] = {0};
    double res_h[N] = {0};
    double mat_h[N*N] = {0};
    for (int i = 0; i < N; i++) {
        mat_h[i*N+i] = 1;
        vec_h[i] = i+1;
    }

    double *vec_d;
    double *res_d;
    double *mat_d;

    // allocate memory on device
    cudaMalloc( (void**) &vec_d, sizeof(double)*N);
    cudaMalloc( (void**) &res_d, sizeof(double)*N);
    cudaMalloc( (void**) &mat_d, sizeof(double)*N*N);

    //copy data from HOST to DEVICE
    cudaMemcpy(vec_d,vec_h,sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(mat_d,mat_h,sizeof(double)*N*N,cudaMemcpyHostToDevice);

    // // NO WARP
    // dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
    // dim3 nblocks ((N + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
    // matVecMulNoWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, N, N);

    // // ONE WARP
    // dim3 nthreads(WARP_SIZE, 1, 1); // Threads per block 
    // dim3 nblocks (N, 1, 1); // one block per row
    // matVecMulOneWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, N, N);

    // // Multiple Rows per Block
    dim3 nthreads(WARP_SIZE*ROWS_PER_BLOCK, 1, 1);
    dim3 nblocks((N + ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, 1, 1); 
    matVecMulMultiRowOneBlock<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, N, N);

    // // Multiple Warps 
    // dim3 nthreads(WARP_SIZE * NWARPS, 1, 1);
    // dim3 nblocks (N, 1, 1); // one block per row

    cudaDeviceSynchronize();

    //copy data from DEVICE to HOST
    cudaMemcpy(res_h,res_d,sizeof(double)*N,cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("y_h[%d] = %g\n", i, res_h[i]);
    }

    //free memory 
    cudaFree(res_d);
    cudaFree(vec_d);
    cudaFree(mat_d);
}

int main() {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    printf("ncuda_devices = %d\n",ncuda_devices);

    if (ncuda_devices == 0) {
        fprintf(stderr,"NO CUDA DEVICES EXITING\n");
        return 0;
    }
    cudaSetDevice(0);

    mat_vec_mul();
}