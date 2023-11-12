#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

// Matrix Vector Multiplication with GPUs
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NWARPS 32
#define ROWS_PER_BLOCK 4

enum Experiment {
    NO_WARP,
    ONE_WARP_ONE_ROW,
    ONE_WARP_MULTI_ROW,
    MULTI_WARP
};

typedef void (*MatVecMulFunc)(double *matrix, double *vector, double *result, int rows, int cols);

// randomization
std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator
std::uniform_real_distribution<double> distribution(1.0, 100.0);

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
    int row_offset = row * cols;
    for (int i = lane + (WARP_SIZE * warp_id); i < cols; i += WARP_SIZE * nwarps) { // TODO: figure out indexing
        sum += matrix[row_offset + i] * vector[i];
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

    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    int offset = row * cols;
    for (int i = lane; i < cols; i += WARP_SIZE) {
        sum += matrix[offset + i] * vector[i];
    }

    __syncthreads();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    if (threadIdx.x % WARP_SIZE == 0) {
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
    int offset = row * cols;
    for (int i = lane; i < cols; i += WARP_SIZE) {
        sum += matrix[offset + i] * vector[i];
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
        int offset = row * cols;
        for (int i = 0; i < cols; i++) {
            // printf("r,c = (%d, %d)\n", (int) row, i);
            sum += matrix[offset + i] * vector[i];
            // printf("%d, %g, %g\n", (row * cols) + i, vector[i], sum);
        }
        result[row] = sum;
    }
}  

double getFlopRate(int flops, int ms)
{
    double flopsms = ((double)flops) / ((double)ms);
    return flopsms / 1000000.0;
}

void instantiateMatVec(int m, int n, double *mat, double *vec) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        mat[n * i + j] = distribution(gen);  // Generate a random double value and store it in the matrix
    }
  }
  for (int i = 0; i < n; i++) {
    vec[i] = distribution(gen);
  }
}

void generateLatexTable(int* N, int* M, int* executionTime, double* flopRate, int size, std::string tableTitle, std::string output) {
    ofstream myfile;
    myfile.open(output);
    myfile << "\\begin{table}[htbp]\n";
    myfile << "  \\centering\n";
    myfile << "  \\caption{" << tableTitle << "}\n";
    myfile << "  \\begin{tabular}{|c|c|c|c|c|}\n";
    myfile << "    \\hline\n";
    myfile << "    \\multirow{2}{*}{N} & \\multirow{2}{*}{M} & \\multicolumn{2}{c|}{Performance Metrics} \\\\\n";
    myfile << "    \\cline{3-4}\n";
    myfile << "    & & Execution Time (ms) & Flop Rate (TFLOP/s) \\\\\n";
    myfile << "    \\hline\n";

    for (int i = 0; i < size; i++) {
        myfile << "    " << N[i] << " & " << M[i] << " & " << executionTime[i] << " & " << flopRate[i] << " \\\\\n";
    }

    myfile << "    \\hline\n";
    myfile << "  \\end{tabular}\n";
    myfile << "\\end{table}\n";
    myfile.close();
}

void runExperiment(Experiment e, std::string output) {

    int rowDims[] = {10, 10, 10, 10, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 10000, 10000, 10000};
    int colDims[] = {10, 100, 1000, 10000, 10, 100, 1000, 10000, 10, 100, 1000, 10000, 10, 100, 1000};
    // int rowDims[] = {10, 10, 100, 100};
    // int colDims[] = {10, 100, 10, 100};

    int size = sizeof(rowDims) / sizeof(rowDims[0]);
    int executionTimes[size];
    double flopRates[size];

    std::string algorithm;

    for (int i = 0; i < size; i++) {
        int M = rowDims[i];
        int N = colDims[i];

        double mat_h[M*N] = {0};
        double vec_h[N] = {0};
        double res_h[M] = {0};

        instantiateMatVec(M, N, mat_h, vec_h);

        double *mat_d;
        double *vec_d;
        double *res_d;
        
        // allocate memory on device
        cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
        cudaMalloc( (void**) &vec_d, sizeof(double)*N);
        cudaMalloc( (void**) &res_d, sizeof(double)*M);

        //copy data from HOST to DEVICE
        cudaMemcpy(vec_d,vec_h,sizeof(double)*N,cudaMemcpyHostToDevice);
        cudaMemcpy(mat_d,mat_h,sizeof(double)*M*N,cudaMemcpyHostToDevice);

        struct timeval startTime;
        struct timeval endTime;

        gettimeofday(&startTime, nullptr);  
        gettimeofday(&endTime, nullptr);

        if (e == NO_WARP) {
            algorithm = "No Warp";

            dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
            dim3 nblocks ((M + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
            gettimeofday(&startTime, nullptr);  
            matVecMulNoWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
        } else if (e == ONE_WARP_ONE_ROW) {
            algorithm = "One warp per row";

            dim3 nthreads(WARP_SIZE, 1, 1); // Threads per block 
            dim3 nblocks (M, 1, 1); // one block per row
            gettimeofday(&startTime, nullptr);  
            matVecMulOneWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
        } else if (e == ONE_WARP_MULTI_ROW) {
            algorithm = "Multiple rows per block";

            dim3 nthreads(WARP_SIZE*ROWS_PER_BLOCK, 1, 1);
            dim3 nblocks((M + ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, 1, 1); 
            gettimeofday(&startTime, nullptr);  
            matVecMulMultiRowOneBlock<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
        } else if (e == MULTI_WARP) {
            algorithm = "Multiple warps per row";

            dim3 nthreads(WARP_SIZE * NWARPS, 1, 1);
            dim3 nblocks (M, 1, 1); // one block per row
            gettimeofday(&startTime, nullptr);  
            matVecMulMultiWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
        }

        cudaDeviceSynchronize();
        gettimeofday(&endTime, nullptr);
        //copy data from DEVICE to HOST
        cudaMemcpy(res_h,res_d,sizeof(double)*M,cudaMemcpyDeviceToHost);

        double flops = 2 * M * N;
        int microseconds = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec - startTime.tv_usec);
        double floprate = getFlopRate(flops, microseconds);

        executionTimes[i] = microseconds;
        flopRates[i] = floprate;

        // for (int i = 0; i < M; i++) {
            // printf("y_h[%d] = %g\n", i, res_h[i]);
        // }

        std::cout << "M: " << M << ", N: " << N << ", Time: " << microseconds << ", Floprate: " << floprate << std::endl;

        //free memory 
        cudaFree(res_d);
        cudaFree(vec_d);
        cudaFree(mat_d);
    }

    generateLatexTable(rowDims, colDims, executionTimes, flopRates, size, algorithm, output);
}

void validate(Experiment e) {
    int M = 100;
    int N = 100;

    double mat_h[M*N] = {0};
    double vec_h[N] = {0};
    double res_h[M] = {0};

    instantiateMatVec(M, N, mat_h, vec_h);

    double *mat_d;
    double *vec_d;
    double *res_d;
    
    // allocate memory on device
    cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
    cudaMalloc( (void**) &vec_d, sizeof(double)*N);
    cudaMalloc( (void**) &res_d, sizeof(double)*M);

    //copy data from HOST to DEVICE
    cudaMemcpy(vec_d,vec_h,sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(mat_d,mat_h,sizeof(double)*M*N,cudaMemcpyHostToDevice);

    if (e == NO_WARP) {
        dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
        dim3 nblocks ((M + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
        matVecMulNoWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
    } else if (e == ONE_WARP_ONE_ROW) {
        dim3 nthreads(WARP_SIZE, 1, 1); // Threads per block 
        dim3 nblocks (M, 1, 1); // one block per row
        matVecMulOneWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
    } else if (e == ONE_WARP_MULTI_ROW) {

        dim3 nthreads(WARP_SIZE*ROWS_PER_BLOCK, 1, 1);
        dim3 nblocks((M + ROWS_PER_BLOCK-1)/ROWS_PER_BLOCK, 1, 1); 
        matVecMulMultiRowOneBlock<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
    } else if (e == MULTI_WARP) {
        dim3 nthreads(WARP_SIZE * NWARPS, 1, 1);
        dim3 nblocks (M, 1, 1); // one block per row
        matVecMulMultiWarp<<<nblocks, nthreads, 0, 0>>>(mat_d, vec_d, res_d, M, N);
    }

    cudaDeviceSynchronize();
    //copy data from DEVICE to HOST
    cudaMemcpy(res_h,res_d,sizeof(double)*M,cudaMemcpyDeviceToHost);

    double expected[M] = {0};
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        int offset = i * N;
        for (int j = 0; j < N; j++) {
            sum += mat_h[offset+j] * vec_h[j];
        }
        expected[i] = sum;
    }

    for (int i = 0; i < M; i++) {
        if (abs(expected[i] - res_h[i]) > 0.0001) {
            printf("DIFF FOUND: expected: %g, actual: %g", expected[i], res_h[i]);
        }
    }

    //free memory 
    cudaFree(res_d);
    cudaFree(vec_d);
    cudaFree(mat_d);
}


int main(int argc, char *argv[]) {

    if ((argc < 2) || (argc > 4))
    {
        std::cerr << "Usage: " << argv[0] << " <algorithm>" << std::endl;
        return 1; // Return an error code indicating incorrect usage
    }

    std::string output;
    if (argc == 4) {
        std::string flag = std::string(argv[2]);
        if (flag == "-o") {
            output = std::string(argv[3]);
        }
    }

    Experiment e;
    std::string algorithm = std::string(argv[1]);
    if (algorithm == "NO_WARP") {
        e = NO_WARP;
    } else if (algorithm == "ONE_WARP_ONE_ROW") {
        e = ONE_WARP_ONE_ROW;
    } else if (algorithm == "ONE_WARP_MULTI_ROW") {
        e = ONE_WARP_MULTI_ROW;
    } else if (algorithm == "MULTI_WARP") {
        e = MULTI_WARP;
    } else {
        std::cerr << "Invalid algorithm" << std::endl;
    }

    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    printf("ncuda_devices = %d\n",ncuda_devices);

    if (ncuda_devices == 0) {
        fprintf(stderr,"NO CUDA DEVICES EXITING\n");
        return 0;
    }
    cudaSetDevice(0);

    runExperiment(e, output);
    validate(e);
}