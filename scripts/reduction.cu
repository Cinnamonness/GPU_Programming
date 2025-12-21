// reduction.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Наивная редукция через shared memory
__global__ void reduceNaive(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Оптимизированная редукция через warp shuffle
__global__ void reduceShuffle(const float* input, float* output, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < n) ? input[i] : 0.0f;

    // Внутри варпа: итеративное суммирование
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Запись результата в shared memory (только первые нити блока по одному на варп)
    extern __shared__ float sdata2[];
    if (tid % 32 == 0) {
        sdata2[tid / 32] = val;
    }
    __syncthreads();

    // Редукция над результатами варпов (в первом варпе)
    if (tid < 32) {
        if (blockDim.x > 32) {
            val = (tid < (blockDim.x + 31) / 32) ? sdata2[tid] : 0.0f;
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
        }
        if (tid == 0) output[blockIdx.x] = val;
    }
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 20;
    size_t bytes = N * sizeof(float);

    float *h_in = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f;

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * (N + 255) / 256));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);

    // Тест наивной версии
    auto start = std::chrono::steady_clock::now();
    reduceNaive<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto naive_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();

    // Тест shuffle-версии
    start = std::chrono::steady_clock::now();
    reduceShuffle<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto shuffle_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << "Naive reduction: " << naive_time << " µs\n";
    std::cout << "Shuffle reduction: " << shuffle_time << " µs\n";

    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
