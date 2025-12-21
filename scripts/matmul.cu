// matmul.cu
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

// Наивное умножение
__global__ void matmulNaive(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Tiled версия
const int TILE_SIZE = 16;

__global__ void matmulTiled(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Загрузка тайлов в shared memory
        if (row < N && (tile * TILE_SIZE + tx) < N)
            As[ty][tx] = A[row * N + tile * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && (tile * TILE_SIZE + ty) < N)
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Вычисление частичного скалярного произведения
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}

void initializeMatrix(float* M, int N, float val = 1.0f) {
    for (int i = 0; i < N * N; ++i) M[i] = val;
}

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 2048; // 512x512 по умолчанию
    size_t bytes = N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    initializeMatrix(h_A, N, 1.0f);
    initializeMatrix(h_B, N, 2.0f);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Наивная версия
    dim3 blockNaive(16, 16);
    dim3 gridNaive((N + blockNaive.x - 1) / blockNaive.x,
                   (N + blockNaive.y - 1) / blockNaive.y);

    auto start = std::chrono::steady_clock::now();
    matmulNaive<<<gridNaive, blockNaive>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto naive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    // Tiled версия
    dim3 blockTiled(TILE_SIZE, TILE_SIZE);
    dim3 gridTiled((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    start = std::chrono::steady_clock::now();
    matmulTiled<<<gridTiled, blockTiled>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto tiled_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << "Naive matmul (" << N << "x" << N << "): " << naive_time << " ms\n";
    std::cout << "Tiled matmul (" << N << "x" << N << "): " << tiled_time << " ms\n";

    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
