// vector_add.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Утилита для проверки ошибок CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    int N = (argc > 1) ? atoi(argv[1]) : 1 << 20; // по умолчанию 1M элементов
    size_t bytes = N * sizeof(float);

    // Выделение памяти на хосте
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Инициализация
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Выделение памяти на устройстве
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Копирование на устройство
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Настройка исполнения
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Запуск ядра
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копирование результата обратно
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Простая проверка
    bool ok = true;
    for (int i = 0; i < std::min(N, 10); ++i) {
        if (h_C[i] != 3.0f) ok = false;
    }
    std::cout << "Vector addition " << (ok ? "PASSED" : "FAILED") << " (first 10 elements)\n";

    // Очистка
    free(h_A); free(h_B); free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
