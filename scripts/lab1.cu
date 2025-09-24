#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1000000000 
#define PI 3.14159265358979323846

// вызов из CPU, выполнение на GPU
// Ядра для float
__global__ void sinfKernel_fast(float *arr, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        arr[i] = __sinf((i % 360) * PI / 180.0f);
}

__global__ void sinfKernel(float *arr, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        arr[i] = sinf((i % 360) * PI / 180.0f);
}

__global__ void sinKernel(float *arr, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        arr[i] = sin((i % 360) * PI / 180.0f);
}

// Ядра для double
__global__ void sinKernel_double(double *arr, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        arr[i] = sin((i % 360) * PI / 180.0);
}

// Функция для вычисления суммы ошибок на CPU
__host__ float computeError_float(float *arr, int n) {
    float err = 0;
    for (int i = 0; i < n; i++) {
        double exact = sin((i % 360) * PI / 180.0); 
        err += fabsf((float)exact - arr[i]);       // сравниваем с float результатом GPU
    }
    return err / n;
}

__host__ double computeError_double(double *arr, int n) {
    double err = 0;
    for (int i = 0; i < n; i++) {
        double exact = sin((i % 360) * PI / 180.0);
        err += fabs(exact - arr[i]);
    }
    return err / n;
}

// Функция для замера времени и выполнения ядра
template<typename T>
void runKernel(T *h_arr, T *d_arr, int n, void (*kernel)(T*, int), const char* name) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, n);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%s kernel execution time: %.3f ms\n", name, ms);

    cudaEventRecord(start);
    cudaMemcpy(h_arr, d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float copyTime = 0;
    cudaEventElapsedTime(&copyTime, start, stop);
    printf("%s data copy time: %.3f ms\n", name, copyTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n = N;
    size_t size_float = n * sizeof(float);
    size_t size_double = n * sizeof(double);

    // Массивы для float
    float *h_arr_f = (float*)malloc(size_float);
    float *d_arr_f;
    cudaMalloc((void**)&d_arr_f, size_float);

    // Сравнение sin
    runKernel(h_arr_f, d_arr_f, n, sinKernel, "sin");
    printf("sin error: %f\n", computeError_float(h_arr_f, n));

    // Сравнение sinf
    runKernel(h_arr_f, d_arr_f, n, sinfKernel, "sinf");
    printf("sinf error: %f\n", computeError_float(h_arr_f, n));

    // Сравнение __sinf
    runKernel(h_arr_f, d_arr_f, n, sinfKernel_fast, "__sinf");
    printf("__sinf error: %f\n", computeError_float(h_arr_f, n));

    cudaFree(d_arr_f);
    free(h_arr_f);

    // Массив для double
    double *h_arr_d = (double*)malloc(size_double);
    double *d_arr_d;
    cudaMalloc((void**)&d_arr_d, size_double);

    runKernel(h_arr_d, d_arr_d, n, sinKernel_double, "double sin");
    printf("double sin error: %lf\n", computeError_double(h_arr_d, n));

    cudaFree(d_arr_d);
    free(h_arr_d);

    return 0;
}

