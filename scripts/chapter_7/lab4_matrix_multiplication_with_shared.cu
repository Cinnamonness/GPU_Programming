#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// размер блока
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE float

// Базовое ядро для умножения матриц без shared memory
__global__ void matrixMult(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, 
                          int Arows, int Acols, int Bcols)
{
    // Вычисление индекса элемента матрицы на GPU
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < Arows && col < Bcols) {
        BASE_TYPE sum = 0;
        for (int k = 0; k < Acols; k++) {
            sum += A[row * Acols + k] * B[k * Bcols + col];
        }
        C[row * Bcols + col] = sum;
    }
}

// Оптимизированное ядро для умножения матриц с использованием shared memory
__global__ void matrixMultTiled(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, 
                               int Arows, int Acols, int Bcols)
{
    // Выделение разделяемой памяти для подматриц
    __shared__ BASE_TYPE As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ BASE_TYPE Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Вычисление индексов элемента в результирующей матрице
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;
    
    BASE_TYPE sum = 0.0;
    
    // Количество шагов (подматриц)
    int numSteps = (Acols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int m = 0; m < numSteps; ++m) {
        // Загрузка элементов подматрицы A в shared memory
        int aCol = m * BLOCK_SIZE + tx;
        int aIndex = row * Acols + aCol;
        if (row < Arows && aCol < Acols) {
            As[ty][tx] = A[aIndex];
        } else {
            As[ty][tx] = 0.0;
        }
        
        // Загрузка элементов подматрицы B в shared memory
        int bRow = m * BLOCK_SIZE + ty;
        int bIndex = bRow * Bcols + col;
        if (bRow < Acols && col < Bcols) {
            Bs[ty][tx] = B[bIndex];
        } else {
            Bs[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Вычисление суммы произведений
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Запись результата
    if (row < Arows && col < Bcols) {
        C[row * Bcols + col] = sum;
    }
}

// Функция умножения матриц на CPU
void matrixMultCPU(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, 
                  int Arows, int Acols, int Bcols)
{
    for (int i = 0; i < Arows; i++) {
        for (int j = 0; j < Bcols; j++) {
            BASE_TYPE sum = 0;
            for (int k = 0; k < Acols; k++) {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            C[i * Bcols + j] = sum;
        }
    }
}

// Функция вычисления числа, которое больше числа а и кратное числу b
int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        mod = b - mod;
        return a + mod;
    }
    return a;
}

// Функция проверки результатов
int verifyResults(const BASE_TYPE *cpu_result, const BASE_TYPE *gpu_result, 
                 int rows, int cols, int max_errors_to_show)
{
    int errors = 0;
    const int samples_to_check = 100;
    
    for (int k = 0; k < samples_to_check; k++) {
        int i = rand() % rows;
        int j = rand() % cols;
        int ind = i * cols + j;
        
        BASE_TYPE diff = fabs(cpu_result[ind] - gpu_result[ind]);
        if (diff > 1e-3) {
            if (errors < max_errors_to_show) {
                printf("ОШИБКА: C[%d][%d]: CPU=%.6f, GPU=%.6f, разница=%.6f\n", 
                       i, j, cpu_result[ind], gpu_result[ind], diff);
            }
            errors++;
        }
    }
    
    return errors;
}

// Функция для проверки ошибок CUDA
bool checkCudaError(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        return false;
    }
    return true;
}

// Главная функция
int main()
{            
    // Создаем события CUDA для замера времени
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_cpu, stop_cpu;
    cudaEvent_t start_mem, stop_mem;
    cudaEvent_t start_kernel, stop_kernel;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&start_mem);
    cudaEventCreate(&stop_mem);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    
    // Начало общего времени выполнения
    cudaEventRecord(start_total, 0);
    
    int Arows = 500;
    int Acols = 400;
    int Brows = Acols;  // Для умножения матриц количество столбцов A должно равняться количеству строк B
    int Bcols = 300;
    
    printf("ПАРАМЕТРЫ ЗАДАЧИ УМНОЖЕНИЯ МАТРИЦ\n");
    printf("  Размер матрицы A: %d x %d\n", Arows, Acols);
    printf("  Размер матрицы B: %d x %d\n", Brows, Bcols);
    printf("  Размер результирующей матрицы C: %d x %d\n", Arows, Bcols);
    printf("  Размер блока: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // Выравниваем размеры для оптимальной работы с блоками
    int Arows_aligned = toMultiple(Arows, BLOCK_SIZE);
    int Acols_aligned = toMultiple(Acols, BLOCK_SIZE);
    int Brows_aligned = toMultiple(Brows, BLOCK_SIZE);
    int Bcols_aligned = toMultiple(Bcols, BLOCK_SIZE);
    
    printf("\nВЫРОВНЕННЫЕ РАЗМЕРЫ:\n");
    printf("  Матрица A: %d x %d\n", Arows_aligned, Acols_aligned);
    printf("  Матрица B: %d x %d\n", Brows_aligned, Bcols_aligned);
    printf("  Матрица C: %d x %d\n", Arows_aligned, Bcols_aligned);
    printf("  Общее количество элементов в C: %d\n", Arows_aligned * Bcols_aligned);
    
    size_t Asize = Arows_aligned * Acols_aligned * sizeof(BASE_TYPE);
    size_t Bsize = Brows_aligned * Bcols_aligned * sizeof(BASE_TYPE);
    size_t Csize = Arows_aligned * Bcols_aligned * sizeof(BASE_TYPE);
    
    printf("  Размер данных:\n");
    printf("    - Матрица A: %.2f MB\n", (float)Asize / (1024 * 1024));
    printf("    - Матрица B: %.2f MB\n", (float)Bsize / (1024 * 1024));
    printf("    - Матрица C: %.2f MB\n", (float)Csize / (1024 * 1024));
    printf("    - Всего: %.2f MB\n\n", (float)(Asize + Bsize + Csize) / (1024 * 1024));
    
    printf("ВЫДЕЛЕНИЕ ПАМЯТИ И ИНИЦИАЛИЗАЦИЯ\n");
    
    // Выделение памяти под матрицы на хосте
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(Asize);
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(Bsize);
    BASE_TYPE *h_C_cpu = (BASE_TYPE *)malloc(Csize);
    BASE_TYPE *h_C_gpu_basic = (BASE_TYPE *)malloc(Csize);
    BASE_TYPE *h_C_gpu_tiled = (BASE_TYPE *)malloc(Csize);
    
    if (h_A == NULL || h_B == NULL || h_C_cpu == NULL || 
        h_C_gpu_basic == NULL || h_C_gpu_tiled == NULL) {
        fprintf(stderr, "Ошибка выделения памяти на CPU!\n");
        return 1;
    }
    
    // Инициализация матриц случайными числами
    printf("Инициализация матриц случайными числами...\n");
    for (int i = 0; i < Arows_aligned * Acols_aligned; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    for (int i = 0; i < Brows_aligned * Bcols_aligned; ++i) {
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    
    printf("\nУМНОЖЕНИЕ НА CPU\n");
    
    cudaEventRecord(start_cpu, 0);
    matrixMultCPU(h_A, h_B, h_C_cpu, Arows_aligned, Acols_aligned, Bcols_aligned);
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    printf("Время умножения на CPU: %.6f мс\n", cpu_time);
    
    printf("\nВЫПОЛНЕНИЕ НА GPU\n");
    
    // Выделение глобальной памяти на девайсе
    BASE_TYPE *d_A = NULL;
    BASE_TYPE *d_B = NULL;
    BASE_TYPE *d_C = NULL;
    
    cudaEventRecord(start_mem, 0);
    
    cudaError_t cudaStatus;
    bool success = true;
    
    // Выделение памяти на GPU с проверкой ошибок
    success = success && checkCudaError(cudaMalloc((void **)&d_A, Asize), "cudaMalloc for d_A");
    success = success && checkCudaError(cudaMalloc((void **)&d_B, Bsize), "cudaMalloc for d_B");
    success = success && checkCudaError(cudaMalloc((void **)&d_C, Csize), "cudaMalloc for d_C");
    
    // Копирование данных на GPU с проверкой ошибок
    success = success && checkCudaError(cudaMemcpy(d_A, h_A, Asize, cudaMemcpyHostToDevice), "cudaMemcpy for d_A");
    success = success && checkCudaError(cudaMemcpy(d_B, h_B, Bsize, cudaMemcpyHostToDevice), "cudaMemcpy for d_B");
    
    if (!success) {
        // Освобождение ресурсов при ошибке
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        return 1;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    float mem_time_1;
    cudaEventElapsedTime(&mem_time_1, start_mem, stop_mem);
    printf("Время выделения памяти и копирования на GPU: %.6f мс\n", mem_time_1);
    
    // Определяем размер блока и сетки
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((Bcols_aligned + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                       (Arows_aligned + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("\nКОНФИГУРАЦИЯ ЗАПУСКА:\n");
    printf("  Блоков в сетке: (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
    printf("  Потоков в блоке: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("  Всего блоков: %d\n", blocksPerGrid.x * blocksPerGrid.y);
    printf("  Всего потоков: %d\n", blocksPerGrid.x * blocksPerGrid.y * 
                                   threadsPerBlock.x * threadsPerBlock.y);
    
    const int iterations = 100;
    printf("  Количество итераций ядра: %d\n", iterations);
    
    // Переменные для хранения времени выполнения
    float kernel_time_basic = 0.0f;
    float avg_kernel_time_basic = 0.0f;
    float kernel_time_tiled = 0.0f;
    float avg_kernel_time_tiled = 0.0f;
    
    // ТЕСТИРОВАНИЕ БАЗОВОГО ЯДРА
    printf("\n=== БАЗОВОЕ ЯДРО (БЕЗ SHARED MEMORY) ===\n");
    
    cudaEventRecord(start_kernel, 0);
    for (int iter = 0; iter < iterations; iter++) {
        matrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Arows_aligned, Acols_aligned, Bcols_aligned);
        cudaStatus = cudaGetLastError();
        if (!checkCudaError(cudaStatus, "kernel launch")) {
            success = false;
            break;
        }
    }
    
    if (success) {
        success = checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }
    
    if (success) {
        cudaEventRecord(stop_kernel, 0);
        cudaEventSynchronize(stop_kernel);
        
        cudaEventElapsedTime(&kernel_time_basic, start_kernel, stop_kernel);
        avg_kernel_time_basic = kernel_time_basic / iterations;
        
        printf("Время выполнения базового ядра:\n");
        printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time_basic);
        printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time_basic);
        
        // Копируем результат базового ядра
        success = checkCudaError(cudaMemcpy(h_C_gpu_basic, d_C, Csize, cudaMemcpyDeviceToHost), 
                                "cudaMemcpy for basic kernel results");
    }
    
    // ТЕСТИРОВАНИЕ ОПТИМИЗИРОВАННОГО ЯДРА
    if (success) {
        printf("\n=== ОПТИМИЗИРОВАННОЕ ЯДРО (SHARED MEMORY) ===\n");
        
        cudaEventRecord(start_kernel, 0);
        for (int iter = 0; iter < iterations; iter++) {
            matrixMultTiled<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, Arows_aligned, Acols_aligned, Bcols_aligned);
            cudaStatus = cudaGetLastError();
            if (!checkCudaError(cudaStatus, "optimized kernel launch")) {
                success = false;
                break;
            }
        }
        
        if (success) {
            success = checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize for optimized kernel");
        }
        
        if (success) {
            cudaEventRecord(stop_kernel, 0);
            cudaEventSynchronize(stop_kernel);
            
            cudaEventElapsedTime(&kernel_time_tiled, start_kernel, stop_kernel);
            avg_kernel_time_tiled = kernel_time_tiled / iterations;
            
            printf("Время выполнения оптимизированного ядра:\n");
            printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time_tiled);
            printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time_tiled);
            
            // Копируем результат оптимизированного ядра
            success = checkCudaError(cudaMemcpy(h_C_gpu_tiled, d_C, Csize, cudaMemcpyDeviceToHost), 
                                    "cudaMemcpy for optimized kernel results");
        }
    }
    
    // Общее время выполнения
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    
    if (success) {
        printf("\n=== СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ ===\n");
        
        // Сравнение времени выполнения
        printf("Время выполнения умножения:\n");
        printf("  CPU (одно умножение):                %9.6f мс\n", cpu_time);
        printf("  GPU базовое ядро (среднее):          %9.6f мс\n", avg_kernel_time_basic);
        printf("  GPU оптимизированное ядро (среднее): %9.6f мс\n", avg_kernel_time_tiled);
        
        if (cpu_time > 0 && avg_kernel_time_basic > 0 && avg_kernel_time_tiled > 0) {
            double speedup_basic = cpu_time / avg_kernel_time_basic;
            double speedup_tiled = cpu_time / avg_kernel_time_tiled;
            double speedup_shared = avg_kernel_time_basic / avg_kernel_time_tiled;
            
            printf("\nКОЭФФИЦИЕНТ УСКОРЕНИЯ:\n");
            printf("  Базовое ядро GPU/CPU:           %.3f x\n", speedup_basic);
            printf("  Оптимизированное ядро GPU/CPU:  %.3f x\n", speedup_tiled);
            printf("  Shared memory ускорило в:       %.3f x\n", speedup_shared);
            
            // Дополнительная статистика
            long long operations_per_mult = (long long)Arows_aligned * Bcols_aligned * Acols_aligned;
            double cpu_gflops = (operations_per_mult / (cpu_time / 1000.0)) / 1e9;
            double basic_gflops = (operations_per_mult / (avg_kernel_time_basic / 1000.0)) / 1e9;
            double tiled_gflops = (operations_per_mult / (avg_kernel_time_tiled / 1000.0)) / 1e9;
            
            printf("\nПРОИЗВОДИТЕЛЬНОСТЬ (GFLOPS):\n");
            printf("  CPU:                            %.2f GFLOPS\n", cpu_gflops);
            printf("  GPU базовое ядро:               %.2f GFLOPS\n", basic_gflops);
            printf("  GPU оптимизированное ядро:      %.2f GFLOPS\n", tiled_gflops);
        }
        
        printf("\n=== ПРОВЕРКА КОРРЕКТНОСТИ РЕЗУЛЬТАТОВ ===\n");
        
        // Проверка базового ядра
        int errors_basic = verifyResults(h_C_cpu, h_C_gpu_basic, Arows_aligned, Bcols_aligned, 5);
        if (errors_basic == 0) {
            printf("Базовое ядро: Умножение выполнено корректно\n");
        } else {
            printf("Базовое ядро: Обнаружено %d ошибок\n", errors_basic);
        }
        
        // Проверка оптимизированного ядра
        int errors_tiled = verifyResults(h_C_cpu, h_C_gpu_tiled, Arows_aligned, Bcols_aligned, 5);
        if (errors_tiled == 0) {
            printf("Оптимизированное ядро: Умножение выполнено корректно\n");
        } else {
            printf("Оптимизированное ядро: Обнаружено %d ошибок\n", errors_tiled);
        }
        
        // Вывод примера умножения
        if (errors_basic == 0 && errors_tiled == 0) {
            printf("\nПРИМЕР УМНОЖЕНИЯ (первые 2x2 элемента):\n");
            printf("Матрица A[0-1][0-1]:\n");
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    printf("%8.4f ", h_A[i * Acols_aligned + j]);
                }
                printf("\n");
            }
            
            printf("\nМатрица B[0-1][0-1]:\n");
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    printf("%8.4f ", h_B[i * Bcols_aligned + j]);
                }
                printf("\n");
            }
            
            printf("\nРезультат C[0-1][0-1] (A × B):\n");
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    printf("%8.4f ", h_C_gpu_tiled[i * Bcols_aligned + j]);
                }
                printf("\n");
            }
        }
        
        printf("\nПРОГРАММА ЗАВЕРШЕНА УСПЕШНО\n");
    }
    
    // Освобождение ресурсов
    if (d_A) cudaFree(d_A);
    if (d_B) cudaFree(d_B);
    if (d_C) cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu_basic);
    free(h_C_gpu_tiled);
    
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_mem);
    cudaEventDestroy(stop_mem);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    
    if (!success) {
        printf("\nПРОГРАММА ЗАВЕРШЕНА С ОШИБКАМИ\n");
        return 1;
    }
    
    return 0;
}