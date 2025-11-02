#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// размер блока
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE float

// Ядро для сложения матриц
__global__ void matrixAdd(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int rows, int cols)
{
    // Вычисление индекса элемента матрицы на GPU
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int ind = row * cols + col;
        C[ind] = A[ind] + B[ind];
    }
}

// Функция сложения матриц на CPU
void matrixAddCPU(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            int ind = i * cols + j;
            C[ind] = A[ind] + B[ind];
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
    
    int rows = 5000;
    int cols = 8000;
    
    printf("Параметры задачи:\n");
    printf("  Размер матриц: %d x %d\n", rows, cols);
    printf("  Размер блока: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // Выравниваем размеры для оптимальной работы с блоками
    rows = toMultiple(rows, BLOCK_SIZE);
    cols = toMultiple(cols, BLOCK_SIZE);
    
    printf("  Выровненные размеры: %d x %d\n", rows, cols);
    printf("  Общее количество элементов: %d\n", rows * cols);
    
    size_t size = rows * cols * sizeof(BASE_TYPE);
    printf("  Общий размер данных (одна матрица): %.2f MB\n", (float)size / (1024 * 1024));
    printf("  Общий размер всех данных: %.2f MB\n\n", (float)(3 * size) / (1024 * 1024));
    
    printf("ВЫДЕЛЕНИЕ ПАМЯТИ И ИНИЦИАЛИЗАЦИЯ\n");
    
    // Выделение памяти под матрицы на хосте
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);      // Матрица A
    BASE_TYPE *h_B = (BASE_TYPE *)malloc(size);      // Матрица B
    BASE_TYPE *h_C_cpu = (BASE_TYPE *)malloc(size);  // Результат (CPU)
    BASE_TYPE *h_C_gpu = (BASE_TYPE *)malloc(size);  // Результат (GPU)
    
    if (h_A == NULL || h_B == NULL || h_C_cpu == NULL || h_C_gpu == NULL) {
        fprintf(stderr, "Ошибка выделения памяти на CPU!\n");
        return 1;
    }
    
    // Инициализация матриц случайными числами
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
        h_B[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    
    printf("\nСЛОЖЕНИЕ НА CPU\n");
    
    cudaEventRecord(start_cpu, 0);
    
    matrixAddCPU(h_A, h_B, h_C_cpu, rows, cols);
    
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    printf("Время сложения на CPU: %.6f мс\n", cpu_time);
    
    printf("\nВЫПОЛНЕНИЕ НА GPU\n");
    
    // Выделение глобальной памяти на девайсе
    BASE_TYPE *d_A = NULL;   // Матрица A на GPU
    BASE_TYPE *d_B = NULL;   // Матрица B на GPU
    BASE_TYPE *d_C = NULL;   // Результат на GPU
    
    cudaEventRecord(start_mem, 0);
    
    cudaError_t cudaStatus = cudaMalloc((void **)&d_A, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    cudaStatus = cudaMalloc((void **)&d_B, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        return 1;
    }
    
    cudaStatus = cudaMalloc((void **)&d_C, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        return 1;
    }
    
    // Копируем матрицы из CPU на GPU
    cudaStatus = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed for d_A: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    
    cudaStatus = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed for d_B: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    float mem_time_1;
    cudaEventElapsedTime(&mem_time_1, start_mem, stop_mem);
    printf("Время выделения памяти и копирования на GPU: %.6f мс\n", mem_time_1);
    
    printf("\nКОНФИГУРАЦИЯ И ЗАПУСК ЯДРА\n");
    
    // Определяем размер блока и сетки
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                       (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    printf("Конфигурация запуска:\n");
    printf("  Блоков в сетке: (%d, %d)\n", blocksPerGrid.x, blocksPerGrid.y);
    printf("  Потоков в блоке: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("  Всего блоков: %d\n", blocksPerGrid.x * blocksPerGrid.y);
    printf("  Всего потоков: %d\n", blocksPerGrid.x * blocksPerGrid.y * 
                                   threadsPerBlock.x * threadsPerBlock.y);
    
    // многократный запуск ядра 1000 раз
    const int iterations = 1000;
    printf("  Количество итераций ядра: %d\n", iterations);
    
    // Запуск ядра сложения многократно
    cudaEventRecord(start_kernel, 0);
    
    for (int iter = 0; iter < iterations; iter++) {
        matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, rows, cols);
        
        // Проверяем ошибки запуска ядра
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed on iteration %d: %s\n", iter, cudaGetErrorString(cudaStatus));
            cudaFree(d_A);
            cudaFree(d_B);
            cudaFree(d_C);
            return 1;
        }
    }
    
    // Ждем завершения работы всех ядер
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    
    // Вычисляем среднее время выполнения одного ядра
    float avg_kernel_time = kernel_time / iterations;
    printf("Время выполнения ядра на GPU:\n");
    printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time);
    printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time);
    
    printf("\nКОПИРОВАНИЕ РЕЗУЛЬТАТОВ\n");
    
    cudaEventRecord(start_mem, 0);
    
    // Копируем результат из GPU на CPU
    cudaStatus = cudaMemcpy(h_C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 1;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    float mem_time_2;
    cudaEventElapsedTime(&mem_time_2, start_mem, stop_mem);
    printf("Время копирования результатов с GPU на CPU: %.6f мс\n", mem_time_2);
    
    // Общее время выполнения
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    
    float total_time;
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    
    printf("\nАНАЛИЗ ВРЕМЕНИ ВЫПОЛНЕНИЯ\n");
    
    float total_mem_time = mem_time_1 + mem_time_2;
    float gpu_total_time = total_time - cpu_time;
    float other_time = total_time - cpu_time - total_mem_time - kernel_time;
    
    printf("Общее время выполнения: %.6f мс\n", total_time);
    printf("Детализация времени GPU:\n");
    printf("  - Работа с памятью:           %7.6f мс (%5.1f%%)\n", 
           total_mem_time, (total_mem_time/gpu_total_time)*100);
    printf("  - Выполнение ядра (%d итераций): %7.6f мс (%5.1f%%)\n", 
           iterations, kernel_time, (kernel_time/gpu_total_time)*100);
    printf("  - Прочие операции:            %7.6f мс (%5.1f%%)\n", 
           other_time, (other_time/gpu_total_time)*100);
    
    printf("\nПРОВЕРКА КОРРЕКТНОСТИ РЕЗУЛЬТАТОВ\n");
    
    int errors = 0;
    const int max_errors_to_show = 5;
    const int samples_to_check = 100;
    
    // Проверяем правильность сложения для выборочных элементов
    for (int k = 0; k < samples_to_check; k++) {
        int i = rand() % rows;
        int j = rand() % cols;
        int ind = i * cols + j;
        BASE_TYPE expected = h_A[ind] + h_B[ind];
        BASE_TYPE result_gpu = h_C_gpu[ind];
        BASE_TYPE result_cpu = h_C_cpu[ind];
        
        // Проверяем совпадение GPU и CPU результатов
        if (fabs(result_gpu - result_cpu) > 1e-5) {
            if (errors < max_errors_to_show) {
                printf("ОШИБКА: [%d][%d]: A=%.6f + B=%.6f = CPU=%.6f, GPU=%.6f\n", 
                       i, j, h_A[ind], h_B[ind], result_cpu, result_gpu);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Сложение выполнено корректно (проверено %d элементов)\n", samples_to_check);
        printf("\nПример сложения (первые 3x3 элемента):\n");
        printf("Матрица A[0-2][0-2]:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int ind = i * cols + j;
                printf("%8.4f ", h_A[ind]);
            }
            printf("\n");
        }
        
        printf("\nМатрица B[0-2][0-2]:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int ind = i * cols + j;
                printf("%8.4f ", h_B[ind]);
            }
            printf("\n");
        }
        
        printf("\nРезультат C[0-2][0-2] (A + B):\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                int ind = i * cols + j;
                printf("%8.4f ", h_C_gpu[ind]);
            }
            printf("\n");
        }
    } else {
        printf("Обнаружено %d ошибок (проверено %d элементов)\n", errors, samples_to_check);
        if (errors > max_errors_to_show) {
            printf("Показаны только первые %d ошибок\n", max_errors_to_show);
        }
    }
    
    printf("\nАНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ\n");
    
    printf("Время выполнения сложения:\n");
    printf("  CPU (одно сложение):          %9.6f мс\n", cpu_time);
    printf("  GPU (общее с накладными):     %9.6f мс\n", gpu_total_time);
    printf("  GPU (вычисления, %d сложений): %9.6f мс\n", iterations, kernel_time);
    printf("  GPU (среднее на сложение):    %9.6f мс\n", avg_kernel_time);
    
    if (cpu_time > 0 && avg_kernel_time > 0) {
        double speedup_kernel = cpu_time / avg_kernel_time;
        double speedup_total = cpu_time / gpu_total_time;
        double speedup_iterations = (cpu_time * iterations) / kernel_time;
        
        printf("\nКоэффициент ускорения:\n");
        printf("  Общее ускорение (CPU/GPU общее):           %.3f x\n", speedup_total);
        printf("  Ускорение вычислений (CPU/GPU на операцию): %.3f x\n", speedup_kernel);
        printf("  Общая производительность (%d операций):    %.3f x\n", iterations, speedup_iterations);
        
        // Дополнительная статистика
        long long total_operations = (long long)rows * cols * iterations;
        double cpu_ops_per_sec = (rows * cols) / (cpu_time / 1000.0);
        double gpu_ops_per_sec = total_operations / (kernel_time / 1000.0);
        
        printf("\nДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:\n");
        printf("  Всего операций на CPU: %d\n", rows * cols);
        printf("  Всего операций на GPU: %lld\n", total_operations);
        printf("  FLOPs в секунду (CPU): %.0f\n", cpu_ops_per_sec);
        printf("  FLOPs в секунду (GPU): %.0f\n", gpu_ops_per_sec);
        
        if (speedup_total > 1.0) {
            printf("\nGPU БЫСТРЕЕ CPU в %.1f раз!\n", speedup_total);
        } else if (speedup_kernel > 1.0) {
            printf("\nВычисления на GPU быстрее в %.1f раз\n", speedup_kernel);
            printf("         Накладные расходы скрывают преимущество\n");
        } else {
            printf("\nЗадача слишком проста для GPU\n");
        }
    }
    
    printf("\nРАСХОДЫ НА ПЕРЕДАЧУ ДАННЫХ:\n");
    printf("   - Копирование %.2f MB на GPU: %.6f мс\n", (float)(2 * size) / (1024 * 1024), mem_time_1);
    printf("   - Копирование %.2f MB с GPU: %.6f мс\n", (float)size / (1024 * 1024), mem_time_2);
    printf("   - Итого на передачу данных: %.6f мс (%.1f%% общего времени GPU)\n", 
           total_mem_time, (total_mem_time/gpu_total_time)*100);
        
    // Освобождаем память на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Освобождаем память на CPU
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    
    // Уничтожаем события CUDA
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_mem);
    cudaEventDestroy(stop_mem);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
        
    return 0;
}