#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// размер блока
#define BLOCK_SIZE 256
// тип, который будут иметь элементы векторов
#define BASE_TYPE float

// Пустое warm-up ядро для инициализации CUDA контекста
__global__ void warmupKernel()
{
    // Пустое ядро - ничего не делает
    // Нужно только для инициализации CUDA runtime
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Пустая операция чтобы компилятор не оптимизировал ядро
    if (tid < 0) {
        printf("This should never happen\n");
    }
}

// Ядро для скалярного произведения с использованием глобальной памяти
__global__ void dotProductGlobal(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int numElem)
{
    // Глобальный индекс для каждой нити
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Каждая нить вычисляет частичную сумму
    BASE_TYPE partial_sum = 0.0;
    
    if (globalIdx < numElem) {
        partial_sum = A[globalIdx] * B[globalIdx];
    }
    
    // Используем атомарную операцию для суммирования результатов
    atomicAdd(C, partial_sum);
}

// Ядро для скалярного произведения с использованием разделяемой (shared) памяти
__global__ void dotProductShared(const BASE_TYPE *A, const BASE_TYPE *B, BASE_TYPE *C, int numElem)
{
    // Создание массива в разделяемой памяти для частичных сумм
    __shared__ BASE_TYPE partial_sums[BLOCK_SIZE];
    
    // Глобальный индекс
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Каждая нить вычисляет свой элемент
    BASE_TYPE thread_sum = 0.0;
    if (globalIdx < numElem) {
        thread_sum = A[globalIdx] * B[globalIdx];
    } else {
        thread_sum = 0.0;  // Явно обнуляем для выровненных элементов
    }
    
    // Сохраняем результат в разделяемую память
    partial_sums[threadIdx.x] = thread_sum;
    
    // Синхронизация нитей в блоке
    __syncthreads();
    
    // Редукция в разделяемой памяти (суммирование элементов в блоке)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Первая нить блока добавляет сумму блока в глобальную память
    if (threadIdx.x == 0) {
        atomicAdd(C, partial_sums[0]);
    }
}

// Функция скалярного произведения на CPU
BASE_TYPE dotProductCPU(const BASE_TYPE *A, const BASE_TYPE *B, int numElem)
{
    BASE_TYPE sum = 0.0;
    for (int i = 0; i < numElem; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

// Функция вычисления числа, которое больше числа а и кратное числу b
int toMultiple(int a, int b)
{
    int mod = a % b;
    if (mod != 0)
    {
        return a + (b - mod);
    }
    return a;
}

// Главная функция
int main()
{
    // Объявляем переменные
    int numElem = 1000000;
    int alignedNumElem = toMultiple(numElem, BLOCK_SIZE);
    size_t vectorSize = alignedNumElem * sizeof(BASE_TYPE);
    size_t resultSize = sizeof(BASE_TYPE);
    
    // Переменные для конфигурации и времени
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (alignedNumElem + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int iterations = 100;
    BASE_TYPE zero = 0.0;
    
    float kernel_time_global = 0.0f;
    float avg_kernel_time_global = 0.0f;
    float kernel_time_shared = 0.0f;
    float avg_kernel_time_shared = 0.0f;
    float total_mem_time = 0.0f;
    float mem_time_1 = 0.0f;
    float mem_time_2 = 0.0f;
    float total_time = 0.0f;
    float cpu_time = 0.0f;
    
    BASE_TYPE tolerance = 1e-2;
    int global_correct = 0;
    int shared_correct = 0;
    
    // Указатели для хоста
    BASE_TYPE *h_A = NULL;
    BASE_TYPE *h_B = NULL;
    BASE_TYPE h_C_cpu = 0.0;
    BASE_TYPE h_C_global = 0.0;
    BASE_TYPE h_C_shared = 0.0;
    
    // Указатели для устройства
    BASE_TYPE *d_A = NULL;
    BASE_TYPE *d_B = NULL;
    BASE_TYPE *d_C_global = NULL;
    BASE_TYPE *d_C_shared = NULL;
    
    // События CUDA
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_cpu, stop_cpu;
    cudaEvent_t start_mem, stop_mem;
    cudaEvent_t start_kernel_global, stop_kernel_global;
    cudaEvent_t start_kernel_shared, stop_kernel_shared;
    
    cudaError_t cudaStatus = cudaSuccess;
    
    // Инициализация событий CUDA
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    cudaEventCreate(&start_mem);
    cudaEventCreate(&stop_mem);
    cudaEventCreate(&start_kernel_global);
    cudaEventCreate(&stop_kernel_global);
    cudaEventCreate(&start_kernel_shared);
    cudaEventCreate(&stop_kernel_shared);
    
    // Начало общего времени выполнения
    cudaEventRecord(start_total, 0);
    
    printf("ПАРАМЕТРЫ ЗАДАЧИ СКАЛЯРНОГО ПРОИЗВЕДЕНИЯ ВЕКТОРОВ\n");
    printf("  Размер векторов: %d элементов\n", numElem);
    printf("  Размер блока: %d\n", BLOCK_SIZE);
    printf("  Тип данных: %s\n", sizeof(BASE_TYPE) == 4 ? "float" : "double");
    
    printf("\nВЫРОВНЕННЫЕ РАЗМЕРЫ:\n");
    printf("  Исходный размер: %d\n", numElem);
    printf("  Выровненный размер: %d\n", alignedNumElem);
    printf("  Дополнительных элементов: %d\n", alignedNumElem - numElem);
    
    printf("  Размер данных:\n");
    printf("    - Вектор A: %.2f MB\n", (float)vectorSize / (1024 * 1024));
    printf("    - Вектор B: %.2f MB\n", (float)vectorSize / (1024 * 1024));
    printf("    - Результат: %zu bytes\n", resultSize);
    printf("    - Всего: %.2f MB\n\n", (float)(2 * vectorSize + resultSize) / (1024 * 1024));
    
    printf("ВЫДЕЛЕНИЕ ПАМЯТИ И ИНИЦИАЛИЗАЦИЯ\n");
    
    // Выделение памяти под векторы на хосте
    h_A = (BASE_TYPE *)malloc(vectorSize);
    h_B = (BASE_TYPE *)malloc(vectorSize);
    
    if (h_A == NULL || h_B == NULL) {
        fprintf(stderr, "Ошибка выделения памяти на CPU!\n");
        cudaStatus = cudaErrorMemoryAllocation;
        goto cleanup_before_init;
    }
    
    // Инициализация векторов случайными числами
    printf("Инициализация векторов случайными числами...\n");
    for (int i = 0; i < alignedNumElem; ++i) {
        if (i < numElem) {
            h_A[i] = (rand() % 100) / 100.0f;
            h_B[i] = (rand() % 100) / 100.0f;
        } else {
            // Заполняем выровненные элементы нулями
            h_A[i] = 0.0;
            h_B[i] = 0.0;
        }
    }
    
    printf("\nСКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ НА CPU\n");
    
    cudaEventRecord(start_cpu, 0);
    
    h_C_cpu = dotProductCPU(h_A, h_B, numElem);
    
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    printf("Время вычисления на CPU: %.6f мс\n", cpu_time);
    printf("Результат CPU: %.6f\n", h_C_cpu);
    
    printf("\nВЫПОЛНЕНИЕ НА GPU\n");
    
    // Выделение глобальной памяти на девайсе
    cudaEventRecord(start_mem, 0);
    
    cudaStatus = cudaMalloc((void **)&d_A, vectorSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_before_init;
    }
    
    cudaStatus = cudaMalloc((void **)&d_B, vectorSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_B: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_dA;
    }
    
    cudaStatus = cudaMalloc((void **)&d_C_global, resultSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C_global: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_dB;
    }
    
    cudaStatus = cudaMalloc((void **)&d_C_shared, resultSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_C_shared: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_dC_global;
    }
    
    // Копируем векторы из CPU на GPU
    cudaStatus = cudaMemcpy(d_A, h_A, vectorSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed for d_A: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaStatus = cudaMemcpy(d_B, h_B, vectorSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed for d_B: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    cudaEventElapsedTime(&mem_time_1, start_mem, stop_mem);
    printf("Время выделения памяти и копирования на GPU: %.6f мс\n", mem_time_1);
    
    printf("\nWARM-UP (ИНИЦИАЛИЗАЦИЯ CUDA КОНТЕКСТА)\n");
    
    // Запускаем пустое ядро для инициализации CUDA runtime
    printf("Запуск warm-up ядра для инициализации CUDA контекста...\n");
    warmupKernel<<<1, 1>>>();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Warm-up kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize after warm-up failed: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    printf("Warm-up завершен успешно\n");
    
    printf("\nКОНФИГУРАЦИЯ И ЗАПУСК ЯДЕР\n");
    
    printf("Конфигурация запуска:\n");
    printf("  Блоков в сетке: %d\n", blocksPerGrid);
    printf("  Потоков в блоке: %d\n", threadsPerBlock);
    printf("  Всего потоков: %d\n", blocksPerGrid * threadsPerBlock);
    printf("  Количество итераций ядер: %d\n", iterations);
    
    printf("\nЯДРО С ГЛОБАЛЬНОЙ ПАМЯТЬЮ\n");
    
    cudaMemcpy(d_C_global, &zero, resultSize, cudaMemcpyHostToDevice);
    
    // Запуск ядра с глобальной памятью многократно
    cudaEventRecord(start_kernel_global, 0);
    
    for (int iter = 0; iter < iterations; iter++) {
        if (iter > 0) { // Сбрасываем результат только после первой итерации
            cudaMemcpy(d_C_global, &zero, resultSize, cudaMemcpyHostToDevice);
        }
        
        dotProductGlobal<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_global, numElem);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Global kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto cleanup_after_all_alloc;
        }
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaEventRecord(stop_kernel_global, 0);
    cudaEventSynchronize(stop_kernel_global);
    
    cudaEventElapsedTime(&kernel_time_global, start_kernel_global, stop_kernel_global);
    avg_kernel_time_global = kernel_time_global / iterations;
    
    printf("Время выполнения ядра с глобальной памятью:\n");
    printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time_global);
    printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time_global);
    
    printf("\nЯДРО С РАЗДЕЛЯЕМОЙ ПАМЯТЬЮ\n");
    
    cudaMemcpy(d_C_shared, &zero, resultSize, cudaMemcpyHostToDevice);
    
    // Запуск ядра с разделяемой памятью многократно
    cudaEventRecord(start_kernel_shared, 0);
    
    for (int iter = 0; iter < iterations; iter++) {
        if (iter > 0) {
            cudaMemcpy(d_C_shared, &zero, resultSize, cudaMemcpyHostToDevice);
        }
        
        dotProductShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C_shared, numElem);
        
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Shared kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto cleanup_after_all_alloc;
        }
    }
    
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaEventRecord(stop_kernel_shared, 0);
    cudaEventSynchronize(stop_kernel_shared);
    
    cudaEventElapsedTime(&kernel_time_shared, start_kernel_shared, stop_kernel_shared);
    avg_kernel_time_shared = kernel_time_shared / iterations;
    
    printf("Время выполнения ядра с разделяемой памятью:\n");
    printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time_shared);
    printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time_shared);
    
    printf("\nКОПИРОВАНИЕ РЕЗУЛЬТАТОВ\n");
    
    cudaEventRecord(start_mem, 0);
    
    // Копируем результаты из GPU на CPU
    cudaStatus = cudaMemcpy(&h_C_global, d_C_global, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed for global result: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaStatus = cudaMemcpy(&h_C_shared, d_C_shared, resultSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed for shared result: %s\n", cudaGetErrorString(cudaStatus));
        goto cleanup_after_all_alloc;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    cudaEventElapsedTime(&mem_time_2, start_mem, stop_mem);
    printf("Время копирования результатов с GPU на CPU: %.6f мс\n", mem_time_2);
    
    // Общее время выполнения
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    
    cudaEventElapsedTime(&total_time, start_total, stop_total);
    
    printf("\nАНАЛИЗ ВРЕМЕНИ ВЫПОЛНЕНИЯ\n");
    
    total_mem_time = mem_time_1 + mem_time_2;
    
    printf("Общее время выполнения: %.6f мс\n", total_time);
    printf("Детализация времени GPU:\n");
    printf("  - Работа с памятью:                  %7.6f мс\n", total_mem_time);
    printf("  - Ядро с глобальной памятью:         %7.6f мс\n", kernel_time_global);
    printf("  - Ядро с разделяемой памятью:        %7.6f мс\n", kernel_time_shared);
    
    printf("\nПРОВЕРКА КОРРЕКТНОСТИ РЕЗУЛЬТАТОВ\n");
    
    printf("Результаты вычислений:\n");
    printf("  CPU:                 %.6f\n", h_C_cpu);
    printf("  Global:              %.6f\n", h_C_global);
    printf("  Shared:              %.6f\n", h_C_shared);
    
    global_correct = fabs(h_C_global - h_C_cpu) < tolerance;
    shared_correct = fabs(h_C_shared - h_C_cpu) < tolerance;
    
    printf("\nПроверка точности (допуск: %.6f):\n", tolerance);
    printf("  Глобальная память:          %s (ошибка: %.6f)\n", 
           global_correct ? "КОРРЕКТНО" : "ОШИБКА", fabs(h_C_global - h_C_cpu));
    printf("  Разделяемая память:         %s (ошибка: %.6f)\n", 
           shared_correct ? "КОРРЕКТНО" : "ОШИБКА", fabs(h_C_shared - h_C_cpu));
    
    printf("\nСРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ\n");
    
    printf("Среднее время выполнения одной операции:\n");
    printf("  CPU:                             %9.6f мс\n", cpu_time);
    printf("  GPU (глобальная память):         %9.6f мс\n", avg_kernel_time_global);
    printf("  GPU (разделяемая память):        %9.6f мс\n", avg_kernel_time_shared);
    
    if (cpu_time > 0) {
        double speedup_global = cpu_time / avg_kernel_time_global;
        double speedup_shared = cpu_time / avg_kernel_time_shared;
        
        printf("\nКОЭФФИЦИЕНТ УСКОРЕНИЯ:\n");
        printf("  Глобальная память:           %.3f x быстрее CPU\n", speedup_global);
        printf("  Разделяемая память:          %.3f x быстрее CPU\n", speedup_shared);
        
        // Сравнение между GPU реализациями
        if (avg_kernel_time_global > 0) {
            double improvement_shared = (avg_kernel_time_global - avg_kernel_time_shared) / avg_kernel_time_global * 100;
            
            printf("\nСРАВНЕНИЕ РЕАЛИЗАЦИЙ GPU:\n");
            printf("  Разделяемая память быстрее глобальной на: %.1f%%\n", improvement_shared);
            
            long long total_operations = (long long)numElem * iterations;
            
            printf("\nДОПОЛНИТЕЛЬНАЯ СТАТИСТИКА:\n");
            printf("  Операций на одно умножение: %d FLOP\n", numElem);
            printf("  Всего операций на GPU: %lld FLOP\n", total_operations);
            
            if (improvement_shared > 0) {
                printf("\nВЫВОД: РАЗДЕЛЯЕМАЯ ПАМЯТЬ ЗНАЧИТЕЛЬНО УСКОРЯЕТ ВЫЧИСЛЕНИЯ!\n");
            }
        }
    }
    
    printf("\nРАСХОДЫ НА ПЕРЕДАЧУ ДАННЫХ:\n");
    printf("   - Копирование %.2f MB на GPU: %.6f мс\n", (float)(2 * vectorSize) / (1024 * 1024), mem_time_1);
    printf("   - Копирование результатов с GPU: %.6f мс\n", mem_time_2);
    printf("   - Итого на передачу данных: %.6f мс\n", total_mem_time);

cleanup_after_all_alloc:
    // Освобождаем память на GPU
    if (d_C_shared) cudaFree(d_C_shared);
cleanup_after_dC_global:
    if (d_C_global) cudaFree(d_C_global);
cleanup_after_dB:
    if (d_B) cudaFree(d_B);
cleanup_after_dA:
    if (d_A) cudaFree(d_A);
cleanup_before_init:
    // Освобождаем память на CPU
    if (h_A) free(h_A);
    if (h_B) free(h_B);
    
    // Уничтожаем события CUDA
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_mem);
    cudaEventDestroy(stop_mem);
    cudaEventDestroy(start_kernel_global);
    cudaEventDestroy(stop_kernel_global);
    cudaEventDestroy(start_kernel_shared);
    cudaEventDestroy(stop_kernel_shared);
    
    return (cudaStatus == cudaSuccess) ? 0 : 1;
}