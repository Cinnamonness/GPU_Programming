#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// размер блока
#define BLOCK_SIZE 16
// тип, который будут иметь элементы матриц
#define BASE_TYPE float

// Ядро для транспонирования матрицы
__global__ void matrixTranspose(const BASE_TYPE *A, BASE_TYPE *AT, int rows, int cols)
{
    // Индекс элемента в исходной матрице A
    // Формула: строка * cols + столбец
    int iA = cols * (blockDim.y * blockIdx.y + threadIdx.y) +  // глобальная координата Y (строка)
             blockDim.x * blockIdx.x + threadIdx.x;            // глобальная координата X (столбец)
    
    // Индекс соответствующего элемента в транспонированной матрице AT
    // Формула: столбец * rows + строка (меняем местами строки и столбцы)
    int iAT = rows * (blockDim.x * blockIdx.x + threadIdx.x) + // глобальная координата X становится строкой
              blockDim.y * blockIdx.y + threadIdx.y;           // глобальная координата Y становится столбцом
    
    // Копируем элемент из A в AT с изменением позиции
    AT[iAT] = A[iA];
}

// Функция транспонирования на CPU для сравнения производительности
void matrixTransposeCPU(const BASE_TYPE *A, BASE_TYPE *AT, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            AT[j * rows + i] = A[i * cols + j];  // Меняем местами индексы
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
    printf("ПОДГОТОВКА ДАННЫХ И ПАРАМЕТРОВ\n");
    
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
    
    // Исходные размеры матрицы
    int rows = 1000;
    int cols = 2000;
    
    printf("Исходные размеры матрицы: %d x %d\n", rows, cols);
    printf("Размер блока: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    
    // Выравниваем размеры для оптимальной работы с блоками
    rows = toMultiple(rows, BLOCK_SIZE);
    cols = toMultiple(cols, BLOCK_SIZE);
    
    printf("Выровненные размеры матрицы: %d x %d\n", rows, cols);
    printf("Общее количество элементов: %d\n", rows * cols);
    
    size_t size = rows * cols * sizeof(BASE_TYPE);
    printf("Общий размер данных: %.2f MB\n", (float)size / (1024 * 1024));
    
    printf("\nВЫДЕЛЕНИЕ ПАМЯТИ И ИНИЦИАЛИЗАЦИЯ\n");
    
    // Выделение памяти под матрицы на хосте
    BASE_TYPE *h_A = (BASE_TYPE *)malloc(size);      // Исходная матрица
    BASE_TYPE *h_AT_cpu = (BASE_TYPE *)malloc(size); // Транспонированная (CPU)
    BASE_TYPE *h_AT_gpu = (BASE_TYPE *)malloc(size); // Транспонированная (GPU)
    
    if (h_A == NULL || h_AT_cpu == NULL || h_AT_gpu == NULL) {
        fprintf(stderr, "Ошибка выделения памяти на CPU!\n");
        return 1;
    }
    
    // Инициализация исходной матрицы случайными числами
    for (int i = 0; i < rows * cols; ++i) {
        h_A[i] = rand() / (BASE_TYPE)RAND_MAX;
    }
    
    printf("\nТРАНСПОНИРОВАНИЕ НА CPU\n");
    
    cudaEventRecord(start_cpu, 0);
    
    matrixTransposeCPU(h_A, h_AT_cpu, rows, cols);
    
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    printf("Время транспонирования на CPU: %.6f мс\n", cpu_time);
    
    printf("\nВЫПОЛНЕНИЕ НА GPU\n");
    
    // Выделение глобальной памяти на девайсе
    BASE_TYPE *d_A = NULL;   // Исходная матрица на GPU
    BASE_TYPE *d_AT = NULL;  // Транспонированная матрица на GPU
    
    cudaEventRecord(start_mem, 0);
    
    cudaError_t cudaStatus = cudaMalloc((void **)&d_A, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_A: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    cudaStatus = cudaMalloc((void **)&d_AT, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for d_AT: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        return 1;
    }
    
    // Копируем исходную матрицу из CPU на GPU
    cudaStatus = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_AT);
        return 1;
    }
    
    cudaEventRecord(stop_mem, 0);
    cudaEventSynchronize(stop_mem);
    
    float mem_time_1;
    cudaEventElapsedTime(&mem_time_1, start_mem, stop_mem);
    printf("Время выделения памяти и копирования на GPU: %.6f мс\n", mem_time_1);
    
    printf("\nКОНФИГУРАЦИЯ И ЗАПУСК ЯДРА\n");
    
    // Определяем размер блока и сетки
    dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid = dim3(cols / BLOCK_SIZE, rows / BLOCK_SIZE);
    
    printf("Конфигурация запуска:\n");
    printf("  Блоков в сетке: (%d, %d, %d)\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
    printf("  Потоков в блоке: (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    printf("  Всего блоков: %d\n", blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z);
    printf("  Всего потоков: %d\n", blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z * 
                                   threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z);
    
    // Запуск ядра транспонирования
    cudaEventRecord(start_kernel, 0);
    
    matrixTranspose<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_AT, rows, cols);
    
    // Проверяем ошибки запуска ядра
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_AT);
        return 1;
    }
    
    // Ждем завершения работы ядра
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_AT);
        return 1;
    }
    
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    
    float kernel_time;
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    printf("Время выполнения ядра на GPU: %.6f мс\n", kernel_time);
    
    printf("\nКОПИРОВАНИЕ РЕЗУЛЬТАТОВ И ОБРАБОТКА\n");
    
    cudaEventRecord(start_mem, 0);
    
    // Копируем транспонированную матрицу из GPU на CPU
    cudaStatus = cudaMemcpy(h_AT_gpu, d_AT, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(cudaStatus));
        cudaFree(d_A);
        cudaFree(d_AT);
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
    
    printf("\nДЕТАЛЬНЫЙ АНАЛИЗ ВРЕМЕНИ ВЫПОЛНЕНИЯ\n");
    
    float total_mem_time = mem_time_1 + mem_time_2;
    float other_time = total_time - cpu_time - total_mem_time - kernel_time;
    
    printf("Общее время выполнения: %.6f мс\n", total_time);
    printf("Детализация времени GPU:\n");
    printf("  - Работа с памятью:      %7.6f мс (%5.1f%%)\n", 
           total_mem_time, (total_mem_time/total_time)*100);
    printf("  - Выполнение ядра:       %7.6f мс (%5.1f%%)\n", 
           kernel_time, (kernel_time/total_time)*100);
    printf("  - Транспонирование на CPU: %7.6f мс (%5.1f%%)\n", 
           cpu_time, (cpu_time/total_time)*100);
    printf("  - Прочие операции:       %7.6f мс (%5.1f%%)\n", 
           other_time, (other_time/total_time)*100);
    
    printf("\nПРОВЕРКА КОРРЕКТНОСТИ РЕЗУЛЬТАТОВ\n");
    
    int errors = 0;
    const int max_errors_to_show = 5;
    
    // Проверяем правильность транспонирования
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            BASE_TYPE original = h_A[i * cols + j];
            BASE_TYPE transposed_gpu = h_AT_gpu[j * rows + i];
            BASE_TYPE transposed_cpu = h_AT_cpu[j * rows + i];
            
            // Проверяем совпадение GPU и CPU результатов
            if (fabs(transposed_gpu - transposed_cpu) > 1e-5) {
                if (errors < max_errors_to_show) {
                    printf("ОШИБКА: [%d][%d] -> [%d][%d]: CPU=%.6f, GPU=%.6f\n", 
                           i, j, j, i, transposed_cpu, transposed_gpu);
                }
                errors++;
            }
        }
    }
    
    if (errors == 0) {
        printf("Транспонирование выполнено корректно\n");
        printf("\nПример транспонирования (первые 3x3 элемента):\n");
        printf("Исходная матрица A[0-2][0-2]:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%8.4f ", h_A[i * cols + j]);
            }
            printf("\n");
        }
        
        printf("\nТранспонированная матрица AT[0-2][0-2]:\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%8.4f ", h_AT_gpu[i * rows + j]);
            }
            printf("\n");
        }
    } else {
        printf("Обнаружено %d ошибок\n", errors);
        if (errors > max_errors_to_show) {
            printf("Показаны только первые %d ошибок\n", max_errors_to_show);
        }
    }
    
    printf("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ\n");
    
    printf("Время выполнения транспонирования:\n");
    printf("  CPU (последовательно):          %9.6f мс\n", cpu_time);
    printf("  GPU (только ядро):              %9.6f мс\n", kernel_time);
    printf("  GPU (общее с накладными):       %9.6f мс\n", total_time - cpu_time);
    
    if (cpu_time > 0 && kernel_time > 0) {
        double speedup_kernel = cpu_time / kernel_time;
        double speedup_total = cpu_time / (total_time - cpu_time);
        
        printf("\nКоэффициент ускорения:\n");
        printf("  Ускорение вычислений (CPU/GPU ядро):    %.3f x\n", speedup_kernel);
        printf("  Общее ускорение (CPU/GPU с накладными): %.3f x\n", speedup_total);
        
        if (speedup_total > 1.0) {
            printf("\nВЫВОД: GPU обеспечивает ускорение для этой задачи\n");
        } else {
            printf("\nВЫВОД: Накладные расходы GPU превышают выгоду\n");
        }
    }
        
    // Освобождаем память на GPU
    cudaFree(d_A);
    cudaFree(d_AT);
    
    // Освобождаем память на CPU
    free(h_A);
    free(h_AT_cpu);
    free(h_AT_gpu);
    
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