#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Ядро для создания матрицы на GPU
__global__ void createMatrix(int *A, const int n)
{
    // Создание элементов матрицы на GPU
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        A[row * n + col] = 10 * row + col;
    }
}

// Функция для создания матрицы на CPU
void createMatrixCPU(int *A, const int n)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = 10 * i + j;  
}

// Главная функция main()
int main()
{    
    // РАЗМЕР МАТРИЦЫ 5000x5000
    const int n = 5000;
    // размер матрицы
    size_t size = n * n * sizeof(int);
    
    printf("Параметры задачи:\n");
    printf("  Размер матрицы: %d x %d\n", n, n);
    printf("  Всего элементов: %d\n", n * n);
    printf("  Общий размер данных: %.2f MB\n\n", size / (1024.0 * 1024.0));

    printf("ПОДГОТОВКА ДАННЫХ НА CPU\n");
    
    // Создаем события для замера времени CPU
    cudaEvent_t start_cpu, stop_cpu;
    cudaEventCreate(&start_cpu);
    cudaEventCreate(&stop_cpu);
    
    // выделяем память для матрицы на CPU
    int *h_A = (int *)malloc(size);
    int *h_B = (int *)malloc(size);
    
    // Замер времени создания матрицы на CPU
    cudaEventRecord(start_cpu, 0);
    
    // инициализируем матрицу - двойной цикл по строкам и столбцам
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            h_A[i * n + j] = 10 * i + j;
    
    cudaEventRecord(stop_cpu, 0);
    cudaEventSynchronize(stop_cpu);
    
    float cpu_time;
    cudaEventElapsedTime(&cpu_time, start_cpu, stop_cpu);
    printf("Время создания матрицы на CPU: %.6f мс\n", cpu_time);

    printf("\nВЫПОЛНЕНИЕ НА GPU\n");
    
    // Создаем события CUDA для замера времени GPU
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_malloc, stop_malloc;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_memcpy, stop_memcpy;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_malloc);
    cudaEventCreate(&stop_malloc);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_memcpy);
    cudaEventCreate(&stop_memcpy);
    
    float total_gpu_time, malloc_time, kernel_time, memcpy_time;
    
    // Начало общего времени GPU
    cudaEventRecord(start_total, 0);
    
    int *d_B = NULL; // d_B - указатель на память GPU
    
    // Замер времени выделения памяти на GPU
    cudaEventRecord(start_malloc, 0);
    
    // выделяем память для матрицы на GPU
    cudaError_t cudaStatus = cudaMalloc((void **)&d_B, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    cudaEventRecord(stop_malloc, 0);
    cudaEventSynchronize(stop_malloc);
    cudaEventElapsedTime(&malloc_time, start_malloc, stop_malloc);
    printf("Время выделения памяти на GPU: %.6f мс\n", malloc_time);

    // определение размеров сетки и блоков для большой матрицы
    dim3 threadsPerBlock(16, 16); // блок из 16*16 потоков
    dim3 blocksPerGrid((n + 15) / 16, (n + 15) / 16); // = (313, 313) блоков
    
    printf("\nКонфигурация запуска ядра:\n");
    printf("  Блоков в сетке: (%d, %d, %d)\n", blocksPerGrid.x, blocksPerGrid.y, blocksPerGrid.z);
    printf("  Потоков в блоке: (%d, %d, %d)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
    printf("  Всего потоков: %d\n", blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z * 
                                   threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z);
    printf("  Всего операций: %d (по одной на поток)\n", n * n);

    // Замер времени выполнения ядра
    cudaEventRecord(start_kernel, 0);
    
    // 1000 раз вызываем ядро
    const int iterations = 1000;
    printf("  Количество итераций ядра: %d\n", iterations);
    
    for (int iter = 0; iter < iterations; iter++) {
        createMatrix<<<blocksPerGrid, threadsPerBlock>>>(d_B, n);
        
        // Проверяем ошибки выполнения ядра
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed on iteration %d: %s\n", iter, cudaGetErrorString(cudaStatus));
            return 1;
        }
    }
    
    // Ждем завершения всех ядер
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }
    
    cudaEventRecord(stop_kernel, 0);
    cudaEventSynchronize(stop_kernel);
    cudaEventElapsedTime(&kernel_time, start_kernel, stop_kernel);
    
    // Вычисляем среднее время выполнения одного ядра
    float avg_kernel_time = kernel_time / iterations;
    printf("Время выполнения ядра на GPU:\n");
    printf("  Общее время %d итераций: %.6f мс\n", iterations, kernel_time);
    printf("  Среднее время одной итерации: %.6f мс\n", avg_kernel_time);

    // Замер времени копирования данных обратно на CPU
    cudaEventRecord(start_memcpy, 0);
    
    // копируем матрицу из GPU на CPU
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop_memcpy, 0);
    cudaEventSynchronize(stop_memcpy);
    cudaEventElapsedTime(&memcpy_time, start_memcpy, stop_memcpy);
    printf("Время копирования данных с GPU на CPU: %.6f мс\n", memcpy_time);
    
    // Конец общего времени GPU
    cudaEventRecord(stop_total, 0);
    cudaEventSynchronize(stop_total);
    cudaEventElapsedTime(&total_gpu_time, start_total, stop_total);
    
    printf("\nАНАЛИЗ ВРЕМЕНИ ВЫПОЛНЕНИЯ\n");
    
    printf("Общее время GPU: %.6f мс\n", total_gpu_time);
    printf("  - Выделение памяти GPU: %7.6f мс (%5.1f%%)\n", 
           malloc_time, (malloc_time/total_gpu_time)*100);
    printf("  - Выполнение ядра (%d итераций): %7.6f мс (%5.1f%%)\n", 
           iterations, kernel_time, (kernel_time/total_gpu_time)*100);
    printf("  - Копирование результатов: %7.6f мс (%5.1f%%)\n", 
           memcpy_time, (memcpy_time/total_gpu_time)*100);
    printf("  - Прочие операции:      %7.6f мс (%5.1f%%)\n", 
           total_gpu_time - malloc_time - kernel_time - memcpy_time,
           ((total_gpu_time - malloc_time - kernel_time - memcpy_time)/total_gpu_time)*100);

    printf("\nПРОВЕРКА РЕЗУЛЬТАТОВ\n");
    
    int errors = 0;    
    const int samples_to_check = 100;
    for (int k = 0; k < samples_to_check; k++) {
        int i = rand() % n;
        int j = rand() % n;
        int expected = 10 * i + j;
        int actual_gpu = h_B[i * n + j];
        
        if (actual_gpu != expected) {
            if (errors < 3) {
                printf("ОШИБКА: [%d][%d] - ожидалось: %d, получено: %d\n", 
                       i, j, expected, actual_gpu);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Матрицы идентичны. Ошибок не обнаружено (проверено %d элементов).\n", samples_to_check);
        printf("\nДемонстрация первых 4x4 элементов матрицы:\n");
        printf("Формат: [строка][столбец] = значение\n");
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                printf("[%d][%d]=%4d  ", i, j, h_A[i * n + j]);
            }
            printf("\n");
        }
    } else {
        printf("Обнаружено %d ошибок (проверено %d элементов)\n", errors, samples_to_check);
        if (errors > 3) {
            printf("Показаны только первые 3 ошибки\n");
        }
    }

    printf("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ\n");
    
    printf("Время выполнения:\n");
    printf("  CPU (одна матрица):          %9.6f мс\n", cpu_time);
    printf("  GPU (общее с накладными):    %9.6f мс\n", total_gpu_time);
    printf("  GPU (вычисления, %d матриц): %9.6f мс\n", iterations, kernel_time);
    printf("  GPU (среднее на матрицу):    %9.6f мс\n", avg_kernel_time);
    
    if (cpu_time > 0 && avg_kernel_time > 0) {
        double speedup_total = cpu_time / total_gpu_time;
        double speedup_kernel = cpu_time / avg_kernel_time;
        double speedup_iterations = (cpu_time * iterations) / kernel_time;
        
        printf("\nКоэффициент ускорения:\n");
        printf("  Общее ускорение (CPU/GPU общее):           %.3f x\n", speedup_total);
        printf("  Ускорение вычислений (CPU/GPU на матрицу): %.3f x\n", speedup_kernel);
        printf("  Общая производительность (%d матриц):      %.3f x\n", iterations, speedup_iterations);
        
        if (speedup_total > 1.0) {
            printf("\nGPU БЫСТРЕЕ CPU в %.1f раз\n", speedup_total);
        } else if (speedup_kernel > 1.0) {
            printf("\nВычисления на GPU быстрее в %.1f раз\n", speedup_kernel);
        } else {
            printf("\nЗадача слишком проста для GPU\n");
        }
        
        printf("  Всего операций на CPU: %d\n", n * n);
        printf("  Всего операций на GPU: %lld\n", (long long)n * n * iterations);
        printf("  Операций в секунду (CPU): %.0f\n", (n * n) / (cpu_time / 1000.0));
        printf("  Операций в секунду (GPU): %.0f\n", ((long long)n * n * iterations) / (kernel_time / 1000.0));
    }

    printf("\nРАСХОДЫ НА ПЕРЕДАЧУ ДАННЫХ:\n");
    printf("   - Копирование %.2f MB на GPU: %.6f мс\n", size / (1024.0 * 1024.0), malloc_time);
    printf("   - Копирование %.2f MB с GPU: %.6f мс\n", size / (1024.0 * 1024.0), memcpy_time);
    printf("   - Итого на передачу данных: %.6f мс (%.1f%% общего времени)\n", 
           malloc_time + memcpy_time, 
           ((malloc_time + memcpy_time)/total_gpu_time)*100);
    
    // освобождаем память на GPU
    cudaFree(d_B);
    
    // освобождаем память на CPU
    free(h_A);
    free(h_B);
    
    // уничтожаем события CUDA
    cudaEventDestroy(start_cpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_malloc);
    cudaEventDestroy(stop_malloc);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_memcpy);
    cudaEventDestroy(stop_memcpy);    
    return 0;
}