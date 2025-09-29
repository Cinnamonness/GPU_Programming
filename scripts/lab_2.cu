#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <cuda_runtime.h>

// Структура для хранения изображения
struct Image {
    int width;
    int height;
    unsigned char *data;
};

// Ядро для сильного размытия (5x5 ядро)
__global__ void strongBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Большее ядро 5x5 для сильного размытия
    float kernel[5][5] = {
        {1/256.0f, 4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {6/256.0f, 24/256.0f, 36/256.0f, 24/256.0f, 6/256.0f},
        {4/256.0f, 16/256.0f, 24/256.0f, 16/256.0f, 4/256.0f},
        {1/256.0f, 4/256.0f,  6/256.0f,  4/256.0f, 1/256.0f}
    };
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    for(int ky = -2; ky <= 2; ky++) {
        for(int kx = -2; kx <= 2; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int idx = (ny * width + nx) * 4;
            
            float weight = kernel[ky + 2][kx + 2];
            r += input[idx + 0] * weight;
            g += input[idx + 1] * weight;
            b += input[idx + 2] * weight;
        }
    }
    
    int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = (unsigned char)min(max(r, 0.0f), 255.0f);
    output[out_idx + 1] = (unsigned char)min(max(g, 0.0f), 255.0f);
    output[out_idx + 2] = (unsigned char)min(max(b, 0.0f), 255.0f);
    output[out_idx + 3] = input[out_idx + 3];
}

// Ядро для обычного размытия (3x3)
__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    for(int ky = -1; ky <= 1; ky++) {
        for(int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);
            int idx = (ny * width + nx) * 4;
            
            float weight = kernel[ky + 1][kx + 1];
            r += input[idx + 0] * weight;
            g += input[idx + 1] * weight;
            b += input[idx + 2] * weight;
        }
    }
    
    int out_idx = (y * width + x) * 4;
    output[out_idx + 0] = (unsigned char)min(max(r, 0.0f), 255.0f);
    output[out_idx + 1] = (unsigned char)min(max(g, 0.0f), 255.0f);
    output[out_idx + 2] = (unsigned char)min(max(b, 0.0f), 255.0f);
    output[out_idx + 3] = input[out_idx + 3];
}

// Ядро для медианного фильтра
__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // Для граничных пикселей просто копируем
        int idx = (y * width + x) * 4;
        output[idx + 0] = input[idx + 0];
        output[idx + 1] = input[idx + 1];
        output[idx + 2] = input[idx + 2];
        output[idx + 3] = input[idx + 3];
        return;
    }
    
    // Обрабатываем каждый цветовой канал отдельно
    for (int channel = 0; channel < 3; channel++) {
        unsigned char window[9];
        int count = 0;
        
        // Собираем окрестность 3x3
        for(int ky = -1; ky <= 1; ky++) {
            for(int kx = -1; kx <= 1; kx++) {
                int idx = ((y + ky) * width + (x + kx)) * 4;
                window[count++] = input[idx + channel];
            }
        }
        
        // Сортировка пузырьком для нахождения медианы
        for(int i = 0; i < 8; i++) {
            for(int j = 0; j < 8 - i; j++) {
                if(window[j] > window[j + 1]) {
                    unsigned char temp = window[j];
                    window[j] = window[j + 1];
                    window[j + 1] = temp;
                }
            }
        }
        
        // Медиана - средний элемент
        int out_idx = (y * width + x) * 4;
        output[out_idx + channel] = window[4];
    }
    
    // Копируем альфа-канал
    int out_idx = (y * width + x) * 4;
    output[out_idx + 3] = input[out_idx + 3];
}

// Ядро для гауссового шума
__global__ void addGaussianNoiseKernel(unsigned char* input, unsigned char* output, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    // Простой генератор шума
    unsigned int noise_seed = (x * 123 + y * 456 + seed) % 123456;
    float noise = (float)(noise_seed % 100) / 100.0f * 50.0f - 25.0f; // ±25
    
    for (int channel = 0; channel < 3; channel++) {
        float value = input[idx + channel] + noise;
        output[idx + channel] = (unsigned char)min(max(value, 0.0f), 255.0f);
    }
    output[idx + 3] = input[idx + 3];
}

// Функция для чтения PNG изображения
Image readPNG(const char* filename) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    
    printf("Reading PNG file: %s\n", filename);
    
    if (!png_image_begin_read_from_file(&image, filename)) {
        fprintf(stderr, "Error reading PNG: %s\n", image.message);
        exit(1);
    }
    
    printf("Image size: %dx%d\n", image.width, image.height);
    
    image.format = PNG_FORMAT_RGBA;
    png_bytep buffer = (png_bytep)malloc(PNG_IMAGE_SIZE(image));
    
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for image data\n");
        exit(1);
    }
    
    printf("Allocated %zu bytes for image data\n", (size_t)PNG_IMAGE_SIZE(image));
    
    if (!png_image_finish_read(&image, NULL, buffer, 0, NULL)) {
        fprintf(stderr, "Error reading PNG data: %s\n", image.message);
        free(buffer);
        exit(1);
    }
    
    Image result;
    result.width = image.width;
    result.height = image.height;
    result.data = buffer;
    
    png_image_free(&image);
    printf("PNG read successfully\n");
    return result;
}

// Функция для записи PNG изображения
void writePNG(const char* filename, Image img) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;
    image.width = img.width;
    image.height = img.height;
    image.format = PNG_FORMAT_RGBA;
    
    printf("Writing PNG file: %s\n", filename);
    
    if (!png_image_write_to_file(&image, filename, 0, img.data, 0, NULL)) {
        fprintf(stderr, "Error writing PNG: %s\n", image.message);
        exit(1);
    }
    
    png_image_free(&image);
    printf("PNG written successfully\n");
}

// функция применения фильтра с замером времени
void applyFilterGPU(Image input, Image* output, const char* filterType, int kernelSize) {
    size_t size = input.width * input.height * 4;
    printf("\n=== Applying %s filter (kernel: %dx%d) ===\n", 
           filterType, kernelSize, kernelSize);
    printf("Image size: %dx%d, data size: %lu bytes\n", 
           input.width, input.height, size);
    
    // События для замера времени
    cudaEvent_t startTotal, stopTotal;
    cudaEvent_t startH2D, stopH2D;
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startD2H, stopD2H;
    
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventCreate(&startH2D);
    cudaEventCreate(&stopH2D);
    cudaEventCreate(&startKernel);
    cudaEventCreate(&stopKernel);
    cudaEventCreate(&startD2H);
    cudaEventCreate(&stopD2H);
    
    // Начало общего времени
    cudaEventRecord(startTotal);
    
    // Выделение памяти на GPU
    unsigned char *d_input, *d_output;
    cudaError_t err;
    
    err = cudaMalloc(&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for input: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    err = cudaMalloc(&d_output, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed for output: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        exit(1);
    }
    
    // Копирование данных на GPU (Host -> Device)
    cudaEventRecord(startH2D);
    err = cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stopH2D);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy HostToDevice failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        exit(1);
    }
    
    // Настройка размеров блоков и сеток
    dim3 blockDim(16, 16);
    dim3 gridDim((input.width + blockDim.x - 1) / blockDim.x, 
                 (input.height + blockDim.y - 1) / blockDim.y);
    
    printf("Grid: %dx%d, Block: %dx%d, Threads: %d\n",
           gridDim.x, gridDim.y, blockDim.x, blockDim.y, 
           gridDim.x * gridDim.y * blockDim.x * blockDim.y);
    
    // Запуск ядра
    cudaEventRecord(startKernel);
    
    if (strcmp(filterType, "blur") == 0) {
        blurKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height);
    } else if (strcmp(filterType, "strong_blur") == 0) {
        strongBlurKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height);
    } else if (strcmp(filterType, "denoise") == 0) {
        medianFilterKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height);
    } else if (strcmp(filterType, "add_noise") == 0) {
        addGaussianNoiseKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height, 12345);
    }
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        exit(1);
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        exit(1);
    }
    
    cudaEventRecord(stopKernel);
    
    // Копирование результата обратно на CPU (Device -> Host)
    cudaEventRecord(startD2H);
    err = cudaMemcpy(output->data, d_output, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopD2H);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy DeviceToHost failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        exit(1);
    }
    
    // Конец общего времени
    cudaEventRecord(stopTotal);
    cudaEventSynchronize(stopTotal);
    
    // Расчет времени
    float totalTime, h2dTime, kernelTime, d2hTime;
    cudaEventElapsedTime(&h2dTime, startH2D, stopH2D);
    cudaEventElapsedTime(&kernelTime, startKernel, stopKernel);
    cudaEventElapsedTime(&d2hTime, startD2H, stopD2H);
    cudaEventElapsedTime(&totalTime, startTotal, stopTotal);
    
    // Вывод результатов
    printf("\n--- Timing Results ---\n");
    printf("Host → Device copy:  %.3f ms\n", h2dTime);
    printf("Kernel execution:    %.3f ms\n", kernelTime);
    printf("Device → Host copy:  %.3f ms\n", d2hTime);
    printf("Total GPU time:      %.3f ms\n", totalTime);
    printf("Throughput:          %.2f MPixels/sec\n", 
           (input.width * input.height) / (totalTime / 1000.0f) / 1000000.0f);
    
    // Освобождение событий
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    cudaEventDestroy(startH2D);
    cudaEventDestroy(stopH2D);
    cudaEventDestroy(startKernel);
    cudaEventDestroy(stopKernel);
    cudaEventDestroy(startD2H);
    cudaEventDestroy(stopD2H);
    
    // Освобождение памяти GPU
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("GPU filter applied successfully\n");
}

// главная функция
int main() {
    printf("=== Advanced GPU Image Filter Application ===\n");
    
    // Чтение исходного изображения
    Image input = readPNG("cube.png");
    printf("Image loaded: %dx%d\n", input.width, input.height);
    
    size_t size = input.width * input.height * 4;
    
    // Создание выходных изображений
    Image output_blur, output_strong_blur, output_noisy, output_denoised;
    
    output_blur.width = output_strong_blur.width = output_noisy.width = output_denoised.width = input.width;
    output_blur.height = output_strong_blur.height = output_noisy.height = output_denoised.height = input.height;
    
    output_blur.data = (unsigned char*)malloc(size);
    output_strong_blur.data = (unsigned char*)malloc(size);
    output_noisy.data = (unsigned char*)malloc(size);
    output_denoised.data = (unsigned char*)malloc(size);
    
    if (!output_blur.data || !output_strong_blur.data || !output_noisy.data || !output_denoised.data) {
        fprintf(stderr, "Memory allocation failed for output images\n");
        free(input.data);
        return 1;
    }
    
    // Применение фильтров с детальным замером времени
    printf("\n" "=== Starting Filter Pipeline ===" "\n");
    
    // 1. Обычное размытие
    applyFilterGPU(input, &output_blur, "blur", 3);
    
    // 2. Сильное размытие
    applyFilterGPU(input, &output_strong_blur, "strong_blur", 5);
    
    // 3. Добавление шума + удаление шума
    applyFilterGPU(input, &output_noisy, "add_noise", 3);
    applyFilterGPU(output_noisy, &output_denoised, "denoise", 3);
    
    // Сохранение результатов
    printf("\n" "=== Saving Results ===" "\n");
    writePNG("output_blur.png", output_blur);
    writePNG("output_strong_blur.png", output_strong_blur);
    writePNG("output_noisy.png", output_noisy);
    writePNG("output_denoised.png", output_denoised);
    
    printf("\n" "=== Results Summary ===" "\n");
    printf("Saved files:\n");
    printf("- output_blur.png (3x3 Gaussian blur)\n");
    printf("- output_strong_blur.png (5x5 Gaussian blur)\n");
    printf("- output_noisy.png (image with added noise)\n");
    printf("- output_denoised.png (noise removed with median filter)\n");
    
    // Освобождение памяти
    free(input.data);
    free(output_blur.data);
    free(output_strong_blur.data);
    free(output_noisy.data);
    free(output_denoised.data);
    
    cudaDeviceReset();
    printf("\nProgram completed successfully!\n");
    return 0;
}