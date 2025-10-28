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

// Структура для работы с двумя GPU
struct DualGPUContext {
    int gpu0;
    int gpu1;
    size_t total_size;
    size_t split_height;
};

// Инициализация контекста для двух GPU
DualGPUContext initDualGPU(int image_height) {
    DualGPUContext context;
    
    // Получаем количество доступных GPU
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Found %d CUDA devices\n", device_count);
    
    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs, but only %d found\n", device_count);
        exit(1);
    }
    
    context.gpu0 = 0;
    context.gpu1 = 1;
    
    // Разделяем изображение по горизонтали
    context.split_height = image_height / 2;
    context.total_size = 0;
    
    printf("Using GPUs %d and %d\n", context.gpu0, context.gpu1);
    printf("Image split at height: %zu\n", context.split_height);
    
    return context;
}

// Ядро для сильного размытия (5x5)
__global__ void strongBlurKernel(unsigned char* input, unsigned char* output, int width, int height, int start_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_y = y + start_y;
    
    if (x >= width || global_y >= height) return;
    
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
            int ny = min(max(global_y + ky, 0), height - 1);
            int idx = (ny * width + nx) * 4;
            
            float weight = kernel[ky + 2][kx + 2];
            r += input[idx + 0] * weight;
            g += input[idx + 1] * weight;
            b += input[idx + 2] * weight;
        }
    }
    
    int out_idx = (global_y * width + x) * 4;
    output[out_idx + 0] = (unsigned char)min(max(r, 0.0f), 255.0f);
    output[out_idx + 1] = (unsigned char)min(max(g, 0.0f), 255.0f);
    output[out_idx + 2] = (unsigned char)min(max(b, 0.0f), 255.0f);
    output[out_idx + 3] = input[out_idx + 3];
}

// Ядро для обычного размытия (3x3)
__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int start_y) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_y = y + start_y;
    
    if (x >= width || global_y >= height) return;
    
    float kernel[3][3] = {
        {1/16.0f, 2/16.0f, 1/16.0f},
        {2/16.0f, 4/16.0f, 2/16.0f},
        {1/16.0f, 2/16.0f, 1/16.0f}
    };
    
    float r = 0.0f, g = 0.0f, b = 0.0f;
    
    for(int ky = -1; ky <= 1; ky++) {
        for(int kx = -1; kx <= 1; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(global_y + ky, 0), height - 1);
            int idx = (ny * width + nx) * 4;
            
            float weight = kernel[ky + 1][kx + 1];
            r += input[idx + 0] * weight;
            g += input[idx + 1] * weight;
            b += input[idx + 2] * weight;
        }
    }
    
    int out_idx = (global_y * width + x) * 4;
    output[out_idx + 0] = (unsigned char)min(max(r, 0.0f), 255.0f);
    output[out_idx + 1] = (unsigned char)min(max(g, 0.0f), 255.0f);
    output[out_idx + 2] = (unsigned char)min(max(b, 0.0f), 255.0f);
    output[out_idx + 3] = input[out_idx + 3];
}

// Ядро для гауссового шума
__global__ void addGaussianNoiseKernel(unsigned char* input, unsigned char* output, int width, int height, int start_y, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int global_y = y + start_y;
    
    if (x >= width || global_y >= height) return;
    
    int idx = (global_y * width + x) * 4;
    
    unsigned int noise_seed = (x * 123 + global_y * 456 + seed) % 123456;
    float noise = (float)(noise_seed % 100) / 100.0f * 50.0f - 25.0f; // ±25
    
    for (int channel = 0; channel < 3; channel++) {
        float value = input[idx + channel] + noise;
        output[idx + channel] = (unsigned char)min(max(value, 0.0f), 255.0f);
    }
    output[idx + 3] = input[idx + 3];
}

// Функции чтения и записи PNG
Image readPNG(const char* filename) {
    png_image image;
    memset(&image, 0, sizeof(image));
    image.version = PNG_IMAGE_VERSION;

    if (!png_image_begin_read_from_file(&image, filename)) {
        fprintf(stderr, "Error reading PNG: %s\n", image.message);
        exit(1);
    }

    image.format = PNG_FORMAT_RGBA;
    png_bytep buffer = (png_bytep)malloc(PNG_IMAGE_SIZE(image));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed for image data\n");
        exit(1);
    }

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
    return result;
}

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

// Применение фильтра на двух GPU
void applyFilterDualGPU(Image input, Image* output, const char* filterType, int kernelSize) {
    size_t total_size = input.width * input.height * 4;
    
    DualGPUContext context = initDualGPU(input.height);
    size_t part1_size = input.width * context.split_height * 4;
    size_t part2_size = input.width * (input.height - context.split_height) * 4;
    
    // События для измерения времени
    cudaEvent_t start, stop;
    float milliseconds_total = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // GPU 0: Верхняя часть изображения
    cudaSetDevice(context.gpu0);
    unsigned char *d_input0, *d_output0;
    cudaMalloc(&d_input0, total_size);
    cudaMalloc(&d_output0, total_size);
    cudaMemcpy(d_input0, input.data, total_size, cudaMemcpyHostToDevice);
    
    // GPU 1: Нижняя часть изображения  
    cudaSetDevice(context.gpu1);
    unsigned char *d_input1, *d_output1;
    cudaMalloc(&d_input1, total_size);
    cudaMemcpy(d_input1, input.data, total_size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_output1, total_size);
    
    // Размер блока: 32×8 потоков (256 потоков на блок)
    dim3 blockDim(32, 8);
    
    // Расчет размеров сетки с учетом разделения изображения
    int gridX = (input.width + blockDim.x - 1) / blockDim.x;
    int gridY1 = (context.split_height + blockDim.y - 1) / blockDim.y;
    int gridY2 = ((input.height - context.split_height) + blockDim.y - 1) / blockDim.y;
    
    printf("Grid configuration: Block(%d,%d), GPU0 Grid(%d,%d), GPU1 Grid(%d,%d)\n",
           blockDim.x, blockDim.y, gridX, gridY1, gridX, gridY2);
    
    // Запуск ядер на обоих GPU
    if (strcmp(filterType, "blur") == 0) {
        // GPU 0: верхняя часть (0 ÷ 1000 строк)
        cudaSetDevice(context.gpu0);
        dim3 gridDim0(gridX, gridY1);
        blurKernel<<<gridDim0, blockDim>>>(d_input0, d_output0, input.width, input.height, 0);
        
        // GPU 1: нижняя часть (1000 ÷ 2000 строк)
        cudaSetDevice(context.gpu1);
        dim3 gridDim1(gridX, gridY2);
        blurKernel<<<gridDim1, blockDim>>>(d_input1, d_output1, input.width, input.height, context.split_height);
        
    } else if (strcmp(filterType, "strong_blur") == 0) {
        // GPU 0: верхняя часть
        cudaSetDevice(context.gpu0);
        dim3 gridDim0(gridX, gridY1);
        strongBlurKernel<<<gridDim0, blockDim>>>(d_input0, d_output0, input.width, input.height, 0);
        
        // GPU 1: нижняя часть
        cudaSetDevice(context.gpu1);
        dim3 gridDim1(gridX, gridY2);
        strongBlurKernel<<<gridDim1, blockDim>>>(d_input1, d_output1, input.width, input.height, context.split_height);
        
    } else if (strcmp(filterType, "add_noise") == 0) {
        // GPU 0: верхняя часть
        cudaSetDevice(context.gpu0);
        dim3 gridDim0(gridX, gridY1);
        addGaussianNoiseKernel<<<gridDim0, blockDim>>>(d_input0, d_output0, input.width, input.height, 0, 12345);
        
        // GPU 1: нижняя часть
        cudaSetDevice(context.gpu1);
        dim3 gridDim1(gridX, gridY2);
        addGaussianNoiseKernel<<<gridDim1, blockDim>>>(d_input1, d_output1, input.width, input.height, context.split_height, 54321);
    }
    
    // Проверка ошибок CUDA после запуска ядер
    cudaError_t err;
    cudaSetDevice(context.gpu0);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU0 Kernel error: %s\n", cudaGetErrorString(err));
    }
    
    cudaSetDevice(context.gpu1);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPU1 Kernel error: %s\n", cudaGetErrorString(err));
    }
    
    // Синхронизация обоих устройств
    cudaSetDevice(context.gpu0);
    cudaDeviceSynchronize();
    cudaSetDevice(context.gpu1);
    cudaDeviceSynchronize();
    
    // Копирование результатов обратно на хост
    cudaSetDevice(context.gpu0);
    cudaMemcpy(output->data, d_output0, total_size, cudaMemcpyDeviceToHost);
    
    // Для нижней части копируем из GPU 1
    unsigned char* lower_part = output->data + part1_size;
    cudaMemcpy(lower_part, d_output1 + part1_size, part2_size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds_total, start, stop);
    printf("[Dual GPU] Total execution time (%s): %.3f ms\n", filterType, milliseconds_total);
    
    // Освобождение памяти
    cudaSetDevice(context.gpu0);
    cudaFree(d_input0);
    cudaFree(d_output0);
    
    cudaSetDevice(context.gpu1);
    cudaFree(d_input1);
    cudaFree(d_output1);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// функция main()
int main() {
    Image input = readPNG("cube.png");
    size_t size = input.width * input.height * 4;

    Image output_blur, output_strong_blur, output_noisy;
    output_blur.width = output_strong_blur.width = output_noisy.width = input.width;
    output_blur.height = output_strong_blur.height = output_noisy.height = input.height;

    output_blur.data = (unsigned char*)malloc(size);
    output_strong_blur.data = (unsigned char*)malloc(size);
    output_noisy.data = (unsigned char*)malloc(size);

    printf("=== Applying filters with Dual GPU ===\n");
    
    applyFilterDualGPU(input, &output_blur, "blur", 3);
    applyFilterDualGPU(input, &output_strong_blur, "strong_blur", 5);
    applyFilterDualGPU(input, &output_noisy, "add_noise", 3);

    writePNG("output_blur_dual.png", output_blur);
    writePNG("output_strong_blur_dual.png", output_strong_blur);
    writePNG("output_noisy_dual.png", output_noisy);

    free(input.data);
    free(output_blur.data);
    free(output_strong_blur.data);
    free(output_noisy.data);

    cudaDeviceReset();
    printf("Dual GPU program completed successfully!\n");
    return 0;
}