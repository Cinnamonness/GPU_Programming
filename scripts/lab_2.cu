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

// Ядро для сильного размытия (5x5)
__global__ void strongBlurKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
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

// Ядро для гауссового шума
__global__ void addGaussianNoiseKernel(unsigned char* input, unsigned char* output, int width, int height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * 4;
    
    unsigned int noise_seed = (x * 123 + y * 456 + seed) % 123456;
    float noise = (float)(noise_seed % 100) / 100.0f * 50.0f - 25.0f; // ±25
    
    for (int channel = 0; channel < 3; channel++) {
        float value = input[idx + channel] + noise;
        output[idx + channel] = (unsigned char)min(max(value, 0.0f), 255.0f);
    }
    output[idx + 3] = input[idx + 3];
}

// Функции чтения и записи PNG остаются без изменений
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

// Применение фильтра GPU 
void applyFilterGPU(Image input, Image* output, const char* filterType, int kernelSize) {
    size_t size = input.width * input.height * 4;

    unsigned char *d_input, *d_output;
    
    // События для измерения времени
    cudaEvent_t start, stop;
    float milliseconds = 0.0f;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Копирование данных на GPU
    cudaEventRecord(start);
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMemcpy(d_input, input.data, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[GPU] H2D copy time: %.3f ms\n", milliseconds);

    dim3 blockDim(16, 16);
    dim3 gridDim((input.width + blockDim.x - 1) / blockDim.x, 
                 (input.height + blockDim.y - 1) / blockDim.y);

    // Запуск ядра
    cudaEventRecord(start);
    if (strcmp(filterType, "blur") == 0) {
        blurKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height);
    } else if (strcmp(filterType, "strong_blur") == 0) {
        strongBlurKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height);
    } else if (strcmp(filterType, "add_noise") == 0) {
        addGaussianNoiseKernel<<<gridDim, blockDim>>>(d_input, d_output, input.width, input.height, 12345);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[GPU] Kernel execution time (%s): %.3f ms\n", filterType, milliseconds);

    // Копирование результата обратно
    cudaEventRecord(start);
    cudaMemcpy(output->data, d_output, size, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[GPU] D2H copy time: %.3f ms\n", milliseconds);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Главная функция
int main() {
    Image input = readPNG("cube.png");
    size_t size = input.width * input.height * 4;

    Image output_blur, output_strong_blur, output_noisy;
    output_blur.width = output_strong_blur.width = output_noisy.width = input.width;
    output_blur.height = output_strong_blur.height = output_noisy.height = input.height;

    output_blur.data = (unsigned char*)malloc(size);
    output_strong_blur.data = (unsigned char*)malloc(size);
    output_noisy.data = (unsigned char*)malloc(size);

    applyFilterGPU(input, &output_blur, "blur", 3);
    applyFilterGPU(input, &output_strong_blur, "strong_blur", 5);
    applyFilterGPU(input, &output_noisy, "add_noise", 3);

    writePNG("output_blur.png", output_blur);
    writePNG("output_strong_blur.png", output_strong_blur);
    writePNG("output_noisy.png", output_noisy);

    free(input.data);
    free(output_blur.data);
    free(output_strong_blur.data);
    free(output_noisy.data);

    cudaDeviceReset();
    printf("Program completed successfully!\n");
    return 0;
}
