#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <png.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// Warm-up kernel
__global__ void warmupKernel() {}

// ---------- Structures ----------
struct Image {
    int width;
    int height;
    unsigned char *data;
};

struct DualGPUContext {
    int gpu0;
    int gpu1;
    size_t total_size;
    size_t split_height;
    int overlap;
};

// ---------- Dual GPU Initialization ----------
DualGPUContext initDualGPU(int image_height, int kernel_radius) {
    DualGPUContext context;

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA devices\n", device_count);

    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs, but only %d found\n", device_count);
        exit(1);
    }

    context.gpu0 = 0;
    context.gpu1 = 1;
    context.overlap = kernel_radius;
    context.split_height = image_height / 2 + kernel_radius;
    if (context.split_height > (size_t)image_height)
        context.split_height = image_height;

    printf("Using GPUs %d and %d\n", context.gpu0, context.gpu1);
    printf("Image split at height: %zu (overlap: %d rows)\n", context.split_height, context.overlap);

    // Print GPU device names
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, context.gpu0);
    printf("GPU0: %s\n", prop.name);
    cudaGetDeviceProperties(&prop, context.gpu1);
    printf("GPU1: %s\n", prop.name);

    return context;
}

// ---------- Kernels ----------
__global__ void blurKernel(unsigned char* input, unsigned char* output, int width, int part_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;

    float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };

    float r=0, g=0, b=0;
    for(int ky=-1; ky<=1; ky++) {
        for(int kx=-1; kx<=1; kx++) {
            int nx = min(max(x+kx,0), width-1);
            int ny = min(max(y+ky,0), part_height-1);
            int idx = (ny*width+nx)*4;
            float w = kernel[ky+1][kx+1];
            r += input[idx]*w; g += input[idx+1]*w; b += input[idx+2]*w;
        }
    }
    int out_idx = (y*width+x)*4;
    output[out_idx]=r; output[out_idx+1]=g; output[out_idx+2]=b; output[out_idx+3]=255;
}

__global__ void strongBlurKernel(unsigned char* input, unsigned char* output, int width, int part_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;

    float kernel[5][5] = {
        {1,4,6,4,1}, {4,16,24,16,4}, {6,24,36,24,6}, {4,16,24,16,4}, {1,4,6,4,1}
    };
    float sum = 256.0f;
    float r=0,g=0,b=0;
    for(int ky=-2; ky<=2; ky++){
        for(int kx=-2; kx<=2; kx++){
            int nx=min(max(x+kx,0),width-1);
            int ny=min(max(y+ky,0),part_height-1);
            int idx=(ny*width+nx)*4;
            float w=kernel[ky+2][kx+2]/sum;
            r+=input[idx]*w; g+=input[idx+1]*w; b+=input[idx+2]*w;
        }
    }
    int out_idx=(y*width+x)*4;
    output[out_idx]=r; output[out_idx+1]=g; output[out_idx+2]=b; output[out_idx+3]=255;
}

__global__ void addGaussianNoiseKernel(unsigned char* input, unsigned char* output, int width, int part_height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;
    int idx = (y*width+x)*4;
    unsigned int noise_seed = (x * 12347 + y * 54321 + seed * 76543) % 1234567;
    float noise = (float)(noise_seed % 200) / 200.0f * 40.0f - 20.0f;
    for(int c=0;c<3;c++){
        float val=input[idx+c]+noise;
        output[idx+c]=(unsigned char)min(max(val,0.0f),255.0f);
    }
    output[idx+3]=255;
}

// ---------- PNG ----------
Image readPNG(const char* filename) {
    png_image img;
    memset(&img,0,sizeof(img));
    img.version=PNG_IMAGE_VERSION;
    if(!png_image_begin_read_from_file(&img,filename)){
        fprintf(stderr,"Error reading PNG: %s\n",img.message); exit(1);
    }
    img.format=PNG_FORMAT_RGBA;
    png_bytep buffer=(png_bytep)malloc(PNG_IMAGE_SIZE(img));
    if(!png_image_finish_read(&img,NULL,buffer,0,NULL)){
        fprintf(stderr,"Error reading PNG data: %s\n",img.message); free(buffer); exit(1);
    }
    Image res={img.width,img.height,buffer};
    png_image_free(&img);
    return res;
}

void writePNG(const char* filename, Image img) {
    png_image image;
    memset(&image,0,sizeof(image));
    image.version=PNG_IMAGE_VERSION;
    image.width=img.width;
    image.height=img.height;
    image.format=PNG_FORMAT_RGBA;
    printf("Writing PNG: %s\n",filename);
    if(!png_image_write_to_file(&image,filename,0,img.data,0,NULL)){
        fprintf(stderr,"Error writing PNG: %s\n",image.message);
        exit(1);
    }
    png_image_free(&image);
}

// ---------- Filter Application ----------
void applyFilterDualGPU(Image input, Image* output, const char* filterType, int kernelSize) {
    size_t total_size = input.width * input.height * 4;
    int kernel_radius = (kernelSize - 1) / 2;
    DualGPUContext context = initDualGPU(input.height, kernel_radius);

    // Pinned host memory for faster transfer
    CUDA_CHECK(cudaHostRegister(input.data, total_size, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(output->data, total_size, cudaHostRegisterDefault));

    // Warm-up both GPUs
    cudaSetDevice(context.gpu0);
    warmupKernel<<<1,1>>>();
    cudaSetDevice(context.gpu1);
    warmupKernel<<<1,1>>>();
    cudaDeviceSynchronize();

    size_t part1_size = input.width * context.split_height * 4;
    size_t part2_size = input.width * (input.height - context.split_height + context.overlap) * 4;

    float ms0=0, ms1=0;
    cudaStream_t s0, s1;
    cudaEvent_t start0, stop0, start1, stop1;
    unsigned char *d_in0,*d_out0,*d_in1,*d_out1;

    dim3 block(32,8);
    int gridX=(input.width+block.x-1)/block.x;
    int gridY0=(context.split_height+block.y-1)/block.y;
    int gridY1=((input.height-context.split_height+context.overlap)+block.y-1)/block.y;

    // GPU0
    cudaSetDevice(context.gpu0);
    CUDA_CHECK(cudaStreamCreate(&s0));
    CUDA_CHECK(cudaEventCreate(&start0));
    CUDA_CHECK(cudaEventCreate(&stop0));
    CUDA_CHECK(cudaMalloc(&d_in0, part1_size));
    CUDA_CHECK(cudaMalloc(&d_out0, part1_size));
    CUDA_CHECK(cudaEventRecord(start0,s0));
    CUDA_CHECK(cudaMemcpyAsync(d_in0,input.data,part1_size,cudaMemcpyHostToDevice,s0));

    if(strcmp(filterType,"blur")==0)
        blurKernel<<<dim3(gridX,gridY0),block,0,s0>>>(d_in0,d_out0,input.width,context.split_height);
    else if(strcmp(filterType,"strong_blur")==0)
        strongBlurKernel<<<dim3(gridX,gridY0),block,0,s0>>>(d_in0,d_out0,input.width,context.split_height);
    else
        addGaussianNoiseKernel<<<dim3(gridX,gridY0),block,0,s0>>>(d_in0,d_out0,input.width,context.split_height,12345);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop0,s0));

    // GPU1
    cudaSetDevice(context.gpu1);
    CUDA_CHECK(cudaStreamCreate(&s1));
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaMalloc(&d_in1, part2_size));
    CUDA_CHECK(cudaMalloc(&d_out1, part2_size));
    size_t offset = (context.split_height - context.overlap) * input.width * 4;
    CUDA_CHECK(cudaEventRecord(start1,s1));
    CUDA_CHECK(cudaMemcpyAsync(d_in1,input.data+offset,part2_size,cudaMemcpyHostToDevice,s1));

    if(strcmp(filterType,"blur")==0)
        blurKernel<<<dim3(gridX,gridY1),block,0,s1>>>(d_in1,d_out1,input.width,input.height-context.split_height+context.overlap);
    else if(strcmp(filterType,"strong_blur")==0)
        strongBlurKernel<<<dim3(gridX,gridY1),block,0,s1>>>(d_in1,d_out1,input.width,input.height-context.split_height+context.overlap);
    else
        addGaussianNoiseKernel<<<dim3(gridX,gridY1),block,0,s1>>>(d_in1,d_out1,input.width,input.height-context.split_height+context.overlap,54321);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop1,s1));

    cudaSetDevice(context.gpu0);
    CUDA_CHECK(cudaEventSynchronize(stop0));
    CUDA_CHECK(cudaEventElapsedTime(&ms0,start0,stop0));
    cudaSetDevice(context.gpu1);
    CUDA_CHECK(cudaEventSynchronize(stop1));
    CUDA_CHECK(cudaEventElapsedTime(&ms1,start1,stop1));

    // Copy back
    cudaSetDevice(context.gpu0);
    CUDA_CHECK(cudaMemcpyAsync(output->data,d_out0,part1_size,cudaMemcpyDeviceToHost,s0));
    cudaSetDevice(context.gpu1);
    size_t gpu1_out_offset=context.split_height*input.width*4;
    size_t gpu1_out_size=(input.height-context.split_height)*input.width*4;
    CUDA_CHECK(cudaMemcpyAsync(output->data+gpu1_out_offset,d_out1+context.overlap*input.width*4,gpu1_out_size,cudaMemcpyDeviceToHost,s1));

    CUDA_CHECK(cudaStreamSynchronize(s0));
    CUDA_CHECK(cudaStreamSynchronize(s1));

    printf("[Dual GPU] Total execution time (%s): %.3f ms (GPU0: %.3f, GPU1: %.3f)\n",
           filterType, fmaxf(ms0,ms1), ms0, ms1);

    cudaFree(d_in0); cudaFree(d_out0);
    cudaFree(d_in1); cudaFree(d_out1);
    cudaEventDestroy(start0); cudaEventDestroy(stop0);
    cudaEventDestroy(start1); cudaEventDestroy(stop1);
    cudaStreamDestroy(s0); cudaStreamDestroy(s1);
    cudaHostUnregister(input.data);
    cudaHostUnregister(output->data);
}

// ---------- main ----------
int main() {
    Image input = readPNG("creative-pebble.png");
    size_t size = input.width * input.height * 4;

    Image out1={input.width,input.height,(unsigned char*)malloc(size)};
    Image out2={input.width,input.height,(unsigned char*)malloc(size)};
    Image out3={input.width,input.height,(unsigned char*)malloc(size)};

    printf("=== Dual GPU Processing ===\nInput: %dx%d\n", input.width, input.height);

    applyFilterDualGPU(input,&out1,"blur",3);
    applyFilterDualGPU(input,&out2,"strong_blur",5);
    applyFilterDualGPU(input,&out3,"add_noise",3);

    writePNG("output_blur_dual.png",out1);
    writePNG("output_strong_blur_dual.png",out2);
    writePNG("output_noisy_dual.png",out3);

    free(input.data); free(out1.data); free(out2.data); free(out3.data);
    cudaDeviceReset();
    printf("Completed successfully!\n");
    return 0;
}
