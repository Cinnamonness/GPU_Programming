#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return 1;
    }

    if (deviceCount == 0) {
        printf("CUDA devices not found!\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);

        printf("\n=== Device %d: %s ===\n", i, deviceProp.name);
        printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Total global memory: %zu MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
        printf("Shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
        printf("Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("Max grid size: (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

        int clockRate = 0;
        cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, i);
        printf("Clock rate: %.2f MHz\n", clockRate / 1000.0);

        printf("Total constant memory: %zu bytes\n", deviceProp.totalConstMem);
        printf("Texture alignment: %zu bytes\n", deviceProp.textureAlignment);
        printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
        printf("Kernel execution timeout: %s\n", 
               deviceProp.kernelExecTimeoutEnabled ? "enabled" : "disabled");
        
        // Дополнительные атрибуты
        int asyncEngineCount;
        cudaDeviceGetAttribute(&asyncEngineCount, cudaDevAttrAsyncEngineCount, i);
        printf("Async engine count: %d\n", asyncEngineCount);
        
        int canMapHostMemory;
        cudaDeviceGetAttribute(&canMapHostMemory, cudaDevAttrCanMapHostMemory, i);
        printf("Can map host memory: %s\n", canMapHostMemory ? "yes" : "no");
    }

    return 0;
}