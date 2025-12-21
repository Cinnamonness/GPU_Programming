#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("CUDA-устройств не найдено!\n");
        return 0;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    printf("Device has Compute Capability %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Device name : %s\n", deviceProp.name);
    printf("Total global memory : %zu MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
    printf("Shared memory per block : %zu\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block : %d\n", deviceProp.regsPerBlock);
    printf("Warp size : %d\n", deviceProp.warpSize);
    printf("Memory pitch : %zu\n", deviceProp.memPitch);
    printf("Max threads per block : %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max threads dimensions : x = %d, y = %d, z = %d\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max grid size: x = %d, y = %d, z = %d\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    int clockRate = 0;
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, 0);
    printf("Clock rate: %.2f MHz\n", clockRate / 1000.0);

    printf("Total constant memory: %zu\n", deviceProp.totalConstMem);
    printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Texture alignment: %zu\n", deviceProp.textureAlignment);
    printf("Device overlap: %d\n", deviceProp.deviceOverlap);
    printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);
    printf("Kernel execution timeout enabled: %s\n", deviceProp.kernelExecTimeoutEnabled ? "true" : "false");

    getchar(); 
    return 0;
}