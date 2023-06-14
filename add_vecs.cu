#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// How to run
// cd simple-cuda
// nvcc -arch=sm_50 -o add_vecs.exe add_vecs.cu
// ./add_vecs 10

__global__ void addVectors(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] + b[i];
}

// __global__ void printVector(float *a){
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     std::cout << a[i];
// }

int main(int argc, char *argv[])
{
    // get information of GPU
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA capable devices found\n";
        return 1;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        int smCount = 0;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, i);
        int coresPerSM = 0;
        cudaDeviceGetAttribute(&coresPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, i);
        int totalCores = smCount * coresPerSM;
        int warpSize = 0;
        cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, i);
        std::cout << "Device #" << i << ": " << props.name << std::endl;                                      // Device #0: NVIDIA GeForce MX130
        std::cout << "  Number of SMs: " << smCount << std::endl;                                             // Number of SMs: 3
        std::cout << "  CUDA cores per SM: " << coresPerSM << std::endl;                                      // CUDA cores per SM: 2048
        std::cout << "  Total CUDA cores: " << totalCores << std::endl;                                       // Total CUDA cores: 6144           
        std::cout << "  Warp size: " << warpSize << std::endl;                                                // Warp size: 32
        std::cout << "  Compute capability: " << props.major << "." << props.minor << std::endl;              // Compute capability: 5.0
        std::cout << "  Total global memory: " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl; // Total global memory: 2047 MB
        std::cout << "  Maximum thread block size: " << props.maxThreadsPerBlock << std::endl;                // Maximum thread block size: 1024
    }

    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " N\n";
        return 1;
    }

    int n = std::atoi(argv[1]);
    size_t size = n * sizeof(float);

    float *a, *b, *c;
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    for (int i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = 2 * i;
    }

    int threadsPerBlock = 1024;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    addVectors<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++)
        std::cout << c[i] << " ";

    // printVector<<<blocksPerGrid, threadsPerBlock>>>(c);
    // cudaDeviceSynchronize();
    std::cout << std::endl;

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}