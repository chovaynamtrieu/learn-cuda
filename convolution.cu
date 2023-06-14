#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 20
#define K 3

__global__ void random_matrix(int *matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * n + idx;
    if (idx < n && idy < n) {
        curandState state;
        curand_init(0, index, 0, &state);
        matrix[index] = curand(&state) % 256;
    }
}

__global__ void convolution(int *input, int *kernel, int *output, int n, int k) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < n && y < n) {
        int sum = 0;
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                int input_idx = (y + i - k/2) * n + (x + j - k/2);
                int kernel_idx = i * k + j;
                if (y + i - k/2 >= 0 && y + i - k/2 < n && x + j - k/2 >= 0 && x + j - k/2 < n) {
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }
        }
        output[y * n + x] = sum;
        printf("threadIdx: (%d, %d), \tblockIdx: (%d, %d), \t(x, y): (%d, %d), \t sum: %d\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, x, y, sum);
    }
}

int main(){
    // Định nghĩa ma trận đầu vào và ma trận kernel
    //     int n = 10;
    int *matrix, *d_matrix;
    size_t size =N * N * sizeof(int);

    // Allocate memory on GPU
    cudaMalloc(&d_matrix, size);

    // Launch kernel to generate random matrix
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    random_matrix<<<numBlocks, threadsPerBlock>>>(d_matrix, N);
        // Copy matrix from GPU to CPU
    matrix = (int*)malloc(size);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Print matrix
    for (int i = 0; i < N * N; i++) {
        printf("%d ", matrix[i]);
        if ((i+1) % N == 0) {
            printf("\n");
        }
    }

    int input[N][N] = { { 1, 2, 3, 4, 5 },
                        { 6, 7, 8, 9, 10 },
                        { 11, 12, 13, 14, 15 },
                        { 16, 17, 18, 19, 20 },
                        { 21, 22, 23, 24, 25 } };
    int kernel[K][K] = { { -1, -1, -1 },
                         { -1,  8, -1 },
                         { -1, -1, -1 } };

    // Cấp phát bộ nhớ trên GPU cho các ma trận đầu vào và sao chép chúng từ bộ nhớ CPU sang bộ nhớ GPU
    int *d_input, *d_kernel, *d_output;
    cudaMalloc((void **)&d_input, N * N * sizeof(int));
    cudaMalloc((void **)&d_kernel, K * K * sizeof(int));
    cudaMalloc((void **)&d_output, N * N * sizeof(int));
    cudaMemcpy(d_input, input, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, K * K * sizeof(int), cudaMemcpyHostToDevice);

    // Định nghĩa kích thước các khối và số lượng khối
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Gọi kernel convolution với các đối số là các ma trận đầu vào và kích thước của chúng
    convolution<<<gridDim, blockDim>>>(d_matrix, d_kernel, d_output, N, K);

    // Sao chép ma trận kết quả từ bộ nhớ GPU sang bộ nhớ CPU
    int output[N][N];
    cudaMemcpy(output, d_output, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // In ra ma trận kết quả
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", output[i][j]);
        }
        printf("\n");
    }

    // Giải phóng bộ nhớ trên GPU
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    cudaFree(d_matrix);
    free(matrix);

    return 0;
}