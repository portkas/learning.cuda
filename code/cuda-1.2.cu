#include <iostream>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// 使用256个线程
// 每个线程处理多个元素
__global__ void add(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
    {
        y[i] = x[i] + y[i];
    }
}

int main()
{
    int N = 1 << 20;
    float *x, *y;

    cudaError err = cudaMallocManaged(&x, N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate unified memory for x: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    err = cudaMallocManaged(&y, N * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Failed to allocate unified memory for y: " << cudaGetErrorString(err) << std::endl;
        cudaFree(x);
        return -1;
    }

    for (int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // 每个线程块有256个线程
    // 只使用一个线程块
    int threadsPerBlock = 256;
    int blocksPerGrid = 1;

    auto start = std::chrono::high_resolution_clock::now();
    add<<<blocksPerGrid, threadsPerBlock>>>(N, x, y);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
    {
        maxError = fmax(maxError, fabs(y[i] - 3.0));
    }
    std::cout << "Max error: " << maxError << std::endl;
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}