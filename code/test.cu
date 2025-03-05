#include <stdio.h>

// CUDA 核函数：将数组中的每个元素加 1
__global__ void addOne(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局索引
    if (idx < n) {
        data[idx] += 1;
    }
}

int main() {
    const int n = 256; // 数组大小
    const int size = n * sizeof(int); // 数组占用的字节数

    // 分配主机端内存
    int *h_data = (int *)malloc(size);
    if (h_data == nullptr) {
        printf("Failed to allocate host memory\n");
        return -1;
    }

    // 初始化数组
    for (int i = 0; i < n; i++) {
        h_data[i] = i;
    }

    // 分配设备端内存
    int *d_data;
    cudaError_t err = cudaMalloc(&d_data, size);
    if (err != cudaSuccess) {
        printf("Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_data);
        return -1;
    }

    // 将数据从主机端拷贝到设备端
    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Failed to copy data to device: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return -1;
    }

    // 设置线程块大小和网格大小
    int blockSize = 256; // 每个线程块的线程数
    int numBlocks = (n + blockSize - 1) / blockSize; // 计算网格大小

    // 调用核函数
    addOne<<<numBlocks, blockSize>>>(d_data, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return -1;
    }

    // 将结果从设备端拷贝回主机端
    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Failed to copy data back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        free(h_data);
        return -1;
    }

    // 打印结果
    for (int i = 0; i < n; i++) {
        printf("h_data[%d] = %d\n", i, h_data[i]);
    }

    // 释放内存
    cudaFree(d_data);
    free(h_data);

    return 0;
}