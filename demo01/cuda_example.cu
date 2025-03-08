// cuda_example.cu
#include <iostream>

__global__ void addVectors(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 100000;
    float *a_host = new float[n];
    float *b_host = new float[n];
    float *c_host = new float[n];
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        a_host[i] = i;
        b_host[i] = 2 * i;
    }

    float *a, *b, *c;
    cudaMalloc((void**)&a, n * sizeof(float));
    cudaMalloc((void**)&b, n * sizeof(float));
    cudaMalloc((void**)&c, n * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(a, a_host, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b, b_host, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(a, b, c, n);

    // Copy result back to host
    cudaMemcpy(c_host, c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < 10; i++) {
        std::cout << "c[" << i << "] = " << c_host[i] << std::endl;
    }

    // Clean up
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    delete[] a_host;
    delete[] b_host;
    delete[] c_host;

    return 0;
}
