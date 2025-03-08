#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_OUTPUT_SIZE 16

__global__ void bilateralFilterKernel(float *src, float *dst, int rows, int cols, 
                                    int d, float sigma_color, float sigma_space) {
    const int halo = d;
    const int block_threads = BLOCK_OUTPUT_SIZE + 2 * halo;
    
    // 动态共享内存声明
    extern __shared__ char shared_memory[];
    float* sharedMem = (float*)shared_memory;
    bool* valid = (bool*)(shared_memory + block_threads * block_threads * sizeof(float));

    const int global_x = blockIdx.x * BLOCK_OUTPUT_SIZE - halo + threadIdx.x;
    const int global_y = blockIdx.y * BLOCK_OUTPUT_SIZE - halo + threadIdx.y;

    // 加载数据到共享内存
    if (threadIdx.x < block_threads && threadIdx.y < block_threads) {
        const int idx = threadIdx.y * block_threads + threadIdx.x;
        if (global_x >= 0 && global_x < cols && global_y >= 0 && global_y < rows) {
            sharedMem[idx] = src[global_y * cols + global_x];
            valid[idx] = true;
        } else {
            sharedMem[idx] = 0;
            valid[idx] = false;
        }
    }
    __syncthreads();

    if (threadIdx.x >= halo && threadIdx.x < halo + BLOCK_OUTPUT_SIZE &&
        threadIdx.y >= halo && threadIdx.y < halo + BLOCK_OUTPUT_SIZE) {
        
        const int out_x = blockIdx.x * BLOCK_OUTPUT_SIZE + (threadIdx.x - halo);
        const int out_y = blockIdx.y * BLOCK_OUTPUT_SIZE + (threadIdx.y - halo);

        if (out_x >= cols || out_y >= rows) return;

        float sum_intensity = 0.0f;
        float sum_weight = 0.0f;
        const int center_idx = threadIdx.y * block_threads + threadIdx.x;
        const float center_val = sharedMem[center_idx];

        for (int i = -d; i <= d; ++i) {
            for (int j = -d; j <= d; ++j) {
                const int sx = threadIdx.x + j;
                const int sy = threadIdx.y + i;
                
                if (sx >= 0 && sx < block_threads && sy >= 0 && sy < block_threads) {
                    const int curr_idx = sy * block_threads + sx;
                    if (valid[curr_idx]) {
                        const float curr_val = sharedMem[curr_idx];
                        const float intensity_diff = center_val - curr_val;
                        const float space_diff = i*i + j*j;

                        const float weight = expf(-(intensity_diff*intensity_diff)/(2*sigma_color*sigma_color) 
                                                - space_diff/(2*sigma_space*sigma_space));
                        sum_intensity += curr_val * weight;
                        sum_weight += weight;
                    }
                }
            }
        }

        dst[out_y * cols + out_x] = (sum_weight > 1e-5f) ? (sum_intensity / sum_weight) : center_val;
    }
}

void cuda_bilateralFilter(cv::Mat &src, cv::Mat &dst, int d, float sigma_color, float sigma_space) {
    if (src.empty()) {
        std::cerr << "Source image is empty!" << std::endl;
        return;
    }

    const int halo = d;
    const int block_threads = BLOCK_OUTPUT_SIZE + 2 * halo;
    const size_t sharedMemSize = block_threads * block_threads * (sizeof(float) + sizeof(bool));

    dim3 threads(block_threads, block_threads);
    dim3 blocks((src.cols + BLOCK_OUTPUT_SIZE - 1) / BLOCK_OUTPUT_SIZE,
                (src.rows + BLOCK_OUTPUT_SIZE - 1) / BLOCK_OUTPUT_SIZE);

    float *dev_src, *dev_dst;
    cudaMalloc(&dev_src, src.rows * src.cols * sizeof(float));
    cudaMalloc(&dev_dst, src.rows * src.cols * sizeof(float));

    cudaMemcpy(dev_src, src.ptr<float>(), src.rows * src.cols * sizeof(float), cudaMemcpyHostToDevice);
    bilateralFilterKernel<<<blocks, threads, sharedMemSize>>>(dev_src, dev_dst, src.rows, src.cols, 
                                                             d, sigma_color, sigma_space);
    cudaMemcpy(dst.ptr<float>(), dev_dst, src.rows * src.cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_src);
    cudaFree(dev_dst);
}

int main() {
    cv::Mat src = cv::imread("../test.bmp", cv::IMREAD_GRAYSCALE);
    if (src.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    src.convertTo(src, CV_32FC1);
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_32FC1);
    cuda_bilateralFilter(src, dst, 5, 10.0f, 10.0f);

    cv::imwrite("../Filtered_Optimized.bmp", dst);
    return 0;
}