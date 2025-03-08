#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// 计算权重的设备函数
__device__ float computeWeight(float intensity_diff, float space_diff_sq, float sigma_color, float sigma_space)
{
    return expf(-(intensity_diff * intensity_diff / (2 * sigma_color * sigma_color)) - (space_diff_sq / (2 * sigma_space * sigma_space)));
}

// CUDA核函数：双边滤波
__global__ void bilateralFilterKernel(float* src, float* dst, int rows, int cols, int d, float sigma_color, float sigma_space)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols) return;

    __shared__ float sharedMem[16 + 2][16 + 2];

    // 加载共享内存
    if (row < rows && col < cols)
    {
        int sharedRow = threadIdx.y + 1;
        int sharedCol = threadIdx.x + 1;
        sharedMem[sharedRow][sharedCol] = src[row * cols + col];

        // 加载边界
        if (threadIdx.x == 0 && col > 0) sharedMem[sharedRow][0] = src[row * cols + (col - 1)];
        if (threadIdx.x == blockDim.x - 1 && col < cols - 1) sharedMem[sharedRow][blockDim.x + 1] = src[row * cols + (col + 1)];
        if (threadIdx.y == 0 && row > 0) sharedMem[0][sharedCol] = src[(row - 1) * cols + col];
        if (threadIdx.y == blockDim.y - 1 && row < rows - 1) sharedMem[blockDim.y + 1][sharedCol] = src[(row + 1) * cols + col];
    }
    __syncthreads();

    if (row >= d && row < rows - d && col >= d && col < cols - d)
    {
        float sum_intensity = 0.0f;
        float sum_weight = 0.0f;

        float centerPixel = sharedMem[threadIdx.y + 1][threadIdx.x + 1];

        for (int i = -d; i <= d; ++i)
        {
            for (int j = -d; j <= d; ++j)
            {
                float intensity_diff = centerPixel - sharedMem[threadIdx.y + 1 + i][threadIdx.x + 1 + j];
                float space_diff_sq = (float)(i * i + j * j);

                float weight = computeWeight(intensity_diff, space_diff_sq, sigma_color, sigma_space);
                sum_intensity += sharedMem[threadIdx.y + 1 + i][threadIdx.x + 1 + j] * weight;
                sum_weight += weight;
            }
        }
        if (sum_weight > 0.0f)
            dst[row * cols + col] = sum_intensity / sum_weight;
        else
            dst[row * cols + col] = centerPixel;
    }
}

// CUDA双边滤波函数
extern "C" void cuda_bilateralFilter(cv::Mat& src, cv::Mat& dst, int d, float sigma_color, float sigma_space)
{
    if (src.empty() || src.type() != CV_32F || src.rows < 16 || src.cols < 16)
    {
        std::cerr << "Source image is empty, not single-channel float, or too small!" << std::endl;
        return;
    }

    int rows = src.rows;
    int cols = src.cols;

    float* dev_src;
    float* dev_dst;
    int size = rows * cols * sizeof(float);

    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y); // 网格大小

    // 分配GPU内存
    cudaError err;
    err = cudaMalloc((void**)&dev_src, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc((void**)&dev_dst, size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device memory: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_src);
        return;
    }

    // 将输入图像从主机复制到设备
    err = cudaMemcpy(dev_src, src.ptr<float>(0), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data to device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_src);
        cudaFree(dev_dst);
        return;
    }

    // 调用CUDA核函数
    bilateralFilterKernel<<<blocks, threads>>>(dev_src, dev_dst, rows, cols, d, sigma_color, sigma_space);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error in kernel launch: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_src);
        cudaFree(dev_dst);
        return;
    }

    // 将结果从设备复制回主机
    dst = cv::Mat(rows, cols, CV_32F);
    err = cudaMemcpy(dst.ptr<float>(0), dev_dst, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying data from device: " << cudaGetErrorString(err) << std::endl;
        cudaFree(dev_src);
        cudaFree(dev_dst);
        return;
    }

    // 释放GPU内存
    cudaFree(dev_src);
    cudaFree(dev_dst);
}

int main()
{
    cv::Mat src = cv::imread("../test.bmp", cv::IMREAD_GRAYSCALE);
    if (src.empty())
    {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }
    
    // 归一化到 [0, 1]
    src.convertTo(src, CV_32F, 1.0 / 255.0);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
    std::cout << "Input Image - Min: " << minVal << ", Max: " << maxVal << std::endl;
    
    cv::Mat dst;
    cuda_bilateralFilter(src, dst, 5, 10.0f, 10.0f);
    
    double minVal_dst, maxVal_dst;
    cv::Point minLoc_dst, maxLoc_dst;
    cv::minMaxLoc(dst, &minVal_dst, &maxVal_dst, &minLoc_dst, &maxLoc_dst);
    std::cout << "Output Image - Min: " << minVal_dst << ", Max: " << maxVal_dst << std::endl;
    
    dst.convertTo(dst, CV_8U, 255.0);
    cv::imwrite("../Filtered.bmp", dst);
    return 0;
}