#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

// CUDA 版双边滤波核
__global__ void bilateralFilterKernel(float *src, float *dst, int rows, int cols, int d, float sigma_color, float sigma_space)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= cols)
    {
        return;
    }

    int centerIndex = row * cols + col;
    dst[centerIndex] = 255;

    // if (d < col && col < cols - d && row > d && row < rows - d)
    // {
    //     int centerIndex = row * cols + col;

    //     float sum_intensity = 0.0f;
    //     float sum_weight = 0.0f;

    //     for (int i = -d; i <= d; ++i)
    //     {
    //         for (int j = -d; j <= d; ++j)
    //         {
    //             int curRow = row + i;
    //             int curCol = col + j;

    //             if (curRow >= 0 && curRow < rows && curCol >= 0 && curCol < cols)
    //             {
    //                 int curIndex = curRow * cols + curCol;
    //                 float intensity_diff = src[centerIndex] - src[curIndex];
    //                 float space_diff = i * i + j * j;

    //                 float weight = expf(-(intensity_diff * intensity_diff / (2 * sigma_color * sigma_color)) - ((float)space_diff / (2 * sigma_space * sigma_space)));
    //                 sum_intensity += src[curIndex] * weight;
    //                 sum_weight += weight;
    //             }
    //         }
    //     }

    //     if (sum_weight > 0.0f)
    //     {
    //         // dst[centerIndex] = sum_intensity / sum_weight;
    //         dst[centerIndex] = 255;
    //     }
    //     else
    //     {
    //         // dst[centerIndex] = src[centerIndex];
    //         dst[centerIndex] = 111;
    //     }
    // }
}

void cuda_bilateralFilter(cv::Mat &src, cv::Mat &dst, int d, float sigma_color, float sigma_space)
{
    if (src.empty())
    {
        std::cerr << "Source image is empty!" << std::endl;
        return;
    }

    int rows = src.rows;
    int cols = src.cols;
    printf("rows : %d\n", rows);
    printf("cols : %d\n", cols);

    float *dev_src;
    float *dev_dst;
    int size = rows * cols * sizeof(float);
    printf("size : %d\n", size);

    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    dim3 threads(16, 16);

    cudaMalloc((float **)&dev_src, size);
    cudaMalloc((float **)&dev_dst, size);
    cudaMemcpy(dev_src, (float *)src.data, size, cudaMemcpyHostToDevice);

    
    bilateralFilterKernel <<<blocks, threads>>> (dev_src, dev_dst, rows, cols, d, sigma_color, sigma_space);

    float* h_src = new float[rows * cols];
    cudaMemcpy(h_src, dev_dst, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i++) {
        std::cout << "c[" << i << "] = " << h_src[i] << std::endl;
    }
    delete[] h_src;

    // cudaMemcpy((float *)dst.data, dev_dst, size, cudaMemcpyDeviceToHost);
    cudaMemcpy((float *)dst.data, dev_src, size, cudaMemcpyDeviceToHost);

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

    src.convertTo(src, CV_32FC1);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(src, &minVal, &maxVal, &minLoc, &maxLoc);
    std::cout << "Input Image - Min: " << minVal << ", Max: " << maxVal << std::endl;

    cv::Mat dst = cv::Mat::zeros(src.size(),CV_32FC1);
    cuda_bilateralFilter(src, dst, 5, 10.0f, 10.0f);

    double minVal_dst, maxVal_dst;
    cv::Point minLoc_dst, maxLoc_dst;
    cv::minMaxLoc(dst, &minVal_dst, &maxVal_dst, &minLoc_dst, &maxLoc_dst);
    std::cout << "Output Image - Min: " << minVal_dst << ", Max: " << maxVal_dst << std::endl;

    cv::imwrite("../Filtered.bmp", dst);
    return 0;
}