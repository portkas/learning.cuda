#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

// 定义中值滤波的核大小
#define KERNEL_SIZE 3

// CUDA内核函数：中值滤波
__global__ void medianFilterKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int index = (y * width + x) * channels;

    for (int c = 0; c < channels; ++c) {
        int count = 0;
        unsigned char values[KERNEL_SIZE * KERNEL_SIZE];
        int kernelIndex = 0;

        for (int ky = -KERNEL_SIZE / 2; ky <= KERNEL_SIZE / 2; ++ky) {
            for (int kx = -KERNEL_SIZE / 2; kx <= KERNEL_SIZE / 2; ++kx) {
                int nx = x + kx;
                int ny = y + ky;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    values[kernelIndex++] = input[((ny * width + nx) * channels) + c];
                    count++;
                }
            }
        }

        // 对values数组进行排序以找到中值
        for (int i = 0; i < count - 1; ++i) {
            for (int j = 0; j < count - i - 1; ++j) {
                if (values[j] > values[j + 1]) {
                    unsigned char temp = values[j];
                    values[j] = values[j + 1];
                    values[j + 1] = temp;
                }
            }
        }

        output[index + c] = values[count / 2];
    }
}

// 主函数
int main() {
    cv::Mat inputImage = cv::imread("../input.bmp", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Unable to load image!" << std::endl;
        return -1;
    }

    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    int imageSize = width * height * channels * sizeof(unsigned char);

    // 分配CUDA内存
    unsigned char* d_input;
    unsigned char* d_output;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);

    // 将图像数据从CPU内存复制到GPU内存
    cudaMemcpy(d_input, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 调用CUDA内核函数
    medianFilterKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, channels);

    // 将结果从GPU内存复制回CPU内存
    cv::Mat outputImage(height, width, CV_8UC3);
    cudaMemcpy(outputImage.data, d_output, imageSize, cudaMemcpyDeviceToHost);

    // 释放CUDA内存
    cudaFree(d_input);
    cudaFree(d_output);

    // 显示结果
    cv::imshow("Input Image", inputImage);
    cv::imshow("Median Filtered Image", outputImage);
    cv::waitKey(0);

    return 0;
}