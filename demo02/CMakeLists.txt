cmake_minimum_required(VERSION 3.22)
project(BilateralFilter LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.8 REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "hello cmake")
message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
message(STATUS "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Toolkit: ${CUDAToolkit_INCLUDE_DIRS}")

set(CMAKE_CUDA_ARCH_BIN "86")
add_executable(${PROJECT_NAME} cuda_example.cu)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})