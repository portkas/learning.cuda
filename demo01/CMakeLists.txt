# CMakeLists.txt
cmake_minimum_required(VERSION 3.22)
project(cuda_example LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDA 12.8 REQUIRED)

# Set CUDA architecture
set(CMAKE_CUDA_ARCH_BIN "75")  # For NVIDIA Ampere architecture

# Add CUDA source files
add_executable(cuda_example cuda_example.cu)

# Set CUDA flags
set_target_properties(cuda_example PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Link against CUDA libraries (optional)
target_link_libraries(cuda_example ${CUDA_LIBRARIES})
