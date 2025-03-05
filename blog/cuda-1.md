---
title: cuda-1-入门基础
ccategories:
	- cuda
---
1. 入门

<!-- more -->

# 入门

## 流程

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

## 函数限定词

1. `__global__`
2. `__device__`
3. `__host__`

### 1. `__global__`

- **定义**：`__global__` 用于定义一个 **CUDA内核函数（Kernel）**，它运行在 **GPU（device）** 上，但必须由 **CPU（host）** 调用。
- **特点**：
  - 返回类型必须是 `void`。
  - 不支持可变参数。
  - 不能作为类的成员函数。
  - 内核函数是异步执行的，即CPU不会等待GPU执行完内核函数后才继续执行后续代码。
- **使用场景**：用于定义需要在GPU上并行执行的函数，通常用于大规模数据处理或计算密集型任务。
- **示例**：
  ```cpp
  __global__ void addKernel(int *a, int *b, int *c, int N) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N) {
          c[idx] = a[idx] + b[idx];
      }
  }
  ```

---

### 2. `__device__`

- **定义**：`__device__` 用于定义一个 **设备函数（Device Function）**，它只能在 **GPU（device）** 上运行，并且只能由其他在GPU上运行的函数调用。
- **特点**：
  - 不能直接从CPU调用。
  - 可以被 `__global__` 或其他 `__device__` 函数调用。
  - 可以有返回值。
- **使用场景**：用于封装在GPU上运行的辅助函数，这些函数通常用于简化内核函数的逻辑。
- **示例**：
  ```cpp
  __device__ int add(int a, int b) {
      return a + b;
  }

  __global__ void addKernel(int *a, int *b, int *c, int N) {
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx < N) {
          c[idx] = add(a[idx], b[idx]);
      }
  }
  ```

---

### 3. `__host__`

- **定义**：`__host__` 用于显式声明一个函数只能在 **CPU（host）** 上运行。
- **特点**：
  - 默认情况下，CUDA函数如果未显式指定 `__device__` 或 `__global__`，则默认为 `__host__`。
  - 可以与 `__device__` 同时使用，表示函数可以在CPU和GPU上编译和运行（这种函数称为 **统一函数**）。
  - 不能与 `__global__` 同时使用。
- **使用场景**：
  - 单独使用时，表示函数仅在CPU上运行。
  - 与 `__device__` 同时使用时，表示函数可以在CPU和GPU上运行，用于代码复用。
- **示例**：
  ```cpp
  // 仅在CPU上运行
  __host__ void printHost() {
      printf("Running on CPU\n");
  }

  // 在CPU和GPU上都可以运行
  __host__ __device__ int add(int a, int b) {
      return a + b;
  }
  ```

---

### 总结

- `__global__`：定义运行在GPU上、由CPU调用的内核函数。
- `__device__`：定义运行在GPU上、只能被GPU函数调用的设备函数。
- `__host__`：显式声明函数运行在CPU上，可与 `__device__` 结合实现统一函数。

## kernel线程层次结构

## 共享内存

共享内存在block内的线程都可以访问，但是不同得block不能互相访问。
