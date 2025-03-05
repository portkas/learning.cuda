#include <stdio.h>

// 如果共享内存数组大小在编译时已知
// 就像在 staticReverse 内核中一样
// 那么我们可以显式地声明一个该大小的数组，例如s[64]
// 由于全局内存总是通过线性对齐索引 t 访问
// 所以读写都可以实现最佳的全局内存合并
__global__ void staticReverse(int *d, int n)
{
  __shared__ int s[64]; // 静态分配共享内存
  int t = threadIdx.x;  // 当前线程索引
  int tr = n-t-1;       // 反转后的索引
  s[t] = d[t];          // 将全局内存中的数据加载到共享内存
  __syncthreads();      // 确保所有线程都已完成对共享内存的加载
  d[t] = s[tr];         // 从共享内存中读取反转后的数据
}

__global__ void dynamicReverse(int *d, int n)
{
  extern __shared__ int s[];    // 动态分配共享内存
  int t = threadIdx.x;          // 当前线程的索引
  int tr = n-t-1;
  s[t] = d[t];
  __syncthreads();
  d[t] = s[tr];
}

int main(void)
{
  const int n = 64;         // 数组大小
  int a[n], r[n], d[n];     // 输入数组，预期反转数组，输出数组
  
  for (int i = 0; i < n; i++) {
    a[i] = i;
    r[i] = n-i-1;
    d[i] = 0;
  }

  int *d_d;     // 设备端输出数组
  cudaMalloc(&d_d, n * sizeof(int)); 
  
  // run version with static shared memory
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  staticReverse<<<1,n>>>(d_d, n); // 静态共享内存
  cudaMemcpy(d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
  
  // run dynamic shared memory version
  cudaMemcpy(d_d, a, n*sizeof(int), cudaMemcpyHostToDevice);
  dynamicReverse<<<1,n,n*sizeof(int)>>>(d_d, n);  // 动态共享内存
  cudaMemcpy(d, d_d, n * sizeof(int), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++) 
    if (d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);
}