/*********************************************************************************************
 * file name  : cuda_shared_memory_static.cu
 * author     : justin
 * date       : 2024-3-19
 * brief      : 动态共享内存变量使用示例
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "../cuda错误检测/error.cuh"


extern __shared__ float s_array[];              // 动态声明定义共享内存变量，不可以为*s_array

__global__ void kernel(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;

    if (n < N)
    {
        s_array[tid] = d_A[n];
    }
    __syncthreads();

    if (tid == 0)
    {
        for (int i = 0; i < 32; ++i)
        {
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }

}


int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int nElems = 64;
    int nbytes = nElems * sizeof(float);

    float* h_A = nullptr;
    h_A = (float*)malloc(nbytes);
    for (int i = 0; i < nElems; ++i)
    {
        h_A[i] = float(i);
    }

    float* d_A = nullptr;
    CHECK(cudaMalloc(&d_A, nbytes));
    CHECK(cudaMemcpy(d_A, h_A, nbytes,cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);
    kernel<<<grid, block, 32>>>(d_A, nElems);           // 此处调用核函数的时候需要对动态共享内存大小进行赋值

    CHECK(cudaFree(d_A));
    free(h_A);
    CHECK(cudaDeviceReset());
}