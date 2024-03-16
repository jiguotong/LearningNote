/*********************************************************************************************
 * file name  : cuda_shared_memory_static.cu
 * author     : justin
 * date       : 2024-3-19
 * brief      : 静态共享内存变量使用示例
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "../cuda错误检测/error.cuh"


__global__ void kernel(float* d_A, const int N)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;
    __shared__ float s_array[32];   // 声明定义静态共享内存，必须指定大小

    if (n < N)
    {
        s_array[tid] = d_A[n];
    }
    __syncthreads();                // 因线程块内所有线程共享共享变量，所以需要进行同步

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
    kernel<<<grid, block>>>(d_A, nElems);

    CHECK(cudaFree(d_A));
    free(h_A);
    CHECK(cudaDeviceReset());

    return 0;
}