/*********************************************************************************************
 * file name  : cuda_constant_memory.cu
 * author     : justin
 * date       : 2024-3-19
 * brief      : 常量内存变量使用示例
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "../cuda错误检测/error.cuh"


__constant__ float c_data;              // 定义常量内存变量，只能在主机端使用特殊函数进行初始化，不可以在设备中直接改变
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1(void)
{
    printf("Constant data c_data = %.2f.\n", c_data);
}

__global__ void kernel_2(int N)
{
    int idx = threadIdx.x;
    if (idx < N)
    {

    }   
}

int main(int argc, char **argv)
{ 
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    float h_data = 8.8f;
    CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));              // 使用主机端的cudaMemcpyToSymbol函数从主机向常量内存拷贝数据

    dim3 block(1);
    dim3 grid(1);
    kernel_1<<<grid, block>>>();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));           // 使用主机端的cudaMemcpyFromSymbol函数从常量内存向主机拷贝数据
    printf("Constant data h_data = %.2f.\n", h_data);

    CHECK(cudaDeviceReset());

    return 0;
}