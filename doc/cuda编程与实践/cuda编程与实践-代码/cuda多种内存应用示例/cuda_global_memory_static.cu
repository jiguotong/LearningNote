/*********************************************************************************************
 * file name  : cuda_global_memory_static.cu
 * author     : justin
 * date       : 2024-3-19
 * brief      : 静态全局内存变量使用示例
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "../cuda错误检测/error.cuh"

__device__ int d_x = 1;             // 使用__device__来静态声明全局内存变量
__device__ int d_y[2];              // 使用__device__来静态声明全局内存变量

__global__ void kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int h_y[2] = {10, 20};      // 主机内存定义变量
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));               // 使用主机端的cudaMemcpyToSymbol函数实现从主机内存到设备静态全局内存的数据拷贝

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));             // 使用主机端的cudaMemcpyFromSymbol函数实现从设备静态全局内存到主机内存的数据拷贝
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);              

    CHECK(cudaDeviceReset());

    return 0;
}