/*********************************************************************************************
 * file name  : cuda_global_memory_static.cu
 * author     : justin
 * date       : 2024-3-19
 * brief      : 动态全局内存变量使用示例
***********************************************************************************************/

#include <cuda_runtime.h>
#include <iostream>
#include "../cuda错误检测/error.cuh"


__global__ void kernel(int *d_x, int d_y[2])
{
    d_y[0] += *d_x;
    d_y[1] += *d_x;

    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", *d_x, d_y[0], d_y[1]);
}



int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProps;
    CHECK(cudaGetDeviceProperties(&deviceProps, devID));
    std::cout << "运行GPU设备:" << deviceProps.name << std::endl;

    int h_x = 1;
    int h_y[2] = {10, 20};              // 主机内存定义变量

    int *d_x, *d_y;
    CHECK(cudaMalloc((int**)&d_x, sizeof(int)));
    CHECK(cudaMalloc((int**)&d_y, sizeof(int)*2));
    CHECK(cudaMemcpy(d_x, &h_x, sizeof(int), cudaMemcpyHostToDevice));      // 注意cudaMemcpy的前两个参数均为指针
    CHECK(cudaMemcpy(d_y, h_y, sizeof(int)*2, cudaMemcpyHostToDevice)); 

    dim3 block(1);
    dim3 grid(1);
    kernel<<<grid, block>>>(d_x, d_y);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h_y, d_y, sizeof(int)*2, cudaMemcpyDeviceToHost)); 
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);

    CHECK(cudaDeviceReset());

    return 0;
}